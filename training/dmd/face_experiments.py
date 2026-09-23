"""Read-only face perception comparisons; coverage and smoothness are not accuracy."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import importlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

from training.common.config import TASKS, canonical_hash, read_json
from training.dmd.face_experiment_config import load_config, validate_config


def _coverage():
    return {name: dict(slots=0, cached_head_available=0, variant_head_available=0,
                      cached_iris_available=0, variant_iris_available=0)
            for name in ('all', 'known_positive', 'known_negative', 'unknown')}


def _coverage_rates(coverage):
    for counts in coverage.values():
        counts['rates'] = {key: counts[key]/counts['slots'] if counts['slots'] else None
                           for key in counts if key not in ('slots', 'rates')}


def _aggregate(records, variants):
    result = {}
    for variant in variants:
        rows = [r for r in records if r['variant']['name'] == variant['name']]
        totals = {key: sum(r[key] for r in rows) for key in ('slots', 'cached_slots', 'unique_inference_frames',
                  'head_mask_mismatches', 'iris_mask_mismatches', 'source_pts_mismatches')}
        coverage = _coverage()
        for group, counts in coverage.items():
            for key in counts:
                counts[key] = sum(r['coverage'][group][key] for r in rows)
        _coverage_rates(coverage)
        for key in ('reason_counts', 'iris_reason_counts', 'head_mask_contingency'):
            counter = Counter()
            for row in rows:
                counter.update(row[key])
            totals[key] = dict(counter)
        inference_ms = sum(r['timing']['inference_total_ms'] for r in rows)
        unique = totals['unique_inference_frames']
        pairs = sum(r['angle_jumps']['pairs'] for r in rows)
        baseline = variant['name'] == 'baseline'
        result[variant['name']] = dict(totals, parameters=variant, records=len(rows), coverage=coverage,
            partial=any(r['partial'] for r in rows),
            baseline_masks_reproduced=all(r['baseline_masks_reproduced'] for r in rows) if baseline else None,
            baseline_runtime_and_masks_match=all(r['baseline_runtime_and_masks_match'] for r in rows) if baseline else None,
            timing=dict(inference_total_ms=inference_ms, inference_mean_ms=inference_ms/unique if unique else None,
                inference_max_ms=max(r['timing']['inference_max_ms'] for r in rows),
                replay_wall_seconds=sum(r['timing']['replay_wall_seconds'] for r in rows),
                inference_frames_per_second=unique/(inference_ms/1000) if inference_ms else None),
            angle_jumps=dict(pairs=pairs, axes=['yaw', 'pitch', 'roll'],
                mean_abs_degrees=[sum(r['angle_jumps']['pairs']*r['angle_jumps']['mean_abs_degrees'][i]
                    for r in rows if r['angle_jumps']['pairs'])/pairs for i in range(3)] if pairs else None,
                max_abs_degrees=[max(r['angle_jumps']['max_abs_degrees'][i]
                    for r in rows if r['angle_jumps']['pairs']) for i in range(3)] if pairs else None))
    return result


def _replay(item, variant, directory, samples_per_reason, max_seconds):
    import numpy as np
    from inference.experts.face_experiment_adapter import FaceExperimentAdapter
    from inference.perception.runtime import runtime_signature
    from inference.perception.video_source import decode_video, sample_frames
    from training.dmd.face_diagnostic_samples import Samples
    from training.dmd.face_diagnostics import _stamp

    feature, config = item['complete']['feature_record'], item['config']
    names = feature['feature_schema']['names']
    heads = [names.index('head_'+axis+'_deg') for axis in ('yaw', 'pitch', 'roll')]
    irises = [names.index('iris_'+side+'_'+axis) for side in ('left', 'right') for axis in ('x', 'y')]
    expected_slots = feature['slots']
    limit = min(expected_slots, math.ceil(min(expected_slots/25, max_seconds)*25)) if max_seconds else expected_slots
    samples, counters, iris_reasons = Samples(directory, samples_per_reason), Counter(), Counter()
    coverage, contingency, clock = _coverage(), Counter(), {}
    count = unique = head_diff = iris_diff = pts_diff = 0
    last_pts = previous_angles = None
    jump_pairs, jump_sum, jump_max = 0, np.zeros(3), np.zeros(3)
    inference_total = inference_max = 0.
    started = perf_counter()
    if _stamp(item['source']) != item['source_stamp']:
        raise ValueError('Source video changed after checksum verification')
    with ExitStack() as stack:
        arrays = {}
        for name in ('values', 'masks', 'targets', 'label_masks', 'source_pts_ms'):
            arrays[name] = np.load(item['attempt']/item['complete']['arrays'][name], mmap_mode='r', allow_pickle=False)
            stack.callback(arrays[name]._mmap.close)
        adapter = FaceExperimentAdapter(config['face_asset'], driver_roi=config['driver_roi'],
            driver_region_confirmed_exclusive=config['driver_region_confirmed_exclusive'], variant=variant)
        stack.callback(adapter.close)
        versions = {name: str(importlib.import_module(name).__version__) for name in ('numpy', 'mediapipe', 'av')}
        baseline_id = (adapter.baseline_face_model_id+':runtime-'+runtime_signature(config)
                       if adapter.baseline_face_model_id else None)
        frames = decode_video(item['source'], clock); stack.callback(frames.close)
        slots = sample_frames(frames); stack.callback(slots.close)
        fields = ['slot_index', 'slot_ms', 'source_pts_ms', 'cached_source_pts_ms', 'source_pts_matches',
            'face_age_ms', 'inference_performed', 'inference_ms', 'reason', 'iris_reason',
            'detected_faces', 'roi_candidates', 'selected_index', 'cached_head_mask', 'variant_head_mask',
            'cached_iris_mask', 'variant_iris_mask', 'cached_head_valid', 'variant_head_valid',
            'cached_iris_valid', 'variant_iris_valid', 'head_mask_matches', 'iris_mask_matches',
            'dist_label_known', 'dist_target', 'head_yaw_deg', 'head_pitch_deg', 'head_roll_deg',
            'cached_head_yaw_deg', 'cached_head_pitch_deg', 'cached_head_roll_deg']
        handle = stack.enter_context((directory/'frames.csv').open('x', encoding='utf-8-sig', newline=''))
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        candidates = stack.enter_context((directory/'candidates.jsonl').open('x', encoding='utf-8'))
        for slot, frame in slots:
            if count >= expected_slots:
                raise ValueError('Replay has more slots than original cache')
            if list(reversed(frame.bgr.shape[:2])) != feature['image_size']:
                raise ValueError('Decoded image size differs from cache')
            inferred = last_pts != frame.timestamp_ms
            duration = 0.
            if inferred:
                start = perf_counter()
                observation = adapter.predict(frame.bgr, frame.timestamp_ms)
                duration = (perf_counter()-start)*1000
                inference_total += duration; inference_max = max(inference_max, duration)
                detail = adapter.last_face_diagnostics
                candidates.write(json.dumps(dict(source_pts_ms=frame.timestamp_ms,
                    crop_pixels=adapter.crop_pixels, **detail), allow_nan=False)+'\n')
                samples.add(frame, detail, config['driver_roi'])
                angles = ([observation['head_pose'][axis+'_deg'] for axis in ('yaw', 'pitch', 'roll')]
                          if detail['head_usable'] else None)
                if angles is not None and previous_angles is not None:
                    # Wrapped per-axis displacement; adjacency resets after any unavailable inference.
                    jump = np.abs((np.asarray(angles)-previous_angles+180)%360-180)
                    jump_sum += jump; jump_max = np.maximum(jump_max, jump); jump_pairs += 1
                previous_angles = None if angles is None else np.asarray(angles)
                last_pts = frame.timestamp_ms; unique += 1
            age = slot-frame.timestamp_ms
            fresh = age <= feature['extractor_profile']['max_age_ms']
            head = bool(detail['head_usable'] and fresh)
            left, right = bool(detail['iris_left_valid'] and fresh), bool(detail['iris_right_valid'] and fresh)
            hmask, imask = [int(head)]*3, [int(left)]*2+[int(right)]*2
            cached_h = arrays['masks'][count, heads].astype(int).tolist()
            cached_i = arrays['masks'][count, irises].astype(int).tolist()
            cached_head, cached_iris = all(cached_h), all(cached_i[:2]) or all(cached_i[2:])
            hmatch, imatch = hmask == cached_h, imask == cached_i
            head_diff += not hmatch; iris_diff += not imatch
            expected_pts = int(arrays['source_pts_ms'][np.searchsorted(arrays['source_pts_ms'], slot, side='right')-1])
            pmatch = expected_pts == frame.timestamp_ms; pts_diff += not pmatch
            reason, iris_reason = (detail['reason'], detail['iris_reason']) if fresh else ('stale_face', 'stale_face')
            counters[reason] += 1; iris_reasons[iris_reason] += 1
            contingency[f'cached_{int(cached_head)}_variant_{int(head)}'] += 1
            known = bool(arrays['label_masks'][count, TASKS.index('distraction')])
            target = int(arrays['targets'][count, TASKS.index('distraction')]) if known else ''
            label_group = ('known_positive' if target == 1 else 'known_negative') if known else 'unknown'
            for group in ('all', label_group):
                counts = coverage[group]
                counts['slots'] += 1
                for key, value in dict(cached_head_available=cached_head, variant_head_available=head,
                                      cached_iris_available=cached_iris, variant_iris_available=left or right).items():
                    counts[key] += int(value)
            row = dict(slot_index=count, slot_ms=slot, source_pts_ms=frame.timestamp_ms,
                cached_source_pts_ms=expected_pts, source_pts_matches=pmatch, face_age_ms=age,
                inference_performed=inferred, inference_ms=duration, reason=reason, iris_reason=iris_reason,
                detected_faces=detail['detected_faces'], roi_candidates=detail['roi_candidates'], selected_index=detail['selected_index'],
                cached_head_mask=json.dumps(cached_h), variant_head_mask=json.dumps(hmask),
                cached_iris_mask=json.dumps(cached_i), variant_iris_mask=json.dumps(imask),
                cached_head_valid=cached_head, variant_head_valid=head, cached_iris_valid=cached_iris,
                variant_iris_valid=left or right, head_mask_matches=hmatch, iris_mask_matches=imatch,
                dist_label_known=known, dist_target=target)
            for index, axis in enumerate(('yaw', 'pitch', 'roll')):
                row['head_'+axis+'_deg'] = angles[index] if head else ''
                row['cached_head_'+axis+'_deg'] = float(arrays['values'][count, heads[index]]) if cached_h[index] else ''
            writer.writerow(row)
            count += 1
            if count % 250 == 0:
                print(f'{item["record"]["record_id"]}/{variant["name"]}: {count}/{limit} slots', file=sys.stderr, flush=True)
            if count == limit and limit < expected_slots:
                break
        if count != limit:
            raise ValueError('Replay slot count differs from cache/requested prefix')
    # Completion can only be returned after model, decoder, CSV and all memory maps close.
    elapsed = perf_counter()-started
    if _stamp(item['source']) != item['source_stamp']:
        raise ValueError('Source video changed during replay')
    _coverage_rates(coverage)
    original_versions = feature['provenance']['versions']
    version_matches = {k: versions[k] == original_versions.get(k) for k in versions}
    mask_match = head_diff == iris_diff == pts_diff == 0
    baseline = variant['name'] == 'baseline'
    return dict(record_id=item['record']['record_id'], view=item['record']['view'], variant=variant,
        directory=directory.name, slots=count, cached_slots=expected_slots, partial=count<expected_slots,
        unique_inference_frames=unique, video=str(item['source']), video_sha256=item['video_sha256'],
        driver_roi=config['driver_roi'], driver_region_confirmed_exclusive=config['driver_region_confirmed_exclusive'],
        face_max_age_ms=feature['extractor_profile']['max_age_ms'],
        face_asset=config['face_asset'], hands_asset_verified=config['hands_asset'],
        original_face_model_id=feature['expert_models']['face'], baseline_face_model_id=baseline_id,
        experiment_model_id=adapter.experiment_model_id, model_id_matches=baseline_id == feature['expert_models']['face'],
        replay_versions=versions, original_versions={k: original_versions.get(k) for k in versions},
        version_matches=version_matches, head_mask_mismatches=head_diff, iris_mask_mismatches=iris_diff,
        source_pts_mismatches=pts_diff, head_mask_contingency=dict(contingency),
        baseline_masks_reproduced=mask_match if baseline else None,
        baseline_runtime_and_masks_match=(mask_match and baseline_id == feature['expert_models']['face'] and
                                          all(version_matches.values())) if baseline else None,
        coverage=coverage, reason_counts=dict(counters), iris_reason_counts=dict(iris_reasons),
        pose_coordinate_frame='crop-experimental' if variant['crop_xyxy'] else 'full-frame',
        training_cache_compatible=False,
        angle_jumps=dict(pairs=jump_pairs, axes=['yaw', 'pitch', 'roll'],
            mean_abs_degrees=(jump_sum/jump_pairs).tolist() if jump_pairs else None,
            max_abs_degrees=jump_max.tolist() if jump_pairs else None,
            counting='Adjacent valid inferred source frames only; resets after missing head. Wrapped Euler differences, not accuracy.'),
        timing=dict(inference_total_ms=inference_total, inference_mean_ms=inference_total/unique if unique else None,
            inference_max_ms=inference_max, replay_wall_seconds=elapsed,
            inference_frames_per_second=unique/(inference_total/1000) if inference_total else None,
            scope='Face inference plus conversion/diagnostics, excludes decode and JPEG; wall time includes setup and cleanup.'),
        clock=clock, samples=samples.items)


def run_experiments(*, manifest, data_root, dataset_root, output_dir, experiment_config,
                    view='body', split='val', subset='distraction', record_ids=None,
                    max_seconds=0, samples_per_reason=2):
    from inference.experts.assets import file_sha256
    from training.dmd.cache import write_json
    from training.dmd.face_diagnostics import _prepare, _stamp
    from training.dmd.face_diagnostic_samples import write_gallery

    experiment = validate_config(experiment_config) if isinstance(experiment_config, dict) else load_config(experiment_config)
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds < 0 or type(samples_per_reason) is not int or samples_per_reason < 0:
        raise ValueError('Invalid prefix duration or sample count')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Output directory already exists; choose a new directory')
    root, dataset = Path(data_root).resolve(strict=True), Path(dataset_root).resolve(strict=True)
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (root, dataset)):
        raise ValueError('Output must be separate from original cache and dataset')
    manifest = Path(manifest).resolve(strict=True)
    manifest_stamp, manifest_hash = _stamp(manifest), file_sha256(manifest)
    data = read_json(manifest)
    if data.get('schema_version') != 'driveguard-feature-manifest-v3':
        raise ValueError('Original v3 feature manifest required')
    selected = [r for r in data['records'] if (view == 'all' or r['view'] == view) and r['split'] == split
                and r['subset'] == subset and (record_ids is None or r['record_id'] in record_ids)]
    if not selected:
        raise ValueError('No records match requested filters')
    identities = [r['record_id'] for r in selected]
    if len(set(identities)) != len(identities) or (record_ids and set(record_ids)-set(identities)):
        raise ValueError('Duplicate or unmatched record IDs')
    items = []
    stamps = {manifest: manifest_stamp}
    for record in selected:
        print(f'Validating original cache/video: {record["record_id"]}', file=sys.stderr, flush=True)
        item = _prepare(record, root, dataset)
        paths = [p for p in item['attempt'].rglob('*') if p.is_file()]
        paths += [item['source']] + [Path(item['config'][key]['path']).resolve(strict=True) for key in ('face_asset', 'hands_asset')]
        stamps.update({p: _stamp(p) for p in paths})
        items.append(item)
    if not isinstance(experiment_config, dict):
        config_path = Path(experiment_config).resolve(strict=True)
        if load_config(config_path) != experiment:
            raise ValueError('Experiment configuration changed during validation')
        stamps[config_path] = _stamp(config_path)
    def unchanged():
        if any(_stamp(p) != stamp for p, stamp in stamps.items()):
            raise ValueError('Original manifest/cache/video/asset/config changed during experiment')
    unchanged()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index, item in enumerate(items, 1):
        for variant in experiment['variants']:
            unchanged()
            directory = output/f'record-{index:03d}-{variant["name"]}'; directory.mkdir()
            records.append(_replay(item, variant, directory, samples_per_reason, max_seconds))
    unchanged()
    report = dict(schema_version='driveguard-face-experiment-report-v1', status='complete',
        manifest=str(manifest), manifest_sha256=manifest_hash, experiment_config=experiment,
        experiment_config_hash=canonical_hash(experiment), max_seconds=max_seconds,
        filters=dict(view=view, split=split, subset=subset, record_ids=record_ids),
        partial=any(r['partial'] for r in records), records=records, variants=_aggregate(records, experiment['variants']),
        interpretation='Only baseline may establish cache mask reproduction. Other differences are candidates, not improvements. '
            'Coverage uses all 25 Hz slots and known positive/negative distraction labels, not windows. '
            'Angle continuity and speed are not accuracy; no automatic winner. '
            'Crop matrix angles are experimental crop coordinates, not calibrated original-image pose. '
            'All outputs are diagnostic only and cannot be published as training caches.')
    gallery_records = [dict(r, record_id=r['record_id']+' / '+r['variant']['name']) for r in records]
    write_gallery(output, gallery_records)
    write_json(output/'summary.pending.json', report)
    unchanged()
    (output/'summary.pending.json').replace(output/'summary.json')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'data-root', 'dataset-root', 'output-dir'):
        parser.add_argument('--'+name)
    parser.add_argument('--experiment-config', default='configs/perception/face_experiments.json')
    parser.add_argument('--check-config', action='store_true', help='Validate experiment JSON without loading any runtime or input')
    parser.add_argument('--view', choices=('body', 'face', 'hands', 'all'), default='body')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--subset', choices=('distraction', 'drowsiness', 'gaze'), default='distraction')
    parser.add_argument('--record-id', dest='record_ids', action='append')
    parser.add_argument('--max-seconds', type=float, default=0, help='Explicit prefix from video start; 0 means complete')
    parser.add_argument('--samples-per-reason', type=int, default=2, help='Bounded JPEGs per reason per record/variant; 0 disables')
    args = vars(parser.parse_args(argv))
    try:
        if args.pop('check_config'):
            config = load_config(args['experiment_config'])
            print(json.dumps(dict(valid=True, variants=len(config['variants']), assets_verified=False)))
            return 0
        for key in ('manifest', 'data_root', 'dataset_root', 'output_dir'):
            if not args[key]:
                parser.error('--'+key.replace('_', '-')+' is required for replay')
        report = run_experiments(**args)
        print(json.dumps(dict(status=report['status'], partial=report['partial'], record_variants=len(report['records']),
                              output_dir=str(Path(args['output_dir']).resolve()))))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f'Face experiments failed: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
