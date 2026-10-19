"""Replay cached DMD face extraction, explain rejection, and compare head/iris masks."""
import argparse
from collections import Counter
from contextlib import ExitStack
import importlib
import json
import math
from pathlib import Path
import csv
import sys

import numpy as np

from inference.experts.assets import file_sha256, verify_asset
from training.common.config import canonical_hash, read_json, TASKS
from training.dmd.cache import METADATA, validate_complete, write_json, _unlinked


def _contained(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('Expected a relative data path')
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Data path is outside its root or is not a file')
    return path


def _stamp(path):
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ino)


def _prepare(record, root, dataset):
    values = _unlinked(_contained(root, record['values']))
    attempt = values.parent.parent
    inputs = read_json(attempt / 'input.json')
    complete = validate_complete(attempt, canonical_hash(inputs))
    if complete is None:
        raise ValueError('Invalid cache: checksum, provenance, layout or dimensions failed')
    job, feature = complete['job'], complete['feature_record']
    if record['record_id'] != job['job_id'] or any(record[k] != job[k] for k in METADATA):
        raise ValueError('Manifest metadata differs from cache')
    for name, relative in complete['arrays'].items():
        if name != 'source_pts_ms' and _contained(root, record[name]) != (attempt / relative).resolve():
            raise ValueError('Manifest arrays differ from cache')
    if feature['schema_version'] != 'driveguard-feature-record-v3':
        raise ValueError('Gaze v3 feature cache required')
    provenance = feature['provenance']
    config = provenance['config']
    if (config['schema_version'] != 'driveguard-runtime-v2' or config['driver_roi'] != feature['driver_roi']
            or any(config[k] != job[k] for k in ('recording_id','view'))):
        raise ValueError('Cached runtime configuration disagrees with feature record')
    source = _contained(dataset, job['video'])
    source_stamp = _stamp(source)
    digest = file_sha256(source)
    if _stamp(source) != source_stamp:
        raise ValueError('Source video changed during checksum verification')
    if digest != inputs.get('video_sha256') or digest != provenance['source_video']['sha256']:
        raise ValueError('Source video checksum differs from original extraction')
    if source.stat().st_size != provenance['source_video']['size_bytes']:
        raise ValueError('Source video size differs from original extraction')
    for name in ('face_asset','hands_asset'):
        verify_asset(config[name])
    return dict(record=record, attempt=attempt, complete=complete, source=source,
                video_sha256=digest, source_stamp=source_stamp, config=config)


def _replay(item, directory, samples_per_reason, max_seconds):
    from inference.experts.mediapipe_adapter import MediaPipeAdapter
    from inference.perception.video_source import decode_video, sample_frames
    from inference.perception.runtime import runtime_signature
    from training.dmd.face_diagnostic_samples import Samples
    feature, config = item['complete']['feature_record'], item['config']
    names = feature['feature_schema']['names']
    head_indices = [names.index('head_' + axis + '_deg') for axis in ('yaw','pitch','roll')]
    iris_indices = [names.index('iris_' + side + '_' + axis) for side in ('left','right') for axis in ('x','y')]
    counters, iris_reasons, candidate_reasons = Counter(), Counter(), Counter()
    disagreements, positive_reasons = Counter(), Counter()
    samples = Samples(directory, samples_per_reason)
    count = unique = head_diff = iris_diff = pts_diff = 0
    last_time, clock = None, {}
    if _stamp(item['source']) != item['source_stamp']:
        raise ValueError('Source video changed after checksum verification')
    expected_slots = feature['slots']
    limit = math.ceil(min(expected_slots/25, max_seconds)*25) if max_seconds else expected_slots
    with ExitStack() as stack:
        arrays = {}
        for name in ('masks','targets','label_masks','source_pts_ms'):
            arrays[name] = np.load(item['attempt']/item['complete']['arrays'][name], mmap_mode='r', allow_pickle=False)
            stack.callback(arrays[name]._mmap.close)
        adapter = MediaPipeAdapter(config['face_asset'], config['hands_asset'], driver_roi=config['driver_roi'],
                    driver_region_confirmed_exclusive=config['driver_region_confirmed_exclusive'],
                    gaze_features=True, diagnostics=True)
        stack.callback(adapter.close)
        versions = {name: str(importlib.import_module(name).__version__) for name in ('numpy','mediapipe','av')}
        replay_model_id = adapter.face_model_id + ':runtime-' + runtime_signature(config)
        original_model_id = feature['expert_models']['face']
        frames = decode_video(item['source'], clock)
        stack.callback(frames.close)
        slots = sample_frames(frames)
        stack.callback(slots.close)
        fields = ['slot_index','slot_ms','source_pts_ms','cached_source_pts_ms','source_pts_matches','face_age_ms',
                  'reason','iris_reason','detected_faces','roi_candidates','selected_index',
                  'cached_head_valid','replay_head_valid','cached_iris_valid','replay_iris_valid',
                  'head_mask_matches','iris_mask_matches','dist_label_known','dist_target']
        csvfile = stack.enter_context((directory/'frames.csv').open('x', encoding='utf-8-sig', newline=''))
        writer = csv.DictWriter(csvfile, fieldnames=fields); writer.writeheader()
        candidates = stack.enter_context((directory/'candidates.jsonl').open('x', encoding='utf-8'))
        for slot, frame in slots:
            if count >= expected_slots:
                raise ValueError('Replay has more slots than cache')
            if list(reversed(frame.bgr.shape[:2])) != feature['image_size']:
                raise ValueError('Decoded image size differs from cache')
            if last_time != frame.timestamp_ms:
                adapter.predict(frame.bgr, frame.timestamp_ms)
                detail = adapter.last_face_diagnostics
                candidates.write(json.dumps(dict(source_pts_ms=frame.timestamp_ms, **detail), allow_nan=False)+'\n')
                samples.add(frame, detail, config['driver_roi'])
                candidate_reasons.update(c['invalid_reason'] for c in detail['candidates'] if c['invalid_reason'])
                last_time = frame.timestamp_ms
                unique += 1
            age = slot-frame.timestamp_ms
            fresh = age <= feature['extractor_profile']['max_age_ms']
            head = bool(detail['head_usable'] and fresh)
            left, right = bool(detail['iris_left_valid'] and fresh), bool(detail['iris_right_valid'] and fresh)
            actual_head = [head]*3
            actual_iris = [left,left,right,right]
            cached_head = arrays['masks'][count, head_indices].astype(bool).tolist()
            cached_iris = arrays['masks'][count, iris_indices].astype(bool).tolist()
            hmatch, imatch = cached_head == actual_head, cached_iris == actual_iris
            head_diff += not hmatch; iris_diff += not imatch
            cached_head_valid, cached_iris_valid = all(cached_head), all(cached_iris[:2]) or all(cached_iris[2:])
            expected_pts = int(arrays['source_pts_ms'][np.searchsorted(arrays['source_pts_ms'], slot, side='right')-1])
            pmatch = frame.timestamp_ms == expected_pts
            pts_diff += not pmatch
            reason = detail['reason'] if fresh else 'stale_face'
            counters[reason] += 1; iris_reasons[detail['iris_reason'] if fresh else 'stale_face'] += 1
            disagreements[f'cached_{int(cached_head_valid)}_replay_{int(head)}'] += 1
            dist = TASKS.index('distraction')
            known = bool(arrays['label_masks'][count,dist])
            target = int(arrays['targets'][count,dist]) if known else ''
            if known and target == 1:
                positive_reasons[reason] += 1
            writer.writerow(dict(slot_index=count,slot_ms=slot,source_pts_ms=frame.timestamp_ms,
                cached_source_pts_ms=expected_pts,source_pts_matches=pmatch,face_age_ms=age,reason=reason,
                iris_reason=detail['iris_reason'] if fresh else 'stale_face',detected_faces=detail['detected_faces'],
                roi_candidates=detail['roi_candidates'],selected_index=detail['selected_index'],
                cached_head_valid=cached_head_valid,replay_head_valid=head,cached_iris_valid=cached_iris_valid,
                replay_iris_valid=left or right,head_mask_matches=hmatch,iris_mask_matches=imatch,
                dist_label_known=known,dist_target=target))
            count += 1
            if count % 250 == 0:
                print(f'{item["record"]["record_id"]}: {count}/{limit} slots', file=sys.stderr, flush=True)
            if count == limit and limit < expected_slots:
                break
        if count != limit:
            raise ValueError('Replay slot count differs from cache/requested prefix')
    if _stamp(item['source']) != item['source_stamp']:
        raise ValueError('Source video changed during replay')
    original_versions = feature['provenance']['versions']
    return dict(**{k:item['record'][k] for k in ('record_id',*METADATA)},directory=directory.name,
        video=str(item['source']),video_sha256=item['video_sha256'],driver_roi=config['driver_roi'],
        slots=count,cached_slots=expected_slots,partial=count<expected_slots,unique_inference_frames=unique,
        reason_counts=dict(counters),iris_reason_counts=dict(iris_reasons),
        invalid_candidate_counts=dict(candidate_reasons),positive_slot_reason_counts=dict(positive_reasons),
        head_mask_mismatches=head_diff,iris_mask_mismatches=iris_diff,source_pts_mismatches=pts_diff,
        head_mask_contingency=dict(disagreements),model_id_matches=replay_model_id==original_model_id,
        original_face_model_id=original_model_id,replay_face_model_id=replay_model_id,
        replay_versions=versions,original_versions={k:original_versions.get(k) for k in versions},
        version_matches={k:versions[k]==original_versions.get(k) for k in versions},
        head_iris_masks_reproduced=(head_diff==iris_diff==pts_diff==0),clock=clock,samples=samples.items)


def diagnose(*, manifest, data_root, dataset_root, output_dir, view='body', split='val',
             subset='distraction', record_ids=None, max_seconds=0, samples_per_reason=2):
    if not math.isfinite(max_seconds) or max_seconds < 0 or type(samples_per_reason) is not int or samples_per_reason < 0:
        raise ValueError('Invalid prefix duration or sample count')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Output directory already exists; choose a new directory')
    root, dataset = Path(data_root).resolve(strict=True), Path(dataset_root).resolve(strict=True)
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (root,dataset)):
        raise ValueError('Output must be separate from source cache and dataset')
    manifest = Path(manifest).resolve(strict=True)
    data = read_json(manifest)
    selected = [r for r in data['records'] if (view=='all' or r['view']==view) and r['split']==split
                and r['subset']==subset and (record_ids is None or r['record_id'] in record_ids)]
    if not selected:
        raise ValueError('No records match the requested filters')
    identities = [r['record_id'] for r in selected]
    if len(set(identities)) != len(identities) or (record_ids and set(record_ids)-set(identities)):
        raise ValueError('Duplicate or unmatched record IDs')
    items = []
    for r in selected:
        print(f'Validating original cache/video: {r["record_id"]}', file=sys.stderr, flush=True)
        items.append(_prepare(r, root, dataset))
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for index,item in enumerate(items, 1):
        print(f'Replaying {index}/{len(items)}: {item["record"]["record_id"]}', file=sys.stderr, flush=True)
        directory = output/f'record-{index:03d}'; directory.mkdir()
        records.append(_replay(item,directory,samples_per_reason,max_seconds))
    groups = {}
    for selected_view in sorted({r['view'] for r in records}):
        rows = [r for r in records if r['view'] == selected_view]
        totals = {key:sum(r[key] for r in rows) for key in ('slots','unique_inference_frames',
                  'head_mask_mismatches','iris_mask_mismatches','source_pts_mismatches')}
        for key in ('reason_counts','iris_reason_counts','positive_slot_reason_counts','invalid_candidate_counts'):
            counter = Counter()
            for r in rows:
                counter.update(r[key])
            totals[key] = dict(counter)
        totals['partial'] = any(r['partial'] for r in rows)
        totals['runtime_and_masks_match'] = all(r['model_id_matches'] and all(r['version_matches'].values())
                                               and r['head_iris_masks_reproduced'] for r in rows)
        groups[selected_view] = totals
    report = dict(schema_version='driveguard-face-diagnostics-v1',status='complete',
        manifest=str(manifest),manifest_sha256=file_sha256(manifest),
        filters=dict(view=view,split=split,subset=subset,record_ids=record_ids),max_seconds=max_seconds,
        partial=any(r['partial'] for r in records),groups=dict(view=groups),records=records,
        counting='Reasons count 25 Hz slots (including reused frames); invalid candidates count unique inference frames. '
                 'Positive counts use slot labels, not training windows. No checkpoint predictions are recomputed. '
                 'Replay causes explain original cache only to the extent masks, timestamps and runtime agree.')
    from training.dmd.face_diagnostic_samples import write_gallery
    write_gallery(output, records)
    write_json(output/'summary.pending.json', report)
    (output/'summary.pending.json').replace(output/'summary.json')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','data-root','dataset-root','output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--view',choices=('body','face','hands','all'),default='body')
    parser.add_argument('--split',choices=('train','val','test'),default='val')
    parser.add_argument('--subset',choices=('distraction','drowsiness','gaze'),default='distraction')
    parser.add_argument('--record-id',dest='record_ids',action='append')
    parser.add_argument('--max-seconds',type=float,default=0,help='Explicit prefix limit per recording; 0 replays full videos')
    parser.add_argument('--samples-per-reason',type=int,default=2,help='Per recording; 0 disables JPEGs')
    args = parser.parse_args(argv)
    try:
        report = diagnose(**vars(args))
        print(json.dumps(dict(status=report['status'],partial=report['partial'],records=len(report['records']),
                              output_dir=str(Path(args.output_dir).resolve()))))
        return 0
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,ImportError) as error:
        print(f'Face diagnostics failed: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
