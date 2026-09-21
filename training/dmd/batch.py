"""DMD init/run/publish: resumable recording caches and validated training manifests."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import uuid

from inference.experts.assets import file_sha256, verify_asset
from inference.perception.runtime import runtime_signature
from inference.perception.video import run_video
from training.common.config import canonical_hash, load_config, read_json
from training.dmd.batch_config import init_config, load_batch
from training.dmd.labels import camera_offsets, compile_labels, MAX_OUTPUT_SLOTS
from training.dmd.prepare import ROOT, _contained, _source_annotation


def decode_video(path):
    # Keep `init` usable with python -S and without NumPy/PyTorch installed.
    from inference.perception.video_source import decode_video as decode
    return decode(path)


def _environment():
    versions = {'python': platform.python_version()}
    for name in ('numpy', 'torch', 'torchvision', 'mediapipe', 'av', 'yolox',
                 'opencv-python', 'opencv-contrib-python', 'torch-npu'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    code = {}
    for directory in ('training/dmd', 'training/common', 'training/relation', 'training/gaze', 'inference'):
        for path in sorted((ROOT / directory).rglob('*.py')):
            code[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    return dict(versions=versions, implementation_sha256=canonical_hash(code))


def _context(config_path, dataset_root, output_dir, *, gaze=False):
    loader = load_batch
    if gaze:
        from training.dmd.gaze_batch import load_batch as loader
    context = loader(config_path)
    for jid, setting in context['config']['jobs'].items():
        if setting['confirmed_exclusive'] is not True:
            raise ValueError(f'ROI exclusive ownership not confirmed: {jid}; inspect video before setting true')
    root = Path(dataset_root).resolve(strict=True)
    output = Path(output_dir).resolve()
    if not root.is_dir():
        raise ValueError('Dataset root must be a directory')
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Output must be outside and not an ancestor of Dataset')
    for spec in (context['runtime']['objects']['asset'], context['runtime']['face_asset'], context['runtime']['hands_asset']):
        verify_asset(spec)
    context.update(root=root, output=output, environment=_environment(),
                   group_profile=read_json(ROOT / 'configs/perception/relation_groups_v1.json'),
                   feature_profile=read_json(ROOT / 'configs/perception/interaction_v1.json'))
    if gaze:
        from training.gaze.profile import default_group_profile
        from inference.perception.gaze_profile import gaze_profile
        context.update(group_profile=default_group_profile(),feature_profile=gaze_profile())
    return context


@contextmanager
def _lock(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / '.batch.lock'
    # Never steal a lock: killed processes require operator verification and removal.
    with path.open('x', encoding='utf-8') as handle:
        handle.write('DMD batch active; remove only after confirming the process has stopped.\n')
    try:
        yield
    finally:
        path.unlink()


def _inputs(context, job):
    root = context['root']
    source = _contained(root, job['video'])
    if not source.is_file() or source.stat().st_size != job['video_size_bytes']:
        raise ValueError(f'Video missing or size changed since preparation: {job["video"]}')
    identity = job['annotation'].split('/')
    record = dict(subset=job['subset'], group=identity[2], subject_id=job['subject_id'],
                  session=identity[4], recording_id=job['recording_id'])
    fresh, annotation = _source_annotation(root, record, dict(path=job['annotation'],
        kind=f'ann_{job["subset"]}.json', sha256=job['annotation_sha256']))
    doc = read_json(annotation)
    if file_sha256(annotation) != fresh['sha256']:
        raise ValueError('Annotation changed during read')
    stream = doc['openlabel']['streams'][job['view'] + '_camera']
    expected_uri = '/'.join(job['video'].split('/')[2:])
    if (stream['uri'] != expected_uri or camera_offsets(doc)[job['view']] != job['annotation_frame_offset']
            or stream['stream_properties']['total_frames'] != job['declared_frames']):
        raise ValueError('Video synchronization differs from preparation')
    for item in job['annotation_sources']:
        if file_sha256(_contained(root, item['path'])) != item['sha256']:
            raise ValueError('Auxiliary annotation changed since preparation')
    stat = source.stat()
    evidence = dict(job={k: job[k] for k in ('job_id', 'recording_id', 'subject_id', 'subset', 'view', 'split',
                    'video', 'annotation', 'annotation_sha256', 'annotation_sources')},
        video_sha256=file_sha256(source), roi=context['config']['jobs'][job['job_id']],
        label_policy=context['label_policy'], group_profile=context['group_profile'],
        feature_profile=context['feature_profile'], runtime_signature=runtime_signature(context['runtime']),
        environment=context['environment'])
    return source, doc, canonical_hash(evidence), evidence, (stat.st_size, stat.st_mtime_ns)


def _existing(context, job, fingerprint):
    from training.dmd.cache import validate_complete
    folder = _contained(context['output'], 'records/' + job['job_id'])
    if folder.exists():
        for attempt in sorted(folder.iterdir(), reverse=True):
            if not attempt.is_dir() or attempt.is_symlink() or (hasattr(attempt, 'is_junction') and attempt.is_junction()):
                continue
            report = validate_complete(attempt, fingerprint)
            if report is not None and report['job'] == job and not (attempt / 'failure.json').exists():
                try:
                    if canonical_hash(read_json(_contained(attempt, 'input.json'))) == fingerprint:
                        return attempt, report
                except (OSError, ValueError, TypeError):
                    pass
    return None


def _extract(context, job, source, doc, fingerprint, evidence, source_stat, progress):
    import numpy as np
    from training.dmd.cache import write_json, validate_complete
    attempt = _contained(context['output'], f'records/{job["job_id"]}/{uuid.uuid4().hex}')
    attempt.mkdir(parents=True, exist_ok=False)
    write_json(attempt / 'input.json', evidence)
    try:
        # Probe only dimensions, then close. Full extraction still starts at source frame 0.
        frames = decode_video(source)
        try:
            first = next(frames)
            height, width = first.bgr.shape[:2]
        finally:
            close = getattr(frames, 'close', None)
            if close:
                close()
        runtime = deepcopy(context['runtime'])
        roi = context['config']['jobs'][job['job_id']]['normalized_roi']
        runtime.update(recording_id=job['recording_id'], view=job['view'],
            driver_roi=[roi[0]*width, roi[1]*height, roi[2]*width, roi[3]*height],
            driver_region_confirmed_exclusive=True)
        pts = []
        def observe(frame):
            if frame.timestamp_ms // 40 + 1 > MAX_OUTPUT_SLOTS:
                raise ValueError('Source exceeds label compiler slot capacity; no truncation')
            pts.append(frame.timestamp_ms)
            if progress and len(pts) % 100 == 0:
                progress(f'{job["job_id"]}: source_frames={len(pts)} video_seconds={frame.timestamp_ms/1000:.2f}')
        options = {'feature_profile':context['feature_profile']} if context.get('gaze') else {}
        feature = run_video(source, attempt / 'features', runtime,
                            group_profile=context['group_profile'], frame_observer=observe, **options)
        compiler = compile_labels
        if context.get('gaze'):
            from training.dmd.gaze_labels import compile_gaze_labels as compiler
        labels = compiler(doc, job['view'], pts,
            closed_eye_min_ms=context['label_policy']['closed_eye_min_ms'], subset=job['subset'])
        if (feature['slots'] != len(labels['slots_ms']) or feature['first_timestamp_ms'] != 0
                or feature['last_timestamp_ms'] != labels['slots_ms'][-1]):
            raise ValueError('Source timeline and extracted slots disagree')
        np.save(attempt / 'source_pts_ms.npy', np.asarray(pts, dtype=np.int64), allow_pickle=False)
        np.save(attempt / 'targets.npy', np.asarray(labels['targets'], dtype=np.float32), allow_pickle=False)
        np.save(attempt / 'label_masks.npy', np.asarray(labels['label_masks'], dtype=np.uint8), allow_pickle=False)
        if context.get('gaze'):
            np.save(attempt/'gaze_targets.npy',np.asarray(labels['gaze_targets'],dtype=np.int64),allow_pickle=False)
            np.save(attempt/'gaze_label_masks.npy',np.asarray(labels['gaze_label_masks'],dtype=np.uint8),allow_pickle=False)
        stat = source.stat()
        if (stat.st_size, stat.st_mtime_ns) != source_stat:
            raise ValueError('Video changed during extraction')
        for item in job['annotation_sources']:
            if file_sha256(_contained(context['root'], item['path'])) != item['sha256']:
                raise ValueError('Annotation changed during extraction')
        arrays = {name: f'features/{name}.npy' for name in ('values', 'masks', 'expert_age_ms')}
        arrays.update({name: f'{name}.npy' for name in ('targets', 'label_masks', 'source_pts_ms')})
        if context.get('gaze'):
            arrays.update({name:f'{name}.npy' for name in ('gaze_targets','gaze_label_masks')})
        report = dict(schema_version='driveguard-dmd-complete-v2' if context.get('gaze') else 'driveguard-dmd-complete-v1', fingerprint=fingerprint,
            job=job, feature_record=feature, arrays=arrays,
            array_sha256={name: file_sha256(attempt / path) for name, path in arrays.items()},
            label_summary=labels['summary'])
        write_json(attempt / 'complete.pending.json', report)
        (attempt / 'complete.pending.json').rename(attempt / 'complete.json')
        if validate_complete(attempt, fingerprint) is None:
            raise ValueError('Newly written cache failed integrity validation')
        return attempt, report
    except Exception as error:
        # This attempt is exclusively ours; remove only its success marker on failure.
        complete = attempt / 'complete.json'
        if complete.exists():
            complete.unlink()
        write_json(attempt / 'failure.json', {'error': str(error), 'type': type(error).__name__})
        raise


def run_batch(config_path, dataset_root, output_dir, progress=None, *, gaze=False):
    from training.dmd.cache import write_json
    context = _context(config_path, dataset_root, output_dir, gaze=gaze)
    result = dict(completed=0, reused=0, failed=0, jobs=[])
    with _lock(context['output']):
        for index, job in enumerate(context['selected'], 1):
            if progress:
                progress(f'[{index}/{len(context["selected"])}] {job["split"]} {job["video"]}')
            try:
                source, doc, fingerprint, evidence, stat = _inputs(context, job)
                existing = _existing(context, job, fingerprint)
                if existing:
                    state = 'reused'
                else:
                    existing = _extract(context, job, source, doc, fingerprint, evidence, stat, progress)
                    state = 'completed'
                result[state] += 1
                result['jobs'].append(dict(job_id=job['job_id'], status=state,
                                           attempt=str(existing[0].relative_to(context['output']).as_posix())))
            except Exception as error:
                result['failed'] += 1
                result['jobs'].append(dict(job_id=job['job_id'], status='failed', error=str(error)))
                if progress:
                    progress(f'FAILED {job["job_id"]}: {error}')
        result['status'] = 'complete' if not result['failed'] else 'errors'
        result['training_ready'] = False  # Only publish can qualify a manifest.
        reports = _contained(context['output'], 'reports')
        reports.mkdir(exist_ok=True)
        path = reports / f'{uuid.uuid4().hex}.json'
        write_json(path, result)
        result['report'] = str(path)
    return result


def publish_batch(config_path, dataset_root, output_dir, training_config, *, gaze=False):
    from training.dmd.cache import publish_manifest
    context = _context(config_path, dataset_root, output_dir, gaze=gaze)
    with _lock(context['output']):
        completed = []
        for job in context['selected']:
            _, _, fingerprint, _, _ = _inputs(context, job)
            existing = _existing(context, job, fingerprint)
            if existing is None:
                raise ValueError(f'No matching complete cache for {job["job_id"]}; run this configuration first')
            completed.append(existing)
        return publish_manifest(context['output'], completed, load_config(training_config), context['label_policy'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='Create editable pilot/all ROI config; no ML imports')
    init.add_argument('--preparation', required=True)
    init.add_argument('--output', required=True)
    init.add_argument('--selection', choices=('pilot', 'all'), default='pilot')
    for name in ('run', 'publish'):
        command = commands.add_parser(name)
        command.add_argument('--config', required=True)
        command.add_argument('--dataset-root', required=True)
        command.add_argument('--output-dir', required=True)
        if name == 'publish':
            command.add_argument('--training-config', default=str(ROOT / 'configs/training/relation_gated.json'))
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            config = init_config(args.preparation, args.output, args.selection)
            result = dict(status='needs_roi_review', jobs=len(config['jobs']), config=str(Path(args.output).resolve()))
        elif args.command == 'run':
            result = run_batch(args.config, args.dataset_root, args.output_dir,
                               progress=lambda message: print(message, file=sys.stderr, flush=True))
        else:
            result = publish_batch(args.config, args.dataset_root, args.output_dir, args.training_config)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 2 if result.get('failed', 0) else 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as error:
        print(f'DMD batch failed: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
