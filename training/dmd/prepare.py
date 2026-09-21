"""Build portable DMD preparation jobs. No decoding, feature export or training."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys

from training.common.config import canonical_hash, read_json
from training.dmd.annotations import parse_annotation
from training.dmd.audit import _identity

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = ROOT / 'configs/dmd/split.subjects.json'
DEFAULT_LABELS = ROOT / 'configs/dmd/labels.v1.json'
TASK_SCOPE = {'distraction': ['phone_use', 'drinking', 'distraction'],
              'drowsiness': ['fatigue'], 'gaze': []}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _contained(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or '\x00' in relative:
        raise ValueError('Invalid relative dataset path')
    parts = PurePosixPath(relative)
    if (parts.is_absolute() or '..' in parts.parts or PureWindowsPath(relative).drive
            or not parts.parts or ':' in parts.parts[0]):
        raise ValueError('Dataset path must be relative and contained')
    candidate = root
    for part in parts.parts:
        candidate = candidate / part
        if candidate.is_symlink() or (hasattr(candidate, 'is_junction') and candidate.is_junction()):
            raise ValueError('Linked dataset paths are not supported')
    if not candidate.resolve().is_relative_to(root):
        raise ValueError('Dataset path escapes root')
    return candidate


def load_split(path):
    split = read_json(path)
    if (not isinstance(split, dict) or set(split) != {'schema_version', 'train', 'val'}
            or split['schema_version'] != 'driveguard-dmd-split-v1'):
        raise ValueError('Invalid split configuration')
    assignments = {}
    for name in ('train', 'val'):
        subjects = split[name]
        if not isinstance(subjects, list) or not subjects:
            raise ValueError('Both train and val subject lists must be nonempty')
        for subject in subjects:
            if not isinstance(subject, str) or not re.fullmatch(r'[1-9][0-9]*', subject):
                raise ValueError('Subject IDs must be canonical positive integer strings')
            if subject in assignments:
                raise ValueError(f'Duplicate subject or split leakage: {subject}')
            assignments[subject] = name
    return split, assignments


def _source_annotation(root, record, annotation):
    relative = annotation.get('path')
    path = _contained(root, relative)
    identity = _identity(relative)
    expected = (record['subset'], record['group'], record['subject_id'],
                record['session'], record['recording_id'])
    if identity is None or identity[:5] != expected or identity[5] != annotation.get('kind'):
        raise ValueError('Annotation identity does not match recording/subject')
    fresh = parse_annotation(path)
    if fresh['sha256'] != annotation.get('sha256'):
        raise ValueError(f'Annotation changed since audit (SHA256): {relative}')
    if fresh['driver_ids'] != [record['subject_id']]:
        raise ValueError('Annotation driver subject does not match recording')
    unsafe = {'OUTSIDE_TIMELINE', 'FRAME_ACTION_MISMATCH', 'FRAME_TIMELINE_MISMATCH',
              'FRAME_OUTSIDE_TIMELINE'}
    if any(i['severity'] == 'error' or i['code'] in unsafe for i in fresh['issues']):
        raise ValueError(f'Annotation has unresolved interval/reference issues: {relative}')
    return fresh, path


def prepare_dataset(*, audit_path, dataset_root, output_dir, split_path=DEFAULT_SPLIT,
                    closed_eye_min_ms=None):
    """Recheck audited annotations and generate new jobs; never mutate source files."""
    from training.dmd.labels import camera_offsets
    if closed_eye_min_ms is not None and (type(closed_eye_min_ms) is not int or closed_eye_min_ms <= 0):
        raise ValueError('closed_eye_min_ms must be an explicit positive integer')
    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('dataset-root must be a directory')
    output = Path(output_dir).resolve()
    if output.is_relative_to(root):
        raise ValueError('Output must be outside the input dataset')
    if output.exists():
        raise FileExistsError('Output already exists; choose a NEW directory')
    report = read_json(audit_path)
    if (not isinstance(report, dict) or report.get('schema_version') != 'driveguard-dmd-audit-v1'
            or report.get('status') != 'ok' or report.get('counts', {}).get('errors') != 0
            or any(i.get('severity') == 'error' for i in report.get('issues', []))):
        raise ValueError('A successful DMD audit without errors is required')
    records = report.get('recordings')
    if not isinstance(records, list) or not records:
        raise ValueError('Audit has no recordings')
    split, assignments = load_split(split_path)
    observed = {r['subject_id'] for r in records}
    if observed - assignments.keys():
        raise ValueError(f'Unassigned subjects: {sorted(observed - assignments.keys())}')
    policy = deepcopy(read_json(DEFAULT_LABELS))
    policy['closed_eye_min_ms'] = closed_eye_min_ms
    jobs, skipped, seen = [], [], set()
    for record in records:
        subset, recording = record['subset'], record['recording_id']
        if subset not in TASK_SCOPE or (subset, recording) in seen:
            raise ValueError('Unsupported subset or duplicate recording in audit')
        seen.add((subset, recording))
        sources, primary = [], None
        for annotation in record['annotations']:
            fresh, path = _source_annotation(root, record, annotation)
            sources.append(dict(path=annotation['path'], sha256=fresh['sha256']))
            if annotation['kind'] == f'ann_{subset}.json':
                if primary is not None:
                    raise ValueError('Duplicate primary annotation')
                primary = fresh, path, annotation['path']
        if primary is None:
            skipped.append(dict(recording_id=recording, subset=subset, reason='missing_primary_annotation'))
            continue
        fresh, path, relative = primary
        # Reread only after hash comparison; a final hash detects modification during planning.
        offsets = camera_offsets(read_json(path))
        streams = {s['name']: s for s in fresh['streams']}
        for view in ('face', 'body'):
            stream = streams.get(view + '_camera')
            if stream is None:
                raise ValueError(f'Missing synchronization declaration for {view}')
            video_relative = (PurePosixPath(subset) / 'dmd' / PurePosixPath(stream['uri'])).as_posix()
            # Check the raw URI separately so absolute components cannot discard the prefix.
            _contained(root / subset / 'dmd', stream['uri'])
            video_path = _contained(root, video_relative)
            identity = _identity(video_relative)
            expected = (subset, record['group'], record['subject_id'], record['session'], recording, view + '.mp4')
            if identity != expected:
                raise ValueError('Video identity does not match recording/subject/view')
            if not video_path.is_file():
                skipped.append(dict(recording_id=recording, subset=subset, view=view, reason='missing_video'))
                continue
            stat = video_path.stat()
            if stat.st_size == 0:
                raise ValueError(f'Video is empty: {video_relative}')
            job_id = canonical_hash([subset, recording, view])[:24]
            jobs.append(dict(job_id=job_id, recording_id=recording, subject_id=record['subject_id'],
                subset=subset, view=view, split=assignments[record['subject_id']],
                video=video_relative, video_size_bytes=stat.st_size, video_mtime_ns=stat.st_mtime_ns,
                annotation=relative, annotation_sha256=fresh['sha256'], annotation_sources=sources,
                annotation_frame_offset=offsets[view], declared_frames=stream['total_frames'],
                driver_roi=None, driver_region_confirmed_exclusive=False,
                supervised_tasks=list(TASK_SCOPE[subset])))
        if _sha256(path) != fresh['sha256']:
            raise ValueError('Annotation changed during preparation')
    if not jobs:
        raise ValueError('No available face/body videos to prepare')
    result = dict(schema_version='driveguard-dmd-preparation-v1', status='needs_configuration',
        video_decoded=False, training_ready=False, audit_sha256=_sha256(audit_path),
        split=split, split_sha256=canonical_hash(split), label_policy=policy,
        label_policy_sha256=canonical_hash(policy), jobs=jobs, skipped=skipped,
        split_summary=dict(recordings_by_split=dict(Counter(assignments[r['subject_id']] for r in records)),
            jobs_by_split=dict(Counter(j['split'] for j in jobs)),
            configured_but_absent=sorted(assignments.keys() - observed, key=int)),
        needs_configuration=(['closed_eye_min_ms'] if closed_eye_min_ms is None else []) + ['driver_rois'],
        not_implemented_here=['batch_perception_execution', 'labeled_array_export', 'training_manifest_publication'])
    # Do not publish a success marker until both files are fully written.
    output.mkdir(parents=True, exist_ok=False)
    pending = output / 'preparation.pending.json'
    pending.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'README.md').write_text(
        '# DMD 训练准备清单\n\n'
        f'视频任务数：{len(jobs)}；跳过项：{len(skipped)}。\n\n'
        '这里只验证标注与路径，未解码、未生成特征或可训练 manifest。不要把 preparation.json 传给训练器。\n\n'
        '需要为各任务确认驾驶员 ROI 与排他性，并显式设置持续闭眼毫秒数。'
        '不要仅凭标注中的 driver ID 将排他性设为 true。\n\n'
        'train/val 是两份被试隔离划分；val 用于选 best.pt，不是独立 test。'
        'gaze 当前不生成监督；全量感知执行与最终数据导出需要后续接入。\n', encoding='utf-8')
    pending.rename(output / 'preparation.json')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output-dir', required=True, help='NEW directory outside Dataset')
    parser.add_argument('--split', default=str(DEFAULT_SPLIT))
    parser.add_argument('--closed-eye-min-ms', type=int, help='Explicit experiment threshold; no default or diagnostic claim')
    args = parser.parse_args(argv)
    try:
        report = prepare_dataset(audit_path=args.audit, dataset_root=args.dataset_root,
            output_dir=args.output_dir, split_path=args.split, closed_eye_min_ms=args.closed_eye_min_ms)
        print(json.dumps(dict(status=report['status'], training_ready=False, jobs=len(report['jobs']),
                              skipped=len(report['skipped']), split_summary=report['split_summary']), ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f'Preparation failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
