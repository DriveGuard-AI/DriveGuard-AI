"""Read-only DMD annotation audit, without ML packages or video decoding."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys

from training.dmd.annotations import parse_annotation


SUBSETS = ('distraction', 'drowsiness', 'gaze')
KINDS = {'ann_distraction.json', 'ann_drowsiness.json', 'ann_gaze.json', 'ann_hands.json',
         'body.mp4', 'face.mp4', 'hands.mp4', 'mosaic.avi'}


def _local_files(root, issues):
    def walk_error(exc):
        issues.append(dict(severity='error', code='DIRECTORY_READ_ERROR',
                           path=str(exc.filename), message=str(exc)))
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
                if name in dirs:
                    dirs.remove(name)
                issues.append(dict(severity='warning', code='SYMLINK_SKIPPED', path=relative,
                                   message='Links are not traversed or opened'))
            elif name in files:
                yield path, relative
        dirs.sort()


def _identity(relative):
    parts = PurePosixPath(relative).parts
    if len(parts) != 6:
        return None
    subset, dmd, group, subject, session, filename = parts
    if (subset not in SUBSETS or dmd != 'dmd' or not re.fullmatch(r'g[A-Za-z0-9]+', group)
            or not re.fullmatch(r'[0-9]+', subject) or not re.fullmatch(r's[0-9]+', session)
            or '_rgb_' not in filename):
        return None
    prefix, kind = filename.rsplit('_rgb_', 1)
    if not prefix.startswith(f'{group}_{subject}_{session}_') or kind not in KINDS:
        return None
    return subset, group, subject, session, prefix, kind


def _check_streams(annotation, record, root, files, issues, source):
    for stream in annotation['streams']:
        uri = stream['uri']
        parts = PurePosixPath(uri)
        # Timestamp colons are legitimate in the filename, not in a URI scheme/drive.
        unsafe = ('\\' in uri or '\x00' in uri or parts.is_absolute()
                  or PureWindowsPath(uri).drive or '..' in parts.parts
                  or not parts.parts or ':' in parts.parts[0])
        if unsafe:
            issues.append(dict(severity='error', code='UNSAFE_URI', path=source,
                               message='Camera URI must be a local relative dataset path', stream=stream['name']))
            stream['file_exists'] = False
            continue
        relative = (PurePosixPath(record['subset']) / 'dmd' / parts).as_posix()
        candidate = root / relative
        try:
            contained = candidate.resolve().is_relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            issues.append(dict(severity='error', code='UNSAFE_URI', path=source,
                               message=f'Camera URI cannot be resolved: {exc}', stream=stream['name']))
            stream['file_exists'] = False
            continue
        if not contained:
            issues.append(dict(severity='error', code='UNSAFE_URI', path=source,
                               message='Camera URI resolves outside dataset', stream=stream['name']))
            stream['file_exists'] = False
            continue
        stream['file_exists'] = relative in files
        identity = _identity(relative)
        if identity is None or identity[:5] != (
                record['subset'], record['group'], record['subject_id'], record['session'], record['recording_id']):
            issues.append(dict(severity='error', code='STREAM_RECORDING_MISMATCH', path=source,
                               message='Camera URI refers to a different recording', stream=stream['name']))
        expected_view = {'face_camera': 'face.mp4', 'body_camera': 'body.mp4',
                         'hands_camera': 'hands.mp4'}.get(stream['name'])
        if identity is not None and (identity[5] not in ('body.mp4', 'face.mp4', 'hands.mp4', 'mosaic.avi')
                                     or (expected_view is not None and identity[5] != expected_view)):
            issues.append(dict(severity='error', code='STREAM_VIEW_MISMATCH', path=source,
                               message='Camera URI is not the expected video view', stream=stream['name']))
        if not stream['file_exists']:
            issues.append(dict(severity='warning', code='MISSING_VIDEO', path=source,
                               message='Declared video is absent or was skipped', video=relative))


def _aggregate(records):
    subjects = {}
    for record in records:
        subject = subjects.setdefault(record['subject_id'], dict(subject_id=record['subject_id'],
            recordings=0, subsets=set(), groups=set(), labels={}))
        subject['recordings'] += 1
        subject['subsets'].add(record['subset'])
        subject['groups'].add(record['group'])
        for annotation in record['annotations']:
            # Keep source axes separate; gaze and hands files are not extra recordings.
            namespace = record['subset'] + '/' + annotation['kind']
            labels = subject['labels'].setdefault(namespace, {})
            for label in annotation['labels']:
                total = labels.setdefault(label['type'], dict(covered_frames=0, recordings=0))
                total['covered_frames'] += label['covered_frames']
                total['recordings'] += 1
    result = []
    for subject in sorted(subjects.values(), key=lambda s: (int(s['subject_id']), s['subject_id'])):
        subject['subsets'] = sorted(subject['subsets'])
        subject['groups'] = sorted(subject['groups'])
        result.append(subject)
    return result


def _cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ').replace('<', '&lt;').replace('>', '&gt;')


def _markdown(report):
    lines = ['# DMD 标注审计报告', '',
             '只读取标注与文件信息；未解码视频，未生成训练标签，未验证训练精度。', '',
             f"状态：{report['status']}；计数：`{json.dumps(report['counts'], ensure_ascii=False)}`", '',
             '## 被试与原始标签覆盖', '',
             '覆盖帧数为原始标注闭区间的并集计数，不是秒数或有效训练窗口数。不同标签轴可以重叠。', '',
             '| 被试 | 录制数 | 子集 |', '| --- | ---: | --- |']
    for subject in report['subjects']:
        lines.append(f"| {_cell(subject['subject_id'])} | {subject['recordings']} | {', '.join(subject['subsets'])} |")
    for subject in report['subjects']:
        lines += ['', f"### 被试 {_cell(subject['subject_id'])}", '',
                  '| 标注来源 | 原始类型 | 覆盖帧数 | 录制数 |', '| --- | --- | ---: | ---: |']
        for namespace, labels in sorted(subject['labels'].items()):
            for label, counts in sorted(labels.items()):
                lines.append(f"| {_cell(namespace)} | {_cell(label)} | {counts['covered_frames']} | {counts['recordings']} |")
    lines += ['', '## 诊断', '', '| 级别 | 代码 | 路径 | 说明 |', '| --- | --- | --- | --- |']
    for issue in report['issues']:
        lines.append('| ' + ' | '.join(_cell(issue.get(k, '')) for k in ('severity', 'code', 'path', 'message')) + ' |')
    if not report['issues']:
        lines.append('| — | — | — | 无已识别问题；不等于视频完整性验证 |')
    lines += ['', '## 后续', '',
              '依据原始标签确认四任务正/负/未知规则与摄像头同步方向，再固定被试划分。',
              '两份划分：train + 留出评估；当前训练器以 val 选 best.pt，不另留 test。',
              'not_valid / unclassified 不自动转为负例；物体存在不自动转为使用行为。', '']
    return '\n'.join(lines)


def audit_dataset(dataset_root, output_dir, progress=None):
    root = Path(dataset_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('dataset-root must be a directory')
    output = Path(output_dir).resolve()
    if output.is_relative_to(root):
        raise ValueError('Output must be outside the input dataset')
    if output.exists():
        raise FileExistsError('Output already exists; choose a NEW directory')
    issues, records, files = [], {}, set()
    annotations = []
    for path, relative in _local_files(root, issues):
        files.add(relative)
        identity = _identity(relative)
        if identity is None:
            issues.append(dict(severity='warning', code='UNRECOGNIZED_LAYOUT', path=relative,
                               message='File does not match the supported DMD recording layout'))
            continue
        subset, group, subject, session, prefix, kind = identity
        record = records.setdefault((subset, prefix), dict(subset=subset, group=group,
            subject_id=subject, session=session, recording_id=prefix, files={}, annotations=[]))
        record['files'][kind] = relative
        try:
            if path.stat().st_size == 0:
                issues.append(dict(severity='error', code='EMPTY_FILE', path=relative, message='File is empty'))
        except OSError as exc:
            issues.append(dict(severity='error', code='FILE_STAT_ERROR', path=relative, message=str(exc)))
        if kind.startswith('ann_'):
            allowed = ('ann_gaze.json', 'ann_hands.json') if subset == 'gaze' else (f'ann_{subset}.json',)
            if kind not in allowed:
                issues.append(dict(severity='error', code='ANNOTATION_KIND_MISMATCH', path=relative,
                                   message='Annotation kind disagrees with subset directory'))
            annotations.append((path, relative, record, kind))
    if not annotations:
        issues.append(dict(severity='error', code='NO_ANNOTATIONS', path='.', message='No supported annotation files found'))
    parsed = 0
    for index, (path, relative, record, kind) in enumerate(sorted(annotations, key=lambda a: a[1]), 1):
        if progress:
            progress(f'[{index}/{len(annotations)}] {relative}')
        try:
            annotation = parse_annotation(path)
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            issues.append(dict(severity='error', code='ANNOTATION_READ_ERROR', path=relative, message=str(exc)))
            continue
        parsed += 1
        annotation.update(path=relative, kind=kind)
        record['annotations'].append(annotation)
        issues.extend(dict(item, path=relative) for item in annotation['issues'])
        if annotation['name'] != record['recording_id'] + '_' + kind[4:-5]:
            issues.append(dict(severity='warning', code='METADATA_RECORDING_MISMATCH', path=relative,
                               message='metadata.name differs from recording filename and annotation kind'))
        if annotation['driver_ids'] != [record['subject_id']]:
            issues.append(dict(severity='error', code='SUBJECT_MISMATCH', path=relative,
                               message='Driver annotation does not match directory subject',
                               driver_ids=annotation['driver_ids'], directory_subject=record['subject_id']))
        _check_streams(annotation, record, root, files, issues, relative)
    ordered = [records[key] for key in sorted(records)]
    for record in ordered:
        expected = ('ann_gaze.json', 'ann_hands.json') if record['subset'] == 'gaze' else (f"ann_{record['subset']}.json",)
        for name in expected:
            if name not in record['files']:
                issues.append(dict(severity='warning', code='MISSING_ANNOTATION',
                    path=record['recording_id'], message=f'Missing {name}', subset=record['subset']))
        for name in ('body.mp4', 'face.mp4', 'hands.mp4'):
            if name not in record['files']:
                issues.append(dict(severity='warning', code='MISSING_VIDEO',
                    path=record['recording_id'], message=f'Missing {name}', subset=record['subset']))
        declarations = {}
        for annotation in record['annotations']:
            for stream in annotation['streams']:
                declaration = (stream['uri'], stream['total_frames'], stream['frame_shift'])
                previous = declarations.setdefault(stream['name'], declaration)
                if previous != declaration:
                    issues.append(dict(severity='warning', code='STREAM_DECLARATION_MISMATCH',
                        path=annotation['path'], message='Same recording has conflicting camera declarations',
                        stream=stream['name']))
    subjects = _aggregate(ordered)
    for subject in subjects:
        if len(subject['groups']) > 1:
            issues.append(dict(severity='error', code='SUBJECT_GROUP_MISMATCH', path=subject['subject_id'],
                               message='Subject number appears in multiple groups', groups=subject['groups']))
    levels = Counter(issue['severity'] for issue in issues)
    report = dict(schema_version='driveguard-dmd-audit-v1',
        created_at=datetime.now(timezone.utc).isoformat(), dataset_root=str(root),
        status='errors' if levels['error'] else 'ok', video_decoded=False, training_ready=False,
        counts=dict(files=len(files), recordings=len(records), subjects=len(subjects),
                    annotation_files=len(annotations), parsed_annotations=parsed,
                    errors=levels['error'], warnings=levels['warning']),
        subsets={subset: dict(recordings=sum(r['subset'] == subset for r in ordered),
                             subjects=sorted({r['subject_id'] for r in ordered if r['subset'] == subset}))
                 for subset in SUBSETS},
        subjects=subjects, recordings=ordered, issues=issues)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'audit.md').write_text(_markdown(report), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output-dir', required=True, help='NEW directory outside the dataset')
    args = parser.parse_args(argv)
    try:
        report = audit_dataset(args.dataset_root, args.output_dir,
                               progress=lambda message: print(message, file=sys.stderr, flush=True))
        print(json.dumps(dict(status=report['status'], **report['counts'],
                              output_dir=str(Path(args.output_dir).resolve())), ensure_ascii=False))
        return 2 if report['counts']['errors'] else 0
    except (OSError, ValueError) as exc:
        print(f'Audit failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
