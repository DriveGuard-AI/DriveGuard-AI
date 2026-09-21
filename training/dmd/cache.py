"""Verify immutable DMD attempts and publish only trainable feature manifests."""
import json
from pathlib import Path
import re
import stat
from uuid import uuid4

import numpy as np

from inference.perception.profile import validate_profile
from training.common.config import TASKS, canonical_hash, read_json, validate_config
from training.common.data import FeatureCorpus, file_sha256
from training.dmd.labels import MAX_OUTPUT_SLOTS
from training.relation.profile import validate_group_profile

ARRAYS = {
    'values': 'features/values.npy', 'masks': 'features/masks.npy',
    'expert_age_ms': 'features/expert_age_ms.npy', 'targets': 'targets.npy',
    'label_masks': 'label_masks.npy', 'source_pts_ms': 'source_pts_ms.npy',
}
RAW_ARRAYS = {name: name + '.npy' for name in
              ('values', 'masks', 'targets', 'label_masks', 'expert_age_ms')}
METADATA = ('subject_id', 'recording_id', 'view', 'subset', 'split')


def write_json(path, value):
    """Create one JSON artifact exclusively; never replace an existing file."""
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(encoded)


def _unlinked(path):
    """Reject links and Windows reparse points in every existing path component."""
    path = Path(path).absolute()
    for component in (path, *path.parents):
        if component.is_symlink() or (hasattr(component, 'is_junction') and component.is_junction()):
            raise ValueError('Linked cache paths are unsupported')
        attributes = getattr(component.lstat(), 'st_file_attributes', 0)
        if attributes & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
            raise ValueError('Reparse cache paths are unsupported')
    return path.resolve()


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _hashed_paths(root, paths, hashes, expected):
    if paths != expected or not isinstance(hashes, dict) or set(hashes) != set(expected):
        raise ValueError('Unexpected array layout')
    result = {}
    for name, relative in expected.items():
        path = _unlinked(root / relative)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Array outside cache directory')
        if not _digest(hashes[name]) or file_sha256(path) != hashes[name]:
            raise ValueError('Array SHA256 mismatch')
        result[name] = path
    return result


def _array(path):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError('Expected one NPY array, not an archive')
    return array


def _check_dimensions(paths, raw_paths, report):
    """Read only mmap headers or bounded per-recording blocks, closing all handles."""
    opened = []
    try:
        arrays = {}
        for name, path in paths.items():
            arrays[name] = _array(path)
            opened.append(arrays[name])
        feature = report['feature_record']
        slots = feature['slots']
        if type(slots) is not int or not 100 <= slots <= MAX_OUTPUT_SLOTS:
            raise ValueError('Invalid slot count')
        dimensions = {'values': (slots, len(feature['feature_schema']['names'])),
                      'masks': (slots, len(feature['feature_schema']['names'])),
                      'expert_age_ms': (slots, 3), 'targets': (slots, 4), 'label_masks': (slots, 4)}
        for name, shape in dimensions.items():
            if arrays[name].shape != shape:
                raise ValueError('Array dimensions differ from feature record')
        for name in ('values', 'expert_age_ms', 'targets'):
            if arrays[name].dtype != np.float32:
                raise ValueError('Invalid float array dtype')
        for name in ('masks', 'label_masks'):
            if arrays[name].dtype not in (np.dtype('bool'), np.dtype('uint8'), np.dtype('float32')):
                raise ValueError('Invalid binary array dtype')
        pts = arrays['source_pts_ms']
        if pts.ndim != 1 or pts.dtype != np.int64 or not len(pts) or pts[0] != 0:
            raise ValueError('Source PTS must start at zero in int64 milliseconds')
        if np.any(pts[1:] <= pts[:-1]) or int(pts[-1]) // 40 + 1 != slots:
            raise ValueError('Source PTS and exported slots disagree')
        if (type(feature['first_timestamp_ms']) is not int or feature['first_timestamp_ms'] != 0
                or type(feature['last_timestamp_ms']) is not int
                or feature['last_timestamp_ms'] != (slots - 1) * 40):
            raise ValueError('Export timestamp range disagrees with slots')
        for start in range(0, slots, 4096):
            chunk = {name: array[start:start + 4096] for name, array in arrays.items()
                     if name != 'source_pts_ms'}
            if any(not np.isfinite(array).all() for array in chunk.values()):
                raise ValueError('Nonfinite array data')
            for name in ('masks', 'targets', 'label_masks'):
                if not np.isin(chunk[name], [0, 1]).all():
                    raise ValueError('Nonbinary array data')
            if np.any(chunk['targets'][chunk['label_masks'] == 0] != 0):
                raise ValueError('Unknown labels must have zero targets')
            ages = chunk['expert_age_ms']
            if np.any(ages < 0) or np.any(ages > feature['group_profile']['max_age_ms'] + 1):
                raise ValueError('Expert ages outside profile range')
        for name in ('targets', 'label_masks'):
            raw = _array(raw_paths[name])
            opened.append(raw)
            if raw.shape != (slots, 4) or raw.dtype != (np.float32 if name == 'targets' else np.uint8):
                raise ValueError('Raw unknown label layout is invalid')
            if np.any(raw != 0):
                raise ValueError('Raw feature export must remain unlabeled')
    finally:
        for array in opened:
            mmap = getattr(array, '_mmap', None)
            if mmap is not None:
                mmap.close()


def validate_complete(attempt_dir, fingerprint):
    """Return a verified completion or None; never repair or overwrite a cache."""
    try:
        if not _digest(fingerprint):
            return None
        root = _unlinked(attempt_dir)
        if (root / 'failure.json').exists() or (root / 'failure.json').is_symlink():
            return None
        report = read_json(_unlinked(root / 'complete.json'))
        if (not isinstance(report, dict)
                or report.get('schema_version') != 'driveguard-dmd-complete-v1'
                or report.get('fingerprint') != fingerprint
                or not isinstance(report.get('label_summary'), dict)):
            return None
        feature, job = report['feature_record'], report['job']
        if not isinstance(feature, dict) or not isinstance(job, dict):
            return None
        required = {'schema_version', 'status', 'slots', 'recording_id', 'view',
                    'first_timestamp_ms', 'last_timestamp_ms', 'image_size', 'driver_roi',
                    'expert_models', 'feature_schema', 'extractor_profile', 'profile_sha256',
                    'group_profile', 'group_profile_sha256', 'arrays', 'array_sha256', 'labels'}
        if not required <= set(feature) or set(feature) - required - {'provenance'}:
            return None
        size, roi, models = feature['image_size'], feature['driver_roi'], feature['expert_models']
        if (not isinstance(size, list) or len(size) != 2
                or any(type(value) is not int or value <= 0 for value in size)
                or not isinstance(roi, list) or len(roi) != 4
                or any(type(value) not in (int, float) for value in roi)
                or not (0 <= roi[0] < roi[2] <= size[0] and 0 <= roi[1] < roi[3] <= size[1])
                or not isinstance(models, dict) or set(models) != {'face', 'objects', 'hands'}
                or any(not isinstance(value, str) or not value.strip() for value in models.values())
                or not isinstance(feature['labels'], str) or not feature['labels'].strip()
                or ('provenance' in feature and not isinstance(feature['provenance'], dict))):
            return None
        if any(not isinstance(job.get(key), str) or not job[key].strip()
               for key in ('job_id', *METADATA)):
            return None
        if job['view'] not in ('face', 'body', 'hands') or job['split'] not in ('train', 'val', 'test'):
            return None
        if job['subset'] not in ('distraction', 'drowsiness', 'gaze'):
            return None
        if (feature['schema_version'] != 'driveguard-feature-record-v2'
                or feature['status'] != 'complete_unlabeled'
                or any(feature[key] != job[key] for key in ('recording_id', 'view'))):
            return None
        persisted = read_json(_unlinked(root / 'features/record.json'))
        if canonical_hash(persisted) != canonical_hash(feature):
            return None
        if (root / 'features/incomplete.json').exists():
            return None
        profile = validate_profile(feature['extractor_profile'])
        group = validate_group_profile(feature['group_profile'], profile['feature_names'])
        expected_schema = {'version': profile['feature_schema_version'],
                           'names': profile['feature_names'], 'sample_rate_hz': 25}
        if (feature['profile_sha256'] != canonical_hash(profile)
                or feature['group_profile_sha256'] != canonical_hash(group)
                or feature['feature_schema'] != expected_schema
                or profile['max_age_ms'] != group['max_age_ms']):
            return None
        paths = _hashed_paths(root, report['arrays'], report['array_sha256'], ARRAYS)
        raw_paths = _hashed_paths(root / 'features', feature['arrays'], feature['array_sha256'], RAW_ARRAYS)
        _check_dimensions(paths, raw_paths, report)
        return report
    except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError, EOFError):
        return None


def publish_manifest(output_root, completed, training_config, label_policy):
    """Publish a fresh manifest only after the real loader and coverage gate pass."""
    config = validate_config(training_config)
    if config['schema_version'] != 'driveguard-training-v2':
        raise ValueError('DMD publication requires a v2 training configuration')
    if not isinstance(label_policy, dict) or not label_policy:
        raise ValueError('A nonempty label policy is required')
    canonical_hash(label_policy)
    if not completed:
        raise ValueError('No completed recordings')
    root = _unlinked(output_root)
    records, sources, identities = [], [], set()
    first = None
    for attempt, supplied in completed:
        if not isinstance(supplied, dict):
            raise ValueError('Invalid supplied completion report')
        report = validate_complete(attempt, supplied.get('fingerprint'))
        if report is None or canonical_hash(supplied) != canonical_hash(report):
            raise ValueError('Invalid or changed completion report')
        attempt = _unlinked(attempt)
        if not attempt.is_relative_to(root):
            raise ValueError('Completed attempt lies outside output root')
        feature, job = report['feature_record'], report['job']
        if job['job_id'] in identities:
            raise ValueError('Duplicate completed job')
        identities.add(job['job_id'])
        if first is None:
            first = feature
        elif any(feature[key] != first[key] for key in
                 ('feature_schema', 'extractor_profile', 'group_profile', 'group_profile_sha256')):
            raise ValueError('Inconsistent feature or group profile across recordings')
        record = {'record_id': job['job_id'], **{key: job[key] for key in METADATA}}
        record.update({key: (attempt / report['arrays'][key]).relative_to(root).as_posix()
                       for key in RAW_ARRAYS})
        records.append(record)
        sources.append({'job_id': job['job_id'], 'fingerprint': report['fingerprint'],
                        'attempt': attempt.relative_to(root).as_posix()})
    manifest = {
        'schema_version': 'driveguard-feature-manifest-v2',
        'dataset_version': 'dmd-event-proxy-v1-' + canonical_hash({'sources': sources, 'labels': label_policy}),
        'feature_schema': first['feature_schema'], 'extractor_profile': first['extractor_profile'],
        'group_profile': first['group_profile'], 'group_profile_sha256': first['group_profile_sha256'],
        'label_mapping': {'version': 'driveguard-behavior-v1', 'tasks': list(TASKS), 'rules': label_policy},
        'records': records,
    }
    parent = root / 'manifests'
    parent.mkdir(exist_ok=True)
    _unlinked(parent)
    publication = parent / uuid4().hex
    publication.mkdir(exist_ok=False)
    candidate = publication / 'candidate.json'
    write_json(candidate, manifest)
    corpus = None
    diagnostic = {'status': 'invalid_corpus', 'sources': sources,
                  'active_tasks': config['training']['active_tasks'],
                  'training_config_sha256': canonical_hash(config)}
    try:
        corpus = FeatureCorpus(config, candidate, root)
        normalizer = corpus.fit_normalizer()
        coverage = corpus.check_supervision([], normalizer)
        diagnostic['coverage'] = coverage
        missing = [f'{split} {task}' for split in ('train', 'val')
                   for task in config['training']['active_tasks']
                   if coverage[split]['positive'][TASKS.index(task)] == 0
                   or coverage[split]['negative'][TASKS.index(task)] == 0]
        if missing:
            diagnostic['status'] = 'insufficient_supervision'
            raise ValueError('Usable positive and negative supervision required: ' + ', '.join(missing))
        diagnostic['status'] = 'complete'
        diagnostic['corpus'] = corpus.summary
    except (OSError, ValueError, TypeError) as error:
        diagnostic['error'] = str(error)
        report_path = publication / 'report.json'
        write_json(report_path, diagnostic)
        raise ValueError(f'{error}. Diagnostic report: {report_path}') from error
    finally:
        if corpus is not None:
            corpus.close()
    write_json(publication / 'coverage.json', coverage)
    write_json(publication / 'report.json', diagnostic)
    manifest_path = publication / 'manifest.json'
    candidate.rename(manifest_path)
    return {'manifest': str(manifest_path), 'coverage': coverage,
            'report': str(publication / 'report.json')}
