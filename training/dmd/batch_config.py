"""Portable, explicit batch selection and normalized driver ROI configuration."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re

from inference.perception.runtime import load_runtime_config
from training.common.config import canonical_hash, read_json
from training.dmd.audit import _identity
from training.dmd.prepare import ROOT, DEFAULT_LABELS, TASK_SCOPE


def _relative(path, base):
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:  # Different Windows drives cannot be expressed relatively.
        return str(path)


def validate_preparation(prep):
    if not isinstance(prep, dict) or prep.get('schema_version') != 'driveguard-dmd-preparation-v1':
        raise ValueError('Unsupported preparation document')
    split = prep.get('split', {})
    if not isinstance(split, dict) or split.get('schema_version') != 'driveguard-dmd-split-v1':
        raise ValueError('Invalid preparation split')
    assignments = {}
    for name in ('train', 'val'):
        if not isinstance(split.get(name), list) or not split[name]:
            raise ValueError('Both subject splits must be nonempty')
        for subject in split[name]:
            if (not isinstance(subject, str) or not re.fullmatch('[1-9][0-9]*', subject)
                    or subject in assignments):
                raise ValueError('Invalid or leaking subject in preparation')
            assignments[subject] = name
    if prep.get('split_sha256') != canonical_hash(split):
        raise ValueError('Preparation split hash mismatch')
    jobs = prep.get('jobs')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('Preparation jobs must be nonempty')
    seen = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError('Invalid preparation job')
        identity = _identity(job.get('video', ''))
        annotation = _identity(job.get('annotation', ''))
        if (identity is None or annotation is None or identity[:5] != annotation[:5]
                or identity[0] != job.get('subset') or identity[2] != job.get('subject_id')
                or identity[4] != job.get('recording_id') or identity[5] != job.get('view', '') + '.mp4'
                or annotation[5] != 'ann_' + identity[0] + '.json'
                or job.get('split') != assignments.get(job.get('subject_id'))):
            raise ValueError('Preparation job identity/subject/split mismatch')
        jid = canonical_hash([job['subset'], job['recording_id'], job['view']])[:24]
        if jid != job.get('job_id') or jid in seen:
            raise ValueError('Duplicate or invalid preparation job_id')
        seen.add(jid)
        if job.get('supervised_tasks') != TASK_SCOPE[job['subset']]:
            raise ValueError('Preparation supervision scope mismatch')
    return prep


def init_config(preparation_path, output_path, selection='pilot'):
    source, output = Path(preparation_path).resolve(strict=True), Path(output_path).resolve()
    prep = validate_preparation(read_json(source))
    if selection not in ('pilot', 'all'):
        raise ValueError('selection must be pilot or all')
    selected, seen = [], set()
    for job in prep['jobs']:
        if not job['supervised_tasks']:
            continue
        key = (job['split'], job['subset'], job['view'])
        if selection == 'all' or key not in seen:
            selected.append(job)
            seen.add(key)
    if not selected:
        raise ValueError('No supervised jobs available')
    threshold = prep.get('label_policy', {}).get('closed_eye_min_ms')
    if threshold is None:
        threshold = read_json(DEFAULT_LABELS)['closed_eye_min_ms']
    if type(threshold) is not int or threshold <= 0:
        raise ValueError('Preparation closed_eye_min_ms must be a positive integer or legacy null')
    result = dict(schema_version='driveguard-dmd-batch-v1',
        preparation=_relative(source, output.parent), preparation_sha256=canonical_hash(prep),
        runtime_config=_relative(ROOT / 'configs/perception/dmd_cpu.json', output.parent),
        closed_eye_min_ms=threshold,
        jobs={job['job_id']: dict(video=job['video'], normalized_roi=[0, 0, 1, 1],
                                  confirmed_exclusive=False) for job in selected})
    # An existing config may contain hours of manual ROI review. Never replace it.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return result


def load_batch(config_path):
    path = Path(config_path).resolve(strict=True)
    config = read_json(path)
    if (not isinstance(config, dict) or set(config) != {'schema_version', 'preparation',
            'preparation_sha256', 'runtime_config', 'closed_eye_min_ms', 'jobs'}
            or config['schema_version'] != 'driveguard-dmd-batch-v1'):
        raise ValueError('Invalid batch configuration')
    threshold = config['closed_eye_min_ms']
    if type(threshold) is not int or threshold <= 0:
        raise ValueError('closed_eye_min_ms must be a positive integer')
    prep = validate_preparation(read_json(path.parent / config['preparation']))
    if canonical_hash(prep) != config['preparation_sha256']:
        raise ValueError('Preparation changed since batch configuration was generated')
    jobs = {j['job_id']: j for j in prep['jobs']}
    selected = []
    if not isinstance(config['jobs'], dict) or not config['jobs']:
        raise ValueError('Select at least one supervised job')
    for jid, setting in config['jobs'].items():
        if jid not in jobs or not jobs[jid]['supervised_tasks']:
            raise ValueError('Unknown or unsupervised job selected')
        if (not isinstance(setting, dict) or set(setting) != {'video', 'normalized_roi', 'confirmed_exclusive'}
                or setting['video'] != jobs[jid]['video'] or type(setting['confirmed_exclusive']) is not bool):
            raise ValueError('Invalid per-job ROI configuration')
        roi = setting['normalized_roi']
        if (not isinstance(roi, list) or len(roi) != 4
                or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in roi)
                or roi[0] >= roi[2] or roi[1] >= roi[3]):
            raise ValueError('normalized_roi must be x1,y1,x2,y2 in [0,1] with positive area')
        selected.append(deepcopy(jobs[jid]))
    runtime = load_runtime_config(path.parent / config['runtime_config'])
    policy = deepcopy(read_json(DEFAULT_LABELS))
    policy['closed_eye_min_ms'] = threshold
    return dict(config=config, preparation=prep, selected=selected, runtime=runtime, label_policy=policy)
