"""Read-only, per-window distraction diagnostics for a joint checkpoint.

Uses the training dataset and saved normalizer, never raw video or new labels.
summary.json is written last and marks a complete report.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from training.common.checkpoint import atomic_json
from training.common.config import TASKS
from training.common.data import file_sha256
from training.common.model_factory import load_model
from training.gaze.data import FeatureCorpus
from training.gaze.profile import FEATURE_NAMES, GROUP_INDICES
from training.gaze.head_evidence import HEAD_MODEL_KINDS,behavior_availability,head_statistics
from training.gaze.head_policy import resolve_head_policy, minimum_head_count

DIST = TASKS.index('distraction')
GROUPS = ('f', 'o', 'ho', 'fo')


def _positive(logit, threshold):
    # Training uses logit >= 0. FP32 sigmoid can round tiny negative logits to .5.
    boundary = math.log(threshold) - math.log1p(-threshold)
    return logit >= boundary


def _evidence(masks):
    masks = np.asarray(masks, dtype=bool)
    head = masks[:, 5:8].all(axis=1)
    iris = masks[:, 19:21].all(axis=1) | masks[:, 21:23].all(axis=1)
    return {
        'head_pose_valid_ratio': float(head.mean()),
        'iris_pair_valid_ratio': float(iris.mean()),
        'head_and_iris_valid_ratio': float((head & iris).mean()),
        'head_pose_current_valid': bool(head[-1]),
        'iris_pair_current_valid': bool(iris[-1]),
    }


def _window_row(dataset, index, inputs, target, labels):
    record, start = dataset.locate(index)
    stop = start + dataset.steps
    arrays = dataset.corpus.arrays(record)
    config = dataset.corpus.config['data']
    raw_masks = arrays['masks'][start:stop].astype(bool)
    effective = inputs['x'][:, 23:].astype(bool)
    known = bool(labels[DIST])
    row = {key: record[key] for key in ('record_id', 'recording_id', 'subject_id', 'view', 'subset', 'split')}
    row.update(window_index=index, start_slot=start, end_slot=stop-1,
               window_start_s=start / config['sample_rate_hz'],
               window_end_s=(stop-1) / config['sample_rate_hz'],
               label_known=known, target=int(target[DIST]) if known else None,
               behavior_available=bool(behavior_availability(inputs,dataset.corpus.config['model']['kind'],
                   dataset.corpus.config['model'].get('head_policy'))[DIST]))
    if dataset.corpus.config['model']['kind'] in HEAD_MODEL_KINDS:
        policy=resolve_head_policy(dataset.corpus.config['model'].get('head_policy'))
        stats=head_statistics(inputs['x'],policy)
        gap=int(stats['gap_steps'])
        reasons=[]
        if gap == dataset.steps: reasons.append('no_valid_head')
        else:
            if gap > policy['max_gap_steps']: reasons.append('head_gap_exceeded')
            if stats['valid_count'] < minimum_head_count(policy, dataset.steps): reasons.append('low_head_valid_ratio')
        row.update(h_eligible=bool(stats['eligible']),h_valid_ratio=float(stats['valid_ratio']),
                   h_current_valid=bool(stats['current_valid']),h_gap_steps=gap if gap < dataset.steps else None,
                   h_gap_ms=gap*1000/config['sample_rate_hz'] if gap < dataset.steps else None,
                   h_block_reasons='|'.join(reasons),
                   distraction_head_evidence_available=bool(stats['eligible']))
    for prefix, masks in (('raw', raw_masks), ('effective', effective)):
        row.update({prefix + '_' + key: value for key, value in _evidence(masks).items()})
        for column, name in enumerate(FEATURE_NAMES):
            row[f'{prefix}_{name}_valid_ratio'] = float(masks[:, column].mean())
    for column, name in enumerate(FEATURE_NAMES[5:8], start=5):
        row[name + '_last'] = float(arrays['values'][stop-1, column]) if effective[-1, column] else None
    for group, eligible, quality in zip(GROUPS, inputs['eligible'], inputs['quality']):
        row[group + '_eligible'] = bool(eligible)
        row[group + '_valid_ratio'] = float(quality[0])
        row[group + '_freshness'] = float(quality[1])
    row['face_age_ms'] = float(arrays['expert_age_ms'][stop-1, 1])
    profile = config['group_profile']
    supported = np.asarray(dataset.normalizer['supported'], dtype=bool)[list(GROUP_INDICES[0])]
    ratio = effective[:, GROUP_INDICES[0]].sum() / (dataset.steps * supported.sum()) if supported.any() else 0.
    reasons = []
    if not supported.any():
        reasons.append('no_supported_features')
    if not effective[-1, GROUP_INDICES[0]].any():
        reasons.append('no_current_valid_feature')
    if ratio < profile['min_valid_ratio']:
        reasons.append('low_valid_ratio')
    if row['face_age_ms'] > profile['max_age_ms']:
        reasons.append('stale_face')
    row['f_block_reasons'] = '|'.join(reasons) if not row['f_eligible'] else ''
    return row


class _Counts:
    def __init__(self):
        self.counts = dict.fromkeys(('windows', 'known', 'positive', 'negative', 'unknown',
                                    'available', 'known_unavailable', 'evaluated', 'tp', 'fp', 'tn', 'fn'), 0)

    def update(self, row, threshold):
        c = self.counts
        c['windows'] += 1
        c['available'] += int(row['behavior_available'])
        if not row['label_known']:
            c['unknown'] += 1
            return
        c['known'] += 1
        positive = row['target'] == 1
        c['positive' if positive else 'negative'] += 1
        if row['probability'] is None:
            c['known_unavailable'] += 1
            return
        c['evaluated'] += 1
        predicted = _positive(row['logit'], threshold)
        c[('tp' if predicted else 'fn') if positive else ('fp' if predicted else 'tn')] += 1

    def report(self):
        c = self.counts.copy()
        def divide(n, d):
            return n / d if d else None
        tp, fp, tn, fn = (c[name] for name in ('tp', 'fp', 'tn', 'fn'))
        # Match training's zero-division convention when any known input is usable.
        c.update(precision=(tp / (tp+fp) if tp+fp else 0.) if c['evaluated'] else None,
                 recall=(tp / (tp+fn) if tp+fn else 0.) if c['evaluated'] else None,
                 f1=(2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.) if c['evaluated'] else None,
                 false_positive_rate=divide(fp, fp+tn),
                 accuracy=divide(tp+tn, c['evaluated']),
                 input_coverage=divide(c['available'], c['windows']),
                 known_input_coverage=divide(c['evaluated'], c['known']),
                 detected_fraction_of_all_known_positives=divide(tp, c['positive']))
        return c


def diagnose(checkpoint, *, output_dir, manifest=None, data_root=None, backend='cpu',
             device_index=0, split='val', view='body', subset='distraction',
             batch_size=64, threshold=.5, progress=None):
    """Export the selected original-corpus windows; never refit or mutate inputs."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('threshold must be finite and between 0 and 1')
    if split not in ('train', 'val', 'test') or view not in ('body', 'face', 'all') or subset not in ('distraction', 'all'):
        raise ValueError('Invalid split/view/subset selection')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a NEW output_dir; existing reports are never overwritten')
    if progress:
        progress('Loading checkpoint and checking original feature hashes...')
    model, payload = load_model(checkpoint, backend, device_index)
    if payload.get('model_kind') not in ('relation_gaze_v1',*HEAD_MODEL_KINDS):
        raise ValueError('Diagnostics require a supported joint checkpoint')
    head_model=payload['model_kind'] in HEAD_MODEL_KINDS
    config = payload['config']
    manifest = manifest or config['runtime']['manifest']
    data_root = data_root or config['runtime']['data_root']
    if not manifest:
        raise ValueError('Supply --manifest; checkpoint has no saved manifest path')
    corpus = FeatureCorpus(config, manifest, data_root)
    try:
        if corpus.summary != payload['data_summary']:
            raise ValueError('Data fingerprint/provenance differs from the checkpoint corpus')
        dataset = corpus.dataset(split, payload['normalizer'])
        indices = []
        for index in range(len(dataset)):
            record, _ = dataset.locate(index)
            if (view == 'all' or record['view'] == view) and (subset == 'all' or record['subset'] == subset):
                indices.append(index)
        if not indices:
            raise ValueError('No windows match the split/view/subset selection')
        output.mkdir(parents=True, exist_ok=False)
        overall = _Counts()
        groups = {name: {} for name in ('view', 'subject_id', 'record_id', 'f_eligibility')}
        if head_model:
            groups['h_eligibility']={}
            groups['view_h_evidence']={}
        thresholds = {value / 10: _Counts() for value in range(1, 10)}
        device = next(model.parameters()).device
        with (output / 'windows.csv').open('x', encoding='utf-8-sig', newline='') as stream:
            writer = None
            for first in range(0, len(indices), batch_size):
                batch_inputs, rows = [], []
                for index in indices[first:first+batch_size]:
                    inputs, target, labels, _, _ = dataset[index]
                    rows.append(_window_row(dataset, index, inputs, target, labels))
                    batch_inputs.append(inputs)
                tensors = {key: torch.as_tensor(np.stack([item[key] for item in batch_inputs]), device=device)
                           for key in batch_inputs[0]}
                with torch.no_grad():
                    out = model(**tensors)
                    logits = out['behavior_logits'][:, DIST].cpu().tolist()
                    probabilities = torch.sigmoid(out['behavior_logits'][:, DIST]).cpu().tolist()
                    weights = out['weights'][:, DIST].cpu().tolist()
                    head_weights=out['head_dist_weight'].cpu().tolist() if head_model else [None]*len(rows)
                for row, logit, probability, gate, head_weight in zip(rows, logits, probabilities, weights,head_weights):
                    if not math.isfinite(logit) or not math.isfinite(probability) or not all(math.isfinite(value) for value in gate):
                        raise ValueError('Nonfinite prediction or gate weight')
                    row['logit'] = float(logit) if row['behavior_available'] else None
                    row['probability'] = float(probability) if row['behavior_available'] else None
                    row['prediction'] = int(_positive(logit, threshold)) if row['behavior_available'] else None
                    row['threshold'] = threshold
                    if not row['label_known']:
                        row['outcome'] = 'unknown'
                    elif not row['behavior_available']:
                        row['outcome'] = 'unavailable'
                    else:
                        row['outcome'] = ('T' if row['prediction'] == row['target'] else 'F') + ('P' if row['prediction'] else 'N')
                    row.update({group + '_dist_weight': float(value) for group, value in zip(GROUPS, gate)})
                    if head_model:
                        if not math.isfinite(head_weight): raise ValueError('Nonfinite head gate weight')
                        row['h_dist_weight']=float(head_weight)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    overall.update(row, threshold)
                    for name in groups:
                        if name in ('f_eligibility','h_eligibility'):
                            prefix=name[0]
                            group=prefix+('_eligible' if row[prefix+'_eligible'] else '_ineligible')
                        elif name == 'view_h_evidence':
                            group=row['view']+':'+(row['h_block_reasons'] or 'h_eligible')
                        else: group=row[name]
                        groups[name].setdefault(group, _Counts()).update(row, threshold)
                    for candidate, counts in thresholds.items():
                        counts.update(row, candidate)
                if progress and (first == 0 or first+batch_size >= len(indices) or first // 512 != (first+batch_size) // 512):
                    progress(f'Diagnosed {min(first+batch_size, len(indices))}/{len(indices)} windows')
        summary = {
            'schema_version': 'driveguard-distraction-diagnostics-v1', 'status': 'complete',
            'checkpoint': str(Path(checkpoint).resolve()), 'checkpoint_sha256': file_sha256(checkpoint),
            'checkpoint_epoch': payload['epoch'], 'device': str(device),
            'model_kind':payload['model_kind'],
            'manifest': str(Path(manifest).resolve()), 'data_root': str(corpus.root),
            'data_fingerprint': corpus.summary['fingerprint'],
            'synthetic': corpus.manifest['extractor_profile'].get('synthetic', False),
            'selection': {'split': split, 'view': view, 'subset': subset},
            'threshold': threshold, 'normalizer_supported': dict(zip(FEATURE_NAMES, payload['normalizer']['supported'])),
            'overall': overall.report(),
            'groups': {name: {key: counts.report() for key, counts in entries.items()} for name, entries in groups.items()},
            'thresholds': [{'threshold': key, **counts.report()} for key, counts in thresholds.items()],
            'notes': [
                'Window times are relative to first decoded frame PTS; end time is the label slot, inclusive.',
                'recording_id + view identify video; the feature manifest does not contain original video paths.',
                'Raw masks are cached extraction masks; effective masks match training after dependencies and normalization.',
                'Distraction availability is ANY original group, or eligible H in the head model; this is not proof of gaze evidence.',
                'In the head model F/O/HO/FO distraction weights plus h_dist_weight sum to one when available.',
                'Precision/recall/F1 exclude unknown labels and unavailable inputs, matching training.',
                'Predictions compare logits to log(threshold/(1-threshold)), avoiding rounded sigmoid boundary errors.',
                'All-known-positive detection fraction includes unavailable positives in its denominator.',
                'Threshold rows are diagnostic only; no threshold is selected or deployed.',
                'Validation participated in checkpoint selection; overlapping windows are not independent events.',
                'F block reasons describe filter conditions, not the causal explanation for an error.',
            ],
        }
        if head_model:
            summary['head_policy']=resolve_head_policy(config['model'].get('head_policy'))
            summary['notes'].append('h_gap_ms is the grid interval since the latest valid head slot, not exact source measurement age; missing evidence is never filled.')
        atomic_json(output / 'summary.json', summary)
        return summary
    finally:
        corpus.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--data-root')
    parser.add_argument('--backend', choices=('cpu', 'cuda', 'npu'), default='cpu')
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--view', choices=('body', 'face', 'all'), default='body')
    parser.add_argument('--subset', choices=('distraction', 'all'), default='distraction')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threshold', type=float, default=.5)
    args = parser.parse_args(argv)
    try:
        summary = diagnose(**vars(args), progress=lambda message: print(message, file=sys.stderr, flush=True))
        print(json.dumps({'summary': str(Path(args.output_dir).resolve() / 'summary.json'),
                          'windows': str(Path(args.output_dir).resolve() / 'windows.csv'),
                          'checkpoint_epoch': summary['checkpoint_epoch'], 'overall': summary['overall']}, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
