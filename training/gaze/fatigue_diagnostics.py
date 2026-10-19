"""Read-only validation fatigue evidence report using a trusted joint checkpoint.

No extraction, retraining, threshold search, or test-set evaluation is performed.
summary.json is written last, only after input hashes have been rechecked.
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
from training.common.data import file_sha256, resolve_data_root
from training.common.model_factory import load_model
from training.gaze.data import FeatureCorpus
from training.gaze.diagnostics import GROUPS, _Counts, _window_row
from training.gaze.head_evidence import HEAD_MODEL_KINDS, behavior_availability
from training.gaze.head_policy import resolve_head_policy
from training.gaze.profile import FEATURE_NAMES, GROUP_INDICES

FATIGUE = TASKS.index('fatigue')
FATIGUE_FEATURES = ('ear_left', 'ear_right', 'mar', 'perclos_4s',
                   'closed_duration_s', 'blink_rate_per_min')


def _quantiles(values):
    if not values:
        return {'count': 0, **dict.fromkeys(('min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max', 'mean'))}
    q = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return dict(count=len(values), **dict(zip(('min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max'), map(float, q))),
                mean=float(np.mean(values)))


def _fatigue_row(dataset, index, inputs, target, labels):
    # Reuse audited timing, per-feature mask ratios and F block reasons. The
    # source helper's distraction label/availability are explicitly replaced.
    row = _window_row(dataset, index, inputs, target, labels)
    row.pop('distraction_head_evidence_available', None)
    config = dataset.corpus.config
    known = bool(labels[FATIGUE])
    row.update(label_known=known, target=int(target[FATIGUE]) if known else None,
               behavior_available=bool(behavior_availability(inputs, config['model']['kind'],
                   config['model'].get('head_policy'))[FATIGUE]))
    row['label_class'] = ('positive' if row['target'] else 'negative') if known else 'unknown'
    record, start = dataset.locate(index)
    stop = start + dataset.steps
    arrays = dataset.corpus.arrays(record)
    raw = arrays['masks'][start:stop].astype(bool)
    effective = inputs['x'][:, len(FEATURE_NAMES):].astype(bool)
    fatigue_indices = [FEATURE_NAMES.index(name) for name in FATIGUE_FEATURES]
    for prefix, masks in (('raw', raw), ('effective', effective)):
        row[prefix + '_face_any_valid_slot_ratio'] = float(masks[:, GROUP_INDICES[0]].any(axis=1).mean())
        row[prefix + '_fatigue_any_valid_slot_ratio'] = float(masks[:, fatigue_indices].any(axis=1).mean())
        row[prefix + '_eye_pair_valid_slot_ratio'] = float(masks[:, :2].all(axis=1).mean())
        for name in FATIGUE_FEATURES:
            col = FEATURE_NAMES.index(name)
            valid = masks[:, col]
            values = arrays['values'][start:stop, col]
            row[f'{prefix}_{name}_current_valid'] = bool(valid[-1])
            row[f'{prefix}_{name}_last'] = float(values[-1]) if valid[-1] else None
            row[f'{prefix}_{name}_valid_mean'] = float(values[valid].mean()) if valid.any() else None
    return row


class _EvidenceCounts(_Counts):
    def __init__(self):
        super().__init__()
        self.scores = {key: [] for key in ('logit', 'probability', *(g + '_fatigue_weight' for g in GROUPS))}
        self.mask_sums = {}
        self.evidence = dict.fromkeys(('face_features_absent', 'fatigue_features_absent', 'f_eligible',
                                     'f_ineligible', 'f_eligible_below_zero', 'f_ineligible_below_zero',
                                     'f_eligible_zero_gate', 'f_ineligible_zero_gate'), 0)

    def update(self, row):
        super().update(row, .5)
        e = self.evidence
        e['face_features_absent'] += int(row['effective_face_any_valid_slot_ratio'] == 0)
        e['fatigue_features_absent'] += int(row['effective_fatigue_any_valid_slot_ratio'] == 0)
        eligibility = 'f_eligible' if row['f_eligible'] else 'f_ineligible'
        e[eligibility] += 1
        if row['behavior_available']:
            e[eligibility + '_below_zero'] += int(row['logit'] < 0)
            e[eligibility + '_zero_gate'] += int(row['f_fatigue_weight'] == 0)
            for name in self.scores:
                self.scores[name].append(row[name])
        for name, value in row.items():
            if name.startswith(('raw_', 'effective_')) and name.endswith(('_valid_ratio', '_valid_slot_ratio')):
                self.mask_sums[name] = self.mask_sums.get(name, 0.) + value

    def report(self):
        result = super().report()
        result.update(evidence=self.evidence.copy(), scores={name: _quantiles(v) for name, v in self.scores.items()},
                      mean_mask_coverage={name: total / result['windows'] for name, total in self.mask_sums.items()})
        return result


def _check_output(output, data_root, inputs):
    if output.exists():
        raise FileExistsError('Use a NEW output_dir; existing reports are never overwritten')
    if output.is_relative_to(data_root) or data_root.is_relative_to(output):
        raise ValueError('Output directory must not overlap the feature data root')
    if any(path == output or path.is_relative_to(output) for path in inputs):
        raise ValueError('Output directory must not overlap input files')


def diagnose(checkpoint, *, output_dir, manifest=None, data_root=None, backend='cpu',
             device_index=0, batch_size=64, split='val', view='all', subset='all', progress=None):
    """Diagnose original validation windows with the checkpoint's own preprocessing."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    if split != 'val' or view not in ('all', 'body', 'face') or subset not in ('all', 'drowsiness', 'distraction', 'gaze'):
        raise ValueError('Only validation split and supported view/subset selections are accepted')
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a NEW output_dir; existing reports are never overwritten')
    sidecar = checkpoint.with_suffix(checkpoint.suffix + '.sha256.json')
    unchanged = {p: file_sha256(p) for p in (checkpoint, sidecar)}
    if progress:
        progress('Loading trusted checkpoint and verifying original corpus fingerprint...')
    model, payload = load_model(checkpoint, backend, device_index)
    if payload.get('model_kind') not in ('relation_gaze_v1', *HEAD_MODEL_KINDS):
        raise ValueError('Fatigue diagnostics require a supported joint checkpoint')
    config = payload['config']
    manifest = manifest or config['runtime']['manifest']
    if not manifest:
        raise ValueError('Supply --manifest; checkpoint has no saved manifest path')
    manifest = Path(manifest).expanduser().resolve()
    root = resolve_data_root(data_root or config['runtime']['data_root'])
    _check_output(output, root, [checkpoint, sidecar, manifest])
    unchanged[manifest] = file_sha256(manifest)
    corpus = FeatureCorpus(config, manifest, root)
    try:
        if corpus.summary != payload['data_summary']:
            raise ValueError('Data fingerprint/provenance differs from the checkpoint corpus')
        for record in corpus.records:
            for key, path in record['paths'].items():
                unchanged[path] = corpus.summary['array_sha256'][record['record_id']][key]
        dataset = corpus.dataset('val', payload['normalizer'])
        indices = []
        for index in range(len(dataset)):
            record, _ = dataset.locate(index)
            if record['view'] in (('body', 'face') if view == 'all' else (view,)) and (subset == 'all' or record['subset'] == subset):
                indices.append(index)
        if not indices:
            raise ValueError('No validation windows match the view/subset selection')
        output.mkdir(parents=True, exist_ok=False)
        overall = _EvidenceCounts()
        groups = {key: {} for key in ('view', 'label_class', 'view_label', 'subject_id', 'subset',
                                     'view_subset_label', 'f_eligibility', 'view_label_f_eligibility')}
        device = next(model.parameters()).device
        with (output / 'windows.csv').open('x', encoding='utf-8-sig', newline='') as stream:
            writer = None
            for first in range(0, len(indices), batch_size):
                batch_inputs, rows = [], []
                for index in indices[first:first + batch_size]:
                    inputs, targets, labels, _, _ = dataset[index]
                    rows.append(_fatigue_row(dataset, index, inputs, targets, labels))
                    batch_inputs.append(inputs)
                tensors = {key: torch.as_tensor(np.stack([item[key] for item in batch_inputs]), device=device)
                           for key in batch_inputs[0]}
                with torch.no_grad():
                    predictions = model(**tensors)
                    logits = predictions['behavior_logits'][:, FATIGUE].cpu().tolist()
                    probabilities = torch.sigmoid(predictions['behavior_logits'][:, FATIGUE]).cpu().tolist()
                    weights = predictions['weights'][:, FATIGUE].cpu().tolist()
                for row, logit, probability, gate in zip(rows, logits, probabilities, weights):
                    if not all(math.isfinite(value) for value in (logit, probability, *gate)):
                        raise ValueError('Nonfinite fatigue prediction or gate weight')
                    available = row['behavior_available']
                    row.update(logit=float(logit) if available else None,
                               probability=float(probability) if available else None,
                               prediction=int(logit >= 0) if available else None,
                               threshold=.5, logit_threshold=0.)
                    row.update({group + '_fatigue_weight': float(value) for group, value in zip(GROUPS, gate)})
                    if not row['label_known']:
                        row['outcome'] = 'unknown'
                    elif not available:
                        row['outcome'] = 'unavailable'
                    else:
                        row['outcome'] = ('T' if row['prediction'] == row['target'] else 'F') + ('P' if row['prediction'] else 'N')
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    overall.update(row)
                    eligibility = 'f_eligible' if row['f_eligible'] else 'f_ineligible'
                    keys = dict(view=row['view'], label_class=row['label_class'],
                                view_label=row['view'] + ':' + row['label_class'], subject_id=row['subject_id'],
                                subset=row['subset'], view_subset_label=':'.join((row['view'], row['subset'], row['label_class'])),
                                f_eligibility=eligibility,
                                view_label_f_eligibility=':'.join((row['view'], row['label_class'], eligibility)))
                    for name, key in keys.items():
                        groups[name].setdefault(key, _EvidenceCounts()).update(row)
                if progress and (first == 0 or first + batch_size >= len(indices) or first // 512 != (first + batch_size) // 512):
                    progress(f'Diagnosed {min(first + batch_size, len(indices))}/{len(indices)} validation windows')
        if progress:
            progress('Rechecking input hashes across the full train/val/test corpus before publishing summary; read I/O may take time...')
        for path, expected in unchanged.items():
            if file_sha256(path) != expected:
                raise ValueError(f'Input changed during diagnosis: {path}')
        summary = dict(schema_version='driveguard-fatigue-diagnostics-v1', status='complete',
            checkpoint=str(checkpoint), checkpoint_sha256=unchanged[checkpoint], checkpoint_epoch=payload['epoch'],
            model_kind=payload['model_kind'], device=str(device), manifest=str(manifest), data_root=str(root),
            data_fingerprint=corpus.summary['fingerprint'], original_inputs_unchanged=True,
            synthetic=corpus.manifest['extractor_profile'].get('synthetic', False),
            selection=dict(split='val', view=view, subset=subset), threshold=.5, logit_threshold=0.,
            normalizer_supported=dict(zip(FEATURE_NAMES, payload['normalizer']['supported'])),
            feature_definitions=corpus.manifest['extractor_profile']['feature_definitions'],
            group_profile=config['data']['group_profile'], overall=overall.report(),
            groups={name: {key: counts.report() for key, counts in values.items()} for name, values in groups.items()},
            notes=[
                'Validation only; no threshold selection or test-set retuning. Validation participated in checkpoint selection.',
                'Window times are inclusive label-slot times relative to first decoded frame PTS, not independent events.',
                'Fatigue availability is any F/O/HO/FO group. H serves distraction only and never rescues fatigue availability.',
                'Feature mask coverage, prediction availability and raw label known/unknown are separate quantities.',
                'Raw masks are extraction masks. Effective masks include dependency/history freshness and saved normalizer support.',
                'EAR-pair history gaps suppress PERCLOS, closed duration and blink rate until the gap leaves the 100-slot history.',
                'Per-feature ratios use window slots; F quality uses supported feature cells, not an all-feature denominator.',
                'Scores and gate quantiles include available windows only; label_class groups separate unknown from negative.',
                'Confusion counts exclude unknown/unavailable windows. All-known-positive detection fraction includes unavailable positives.',
                'face_features_absent and fatigue_features_absent mean zero effective slots for any feature in the named set.',
                'F gate zero when ineligible is enforced masking; low eligible F weight is descriptive, not causal attribution.',
                'F eligibility can be driven by head/iris features without valid EAR/MAR; eligibility is not fatigue-feature completeness.',
                'No causal explanation is established by correlations between masks, gates, and scores.',
                'A complete summary requires unchanged input file hashes at the final check; partial CSV is not a complete report.',
            ])
        if payload['model_kind'] in HEAD_MODEL_KINDS:
            summary['head_policy'] = resolve_head_policy(config['model'].get('head_policy'), dataset.steps)
        summary.update(window_steps=dataset.steps, stride_steps=dataset.stride,
                       manifest_file_sha256=unchanged[manifest])
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
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--split', choices=('val',), default='val')
    parser.add_argument('--view', choices=('all', 'body', 'face'), default='all')
    parser.add_argument('--subset', choices=('all', 'drowsiness', 'distraction', 'gaze'), default='all')
    args = parser.parse_args(argv)
    try:
        summary = diagnose(**vars(args), progress=lambda message: print(message, file=sys.stderr, flush=True))
        print(json.dumps(dict(summary=str(Path(args.output_dir).resolve() / 'summary.json'),
                              windows=str(Path(args.output_dir).resolve() / 'windows.csv'),
                              checkpoint_epoch=summary['checkpoint_epoch'], overall=summary['overall']), allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
