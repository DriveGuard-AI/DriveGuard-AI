"""Explain missed eye events using one verified action checkpoint; no training."""
import argparse
from contextlib import contextmanager
import csv
import json
from pathlib import Path
import sys

import numpy as np

from training.common.checkpoint import atomic_json
from training.common.config import read_json
from training.common.data import file_sha256
from training.visual.action_eval import evaluate_events, replay_events, action_metrics, subtype_metrics
from training.visual.action_labels import ActionLabels
from training.visual.action_train import load_actions
from training.visual.data import PreparedData
from training.visual.eye_scores import score_summary
from training.visual.fatigue_spatial_suite import _summary
from training.visual.head_adaptation import paired_changes
from training.visual.head_adapt_train import _verify_dataset_identity
from training.visual.head_frames import FrameCache, FrameDataset
from training.visual.report import write_rows


DEFAULT_VARIANT = 'B_action_supervision-seed42-best-event'
IDENTITY = ('variant', 'arm', 'seed', 'checkpoint_kind', 'checkpoint', 'checkpoint_sha256', 'epoch')


def validate_selection(report, selected, payload, duration):
    policy = dict(action_logit_threshold=0., closed_eye_min_ms=duration,
                  reset_on_source_gap_ms=80, threshold_search=False)
    if (report.get('schema_version') != 'driveguard-fatigue-actions-comparison-v1'
            or report.get('status') != 'complete' or report.get('split') != 'val'
            or report.get('task') != 'fatigue'
            or selected['arm'] != 'B_action_supervision_event'
            or not selected['variant'].endswith('-best-event') or selected['checkpoint_kind'] != 'best'
            or payload['kind'] != 'weights' or payload.get('checkpoint_role') != 'best'
            or payload['epoch'] != selected['epoch'] or payload['settings']['seed'] != selected['seed']
            or not payload['auxiliary'] or payload['selection_metric'] != 'action_validation_bce'
            or selected['selection_metric'] != 'action_validation_bce'
            or report['plan']['provenance'] != payload['provenance']
            or report['plan']['event_policy'] != policy):
        raise ValueError('Action diagnostic reference metadata/provenance/policy mismatch')


@contextmanager
def _inputs(source, variant, dataset_root, backend, device_index):
    digest = file_sha256(source)
    report = read_json(source)
    matches = [r for r in report['comparisons'] if r['variant'] == variant]
    if len(matches) != 1: raise ValueError('Selected event variant missing or duplicated')
    selected = matches[0]
    path = Path(selected['checkpoint']).resolve(strict=True)
    if file_sha256(path) != selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint checksum mismatch')
    model, payload = load_actions(path, backend, device_index)
    provenance = payload['provenance']
    validate_selection(report, selected, payload, report['plan']['event_policy']['closed_eye_min_ms'])
    base = cache = None
    try:
        print(json.dumps(dict(stage='verify_head_cache_and_labels', checkpoint=str(path))), flush=True)
        cache = FrameCache(provenance['backbone_provenance']['frame_cache'])
        run = read_json(cache.root/'input.json')
        if Path(run['dataset_root']).resolve() != Path(dataset_root).resolve():
            raise ValueError('Dataset root differs from original head pixel cache')
        base = PreparedData(run['prepared_root'])
        for split in ('train', 'val'):
            _verify_dataset_identity(FrameDataset(base, cache, split, 32), provenance['backbone_provenance'])
        labels = ActionLabels(base, cache, dataset_root)
        if labels.digest != provenance['action_labels_sha256'] or labels.provenance != provenance['action_labels']:
            raise ValueError('Action label provenance differs from training')
        validate_selection(report, selected, payload, labels.closed_eye_min_ms)
        if file_sha256(source) != digest or file_sha256(path) != selected['checkpoint_sha256']:
            raise ValueError('Diagnostic reference changed during read')
        yield dict(base=base, cache=cache, labels=labels, model=model, selected=selected, report=report,
                   provenance=dict(action_report=str(source), action_report_sha256=digest,
                                   checkpoint=str(path), checkpoint_sha256=selected['checkpoint_sha256'],
                                   model_provenance=provenance),
                   sources=[source.parent, path.parent, cache.root, base.root, base.cache.root,
                            Path(base.report['data_root']), Path(base.report['manifest']), Path(dataset_root)])
    finally:
        if cache is not None: cache.close()
        if base is not None: base.close()


def _scores(predictions, cache, labels):
    def group(ids):
        result = {}
        for column, name in enumerate(('eye', 'yawn')):
            result[name] = score_summary(
                np.concatenate([predictions[r][:, column] for r in ids]),
                np.concatenate([labels.records[r]['targets'][:, column] for r in ids]),
                np.concatenate([labels.records[r]['masks'][:, column] & cache.arrays(r)['valid'] for r in ids]))
        return result
    if not predictions: raise ValueError('Diagnostic split has no records')
    subjects = sorted({cache.records[r]['subject_id'] for r in predictions})
    return dict(overall=group(sorted(predictions)), by_subject={s: group(
        [r for r in sorted(predictions) if cache.records[r]['subject_id'] == s]) for s in subjects})


def _write_frames(path, predictions, cache, labels):
    fields = ('split', 'record_id', 'subject_id', 'slot_index', 'source_pts_ms', 'valid', 'source_reset',
              'eye_logit', 'yawn_logit', 'eye_known', 'eye_target', 'yawn_known', 'yawn_target')
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for split, records in predictions.items():
            for rid, logits in sorted(records.items()):
                arrays, label = cache.arrays(rid), labels.records[rid]
                for i, values in enumerate(logits):
                    valid = bool(arrays['valid'][i])
                    row = dict(split=split, record_id=rid, subject_id=cache.records[rid]['subject_id'],
                               slot_index=int(arrays['slot_indices'][i]), source_pts_ms=int(arrays['source_pts_ms'][i]),
                               valid=valid, source_reset=bool(label['reset'][i]))
                    for col, name in enumerate(('eye', 'yawn')):
                        known = bool(label['masks'][i, col])
                        row.update({name+'_logit': float(values[col]) if valid else None,
                                    name+'_known': known,
                                    name+'_target': int(label['targets'][i, col]) if known else None})
                    writer.writerow(row)


def run(*, action_report, dataset_root, output_dir, variant=DEFAULT_VARIANT, backend='cpu', device_index=0):
    source = Path(action_report).resolve(strict=True); output = Path(output_dir).resolve()
    if output.exists(): raise FileExistsError('Use a NEW output_dir; diagnostics never overwrite reports')
    with _inputs(source, variant, dataset_root, backend, device_index) as data:
        if any(output.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(output) for p in data['sources']):
            raise ValueError('Diagnostic output overlaps source artifacts')
        cache, labels = data['cache'], data['labels']
        resets = {rid: r['reset'] for rid, r in labels.records.items()}
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output/'input.json', dict(provenance=data['provenance'], variant=variant,
                                             backend=backend, device_index=device_index))
        print(json.dumps(dict(stage='reproduce_selected_event', variant=variant)), flush=True)
        baseline, validation = evaluate_events(cache, data['model'], data['base'].rows['val'], resets, labels.closed_eye_min_ms)
        selected = data['selected']
        actual = _summary(baseline, **{k: selected[k] for k in IDENTITY})
        if any(selected.get(k) != v for k, v in actual.items()):
            raise ValueError('Replayed predictions differ from published event report')
        actions = action_metrics(validation, labels, cache)
        if actions != data['report']['action_metrics'][variant.removesuffix('-event')]:
            raise ValueError('Replayed action metrics differ from published report')
        print(json.dumps(dict(stage='infer_training_scores', reference_reproduced=True, **actual['overall'])), flush=True)
        _, training = evaluate_events(cache, data['model'], data['base'].rows['train'], resets, labels.closed_eye_min_ms)
        predictions = dict(train=training, val=validation)
        scores = {s: _scores(p, cache, labels) for s, p in predictions.items()}
        train_eye = scores['train']['overall']['eye']
        thresholds = dict(fpr03=train_eye['fpr03_threshold'], balanced_accuracy=train_eye['balanced_accuracy_threshold'])
        variants = [('original_event', 0., 'both', 'fixed_reference'),
                    ('yawn_only', None, 'yawn', 'branch_ablation'),
                    ('eye_only', 0., 'eye', 'branch_ablation'),
                    ('train_eye_fpr03', thresholds['fpr03'], 'both', 'training_selected_exploratory'),
                    ('train_eye_balanced', thresholds['balanced_accuracy'], 'both', 'training_selected_exploratory')]
        comparisons = []; subject_rows = []; subtype_rows = []
        for name, threshold, branches, role in variants:
            rows = replay_events(cache, validation, baseline, resets, labels.closed_eye_min_ms,
                                 eye_threshold=threshold, branches=branches)
            summary = _summary(rows, variant=name)
            if summary['window_label_hash'] != actual['window_label_hash']:
                raise ValueError('Diagnostic cohort changed')
            summary.update(role=role, eye_threshold_logit=threshold, branches=branches,
                           by_subtype=subtype_metrics(rows, labels), changes_from_reference=paired_changes(rows, baseline),
                           by_subject_subtype={s: subtype_metrics([r for r in rows if r['subject_id']==s], labels)
                                               for s in summary['by_subject']})
            comparisons.append(summary)
            write_rows(output/(name+'-windows.csv'), rows)
            for subject, values in summary['by_subject'].items():
                subject_rows.append(dict(variant=name, subject_id=subject, **values))
            for subject, values in dict(all=summary['by_subtype'], **summary['by_subject_subtype']).items():
                for subtype, counts in values.items():
                    subtype_rows.append(dict(variant=name, subject_id=subject, subtype=subtype, **counts))
        _write_frames(output/'frames.csv', predictions, cache, labels)
        score_rows = []
        with (output/'score-curves.csv').open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['split', 'subject_id', 'action', 'threshold_logit',
                                                       'tp', 'fp', 'fn', 'tn', 'recall', 'false_positive_rate', 'precision'])
            writer.writeheader()
            for split, group in scores.items():
                for subject, summary in dict(all=group['overall'], **group['by_subject']).items():
                    for action, result in summary.items():
                        key = dict(split=split, subject_id=subject, action=action)
                        for point in result.pop('curve'): writer.writerow(dict(key, **point))
                        score_rows.append(dict(key, known=result['known'], positive=result['positive'],
                                               negative=result['negative'], unknown_or_invalid=result['unknown_or_invalid'],
                                               roc_auc=result['roc_auc'], average_precision=result['average_precision'],
                                               **result['fixed_zero']))
        write_rows(output/'scores.csv', score_rows)
        write_rows(output/'comparison.csv', [dict(variant=r['variant'], role=r['role'],
                   eye_threshold_logit=r['eye_threshold_logit'], **r['overall']) for r in comparisons])
        write_rows(output/'by-subject.csv', subject_rows)
        write_rows(output/'by-subtype.csv', subtype_rows)
        result = dict(schema_version='driveguard-eye-diagnostics-v1', status='complete', training_runs=0,
                      reference_reproduced=True, reference=actual, action_metrics=actions,
                      provenance=data['provenance'], backend=backend, device_index=device_index,
                      scores=scores, train_selected_thresholds=thresholds, comparisons=comparisons,
                      event_policy=dict(yawn_threshold_logit=0., closed_eye_min_ms=labels.closed_eye_min_ms,
                                        reset_on_source_gap_ms=80),
                      notes=['Unique sampled frames; unknown/invalid frames excluded from binary score statistics.',
                             'Both frame ROC/AP and fixed-zero confusion are reported. Tied scores enter together.',
                             'Curves and within-split thresholds are diagnostic, not independent calibration.',
                             'Training-derived eye thresholds use training labels only; this model already fit those labels.',
                             'The checkpoint was validation-selected. This is not an independent test.',
                             'Frame FPR03 does not constrain fatigue window FPR; see comparison.csv for actual window errors.',
                             'None eye threshold disables eye positives; undefined rankings/thresholds remain null.',
                             'Eye threshold changes recompute causal closure duration. Yawn threshold stays zero.',
                             'Branch ablation measures predictions lost without each branch; overlapping positives are possible.',
                             'No automatic winner, model training, deployment threshold change or other-task routing change.'])
        atomic_json(output/'summary.json', result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('action-report', 'dataset-root', 'output-dir'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--variant', default=DEFAULT_VARIANT)
    parser.add_argument('--backend', default='cpu', choices=('cpu', 'cuda', 'npu'))
    parser.add_argument('--device-index', type=int, default=0)
    args = vars(parser.parse_args(argv))
    try:
        result = run(**args)
        print(json.dumps(dict(status=result['status'], training_runs=0, output_dir=args['output_dir'])))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f'error: {error}', file=sys.stderr); return 2


if __name__ == '__main__': raise SystemExit(main())
