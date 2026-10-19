"""Read-only validation of frozen gap80 with body-only C_fusion distraction."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch

from training.common.checkpoint import atomic_json
from training.common.config import TASKS, canonical_hash, read_json
from training.common.data import file_sha256
from training.common.model_factory import load_model
from training.gaze.data import FeatureCorpus
from training.gaze.metrics import GazeMetrics
from training.visual.data import METADATA, PreparedData
from training.visual.report import evaluate, metrics, summarize, write_rows
from training.visual.evaluation_cost import measure_pass, model_footprint, modelstate_footprint
from training.visual.model import VisualProbe

LOGIT_ATOL = 1e-4


def _key(row):
    return row['record_id'], row['start_slot'], row['end_slot']


def _indexed(rows):
    result = {_key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate window identity')
    return result


def _finite(values):
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite prediction')


def _match(full, body):
    if any(full[k] != body[k] for k in (*METADATA, 'h_eligible')):
        raise ValueError('Body window metadata mismatch')
    if full['targets'][3] != body['target'] or full['label_known'][3] != body['label_known']:
        raise ValueError('Body window label mismatch')


def compose(baseline, probe_rows, threshold):
    """Copy predictions, replacing only available body distraction scores/masks."""
    if threshold is not None:
        _finite(threshold)
    _indexed(baseline)
    body = _indexed([r for r in baseline if r['view'] == 'body'])
    probes = _indexed(probe_rows)
    if set(body) != set(probes):
        raise ValueError('Body window identity set mismatch')
    for key, row in probes.items():
        _match(body[key], row)
        if row['logit'] is not None:
            _finite(row['logit'])
        if row['available'] and row['logit'] is None:
            raise ValueError('Available probe missing logit')
    result = deepcopy(baseline)
    fallback = replaced = 0
    for row in result:
        _finite(row['behavior_logits']); _finite(row['gaze_logits'])
        if row['view'] != 'body':
            continue
        probe = probes[_key(row)]
        row['body_probe_available'] = probe['available']
        row['body_probe_logit'] = probe['logit']
        row['body_probe_threshold'] = threshold
        if probe['available']:
            row['behavior_logits'][3] = probe['logit']
            row['task_available'][3] = True
            row['thresholds'][3] = threshold
            row['body_distraction_source'] = 'C_fusion'
            replaced += 1
        else:
            row['body_distraction_source'] = 'baseline_rgb_unavailable'
            fallback += 1
    score_equal = mask_equal = gaze_equal = True
    for a, b in zip(baseline, result):
        columns = range(3) if a['view'] == 'body' else range(4)
        score_equal &= all(a['behavior_logits'][j] == b['behavior_logits'][j] and
                           a['thresholds'][j] == b['thresholds'][j] for j in columns)
        mask_equal &= all(a['task_available'][j] == b['task_available'][j] for j in columns)
        gaze_equal &= all(a[k] == b[k] for k in ('gaze_logits', 'gaze_available',
                                               'gaze_target', 'gaze_label_known'))
    audit = dict(unchanged_behavior_logits_and_thresholds_equal=score_equal,
                 unchanged_behavior_masks_equal=mask_equal, gaze_equal=gaze_equal,
                 replaced_body_windows=replaced, fallback_body_windows=fallback,
                 passed=score_equal and mask_equal and gaze_equal)
    if not audit['passed']:
        raise ValueError('Composition equality audit failed')
    return result, audit


def _align_output_base(baseline, alternate):
    left, right = _indexed(baseline), _indexed(alternate)
    if set(left) != set(right):
        raise ValueError('Output-base window identity set mismatch')
    for key, row in left.items():
        other = right[key]
        if any(row[field] != other[field] for field in METADATA):
            raise ValueError('Output-base window metadata mismatch')
        if any(row[field] != other[field] for field in
               ('targets', 'label_known', 'gaze_target', 'gaze_label_known')):
            raise ValueError('Output-base window label mismatch')
        for prediction in (row, other):
            _finite(prediction['behavior_logits'])
            _finite(prediction['gaze_logits'])
            for threshold in prediction['thresholds']:
                if threshold is not None:
                    _finite(threshold)
    return right


def replace_output_base(gap80_composite, coverage_rows):
    """Replace non-C outputs by identity; retain gap80 fallback and H cohorts."""
    coverage = _align_output_base(gap80_composite, coverage_rows)
    result = deepcopy(gap80_composite)
    behavior_fields = ('behavior_logits', 'task_available', 'thresholds')
    gaze_fields = ('gaze_logits', 'gaze_available', 'gaze_target', 'gaze_label_known')
    branch_fields = ('body_probe_available', 'body_probe_logit', 'body_probe_threshold',
                     'body_distraction_source')
    for row in result:
        alternate = coverage[_key(row)]
        columns = range(3) if row['view'] == 'body' else range(4)
        for field in behavior_fields:
            for column in columns:
                row[field][column] = deepcopy(alternate[field][column])
        for field in gaze_fields:
            row[field] = deepcopy(alternate[field])
        row['output_base_h_eligible'] = alternate['h_eligible']
    branch_equal = behavior_equal = gaze_equal = cohorts_equal = True
    for original, row in zip(gap80_composite, result):
        alternate = coverage[_key(row)]
        columns = range(3) if row['view'] == 'body' else range(4)
        behavior_equal &= all(row[field][column] == alternate[field][column]
                              for field in behavior_fields for column in columns)
        gaze_equal &= all(row[field] == alternate[field] for field in gaze_fields)
        cohorts_equal &= row['h_eligible'] == original['h_eligible']
        if row['view'] == 'body':
            branch_equal &= all(row[field][3] == original[field][3] for field in behavior_fields)
            branch_equal &= all(row.get(field) == original.get(field) for field in branch_fields)
    audit = dict(c_branch_and_gap80_fallback_equal=branch_equal,
                 output_base_behavior_logits_thresholds_masks_equal=behavior_equal,
                 output_base_gaze_equal=gaze_equal, frozen_gap80_h_cohorts_equal=cohorts_equal,
                 passed=branch_equal and behavior_equal and gaze_equal and cohorts_equal)
    if not audit['passed']:
        raise ValueError('Output-base composition equality audit failed')
    return result, audit


def _decision(row, column=3):
    threshold = row['thresholds'][column]
    return (row['task_available'][column] and threshold is not None and
            row['behavior_logits'][column] >= threshold)


def _summary(rows):
    behavior = {}
    for column, task in enumerate(TASKS):
        # Decisions are evaluated with each row's original threshold, including
        # baseline zero for RGB fallbacks. Raw logits are never shifted.
        decisions = [dict(label_known=r['label_known'][column], target=r['targets'][column],
                          available=r['task_available'][column],
                          logit=1. if _decision(r, column) else -1.) for r in rows]
        value = metrics(decisions)
        value['threshold_logit'] = 'per_window_see_predictions'
        value['effective_f1'] = (2*value['tp']/(2*value['tp']+value['fp']+value['fn'])
                                 if 2*value['tp']+value['fp']+value['fn'] else 0.)
        if not value['available']:
            value['effective_f1'] = value['precision'] = None
        if not value['known']:
            value['all_positive_f1'] = None
        value['raw_known'] = value['known']
        value['input_available_windows'] = sum(r['task_available'][column] for r in rows)
        behavior[task] = value
    gaze = GazeMetrics()
    if rows:
        gaze.update(torch.tensor([r['gaze_logits'] for r in rows]),
                    torch.tensor([r['gaze_target'] for r in rows]),
                    torch.tensor([r['gaze_label_known'] and r['gaze_available'] for r in rows]))
    result = gaze.report()
    raw_known = sum(r['gaze_label_known'] for r in rows)
    result.update(raw_known=raw_known, raw_unknown=len(rows)-raw_known,
                  available_known=result['known'], unavailable_known=raw_known-result['known'],
                  input_available_windows=sum(r['gaze_available'] for r in rows),
                  coverage=result['known']/raw_known if raw_known else None,
                  raw_class_support=[sum(r['gaze_label_known'] and r['gaze_target'] == c for r in rows)
                                     for c in range(9)])
    return dict(windows=len(rows), behavior=behavior, gaze=result)


def summarize_all(rows):
    result = dict(overall=_summary(rows))
    for field in ('view', 'subject_id', 'subset'):
        name = 'subject' if field == 'subject_id' else field
        result['by_'+name] = {value: _summary([r for r in rows if r[field] == value])
                             for value in sorted({r[field] for r in rows})}
    return result


def paired_errors(baseline, composite):
    """All labeled body rows, and the strict both-available comparison cohort."""
    a, b = _indexed(baseline), _indexed(composite)
    if set(a) != set(b):
        raise ValueError('Paired window identity mismatch')
    pairs = [(row, b[key]) for key, row in a.items()
             if row['view'] == 'body' and row['label_known'][3]]
    def count(values):
        result = dict(windows=len(values), rescued_positive=0, lost_positive=0,
                      introduced_fp=0, removed_fp=0)
        for left, right in values:
            was, now = _decision(left), _decision(right)
            if was == now:
                continue
            name = ('rescued_positive' if now else 'lost_positive') if left['targets'][3] else (
                'introduced_fp' if now else 'removed_fp')
            result[name] += 1
        return result
    result = {}
    for name, values in (('overall', pairs), ('baseline_h_eligible', [p for p in pairs if p[0]['h_eligible']]),
                         ('baseline_h_ineligible', [p for p in pairs if not p[0]['h_eligible']])):
        result[name] = count(values)
        result[name]['strict_both_available'] = count([p for p in values
            if p[0]['task_available'][3] and p[1]['task_available'][3]])
    return result


@torch.no_grad()
def _baseline_rows(model, dataset, batch_size):
    device = next(model.parameters()).device
    rows = []
    for start in range(0, len(dataset), batch_size):
        items = [dataset[i] for i in range(start, min(start+batch_size, len(dataset)))]
        tensors = {k: torch.as_tensor(np.stack([item[0][k] for item in items]), device=device)
                   for k in items[0][0]}
        output = model(**tensors)
        for key in ('behavior_logits', 'gaze_logits'):
            if not torch.isfinite(output[key]).all():
                raise ValueError('Nonfinite baseline prediction')
        values = {k: v.detach().cpu().tolist() for k, v in output.items()
                  if k in ('behavior_logits', 'gaze_logits', 'task_available', 'gaze_available', 'head_available')}
        for offset, (_, target, known, gaze, gaze_known) in enumerate(items):
            record, slot = dataset.locate(start+offset)
            rows.append(dict(**{k: record[k] for k in METADATA}, start_slot=slot,
                end_slot=slot+dataset.steps-1, targets=target.astype(int).tolist(),
                label_known=known.tolist(), behavior_logits=values['behavior_logits'][offset],
                task_available=values['task_available'][offset], thresholds=[0.]*4,
                gaze_logits=values['gaze_logits'][offset], gaze_available=values['gaze_available'][offset],
                gaze_target=int(gaze), gaze_label_known=bool(gaze_known),
                h_eligible=values['head_available'][offset]))
    return rows


def _verify_baseline(rows, prepared_rows):
    full = _indexed([r for r in rows if r['view'] == 'body'])
    cached = _indexed(prepared_rows)
    if set(full) != set(cached):
        raise ValueError('Prepared/full body window identity set mismatch')
    maximum = 0.
    for key, old in cached.items():
        row = full[key]
        _match(row, old)
        if row['task_available'][3] != old['baseline_available']:
            raise ValueError('Prepared baseline availability mismatch')
        if old['baseline_available']:
            if old['baseline_logit'] is None:
                raise ValueError('Prepared baseline missing logit')
            _finite(old['baseline_logit'])
            maximum = max(maximum, abs(row['behavior_logits'][3]-old['baseline_logit']))
    if maximum > LOGIT_ATOL:
        raise ValueError('Prepared baseline logit drift exceeds tolerance')
    return dict(max_absolute_logit_difference=maximum, absolute_tolerance=LOGIT_ATOL,
                body_windows=len(full), metadata_labels_masks_equal=True)


def _equivalent(actual, expected):
    if isinstance(actual, dict):
        return (isinstance(expected, dict) and actual.keys() == expected.keys() and
                all(_equivalent(value, expected[key]) for key, value in actual.items()))
    if isinstance(actual, float) and isinstance(expected, (float, int)):
        return math.isfinite(actual) and math.isfinite(expected) and abs(actual-expected) <= LOGIT_ATOL
    return actual == expected


def _source_entries(source):
    if (source.get('schema_version') != 'driveguard-body-visual-stability-v1' or
            source.get('status') != 'complete' or source.get('split') != 'val'):
        raise ValueError('A complete validation stability report is required')
    selected = []
    for seed in (42, 43):
        matches = [r for r in source['comparisons'] if r.get('arm') == 'C_fusion' and
                   r.get('seed') == seed and r.get('checkpoint_kind') == 'best']
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one C_fusion seed{seed} best checkpoint')
        selected.append(matches[0])
    return selected


def run(*, prepared_dir, stability_report, output_dir, backend='cpu', device_index=0,
        batch_size=64, split='val', output_base_checkpoint=None):
    """Evaluate three preselected candidates without training or threshold tuning."""
    if split != 'val':
        raise ValueError('Only val is supported; test threshold selection is prohibited')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    source_path = Path(stability_report).resolve(strict=True)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a NEW output_dir')
    source = read_json(source_path)
    entries = _source_entries(source)
    prepared = PreparedData(prepared_dir)
    corpus = output_corpus = None
    try:
        if prepared.task != 'distraction':
            raise ValueError('Body distraction composition requires distraction prepared data')
        reference = prepared.report
        baseline_path = Path(reference['baseline_checkpoint']).resolve(strict=True)
        output_base_path = (Path(output_base_checkpoint).resolve(strict=True)
                            if output_base_checkpoint is not None else None)
        checkpoints = [Path(e['checkpoint']).resolve(strict=True) for e in entries]
        sources = [prepared.root, prepared.cache.root, Path(reference['data_root']).resolve(strict=True),
                   baseline_path.parent, *(p.parent for p in checkpoints),
                   *([output_base_path.parent] if output_base_path else [])]
        if (any(output.is_relative_to(p) or p.is_relative_to(output) for p in sources) or
                source_path.is_relative_to(output) or Path(reference['manifest']).resolve().is_relative_to(output)):
            raise ValueError('Output overlaps input artifacts')
        watched = [baseline_path, *checkpoints, source_path, prepared.root/'prepared.json',
                   Path(reference['manifest']), prepared.cache.root/'manifest.json',
                   *([output_base_path] if output_base_path else [])]
        watched += [p.with_suffix(p.suffix+'.sha256.json')
                    for p in [baseline_path, *checkpoints, *([output_base_path] if output_base_path else [])]]
        watched += [prepared.root/entry['rows'] for entry in reference['splits'].values()]
        watched += [prepared.cache.root/'input.json',
                    *(prepared.cache.root/entry['directory']/'record.json'
                      for entry in prepared.cache.records.values())]
        hashes = {p: file_sha256(p) for p in watched}
        if hashes[baseline_path] != reference['baseline_checkpoint_sha256']:
            raise ValueError('Baseline checkpoint hash mismatch')
        for entry, checkpoint in zip(entries, checkpoints):
            if hashes[checkpoint] != entry['checkpoint_sha256']:
                raise ValueError('Candidate checkpoint hash mismatch')
        if file_sha256(reference['manifest']) != prepared.cache.report['manifest_sha256']:
            raise ValueError('Manifest hash mismatch')
        model, payload = load_model(baseline_path, backend, device_index)
        model.eval().requires_grad_(False)
        config = payload['config']
        if (config['model']['kind'] != 'relation_gaze_head_policy_v1' or
                config['model']['head_policy'] != dict(min_valid_ratio=.25, max_gap_steps=2)):
            raise ValueError('Expected frozen coverage25+gap80 baseline')
        corpus = FeatureCorpus(config, reference['manifest'], reference['data_root'])
        if corpus.summary != payload['data_summary'] or corpus.summary['fingerprint'] != reference['data_fingerprint']:
            raise ValueError('Corpus/baseline data fingerprint mismatch')
        print('Evaluating frozen gap80 baseline on all validation views', flush=True)
        inventory = {'gap80': dict(checkpoint=str(baseline_path), checkpoint_sha256=hashes[baseline_path],
                                   **model_footprint(model))}
        baseline, baseline_cost = measure_pass(
            lambda: _baseline_rows(model, corpus.dataset(split, payload['normalizer']), batch_size),
            backend, device_index)
        stages = [dict(name='gap80_cached_feature_inference', includes_model_load=False, **baseline_cost)]
        alignment = _verify_baseline(baseline, prepared.rows[split])
        del model
        composites = []
        predictions = [('A_gap80', baseline)]
        output_base = None
        coverage = None
        if output_base_path is not None:
            model, output_payload = load_model(output_base_path, backend, device_index)
            model.eval().requires_grad_(False)
            output_config = output_payload['config']
            if (output_config['model']['kind'] != 'relation_gaze_head_policy_v1' or
                    output_config['model']['head_policy'] != dict(min_valid_ratio=.25, max_gap_steps=0)):
                raise ValueError('Expected frozen coverage25 output base')
            output_corpus = FeatureCorpus(output_config, reference['manifest'], reference['data_root'])
            if (output_corpus.summary != output_payload['data_summary'] or
                    output_corpus.summary['fingerprint'] != corpus.summary['fingerprint']):
                raise ValueError('Output-base corpus/checkpoint data fingerprint mismatch')
            inventory['coverage25'] = dict(checkpoint=str(output_base_path),
                checkpoint_sha256=hashes[output_base_path], **model_footprint(model))
            print('Evaluating frozen coverage25 output base with its own normalizer', flush=True)
            coverage, cost = measure_pass(lambda: _baseline_rows(
                model, output_corpus.dataset(split, output_payload['normalizer']), batch_size), backend, device_index)
            stages.append(dict(name='coverage25_cached_feature_inference', includes_model_load=False, **cost))
            _align_output_base(baseline, coverage)
            output_base = dict(name='A_coverage25', checkpoint=str(output_base_path),
                checkpoint_sha256=hashes[output_base_path], checkpoint_epoch=output_payload['epoch'],
                normalizer_sha256=canonical_hash(output_payload['normalizer']),
                head_policy=output_config['model']['head_policy'],
                data_fingerprint=output_corpus.summary['fingerprint'], model_dependencies=['coverage25'],
                **summarize_all(coverage))
            predictions.append((output_base['name'], coverage))
            del model
        for entry, checkpoint in zip(entries, checkpoints):
            print(f"Evaluating C_fusion seed{entry['seed']} best body predictions", flush=True)
            (probe_rows, probe_payload), cost = measure_pass(
                lambda: evaluate(prepared.dataset(split), checkpoint, backend, device_index, batch_size),
                backend, device_index)
            probe_name = f"C_fusion_seed{entry['seed']}"
            stages.append(dict(name=probe_name+'_cached_feature_inference', includes_model_load=True, **cost))
            # A metadata-only model distinguishes parameters from buffers without
            # loading another checkpoint, running inference, or allocating weights.
            with torch.device('meta'):
                probe_shape = VisualProbe(**probe_payload['model_params'])
            inventory[probe_name] = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashes[checkpoint],
                state_payload=modelstate_footprint(probe_payload['model_state']), **model_footprint(probe_shape))
            if (probe_payload['model_params']['mode'] != 'fusion' or
                    probe_payload['settings']['seed'] != entry['seed'] or probe_payload['epoch'] != entry['epoch']):
                raise ValueError('Candidate checkpoint identity mismatch')
            fresh = summarize(probe_rows)
            if fresh['window_label_hash'] != entry['window_label_hash']:
                raise ValueError('Source report window-label hash mismatch')
            for field in ('overall', 'validation_fpr03'):
                if not _equivalent(fresh[field], entry[field]):
                    raise ValueError('Source report metrics/threshold mismatch')
            candidates = [('default', 0.)]
            if entry['seed'] == 43:
                point = entry['validation_fpr03']
                # Check the actual imported threshold, not merely a newly swept one.
                if not _equivalent(metrics(probe_rows, point['threshold_logit']),
                                   {k: v for k, v in point.items() if k not in ('fpr_budget', 'threshold_selection')}):
                    raise ValueError('Source operating-point metrics mismatch')
                candidates.append(('validation_fpr03', point['threshold_logit']))
            for kind, threshold in candidates:
                name = f"C_fusion_seed{entry['seed']}_{kind}"
                rows, audit = compose(baseline, probe_rows, threshold)
                composites.append(dict(name=name, seed=entry['seed'], checkpoint=str(checkpoint),
                    output_base='A_gap80', model_dependencies=['gap80', probe_name],
                    checkpoint_sha256=hashes[checkpoint], checkpoint_epoch=entry['epoch'],
                    body_threshold_logit=threshold, threshold_source=kind,
                    window_label_hash=entry['window_label_hash'], equality_audit=audit,
                    paired_body_errors=paired_errors(baseline, rows), **summarize_all(rows)))
                predictions.append((name, rows))
                if coverage is not None:
                    alternate_rows, alternate_audit = replace_output_base(rows, coverage)
                    alternate_name = name+'_coverage25_outputs'
                    item = deepcopy(composites[-1])
                    item.update(name=alternate_name, output_base='A_coverage25',
                        model_dependencies=['gap80', 'coverage25', probe_name], equality_audit=alternate_audit,
                        paired_body_errors=paired_errors(baseline, alternate_rows), **summarize_all(alternate_rows))
                    composites.append(item)
                    predictions.append((alternate_name, alternate_rows))
        for path, digest in hashes.items():
            if file_sha256(path) != digest:
                raise ValueError('Input artifact changed during evaluation')
        report = dict(schema_version='driveguard-visual-all-tasks-v1', status='complete', split=split,
            backend=backend, device_index=device_index, batch_size=batch_size,
            provenance=dict(prepared_sha256=prepared.provenance['prepared_sha256'],
                stability_report=str(source_path), stability_report_sha256=hashes[source_path],
                baseline_checkpoint=str(baseline_path), baseline_checkpoint_sha256=hashes[baseline_path],
                data_fingerprint=corpus.summary['fingerprint'], manifest_sha256=file_sha256(reference['manifest'])),
            baseline_alignment=alignment, baseline=dict(name='A_gap80', model_dependencies=['gap80'], **summarize_all(baseline)),
            evaluation_cost=dict(stages=stages, model_inventory=inventory,
                scope='Cached-feature evaluation only; stage timings are not combined deployment latency.',
                excluded='RGB encoder, raw-video decode/extraction, and combined-device production peak memory.'),
            composites=composites, notes=[
                'Fresh validation inference only; no training, held-out evaluation or deployment claim.',
                'Body distraction uses C_fusion only when RGB is available; otherwise baseline logit, mask and zero threshold remain.',
                'Unknown labels are excluded; unavailable positives remain in all-positive denominators.',
                'Effective F1/recall use known available rows; gaze accuracy/macro F1/confusion use known available rows.',
                'Gaze and non-body-distraction behavior outputs are copied exactly from each composition output base.',
                'Coverage25 composites retain original gap80 missing-RGB fallback and frozen gap80 H cohorts.',
                'Source thresholds are validation-selected; recomputed sweeps verify source identity and never choose a replacement threshold.',
                'Baseline/probe floating-point comparisons allow absolute tolerance 0.0001; counts and labels must match.',
                'Paired all-labeled errors treat unavailable predictions as no detection; strict counts require both outputs available.'])
        if output_base is not None:
            report['output_base'] = output_base
        output.mkdir(parents=True, exist_ok=False)
        with (output/'predictions.jsonl').open('w', encoding='utf-8') as handle:
            for name, rows in predictions:
                for row in rows:
                    handle.write(json.dumps(dict(candidate=name, **row), allow_nan=False)+'\n')
        flat = []
        for item in (report['baseline'], *([output_base] if output_base else []), *composites):
            for group, values in [('overall', {'all': item['overall']}),
                                  *((key, item[key]) for key in ('by_view', 'by_subject', 'by_subset'))]:
                for value, summary in values.items():
                    for task, metric in summary['behavior'].items():
                        flat.append(dict(candidate=item['name'], group=group, value=value, task=task, **metric))
                    gaze = summary['gaze']
                    flat.append(dict(candidate=item['name'], group=group, value=value, task='gaze',
                        windows=summary['windows'], **{key: gaze[key] for key in (
                            'accuracy', 'macro_f1', 'known', 'raw_known', 'raw_unknown', 'available_known',
                            'unavailable_known', 'input_available_windows', 'coverage')}))
        columns = dict.fromkeys(key for row in flat for key in row)
        flat = [dict.fromkeys(columns) | row for row in flat]
        write_rows(output/'metrics.csv', flat)
        atomic_json(output/'report.json', report)
        return report
    finally:
        prepared.close()
        if corpus is not None:
            corpus.close()
        if output_corpus is not None:
            output_corpus.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared-dir', 'stability-report', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda', 'npu'), default='cpu')
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--split', choices=('val',), default='val')
    parser.add_argument('--output-base-checkpoint')
    try:
        args = parser.parse_args(argv)
        result = run(**vars(args))
        print(json.dumps(dict(status=result['status'], output_dir=args.output_dir)))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
