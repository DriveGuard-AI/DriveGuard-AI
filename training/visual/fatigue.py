"""Train body fatigue temporal probes using existing verified RGB features."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics

import numpy as np

from training.common.checkpoint import atomic_json
from training.common.config import canonical_hash, read_json
from training.common.data import file_sha256
from training.visual.data import PreparedData
from training.visual.prepare import prepare
from training.visual.report import evaluate, metrics, operating_point, write_rows
from training.visual.suite import DEFAULTS, _train
from training.visual.train import _settings


ARMS = (('B_rgb', 'visual', False), ('C_rgb_balanced', 'visual', True),
        ('D_fusion_balanced', 'fusion', True))


def class_balance(dataset):
    """Count only training rows that can actually contribute to the loss."""
    positive = negative = 0
    for index in range(len(dataset)):
        inputs, target, known = dataset[index]
        if not known or not bool(np.asarray(inputs['visual_mask']).all()):
            continue
        if float(target) not in (0., 1.):
            raise ValueError('Known fatigue target must be binary')
        positive += int(target == 1)
        negative += int(target == 0)
    if not positive or not negative:
        raise ValueError('Fatigue training requires RGB-available known positives AND negatives')
    return dict(positive=positive, negative=negative, positive_weight=negative/positive)


def fatigue_metrics(rows, threshold=0.):
    result = metrics(rows, threshold)
    result['fn_total'] = result['positive'] - result['tp']
    return result


def summarize(rows, **identity):
    labels = [{k: r[k] for k in ('record_id', 'start_slot', 'end_slot',
                                'label_known', 'target', 'f_eligible')} for r in rows]
    overall = fatigue_metrics(rows)
    point = operating_point(rows, .03)
    point['fn_total'] = point['positive'] - point['tp']
    return dict(identity, window_label_hash=canonical_hash(labels), overall=overall,
                baseline_f_ineligible=fatigue_metrics([r for r in rows if not r['f_eligible']]),
                baseline_f_eligible=fatigue_metrics([r for r in rows if r['f_eligible']]),
                by_subject={s: fatigue_metrics([r for r in rows if r['subject_id'] == s])
                            for s in sorted({r['subject_id'] for r in rows})},
                by_record={s: fatigue_metrics([r for r in rows if r['record_id'] == s])
                           for s in sorted({r['record_id'] for r in rows})},
                validation_fpr03=point,
                nonzero_tp_with_fpr03_at_fixed_threshold=(overall['tp'] > 0
                    and overall['false_positive_rate'] is not None
                    and overall['false_positive_rate'] <= .03))


def _prepared(output, reference, settings):
    # Failed attempts are retained. Only published, verified preparations are reused.
    attempts = sorted(output.glob('prepared-*'))
    expected = dict(task='fatigue', baseline_checkpoint=reference['baseline_checkpoint'],
                    baseline_checkpoint_sha256=reference['baseline_checkpoint_sha256'],
                    manifest=reference['manifest'], data_root=reference['data_root'],
                    visual_cache=reference['visual_cache'], visual_cache_sha256=reference['visual_cache_sha256'])
    for attempt in attempts:
        if not (attempt/'prepared.json').exists():
            continue
        data = PreparedData(attempt)
        if any(data.report.get(k) != v for k, v in expected.items()):
            data.close()
            raise ValueError('Fatigue preparation/reference identity mismatch')
        return data
    target = output/f'prepared-{len(attempts)+1:03d}'
    prepare(checkpoint=reference['baseline_checkpoint'], visual_cache=reference['visual_cache'],
            output_dir=target, manifest=reference['manifest'], data_root=reference['data_root'],
            task='fatigue', backend=settings['backend'], device_index=settings['device_index'],
            batch_size=settings['batch_size'])
    return PreparedData(target)


def run(*, reference_prepared_dir, output_dir, settings=None, seeds=(42, 43), continue_run=False):
    settings = _settings(deepcopy(DEFAULTS if settings is None else settings))
    if 'positive_weight' in settings:
        raise ValueError('Fatigue positive_weight is derived from training labels, not supplied')
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide distinct seeds')
    for seed in seeds:
        _settings(dict(settings, seed=seed))
    output = Path(output_dir).resolve()
    source = PreparedData(reference_prepared_dir)
    try:
        if source.task != 'distraction':
            raise ValueError('Reference must be the existing distraction preparation')
        reference = deepcopy(source.report)
        for path in (source.root, source.cache.root, Path(reference['data_root']).resolve(),
                     Path(reference['baseline_checkpoint']).parent, Path(reference['manifest'])):
            if output.is_relative_to(path) or path.is_relative_to(output):
                raise ValueError('Fatigue output overlaps source artifacts')
        if file_sha256(reference['baseline_checkpoint']) != reference['baseline_checkpoint_sha256']:
            raise ValueError('Reference checkpoint checksum mismatch')
        if file_sha256(reference['manifest']) != source.cache.report['manifest_sha256']:
            raise ValueError('Reference manifest checksum mismatch')
        plan = dict(schema_version='driveguard-fatigue-visual-plan-v1', task='fatigue',
                    reference=source.provenance, settings=settings, seeds=seeds, arms=[list(a) for a in ARMS])
    finally:
        source.close()
    if output.exists():
        if not continue_run:
            raise FileExistsError('Use a NEW output_dir or --continue-run')
        if read_json(output/'plan.json') != plan:
            raise ValueError('Fatigue run configuration/identity mismatch')
    else:
        output.mkdir(parents=True)
        atomic_json(output/'plan.json', plan)
    print(json.dumps(dict(stage='preparing_fatigue_labels', visual_cache=reference['visual_cache'],
                          downloads=False, video_extraction=False)), flush=True)
    prepared = _prepared(output, reference, settings)
    try:
        train, val = prepared.dataset('train'), prepared.dataset('val')
        balance = class_balance(train)
        atomic_json(output/'training-labels.json', balance)
        print(json.dumps(dict(stage='prepared', task='fatigue', train=balance,
                              validation_windows=len(val), training_runs=len(ARMS)*len(seeds))), flush=True)
        rows, _ = evaluate(val)
        comparisons = [summarize(rows, variant='A_original_fatigue', arm='A_original_fatigue',
                                seed=None, checkpoint_kind='baseline',
                                checkpoint=reference['baseline_checkpoint'],
                                checkpoint_sha256=reference['baseline_checkpoint_sha256'],
                                epoch=reference['baseline_epoch'])]
        write_rows(output/'A_original_fatigue-windows.csv', rows)
        for arm, mode, balanced in ARMS:
            for seed in seeds:
                label = f'{arm}-seed{seed}'
                current = dict(settings, seed=seed, positive_weight=balance['positive_weight'] if balanced else 1.)
                print(json.dumps(dict(stage='training', variant=label, settings=current)), flush=True)
                result = _train(output/label, train, val, mode, current, prepared.provenance)
                for kind in ('best', 'last'):
                    checkpoint = result[kind+'_checkpoint']
                    rows, payload = evaluate(val, checkpoint, settings['backend'], settings['device_index'], settings['batch_size'])
                    summary = summarize(rows, variant=label+'-'+kind, arm=arm, seed=seed,
                                        checkpoint_kind=kind, checkpoint=checkpoint,
                                        checkpoint_sha256=file_sha256(checkpoint), epoch=payload['epoch'])
                    if summary['window_label_hash'] != comparisons[0]['window_label_hash']:
                        raise ValueError('Fatigue comparison label/cohort mismatch')
                    comparisons.append(summary)
                    write_rows(output/(label+'-'+kind+'-windows.csv'), rows)
                    atomic_json(output/(label+'-'+kind+'-summary.json'), summary)
                    print(json.dumps(dict(stage='evaluated', variant=summary['variant'], **summary['overall'])), flush=True)
        aggregate = {}
        for arm, _, _ in ARMS:
            selected = [r for r in comparisons if r['arm'] == arm and r['checkpoint_kind'] == 'best']
            aggregate[arm] = {}
            for metric in ('tp', 'fp', 'fn_total', 'all_positive_detection_rate', 'all_positive_f1', 'false_positive_rate'):
                values = [r['overall'][metric] for r in selected if r['overall'][metric] is not None]
                aggregate[arm][metric] = (dict(count=len(values), mean=statistics.mean(values),
                    std=statistics.pstdev(values), min=min(values), max=max(values)) if values else dict(count=0))
        report = dict(schema_version='driveguard-fatigue-visual-comparison-v1', status='complete', task='fatigue',
                      split='val', training_balance=balance, aggregate_best=aggregate, comparisons=comparisons,
                      prepared_dir=str(prepared.root), prepared_sha256=prepared.provenance['prepared_sha256'],
                      notes=[
                          'All original body validation windows retained; labels are fatigue, not distraction.',
                          'Availability requires eight valid RGB samples, independent of F/head/eye eligibility.',
                          'fn_total counts all missed positives, including unavailable predictions; fn counts available misses.',
                          'Primary metrics use fixed logit 0 (probability 0.5). FPR03 sweeps are validation-only diagnostics.',
                          'Weights use only known available training labels. Best epoch uses unweighted validation BCE.',
                          'Frozen existing RGB encoder, eight samples over four seconds; this is not a high-rate blink detector.',
                          'Other task models and outputs are unchanged; this report evaluates fatigue only.',
                          'No test metrics or test thresholds used. Synthetic local checks do not establish DMD improvement.'])
        flat = [dict(variant=r['variant'], seed=r['seed'], checkpoint=r['checkpoint'], epoch=r['epoch'],
                     **r['overall'], f_missing_positive=r['baseline_f_ineligible']['positive'],
                     f_missing_tp=r['baseline_f_ineligible']['tp'], f_missing_fp=r['baseline_f_ineligible']['fp'],
                     fixed_threshold_pass=r['nonzero_tp_with_fpr03_at_fixed_threshold'],
                     diagnostic_fpr03_tp=r['validation_fpr03']['tp'], diagnostic_fpr03_fp=r['validation_fpr03']['fp'])
                for r in comparisons]
        write_rows(output/'comparison.csv', flat)
        atomic_json(output/'comparison.json', report)
        return report
    finally:
        prepared.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-prepared-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--continue-run', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43])
    parser.add_argument('--patience', type=int, default=4)
    for key, value in DEFAULTS.items():
        parser.add_argument('--'+key.replace('_','-'), type=type(value), default=value)
    args = vars(parser.parse_args(argv))
    settings = {key: args.pop(key) for key in DEFAULTS}
    settings.update(early_stopping_patience=args.pop('patience'))
    try:
        report = run(settings=settings, **args)
        print(json.dumps(dict(status=report['status'], output_dir=args['output_dir'])), flush=True)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
