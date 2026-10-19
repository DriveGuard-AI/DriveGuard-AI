"""Matched whole/head RGB fatigue probes, with reproduced published references."""
from copy import deepcopy
import json
from pathlib import Path
import statistics

from training.common.checkpoint import atomic_json, load_checkpoint
from training.common.config import canonical_hash, read_json
from training.common.data import file_sha256
from training.visual.data import PreparedData
from training.visual.fatigue import summarize
from training.visual.report import evaluate, write_rows
from training.visual.spatial import SpatialData
from training.visual.suite import DEFAULTS, _train
from training.visual.train import _settings


ARMS = (('A_whole', 'whole'), ('B_head', 'head'), ('C_whole_head', 'both'))
METRICS = ('positive', 'negative', 'tp', 'fp', 'fn_total',
           'all_positive_detection_rate', 'all_positive_f1', 'false_positive_rate')


def _summary(rows, **identity):
    return dict(summarize(rows, **identity), prediction_rows_hash=canonical_hash(rows))


def _reference(published, dataset, settings, *, baseline=False):
    """Load/evaluate old RGB weights; baseline logits are the verified preparation."""
    checkpoint = published['checkpoint']
    if file_sha256(checkpoint) != published['checkpoint_sha256']:
        raise ValueError('Reference checkpoint checksum mismatch')
    # Reproduce the published evaluation geometry; a new training batch size can
    # change floating-point diagnostic thresholds even when classifications agree.
    replay_batch = settings['batch_size'] if baseline else load_checkpoint(checkpoint)['settings']['batch_size']
    rows, payload = evaluate(dataset, None if baseline else checkpoint,
                             settings['backend'], settings['device_index'], replay_batch)
    if baseline:
        preparation = dataset.prepared.report
        if (Path(checkpoint).resolve() != Path(preparation['baseline_checkpoint']).resolve()
                or published['checkpoint_sha256'] != preparation['baseline_checkpoint_sha256']
                or published['epoch'] != preparation['baseline_epoch']
                or published['seed'] is not None or published['checkpoint_kind'] != 'baseline'):
            raise ValueError('Original fatigue reference identity mismatch')
    elif (type(published['seed']) is not int or payload['settings']['seed'] != published['seed']
          or payload['settings'].get('positive_weight', 1.) != 1.
          or payload['model_params']['mode'] != 'visual'
          or payload['model_params']['visual_dim'] != 512
          or published['epoch'] != payload['epoch'] or payload['kind'] != 'weights'
          or published['checkpoint_kind'] != 'best'):
        raise ValueError('RGB reference seed/settings/checkpoint mismatch')
    identity = {key: published[key] for key in
                ('variant', 'arm', 'seed', 'checkpoint_kind', 'checkpoint', 'checkpoint_sha256', 'epoch')}
    result = _summary(rows, **identity)
    for key, value in result.items():
        if key == 'prediction_rows_hash' and key not in published:
            continue
        if published.get(key) != value:
            raise ValueError(f'Published reference metrics/cohort mismatch: {key}')
    return result, rows


def _stats(rows):
    result = {}
    for metric in METRICS:
        values = [r[metric] for r in rows if r[metric] is not None]
        result[metric] = (dict(count=len(values), mean=statistics.mean(values),
                               std=statistics.pstdev(values), min=min(values), max=max(values))
                          if values else dict(count=0))
    return result


def _aggregate(comparisons):
    result = {}
    for arm, _ in ARMS:
        rows = [r for r in comparisons if r['arm'] == arm and r['checkpoint_kind'] == 'best']
        result[arm] = dict(overall=_stats([r['overall'] for r in rows]),
                           by_subject={s: _stats([r['by_subject'][s] for r in rows])
                                       for s in sorted(rows[0]['by_subject'])},
                           seeds=[r['seed'] for r in rows])
    return result


def _difference(after, before):
    return {key: after[key]-before[key] if after[key] is not None and before[key] is not None else None
            for key in METRICS}


def _paired(comparisons, seeds):
    rows = {(r['arm'], r['seed']): r for r in comparisons if r['checkpoint_kind'] == 'best'}
    result = {}
    for seed in seeds:
        before = rows['A_whole', seed]
        result[str(seed)] = {}
        for arm, label in (('B_head', 'B_minus_A'), ('C_whole_head', 'C_minus_A')):
            after = rows[arm, seed]
            result[str(seed)][label] = dict(
                overall=_difference(after['overall'], before['overall']),
                baseline_f_ineligible=_difference(after['baseline_f_ineligible'], before['baseline_f_ineligible']),
                by_subject={s: _difference(after['by_subject'][s], before['by_subject'][s])
                            for s in sorted(before['by_subject'])})
    return result


def run(*, fatigue_report, head_cache, output_dir, settings=None, seeds=(42, 43), continue_run=False):
    settings = _settings(deepcopy(dict(DEFAULTS, early_stopping_patience=4) if settings is None else settings))
    if settings.get('positive_weight', 1.) != 1.:
        raise ValueError('Spatial input comparison requires ordinary BCE / positive_weight 1')
    settings['positive_weight'] = 1.
    seeds = list(seeds)
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Provide distinct integer seeds')
    for seed in seeds:
        _settings(dict(settings, seed=seed))
    report_path, output = Path(fatigue_report).resolve(), Path(output_dir).resolve()
    reference = read_json(report_path)
    if (reference.get('schema_version') != 'driveguard-fatigue-visual-comparison-v1'
            or reference.get('task') != 'fatigue' or reference.get('split') != 'val'
            or reference.get('status') != 'complete'):
        raise ValueError('Reference must be a complete fatigue validation comparison')
    published = reference.get('comparisons', [])
    if not isinstance(published, list) or any(not isinstance(r, dict) for r in published):
        raise ValueError('Invalid fatigue reference comparisons')
    variants = {r.get('variant'): r for r in published}
    if len(variants) != len(published):
        raise ValueError('Duplicate reference variants')
    names = ['A_original_fatigue'] + [f'B_rgb-seed{seed}-best' for seed in seeds]
    if any(name not in variants for name in names):
        raise ValueError('Missing original fatigue or matching RGB seed reference')
    for seed, name in zip(seeds, names[1:]):
        if variants[name].get('seed') != seed or variants[name].get('arm') != 'B_rgb':
            raise ValueError('Reference seed/arm mismatch')
    if variants[names[0]].get('arm') != 'A_original_fatigue':
        raise ValueError('Original fatigue reference arm mismatch')
    base = PreparedData(reference['prepared_dir'])
    spatial = None
    try:
        if base.task != 'fatigue' or reference.get('prepared_sha256') != base.provenance['prepared_sha256']:
            raise ValueError('Reference fatigue preparation checksum/cohort mismatch')
        sources = [base.root, base.cache.root, Path(head_cache).resolve(), report_path,
                   Path(base.report['data_root']).resolve(), Path(base.report['manifest']).resolve(),
                   Path(base.report['baseline_checkpoint']).resolve().parent]
        sources.extend(Path(variants[name]['checkpoint']).resolve().parent for name in names)
        if any(output.is_relative_to(source) or source.is_relative_to(output) for source in sources):
            raise ValueError('Spatial output overlaps source artifacts')
        spatial = SpatialData(base, head_cache)
        comparisons, reference_rows = [], {}
        for index, name in enumerate(names):
            summary, rows = _reference(variants[name], base.dataset('val'), settings, baseline=index == 0)
            if comparisons and summary['window_label_hash'] != comparisons[0]['window_label_hash']:
                raise ValueError('Reference fatigue label/cohort mismatch')
            comparisons.append(summary)
            reference_rows[name] = rows
        provenance = {arm: spatial.dataset('train', kind).prepared.provenance for arm, kind in ARMS}
        plan = dict(schema_version='driveguard-fatigue-spatial-plan-v1', task='fatigue', split='val',
                    fatigue_report=str(report_path), fatigue_report_sha256=file_sha256(report_path),
                    references=comparisons, settings=settings, seeds=seeds,
                    arms=[list(a) for a in ARMS], provenance=provenance)
        if output.exists():
            if not continue_run:
                raise FileExistsError('Use a NEW output_dir or --continue-run')
            if read_json(output/'plan.json') != plan:
                raise ValueError('Spatial run configuration/identity mismatch')
        else:
            output.mkdir(parents=True)
            atomic_json(output/'plan.json', plan)
        for row in comparisons:
            write_rows(output/(row['variant']+'-windows.csv'), reference_rows[row['variant']])
            atomic_json(output/(row['variant']+'-summary.json'), row)
        for arm, kind in ARMS:
            train, val = spatial.dataset('train', kind), spatial.dataset('val', kind)
            for seed in seeds:
                label = f'{arm}-seed{seed}'
                current = dict(settings, seed=seed)
                print(json.dumps(dict(stage='training_spatial', variant=label, settings=current)), flush=True)
                result = _train(output/label, train, val, 'visual', current, provenance[arm])
                for which in ('best', 'last'):
                    checkpoint = result[which+'_checkpoint']
                    rows, payload = evaluate(val, checkpoint, settings['backend'],
                                             settings['device_index'], settings['batch_size'])
                    summary = _summary(rows, variant=label+'-'+which, arm=arm, seed=seed,
                                       checkpoint_kind=which, checkpoint=checkpoint,
                                       checkpoint_sha256=file_sha256(checkpoint), epoch=payload['epoch'])
                    if summary['window_label_hash'] != comparisons[0]['window_label_hash']:
                        raise ValueError('Spatial fatigue label/cohort mismatch')
                    comparisons.append(summary)
                    write_rows(output/(summary['variant']+'-windows.csv'), rows)
                    atomic_json(output/(summary['variant']+'-summary.json'), summary)
                    print(json.dumps(dict(stage='evaluated_spatial', variant=summary['variant'],
                                          **summary['overall'])), flush=True)
        report = dict(schema_version='driveguard-fatigue-spatial-comparison-v1', status='complete',
                      task='fatigue', split='val', training_runs=len(ARMS)*len(seeds), seeds=seeds,
                      settings=settings, comparisons=comparisons, aggregate_best=_aggregate(comparisons),
                      paired_best_deltas=_paired(comparisons, seeds),
                      prepared_dir=str(base.root), prepared_sha256=base.provenance['prepared_sha256'],
                      fatigue_report=str(report_path), fatigue_report_sha256=plan['fatigue_report_sha256'],
                      head_cache=str(spatial.head.root), head_cache_sha256=spatial.head.manifest_sha256,
                      notes=[
                          'Whole, head, and combined inputs use the same eight frozen RGB samples over four seconds.',
                          'All arms use 1024 input dimensions and equal parameter counts; an unused 512-D half is zero.',
                          'Source identities, encoder, sampling PTS and validity masks match exactly across spatial inputs.',
                          'All original fatigue validation windows remain; shared RGB availability has no F/head/eye gate.',
                          'Normalization uses training records only. Ordinary BCE and training settings match across arms.',
                          'Best epoch uses unweighted validation BCE; both best and last checkpoints are reported.',
                          'Original fatigue preparation and published ordinary RGB best checkpoints are verified references.',
                          'Fixed logit 0 is primary; FPR03 sweeps are validation-only diagnostics.',
                          'fn_total includes unavailable positive predictions; all denominators come from the retained cohort.',
                          'No test split was used. This report does not select or deploy a winner from validation results.'])
        if (output/'comparison.json').exists() and read_json(output/'comparison.json') != report:
            raise ValueError('Completed spatial report identity/results mismatch')
        flat = [dict(variant=r['variant'], arm=r['arm'], seed=r['seed'], checkpoint_kind=r['checkpoint_kind'],
                     checkpoint=r['checkpoint'], checkpoint_sha256=r['checkpoint_sha256'], epoch=r['epoch'],
                     window_label_hash=r['window_label_hash'], prediction_rows_hash=r['prediction_rows_hash'],
                     **r['overall'], f_missing_positive=r['baseline_f_ineligible']['positive'],
                     f_missing_tp=r['baseline_f_ineligible']['tp'], f_missing_fp=r['baseline_f_ineligible']['fp'],
                     diagnostic_fpr03_tp=r['validation_fpr03']['tp'],
                     diagnostic_fpr03_fp=r['validation_fpr03']['fp']) for r in comparisons]
        write_rows(output/'comparison.csv', flat)
        atomic_json(output/'comparison.json', report)
        return report
    finally:
        if spatial is not None:
            spatial.close()
        base.close()
