"""Compare head-only fatigue learning rate and regularization using verified caches."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from training.common.checkpoint import atomic_json, load_checkpoint
from training.common.config import read_json
from training.common.data import file_sha256
from training.visual.data import PreparedData
from training.visual.fatigue_spatial_suite import _difference, _stats, _summary
from training.visual.report import evaluate, write_rows
from training.visual.spatial import SpatialData
from training.visual.suite import _train
from training.visual.train import _settings


ARMS = ('A_control', 'B_slow', 'C_regularized', 'D_slow_regularized')


def arm_settings(settings):
    """A small factorial experiment; all unlisted training settings stay fixed."""
    control = _settings(settings)
    if control.get('positive_weight', 1.) != 1.:
        raise ValueError('Head stability comparison requires ordinary BCE')
    if control['dropout'] >= .4 or control['weight_decay'] >= .01:
        raise ValueError('Reference already exceeds the regularization intervention')
    return {
        'A_control': control,
        'B_slow': dict(control, learning_rate=control['learning_rate']/10),
        'C_regularized': dict(control, dropout=.4, weight_decay=.01),
        'D_slow_regularized': dict(control, learning_rate=control['learning_rate']/10,
                                  dropout=.4, weight_decay=.01),
    }


def _delta(after, before):
    return dict(overall=_difference(after['overall'], before['overall']),
                by_subject={s: _difference(after['by_subject'][s], before['by_subject'][s])
                            for s in before['by_subject']})


def _replay(published, val, expected, backend, device_index):
    path = published['checkpoint']
    if file_sha256(path) != published['checkpoint_sha256']:
        raise ValueError('Reference checkpoint checksum mismatch')
    payload = load_checkpoint(path)
    if (payload['settings'] != expected or payload['epoch'] != published['epoch']
            or payload['kind'] != 'weights' or payload['model_params']['mode'] != 'visual'
            or payload['model_params']['visual_dim'] != 1024):
        raise ValueError('Reference checkpoint settings/identity mismatch')
    rows, _ = evaluate(val, path, backend, device_index, payload['settings']['batch_size'])
    identity = {k: published[k] for k in ('variant','arm','seed','checkpoint_kind',
                                        'checkpoint','checkpoint_sha256','epoch')}
    replayed = _summary(rows, **identity)
    if replayed != published:
        raise ValueError('Reference metrics/cohort/predictions mismatch')
    return replayed, rows


def run(*, spatial_report, output_dir, seeds=(42,43), backend=None, device_index=None,
        epochs=None, continue_run=False):
    path, output = Path(spatial_report).resolve(), Path(output_dir).resolve()
    reference = read_json(path)
    if (reference.get('schema_version') != 'driveguard-fatigue-spatial-comparison-v1'
            or reference.get('status') != 'complete' or reference.get('task') != 'fatigue'
            or reference.get('split') != 'val'):
        raise ValueError('Expected a complete spatial fatigue validation report')
    seeds = list(seeds)
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Provide distinct integer seeds')
    original = _settings(reference['settings'])
    settings = deepcopy(original)
    for key, value in (('backend',backend),('device_index',device_index),('epochs',epochs)):
        if value is not None: settings[key] = value
    arms = arm_settings(settings)
    for seed in seeds: _settings(dict(settings, seed=seed))
    published = reference['comparisons']
    variants = {r['variant']:r for r in published}
    if len(variants) != len(published): raise ValueError('Duplicate reference variants')
    names = [f'B_head-seed{s}-best' for s in seeds]
    for seed, name in zip(seeds, names):
        row = variants.get(name, {})
        if row.get('arm') != 'B_head' or row.get('seed') != seed or row.get('checkpoint_kind') != 'best':
            raise ValueError('Missing matching head best reference')
    base = PreparedData(reference['prepared_dir'])
    spatial = None
    try:
        if base.task != 'fatigue' or base.provenance['prepared_sha256'] != reference['prepared_sha256']:
            raise ValueError('Reference preparation checksum/task mismatch')
        sources = [path.parent, base.root, base.cache.root, Path(reference['head_cache']).resolve(),
                   Path(base.report['data_root']).resolve(), Path(base.report['manifest']).resolve(),
                   Path(base.report['baseline_checkpoint']).resolve().parent]
        sources.extend(Path(variants[n]['checkpoint']).resolve().parent for n in names)
        if any(output.is_relative_to(p) or p.is_relative_to(output) for p in sources):
            raise ValueError('Stability output overlaps source artifacts')
        spatial = SpatialData(base, reference['head_cache'])
        if spatial.head.manifest_sha256 != reference['head_cache_sha256']:
            raise ValueError('Reference head cache checksum mismatch')
        train, val = spatial.dataset('train','head'), spatial.dataset('val','head')
        provenance = train.prepared.provenance
        references, reference_rows = [], {}
        for seed, name in zip(seeds, names):
            summary, rows = _replay(variants[name], val, dict(original, seed=seed),
                                    settings['backend'], settings['device_index'])
            references.append(summary)
            reference_rows[name] = rows
        if len({r['window_label_hash'] for r in references}) != 1:
            raise ValueError('Reference cohort mismatch')
        plan = dict(schema_version='driveguard-fatigue-head-stability-plan-v1',
                    spatial_report=str(path), spatial_report_sha256=file_sha256(path),
                    references=references, seeds=seeds, arms=arms, provenance=provenance)
        if output.exists():
            if not continue_run: raise FileExistsError('Use a NEW output_dir or --continue-run')
            if read_json(output/'plan.json') != plan:
                raise ValueError('Stability configuration/identity mismatch')
        else:
            output.mkdir(parents=True)
            atomic_json(output/'plan.json', plan)
        comparisons = list(references)
        for row in references:
            write_rows(output/(row['variant']+'-windows.csv'), reference_rows[row['variant']])
        training_results = {}
        for arm, current in arms.items():
            for seed in seeds:
                label = f'{arm}-seed{seed}'
                current = dict(current, seed=seed)
                print(json.dumps(dict(stage='training_head_stability', variant=label, settings=current)), flush=True)
                result = _train(output/label, train, val, 'visual', current, provenance)
                training_results[label] = result
                for which in ('best','last'):
                    checkpoint = result[which+'_checkpoint']
                    rows, payload = evaluate(val, checkpoint, current['backend'], current['device_index'],
                                             current['batch_size'])
                    summary = _summary(rows, variant=label+'-'+which, arm=arm, seed=seed,
                                       checkpoint_kind=which, checkpoint=checkpoint,
                                       checkpoint_sha256=file_sha256(checkpoint), epoch=payload['epoch'])
                    if summary['window_label_hash'] != references[0]['window_label_hash']:
                        raise ValueError('Stability fatigue cohort mismatch')
                    comparisons.append(summary)
                    write_rows(output/(summary['variant']+'-windows.csv'), rows)
                    atomic_json(output/(summary['variant']+'-summary.json'), summary)
                    print(json.dumps(dict(stage='evaluated_head_stability', variant=summary['variant'],
                                          **summary['overall'])), flush=True)
        best = {(r['arm'],r['seed']):r for r in comparisons if r['checkpoint_kind']=='best'}
        aggregate = {}
        for arm in ARMS:
            selected = [best[arm,seed] for seed in seeds]
            aggregate[arm] = dict(seeds=seeds, overall=_stats([r['overall'] for r in selected]),
                by_subject={s:_stats([r['by_subject'][s] for r in selected]) for s in selected[0]['by_subject']})
        paired = {str(seed):{arm:_delta(best[arm,seed],best['A_control',seed]) for arm in ARMS[1:]}
                  for seed in seeds}
        drift = {r['variant']:_delta(r,best[r['arm'],r['seed']])
                 for r in comparisons if r['checkpoint_kind']=='last'}
        report = dict(schema_version='driveguard-fatigue-head-stability-comparison-v1',
            status='complete', task='fatigue', split='val', training_runs=len(ARMS)*len(seeds),
            seeds=seeds, arms=arms, references=references, comparisons=comparisons,
            aggregate_best=aggregate, paired_best_deltas=paired, last_minus_best=drift,
            training_results=training_results, spatial_report=str(path),
            spatial_report_sha256=plan['spatial_report_sha256'], provenance=provenance,
            notes=[
                'Head-only frozen features, labels, masks, training-only normalization and BCE are unchanged.',
                'Four arms compare learning rate /10 and bundled dropout .4 + weight decay .01.',
                'Each run starts from its seed; no initialization from a validation-selected checkpoint.',
                'Best uses validation BCE. Fixed logit 0 is primary; FPR03 is validation-only diagnostic.',
                'Unknown labels are not negatives; unavailable positives remain in denominators.',
                'Per-subject metrics and last-minus-best deltas expose regressions hidden by pooled averages.',
                'This experiment does not evaluate other tasks, use held-out test data, or deploy a winner.'])
        if (output/'comparison.json').exists() and read_json(output/'comparison.json') != report:
            raise ValueError('Completed stability report identity/results mismatch')
        flat, subjects = [], []
        for r in comparisons:
            identity = {k:r[k] for k in ('variant','arm','seed','checkpoint_kind','checkpoint',
                                        'checkpoint_sha256','epoch','window_label_hash')}
            flat.append(dict(identity, **r['overall'],
                fixed_threshold_pass=r['nonzero_tp_with_fpr03_at_fixed_threshold'],
                diagnostic_fpr03_tp=r['validation_fpr03']['tp'], diagnostic_fpr03_fp=r['validation_fpr03']['fp']))
            subjects.extend(dict(identity, subject_id=s, **m) for s,m in r['by_subject'].items())
        write_rows(output/'comparison.csv', flat)
        write_rows(output/'by-subject.csv', subjects)
        atomic_json(output/'comparison.json', report)
        return report
    finally:
        if spatial is not None: spatial.close()
        base.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spatial-report', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43])
    parser.add_argument('--backend', choices=['cpu','cuda','npu'])
    parser.add_argument('--device-index', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--continue-run', action='store_true')
    try:
        report = run(**vars(parser.parse_args(argv)))
        print(json.dumps(dict(status=report['status'], training_runs=report['training_runs'])))
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__': main()
