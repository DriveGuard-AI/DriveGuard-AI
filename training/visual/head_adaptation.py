"""Warm-start four head fatigue experiments from one verified selected checkpoint."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path

from training.common.checkpoint import atomic_json, load_checkpoint, seed_everything
from training.common.config import read_json
from training.common.data import file_sha256
from training.common.device import select_device
from training.visual.fatigue_spatial_suite import _summary, _stats, _difference
from training.visual.report import write_rows


ARMS = {'A_frozen8':(8,False), 'B_tune8':(8,True),
        'C_frozen32':(32,False), 'D_tune32':(32,True)}


def _aligned(after, before):
    keys = ('record_id','start_slot','end_slot','subject_id','label_known','target')
    if len(after) != len(before) or any(any(a[k]!=b[k] for k in keys) for a,b in zip(after,before)):
        raise ValueError('Prediction cohort mismatch')


def _positive(row):
    return row['available'] and row['logit'] >= 0


def paired_changes(after, before):
    """Track recovered AND lost positives, so a net gain cannot hide regressions."""
    _aligned(after,before)
    def counts(pairs):
        result = dict(recovered_positive=0,lost_positive=0,new_false_positive=0,removed_false_positive=0)
        for a,b in pairs:
            if not b['label_known']: continue
            old,new = _positive(b),_positive(a)
            if new==old: continue
            key = ('recovered_positive' if new else 'lost_positive') if b['target']==1 else (
                'new_false_positive' if new else 'removed_false_positive')
            result[key] += 1
        return result
    return dict(overall=counts(zip(after,before)),
                by_subject={s:counts((a,b) for a,b in zip(after,before) if b['subject_id']==s)
                            for s in sorted({r['subject_id'] for r in before})})


def check_sparse_replay(actual, expected):
    _aligned(actual,expected)
    maximum = 0.
    for a,b in zip(actual,expected):
        if a['available'] != b['available']:
            raise ValueError('Sparse online replay availability mismatch')
        if not b['available']: continue
        x,y = a['logit'],b['logit']
        if (not math.isfinite(x) or not math.isfinite(y) or _positive(a)!=_positive(b)
                or not math.isclose(x,y,rel_tol=1e-4,abs_tol=1e-4)):
            raise ValueError('Sparse online replay predictions differ from selected checkpoint')
        maximum = max(maximum,abs(x-y))
    return dict(status='passed', max_abs_logit_difference=maximum,
                fixed_threshold_predictions_identical=True, availability_identical=True,
                absolute_tolerance=1e-4,relative_tolerance=1e-4)


@contextmanager
def _inputs(report_path, variant, weights, backend, device_index):
    from training.visual.data import PreparedData
    from training.visual.spatial import SpatialData
    from training.visual.encoder import FrozenEncoder
    from training.visual.report import evaluate
    report = read_json(report_path)
    if (report.get('schema_version')!='driveguard-fatigue-head-stability-comparison-v1'
            or report.get('status')!='complete' or report.get('task')!='fatigue'
            or report.get('split')!='val'):
        raise ValueError('Expected complete head stability validation comparison')
    matches = [r for r in report['comparisons'] if r['variant']==variant]
    if len(matches)!=1: raise ValueError('Selected reference variant missing or duplicated')
    selected = matches[0]
    if selected['checkpoint_kind']!='best' or selected['arm'] not in report['arms']:
        raise ValueError('Select a trained best checkpoint from the stability suite')
    path = Path(selected['checkpoint']).resolve(strict=True)
    if file_sha256(path)!=selected['checkpoint_sha256']:
        raise ValueError('Selected reference checkpoint hash mismatch')
    payload = load_checkpoint(path)
    if (payload['settings']!=dict(report['arms'][selected['arm']],seed=selected['seed'])
            or payload['epoch']!=selected['epoch'] or payload['kind']!='weights'
            or payload['provenance']!=report['provenance']
            or payload['provenance']['spatial']['arm']!='head'):
        raise ValueError('Selected reference settings/provenance mismatch')
    spatial_path = Path(report['spatial_report']).resolve(strict=True)
    if file_sha256(spatial_path)!=report['spatial_report_sha256']:
        raise ValueError('Spatial report hash mismatch')
    spatial_report = read_json(spatial_path)
    base = PreparedData(spatial_report['prepared_dir'])
    spatial = None
    try:
        if base.provenance['prepared_sha256']!=payload['provenance']['prepared_sha256']:
            raise ValueError('Reference preparation hash mismatch')
        spatial = SpatialData(base,payload['provenance']['spatial']['head_cache'])
        rows,_ = evaluate(spatial.dataset('val','head'),path,backend,device_index,
                          payload['settings']['batch_size'])
        identity = {k:selected[k] for k in ('variant','arm','seed','checkpoint_kind',
                                          'checkpoint','checkpoint_sha256','epoch')}
        reproduced = _summary(rows,**identity)
        if reproduced!=selected: raise ValueError('Selected reference metrics/predictions mismatch')
        # This verifies the original official weight digest recorded with the RGB cache.
        encoder = FrozenEncoder(weights,base.cache.report['encoder']['weights_sha256'])
        yield dict(base=base,payload=payload,reference=reproduced,rows=rows,
                   encoder_state=encoder.model.state_dict(), crop=payload['provenance']['spatial']['crop'],
                   sources=[Path(report_path).parent,spatial_path.parent,base.root,base.cache.root,
                            spatial.head.root,Path(base.report['data_root']).resolve(),path.parent,
                            Path(base.report['manifest']).resolve(),Path(weights).resolve()])
    finally:
        if spatial is not None: spatial.close()
        base.close()


def _stage(stage, train, val, payload, encoder_state, settings, provenance, frames, tune):
    from training.visual.head_adapt_train import train_adaptation, validate_adaptation
    stage.mkdir(exist_ok=True)
    params = dict(frames=frames,tune_encoder=tune,frame_chunk=settings['frame_chunk'],
                  probe_params=payload['model_params'])
    def verified(path):
        state = load_checkpoint(path)
        validate_adaptation(state)
        if state['settings']!=settings or state['provenance']!=provenance or state['model_params']!=params:
            raise ValueError('Adaptation stage configuration/provenance mismatch')
        return state
    attempts = sorted(stage.glob('train-*'))
    latest, resume = None,None
    for attempt in attempts:
        result_path = attempt/'result.json'
        if result_path.exists():
            result = read_json(result_path)
            states = {}
            for kind in ('initial','best','last'):
                expected = attempt/(kind+'.pt')
                if Path(result[kind+'_checkpoint']).resolve()!=expected.resolve():
                    raise ValueError('Stage result checkpoint path mismatch')
                states[kind] = verified(expected)
            terminal = states['last']
            stopped = terminal.get('early_stopping_state',{}).get('stopped_early',False)
            if (terminal['kind']!='training' or result['completed_epochs']!=terminal['epoch']
                    or result['best_epoch']!=states['best']['epoch'] or states['initial']['epoch']!=0
                    or states['best']['epoch']!=terminal['best_epoch']
                    or states['best']['kind']!='weights' or states['initial']['kind']!='weights'
                    or result['best_val_loss']!=terminal['best_val_loss']
                    or states['best']['best_val_loss']!=terminal['best_val_loss']
                    or result['stopped_early']!=stopped
                    or not (terminal['epoch']>=settings['epochs'] or stopped)):
                raise ValueError('Incomplete/inconsistent adaptation stage')
            import torch
            for kind, saved in (('initial','initial_model_state'),('best','best_model_state')):
                actual,expected = states[kind]['model_state'],terminal[saved]
                if set(actual)!=set(expected) or any(not torch.equal(actual[k],expected[k]) for k in actual):
                    raise ValueError('Adaptation stage saved model state mismatch')
            return result
        if (attempt/'last.pt').exists():
            try:
                # A partial atomic checkpoint/sidecar publication can be bypassed, never published results.
                load_checkpoint(attempt/'last.pt')
            except (OSError,ValueError,EOFError): continue
            state = verified(attempt/'last.pt')
            if latest is None or state['epoch']>=latest['epoch']:
                latest,resume = state,attempt/'last.pt'
    return train_adaptation(train,val,payload,encoder_state,settings,
                            stage/f'train-{len(attempts)+1:03d}',provenance,frames,tune,resume=resume)


def run(*, stability_report, dataset_root, weights, output_dir,
        variant='C_regularized-seed42-best', seeds=(42,43), settings=None, continue_run=False):
    from training.visual.head_adapt_train import (DEFAULTS,validate_settings,AdaptModel,
                                                  evaluate_model,evaluate_adaptation)
    from training.visual.head_frames import build_frame_cache,FrameCache,FrameDataset
    settings = validate_settings(dict(DEFAULTS if settings is None else settings))
    seeds = list(seeds)
    if not seeds or any(type(s)!=int for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('Distinct integer seeds required')
    for seed in seeds: validate_settings(dict(settings,seed=seed))
    path,output = Path(stability_report).resolve(strict=True),Path(output_dir).resolve()
    dataset_root,weights = Path(dataset_root).resolve(strict=True),Path(weights).resolve(strict=True)
    with _inputs(path,variant,weights,settings['backend'],settings['device_index']) as inputs:
        sources = inputs['sources']+[dataset_root]
        if any(output.is_relative_to(p) or p.is_relative_to(output) for p in sources):
            raise ValueError('Adaptation output overlaps source artifacts')
        plan = dict(schema_version='driveguard-head-adaptation-plan-v1',
                    stability_report=str(path),stability_report_sha256=file_sha256(path),
                    selected_reference=inputs['reference'],dataset_root=str(dataset_root),
                    weights=str(weights),weights_sha256=file_sha256(weights),settings=settings,seeds=seeds,
                    arms={k:list(v) for k,v in ARMS.items()},provenance=inputs['payload']['provenance'])
        if output.exists():
            if not continue_run: raise FileExistsError('Use a NEW output_dir or --continue-run')
            if read_json(output/'plan.json')!=plan: raise ValueError('Adaptation configuration/identity mismatch')
        else:
            output.mkdir(parents=True)
            atomic_json(output/'plan.json',plan)
        print(json.dumps(dict(stage='build_head_pixels',output=str(output/'head-frames'))),flush=True)
        build_frame_cache(inputs['base'],inputs['crop'],dataset_root,output/'head-frames',
                          continue_run=(output/'head-frames').exists())
        cache = FrameCache(output/'head-frames')
        try:
            datasets = {n:{s:FrameDataset(inputs['base'],cache,s,n) for s in ('train','val')}
                        for n in (8,32)}
            provenance = dict(source=inputs['payload']['provenance'],
                warm_start_checkpoint=inputs['reference']['checkpoint'],
                warm_start_sha256=inputs['reference']['checkpoint_sha256'],
                frame_cache=str(cache.root),frame_cache_sha256=cache.manifest_sha256,
                encoder_weights_sha256=plan['weights_sha256'])
            device = select_device(settings['backend'],settings['device_index'])
            seed_everything(seeds[0],device)
            initial = AdaptModel(inputs['payload'],inputs['encoder_state'],8,False,
                                 frame_chunk=settings['frame_chunk']).to(device).eval()
            rows = evaluate_model(datasets[8]['val'],initial,batch_size=settings['batch_size'])
            replay = check_sparse_replay(rows,inputs['rows'])
            del initial
            atomic_json(output/'reference-replay.json',replay)
            comparisons, predictions, results = [inputs['reference']],{},{}
            write_rows(output/'reference-windows.csv',inputs['rows'])
            for arm,(frames,tune) in ARMS.items():
                for seed in seeds:
                    label = f'{arm}-seed{seed}'
                    current = dict(settings,seed=seed)
                    print(json.dumps(dict(stage='training_head_adaptation',variant=label,
                                          frames=frames,tune_encoder=tune,settings=current)),flush=True)
                    result = _stage(output/label,datasets[frames]['train'],datasets[frames]['val'],
                                    inputs['payload'],inputs['encoder_state'],current,provenance,frames,tune)
                    results[label] = result
                    for kind in ('initial','best','last'):
                        checkpoint = result[kind+'_checkpoint']
                        rows,trained = evaluate_adaptation(datasets[frames]['val'],checkpoint,
                            settings['backend'],settings['device_index'],settings['batch_size'])
                        summary = _summary(rows,variant=label+'-'+kind,arm=arm,seed=seed,
                            checkpoint_kind=kind,checkpoint=checkpoint,checkpoint_sha256=file_sha256(checkpoint),
                            epoch=trained['epoch'])
                        if summary['window_label_hash']!=inputs['reference']['window_label_hash']:
                            raise ValueError('Adaptation evaluation cohort mismatch')
                        summary['changes_from_original'] = paired_changes(rows,inputs['rows'])
                        comparisons.append(summary)
                        predictions[summary['variant']] = rows
                        write_rows(output/(summary['variant']+'-windows.csv'),rows)
                        atomic_json(output/(summary['variant']+'-summary.json'),summary)
                        print(json.dumps(dict(stage='evaluated_head_adaptation',variant=summary['variant'],
                                              **summary['overall'])),flush=True)
            report = _report(plan,comparisons,results,replay,predictions)
            if (output/'comparison.json').exists() and read_json(output/'comparison.json')!=report:
                raise ValueError('Completed adaptation report differs')
            _write_report(output,report)
            return report
        finally: cache.close()


def _report(plan,comparisons,results,replay,predictions):
    best = {(r['arm'],r['seed']):r for r in comparisons if r['checkpoint_kind']=='best'}
    aggregate = {}
    for arm in ARMS:
        rows = [best[arm,s] for s in plan['seeds']]
        aggregate[arm] = dict(overall=_stats([r['overall'] for r in rows]),
            by_subject={s:_stats([r['by_subject'][s] for r in rows]) for s in rows[0]['by_subject']})
    paired = {}
    for seed in plan['seeds']:
        baseline = best['A_frozen8',seed]
        paired[str(seed)] = {arm:dict(
            overall=_difference(best[arm,seed]['overall'],baseline['overall']),
            by_subject={s:_difference(best[arm,seed]['by_subject'][s],baseline['by_subject'][s])
                        for s in baseline['by_subject']},
            changes=paired_changes(predictions[best[arm,seed]['variant']],predictions[baseline['variant']]))
            for arm in list(ARMS)[1:]}
    return dict(schema_version='driveguard-head-adaptation-comparison-v1',status='complete',task='fatigue',
        split='val',training_runs=len(ARMS)*len(plan['seeds']),plan=plan,reference_replay=replay,
        comparisons=comparisons,training_results=results,aggregate_best=aggregate,paired_best_deltas=paired,
        notes=[
            'All arms warm-start the same selected best probe with a new optimizer; seeds change order/dropout.',
            'Frozen original normalization, frozen BN, and identical head crop; only layer4 is fine-tuned in B/D.',
            'All arms add the same zero-gated temporal residual; 8-frame initial predictions must reproduce reference.',
            'The fourth dilation9 layer gives a 33-frame receptive field; epoch0 gate zero preserves old predictions.',
            'Best includes epoch0; further training cannot silently replace it with a worse validation BCE candidate.',
            'All original validation windows remain. Unknown labels are not negatives; missing positives remain counted.',
            'Fixed logit0 is primary; FPR03 thresholds are validation-only diagnostics.',
            'No test split is trained/evaluated. This does not replace the composite model or evaluate other tasks.'])


def _write_report(output,report):
    flat,subjects = [],[]
    for r in report['comparisons']:
        identity = {k:r[k] for k in ('variant','arm','seed','checkpoint_kind','checkpoint',
                                    'checkpoint_sha256','epoch','window_label_hash')}
        flat.append(dict(identity,**r['overall'],
            fixed_threshold_pass=r['nonzero_tp_with_fpr03_at_fixed_threshold'],
            diagnostic_fpr03_tp=r['validation_fpr03']['tp'],diagnostic_fpr03_fp=r['validation_fpr03']['fp']))
        subjects.extend(dict(identity,subject_id=s,**m) for s,m in r['by_subject'].items())
    write_rows(output/'comparison.csv',flat)
    write_rows(output/'by-subject.csv',subjects)
    atomic_json(output/'comparison.json',report)


def main(argv=None):
    from training.visual.head_adapt_train import DEFAULTS
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stability-report','dataset-root','weights','output-dir'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--variant',default='C_regularized-seed42-best')
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43])
    parser.add_argument('--continue-run',action='store_true')
    for k,v in DEFAULTS.items():
        if k!='seed': parser.add_argument('--'+k.replace('_','-'),type=type(v),default=v)
    args = vars(parser.parse_args(argv))
    settings = dict(DEFAULTS)
    for key in DEFAULTS:
        if key!='seed': settings[key]=args.pop(key)
    try:
        result = run(**args,settings=settings)
        print(json.dumps(dict(status=result['status'],training_runs=result['training_runs'])))
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as exc:
        parser.exit(2,f'error: {exc}\n')


if __name__=='__main__': main()
