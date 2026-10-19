"""Compare direct fatigue, explicit action supervision and causal event readout."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys

import torch

from training.common.checkpoint import atomic_json,load_checkpoint
from training.common.config import read_json
from training.common.data import file_sha256
from training.visual.action_labels import ActionLabels,ActionDataset
from training.visual.action_model import ActionModel
from training.visual.action_train import train_actions,load_actions
from training.visual.action_eval import evaluate_direct,evaluate_events,action_metrics,subtype_metrics
from training.visual.data import PreparedData
from training.visual.head_frames import FrameCache,FrameDataset
from training.visual.head_adapt_train import (DEFAULTS,validate_settings,load_adaptation,
                                              evaluate_model,_verify_dataset_identity)
from training.visual.head_adaptation import check_sparse_replay,paired_changes,_write_report
from training.visual.fatigue_spatial_suite import _summary,_stats
from training.visual.report import write_rows


@contextmanager
def _inputs(report_path,variant,dataset_root,settings):
    print(json.dumps(dict(stage='verify_action_reference',variant=variant)),flush=True)
    report=read_json(report_path)
    if (report.get('schema_version')!='driveguard-head-adaptation-comparison-v1'
            or report.get('status')!='complete' or report.get('split')!='val' or report.get('task')!='fatigue'):
        raise ValueError('Expected completed head adaptation validation report')
    matches=[r for r in report['comparisons'] if r['variant']==variant]
    if len(matches)!=1: raise ValueError('Selected adaptation variant missing or duplicated')
    selected=matches[0]
    if selected['checkpoint_kind']!='best' or selected['arm']!='D_tune32':
        raise ValueError('Expected a D_tune32 best checkpoint')
    path=Path(selected['checkpoint']).resolve(strict=True)
    if file_sha256(path)!=selected['checkpoint_sha256']: raise ValueError('Reference checkpoint hash mismatch')
    backbone,payload=load_adaptation(path,settings['backend'],settings['device_index'])
    if (payload['kind']!='weights' or payload['epoch']!=selected['epoch']
            or payload['settings']['seed']!=selected['seed'] or payload['model_params']['frames']!=32
            or not payload['model_params']['tune_encoder']):
        raise ValueError('Reference checkpoint metadata mismatch')
    cache=base=None
    try:
        print(json.dumps(dict(stage='verify_existing_head_pixels',path=payload['provenance']['frame_cache'])),flush=True)
        cache=FrameCache(payload['provenance']['frame_cache'])
        run=read_json(cache.root/'input.json')
        if Path(run['dataset_root']).resolve()!=Path(dataset_root).resolve():
            raise ValueError('Dataset root differs from original pixel cache')
        base=PreparedData(run['prepared_root'])
        dataset=FrameDataset(base,cache,'val',32)
        _verify_dataset_identity(dataset,payload['provenance'])
        rows=evaluate_model(dataset,backbone,payload['settings']['batch_size'])
        identity={k:selected[k] for k in ('variant','arm','seed','checkpoint_kind','checkpoint','checkpoint_sha256','epoch')}
        reference=_summary(rows,**identity)
        if any(selected.get(k)!=v for k,v in reference.items()):
            raise ValueError('Reference metrics/predictions differ from published adaptation report')
        print(json.dumps(dict(stage='compile_action_labels',reference=variant)),flush=True)
        labels=ActionLabels(base,cache,dataset_root)
        provenance=dict(reference_checkpoint=str(path),reference_sha256=selected['checkpoint_sha256'],
                        reference_report=str(report_path),reference_report_sha256=file_sha256(report_path),
                        backbone_provenance=payload['provenance'],action_labels=labels.provenance,
                        action_labels_sha256=labels.digest)
        yield dict(base=base,cache=cache,labels=labels,backbone=backbone,reference=reference,rows=rows,
                   provenance=provenance,sources=[Path(report_path).parent,path.parent,cache.root,base.root,
                          base.cache.root,Path(base.report['data_root']),Path(base.report['manifest']),Path(dataset_root)])
    finally:
        if cache is not None: cache.close()
        if base is not None: base.close()


def _stage(stage,train,val,backbone,settings,provenance,auxiliary):
    stage.mkdir(exist_ok=True)
    latest=None; latest_epoch=-1
    attempts=sorted(stage.glob('train-*'))
    def checked(path):
        model,payload=load_actions(path)
        del model
        if (payload['settings']!=settings or payload['provenance']!=provenance or payload['auxiliary']!=auxiliary):
            raise ValueError('Action stage configuration/provenance mismatch')
        return payload
    for attempt in attempts:
        if (attempt/'result.json').exists():
            result=read_json(attempt/'result.json')
            states={}
            for kind in ('initial','best','last','fatigue-best'):
                expected=attempt/(kind+'.pt')
                if Path(result[kind.replace('-','_')+'_checkpoint']).resolve()!=expected.resolve():
                    raise ValueError('Action result checkpoint path mismatch')
                states[kind]=checked(expected)
            last=states['last']; stopped=last['bad_epochs']>=settings['early_stopping_patience']
            if (last['kind']!='training' or result['completed_epochs']!=last['epoch']
                    or result['best_epoch']!=last['best_epoch'] or result['best_loss']!=last['best_loss']
                    or result['fatigue_best_epoch']!=last['fatigue_epoch'] or result['stopped_early']!=stopped
                    or not (last['epoch']>=settings['epochs'] or stopped)):
                raise ValueError('Action result is not a consistent terminal checkpoint')
            for kind,field,epoch in (('initial','initial_state',0),('best','best_state',last['best_epoch']),
                                      ('fatigue-best','fatigue_state',last['fatigue_epoch'])):
                state=states[kind]
                if (state['kind']!='weights' or state['epoch']!=epoch or
                        set(state['model_state'])!=set(last[field]) or
                        any(not torch.equal(v,last[field][k]) for k,v in state['model_state'].items())):
                    raise ValueError('Action published snapshot mismatch')
            return result
        if (attempt/'last.pt').exists():
            try: load_checkpoint(attempt/'last.pt')
            except (OSError,ValueError,EOFError): continue
            state=checked(attempt/'last.pt')
            if state['epoch']>latest_epoch: latest,latest_epoch=attempt/'last.pt',state['epoch']
    number=max([int(p.name.split('-')[-1]) for p in attempts]+[0])+1
    return train_actions(train,val,deepcopy(backbone),settings,stage/f'train-{number:03d}',provenance,auxiliary,resume=latest)


def run(*,adaptation_report,dataset_root,output_dir,variant='D_tune32-seed43-best',
        seeds=(42,43),settings=None,continue_run=False):
    settings=validate_settings(dict(DEFAULTS if settings is None else settings))
    seeds=list(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s)!=int for s in seeds):
        raise ValueError('Distinct integer seeds required')
    for seed in seeds: validate_settings(dict(settings,seed=seed))
    source=Path(adaptation_report).resolve(strict=True); output=Path(output_dir).resolve()
    with _inputs(source,variant,dataset_root,settings) as data:
        if any(output.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(output) for p in data['sources']):
            raise ValueError('Action output overlaps source artifacts')
        labels,cache=data['labels'],data['cache']
        plan=dict(schema_version='driveguard-fatigue-actions-plan-v1',settings=settings,seeds=seeds,
                  provenance=data['provenance'],reference=data['reference'],
                  event_policy=dict(action_logit_threshold=0.,closed_eye_min_ms=labels.closed_eye_min_ms,
                                    reset_on_source_gap_ms=80,threshold_search=False),
                  auxiliary_loss_weights=dict(eye=.5,yawn=.5,fatigue=1.))
        if output.exists():
            if not continue_run: raise FileExistsError('Use a NEW output_dir or --continue-run')
            if read_json(output/'plan.json')!=plan: raise ValueError('Action plan identity mismatch')
        else:
            output.mkdir(parents=True); atomic_json(output/'plan.json',plan)
        datasets={s:ActionDataset(data['base'],cache,s,labels) for s in ('train','val')}
        coverage=labels.coverage(cache)
        atomic_json(output/'action-label-audit.json',dict(status='complete',provenance=labels.provenance,
                  coverage=coverage,notes=['Counts are unique sampled frames. Unknown is never a negative.']))
        replay=check_sparse_replay(evaluate_direct(datasets['val'],ActionModel(deepcopy(data['backbone'])).to(
                                  next(data['backbone'].parameters()).device),settings['batch_size']),data['rows'])
        atomic_json(output/'reference-replay.json',replay)
        reference=deepcopy(data['reference']); reference['by_subtype']=subtype_metrics(data['rows'],labels)
        reference['by_subject_subtype']={s:subtype_metrics([r for r in data['rows'] if r['subject_id']==s],labels)
                                        for s in reference['by_subject']}
        comparisons=[reference]; results={}; action_reports={}
        write_rows(output/'reference-windows.csv',data['rows'])
        resets={rid:value['reset'] for rid,value in labels.records.items()}
        for arm,auxiliary in (('A_direct_control',False),('B_action_supervision',True)):
            for seed in seeds:
                name=f'{arm}-seed{seed}'; current=dict(settings,seed=seed)
                print(json.dumps(dict(stage='train_fatigue_actions',variant=name)),flush=True)
                result=_stage(output/name,datasets['train'],datasets['val'],data['backbone'],current,data['provenance'],auxiliary)
                results[name]=result
                for kind in ('best','fatigue-best','last'):
                    if not auxiliary and kind=='fatigue-best': continue
                    path=result[kind.replace('-','_')+'_checkpoint']
                    model,payload=load_actions(path,settings['backend'],settings['device_index'])
                    direct=evaluate_direct(datasets['val'],model,settings['batch_size'])
                    outputs=[('direct',direct)]
                    if auxiliary:
                        event,predictions=evaluate_events(cache,model,direct,resets,labels.closed_eye_min_ms)
                        outputs.append(('event',event))
                        action_reports[name+'-'+kind]=action_metrics(predictions,labels,cache)
                    for readout,rows in outputs:
                        tag=name+'-'+kind+'-'+readout
                        summary=_summary(rows,variant=tag,arm=arm+'_'+readout,seed=seed,
                           checkpoint_kind=kind,checkpoint=str(path),checkpoint_sha256=file_sha256(path),epoch=payload['epoch'])
                        if summary['window_label_hash']!=reference['window_label_hash']:
                            raise ValueError('Action evaluation cohort changed')
                        summary.update(changes_from_original=paired_changes(rows,data['rows']),
                                       by_subtype=subtype_metrics(rows,labels),
                                       by_subject_subtype={s:subtype_metrics([r for r in rows if r['subject_id']==s],labels)
                                                           for s in summary['by_subject']},
                                       selection_metric=payload.get('checkpoint_selection_metric',payload['selection_metric']),
                                       evaluation_role='diagnostic_fatigue_selected' if kind=='fatigue-best' else kind)
                        comparisons.append(summary)
                        write_rows(output/(tag+'-windows.csv'),[dict(r,positive_subtype=str(labels.records[r['record_id']]['subtypes'][r['end_slot']])) for r in rows])
                        print(json.dumps(dict(stage='evaluated_fatigue_actions',variant=tag,**summary['overall'])),flush=True)
                    del model
        aggregate={}
        for arm in ('A_direct_control_direct','B_action_supervision_direct','B_action_supervision_event'):
            best=[r for r in comparisons if r['arm']==arm and r['checkpoint_kind']=='best']
            aggregate[arm]=dict(overall=_stats([r['overall'] for r in best]),
                  by_subject={s:_stats([r['by_subject'][s] for r in best]) for s in best[0]['by_subject']})
        report=dict(schema_version='driveguard-fatigue-actions-comparison-v1',status='complete',task='fatigue',split='val',
                    training_runs=2*len(seeds),plan=plan,reference_replay=replay,comparisons=comparisons,
                    training_results=results,action_metrics=action_reports,aggregate_best=aggregate,
                    notes=['D43 and these checkpoints are validation-selected; this is not an independent test.',
                           'Direct control and auxiliary arms warm-start the SAME D43 and sample the same supervision union.',
                           'Auxiliary best uses mean known eye/yawn validation BCE; direct and event readouts share weights.',
                           'fatigue-best is a separate diagnostic snapshot selected by fatigue BCE, not event performance.',
                           'Action labels use original source PTS and camera shifts, not broadcast window-end fatigue.',
                           'Yawning absence is negative only with explicit open-eye evidence; other absence stays unknown.',
                           'Event inference uses predicted actions and source-clock gaps only; annotations are evaluation-only.',
                           '8Hz closure duration is sampled evidence, not exact frame-level duration; unseen short blinks can be missed.',
                           'Fixed logit0 thresholds and original duration; any FPR03 sweep is diagnostic only.',
                           'All original validation windows retained. No test data or other task routing changed.'])
        if (output/'comparison.json').exists() and read_json(output/'comparison.json')!=report:
            raise ValueError('Completed action report differs')
        subtype_rows=[]
        for row in comparisons:
            groups=dict(all=row['by_subtype'])
            groups.update(row.get('by_subject_subtype',{}))
            for subject,subtypes in groups.items():
                for label,metrics in subtypes.items():
                    subtype_rows.append(dict(variant=row['variant'],subject_id=subject,subtype=label,**metrics))
        write_rows(output/'by-subtype.csv',subtype_rows)
        atomic_json(output/'action-metrics.json',action_reports)
        _write_report(output,report)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('adaptation-report','dataset-root','output-dir'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--variant',default='D_tune32-seed43-best')
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43])
    parser.add_argument('--continue-run',action='store_true')
    for k,v in DEFAULTS.items():
        if k!='seed': parser.add_argument('--'+k.replace('_','-'),type=type(v),default=v)
    args=vars(parser.parse_args(argv)); settings=dict(DEFAULTS)
    for k in DEFAULTS:
        if k!='seed': settings[k]=args.pop(k)
    try:
        report=run(**args,settings=settings)
        print(json.dumps(dict(status=report['status'],training_runs=report['training_runs'],output_dir=args['output_dir'])))
        return 0
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        print(f'error: {error}',file=sys.stderr); return 2


if __name__=='__main__': raise SystemExit(main())
