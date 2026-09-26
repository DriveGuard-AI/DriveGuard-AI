"""Three independent eye experiments with an immutable reference yawn branch."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import torch

from training.common.checkpoint import atomic_json, cpu_tree, load_checkpoint
from training.common.config import read_json
from training.common.data import file_sha256
from training.visual.action_eval import evaluate_events, replay_events, action_metrics, subtype_metrics
from training.visual.eye_audit import write_audit
from training.visual.eye_composition import infer_eyes, compose_events, frame_metrics
from training.visual.eye_diagnostics import _inputs, DEFAULT_VARIANT, IDENTITY
from training.visual.eye_frames import EyeCache
from training.visual.eye_train import EyeDataset, train_eye, load_eye, SELECTION
from training.visual.fatigue_spatial_suite import _summary, _stats
from training.visual.head_adaptation import paired_changes
from training.visual.head_adapt_train import DEFAULTS, validate_settings
from training.visual.report import write_rows


ARMS = (('A_head_independent','head',False),('B_eye_native','eye',False),('C_eye_subject_balanced','eye',True))


def _stage(stage, train, val, reference, settings, provenance, balanced):
    stage.mkdir(exist_ok=True)
    attempts=sorted(stage.glob('train-*'));latest=None;latest_epoch=-1
    def checked(path):
        model,payload=load_eye(path);del model
        if (payload['settings']!=settings or payload['provenance']!=provenance
                or payload['subject_balanced']!=balanced or payload['input_mode']!=train.mode
                or payload['data_summary']['train']!=train.summary or payload['data_summary']['val']!=val.summary):
            raise ValueError('Eye stage configuration/provenance mismatch')
        return payload
    for attempt in attempts:
        if (attempt/'result.json').exists():
            result=read_json(attempt/'result.json');states={}
            for kind in ('initial','best','last'):
                path=attempt/(kind+'.pt')
                if Path(result[kind+'_checkpoint']).resolve()!=path.resolve():
                    raise ValueError('Eye result checkpoint path mismatch')
                states[kind]=checked(path)
            last=states['last'];stopped=last['bad_epochs']>=settings['early_stopping_patience']
            if (last['kind']!='training' or result['completed_epochs']!=last['epoch']
                    or result['best_epoch']!=last['best_epoch'] or result['best_loss']!=last['best_loss']
                    or result['stopped_early']!=stopped or result['history']!=last['history']
                    or result['selection_metric']!=SELECTION or not (last['epoch']>=settings['epochs'] or stopped)):
                raise ValueError('Eye result is not a consistent terminal checkpoint')
            for kind,field,epoch,loss in [('initial','initial_state',0,last['initial_report']['loss']),
                                          ('best','best_state',last['best_epoch'],last['best_loss'])]:
                state=states[kind]
                if (state['kind']!='weights' or state['checkpoint_role']!=kind or state['epoch']!=epoch
                        or state['selected_loss']!=loss or set(state['model_state'])!=set(last[field])
                        or any(not torch.equal(v,last[field][k]) for k,v in state['model_state'].items())):
                    raise ValueError('Eye published snapshot mismatch')
            return result
        if (attempt/'last.pt').exists():
            try: load_checkpoint(attempt/'last.pt')
            except (OSError,ValueError,EOFError): continue
            payload=checked(attempt/'last.pt')
            if payload['epoch']>latest_epoch:latest,latest_epoch=attempt/'last.pt',payload['epoch']
    number=max([int(p.name.split('-')[-1]) for p in attempts]+[0])+1
    return train_eye(train,val,reference,settings,stage/f'train-{number:03d}',provenance,
                     subject_balanced=balanced,resume=latest)


def _state_hash(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        tensor=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(tensor.dtype).encode());digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _with_subtypes(summary,rows,labels):
    summary.update(by_subtype=subtype_metrics(rows,labels),
                   by_subject_subtype={s:subtype_metrics([r for r in rows if r['subject_id']==s],labels)
                                       for s in summary['by_subject']})
    return summary


def run(*,action_report,dataset_root,eye_cache,output_dir,variant=DEFAULT_VARIANT,
        seeds=(42,43),settings=None,continue_run=False):
    settings=validate_settings(dict(DEFAULTS if settings is None else settings))
    seeds=list(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s)!=int for s in seeds):
        raise ValueError('Distinct integer seeds required')
    for seed in seeds:validate_settings(dict(settings,seed=seed))
    source=Path(action_report).resolve(strict=True);output=Path(output_dir).resolve()
    if output.exists() and not continue_run:raise FileExistsError('Use a NEW output_dir or --continue-run')
    with _inputs(source,variant,dataset_root,settings['backend'],settings['device_index']) as data:
        eyes=EyeCache(eye_cache,data['cache'])
        try:
            sources=[*data['sources'],eyes.root]
            if any(output.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(output) for p in sources):
                raise ValueError('Eye comparison output overlaps source artifacts')
            head,labels=data['cache'],data['labels']
            datasets={s:{m:EyeDataset(head,eyes,labels,s,m) for m in ('head','eye')} for s in ('train','val')}
            for split in datasets:
                if datasets[split]['head'].summary!=datasets[split]['eye'].summary:
                    raise ValueError('Head/eye comparison supervision cohort differs')
            cohort={s:{m:d.summary['frame_cohort_sha256'] for m,d in modes.items()} for s,modes in datasets.items()}
            provenance=dict(source_reference=data['provenance'],head_cache=str(head.root),head_cache_sha256=head.manifest_sha256,
                            eye_cache=str(eyes.root),eye_cache_sha256=eyes.manifest_sha256,
                            action_labels_sha256=labels.digest,matched_cohort={s:cohort[s]['head'] for s in cohort})
            plan=dict(schema_version='driveguard-independent-eye-plan-v1',settings=settings,seeds=seeds,
                      reference={k:data['selected'][k] for k in IDENTITY},provenance=provenance,
                      arms=[dict(name=n,mode=m,subject_balanced=b) for n,m,b in ARMS],selection_metric=SELECTION,
                      event_policy=dict(eye_threshold_logit=0.,yawn_threshold_logit=0.,
                                        closed_eye_min_ms=labels.closed_eye_min_ms,reset_on_source_gap_ms=80,
                                        missing_eye='reset_eye_only_preserve_reference_yawn'))
            if output.exists():
                if read_json(output/'plan.json')!=plan:raise ValueError('Eye comparison plan identity mismatch')
            else:output.mkdir(parents=True);atomic_json(output/'plan.json',plan)
            resets={rid:r['reset'] for rid,r in labels.records.items()}
            print(json.dumps(dict(stage='verify_frozen_yawn_reference',variant=variant)),flush=True)
            baseline,reference_predictions=evaluate_events(head,data['model'],data['base'].rows['val'],resets,labels.closed_eye_min_ms)
            reference=_summary(baseline,**{k:data['selected'][k] for k in IDENTITY})
            if any(data['selected'].get(k)!=v for k,v in reference.items()):
                raise ValueError('Published reference event predictions differ')
            if action_metrics(reference_predictions,labels,head)!=data['report']['action_metrics'][variant.removesuffix('-event')]:
                raise ValueError('Published reference action metrics differ')
            yawn=replay_events(head,reference_predictions,baseline,resets,labels.closed_eye_min_ms,branches='yawn')
            yawn_changes=paired_changes(yawn,baseline)
            if yawn_changes['overall']['lost_positive']:
                raise ValueError('Selected reference has positive detections not preserved by frozen yawn alone')
            reference_state=cpu_tree(data['model'].state_dict())
            _with_subtypes(reference,baseline,labels)
            coverage=write_audit(output,head,eyes,labels,baseline,labels.closed_eye_min_ms)
            print(json.dumps(dict(stage='eye_input_audit_ready',path=str(output/'input-audit/index.html'),
                                  train=datasets['train']['head'].summary,val=datasets['val']['head'].summary)),flush=True)
            write_rows(output/'reference-windows.csv',baseline)
            comparisons=[reference];results={};initials={};initial_digest=None
            frames={'original_reference_roi_matched':frame_metrics(
                {rid:values[:,0] for rid,values in reference_predictions.items()},head,eyes,labels)}
            for arm,mode,balanced in ARMS:
                for seed in seeds:
                    name=f'{arm}-seed{seed}';current=dict(settings,seed=seed)
                    print(json.dumps(dict(stage='train_independent_eye',variant=name)),flush=True)
                    result=_stage(output/name,datasets['train'][mode],datasets['val'][mode],data['model'],current,provenance,balanced)
                    results[name]=result
                    _,initial=load_eye(result['initial_checkpoint'])
                    digest=_state_hash(initial['model_state'])
                    if initial_digest is None:initial_digest=digest
                    if digest!=initial_digest:raise ValueError('Independent arms did not start from identical weights')
                    initials[name]=dict(epoch=0,selected_loss=initial['selected_loss'],state_sha256=digest,
                                        data_summary=initial['data_summary'])
                    for kind in ('best','last'):
                        path=result[kind+'_checkpoint'];model,payload=load_eye(path,settings['backend'],settings['device_index'])
                        prediction=infer_eyes(head,eyes,model,baseline,mode,batch_size=32)
                        tag=name+'-'+kind;frames[tag]=frame_metrics(prediction,head,eyes,labels)
                        combined,eye_only=compose_events(head,eyes,reference_predictions,prediction,baseline,resets,labels.closed_eye_min_ms)
                        for readout,rows in [('combined',combined),('eye_only',eye_only)]:
                            summary=_summary(rows,variant=tag+'-'+readout,arm=arm,seed=seed,readout=readout,
                                checkpoint_kind=kind,checkpoint=path,checkpoint_sha256=file_sha256(path),epoch=payload['epoch'])
                            if summary['window_label_hash']!=reference['window_label_hash']:
                                raise ValueError('Independent eye evaluation cohort changed')
                            _with_subtypes(summary,rows,labels)
                            summary['changes_from_reference']=paired_changes(rows,baseline)
                            if readout=='combined' and summary['changes_from_reference']['overall']['lost_positive']:
                                raise ValueError('Frozen yawn positive detection lost')
                            comparisons.append(summary);write_rows(output/(tag+'-'+readout+'-windows.csv'),rows)
                            print(json.dumps(dict(stage='evaluated_independent_eye',variant=summary['variant'],**summary['overall'])),flush=True)
                        del model
            for k,v in reference_state.items():
                if not torch.equal(v,data['model'].state_dict()[k].detach().cpu()):
                    raise ValueError('Reference model weights changed during independent training')
            aggregate={}
            for arm,_,_ in ARMS:
                aggregate[arm]={}
                for readout in ('combined','eye_only'):
                    best=[r for r in comparisons if r.get('arm')==arm and r.get('readout')==readout and r['checkpoint_kind']=='best']
                    aggregate[arm][readout]=dict(overall=_stats([r['overall'] for r in best]),
                        by_subject={s:_stats([r['by_subject'][s] for r in best]) for s in best[0]['by_subject']})
            report=dict(schema_version='driveguard-independent-eye-comparison-v1',status='complete',task='fatigue',split='val',
                        training_runs=len(ARMS)*len(seeds),plan=plan,reference_reproduced=True,reference_weights_unchanged=True,
                        frozen_yawn_changes=yawn_changes,matched_cohort=cohort,initials=initials,
                        comparisons=comparisons,aggregate_best=aggregate,frame_metrics=frames,training_results=results,
                        reference_action_metrics=data['report']['action_metrics'][variant.removesuffix('-event')],
                        input_coverage=coverage['records'],
                        notes=['A/B train on identical unique known ROI-valid frames; C changes only subject sampling.',
                               'Reference yawn logits and original windows/duration are fixed; missing eye resets only eye.',
                               'Eye-only availability additionally requires an observed eye at the window endpoint.',
                               'Eye selection uses mean per-subject balanced eye BCE; no yawn loss or event threshold search.',
                               'Both class weights and subject sampling frequencies come from training only.',
                               'Initial checkpoint can remain best if training does not improve eye validation loss.',
                               'Input audit labels are instantaneous eye states, not fatigue window labels.',
                               'All original validation windows, including missing-eye positives and unknown labels, remain reported.',
                               'These are validation-selected experiments, not independent test results or automatic promotion.',
                               'Inspect pure-eye recovery, actual window false positives and subject14 crops; no other task routing changed.'])
            if (output/'comparison.json').exists() and read_json(output/'comparison.json')!=report:
                raise ValueError('Completed independent eye report differs')
            write_rows(output/'comparison.csv',[dict(variant=r['variant'],arm=r['arm'],seed=r['seed'],
                checkpoint_kind=r['checkpoint_kind'],checkpoint=r['checkpoint'],epoch=r['epoch'],readout=r.get('readout','reference'),
                **r['overall']) for r in comparisons])
            write_rows(output/'by-subject.csv',[dict(variant=r['variant'],subject_id=s,**m) for r in comparisons for s,m in r['by_subject'].items()])
            write_rows(output/'by-subtype.csv',[dict(variant=r['variant'],subject_id=s,subtype=t,**m) for r in comparisons
                for s,types in dict(all=r['by_subtype'],**r['by_subject_subtype']).items() for t,m in types.items()])
            atomic_json(output/'frame-metrics.json',frames)
            atomic_json(output/'comparison.json',report)
            return report
        finally:eyes.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('action-report','dataset-root','eye-cache','output-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--variant',default=DEFAULT_VARIANT)
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43])
    parser.add_argument('--continue-run',action='store_true')
    for k,v in DEFAULTS.items():
        if k!='seed':parser.add_argument('--'+k.replace('_','-'),type=type(v),default=v)
    args=vars(parser.parse_args(argv));settings=dict(DEFAULTS)
    for k in DEFAULTS:
        if k!='seed':settings[k]=args.pop(k)
    try:
        report=run(**args,settings=settings)
        print(json.dumps(dict(status=report['status'],training_runs=report['training_runs'],output_dir=args['output_dir'])))
        return 0
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        print(f'error: {error}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
