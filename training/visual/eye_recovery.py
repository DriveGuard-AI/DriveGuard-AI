"""Compare eye localization policies with frozen eye/yawn weights; zero training."""
import argparse
from contextlib import contextmanager, ExitStack
from pathlib import Path
from types import SimpleNamespace
import json
import sys

from training.common.checkpoint import atomic_json
from training.common.config import read_json
from training.common.data import file_sha256
from training.visual.action_eval import evaluate_events, action_metrics
from training.visual.eye_audit import write_audit
from training.visual.eye_composition import compose_events, infer_eyes, frame_metrics
from training.visual.eye_diagnostics import _inputs
from training.visual.eye_frames import EyeCache
from training.visual.eye_support import summarize_support
from training.visual.eye_train import load_eye
from training.visual.fatigue_eyes import _state_hash, _with_subtypes
from training.visual.fatigue_spatial_suite import _summary
from training.visual.head_adaptation import paired_changes
from training.visual.report import write_rows


VARIANT='B_eye_native-seed42-best-combined'
MODES=('sampled8','dense','dense_track')


@contextmanager
def _reference(eye_report,variant,dataset_root,backend,device_index):
    if variant!=VARIANT:raise ValueError('This comparison fixes B_eye_native seed42 best; variant cannot change')
    source=Path(eye_report).resolve(strict=True);digest=file_sha256(source);report=read_json(source)
    if (report.get('schema_version')!='driveguard-independent-eye-comparison-v1' or report.get('status')!='complete'
            or report.get('task')!='fatigue' or report.get('split')!='val' or not report.get('reference_reproduced')):
        raise ValueError('Expected a complete independent-eye comparison')
    choices=[r for r in report['comparisons'] if r['variant']==variant]
    if len(choices)!=1:raise ValueError('Frozen eye variant missing or ambiguous')
    selected=choices[0];path=Path(selected['checkpoint']).resolve(strict=True)
    if (selected['readout']!='combined' or selected['checkpoint_kind']!='best'
            or selected['arm']!='B_eye_native' or selected['seed']!=42
            or file_sha256(path)!=selected['checkpoint_sha256']):
        raise ValueError('Frozen native eye checkpoint selection/hash mismatch')
    model,payload=load_eye(path,backend,device_index)
    if (payload['input_mode']!='eye' or payload['subject_balanced'] is not False
            or payload['kind']!='weights' or payload['checkpoint_role']!='best'
            or payload['epoch']!=selected['epoch'] or payload['settings']['seed']!=selected['seed']
            or payload['provenance']!=report['plan']['provenance']):
        raise ValueError('Frozen eye checkpoint provenance mismatch')
    original=payload['provenance']['source_reference']
    action_source=Path(original['action_report']).resolve(strict=True)
    if file_sha256(action_source)!=original['action_report_sha256']:
        raise ValueError('Original action report checksum mismatch')
    with _inputs(action_source,report['plan']['reference']['variant'],dataset_root,backend,device_index) as data:
        if data['provenance']!=original or data['cache'].manifest_sha256!=payload['provenance']['head_cache_sha256']:
            raise ValueError('Original action/head identity mismatch')
        old=EyeCache(payload['provenance']['eye_cache'],data['cache'])
        try:
            if old.manifest_sha256!=payload['provenance']['eye_cache_sha256'] or old.report['run'].get('recovery'):
                raise ValueError('Original native eye cache identity mismatch')
            if file_sha256(source)!=digest or file_sha256(path)!=selected['checkpoint_sha256']:
                raise ValueError('Reference changed during verification')
            yield dict(data,eye_model=model,eye_payload=payload,old_eyes=old,selected_eye=selected,
                source_eye_report=report,provenance=dict(eye_report=str(source),eye_report_sha256=digest,
                    eye_checkpoint=str(path),eye_checkpoint_sha256=selected['checkpoint_sha256'],source_reference=original),
                sources=[*data['sources'],source.parent,path.parent,old.root])
        finally:old.close()


def _reproduce(rows,published,name):
    if any(published.get(k)!=v for k,v in _summary(rows).items()):
        raise ValueError(name+' predictions differ from published reference')


def _rejections(cache,variant):
    result=[]
    for r in cache.report['records']:
        audit=r['audit'];detail=audit.get('localization',{})
        for stage,counts in [('sample_geometry',audit['reasons']),('sample_detector',detail.get('sampled_detector_reasons',{})),
                             ('source_detector_calls',detail.get('detector_calls',{})),('origin',detail.get('origins',{}))]:
            result.extend(dict(variant=variant,record_id=r['record_id'],subject_id=r['subject_id'],split=r['split'],
                               stage=stage,reason=reason,count=count) for reason,count in counts.items())
    return result


def run(*,eye_report,dataset_root,recovery_root,output_dir,variant=VARIANT,backend='cpu',device_index=0,continue_run=False):
    output=Path(output_dir).resolve();recovery_root=Path(recovery_root).resolve()
    if output.exists() and not continue_run:raise FileExistsError('Use a NEW output_dir or --continue-run')
    with _reference(eye_report,variant,dataset_root,backend,device_index) as data,ExitStack() as stack:
        head,labels=data['cache'],data['labels'];caches={'original':data['old_eyes']}
        expected={rid for rid,r in head.records.items() if r['split']=='val'}
        for mode in MODES:
            eyes=EyeCache(recovery_root/mode,head);stack.callback(eyes.close)
            policy=eyes.report['run'].get('recovery',{})
            if set(eyes.records)!=expected or policy.get('mode')!=mode or policy.get('split')!='val':
                raise ValueError('Recovery cache must contain all and only original validation records with matching policy')
            caches[mode]=eyes
        for source in [*data['sources'],*[c.root for c in caches.values()]]:
            path=Path(source).resolve()
            if output.is_relative_to(path) or path.is_relative_to(output):
                raise ValueError('Recovery output overlaps source artifacts')
        plan=dict(schema_version='driveguard-eye-recovery-plan-v1',provenance=data['provenance'],variant=variant,
            backend=backend,device_index=device_index,training_runs=0,
            caches={m:dict(path=str(c.root),manifest_sha256=c.manifest_sha256,policy=c.report['run'].get('recovery')) for m,c in caches.items()},
            event_policy=dict(closed_eye_min_ms=labels.closed_eye_min_ms,threshold_logit=0.,missing='reset-eye-only'))
        if output.exists():
            if read_json(output/'plan.json')!=plan:raise ValueError('Recovery resume plan mismatch')
        else:output.mkdir(parents=True);atomic_json(output/'plan.json',plan)
        # Audits come before new-eye inference. These bounds do not use scores.
        val_head=SimpleNamespace(records={rid:head.records[rid] for rid in sorted(expected)},arrays=head.arrays,report=head.report)
        audits={};support={};rejections=[]
        for mode,eyes in caches.items():
            directory=output/mode;directory.mkdir(exist_ok=True)
            audits[mode]=write_audit(directory,val_head,eyes,labels,data['base'].rows['val'],labels.closed_eye_min_ms)
            support[mode]=summarize_support(audits[mode]['windows'])
            rejections.extend(_rejections(eyes,mode))
            print(json.dumps(dict(stage='eye_input_support',variant=mode,**support[mode]['overall'])),flush=True)
        atomic_json(output/'input-support.json',support)
        if rejections:write_rows(output/'rejection-reasons.csv',rejections)
        atomic_json(output/'preflight.json',dict(status='complete',training_runs=0,input_support=support,
            note='Zero support blocks new training, but fixed-model diagnostic inference still runs to measure negative errors.'))
        states={name:_state_hash(data[name].state_dict()) for name in ('model','eye_model')}
        resets={rid:r['reset'] for rid,r in labels.records.items()}
        baseline,reference=evaluate_events(head,data['model'],data['base'].rows['val'],resets,labels.closed_eye_min_ms)
        _reproduce(baseline,data['selected'],'Original yawn/action')
        if action_metrics(reference,labels,head)!=data['report']['action_metrics'][data['selected']['variant'].removesuffix('-event')]:
            raise ValueError('Original action frame metrics differ')
        comparisons=[];frames={};old_combined=None
        for mode,eyes in caches.items():
            prediction=infer_eyes(head,eyes,data['eye_model'],baseline,'eye',batch_size=32)
            frames[mode]=frame_metrics(prediction,head,eyes,labels)
            combined,eye_only=compose_events(head,eyes,reference,prediction,baseline,resets,labels.closed_eye_min_ms)
            if mode=='original':
                _reproduce(combined,data['selected_eye'],'Frozen eye')
                if frames[mode]!=data['source_eye_report']['frame_metrics'][variant.removesuffix('-combined')]:
                    raise ValueError('Frozen eye frame metrics differ')
                old_combined=combined
            elif mode=='sampled8':
                _reproduce(combined,data['selected_eye'],'Sampled detector control')
                if frames[mode]!=frames['original']:raise ValueError('Sampled detector control frame metrics differ')
            for readout,rows in [('combined',combined),('eye_only',eye_only)]:
                summary=_with_subtypes(_summary(rows,variant=mode+'-'+readout,readout=readout),rows,labels)
                if summary['window_label_hash']!=data['selected_eye']['window_label_hash']:
                    raise ValueError('Recovery evaluation cohort changed')
                summary['changes_from_original_eye']=paired_changes(rows,old_combined)
                summary['changes_from_original_event']=paired_changes(rows,baseline)
                if readout=='combined' and summary['changes_from_original_event']['overall']['lost_positive']:
                    raise ValueError('Recovery suppressed an original event positive')
                comparisons.append(summary);write_rows(output/(mode+'-'+readout+'-windows.csv'),rows)
            print(json.dumps(dict(stage='evaluated_eye_recovery',variant=mode,**comparisons[-2]['overall'])),flush=True)
        if states!={name:_state_hash(data[name].state_dict()) for name in states}:
            raise ValueError('Frozen model weights changed')
        report=dict(schema_version='driveguard-eye-recovery-comparison-v1',status='complete',task='fatigue',split='val',
            training_runs=0,plan=plan,reference_reproduced=True,frozen_weights_unchanged=True,
            comparisons=comparisons,frame_metrics=frames,input_support=support,
            notes=['All original validation windows retained, including unavailable positives and unknown labels.',
                   'Frame metric cohorts vary with localization coverage; event denominators stay fixed.',
                   'Sampled8 must exactly reproduce original fixed-eye inference before comparing dense recovery.',
                   'Tracked ROIs use current-frame pixels; bounded flow does not certify visibility or closure.',
                   'No label-dependent localization, threshold search, new training, or other-task routing change.',
                   'Validation-only exploratory replay of a validation-selected checkpoint; no automatic promotion.'])
        if (output/'comparison.json').exists() and read_json(output/'comparison.json')!=report:
            raise ValueError('Completed recovery report differs')
        write_rows(output/'comparison.csv',[dict(variant=r['variant'],readout=r['readout'],**r['overall']) for r in comparisons])
        write_rows(output/'by-subject.csv',[dict(variant=r['variant'],subject_id=s,**v) for r in comparisons for s,v in r['by_subject'].items()])
        write_rows(output/'by-subtype.csv',[dict(variant=r['variant'],subject_id=s,subtype=t,**v) for r in comparisons
            for s,types in dict(all=r['by_subtype'],**r['by_subject_subtype']).items() for t,v in types.items()])
        atomic_json(output/'frame-metrics.json',frames);atomic_json(output/'comparison.json',report)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('eye-report','dataset-root','recovery-root','output-dir'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--variant',default=VARIANT)
    parser.add_argument('--backend',choices=['cpu','cuda','npu'],default='cpu')
    parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--continue-run',action='store_true')
    args=vars(parser.parse_args(argv))
    try:
        report=run(**args);print(json.dumps(dict(status=report['status'],training_runs=0,output_dir=args['output_dir'])));return 0
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        print(f'error: {error}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
