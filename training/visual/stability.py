"""Cached-feature input ablation, repeated seeds and smaller fusion heads."""
import argparse
import json
from pathlib import Path
import statistics


ARMS=(('D_context_control','context',64),('C_fusion','fusion',64),('E_fusion_small','fusion',32))
METRICS={
    'tp':('overall','tp'),'fp':('overall','fp'),
    'recall':('overall','all_positive_detection_rate'),'precision':('overall','precision'),
    'f1':('overall','all_positive_f1'),'fpr':('overall','false_positive_rate'),
    'h_missing_tp':('baseline_h_ineligible','tp'),'h_missing_fp':('baseline_h_ineligible','fp'),
    'fpr03_tp':('validation_fpr03','tp'),'fpr03_fp':('validation_fpr03','fp'),
    'fpr03_recall':('validation_fpr03','all_positive_detection_rate'),
    'fpr03_fpr':('validation_fpr03','false_positive_rate'),
    'fpr03_h_missing_tp':('validation_fpr03_groups','baseline_h_ineligible','tp'),
    'fpr03_h_missing_fp':('validation_fpr03_groups','baseline_h_ineligible','fp')}


def _value(row,path):
    for key in path:row=row[key]
    return row


def _statistics(values):
    present=[v for v in values if v is not None]
    if not present:return dict(count=0,mean=None,std=None,min=None,max=None)
    return dict(count=len(present),mean=statistics.mean(present),std=statistics.pstdev(present),
                min=min(present),max=max(present))


def aggregates(comparisons,seeds):
    best={(r['arm'],r['seed']):r for r in comparisons if r.get('checkpoint_kind')=='best'}
    grouped={}
    for arm,_,_ in ARMS:
        rows=[best[(arm,seed)] for seed in seeds]
        grouped[arm]=dict(seeds=seeds,seed_count=len(seeds),
                         metrics={name:_statistics([_value(row,path) for row in rows]) for name,path in METRICS.items()})
    paired={}
    for label,positive,negative in (('C_minus_D','C_fusion','D_context_control'),
                                    ('E_minus_C','E_fusion_small','C_fusion')):
        differences=[]
        for seed in seeds:
            a,b=best[(positive,seed)],best[(negative,seed)]
            values={}
            for name,path in METRICS.items():
                av,bv=_value(a,path),_value(b,path)
                values[name]=av-bv if av is not None and bv is not None else None
            differences.append(dict(seed=seed,metrics=values))
        paired[label]=dict(seed_count=len(seeds),per_seed=differences,
                           metrics={name:_statistics([r['metrics'][name] for r in differences]) for name in METRICS})
    return grouped,paired


def _publish(output,comparisons,seeds,plan):
    from training.common.checkpoint import atomic_json
    from training.visual.report import write_rows
    grouped,paired=aggregates(comparisons,seeds)
    report=dict(schema_version='driveguard-body-visual-stability-v1',status='complete',split='val',
                training_runs=len(ARMS)*len(seeds),seeds=seeds,plan=plan,comparisons=comparisons,
                aggregate_best=grouped,paired_best=paired,notes=[
                    'All model variants use identical windows, labels and baseline H cohorts.',
                    'D and C differ only in mode: D replaces normalized RGB input with zeros before projection.',
                    'D retains the RGB availability mask to match the fusion supervision population.',
                    'E changes only fusion hidden width from 64 to 32; other settings match C.',
                    'All seeds are reported; aggregates use best validation BCE checkpoint from each run.',
                    'Standard deviation is population std over seeds, not a confidence interval over subjects.',
                    'Best selection and FPR thresholds use validation; these are not held-out test estimates.',
                    'FPR03 subgroup metrics apply the single global validation threshold without subgroup retuning.',
                    'A uses original joint validation loss selection; new heads use binary validation BCE.'])
    flat=[]
    for row in comparisons:
        flat.append(dict(variant=row['variant'],seed=row.get('seed'),checkpoint_kind=row['checkpoint_kind'],
                         checkpoint=row['checkpoint'],epoch=row['epoch'],
                         completed_epochs=row.get('completed_epochs'),stopped_early=row.get('stopped_early'),
                         **{name:_value(row,path) for name,path in METRICS.items()},
                         fpr03_threshold_logit=row['validation_fpr03']['threshold_logit']))
    write_rows(output/'comparison.csv',flat)
    aggregate_rows=[]
    for category,groups in (('arm',grouped),('paired_delta',paired)):
        for label,group in groups.items():
            for metric,value in group['metrics'].items():
                aggregate_rows.append(dict(category=category,name=label,metric=metric,**value))
    write_rows(output/'aggregate.csv',aggregate_rows)
    atomic_json(output/'comparison.json',report)
    atomic_json(output/'progress.json',dict(status='complete',completed_stages=report['training_runs'],total_stages=report['training_runs']))
    atomic_json(output/'result.json',dict(status='complete',training_runs=report['training_runs'],comparison=str(output/'comparison.json')))
    return report


def run_stability(*,prepared_dir,output_dir,backend='npu',device_index=0,epochs=20,batch_size=64,
                  seeds=None,patience=3,min_delta=.0001,continue_run=False):
    from training.common.checkpoint import atomic_json
    from training.common.config import read_json
    from training.common.data import file_sha256
    from training.visual.data import PreparedData
    from training.visual.report import evaluate,summarize,write_rows
    from training.visual.suite import DEFAULTS,_train
    from training.visual.train import _settings
    seeds=[42,43,44] if seeds is None else list(seeds)
    if not seeds or any(type(s)!=int or s<0 or s>=2**32 for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('Seeds must be unique nonnegative integers below 2**32')
    if type(patience)!=int or patience<1:raise ValueError('Stability experiments require positive early-stop patience')
    base=dict(DEFAULTS,backend=backend,device_index=device_index,epochs=epochs,batch_size=batch_size,
              early_stopping_patience=patience,early_stopping_min_delta=min_delta)
    # Validate options before creating output or opening expensive caches.
    base=_settings(base)
    prepared=PreparedData(prepared_dir);output=Path(output_dir).resolve()
    try:
        sources=(prepared.root,prepared.cache.root,Path(prepared.report['baseline_checkpoint']).parent,
                 Path(prepared.report['data_root']))
        if any(output.is_relative_to(p) or p.is_relative_to(output) for p in sources):
            raise ValueError('Stability output overlaps input artifacts')
        stages=[]
        for seed in seeds:
            for arm,mode,width in ARMS:
                settings=_settings(dict(base,seed=seed,hidden_dim=width))
                stages.append(dict(arm=arm,mode=mode,settings=settings,stage=f'{arm}-seed{seed}'))
        plan=dict(schema_version='driveguard-visual-stability-plan-v1',provenance=prepared.provenance,seeds=seeds,stages=stages)
        if output.exists():
            if not continue_run:raise FileExistsError('Use a NEW output_dir or --continue-run')
            if read_json(output/'plan.json')!=plan:raise ValueError('Stability configuration/identity mismatch')
        else:output.mkdir(parents=True);atomic_json(output/'plan.json',plan)
        train=prepared.dataset('train');val=prepared.dataset('val')
        rows,_=evaluate(val);reference=prepared.report
        baseline=summarize(rows,variant='A_gap80',arm='A_gap80',checkpoint_kind='reference',
                           checkpoint=reference['baseline_checkpoint'],checkpoint_sha256=reference['baseline_checkpoint_sha256'],
                           epoch=reference['baseline_epoch'])
        comparisons=[baseline];write_rows(output/'A_gap80-windows.csv',rows)
        for index,stage in enumerate(stages,1):
            print(f"[{index}/{len(stages)}] {stage['stage']} (max {epochs} epochs, patience {patience})",flush=True)
            settings=stage['settings']
            result=_train(output/stage['stage'],train,val,stage['mode'],settings,prepared.provenance)
            for kind in ('best','last'):
                path=result[kind+'_checkpoint']
                rows,payload=evaluate(val,path,backend,device_index,batch_size)
                label=stage['stage']+'-'+kind
                row=summarize(rows,variant=label,arm=stage['arm'],seed=settings['seed'],checkpoint_kind=kind,
                              checkpoint=path,checkpoint_sha256=file_sha256(path),epoch=payload['epoch'],settings=settings,
                              completed_epochs=result['completed_epochs'],stopped_early=result.get('stopped_early',False))
                if row['window_label_hash']!=baseline['window_label_hash']:raise ValueError('Stability cohort mismatch')
                comparisons.append(row);write_rows(output/(label+'-windows.csv'),rows)
                atomic_json(output/(label+'-summary.json'),row)
            atomic_json(output/'progress.json',dict(status='running',completed_stages=index,total_stages=len(stages),
                                                    latest_stage=stage['stage']))
        return _publish(output,comparisons,seeds,plan)
    finally:prepared.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--backend',choices=('cpu','cuda','npu'),default='npu')
    parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=20);parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--patience',type=int,default=3);parser.add_argument('--min-delta',type=float,default=.0001)
    parser.add_argument('--continue-run',action='store_true')
    options=vars(parser.parse_args(argv))
    try:
        report=run_stability(**options)
        print(json.dumps(dict(status=report['status'],training_runs=report['training_runs'],output_dir=options['output_dir'])))
    except (OSError,ValueError,RuntimeError) as exc:parser.exit(2,f'error: {exc}\n')


if __name__=='__main__':main()
