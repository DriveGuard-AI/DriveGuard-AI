"""Train RGB-only and RGB+frozen-context probes against the frozen gap80 reference."""
import argparse
import json
from pathlib import Path

from training.common.checkpoint import atomic_json,load_checkpoint,save_checkpoint
from training.common.config import read_json
from training.common.data import file_sha256
from training.visual.data import PreparedData
from training.visual.train import train_probe,_settings
from training.visual.report import evaluate,summarize,write_rows,write_comparison


DEFAULTS=dict(epochs=20,batch_size=64,seed=42,learning_rate=.001,weight_decay=.0001,
              gradient_clip_norm=1.,hidden_dim=64,dropout=.1,backend='cpu',device_index=0)


def _verify(path,mode,settings,provenance,payload=None):
    payload=load_checkpoint(path) if payload is None else payload
    if (payload.get('model_kind')!='visual_probe_v1' or payload.get('settings')!=settings
        or payload.get('provenance')!=provenance or payload['model_params']['mode']!=mode):
        raise ValueError('Checkpoint configuration/provenance mismatch')
    return payload


def _train(stage,train,val,mode,settings,provenance):
    stage.mkdir(exist_ok=True)
    attempts=sorted(stage.glob('train-*'))
    resume=None;latest=None
    for attempt in attempts:
        path=attempt/'last.pt'
        if path.exists():
            try:payload=load_checkpoint(path)
            except (OSError,ValueError,EOFError):
                if (attempt/'result.json').exists():raise
                # Unpublished checksum/partial write: retain this attempt and use an older verified epoch.
                payload=None
            if payload is not None:
                _verify(path,mode,settings,provenance,payload)
                if latest is None or payload['epoch']>=latest['epoch']:resume=path;latest=payload
        result_path=attempt/'result.json'
        if result_path.exists():
            result=read_json(result_path)
            for key in ('best_checkpoint','last_checkpoint'):
                expected=attempt/('best.pt' if key=='best_checkpoint' else 'last.pt')
                if Path(result[key]).resolve()!=expected.resolve():raise ValueError('Stage checkpoint identity mismatch')
                _verify(expected,mode,settings,provenance)
            if result['completed_epochs']!=settings['epochs']:raise ValueError('Incomplete stage result')
            return result
    if latest is not None and latest['epoch']==settings['epochs']:
        # A crash after final last.pt publication can be finalized from its saved best state.
        attempt=resume.parent
        best=dict(latest,kind='weights',epoch=latest['best_epoch'],model_state=latest['best_model_state'])
        for key in ('optimizer_state','rng_state','history','best_model_state','global_step'):best.pop(key,None)
        save_checkpoint(attempt/'best.pt',best)
        atomic_json(attempt/'history.json',latest['history'])
        result=dict(completed_epochs=settings['epochs'],global_step=latest['global_step'],mode=mode,
                    best_epoch=latest['best_epoch'],best_val_loss=latest['best_val_loss'],
                    selection_metric='binary_validation_bce',best_checkpoint=str(attempt/'best.pt'),
                    last_checkpoint=str(resume),device=settings['backend'])
        atomic_json(attempt/'result.json',result);return result
    attempt=stage/f'train-{len(attempts)+1:03d}'
    return train_probe(train,val,mode,settings,attempt,provenance,resume=resume)


def run_suite(prepared_dir,output_dir,settings=None,continue_run=False):
    settings=_settings(dict(DEFAULTS if settings is None else settings))
    output=Path(output_dir).resolve();prepared=PreparedData(prepared_dir)
    try:
        for source in (prepared.root,prepared.cache.root,Path(prepared.report['baseline_checkpoint']).parent):
            if output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('Suite output overlaps input artifacts')
        plan=dict(schema_version='driveguard-visual-suite-v1',settings=settings,provenance=prepared.provenance)
        if output.exists():
            if not continue_run:raise FileExistsError('Use a NEW suite output_dir or --continue-run')
            if read_json(output/'plan.json')!=plan:raise ValueError('Suite configuration/identity mismatch')
        else:
            output.mkdir(parents=True);atomic_json(output/'plan.json',plan)
        train=prepared.dataset('train');val=prepared.dataset('val')
        comparisons=[]
        rows,_=evaluate(val)
        baseline=prepared.report
        comparisons.append(summarize(rows,variant='A_gap80',checkpoint=baseline['baseline_checkpoint'],
                                     checkpoint_sha256=baseline['baseline_checkpoint_sha256'],epoch=baseline['baseline_epoch']))
        write_rows(output/'A_gap80-windows.csv',rows)
        for mode,label in (('visual','B_rgb'),('fusion','C_rgb_context')):
            result=_train(output/label,train,val,mode,settings,prepared.provenance)
            for which in ('best','last'):
                checkpoint=result[which+'_checkpoint']
                rows,payload=evaluate(val,checkpoint,settings['backend'],settings['device_index'],settings['batch_size'])
                row=summarize(rows,variant=label+'_'+which,checkpoint=checkpoint,
                              checkpoint_sha256=file_sha256(checkpoint),epoch=payload['epoch'])
                if row['window_label_hash']!=comparisons[0]['window_label_hash']:raise ValueError('Comparison cohort mismatch')
                comparisons.append(row);write_rows(output/(label+'_'+which+'-windows.csv'),rows)
                atomic_json(output/(label+'_'+which+'-summary.json'),row)
        return write_comparison(output,comparisons)
    finally:prepared.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--continue-run',action='store_true')
    for key,value in DEFAULTS.items():parser.add_argument('--'+key.replace('_','-'),type=type(value),default=value)
    args=vars(parser.parse_args(argv));settings={k:args.pop(k) for k in DEFAULTS}
    try:
        report=run_suite(settings=settings,**args);print(json.dumps(dict(status=report['status'],comparisons=len(report['comparisons']))))
    except (OSError,ValueError,RuntimeError) as exc:parser.exit(2,f'error: {exc}\n')


if __name__=='__main__':main()
