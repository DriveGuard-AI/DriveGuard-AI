"""Four controlled body-distraction experiments with restartable sequential jobs."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from training.common.config import canonical_hash, read_json
from training.gaze.config import load_config, validate_config
from training.gaze.body_dist_report import compare_reports, write_csv

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'configs/training/relation_gaze_head_coverage25.json'
ARMS=('baseline','sampling','face_dropout','gap80')


def build_variants(base):
    validate_config(base)
    if (base['model']['kind'] != 'relation_gaze_head_policy_v1' or
            base['model']['head_policy'] != dict(min_valid_ratio=.25,max_gap_steps=0)):
        raise ValueError('Suite requires the coverage25, zero-gap head-policy baseline')
    neutral=dict(missing_head_positive_multiplier=1.,face_dropout_probability=0.)
    if base['training'].get('body_dist_experiment',neutral) != neutral:
        raise ValueError('Suite baseline must not already enable an intervention')
    result={}
    for name in ARMS:
        config=deepcopy(base)
        config['experiment']='body-dist-'+name
        config['training']['body_dist_experiment']=dict(neutral)
        if name == 'sampling': config['training']['body_dist_experiment']['missing_head_positive_multiplier']=3.
        if name == 'face_dropout': config['training']['body_dist_experiment']['face_dropout_probability']=.3
        if name == 'gap80': config['model']['head_policy']['max_gap_steps']=2
        result[name]=validate_config(config)
    return result


def _hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def _json(path,value):
    path=Path(path)
    fd,temporary=tempfile.mkstemp(dir=path.parent,prefix=path.name,suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _checkpoint(path):
    if read_json(str(path)+'.sha256.json')['sha256'] != _hash(path):
        raise ValueError(f'Checkpoint checksum mismatch: {path}')


def _fresh(parent,stem):
    number=1
    while (parent/f'{stem}-{number:03d}').exists(): number+=1
    return parent/f'{stem}-{number:03d}'


def _worker(arguments,log):
    print(f'Running {" ".join(arguments[:3])}; log: {log}',flush=True)
    environment=os.environ.copy(); environment['PYTHONUNBUFFERED']='1'
    # Prevent small CPU test jobs from oversubscribing; preserve explicit user settings.
    environment.setdefault('OMP_NUM_THREADS','1')
    environment.setdefault('MKL_NUM_THREADS','1')
    with Path(log).open('a',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,*arguments],cwd=ROOT,env=environment,
                              stdout=stream,stderr=subprocess.STDOUT,check=False)
    if result.returncode: raise RuntimeError(f'Job failed ({result.returncode}); inspect {log}; rerun suite with --continue')


def _training_config_matches(directory,expected):
    actual=read_json(directory/'config.json'); wanted=deepcopy(expected)
    wanted['runtime']['output_dir']=str(directory)
    if actual != wanted: raise ValueError(f'Training configuration mismatch in {directory}')


def run_suite(base,output_dir,*,variants=None,continue_run=False,prepare_only=False):
    selected=list(ARMS if variants is None else variants)
    if not selected or len(set(selected)) != len(selected) or any(name not in ARMS for name in selected):
        raise ValueError('Choose distinct supported variants')
    configs=build_variants(base); output=Path(output_dir).resolve()
    runtime=base['runtime']
    if not runtime['manifest'] or not runtime['data_root']:
        raise ValueError('Suite requires manifest and data_root')
    manifest=Path(runtime['manifest']).resolve(); data_root=Path(runtime['data_root']).resolve()
    if output == data_root or output.is_relative_to(data_root):
        raise ValueError('Suite output must be outside immutable feature data_root')
    configs={name:configs[name] for name in selected}
    for name,config in configs.items():
        config['runtime'].update(manifest=str(manifest),data_root=str(data_root),output_dir=str(output/name/'train-001'))
    plan=dict(schema_version='driveguard-body-dist-suite-v1',manifest_sha256=_hash(manifest),variants=configs)
    if output.exists():
        if not continue_run: raise FileExistsError('Use a NEW output_dir, or --continue for the identical suite')
        if read_json(output/'suite.json') != plan: raise ValueError('Suite configuration/identity mismatch')
    else:
        output.mkdir(parents=True,exist_ok=False); _json(output/'suite.json',plan)
    for name,config in configs.items():
        directory=output/name; directory.mkdir(exist_ok=True)
        path=directory/'config.json'
        if path.exists():
            if read_json(path) != config: raise ValueError('Saved variant configuration mismatch')
        else: _json(path,config)
    state_path=output/'state.json'
    state=read_json(state_path) if state_path.exists() else dict(plan_hash=canonical_hash(plan),variants={})
    if state['plan_hash'] != canonical_hash(plan): raise ValueError('Suite state identity mismatch')
    _json(state_path,state)
    if prepare_only:
        return dict(status='prepared',suite=str(output/'suite.json'),variants=selected)

    entries=[]
    def publish(status,error=None):
        report=compare_reports(entries)
        report.update(status=status,planned_variants=selected)
        if error is not None: report['error']=str(error)
        _json(output/'comparison.json',report); write_csv(output/'comparison.csv',report)
        return report

    try:
        for name,config in configs.items():
            directory=output/name; phase=state['variants'].setdefault(name,{})
            train_dir=Path(phase['train_dir']) if phase.get('train_dir') else directory/'train-001'
            # State only refers to this arm's direct children; reject edited paths.
            if train_dir.parent != directory: raise ValueError('Invalid training attempt path')
            result_path=train_dir/'result.json'; resume=phase.get('resume_checkpoint')
            if not result_path.exists():
                if train_dir.exists():
                    if (train_dir/'config.json').exists(): _training_config_matches(train_dir,config)
                    last=train_dir/'last.pt'
                    if last.exists():
                        resume=str(last)
                    train_dir=_fresh(directory,'train')
                if resume:
                    source=Path(resume)
                    if source.parent.parent != directory: raise ValueError('Invalid resume source path')
                    _training_config_matches(source.parent,config)
                    _checkpoint(source)
                    from training.common.checkpoint import load_checkpoint
                    payload=load_checkpoint(source)
                    if payload['epoch'] >= config['training']['epochs']:
                        raise ValueError(f'Final epoch exists without result.json in {source.parent}; inspect interrupted finalization')
                phase.update(train_dir=str(train_dir),resume_checkpoint=resume); _json(state_path,state)
                module={'cpu':'cpu_train','cuda':'gpu_train','npu':'npu_train'}[runtime['backend']]
                command=['-m','training.gaze.'+module,'--config',str(directory/'config.json'),'--output-dir',str(train_dir)]
                if resume: command+=['--resume',resume]
                print(f'[{name}] training -> {train_dir}',flush=True)
                _worker(command,directory/'train.log')
                result_path=train_dir/'result.json'
            _training_config_matches(train_dir,config)
            result=read_json(result_path)
            if result['completed_epochs'] != config['training']['epochs']: raise ValueError('Incomplete training result')
            for checkpoint in ('best','last'):
                weight=train_dir/(checkpoint+'.pt'); _checkpoint(weight)
                key=checkpoint+'_diagnostics'
                diagnostic=Path(phase[key]) if key in phase else _fresh(directory,checkpoint+'-diagnostics')
                if diagnostic.parent != directory: raise ValueError('Invalid diagnostics attempt path')
                if not (diagnostic/'summary.json').exists():
                    if diagnostic.exists(): diagnostic=_fresh(directory,checkpoint+'-diagnostics')
                    phase[key]=str(diagnostic); _json(state_path,state)
                    print(f'[{name}] {checkpoint} diagnostics',flush=True)
                    _worker(['-m','training.gaze.diagnostics','--checkpoint',str(weight),
                             '--backend',runtime['backend'],'--device-index',str(runtime['device_index']),
                             '--view','all','--subset','all','--output-dir',str(diagnostic)],directory/(checkpoint+'-diagnostics.log'))
                summary=read_json(diagnostic/'summary.json')
                if summary['checkpoint_sha256'] != _hash(weight): raise ValueError('Diagnostic checkpoint identity mismatch')
                candidate=dict(variant=name,checkpoint=checkpoint,directory=str(diagnostic),train_dir=str(train_dir))
                # Validate before mutating the publishable prefix. A rejected
                # report must not make the failure report itself unpublishable.
                compare_reports(entries+[candidate])
                entries.append(candidate)
                publish('running')
        return publish('complete')
    except (Exception,KeyboardInterrupt) as exc:
        publish('failed',exc)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(DEFAULT))
    parser.add_argument('--manifest'); parser.add_argument('--data-root'); parser.add_argument('--output-dir')
    parser.add_argument('--backend',choices=('cpu','cuda','npu'),default='npu')
    parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--variants',nargs='+',choices=ARMS,default=list(ARMS))
    parser.add_argument('--check-config',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--continue',dest='continue_run',action='store_true')
    args=parser.parse_args(argv)
    try:
        config=load_config(args.config); config['training']['epochs']=args.epochs
        config['runtime'].update(backend=args.backend,device_index=args.device_index)
        for name in ('manifest','data_root'):
            if getattr(args,name) is not None: config['runtime'][name]=getattr(args,name)
        if args.check_config:
            print(json.dumps(dict(status='config_valid',device_verified=False,variants=build_variants(config)),allow_nan=False)); return 0
        if not args.output_dir: raise ValueError('Supply --output-dir')
        result=run_suite(config,args.output_dir,variants=args.variants,continue_run=args.continue_run,prepare_only=args.prepare_only)
        print(json.dumps(dict(status=result['status'],output_dir=str(Path(args.output_dir).resolve())),allow_nan=False)); return 0
    except (ValueError,RuntimeError,OSError,ImportError) as exc: parser.exit(2,f'error: {exc}\n')


if __name__ == '__main__': raise SystemExit(main())
