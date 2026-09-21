"""Explicit full-data joint training preparation; old preparation stays immutable."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from training.common.config import canonical_hash, read_json
from training.dmd.batch_config import _relative, validate_preparation
from training.dmd.prepare import ROOT, DEFAULT_LABELS
from inference.perception.runtime import load_runtime_config


def _jobs(prep):
    return [deepcopy(job) for job in prep['jobs'] if job['subset'] != 'gaze' or job['view'] == 'face']


def init_config(preparation_path,output_path,*,roi_from=()):
    source,output=Path(preparation_path).resolve(strict=True),Path(output_path).resolve()
    prep=validate_preparation(read_json(source))
    settings={job['job_id']:dict(video=job['video'],normalized_roi=[0,0,1,1],confirmed_exclusive=False)
              for job in _jobs(prep)}
    for previous_path in roi_from:
        previous=read_json(previous_path)
        if previous.get('schema_version') == 'driveguard-dmd-batch-v2':
            loaded=load_batch(previous_path)
        else:
            from training.dmd.batch_config import load_batch as legacy_load
            loaded=legacy_load(previous_path)
        if canonical_hash(loaded['preparation']) != canonical_hash(prep):
            raise ValueError('ROI source refers to a different preparation')
        for jid,setting in loaded['config']['jobs'].items():
            if jid in settings and setting['confirmed_exclusive']:
                if setting['video'] != settings[jid]['video']:
                    raise ValueError('ROI source video differs')
                if settings[jid]['confirmed_exclusive'] and settings[jid] != setting:
                    raise ValueError('Conflicting confirmed ROI sources')
                settings[jid]=deepcopy(setting)
    threshold=prep.get('label_policy',{}).get('closed_eye_min_ms')
    if threshold is None:
        threshold=read_json(DEFAULT_LABELS)['closed_eye_min_ms']
    if type(threshold) is not int or threshold <= 0:
        raise ValueError('Invalid closed-eye threshold')
    config=dict(schema_version='driveguard-dmd-batch-v2',preparation=_relative(source,output.parent),
        preparation_sha256=canonical_hash(prep),runtime_config=_relative(ROOT/'configs/perception/dmd_gaze_cpu.json',output.parent),
        closed_eye_min_ms=threshold,jobs=settings)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as handle:
        json.dump(config,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n')
    return config


def load_batch(config_path):
    # Keep legacy validation rules for ROI/path spelling without allowing legacy
    # supervision scope to silently change. Original preparation is hash-checked.
    import math
    path=Path(config_path).resolve(strict=True)
    config=read_json(path)
    keys={'schema_version','preparation','preparation_sha256','runtime_config','closed_eye_min_ms','jobs'}
    if not isinstance(config,dict) or set(config)!=keys or config['schema_version']!='driveguard-dmd-batch-v2':
        raise ValueError('Joint extraction requires batch-v2 config')
    prep=validate_preparation(read_json(path.parent/config['preparation']))
    if canonical_hash(prep)!=config['preparation_sha256']:
        raise ValueError('Preparation changed since joint configuration')
    if type(config['closed_eye_min_ms']) is not int or config['closed_eye_min_ms']<=0:
        raise ValueError('Invalid closed-eye threshold')
    known={j['job_id']:j for j in _jobs(prep)}
    if not isinstance(config['jobs'],dict) or not config['jobs']:
        raise ValueError('Joint extraction needs selected jobs')
    selected=[]
    for jid,setting in config['jobs'].items():
        if jid not in known or not isinstance(setting,dict) or set(setting)!={'video','normalized_roi','confirmed_exclusive'}:
            raise ValueError('Unknown joint job or ROI fields')
        if setting['video']!=known[jid]['video'] or type(setting['confirmed_exclusive']) is not bool:
            raise ValueError('Joint video/ROI identity mismatch')
        roi=setting['normalized_roi']
        if (not isinstance(roi,list) or len(roi)!=4 or any(type(v) not in (int,float) or not math.isfinite(v)
            or not 0<=v<=1 for v in roi) or roi[0]>=roi[2] or roi[1]>=roi[3]):
            raise ValueError('Invalid normalized ROI')
        job=known[jid]
        if job['subset']=='gaze':
            job['supervised_tasks']=['gaze_zone']
        selected.append(job)
    runtime=load_runtime_config(path.parent/config['runtime_config'])
    if runtime['schema_version']!='driveguard-runtime-v2':
        raise ValueError('Joint extraction requires gaze runtime-v2')
    policy=deepcopy(read_json(DEFAULT_LABELS))
    policy['closed_eye_min_ms']=config['closed_eye_min_ms']
    policy['gaze_rules']={'scope':'gaze subset face only','unknown':'not_valid, unknown type, gap, conflict => mask0',
                          'alignment':'same actual-PTS causal endpoint policy as behavior',
                          'risk':'zone classification is not a danger label'}
    return dict(config=config,preparation=prep,selected=selected,runtime=runtime,label_policy=policy,gaze=True)


def run_batch(config_path,dataset_root,output_dir,progress=None):
    from training.dmd.batch import run_batch as run
    return run(config_path,dataset_root,output_dir,progress,gaze=True)


def publish_batch(config_path,dataset_root,output_dir,training_config):
    from training.dmd.batch import publish_batch as publish
    return publish(config_path,dataset_root,output_dir,training_config,gaze=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    init=commands.add_parser('init',help='Full Dist/Drows face+body and Gaze face; no ML imports')
    init.add_argument('--preparation',required=True)
    init.add_argument('--output',required=True)
    init.add_argument('--roi-from',action='append',default=[],help='Copy exact reviewed video ROI; repeatable')
    for name in ('run','publish'):
        sub=commands.add_parser(name)
        sub.add_argument('--config',required=True)
        sub.add_argument('--dataset-root',required=True)
        sub.add_argument('--output-dir',required=True)
        if name=='publish':
            sub.add_argument('--training-config',default=str(ROOT/'configs/training/relation_gaze.json'))
    args=parser.parse_args(argv)
    try:
        if args.command=='init':
            config=init_config(args.preparation,args.output,roi_from=args.roi_from)
            result=dict(status='needs_roi_review',jobs=len(config['jobs']),
                        confirmed=sum(s['confirmed_exclusive'] for s in config['jobs'].values()),config=args.output)
        elif args.command=='run':
            result=run_batch(args.config,args.dataset_root,args.output_dir,
                             progress=lambda s:print(s,file=sys.stderr,flush=True))
        else:
            result=publish_batch(args.config,args.dataset_root,args.output_dir,args.training_config)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return 2 if result.get('failed',0) else 0
    except (OSError,ValueError,TypeError,KeyError,ImportError,RuntimeError) as error:
        print(f'Joint DMD batch failed: {error}',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
