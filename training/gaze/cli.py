"""Thin platform entrypoints; prechecks deliberately use only the stdlib."""
import argparse
import json
from training.common.config import read_json,apply_platform
from training.gaze.config import load_config,validate_config


def main(backend,argv=None):
    parser=argparse.ArgumentParser(description='Joint four-behavior and nine-class gaze training')
    for name in ('config','platform-config','manifest','data-root','output-dir','resume'): parser.add_argument('--'+name)
    parser.add_argument('--epochs',type=int); parser.add_argument('--check-config',action='store_true')
    args=parser.parse_args(argv)
    try:
        config=load_config(args.config)
        if args.platform_config:
            config=apply_platform(config,read_json(args.platform_config))
            if config['runtime']['backend'] != backend: raise ValueError('Platform backend disagrees with entrypoint')
        config['runtime']['backend']=backend
        for key in ('manifest','data_root','output_dir'):
            if getattr(args,key) is not None: config['runtime'][key]=getattr(args,key)
        if args.epochs is not None: config['training']['epochs']=args.epochs
        validate_config(config)
        if args.check_config:
            print(json.dumps({'status':'config_valid','device_verified':False,'config':config},allow_nan=False)); return 0
        from training.gaze.train import run_training
        print(json.dumps(run_training(config,resume=args.resume),allow_nan=False)); return 0
    except (ValueError,RuntimeError,OSError,ImportError) as exc: parser.exit(2,f'error: {exc}\n')
