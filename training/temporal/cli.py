"""Shared GPU/NPU command-line interface; precheck uses only stdlib."""
import argparse
import json
from training.common.config import apply_platform, load_config, read_json, validate_config


def main(backend, argv=None, *, default_config=None):
    parser = argparse.ArgumentParser(description='DriveGuard feature TCN training')
    parser.add_argument('--config')
    parser.add_argument('--platform-config')
    parser.add_argument('--check-config', action='store_true')
    parser.add_argument('--manifest')
    parser.add_argument('--data-root')
    parser.add_argument('--output-dir')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--resume', help='Trusted last.pt from a completed epoch')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config or default_config)
        if args.platform_config:
            config = apply_platform(config, read_json(args.platform_config))
            if config['runtime']['backend'] != backend:
                raise ValueError('Platform config backend disagrees with requested entrypoint')
        config['runtime']['backend'] = backend
        for key in ('manifest', 'data_root', 'output_dir'):
            value = getattr(args, key)
            if value is not None:
                config['runtime'][key] = value
        if args.epochs is not None:
            config['training']['epochs'] = args.epochs
        validate_config(config)
        if args.check_config:
            print(json.dumps({'status': 'config_valid', 'device_verified': False,
                              'config': config}, ensure_ascii=False, allow_nan=False))
            return 0
        if not config['runtime']['manifest'] or not config['runtime']['output_dir']:
            raise ValueError('Training requires --manifest and --output-dir (or runtime config values)')
        from training.temporal.train import run_training
        result = run_training(config, resume=args.resume)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f'error: {exc}\n')
