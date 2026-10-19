"""Explicit CPU-only synthetic engineering smoke; never accesses real datasets."""
import argparse
import json
from pathlib import Path


def run_smoke(output_dir):
    import numpy as np
    import torch
    from training.common.checkpoint import atomic_json
    from training.common.config import TASKS, load_config
    from training.temporal.train import run_training

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    data_root = output / 'synthetic_data'
    data_root.mkdir()
    config = load_config()
    features = config['data']['feature_names']
    rng = np.random.default_rng(17)
    records = []
    for i, split in enumerate(('train', 'train', 'val', 'val')):
        record = {'record_id': f'synthetic-{i}', 'subject_id': f'synthetic-subject-{i}',
                  'recording_id': f'synthetic-recording-{i}', 'view': 'body',
                  'subset': 'synthetic-only', 'split': split}
        values = rng.normal(size=(125, len(features))).astype(np.float32)
        masks = (rng.uniform(size=values.shape) > .05).astype(np.uint8)
        targets = np.full((125, 4), i % 2, dtype=np.float32)
        label_masks = np.ones((125, 4), dtype=np.uint8)
        # Some explicitly unknown labels; known endpoint supervision is retained.
        label_masks[:10, 2:] = 0
        targets[label_masks == 0] = 0
        for name, array in [('values', values), ('masks', masks), ('targets', targets), ('label_masks', label_masks)]:
            record[name] = f'record-{i}-{name}.npy'
            np.save(data_root / record[name], array, allow_pickle=False)
        records.append(record)
    manifest = {'schema_version': 'driveguard-feature-manifest-v1', 'dataset_version': 'synthetic-smoke-v1',
                'feature_schema': {'version': config['data']['feature_schema_version'], 'names': features,
                                   'sample_rate_hz': config['data']['sample_rate_hz']},
                'extractor_profile': {'name': 'synthetic-only', 'synthetic': True,
                                      'feature_definitions': {name: 'Synthetic normal random feature; not a physical measurement' for name in features}},
                'label_mapping': {'version': config['data']['label_schema_version'], 'tasks': list(TASKS),
                                  'rules': {'synthetic': 'Constant binary targets by record; first ten slots partially unknown'}},
                'records': records}
    manifest_path = data_root / 'manifest.json'
    atomic_json(manifest_path, manifest)
    config['training'].update(epochs=2, batch_size=2)
    config['runtime'].update(backend='cpu', data_root=str(data_root), manifest=str(manifest_path),
                             output_dir=str(output / 'initial'))
    torch.set_num_threads(1)
    initial = run_training(config)
    config['training']['epochs'] = 3
    config['runtime']['output_dir'] = str(output / 'resumed')
    resumed = run_training(config, resume=initial['last_checkpoint'])
    report = {'synthetic_only': True, 'purpose': 'Engineering test only; NOT DMD model accuracy or accelerator validation',
              'initial': initial, 'resumed': resumed, 'torch': str(torch.__version__)}
    atomic_json(output / 'smoke_report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, help='New directory, usually under ignored runs/')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_smoke(args.output_dir), ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
