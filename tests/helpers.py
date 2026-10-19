"""Small, explicitly synthetic fixtures; never read DMD or download anything."""
import json
from pathlib import Path
import numpy as np

from training.common.config import TASKS


def write_fixture(root, config, frames=125):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    features = config['data']['feature_names']
    records = []
    for i, split in enumerate(('train', 'train', 'val', 'val')):
        record_id = f'r{i}'
        values = rng.normal(size=(frames, len(features))).astype('float32')
        masks = np.ones_like(values, dtype='uint8')
        targets = np.full((frames, 4), i % 2, dtype='float32')
        label_masks = np.ones((frames, 4), dtype='uint8')
        paths = {}
        for name, array in [('values', values), ('masks', masks),
                            ('targets', targets), ('label_masks', label_masks)]:
            paths[name] = f'{record_id}_{name}.npy'
            np.save(root / paths[name], array, allow_pickle=False)
        records.append({'record_id': record_id, 'subject_id': f's{i}',
                        'recording_id': f'video{i}', 'view': 'body', 'subset': 'synthetic',
                        'split': split, **paths})
    manifest = {
        'schema_version': 'driveguard-feature-manifest-v1',
        'dataset_version': 'synthetic-fixture-v1',
        'feature_schema': {'version': config['data']['feature_schema_version'],
                           'names': features, 'sample_rate_hz': config['data']['sample_rate_hz']},
        'extractor_profile': {'name': 'synthetic-fixture', 'synthetic': True,
                              'feature_definitions': {name: 'synthetic normal value' for name in features}},
        'label_mapping': {'version': config['data']['label_schema_version'],
                          'tasks': list(TASKS), 'rules': {'synthetic': 'constant binary label per recording'}},
        'records': records,
    }
    path = root / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return path, manifest


def write_manifest(path, manifest):
    Path(path).write_text(json.dumps(manifest), encoding='utf-8')
