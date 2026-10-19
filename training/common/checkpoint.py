"""CPU-only state dict checkpoints, integrity checks and epoch-level RNG state."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from training.common.data import file_sha256


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return deepcopy(value)


def atomic_json(path, value):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(path, payload):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(cpu_tree(payload), temporary)
        os.replace(temporary, path)
        atomic_json(path.with_suffix(path.suffix + '.sha256.json'), {'sha256': file_sha256(path)})
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path):
    path = Path(path)
    try:
        metadata = json.loads(path.with_suffix(path.suffix + '.sha256.json').read_text(encoding='utf-8'))
        expected = metadata['sha256']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError('Missing or invalid checkpoint checksum sidecar') from exc
    if file_sha256(path) != expected:
        raise ValueError('Checkpoint checksum mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format')
    required = {'kind', 'model_state', 'config', 'epoch', 'normalizer', 'data_summary'}
    if not required <= payload.keys() or payload['kind'] not in ('training', 'weights'):
        raise ValueError('Incomplete checkpoint metadata')
    return payload


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type in ('cuda', 'npu'):
        getattr(torch, device.type).manual_seed_all(seed)


def capture_rng(device, generator):
    state = np.random.get_state()
    result = {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
              'torch': torch.get_rng_state(), 'loader': generator.get_state(), 'backend': device.type}
    if device.type in ('cuda', 'npu'):
        result['accelerator'] = getattr(torch, device.type).get_rng_state(device)
    return cpu_tree(result)


def restore_rng(state, device, generator):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    generator.set_state(state['loader'])
    if device.type == state['backend'] and device.type in ('cuda', 'npu'):
        getattr(torch, device.type).set_rng_state(state['accelerator'], device)


def validate_resume(payload, config, data_summary, normalizer):
    if payload['kind'] != 'training':
        raise ValueError('Resume requires last.pt training state, not best weights')
    required = {'optimizer_state', 'global_step', 'rng_state', 'best_model_state', 'best_epoch',
                'best_val_loss', 'history'}
    if not required <= payload.keys():
        raise ValueError('Incomplete training checkpoint')
    previous, current = deepcopy(payload['config']), deepcopy(config)
    previous.pop('runtime')
    current.pop('runtime')
    old_epochs = previous['training'].pop('epochs')
    new_epochs = current['training'].pop('epochs')
    if previous != current:
        raise ValueError('Resume configuration mismatch (only runtime/total epochs may change)')
    if new_epochs < old_epochs or new_epochs <= payload['epoch']:
        raise ValueError('Resume total epochs must not shrink and must exceed completed epoch')
    if payload['data_summary'] != data_summary:
        raise ValueError('Resume data fingerprint/split/profile mismatch')
    if payload['normalizer'] != normalizer:
        raise ValueError('Resume normalizer mismatch')
