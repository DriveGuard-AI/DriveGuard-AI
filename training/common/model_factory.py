"""Explicit model construction and integrity-checked feature-model reload."""
from copy import deepcopy

from training.common.config import TASKS, canonical_hash, validate_config


def model_kind(config):
    return config['model'].get('kind', 'temporal_tcn_v1')


def build_model(config):
    validate_config(config)
    settings = deepcopy(config['model'])
    settings['window_steps'] = config['data']['window_steps']
    if config['schema_version'] == 'driveguard-training-v3':
        from training.gaze.model import RelationGazeTCN
        settings.pop('kind')
        return RelationGazeTCN(**settings)
    if config['schema_version'] == 'driveguard-training-v2':
        from training.relation.model import RelationGatedTCN
        from training.relation.profile import GROUP_INDICES
        settings.pop('kind')
        return RelationGatedTCN(**settings,group_indices=GROUP_INDICES)
    from training.temporal.model import TemporalTCN
    return TemporalTCN(**settings)


def load_model(path, backend='cpu', device_index=0):
    """Return eval model and metadata; not a production risk/availability adapter.

    Only load trusted checkpoints with a valid checksum sidecar. Normalizer,
    active_tasks and feature/group profiles must be used by the caller; raw logits
    alone do not authorize a business task or a missing-evidence combination.
    """
    from training.common.checkpoint import load_checkpoint
    from training.common.device import select_device
    payload = load_checkpoint(path)
    config = validate_config(payload['config'])
    if payload.get('task_order') != list(TASKS) or payload.get('active_tasks') != config['training']['active_tasks']:
        raise ValueError('Checkpoint task metadata mismatch')
    data = config['data']
    if payload['normalizer'].get('feature_names') != data['feature_names']:
        raise ValueError('Checkpoint normalizer feature mismatch')
    if config['schema_version'] in ('driveguard-training-v2','driveguard-training-v3'):
        profile = data['group_profile']
        if (payload.get('model_kind') != model_kind(config) or payload.get('group_profile') != profile or
                payload.get('group_profile_sha256') != canonical_hash(profile)):
            raise ValueError('Checkpoint model/group profile mismatch')
    if config['schema_version'] == 'driveguard-training-v3':
        from training.gaze.checkpoint import validate_metadata
        validate_metadata(payload)
    device = select_device(backend,device_index)
    model = build_model(config).to(device)
    model.load_state_dict(payload['model_state'],strict=True)
    return model.eval(),payload
