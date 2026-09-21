"""Cross-check joint checkpoint metadata against its training provenance."""
from training.common.config import TASKS,canonical_hash
from training.gaze.config import validate_config


def validate_metadata(payload):
    config=validate_config(payload['config']); data=config['data']
    if payload.get('kind') == 'training':
        snapshot=payload.get('best_snapshot')
        if not isinstance(snapshot,dict) or not {'optimizer_state','rng_state','global_step'} <= snapshot.keys():
            raise ValueError('Incomplete joint checkpoint best_snapshot metadata')
    if payload.get('model_kind') != 'relation_gaze_v1' or payload.get('task_order') != list(TASKS) or payload.get('active_tasks') != list(TASKS):
        raise ValueError('Joint checkpoint task/model metadata mismatch')
    if payload.get('gaze_classes') != data['gaze_classes']:
        raise ValueError('Joint checkpoint gaze class metadata mismatch')
    schema={'version':data['feature_schema_version'],'names':data['feature_names'],'sample_rate_hz':data['sample_rate_hz']}
    if payload.get('feature_schema') != schema or payload['normalizer'].get('feature_names') != data['feature_names']:
        raise ValueError('Joint checkpoint feature metadata mismatch')
    if payload.get('group_profile') != data['group_profile'] or payload.get('group_profile_sha256') != canonical_hash(data['group_profile']):
        raise ValueError('Joint checkpoint group metadata mismatch')
    mapping=payload.get('gaze_mapping',{})
    if mapping.get('version') != data['gaze_label_schema_version'] or mapping.get('classes') != data['gaze_classes']:
        raise ValueError('Joint checkpoint gaze mapping mismatch')
    summary=payload['data_summary']
    for field in ('gaze_mapping','label_mapping','extractor_profile','group_profile'):
        if canonical_hash(payload.get(field)) != summary.get(field+'_sha256'):
            raise ValueError(f'Joint checkpoint {field} metadata hash mismatch')
