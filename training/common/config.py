"""Strict, dependency-free configuration; safe to import with python -S."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

TASKS = ('fatigue', 'phone_use', 'drinking', 'distraction')
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'configs/training/temporal_baseline.json'
KEYS = {
    'data': {'feature_schema_version', 'label_schema_version', 'feature_names',
             'sample_rate_hz', 'window_steps', 'stride_steps'},
    'model': {'input_channels', 'channels', 'dilations', 'kernel_size', 'dropout'},
    'training': {'active_tasks', 'epochs', 'batch_size', 'seed', 'learning_rate',
                 'weight_decay', 'gradient_clip_norm', 'precision', 'num_workers'},
    'runtime': {'backend', 'device_index', 'data_root', 'manifest', 'output_dir'},
}


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path):
    def reject_constant(value):
        raise ValueError(f'Nonfinite JSON value: {value}')
    with Path(path).open(encoding='utf-8') as handle:
        return json.load(handle, object_pairs_hook=_pairs, parse_constant=reject_constant)


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def _number(value, name, minimum=0, exclusive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if value < minimum or (exclusive and value == minimum):
        raise ValueError(f'{name} out of range')


def validate_config(config):
    top = {'schema_version', 'experiment', *KEYS}
    if not isinstance(config, dict) or set(config) != top:
        raise ValueError('Invalid configuration keys')
    if config['schema_version'] not in ('driveguard-training-v1', 'driveguard-training-v2', 'driveguard-training-v3'):
        raise ValueError('Unsupported configuration schema')
    gaze = config['schema_version'] == 'driveguard-training-v3'
    relation = config['schema_version'] in ('driveguard-training-v2','driveguard-training-v3')
    if not isinstance(config['experiment'], str) or not config['experiment'].strip():
        raise ValueError('experiment must be nonempty')
    for section, keys in KEYS.items():
        if relation and section == 'data':
            keys = keys | {'group_profile'}
        if relation and section == 'model':
            keys = keys | {'kind', 'projection_dim', 'gate_hidden_dim'}
        if gaze and section == 'data':
            keys = keys | {'gaze_label_schema_version','gaze_classes'}
        if gaze and section == 'training':
            keys = keys | {'gaze_enabled','task_weights'}
        if not isinstance(config[section], dict) or set(config[section]) != keys:
            raise ValueError(f'Invalid {section} keys')
    data, model, train, runtime = (config[k] for k in KEYS)
    for key in ('feature_schema_version', 'label_schema_version'):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f'{key} must be nonempty')
    if data['label_schema_version'] != 'driveguard-behavior-v1':
        raise ValueError('Unsupported label schema')
    names = data['feature_names']
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError('feature_names must be nonempty strings')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate feature names')
    for key in ('sample_rate_hz', 'window_steps', 'stride_steps'):
        _integer(data[key], key)
    if data['stride_steps'] > data['window_steps']:
        raise ValueError('stride_steps must not exceed window_steps')
    _integer(model['input_channels'], 'input_channels')
    if model['input_channels'] != 2 * len(names):
        raise ValueError('input_channels must equal twice the number of features')
    for key in ('channels', 'dilations'):
        if not isinstance(model[key], list) or not model[key]:
            raise ValueError(f'{key} must be a nonempty list')
        for value in model[key]:
            _integer(value, key)
    if len(model['channels']) != len(model['dilations']):
        raise ValueError('channels and dilations length mismatch')
    _integer(model['kernel_size'], 'kernel_size')
    _number(model['dropout'], 'dropout')
    if model['dropout'] >= 1:
        raise ValueError('dropout must be < 1')
    if relation:
        if gaze:
            from training.gaze.profile import validate_group_profile, GAZE_CLASSES
            if data['gaze_label_schema_version'] != 'driveguard-gaze-v1' or data['gaze_classes'] != list(GAZE_CLASSES):
                raise ValueError('Gaze schema/class order mismatch')
        else:
            from training.relation.profile import validate_group_profile
        validate_group_profile(data['group_profile'], names)
        allowed_kinds = ('relation_gaze_v1','relation_gaze_head_v1') if gaze else ('relation_gated_v1',)
        if model['kind'] not in allowed_kinds:
            raise ValueError('Unsupported v2 model kind')
        for key in ('projection_dim', 'gate_hidden_dim'):
            _integer(model[key], key)
        if (data['feature_schema_version'] != ('driveguard-interaction-v2' if gaze else 'driveguard-interaction-v1') or
                data['sample_rate_hz'] != 25 or data['window_steps'] != 100 or
                model['channels'] != [model['projection_dim']] * len(model['dilations'])):
            raise ValueError('Relation model requires interaction-v1/25Hz/100 slots and shared projection width')
    for key in ('epochs', 'batch_size'):
        _integer(train[key], key)
    _integer(train['seed'], 'seed', 0)
    if train['seed'] >= 2**32:
        raise ValueError('seed must be < 2**32')
    _integer(train['num_workers'], 'num_workers', 0)
    if train['num_workers'] != 0 or train['precision'] != 'fp32':
        raise ValueError('First implementation supports FP32 and num_workers=0 only')
    for key in ('learning_rate', 'gradient_clip_norm'):
        _number(train[key], key, exclusive=True)
    _number(train['weight_decay'], 'weight_decay')
    active = train['active_tasks']
    if not isinstance(active, list) or not active or any(t not in TASKS for t in active):
        raise ValueError('active_tasks must be a nonempty subset of the four tasks')
    if len(set(active)) != len(active):
        raise ValueError('Duplicate active tasks')
    if gaze:
        if active != list(TASKS) or train['gaze_enabled'] is not True:
            raise ValueError('Joint training requires all four active tasks and gaze_enabled=true')
        weights = train['task_weights']
        if not isinstance(weights,list) or len(weights) != 5:
            raise ValueError('task_weights requires four behavior weights and one gaze weight')
        for weight in weights:
            _number(weight,'task_weights',exclusive=True)
    if runtime['backend'] not in ('cpu', 'cuda', 'npu'):
        raise ValueError('Unsupported backend')
    _integer(runtime['device_index'], 'device_index', 0)
    for key in ('data_root', 'manifest', 'output_dir'):
        if runtime[key] is not None and (not isinstance(runtime[key], str) or not runtime[key].strip()):
            raise ValueError(f'{key} must be null or a nonempty path string')
    return config


def load_config(path=None):
    return validate_config(read_json(path or DEFAULT_CONFIG))


def apply_platform(config, platform):
    if not isinstance(platform, dict) or not set(platform) <= KEYS['runtime']:
        raise ValueError('Platform overlay may only change runtime fields')
    result = deepcopy(config)
    result['runtime'].update(platform)
    return validate_config(result)
