"""Strict joint configuration, importable without site packages."""
from pathlib import Path
from training.common.config import validate_config as validate_common, read_json, apply_platform

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / 'configs/training/relation_gaze.json'


def validate_config(config):
    validate_common(config)
    if config['schema_version'] != 'driveguard-training-v3':
        raise ValueError('Joint training requires training-v3')
    return config


def load_config(path=None):
    return validate_config(read_json(path or DEFAULT_CONFIG))
