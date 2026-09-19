"""Local artifact verification, not a downloader or a license approval service."""
import hashlib
from pathlib import Path
import re

from training.common.config import canonical_hash


def validate_asset_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {'path', 'sha256', 'source', 'license'}:
        raise ValueError('Asset requires exactly path, sha256, source, license')
    for key in ('path', 'source', 'license'):
        if not isinstance(spec[key], str) or not spec[key].strip():
            raise ValueError(f'Asset {key} must be nonempty')
    if '://' in spec['path']:
        raise ValueError('Asset path must be local; automatic downloads are disabled')
    if not isinstance(spec['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', spec['sha256']):
        raise ValueError('Asset sha256 must be 64 lowercase hexadecimal characters')
    return dict(spec)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_asset(spec):
    result = validate_asset_spec(spec)
    path = Path(result['path']).resolve(strict=True)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Asset must be a nonempty local file: {path}')
    if file_sha256(path) != result['sha256']:
        raise ValueError(f'Asset SHA-256 mismatch: {path}')
    result['path'] = str(path)
    return result


def model_identity(kind, asset, settings):
    validate_asset_spec(asset)
    return f'{kind}:{canonical_hash({"artifact": {k: asset[k] for k in ("sha256", "source", "license")}, "settings": settings})}'
