"""Stream canonical expert JSONL to unlabeled recording arrays; never read videos."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from inference.perception.profile import load_profile
from inference.perception.sequence import FeatureExtractor
from training.common.config import canonical_hash, read_json

CHUNK_ROWS = 4096
MAX_LINE_CHARS = 1_000_000


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'Nonfinite JSON constant: {value}')


def read_packets(path):
    """Bound JSONL line size as well as total buffering; a blank line is malformed."""
    with Path(path).open(encoding='utf-8') as handle:
        line_number = 0
        while True:
            line = handle.readline(MAX_LINE_CHARS + 1)
            if not line:
                return
            line_number += 1
            if len(line) > MAX_LINE_CHARS:
                raise ValueError(f'Line {line_number}: exceeds maximum observation size')
            try:
                value = json.loads(line, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
                if not isinstance(value, dict):
                    raise ValueError('Expected a packet object')
            except ValueError as error:
                raise ValueError(f'Line {line_number}: {error}') from error
            yield value


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _materialize(raw_path, output, slots, features, relation=False):
    arrays = {}
    try:
        layout = [('values', features, 'float32'), ('masks', features, 'uint8'),
                  ('targets', 4, 'float32'), ('label_masks', 4, 'uint8')]
        if relation:
            layout.append(('expert_age_ms',3,'float32'))
        row_width = features * 2 + (3 if relation else 0)
        for name, columns, dtype in layout:
            arrays[name] = np.lib.format.open_memmap(output / f'{name}.npy', mode='w+',
                                                     dtype=dtype, shape=(slots, columns))
        with raw_path.open('rb') as handle:
            for start in range(0, slots, CHUNK_ROWS):
                count = min(CHUNK_ROWS, slots - start)
                block = np.fromfile(handle, dtype=np.float32, count=count * row_width).reshape(count, row_width)
                arrays['values'][start:start+count] = block[:, :features]
                arrays['masks'][start:start+count] = block[:, features:features*2]
                if relation:
                    arrays['expert_age_ms'][start:start+count] = block[:,features*2:]
                # Labels are unknown, not negatives. Only a later audited mapping may fill them.
                arrays['targets'][start:start+count] = 0
                arrays['label_masks'][start:start+count] = 0
        for array in arrays.values():
            array.flush()
    finally:
        for array in arrays.values():
            array._mmap.close()


def export_recording(packets, output_dir, profile, *, provenance=None, group_profile=None):
    extractor = FeatureExtractor(profile)
    profile = extractor.profile
    if profile['feature_schema_version'] == 'driveguard-interaction-v2' and group_profile is None:
        raise ValueError('Gaze extraction requires its matching group profile')
    if group_profile is not None:
        if profile['feature_schema_version'] == 'driveguard-interaction-v2':
            from training.gaze.profile import validate_group_profile
        else:
            from training.relation.profile import validate_group_profile
        group_profile = validate_group_profile(group_profile,profile['feature_names'])
        if profile['sample_rate_hz'] != 25 or profile['max_age_ms'] != group_profile['max_age_ms']:
            raise ValueError('Extractor rate/age differs from group profile')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    marker = output / 'incomplete.json'
    _write_json(marker, {'status': 'incomplete', 'reason': 'No successful record.json published yet'})
    raw_path = None
    try:
        slots = 0
        first = last = None
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.tmp', dir=output, delete=False) as raw:
            raw_path = Path(raw.name)
            for packet in packets:
                result = extractor.step(packet)
                row = np.concatenate([result['values'], result['masks']]).astype(np.float32)
                if group_profile is not None:
                    maximum = group_profile['max_age_ms'] + 1
                    ages = [min(maximum,packet['timestamp_ms']-packet[name]['timestamp_ms'])
                            if packet[name]['status'] == 'ok' else maximum
                            for name in group_profile['age_columns']]
                    row = np.concatenate([row,np.asarray(ages,dtype=np.float32)])
                raw.write(row.tobytes())
                if first is None:
                    first = {key: packet[key] for key in ('recording_id', 'view', 'timestamp_ms', 'width', 'height', 'driver_roi')}
                    first['driver_roi'] = list(packet['driver_roi'])
                    first['expert_models'] = {key: packet[key]['model_id'] for key in ('face', 'objects', 'hands')}
                last = packet['timestamp_ms']
                slots += 1
        if slots < 100:
            raise ValueError('Recording needs at least 100 slots for the configured TCN window')
        _materialize(raw_path, output, slots, len(profile['feature_names']),group_profile is not None)
        paths = {name: f'{name}.npy' for name in ('values', 'masks', 'targets', 'label_masks')}
        if group_profile is not None:
            paths['expert_age_ms'] = 'expert_age_ms.npy'
        report = {'schema_version': 'driveguard-feature-record-v1', 'status': 'complete_unlabeled',
                  'slots': slots, 'recording_id': first['recording_id'], 'view': first['view'],
                  'first_timestamp_ms': first['timestamp_ms'], 'last_timestamp_ms': last,
                  'image_size': [first['width'], first['height']], 'driver_roi': first['driver_roi'],
                  'expert_models': first['expert_models'], 'feature_schema': {
                      'version': profile['feature_schema_version'], 'names': profile['feature_names'], 'sample_rate_hz': 25},
                  'extractor_profile': profile, 'profile_sha256': canonical_hash(profile),
                  'arrays': paths, 'array_sha256': {name: _sha256(output / path) for name, path in paths.items()},
                  'labels': 'All unknown: attach audited label mapping and subject/split manifest before training'}
        if group_profile is not None:
            report.update(schema_version='driveguard-feature-record-v2',group_profile=group_profile,
                          group_profile_sha256=canonical_hash(group_profile))
            if profile['feature_schema_version'] == 'driveguard-interaction-v2':
                report['schema_version'] = 'driveguard-feature-record-v3'
        if provenance is not None:
            # Capture decoder metadata populated during iteration, before successful publication.
            if not isinstance(provenance, dict):
                raise ValueError('provenance must be a JSON object')
            canonical_hash(provenance)
            report['provenance'] = json.loads(json.dumps(provenance, allow_nan=False))
        # Publish the success index only after all arrays are flushed and hashed.
        # The directory is exclusively owned by this invocation and was required to be new.
        index = output / 'record.json'
        pending = output / 'record.pending.json'
        _write_json(pending, report)
        raw_path.unlink()
        raw_path = None
        marker.unlink()
        pending.replace(index)  # Final fallible action: no successful index on cleanup failure.
        return report
    except Exception:
        # If final publication failed after marker removal, restore the hint.
        # Absence of record.json is authoritative even if the filesystem refuses this write.
        if not marker.exists():
            try:
                _write_json(marker, {'status': 'incomplete', 'reason': 'Export or index publication failed'})
            except OSError:
                pass
        raise
    finally:
        if raw_path is not None and raw_path.exists():
            raw_path.unlink()  # Only this invocation's temporary cache; never delete output_dir.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Canonical driveguard-experts-v1 JSONL, not video')
    parser.add_argument('--output-dir', required=True, help='New recording directory; never overwrite')
    parser.add_argument('--profile', help='Feature profile JSON; defaults to interaction_v1')
    parser.add_argument('--group-profile',help='Explicit v2 groups JSON; absent preserves v1 export')
    args = parser.parse_args(argv)
    try:
        report = export_recording(read_packets(args.input), args.output_dir, load_profile(args.profile),
                                  group_profile=read_json(args.group_profile) if args.group_profile else None)
    except (OSError, ValueError, TypeError) as error:
        print(f'Feature export failed: {error}', file=sys.stderr)
        return 2
    print(json.dumps({'status': report['status'], 'slots': report['slots'],
                      'profile_sha256': report['profile_sha256'], 'output_dir': str(Path(args.output_dir).resolve())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
