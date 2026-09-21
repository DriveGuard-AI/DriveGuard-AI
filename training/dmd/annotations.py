"""Compact OpenLABEL summaries. Frame intervals are inclusive, not milliseconds."""
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path

from training.common.config import read_json


# Observed in the four supplied samples; NOT a closed DMD taxonomy or label mapping.
REVIEWED_TYPES = {
    f'{axis}/{value}' for axis, values in {
        'driver_actions': ('change_gear', 'drinking', 'radio', 'reach_side', 'safe_drive',
                           'talking_to_passenger', 'unclassified'),
        'gaze_on_road': ('looking_road', 'not_looking_road'),
        'hand_on_gear': ('hand_on_gear',),
        'hands_using_wheel': ('both', 'none', 'only_left', 'only_right'),
        'talking': ('talking',),
        'blinks': ('blinking',),
        'eyes_state': ('close', 'closing', 'open', 'opening'),
        'yawning': ('Yawning with hand', 'Yawning without hand'),
        'gaze_zone': ('center_mirror', 'front', 'front_right', 'infotainment', 'left',
                      'left_mirror', 'not_valid', 'right', 'right_mirror', 'steering_wheel'),
        'hands_on_wheel': ('both_hands', 'none', 'only_left', 'only_right'),
        'moving_hands': ('moving', 'not_moving'),
        'transition': ('giving control', 'taking control'),
    }.items() for value in values
}


def merge_intervals(intervals):
    """Inclusive interval union, without allocating a value for every frame."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _intervals(raw):
    if not isinstance(raw, list):
        raise ValueError('Expected a list of frame intervals')
    result = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError('Frame interval must be an object')
        start, end = item.get('frame_start'), item.get('frame_end')
        if (type(start) is not int or type(end) is not int
                or not 0 <= start <= end < 2**63):
            raise ValueError('Expected integer 0 <= frame_start <= frame_end < 2**63')
        result.append((start, end))
    return result


def _contains(intervals, frame):
    index = bisect_right(intervals, (frame, float('inf'))) - 1
    return index >= 0 and frame <= intervals[index][1]


def _count(intervals):
    return sum(end - start + 1 for start, end in intervals)


def _check_json_values(document):
    # json rejects NaN literals via read_json, but 1e400 can still become infinity.
    pending = [document]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Nonfinite numeric value in annotation')
        elif isinstance(value, str):
            try:
                value.encode('utf-8')
            except UnicodeEncodeError as exc:
                raise ValueError('Unpaired Unicode surrogate in annotation') from exc


def parse_annotation(path):
    """Read one annotation. Bad JSON/root raises; inner issues stay in the report."""
    path = Path(path)
    before = path.stat()
    doc = read_json(path)
    _check_json_values(doc)
    root = doc.get('openlabel') if isinstance(doc, dict) else None
    if not isinstance(root, dict):
        raise ValueError('Expected an openlabel object')
    metadata = root.get('metadata')
    if not isinstance(metadata, dict) or metadata.get('schema_version') != '1.0.0':
        raise ValueError('Expected OpenLABEL schema_version 1.0.0')
    if not isinstance(metadata.get('name'), str) or not metadata['name'].strip():
        raise ValueError('metadata.name must be nonempty text')
    if not isinstance(root.get('actions'), dict):
        raise ValueError('Expected an actions object')
    timeline = merge_intervals(_intervals(root.get('frame_intervals')))
    if not timeline:
        raise ValueError('Annotation timeline must not be empty')
    issues = []

    def issue(code, message, severity='warning', **details):
        issues.append(dict(code=code, severity=severity, message=message, **details))

    action_intervals, by_type = {}, defaultdict(list)
    raw_counts = Counter()
    for action_id, action in root['actions'].items():
        if (not isinstance(action, dict) or not isinstance(action.get('type'), str)
                or not action['type'].strip()):
            issue('INVALID_ACTION', 'Action requires a nonempty type', 'error', action_id=action_id)
            continue
        try:
            spans = _intervals(action.get('frame_intervals'))
        except ValueError as exc:
            issue('INVALID_INTERVALS', str(exc), 'error', action_id=action_id)
            continue
        merged = merge_intervals(spans)
        action_intervals[action_id] = merged
        by_type[action['type']].extend(spans)
        raw_counts[action['type']] += len(spans)
        if action['type'] not in REVIEWED_TYPES:
            issue('UNREVIEWED_LABEL_TYPE', 'Type not seen in reviewed samples; retained without mapping',
                  action_id=action_id, action_type=action['type'])
        if merge_intervals(timeline + merged) != timeline:
            issue('OUTSIDE_TIMELINE', 'Action interval extends outside declared timeline',
                  action_id=action_id)

    frames = root.get('frames')
    observed, frame_indices, problems = defaultdict(list), [], Counter()
    if frames is None:
        issue('FRAMES_NOT_PROVIDED', 'No per-frame references to compare with intervals')
        frames = {}
    elif not isinstance(frames, dict):
        issue('INVALID_FRAMES', 'frames must be an object', 'error')
        frames = {}
    for key, frame in frames.items():
        if not key.isascii() or not key.isdecimal() or len(key) > 19:
            problems['INVALID_FRAME_INDEX'] += 1
            continue
        index = int(key)
        if str(index) != key or index >= 2**63:
            problems['INVALID_FRAME_INDEX'] += 1
            continue
        frame_indices.append((index, index))
        if not _contains(timeline, index):
            problems['FRAME_OUTSIDE_TIMELINE'] += 1
        if not isinstance(frame, dict) or not isinstance(frame.get('actions', {}), dict):
            problems['INVALID_FRAME_ACTIONS'] += 1
            continue
        for action_id in frame.get('actions', {}):
            if action_id not in root['actions']:
                problems['UNDEFINED_ACTION'] += 1
            else:
                observed[action_id].append((index, index))
    for code, count in sorted(problems.items()):
        issue(code, 'Per-frame reference diagnostic',
              'warning' if code == 'FRAME_OUTSIDE_TIMELINE' else 'error', count=count)
    if 'frames' in root:
        if merge_intervals(frame_indices) != timeline:
            issue('FRAME_TIMELINE_MISMATCH', 'Per-frame keys do not exactly cover declared timeline')
        for action_id, intervals in action_intervals.items():
            referenced = merge_intervals(observed[action_id])
            if referenced != intervals:
                issue('FRAME_ACTION_MISMATCH', 'Action intervals differ from per-frame references',
                      action_id=action_id, interval_frames=_count(intervals),
                      referenced_frames=_count(referenced))

    streams = []
    raw_streams = root.get('streams', {})
    if not isinstance(raw_streams, dict):
        issue('INVALID_STREAMS', 'streams must be an object', 'error')
        raw_streams = {}
    for name, stream in raw_streams.items():
        if not isinstance(stream, dict) or not isinstance(stream.get('stream_properties'), dict):
            issue('INVALID_STREAM', 'Stream requires stream_properties', 'error', stream=name)
            continue
        properties = stream['stream_properties']
        sync = properties.get('sync')
        shift = sync.get('frame_shift') if isinstance(sync, dict) else None
        count, uri = properties.get('total_frames'), stream.get('uri')
        if (type(count) is not int or not 0 < count < 2**63 or type(shift) is not int
                or not -(2**63) < shift < 2**63 or not isinstance(uri, str) or not uri):
            issue('INVALID_STREAM', 'Expected uri, positive total_frames and integer frame_shift',
                  'error', stream=name)
            continue
        streams.append(dict(name=name, uri=uri, total_frames=count, frame_shift=shift))
        if shift == 0 and count != timeline[-1][1] + 1:
            issue('FRAME_COUNT_DECLARATION',
                  'Zero-shift stream declared frame count differs from annotation extent; decode required',
                  stream=name, total_frames=count, annotation_end=timeline[-1][1])
    if not streams:
        issue('NO_STREAMS', 'No usable camera synchronization declarations')

    drivers = []
    objects = root.get('objects', {})
    if not isinstance(objects, dict):
        issue('INVALID_OBJECTS', 'objects must be an object', 'error')
    else:
        for obj in objects.values():
            if isinstance(obj, dict) and obj.get('type') == 'driver':
                name = obj.get('name')
                if isinstance(name, str) and name.isascii() and name.isdecimal():
                    drivers.append(name)
                else:
                    issue('INVALID_DRIVER_ID', 'Driver name must be a numeric subject ID', 'error')

    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Annotation changed while being read')
    return dict(schema_version='1.0.0', name=metadata.get('name'), sha256=digest.hexdigest(),
                size_bytes=after.st_size, annotation_frames=_count(timeline),
                frame_intervals=[dict(frame_start=s, frame_end=e) for s, e in timeline],
                frame_entries=len(frames), driver_ids=sorted(set(drivers)),
                labels=[dict(type=label, interval_count=raw_counts[label],
                             covered_frames=_count(merge_intervals(spans)))
                        for label, spans in sorted(by_type.items())],
                streams=streams, issues=issues, video_decoded=False)
