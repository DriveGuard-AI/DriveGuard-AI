"""Pure DMD event-proxy labels aligned causally to actual decoded frame PTS.

Fatigue is a yawning/sustained-closed-eye event proxy, not a diagnosis. Missing
evidence is unknown; an unknown target is always stored as zero with mask zero.
"""
from bisect import bisect_right
from collections import defaultdict

from training.common.config import TASKS
from training.dmd.annotations import _check_json_values, _contains, _intervals, merge_intervals


VIEWS = ('face', 'body', 'hands')
PHONE_TYPES = {'phonecall_left', 'phonecall_right', 'texting_left', 'texting_right'}
DRIVER_TYPES = PHONE_TYPES | {
    'drinking', 'safe_drive', 'change_gear', 'radio', 'reach_side', 'talking_to_passenger',
}
YAWN_TYPES = {'Yawning with hand', 'Yawning without hand'}
MAX_SOURCE_AGE_MS = 80
# Dense per-recording output: approximately 66.7 minutes at 25 Hz.
MAX_OUTPUT_SLOTS = 100_000


def _root(document):
    _check_json_values(document)
    root = document.get('openlabel') if isinstance(document, dict) else None
    if not isinstance(root, dict):
        raise ValueError('Expected an openlabel object')
    metadata = root.get('metadata')
    if (not isinstance(metadata, dict) or metadata.get('schema_version') != '1.0.0'
            or not isinstance(metadata.get('name'), str) or not metadata['name'].strip()):
        raise ValueError('Expected OpenLABEL schema_version 1.0.0 and nonempty metadata.name')
    return root


def _streams(root):
    raw = root.get('streams')
    if not isinstance(raw, dict):
        raise ValueError('Expected camera stream declarations')
    result = {}
    for view in VIEWS:
        stream = raw.get(view + '_camera')
        if not isinstance(stream, dict):
            raise ValueError(f'Missing {view}_camera stream')
        properties = stream.get('stream_properties')
        if not isinstance(properties, dict):
            raise ValueError(f'Missing {view} stream_properties')
        sync = properties.get('sync')
        shift = sync.get('frame_shift') if isinstance(sync, dict) else None
        count, uri = properties.get('total_frames'), stream.get('uri')
        if (type(shift) is not int or not -(2**63) < shift < 2**63
                or type(count) is not int or not 0 < count < 2**63
                or not isinstance(uri, str) or not uri.strip()):
            raise ValueError(f'Invalid {view} camera declaration')
        result[view] = {'shift': shift, 'count': count}
    if result['face']['shift'] != 0:
        raise ValueError('DMD face camera frame_shift must be zero')
    return result


def camera_offsets(document):
    """Return source-to-annotation offsets relative to the earliest camera.

    DMD DEx's vcd4reader.get_shifts and accessDMDAnn.checkFrameInStream use
    mosaic time starting at the earliest of face (zero), body, and hands.
    Annotation frame equals decoded source index plus this normalized offset.
    """
    streams = _streams(_root(document))
    earliest = min(item['shift'] for item in streams.values())
    return {view: item['shift'] - earliest for view, item in streams.items()}


def _annotations(root):
    timeline = merge_intervals(_intervals(root.get('frame_intervals')))
    if not timeline:
        raise ValueError('Annotation timeline must not be empty')
    actions = root.get('actions')
    if not isinstance(actions, dict):
        raise ValueError('Expected an actions object')
    by_id, by_type = {}, defaultdict(list)
    for action_id, action in actions.items():
        if (not isinstance(action_id, str) or not isinstance(action, dict)
                or not isinstance(action.get('type'), str) or not action['type'].strip()):
            raise ValueError('Action requires a string ID and nonempty type')
        intervals = merge_intervals(_intervals(action.get('frame_intervals')))
        if merge_intervals(timeline + intervals) != timeline:
            raise ValueError(f'Action {action_id} interval outside annotation timeline')
        by_id[action_id] = intervals
        by_type[action['type']].extend(intervals)
    if 'frames' in root:
        frames = root['frames']
        if not isinstance(frames, dict):
            raise ValueError('frames must be an object')
        observed, frame_indices = defaultdict(list), []
        for key, frame in frames.items():
            if (not isinstance(key, str) or not key.isascii() or not key.isdecimal()
                    or len(key) > 19 or str(int(key)) != key or int(key) >= 2**63):
                raise ValueError('Invalid per-frame index')
            index = int(key)
            if not _contains(timeline, index):
                raise ValueError('Per-frame reference outside annotation timeline')
            if not isinstance(frame, dict) or not isinstance(frame.get('actions', {}), dict):
                raise ValueError('Invalid per-frame actions')
            frame_indices.append((index, index))
            for action_id, reference in frame.get('actions', {}).items():
                if action_id not in by_id or not isinstance(reference, dict):
                    raise ValueError('Invalid or undefined per-frame action reference')
                observed[action_id].append((index, index))
        if merge_intervals(frame_indices) != timeline:
            raise ValueError('Per-frame keys do not match annotation timeline')
        for action_id, intervals in by_id.items():
            if merge_intervals(observed[action_id]) != intervals:
                raise ValueError(f'Per-frame references do not match action {action_id} intervals')
    return timeline, {kind: merge_intervals(spans) for kind, spans in by_type.items()}


def _active_axes(by_type, frame):
    axes = defaultdict(set)
    for kind, intervals in by_type.items():
        if _contains(intervals, frame):
            axis, _, value = kind.partition('/')
            axes[axis].add(value)
    return axes


def _distraction_labels(axes):
    targets, masks = [0]*4, [0]*4
    driver = axes['driver_actions']
    if driver and not (driver - DRIVER_TYPES):
        safe = 'safe_drive' in driver
        for column, positive in ((1, bool(driver & PHONE_TYPES)), (2, 'drinking' in driver)):
            if positive and not safe:
                targets[column], masks[column] = 1, 1
            elif safe and not positive:
                masks[column] = 1
    gaze = axes['gaze_on_road']
    if gaze == {'not_looking_road'}:
        targets[3], masks[3] = 1, 1
    elif gaze == {'looking_road'}:
        masks[3] = 1
    return targets, masks


def compile_labels(document, view, frame_times_ms, *, closed_eye_min_ms, subset):
    """Compile inclusive annotation intervals onto causal 40 ms slots.

    ``frame_times_ms`` contains every actually decoded source frame PTS, rebased
    to zero, as strictly increasing integer milliseconds. Declared stream frame
    counts are diagnostic only. At each slot the most recent past frame supplies
    labels, unless it is over 80 ms old or outside the annotation timeline.
    Recordings requiring more than MAX_OUTPUT_SLOTS are rejected, never truncated.

    A closed-eye run accrues time only at source observations, including those
    not selected for an output slot. Unknown/conflicting eye states, annotation
    gaps, and source PTS gaps over 80 ms reset it. The threshold is mandatory.
    ``subset`` is explicitly distraction, drowsiness, or gaze; gaze has no
    supervision until a separate gaze-zone mapping has been validated.
    """
    if view not in VIEWS:
        raise ValueError('view must be face, body, or hands')
    if subset not in ('distraction', 'drowsiness', 'gaze'):
        raise ValueError('subset must be distraction, drowsiness, or gaze')
    if type(closed_eye_min_ms) is not int or closed_eye_min_ms <= 0:
        raise ValueError('closed_eye_min_ms must be a positive integer')
    if (not isinstance(frame_times_ms, list) or not frame_times_ms
            or any(type(time) is not int or not 0 <= time < 2**63 for time in frame_times_ms)
            or frame_times_ms[0] != 0
            or any(b <= a for a, b in zip(frame_times_ms, frame_times_ms[1:]))):
        raise ValueError('frame_times_ms must start at 0 and strictly increase in integer ms within [0, 2**63)')
    if frame_times_ms[-1] // 40 + 1 > MAX_OUTPUT_SLOTS:
        raise ValueError(f'Output capacity exceeds {MAX_OUTPUT_SLOTS} slots; labels are not truncated')
    root = _root(document)
    streams = _streams(root)
    offset = streams[view]['shift'] - min(item['shift'] for item in streams.values())
    timeline, by_type = _annotations(root)

    source_labels = []
    closed_since = None
    for index, time in enumerate(frame_times_ms):
        frame = index + offset
        targets, masks = [0]*4, [0]*4
        if index and time - frame_times_ms[index - 1] > MAX_SOURCE_AGE_MS:
            closed_since = None
        if not _contains(timeline, frame):
            closed_since = None
        elif subset != 'gaze':
            axes = _active_axes(by_type, frame)
            if subset == 'distraction':
                targets, masks = _distraction_labels(axes)
            else:
                eyes, yawns = axes['eyes_state'], axes['yawning']
                if eyes == {'close'}:
                    if closed_since is None:
                        closed_since = time
                else:
                    closed_since = None
                positive_yawn = bool(yawns) and yawns <= YAWN_TYPES
                if positive_yawn or (closed_since is not None
                                          and time - closed_since >= closed_eye_min_ms):
                    targets[0], masks[0] = 1, 1
                elif eyes == {'open'} and not yawns:
                    masks[0] = 1
        source_labels.append((targets, masks))

    slots = list(range(0, frame_times_ms[-1] + 1, 40))
    source_indices = [bisect_right(frame_times_ms, slot) - 1 for slot in slots]
    annotation_indices = [index + offset for index in source_indices]
    targets, masks = [], []
    stale_slots = outside_slots = 0
    for slot, source, frame in zip(slots, source_indices, annotation_indices):
        stale = slot - frame_times_ms[source] > MAX_SOURCE_AGE_MS
        outside = not _contains(timeline, frame)
        stale_slots += int(stale)
        outside_slots += int(outside)
        target, mask = source_labels[source]
        targets.append([0]*4 if stale or outside else target.copy())
        masks.append([0]*4 if stale or outside else mask.copy())
    class_counts = {}
    for column, task in enumerate(TASKS):
        known = sum(row[column] for row in masks)
        positive = sum(row[column] for row in targets)
        class_counts[task] = {'positive': positive, 'negative': known - positive,
                              'unknown': len(slots) - known}
    summary = {
        'class_counts': class_counts, 'source_frame_count': len(frame_times_ms),
        'declared_frame_count': streams[view]['count'],
        'frame_count_mismatch': len(frame_times_ms) != streams[view]['count'],
        'stale_slots': stale_slots, 'outside_timeline_slots': outside_slots,
        'source_pts_gaps': sum(b - a > MAX_SOURCE_AGE_MS
                               for a, b in zip(frame_times_ms, frame_times_ms[1:])),
        'view': view, 'subset': subset, 'camera_offset_frames': offset,
        'closed_eye_min_ms': closed_eye_min_ms,
    }
    return {'schema_version': 'driveguard-behavior-v1', 'tasks': list(TASKS),
            'slots_ms': slots, 'source_indices': source_indices,
            'annotation_indices': annotation_indices, 'targets': targets,
            'label_masks': masks, 'summary': summary}
