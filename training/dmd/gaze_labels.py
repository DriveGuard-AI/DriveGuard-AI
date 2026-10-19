"""Separate categorical gaze supervision on the audited causal source timeline."""
from training.dmd.labels import compile_labels, _root, _annotations, _active_axes
from training.gaze.profile import GAZE_CLASSES


def compile_gaze_labels(document, view, frame_times_ms, *, closed_eye_min_ms, subset):
    result = compile_labels(document,view,frame_times_ms,closed_eye_min_ms=closed_eye_min_ms,subset=subset)
    targets, masks = [], []
    _, by_type = _annotations(_root(document))
    for slot, source, frame in zip(result['slots_ms'],result['source_indices'],result['annotation_indices']):
        zones = _active_axes(by_type,frame)['gaze_zone']
        valid = (subset == 'gaze' and view == 'face' and slot-frame_times_ms[source] <= 80
                 and len(zones) == 1 and next(iter(zones)) in GAZE_CLASSES)
        targets.append(GAZE_CLASSES.index(next(iter(zones))) if valid else 0)
        masks.append(int(valid))
    result.update(gaze_targets=targets,gaze_label_masks=masks)
    result['summary']['gaze'] = dict(classes=list(GAZE_CLASSES),
        counts=[sum(m and t == c for t,m in zip(targets,masks)) for c in range(len(GAZE_CLASSES))],
        unknown=len(masks)-sum(masks))
    return result
