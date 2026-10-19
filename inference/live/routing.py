"""Label-free task routing; evidence availability is separate from model scores."""
import math

TASKS = ('fatigue','phone_use','drinking','distraction')
GAZE_CLASSES = ('left_mirror','left','front','center_mirror','front_right',
                'right_mirror','right','infotainment','steering_wheel')


def task(score, available, threshold, source):
    if type(available) is not bool or not math.isfinite(threshold):
        raise ValueError('Invalid task availability/threshold')
    if available and (score is None or not math.isfinite(score)):
        raise ValueError('Available task needs a finite logit')
    value = float(score) if available else None
    sigmoid = (1/(1+math.exp(-value)) if value >= 0 else math.exp(value)/(1+math.exp(value))) if available else None
    return dict(available=available, logit=value, score=sigmoid,
                threshold_logit=float(threshold), detected=value >= threshold if available else None,
                source=source, score_semantics='uncalibrated_sigmoid')


def compose_tasks(view, coverage, gap80, distraction, dist_threshold, fatigue):
    if view not in ('body','face'):
        raise ValueError('Expected body or face view')
    tasks = {name:task(coverage['behavior_logits'][i],coverage['task_available'][i],0.,'coverage25')
             for i,name in enumerate(TASKS)}
    if view == 'body':
        if distraction and distraction['available']:
            tasks['distraction'] = task(distraction['score'],True,dist_threshold,'C_fusion')
        else:
            tasks['distraction'] = task(gap80['behavior_logits'][3],gap80['task_available'][3],0.,'gap80_rgb_fallback')
        tasks['fatigue'] = task(fatigue['score'] if fatigue else None,
                               fatigue['available'] if fatigue else False,0.,'yawn_eye_dense_track')
    logits = coverage['gaze_logits']
    if len(logits)!=9 or not all(math.isfinite(x) for x in logits):
        raise ValueError('Invalid gaze logits')
    available = bool(coverage['gaze_available'])
    exp = [math.exp(x-max(logits)) for x in logits]
    probabilities = [x/sum(exp) for x in exp]
    best = max(range(9),key=logits.__getitem__)
    gaze = dict(available=available, source='coverage25',
        class_name=GAZE_CLASSES[best] if available else None,
        scores=dict(zip(GAZE_CLASSES,probabilities)) if available else None,
        validation_status='unvalidated_body_view' if view=='body' else 'evaluated_face_view')
    return dict(tasks=tasks,gaze=gaze)
