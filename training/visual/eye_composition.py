"""Private eye inference with immutable reference yawn logits and clocks."""
import numpy as np
import torch

from training.visual.action_eval import replay_events
from training.visual.eye_scores import score_summary


def infer_eyes(head_cache, eye_cache, model, rows, mode, batch_size=32):
    if mode not in ('head', 'eye') or type(batch_size) is not int or batch_size<1:
        raise ValueError('Invalid eye inference settings')
    model.eval(); device=next(model.parameters()).device
    predictions={}
    for rid in sorted({r['record_id'] for r in rows}):
        print(f'Evaluating independent eye: {mode} {rid}', flush=True)
        head=head_cache.arrays(rid); eye=eye_cache.arrays(rid)
        valid=head['valid'] & eye['valid']
        source=head if mode=='head' else eye
        scores=np.full(len(valid), np.nan, np.float32)
        with torch.no_grad():
            for start in range(0,len(valid),batch_size):
                mask=valid[start:start+batch_size]
                if not mask.any(): continue
                images=torch.from_numpy(source['images'][start:start+batch_size][mask].copy()).to(device)
                values=model(images).detach().cpu().numpy()
                if values.shape!=(int(mask.sum()),) or not np.isfinite(values).all():
                    raise ValueError('Invalid independent eye predictions')
                scores[start:start+len(mask)][mask]=values
        predictions[rid]=scores
    return predictions


def compose_events(head_cache, eye_cache, reference, predictions, rows, resets, duration):
    """Missing eye suppresses/resets ONLY the eye branch, never reference yawn.

    Combined availability stays the original head-window availability. Eye-only
    availability additionally requires an observed eye at the endpoint; missing
    history resets sustained closure but is never used as a negative eye label.
    """
    mixed={}
    for rid in sorted({r['record_id'] for r in rows}):
        head,eye=head_cache.arrays(rid),eye_cache.arrays(rid)
        if any(not np.array_equal(head[k],eye[k]) for k in ('slot_indices','source_pts_ms')):
            raise ValueError('Eye inference clocks differ from reference')
        values=np.asarray(predictions[rid])
        valid=head['valid'] & eye['valid']
        if values.shape!=valid.shape or reference[rid].shape!=(len(valid),2):
            raise ValueError('Eye inference arrays differ from reference')
        if not np.isfinite(values[valid]).all(): raise ValueError('Nonfinite valid eye prediction')
        mixed[rid]=reference[rid].copy()
        mixed[rid][:,0]=np.where(valid,values,-1e6)
        if not np.array_equal(mixed[rid][:,1],reference[rid][:,1],equal_nan=True):
            raise ValueError('Frozen yawn predictions changed')
    combined=replay_events(head_cache,mixed,rows,resets,duration)
    eye_only=replay_events(head_cache,mixed,rows,resets,duration,branches='eye')
    for row, eye_row in zip(combined,eye_only):
        arrays=eye_cache.arrays(row['record_id'])
        index=int(np.searchsorted(arrays['slot_indices'],row['end_slot']))
        start=row['start_slot']//25*8
        evidence=dict(eye_endpoint_available=bool(arrays['valid'][index]),
                      eye_valid_samples=int(arrays['valid'][start:start+32].sum()),eye_total_samples=32)
        row.update(evidence); eye_row.update(evidence)
        if not evidence['eye_endpoint_available']:
            eye_row.update(available=False,logit=None)
    return combined,eye_only


def frame_metrics(predictions, head_cache, eye_cache, labels):
    def group(ids):
        scores=score_summary(np.concatenate([predictions[r] for r in ids]),
                            np.concatenate([labels.records[r]['targets'][:,0] for r in ids]),
                            np.concatenate([labels.records[r]['masks'][:,0] & head_cache.arrays(r)['valid']
                                            & eye_cache.arrays(r)['valid'] for r in ids]))
        scores.pop('curve')
        return scores
    return dict(unit='unique ROI-valid labelled sampled frames',overall=group(sorted(predictions)),
                by_subject={s:group([r for r in sorted(predictions) if head_cache.records[r]['subject_id']==s])
                            for s in sorted({head_cache.records[r]['subject_id'] for r in predictions})})
