"""Annotation-free event inference, followed by separate label-based accounting."""
import numpy as np
import torch

from training.visual.action_model import EventDecoder


def evaluate_direct(dataset,model,batch_size=2):
    model.eval(); device=next(model.parameters()).device
    rows=[dataset.metadata(i) for i in range(len(dataset))]
    with torch.no_grad():
        for start in range(0,len(dataset),batch_size):
            inputs=[dataset[i][0] for i in range(start,min(start+batch_size,len(dataset)))]
            # Auxiliary targets are deliberately never passed into inference.
            images=torch.stack([torch.as_tensor(x['images']) for x in inputs]).to(device)
            masks=torch.stack([torch.as_tensor(x['visual_mask']) for x in inputs]).to(device)
            output=model(images,masks)
            for offset,(value,valid) in enumerate(zip(output['logits'].cpu().tolist(),output['available'].cpu().tolist())):
                if valid and not np.isfinite(value): raise ValueError('Nonfinite fatigue prediction')
                rows[start+offset].update(logit=float(value) if valid else None,available=bool(valid))
    return rows


def evaluate_events(cache,model,rows,resets,closed_eye_min_ms):
    """Labels are never read here. Only images, sample clocks and source gaps.

    Process each recording once in causal order (including its early history),
    rather than pretending each overlapping four-second window is a new event.
    """
    model.eval(); device=next(model.parameters()).device
    predictions={}; events={}
    for rid in sorted({r['record_id'] for r in rows}):
        arrays=cache.arrays(rid)
        count=len(arrays['slot_indices'])
        logits=np.full((count,2),np.nan,np.float32)
        with torch.no_grad():
            for start in range(0,count,32):
                valid=arrays['valid'][start:start+32]
                if not valid.any(): continue
                images=torch.from_numpy(arrays['images'][start:start+32][valid].copy()).to(device)
                values=model.frame_actions(images).cpu().numpy()
                if not np.isfinite(values).all(): raise ValueError('Nonfinite frame action logits')
                logits[start:start+len(valid)][valid]=values
        decoder=EventDecoder(closed_eye_min_ms)
        scores=[]
        for i,t in enumerate(arrays['source_pts_ms']):
            scores.append(decoder.step(int(t),float(logits[i,0]),float(logits[i,1]),
                                       bool(arrays['valid'][i]),reset=bool(resets[rid][i])))
        predictions[rid]=logits; events[rid]=scores
    result=[]
    for row in rows:
        rid=row['record_id']; arrays=cache.arrays(rid)
        index=int(np.searchsorted(arrays['slot_indices'],row['end_slot'],side='right')-1)
        if index<0 or arrays['slot_indices'][index]!=row['end_slot']:
            raise ValueError('Event readout must align with original window endpoint')
        first=row['start_slot']//25*8
        available=bool(arrays['valid'][first:first+32].all()) and events[rid][index] is not None
        result.append(dict(row,available=available,logit=events[rid][index] if available else None))
    return result,predictions


def action_metrics(predictions,labels,cache):
    """Unique sampled frames, not repeated overlapping-window occurrences."""
    def summarize(record_ids):
        result={}
        for col,name in enumerate(('eye','yawn')):
            tp=fp=fn=tn=unknown=0
            for rid in record_ids:
                raw=labels.records[rid]
                valid=cache.arrays(rid)['valid']&raw['masks'][:,col]
                pred=predictions[rid][:,col]>=0
                target=raw['targets'][:,col]==1
                tp+=int((valid&pred&target).sum()); fp+=int((valid&pred&~target).sum())
                fn+=int((valid&~pred&target).sum()); tn+=int((valid&~pred&~target).sum())
                unknown+=int((~valid).sum())
            result[name]=dict(tp=tp,fp=fp,fn=fn,tn=tn,known=tp+fp+fn+tn,unknown_or_invalid=unknown,
                              recall=tp/(tp+fn) if tp+fn else None,
                              precision=tp/(tp+fp) if tp+fp else 0.,
                              f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        return result
    return dict(unit='unique sampled frames',overall=summarize(predictions),
                by_subject={s:summarize([rid for rid in predictions if cache.records[rid]['subject_id']==s])
                            for s in sorted({cache.records[rid]['subject_id'] for rid in predictions})})


def subtype_metrics(rows,labels):
    result={}
    for name in ('closed_eye','yawn','overlap','unresolved'):
        selected=[]
        for r in rows:
            if not r['label_known'] or r['target']!=1: continue
            label=labels.records[r['record_id']]['subtypes'][r['end_slot']]
            if label not in ('closed_eye','yawn','overlap'): label='unresolved'
            if label==name: selected.append(r)
        tp=sum(r['available'] and r['logit']>=0 for r in selected)
        result[name]=dict(positive=len(selected),tp=tp,fn=len(selected)-tp,
                          recall=tp/len(selected) if selected else None)
    return result
