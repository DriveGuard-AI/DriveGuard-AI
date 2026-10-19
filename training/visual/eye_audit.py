"""Labelled crop review and visibility coverage, separate from model predictions."""
from html import escape
import math

import numpy as np
from PIL import Image, ImageDraw

from training.common.checkpoint import atomic_json
from training.common.config import canonical_hash
from training.visual.action_model import EventDecoder
from training.visual.report import write_rows
from training.visual.spatial import crop_roi


def write_audit(output, head_cache, eye_cache, labels, windows, duration):
    directory=output/'input-audit'; directory.mkdir(exist_ok=True)
    coverage=[]; samples=[]; eligible={}
    for rid, record in sorted(head_cache.records.items()):
        head,eye=head_cache.arrays(rid),eye_cache.arrays(rid)
        label=labels.records[rid]
        known=label['masks'][:,0] & head['valid']
        positive=known & (label['targets'][:,0]==1)
        negative=known & (label['targets'][:,0]==0)
        valid=head['valid'] & eye['valid']
        widths=eye['native_eye_width'][valid]
        counts=dict(record_id=rid,subject_id=record['subject_id'],split=record['split'],
                    sampled_frames=len(known),eye_available=int(valid.sum()),
                    positive=int(positive.sum()),negative=int(negative.sum()),
                    positive_with_eye=int((positive&valid).sum()),negative_with_eye=int((negative&valid).sum()),
                    unknown_or_stale=int((~known).sum()),
                    native_eye_width_min=float(widths.min()) if len(widths) else None,
                    native_eye_width_median=float(np.median(widths)) if len(widths) else None)
        counts['positive_coverage']=counts['positive_with_eye']/counts['positive'] if counts['positive'] else None
        counts['negative_coverage']=counts['negative_with_eye']/counts['negative'] if counts['negative'] else None
        coverage.append(counts)
        if 'origin' in eye:
            counts.update(detected_frames=int((eye['origin']==1).sum()),tracked_frames=int((eye['origin']==2).sum()))
        decoder=EventDecoder(duration); evidence=[]
        for i,t in enumerate(head['source_pts_ms']):
            value=decoder.step(int(t),1.,-1e6,bool(valid[i]),reset=bool(label['reset'][i]))
            evidence.append(value is not None and value>=0)
        eligible[rid]=evidence
        for name,mask in [('positive_with_eye',positive&valid),('negative_with_eye',negative&valid),
                          ('positive_missing_eye',positive&~valid),('negative_missing_eye',negative&~valid)]:
            indices=np.flatnonzero(mask)
            chosen=indices[np.unique(np.linspace(0,len(indices)-1,min(2,len(indices)),dtype=int))] if len(indices) else []
            for j,i in enumerate(chosen):
                left=Image.fromarray(head['images'][i].transpose(1,2,0).copy())
                if valid[i]:
                    roi=crop_roi(record['source']['driver_roi'],head_cache.report['crop'])
                    x0,y0,x1,y1=math.floor(roi[0]),math.floor(roi[1]),math.ceil(roi[2]),math.ceil(roi[3])
                    scale=224/max(x1-x0,y1-y0)
                    w,h=max(1,round((x1-x0)*scale)),max(1,round((y1-y0)*scale))
                    ex0,ey0,ex1,ey1=map(float,eye['roi_xyxy'][i])
                    box=[(ex0-x0)*w/(x1-x0)+(224-w)//2,(ey0-y0)*h/(y1-y0)+(224-h)//2,
                         (ex1-x0)*w/(x1-x0)+(224-w)//2,(ey1-y0)*h/(y1-y0)+(224-h)//2]
                    ImageDraw.Draw(left).rectangle(box,outline='yellow',width=2)
                right=Image.fromarray(eye['images'][i].transpose(1,2,0).copy()) if valid[i] else Image.new('RGB',(224,224))
                canvas=Image.new('RGB',(448,280)); canvas.paste(left,(0,36));canvas.paste(right,(224,36))
                text=f"subject={record['subject_id']} {record['split']} {name}\nslot={int(head['slot_indices'][i])} PTS={int(head['source_pts_ms'][i])}ms"
                if 'origin' in eye:text+=f" origin={int(eye['origin'][i])} age={int(eye['track_age_ms'][i])}ms"
                draw=ImageDraw.Draw(canvas);draw.text((4,2),text,fill='white')
                draw.text((4,263),'head input / native eye input; labels are annotations',fill='white')
                file=f'{canonical_hash(rid)[:12]}-{name}-{j+1}.png'
                canvas.save(directory/file)
                samples.append(dict(file=file,record_id=rid,subject_id=record['subject_id'],split=record['split'],
                                    category=name,slot_index=int(head['slot_indices'][i]),source_pts_ms=int(head['source_pts_ms'][i]),
                                    native_eye_width=float(eye['native_eye_width'][i]),target=int(label['targets'][i,0])))
    window_rows=[]
    for row in windows:
        rid=row['record_id']; head=head_cache.arrays(rid);eye=eye_cache.arrays(rid)
        index=int(np.searchsorted(head['slot_indices'],row['end_slot']))
        first=row['start_slot']//25*8
        window_rows.append(dict(record_id=rid,subject_id=row['subject_id'],start_slot=row['start_slot'],end_slot=row['end_slot'],
                                label_known=row['label_known'],target=row['target'],
                                subtype=str(labels.records[rid]['subtypes'][row['end_slot']]),
                                eye_endpoint_available=bool(eye['valid'][index]),
                                eye_duration_eligible=bool(eligible[rid][index] and head['valid'][first:first+32].all())))
    report=dict(records=coverage,windows=window_rows,samples=samples,
                notes=['Eye crops are detector candidates, not human-verified eyes.',
                       'Positive/negative here means annotated instantaneous eye state, not window fatigue.',
                       'Duration eligibility tests only continuous ROI visibility at the original event duration; it is not model performance.',
                       'Unknown/stale frames never become negatives. Missing eye does not disable frozen yawn.'])
    atomic_json(output/'coverage.json',report)
    write_rows(output/'coverage.csv',coverage)
    if window_rows: write_rows(output/'eye-window-coverage.csv',window_rows)
    cards=''.join(f'<figure><img loading="lazy" src="{escape(s["file"])}"><figcaption>{escape(str(s))}</figcaption></figure>' for s in samples)
    (directory/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Eye input audit</title>'
        '<style>body{font:16px sans-serif}figure{display:inline-block;width:448px;vertical-align:top}figcaption{overflow-wrap:anywhere}</style>'
        '<h1>Eye input audit</h1><p>Head input with crop boundary / native eye input. Annotation labels, not model predictions. '
        'Missing ROI is shown black. Review positive and negative crops, especially subject14.</p>'+cards,encoding='utf-8')
    return report
