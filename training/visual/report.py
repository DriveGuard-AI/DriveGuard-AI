"""Fixed-window body distraction comparison; unavailable positives stay counted."""
import csv
import math
from pathlib import Path

import numpy as np
import torch

from training.common.checkpoint import atomic_json
from training.common.config import canonical_hash
from training.visual.train import load_probe


def metrics(rows, threshold=0.):
    known=[r for r in rows if r['label_known']]
    positive=sum(r['target']==1 for r in known);negative=len(known)-positive
    available=[r for r in known if r['available']]
    predicted=[r for r in available if threshold is not None and r['logit']>=threshold]
    tp=sum(r['target']==1 for r in predicted);fp=len(predicted)-tp
    ap=sum(r['target']==1 for r in available);an=len(available)-ap
    return dict(windows=len(rows),known=len(known),unknown=len(rows)-len(known),
                positive=positive,negative=negative,available=len(available),
                unavailable_positive=positive-ap,unavailable_negative=negative-an,
                tp=tp,fp=fp,fn=ap-tp,tn=an-fp,
                precision=tp/(tp+fp) if tp+fp else 0.,
                available_recall=tp/ap if ap else None,
                all_positive_detection_rate=tp/positive if positive else None,
                all_positive_f1=2*tp/(positive+tp+fp) if positive+tp+fp else 0.,
                false_positive_rate=fp/negative if negative else None,
                available_false_positive_rate=fp/an if an else None,
                coverage=len(available)/len(known) if known else None,
                threshold_logit=threshold)


def operating_point(rows,fpr_budget=.03):
    """Diagnostic validation sweep, grouping tied scores without splitting them."""
    if not 0<=fpr_budget<=1:raise ValueError('Invalid FPR budget')
    known=[r for r in rows if r['label_known']]
    negative=sum(r['target']==0 for r in known)
    allowed=math.floor(fpr_budget*negative+1e-12)
    groups={}
    for row in known:
        if row['available']:
            score=float(row['logit'])
            if not math.isfinite(score):raise ValueError('Nonfinite available logit')
            counts=groups.setdefault(score,[0,0]);counts[int(row['target'])]+=1
    best_threshold=None;best_tp=tp=fp=0
    for threshold,counts in sorted(groups.items(),reverse=True):
        fp+=counts[0];tp+=counts[1]
        if fp>allowed:break
        if tp>best_tp:best_threshold=threshold;best_tp=tp
    return dict(metrics(rows,best_threshold),fpr_budget=fpr_budget,
                threshold_selection='validation_only_diagnostic_not_heldout')


def evaluate(dataset,checkpoint=None,backend='cpu',device_index=0,batch_size=64):
    rows=[dataset.metadata(i) for i in range(len(dataset))]
    if checkpoint is None:
        for row in rows:row.update(available=row['baseline_available'],logit=row['baseline_logit'])
        return rows,None
    model,payload=load_probe(checkpoint,backend,device_index)
    if payload['provenance']!=dataset.prepared.provenance:raise ValueError('Evaluation provenance mismatch')
    device=next(model.parameters()).device
    with torch.no_grad():
        for start in range(0,len(dataset),batch_size):
            items=[dataset[i][0] for i in range(start,min(start+batch_size,len(dataset)))]
            tensors={k:torch.as_tensor(np.stack([item[k] for item in items]),device=device) for k in items[0]}
            output=model(**tensors)
            logits=output['logits'].cpu().tolist();available=output['available'].cpu().tolist()
            for i,(score,valid) in enumerate(zip(logits,available)):
                if valid and not math.isfinite(score):raise ValueError('Nonfinite probe prediction')
                rows[start+i].update(available=bool(valid),logit=float(score) if valid else None)
    return rows,payload


def summarize(rows,**identity):
    labels=[{k:r[k] for k in ('record_id','start_slot','end_slot','label_known','target','h_eligible')} for r in rows]
    return dict(identity,window_label_hash=canonical_hash(labels),overall=metrics(rows),
                baseline_h_ineligible=metrics([r for r in rows if not r['h_eligible']]),
                baseline_h_eligible=metrics([r for r in rows if r['h_eligible']]),
                validation_fpr03=operating_point(rows,.03),
                by_subject={s:metrics([r for r in rows if r['subject_id']==s]) for s in sorted({r['subject_id'] for r in rows})},
                by_record={s:metrics([r for r in rows if r['record_id']==s]) for s in sorted({r['record_id'] for r in rows})})


def write_rows(path,rows):
    with Path(path).open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def write_comparison(output,comparisons):
    report=dict(schema_version='driveguard-body-visual-comparison-v1',status='complete',split='val',
                comparisons=comparisons,notes=[
                    'All arms use identical body validation windows and labels; missing predictions remain in positive denominators.',
                    'H cohorts are fixed by the frozen gap80 baseline, not recomputed per arm.',
                    'Threshold 0.5 metrics use logit 0; 3% FPR thresholds are validation-only diagnostics.',
                    'A uses the original joint-loss-selected checkpoint; B/C best use binary validation BCE.',
                    'Frozen ImageNet encoder; B/C train only body distraction. Gaze/other tasks are not evaluated here.',
                    'No held-out test metrics are used to choose a variant or threshold.'])
    flat=[]
    for row in comparisons:
        flat.append(dict(variant=row['variant'],checkpoint=row['checkpoint'],epoch=row['epoch'],
                         **row['overall'],h_missing_positive=row['baseline_h_ineligible']['positive'],
                         h_missing_tp=row['baseline_h_ineligible']['tp'],h_missing_fp=row['baseline_h_ineligible']['fp'],
                         fpr03_tp=row['validation_fpr03']['tp'],fpr03_fp=row['validation_fpr03']['fp'],
                         fpr03_threshold_logit=row['validation_fpr03']['threshold_logit']))
    write_rows(Path(output)/'comparison.csv',flat)
    atomic_json(Path(output)/'comparison.json',report)
    return report
