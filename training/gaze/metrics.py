"""Categorical metrics with explicit unsupported classes."""
import torch
from training.gaze.profile import GAZE_CLASSES


class GazeMetrics:
    def __init__(self):
        self.confusion=torch.zeros(9,9,dtype=torch.int64)
        self.unknown=0

    def update(self,logits,targets,masks):
        known=masks.detach().cpu().bool()
        truth=targets.detach().cpu()[known]
        predicted=logits.detach().cpu().argmax(-1)[known]
        self.confusion += torch.bincount(truth*9+predicted,minlength=81).reshape(9,9)
        self.unknown += int((~known).sum())

    def report(self):
        support=self.confusion.sum(1); predicted=self.confusion.sum(0); tp=self.confusion.diag()
        classes={}; f1s=[]
        for i,name in enumerate(GAZE_CLASSES):
            recall=float(tp[i]/support[i]) if support[i] else None
            precision=float(tp[i]/predicted[i]) if predicted[i] else 0.
            f1=float(2*tp[i]/(support[i]+predicted[i])) if support[i] else None
            classes[name]={'support':int(support[i]),'recall':recall,'precision':precision,'f1':f1}
            if f1 is not None: f1s.append(f1)
        return {'class_order':list(GAZE_CLASSES),'confusion_matrix':self.confusion.tolist(),'per_class':classes,
            'accuracy':float(tp.sum()/support.sum()) if support.sum() else None,
            'macro_f1':sum(f1s)/len(f1s) if f1s else None,
            'missing_classes':[name for i,name in enumerate(GAZE_CLASSES) if not support[i]],
            'known':int(support.sum()),'unknown_or_unavailable':self.unknown}
