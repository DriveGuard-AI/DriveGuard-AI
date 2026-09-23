"""Reload a joint checkpoint and predict from measured feature history."""
import argparse
import json
import numpy as np
import torch
from training.common.config import TASKS
from training.common.model_factory import load_model
from training.gaze.config import validate_config
from training.gaze.preprocessing import dependency_masks,prepare_window
from training.gaze.profile import GAZE_CLASSES


class JointPredictor:
    """Feature-level inference; supply real measurements, masks and expert ages.

    Supply preceding history as well as the latest window when available. Cached
    history masks follow the same conservative dependency filtering as training.
    The caller is responsible for the canonical 25 Hz feature extraction profile.
    """
    def __init__(self,checkpoint,backend='cpu',device_index=0):
        self.model,self.metadata=load_model(checkpoint,backend,device_index)
        self.config=validate_config(self.metadata['config'])
        self.device=next(self.model.parameters()).device

    def predict_window(self,values,masks,expert_age_ms):
        values,masks,ages=np.asarray(values),np.asarray(masks),np.asarray(expert_age_ms)
        steps=self.config['data']['window_steps']
        if values.ndim != 2 or values.shape[1] != 23 or len(values)<steps or masks.shape != values.shape or ages.shape != (len(values),3):
            raise ValueError('Prediction requires matching values/masks [N>=100,23] and ages [N,3]')
        profile=self.config['data']['group_profile']
        masks=dependency_masks(masks,ages,profile)
        inputs=prepare_window(values[-steps:],masks[-steps:],ages[-steps:],self.metadata['normalizer'],profile)
        with torch.no_grad():
            out=self.model(**{key:torch.as_tensor(value,device=self.device)[None] for key,value in inputs.items()})
            behavior=torch.sigmoid(out['behavior_logits'][0]).cpu().tolist()
            gaze=torch.softmax(out['gaze_logits'][0],dim=-1).cpu().tolist()
        available=bool(out['available'][0]); gaze_valid=bool(out['gaze_available'][0])
        result={'available':available,'gaze_available':gaze_valid,
            'behavior_probabilities':{task:float(behavior[i]) if available else None for i,task in enumerate(TASKS)},
            'gaze_probabilities':{name:float(gaze[i]) for i,name in enumerate(GAZE_CLASSES)} if gaze_valid else None,
            'gaze_class':GAZE_CLASSES[max(range(9),key=gaze.__getitem__)] if gaze_valid else None}
        if 'task_available' in out:
            task_available=out['task_available'][0].cpu().tolist()
            result['behavior_available']=dict(zip(TASKS,task_available))
            result['behavior_probabilities']={task:float(behavior[i]) if task_available[i] else None for i,task in enumerate(TASKS)}
            result['distraction_head_evidence_available']=bool(out['head_available'][0])
            result['distraction_evidence_status']='head_supported' if result['distraction_head_evidence_available'] else (
                'limited_head_evidence' if task_available[3] else 'unavailable')
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description='Predict joint behavior/gaze from a trusted checkpoint and feature history')
    for name in ('checkpoint','values','masks','expert-age-ms'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--backend',choices=('cpu','cuda','npu'),default='cpu')
    parser.add_argument('--device-index',type=int,default=0)
    args=parser.parse_args(argv)
    try:
        predictor=JointPredictor(args.checkpoint,args.backend,args.device_index)
        arrays=[np.load(path,mmap_mode='r',allow_pickle=False) for path in (args.values,args.masks,args.expert_age_ms)]
        print(json.dumps(predictor.predict_window(*arrays),allow_nan=False)); return 0
    except (ValueError,RuntimeError,OSError,ImportError) as exc: parser.exit(2,f'error: {exc}\n')


if __name__ == '__main__': raise SystemExit(main())
