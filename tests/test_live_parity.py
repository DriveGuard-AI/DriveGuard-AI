from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
import torch

from inference.live.models import ModelSet
from inference.live.routing import compose_tasks
from training.common.model_factory import build_model
from training.gaze.preprocessing import dependency_masks,prepare_window
from training.visual.model import VisualProbe
from training.visual.prepare import frozen_context
from training.visual.data import scale


class ModelParityTests(unittest.TestCase):
    def test_real_torch_heads_match_frozen_offline_inputs_and_routing(self):
        old=torch.get_num_threads();torch.set_num_threads(1)
        try:
            torch.manual_seed(5)
            cfg=json.loads(Path('configs/training/relation_gaze_head_coverage25.json').read_text())
            gap=deepcopy(cfg);gap['model']['head_policy']['max_gap_steps']=2
            normalizer=dict(feature_names=cfg['data']['feature_names'],supported=[True]*23,mean=[.2]*23,std=[2.]*23)
            models=ModelSet.__new__(ModelSet);models.coverage=build_model(cfg).eval();models.gap=build_model(gap).eval()
            models.coverage_meta=dict(config=cfg,normalizer=normalizer)
            models.gap_meta=dict(config=gap,normalizer=dict(normalizer,mean=[.5]*23))
            models.device=torch.device('cpu');models.probe=VisualProbe(512,96,mode='fusion').eval();models.threshold=-.6933670043945312
            models.normalizer=dict(visual=dict(mean=[.1]*512,std=[2.]*512),context=dict(mean=[.3]*96,std=[3.]*96))
            rng=np.random.default_rng(2);values=rng.normal(size=(220,23)).astype(np.float32)
            masks=np.ones_like(values);masks[70,0]=0;ages=np.zeros((220,3),np.float32)
            rgb=rng.normal(size=(8,512)).astype(np.float32);mask=np.ones(8,bool)
            fatigue=dict(score=.4,available=True)
            actual=models.window(values[-199:],masks[-199:],ages[-199:],rgb,mask,'body',fatigue)
            def reference(model,meta):
                profile=meta['config']['data']['group_profile']
                effective=dependency_masks(masks,ages,profile)
                x=prepare_window(values[-100:],effective[-100:],ages[-100:],meta['normalizer'],profile)
                return {key:torch.from_numpy(value[None]) for key,value in x.items()}
            with torch.no_grad():
                cov=models.coverage(**reference(models.coverage,models.coverage_meta))
                context,base=frozen_context(models.gap,reference(models.gap,models.gap_meta))
                out=models.probe(torch.from_numpy(scale(rgb,models.normalizer['visual'])[None]),
                    torch.from_numpy(scale(context.numpy(),models.normalizer['context'])),torch.from_numpy(mask[None]))
            def unpack(output):return {key:output[key][0].tolist() for key in
                ('behavior_logits','task_available','gaze_logits','gaze_available')}
            expected=compose_tasks('body',unpack(cov),unpack(base),dict(score=float(out['logits'][0]),available=True),models.threshold,fatigue)
            self.assertEqual(actual,expected)
        finally:torch.set_num_threads(old)


if __name__=='__main__':unittest.main()
