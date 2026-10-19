"""Frozen live model assembly; never consumes annotations or training caches."""
from collections import deque
import base64
import numpy as np

from inference.live.routing import compose_tasks


def pixels(value):
    if value is None or isinstance(value,np.ndarray):return value
    raw=base64.b64decode(value,validate=True)
    if len(raw)!=3*224*224:raise ValueError('Invalid RGB crop payload')
    return np.frombuffer(raw,dtype=np.uint8).reshape(3,224,224).copy()


class ModelSet:
    def __init__(self,bundle_path,backend='cpu',device_index=0):
        import torch
        from inference.live.bundle import load_bundle
        from training.common.model_factory import load_model
        from training.visual.train import load_probe
        from training.visual.action_train import load_actions
        from training.visual.eye_train import load_eye
        from training.visual.encoder import FrozenEncoder
        self.bundle,paths=load_bundle(bundle_path)
        self.coverage,self.coverage_meta=load_model(paths['coverage25'],backend,device_index)
        self.gap,self.gap_meta=load_model(paths['gap80'],backend,device_index)
        self.probe,self.probe_meta=load_probe(paths['distraction'],backend,device_index)
        self.actions,action_meta=load_actions(paths['actions'],backend,device_index)
        self.eye,eye_meta=load_eye(paths['eye'],backend,device_index)
        self.encoder=FrozenEncoder(paths['rgb_encoder'],self.bundle['models']['rgb_encoder']['sha256'],backend,device_index)
        self.device=next(self.coverage.parameters()).device
        self.threshold=self.bundle['distraction_threshold']
        self.normalizer=self.probe_meta['provenance']['normalization']
        if (self.probe_meta['model_params']['mode']!='fusion' or self.probe_meta['model_params']['visual_dim']!=512 or
                self.probe_meta['provenance']['preparation']['baseline_checkpoint_sha256']!=self.bundle['models']['gap80']['sha256'] or
                eye_meta['provenance']['source_reference']['checkpoint_sha256']!=self.bundle['models']['actions']['sha256'] or
                eye_meta['input_mode']!='eye' or not action_meta['auxiliary']):
            raise ValueError('Live specialist dependency mismatch')
        for meta,gap in ((self.coverage_meta,0),(self.gap_meta,2)):
            if (meta['config']['model']['kind']!='relation_gaze_head_policy_v1' or
                    meta['config']['model']['head_policy']!={'min_valid_ratio':.25,'max_gap_steps':gap} or
                    meta['config']['data']['window_steps']!=100 or meta['active_tasks']!=
                    ['fatigue','phone_use','drinking','distraction']):
                raise ValueError('Live base architecture/task policy mismatch')
        if self.coverage_meta['config']['data']!=self.gap_meta['config']['data']:
            raise ValueError('Base models must share feature/group definitions')
        for model in (self.coverage,self.gap,self.probe,self.actions,self.eye):
            model.eval().requires_grad_(False)
            if any(not torch.isfinite(v).all() for v in model.state_dict().values() if v.is_floating_point()):
                raise ValueError('Nonfinite model state')

    def encode_rgb(self,image):
        from training.visual.encoder import MEAN,STD
        return self.encoder.encode(((image.astype(np.float32)/255-MEAN[:,None,None])/STD[:,None,None])[None])[0]

    def action_scores(self,head,eye):
        import torch
        with torch.inference_mode():
            yawn=float(self.actions.frame_actions(torch.from_numpy(head[None]).to(self.device))[0,1])
            closed=float(self.eye(torch.from_numpy(eye[None]).to(self.device))[0]) if eye is not None else -1e6
        return closed,yawn

    def window(self,values,masks,ages,rgb,rgb_mask,view,fatigue):
        import torch
        from training.gaze.preprocessing import dependency_masks,prepare_window
        from training.visual.prepare import frozen_context
        from training.visual.data import scale
        def inputs(meta):
            profile=meta['config']['data']['group_profile']
            effective=dependency_masks(masks,ages,profile)
            prepared=prepare_window(values[-100:],effective[-100:],ages[-100:],meta['normalizer'],profile)
            return {key:torch.from_numpy(value[None]).to(self.device) for key,value in prepared.items()}
        def unpack(output):
            return {key:output[key][0].cpu().tolist() for key in
                    ('behavior_logits','task_available','gaze_logits','gaze_available')}
        with torch.inference_mode():
            coverage=unpack(self.coverage(**inputs(self.coverage_meta)))
            gap=coverage;dist=None
            if view=='body':
                context,gap_result=frozen_context(self.gap,inputs(self.gap_meta))
                gap=unpack(gap_result)
                if len(rgb)==8:
                    visual=scale(rgb,self.normalizer['visual']);visual[~rgb_mask]=0
                    normalized=scale(context.cpu().numpy(),self.normalizer['context'])
                    result=self.probe(torch.from_numpy(visual[None]).to(self.device),
                        torch.from_numpy(normalized).to(self.device),torch.from_numpy(rgb_mask[None]).to(self.device))
                    dist=dict(score=float(result['logits'][0]),available=bool(result['available'][0]))
        return compose_tasks(view,coverage,gap,dist,self.threshold,fatigue)


class LiveStream:
    def __init__(self,models,view):
        from training.visual.action_model import EventDecoder
        self.models,self.view=models,view
        # 99 preceding rows are needed for dependency history before a 100-slot window.
        self.history=deque(maxlen=199);self.rgb=deque(maxlen=8);self.rgb_mask=deque(maxlen=8)
        self.head_valid=deque(maxlen=32);self.decoder=EventDecoder(1000)
        self.event=None;self.slot=0

    def consume(self,samples):
        latest=None
        for item in samples:
            slot=item['slot']
            if type(slot) is not int or slot!=self.slot:raise ValueError('Noncontiguous live feature slots')
            self.slot+=1
            self.history.append((item['values'],item['masks'],item['ages']))
            if self.view=='body':
                if slot%25 in (12,24):
                    image=pixels(item.get('rgb_pixels'));valid=image is not None and item['fresh']
                    self.rgb.append(self.models.encode_rgb(image) if valid else np.zeros(512,np.float32))
                    self.rgb_mask.append(valid)
                if item['head_sample']:
                    head=pixels(item.get('head_pixels'));eye=pixels(item.get('eye_pixels'))
                    valid=head is not None and item['fresh'];self.head_valid.append(valid)
                    closed,yawn=self.models.action_scores(head,eye) if valid else (-1e6,-1e6)
                    self.event=self.decoder.step(item['source_time_ms'],closed,yawn,valid,reset=item['reset'])
            if slot>=99 and slot%25==24:
                values,masks,ages=(np.asarray(x,dtype=np.float32) for x in zip(*self.history))
                fatigue=dict(score=self.event,available=len(self.head_valid)==32 and all(self.head_valid) and self.event is not None)
                latest=self.models.window(values,masks,ages,np.asarray(self.rgb,np.float32),
                    np.asarray(self.rgb_mask,bool),self.view,fatigue)
                latest.update(window_start_ms=(slot-99)*40,window_end_ms=slot*40,
                              source_timestamp_ms=item['source_time_ms'],view=self.view)
        return latest
