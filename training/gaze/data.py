"""Strict joint corpus with usable supervision coverage diagnostics."""
import numpy as np
from training.common.config import TASKS
from training.common.data import FeatureCorpus as BaseCorpus
from training.gaze.config import validate_config
from training.gaze.preprocessing import gaze_available
from training.gaze.profile import GAZE_CLASSES
from training.gaze.head_evidence import behavior_availability


class FeatureCorpus(BaseCorpus):
    def __init__(self,config,manifest_path,data_root=None):
        super().__init__(validate_config(config),manifest_path,data_root)

    def check_supervision(self,active_tasks,normalizer=None):
        if list(active_tasks) not in ([],list(TASKS)):
            raise ValueError('Joint coverage requires all four behavior tasks')
        normalizer=normalizer or self.fit_normalizer()
        result={}; problems=[]
        for split in ('train','val'):
            positive=np.zeros(4,np.int64); negative=np.zeros(4,np.int64); unknown=np.zeros(4,np.int64)
            raw_positive=np.zeros(4,np.int64); raw_negative=np.zeros(4,np.int64)
            support=np.zeros(9,np.int64); raw_support=np.zeros(9,np.int64)
            unavailable=gaze_unavailable=gaze_unknown=0
            task_unavailable=np.zeros(4,np.int64)
            groups={key:{} for key in ('record_id','subject_id','view','subset')}
            dataset=self.dataset(split,normalizer)
            for index in range(len(dataset)):
                inputs,target,labels,gaze,known=dataset[index]
                task_available=behavior_availability(inputs,self.config['model']['kind'])
                available=bool(task_available.any())
                task_unavailable+=~task_available
                gaze_valid=gaze_available(inputs['x'][:,23:],inputs['eligible'])
                unavailable+=int(not available); gaze_unavailable+=int(not gaze_valid)
                positive+=(target==1)&labels&task_available; negative+=(target==0)&labels&task_available; unknown+=~labels
                raw_positive+=(target==1)&labels; raw_negative+=(target==0)&labels
                gaze_unknown+=int(not known)
                if known:
                    raw_support[int(gaze)]+=1
                    if gaze_valid: support[int(gaze)]+=1
                metadata=dataset.metadata(index)
                for key in groups:
                    group=groups[key].setdefault(metadata[key],{'positive':[0]*4,'negative':[0]*4,'unknown':[0]*4,
                        'raw_positive':[0]*4,'raw_negative':[0]*4,'gaze_support':[0]*9,'gaze_raw_support':[0]*9,
                        'gaze_unknown':0,'unavailable_windows':0,'gaze_unavailable_windows':0})
                    for column in range(4):
                        group['raw_positive'][column]+=int(target[column]==1 and labels[column])
                        group['raw_negative'][column]+=int(target[column]==0 and labels[column])
                        group['positive'][column]+=int(target[column]==1 and labels[column] and task_available[column])
                        group['negative'][column]+=int(target[column]==0 and labels[column] and task_available[column])
                        group['unknown'][column]+=int(not labels[column])
                    if known:
                        group['gaze_raw_support'][int(gaze)]+=1
                        group['gaze_support'][int(gaze)]+=int(gaze_valid)
                    group['gaze_unknown']+=int(not known)
                    group['unavailable_windows']+=int(not available)
                    group['gaze_unavailable_windows']+=int(not gaze_valid)
            for column,task in enumerate(TASKS):
                if not positive[column] or not negative[column]: problems.append(f'{split} {task}: requires usable positive and negative windows')
            missing=[name for i,name in enumerate(GAZE_CLASSES) if not support[i]]
            if missing: problems.append(f'{split} gaze: missing usable classes {missing}')
            result[split]={'positive':positive.tolist(),'negative':negative.tolist(),'unknown':unknown.tolist(),
                'raw_positive':raw_positive.tolist(),'raw_negative':raw_negative.tolist(),'groups':groups,
                'gaze_support':support.tolist(),'gaze_raw_support':raw_support.tolist(),'gaze_unknown':gaze_unknown,
                'gaze_missing_classes':missing,'unavailable_windows':unavailable,'task_unavailable_windows':task_unavailable.tolist(),
                'gaze_unavailable_windows':gaze_unavailable}
        if problems and active_tasks:
            raise ValueError('Joint supervision coverage failed: '+'; '.join(problems))
        return result
