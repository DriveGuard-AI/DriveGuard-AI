"""Exercise all warm-start arms with real training and a tiny image encoder."""
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from tests.test_head_adapt_train import TinyEncoder, probe
from training.common.checkpoint import load_checkpoint,save_checkpoint
from training.visual.fatigue_spatial_suite import _summary


class PipelinePixels:
    def __init__(self,prepared,cache,split,frames):
        self.prepared,self.cache,self.split,self.frames=prepared,cache,split,frames
    def __len__(self): return 3
    def __getitem__(self,i):
        return dict(images=torch.full((self.frames,3,224,224),30+i,dtype=torch.uint8),
                    visual_mask=torch.ones(self.frames,dtype=torch.bool)),float(i%2),i<2
    def metadata(self,i):
        return dict(record_id=f'video-{i}',start_slot=0,end_slot=99,subject_id=str(i%2),
                    label_known=i<2,target=i%2 if i<2 else 0,f_eligible=False)
    def availability(self,i):return True


class AdaptationPipelineTests(unittest.TestCase):
    def test_all_arms_real_training_completed_reuse_and_state_swap_rejection(self):
        from training.visual import head_adapt_train as training
        from training.visual import head_adaptation as suite
        torch.set_num_threads(1)
        torch.manual_seed(81)
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root=Path(tmp).resolve()
            source=root/'source'; source.mkdir()
            report_path=source/'report.json'; report_path.write_text('{}',encoding='utf-8')
            weights=source/'weights.pt'; weights.write_bytes(b'local-tiny-encoder')
            base=SimpleNamespace(provenance=dict(prepared_sha256='test',preparation={}))
            payload=probe()
            payload['provenance'].update(base.provenance)
            state=TinyEncoder().state_dict()
            cache=SimpleNamespace(root=root/'output/head-frames',manifest_sha256='pixels',
                                   report=dict(crop={}),close=lambda:None)
            settings=dict(training.DEFAULTS,epochs=1,batch_size=2,accumulation_steps=1)
            with patch.object(training,'ResNet18',TinyEncoder):
                model=training.AdaptModel(payload,state,8,False)
                rows=training.evaluate_model(PipelinePixels(base,cache,'val',8),model)
                reference=_summary(rows,variant='C_regularized-seed42-best',arm='C_regularized',seed=42,
                                   checkpoint_kind='best',checkpoint=str(source/'reference.pt'),
                                   checkpoint_sha256='original',epoch=1)
                @contextmanager
                def inputs(*args):
                    yield dict(base=base,payload=payload,reference=reference,rows=rows,
                               encoder_state=state,crop={},sources=[source])
                with patch.object(suite,'_inputs',inputs),\
                     patch('training.visual.head_frames.build_frame_cache',return_value={}),\
                     patch('training.visual.head_frames.FrameCache',return_value=cache),\
                     patch('training.visual.head_frames.FrameDataset',PipelinePixels),redirect_stdout(io.StringIO()):
                    kwargs=dict(stability_report=report_path,dataset_root=source,weights=weights,
                                output_dir=root/'output',settings=settings)
                    report=suite.run(**kwargs)
                    with patch.object(training,'train_adaptation',side_effect=AssertionError('must reuse')):
                        continued=suite.run(**kwargs,continue_run=True)
                    self.assertEqual(report,continued)
                    self.assertEqual(report['training_runs'],8)
                    self.assertEqual(len(report['comparisons']),25)
                    self.assertEqual(len(report['aggregate_best']),4)
                    self.assertEqual(set(report['paired_best_deltas']),{'42','43'})
                    for row in report['comparisons'][1:]:
                        self.assertEqual(row['window_label_hash'],reference['window_label_hash'])
                        self.assertIn('changes_from_original',row)
                        ck=load_checkpoint(row['checkpoint'])
                        self.assertEqual(ck['provenance']['warm_start_sha256'],'original')
                    with self.assertRaises(FileExistsError):suite.run(**kwargs)
                    with self.assertRaisesRegex(ValueError,'configuration'):
                        suite.run(**dict(kwargs,settings=dict(settings,learning_rate=.0002)),continue_run=True)
                    # Same metadata and valid SHA sidecar are insufficient if weights disagree with last.pt.
                    best=Path(report['training_results']['A_frozen8-seed42']['best_checkpoint'])
                    altered=deepcopy(load_checkpoint(best))
                    altered['model_state']['probe.head.bias'] += .1
                    save_checkpoint(best,altered)
                    with self.assertRaisesRegex(ValueError,'model state mismatch'):
                        suite.run(**kwargs,continue_run=True)


if __name__=='__main__':unittest.main()
