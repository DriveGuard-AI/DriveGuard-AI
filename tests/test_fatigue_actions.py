"""Action supervision must preserve causal labels, masks and source predictions."""
from copy import deepcopy
import unittest
import tempfile
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
import io
from contextlib import redirect_stdout

import numpy as np
import torch
from unittest.mock import patch
from tests.test_dmd_labels import document
from tests.test_head_adapt_train import TinyEncoder, probe


class ActionLabelsTests(unittest.TestCase):
    def test_original_immutable_cache_annotation_alignment_and_tampering(self):
        from tests.test_face_diagnostic_replay import replay_fixture
        from training.visual.action_labels import ActionLabels
        from training.visual.head_frames import selected_slots
        from training.common.config import read_json,canonical_hash
        from training.common.data import file_sha256
        from training.common.checkpoint import atomic_json
        from training.dmd.labels import compile_labels
        with tempfile.TemporaryDirectory(dir='runs') as tmp,replay_fixture(tmp) as (paths,_,_):
            root=Path(tmp); attempt=paths['data_root']/'records/job1/attempt1'
            annotation=paths['dataset_root']/'ann_drowsiness.json'
            doc=document(('eyes_state/open',0,24),('eyes_state/close',25,49),
                         ('eyes_state/open',50,99),('yawning/Yawning without hand',30,40),count=100)
            atomic_json(annotation,doc)
            original=compile_labels(doc,'body',list(range(0,4000,40)),closed_eye_min_ms=500,subset='drowsiness')
            complete=read_json(attempt/'complete.json'); inputs=read_json(attempt/'input.json')
            job=complete['job']; job.update(subset='drowsiness',annotation=annotation.name,
                 annotation_sha256=file_sha256(annotation),annotation_sources=[dict(path=annotation.name,sha256=file_sha256(annotation))])
            inputs.update(job=job,label_policy={'closed_eye_min_ms':500})
            atomic_json(attempt/'input.json',inputs)
            for field,dtype in (('targets',np.float32),('label_masks',np.uint8)):
                np.save(attempt/complete['arrays'][field],np.asarray(original[field],dtype))
                complete['array_sha256'][field]=file_sha256(attempt/complete['arrays'][field])
            complete.update(job=job,fingerprint=canonical_hash(inputs))
            atomic_json(attempt/'complete.json',complete)
            manifest=read_json(paths['manifest']); record=manifest['records'][0]; record['subset']='drowsiness'
            atomic_json(paths['manifest'],manifest)
            slots=selected_slots(100)
            cache=SimpleNamespace(records={'job1':record},arrays=lambda rid:dict(
                slot_indices=slots,source_pts_ms=slots*40,valid=np.ones(len(slots),np.bool_)))
            prepared=SimpleNamespace(report={'manifest':str(paths['manifest']),'data_root':str(paths['data_root'])},
                rows={'val':[dict(record_id='job1',end_slot=99,target=0,label_known=True)]})
            labels=ActionLabels(prepared,cache,paths['dataset_root'])
            self.assertEqual(labels.closed_eye_min_ms,500)
            self.assertEqual(labels.records['job1']['targets'].shape,(32,2))
            self.assertIn('overlap',labels.records['job1']['subtypes'])
            annotation.write_text('{}')
            with self.assertRaisesRegex(ValueError,'annotation checksum'):
                ActionLabels(prepared,cache,paths['dataset_root'])

    def test_closed_eye_is_supervised_before_fatigue_and_end_label_not_broadcast(self):
        from training.visual.action_labels import compile_actions
        doc = document(('eyes_state/open',0,1),('eyes_state/close',2,7))
        result = compile_actions(doc,'body',list(range(0,320,40)),80,'drowsiness')
        np.testing.assert_array_equal(result['targets'][:,0],[0,0,1,1,1,1,1,1])
        self.assertTrue(result['masks'][:,0].all())
        self.assertEqual(result['fatigue_targets'][2],0)
        self.assertEqual(result['fatigue_masks'][2],0)
        self.assertEqual(result['fatigue_targets'][4],1)
        self.assertEqual(result['subtypes'][4],'closed_eye')

    def test_shift_conflicts_missing_and_no_future_leak(self):
        from training.visual.action_labels import compile_actions
        doc = document(('eyes_state/open',2,3),('eyes_state/close',3,4),
                       ('yawning/Yawning without hand',4,5),count=12,shifts=(0,2,0))
        times=[0,40,80,120,160]
        result = compile_actions(doc,'body',times,40,'drowsiness')
        self.assertEqual(result['masks'][1,0],False)  # conflicting eyes
        self.assertEqual(result['masks'][4,1],False)  # absence alone is unknown
        self.assertEqual(result['targets'][2,1],1)
        changed=deepcopy(doc)
        changed['openlabel']['actions']['new']={'type':'eyes_state/close',
             'frame_intervals':[{'frame_start':7,'frame_end':9}]}
        other=compile_actions(changed,'body',times,40,'drowsiness')
        np.testing.assert_array_equal(result['targets'],other['targets'])
        np.testing.assert_array_equal(result['masks'],other['masks'])
        distraction=compile_actions(doc,'body',times,40,'distraction')
        self.assertFalse(distraction['masks'].any())

    def test_source_gaps_reset_ground_truth_duration(self):
        from training.visual.action_labels import compile_actions
        result=compile_actions(document(('eyes_state/close',0,7)), 'body',
                               [0,40,80,240,280,320],80,'drowsiness')
        self.assertEqual(result['subtypes'][2],'closed_eye')
        self.assertEqual(result['subtypes'][6],'unknown')
        self.assertEqual(result['subtypes'][8],'closed_eye')

    def test_inference_resets_use_full_source_clock_not_dense_sample_spacing(self):
        from training.visual.action_labels import pixel_resets
        # 120/160ms between sampled frames is normal; only an actual gap in
        # decoded source frames should reset the event clock.
        np.testing.assert_array_equal(pixel_resets(np.arange(0,520,40),[80,200,320,480]),
                                      [True,False,False,False])
        np.testing.assert_array_equal(pixel_resets([0,40,80,240,280,320],[80,240,320]),
                                      [True,True,False])
        with self.assertRaisesRegex(ValueError,'PTS'): pixel_resets([0,40,80],[60])


class EventTests(unittest.TestCase):
    def test_duration_reset_and_no_annotation_input(self):
        from training.visual.action_model import EventDecoder
        decoder=EventDecoder(500)
        values=[decoder.step(t,2.,-2.,True) for t in (0,125,250,375,500)]
        self.assertTrue(all(x<0 for x in values[:4]))
        self.assertGreaterEqual(values[-1],0)
        self.assertLess(decoder.step(625,2.,-2.,True,reset=True),0)
        self.assertGreater(decoder.step(750,-2.,3.,True),0) # yawn alone
        self.assertIsNone(decoder.step(875,2.,2.,False))
        self.assertLess(decoder.step(1000,2.,-2.,True),0)
        with self.assertRaises(ValueError): decoder.step(999,2.,-2.,True)


class ActionModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_warm_start_predictions_and_unknown_masked_loss(self):
        from training.visual.head_adapt_train import AdaptModel
        from training.visual.action_model import ActionModel, loss_terms
        with patch('training.visual.head_adapt_train.ResNet18',TinyEncoder):
            base=AdaptModel(probe(),TinyEncoder().state_dict(),32,True,16).eval()
            model=ActionModel(base).eval()
            pixels=torch.full((1,32,3,224,224),120,dtype=torch.uint8)
            mask=torch.ones(1,32,dtype=torch.bool)
            with torch.no_grad():
                expected=base(pixels,mask)
                actual=model(pixels,mask)
            torch.testing.assert_close(expected['logits'],actual['logits'],atol=0,rtol=0)
            self.assertEqual(actual['actions'].shape,(1,32,2))
            later=pixels.clone(); later[:,16:]=255
            with torch.no_grad():
                torch.testing.assert_close(model(later,mask)['actions'][:,:16],actual['actions'][:,:16],atol=0,rtol=0)
            actual={k:v.detach().requires_grad_(True) if v.is_floating_point() else v
                    for k,v in actual.items()}
            targets=torch.zeros(1,32,2); targets[0,0,0]=1
            masks=torch.zeros(1,32,2,dtype=torch.bool); masks[0,0,0]=True
            targets[~masks]=float('nan')
            terms=loss_terms(actual,torch.tensor([float('nan')]),torch.tensor([False]),targets,masks)
            self.assertEqual(terms['fatigue'][1],0)
            self.assertEqual(terms['eye'][1],1)
            self.assertEqual(terms['yawn'][1],0)
            terms['eye'][0].backward()
            self.assertEqual(torch.count_nonzero(actual['actions'].grad).item(),1)


class ActionTrainTests(unittest.TestCase):
    def test_tiny_training_checkpoint_reload_and_epoch_resume(self):
        from training.visual.action_train import train_actions, load_actions
        from training.visual.head_adapt_train import AdaptModel, DEFAULTS
        from training.common.checkpoint import load_checkpoint
        from torch.utils.data import Dataset
        class TinyData(Dataset):
            def __len__(self): return 3
            def metadata(self,i): return dict(target=i%2,label_known=i!=2)
            def availability(self,i): return True
            def supervision(self,i):
                return np.full((32,2),i%2,np.float32),np.ones((32,2),np.bool_)
            def __getitem__(self,i):
                actions,masks=self.supervision(i)
                return dict(images=torch.full((32,3,224,224),80+i*30,dtype=torch.uint8),
                            visual_mask=torch.ones(32,dtype=torch.bool),
                            action_targets=actions,action_masks=masks),float(i%2),i!=2
        torch.set_num_threads(1)
        torch.manual_seed(5)
        with patch('training.visual.head_adapt_train.ResNet18',TinyEncoder):
            backbone=AdaptModel(probe(),TinyEncoder().state_dict(),32,True,16)
            data=TinyData()
            settings=dict(DEFAULTS,epochs=2,batch_size=1,accumulation_steps=2,
                          early_stopping_patience=9,backend='cpu')
            with tempfile.TemporaryDirectory(dir='runs') as tmp:
                root=Path(tmp)
                full=train_actions(data,data,deepcopy(backbone),settings,root/'full',{'fixture':1},True)
                first=train_actions(data,data,deepcopy(backbone),dict(settings,epochs=1),root/'first',{'fixture':1},True)
                resumed=train_actions(data,data,deepcopy(backbone),settings,root/'resumed',{'fixture':1},True,
                                      resume=first['last_checkpoint'])
                a=load_checkpoint(full['last_checkpoint']); b=load_checkpoint(resumed['last_checkpoint'])
                self.assertEqual(a['history'],b['history'])
                for k,v in a['model_state'].items(): torch.testing.assert_close(v,b['model_state'][k],atol=0,rtol=0)
                model,payload=load_actions(full['best_checkpoint'])
                self.assertEqual(payload['selection_metric'],'action_validation_bce')
                self.assertEqual(model(data[0][0]['images'][None],torch.ones(1,32,dtype=torch.bool))['actions'].shape,(1,32,2))


class ActionPipelineTests(unittest.TestCase):
    def test_four_trainings_reference_same_cohort_events_and_idempotent_continue(self):
        from training.visual.fatigue_actions import run
        from training.visual.head_adapt_train import AdaptModel,DEFAULTS
        from training.visual.head_frames import selected_slots,FrameDataset
        from training.visual.action_eval import evaluate_direct
        from training.visual.fatigue_spatial_suite import _summary
        from training.visual.action_labels import ActionLabels
        from training.common.config import canonical_hash,read_json
        torch.set_num_threads(1); torch.manual_seed(5)
        with tempfile.TemporaryDirectory(dir='runs') as tmp,patch('training.visual.head_adapt_train.ResNet18',TinyEncoder):
            root=Path(tmp).resolve(); source=root/'source'; source.mkdir()
            report_path=source/'comparison.json'; report_path.write_text('{}')
            backbone=AdaptModel(probe(),TinyEncoder().state_dict(),32,True,16).eval()
            arrays={}; metadata={}; records={}; rows={}
            for split,rid,subject in (('train','train1','1'),('val','val1','7')):
                slots=selected_slots(150); n=len(slots)
                arrays[rid]=dict(images=np.full((n,3,224,224),110,np.uint8),
                                 slot_indices=slots,source_pts_ms=slots*40,valid=np.ones(n,np.bool_))
                metadata[rid]=dict(record_id=rid,recording_id=rid,subject_id=subject,
                                   view='body',subset='drowsiness',split=split)
                targets=np.zeros((n,2),np.float32); targets[n//2:]=1
                records[rid]=dict(targets=targets,masks=np.ones((n,2),np.bool_),
                                 subtypes=np.full(150,'closed_eye'),reset=np.r_[True,np.zeros(n-1,np.bool_)])
                rows[split]=[dict(metadata[rid],start_slot=i*25,end_slot=i*25+99,
                                  target=i%2,label_known=i!=2,f_eligible=False) for i in range(3)]
            base=SimpleNamespace(rows=rows,provenance={'prepared_sha256':'fixture'})
            cache=SimpleNamespace(records=metadata,arrays=lambda rid:arrays[rid],report={'prepared_sha256':'fixture'})
            labels=ActionLabels.__new__(ActionLabels)
            labels.records=records; labels.closed_eye_min_ms=500
            labels.provenance={'fixture':True}; labels.digest=canonical_hash(labels.provenance)
            reference_rows=evaluate_direct(FrameDataset(base,cache,'val',32),backbone,1)
            reference=_summary(reference_rows,variant='D_tune32-seed43-best',arm='D_tune32',seed=43,
                               checkpoint_kind='best',checkpoint='reference.pt',checkpoint_sha256='fixture',epoch=2)
            @contextmanager
            def inputs(*args):
                yield dict(base=base,cache=cache,labels=labels,backbone=backbone,reference=reference,
                           rows=reference_rows,provenance={'fixture':True},sources=[source])
            settings=dict(DEFAULTS,epochs=1,batch_size=1,accumulation_steps=2,backend='cpu')
            with patch('training.visual.fatigue_actions._inputs',inputs),redirect_stdout(io.StringIO()):
                report=run(adaptation_report=report_path,dataset_root=source,output_dir=root/'out',settings=settings)
                repeated=run(adaptation_report=report_path,dataset_root=source,output_dir=root/'out',settings=settings,continue_run=True)
            self.assertEqual(report,repeated)
            self.assertEqual(report['training_runs'],4)
            self.assertEqual(len(report['comparisons']),17)
            for row in report['comparisons']:
                self.assertEqual(row['window_label_hash'],reference['window_label_hash'])
                self.assertEqual(row['overall']['windows'],3)
                self.assertEqual(sum(v['positive'] for v in row['by_subtype'].values()),1)
            self.assertEqual(read_json(root/'out/comparison.json'),report)
            # Alter only labels; inference must be unchanged, including event
            # results. No annotations enter frame_actions or EventDecoder.
            from training.visual.action_eval import evaluate_events
            from training.visual.action_model import ActionModel
            model=ActionModel(backbone).eval()
            resets={rid:r['reset'] for rid,r in records.items()}
            before,_=evaluate_events(cache,model,reference_rows,resets,500)
            changed=[dict(r,target=1-r['target'],label_known=not r['label_known']) for r in reference_rows]
            after,_=evaluate_events(cache,model,changed,resets,500)
            self.assertEqual([r['logit'] for r in before],[r['logit'] for r in after])


if __name__=='__main__': unittest.main()
