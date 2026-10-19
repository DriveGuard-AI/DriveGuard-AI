"""Matched input ablation and seed-level experiment reporting."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_visual_data import ROOT,prepared_fixture
from training.common.config import read_json


class VisualStabilityTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_subgroups_use_single_global_operating_threshold(self):
        from training.visual.report import summarize
        rows=[]
        for i,(subject,y,score) in enumerate([('s1',1,3.),('s1',0,2.),('s2',1,1.),('s2',0,0.)]):
            rows.append(dict(record_id=subject,subject_id=subject,start_slot=i*25,end_slot=i*25+99,
                             label_known=True,target=y,h_eligible=subject=='s1',available=True,logit=score))
        summary=summarize(rows)
        point=summary['validation_fpr03'];groups=summary['validation_fpr03_groups']
        self.assertEqual(point['tp'],1)
        self.assertEqual(groups['by_subject']['s2']['tp'],0) # per-subject sweep would incorrectly recover this positive
        for group in groups['by_subject'].values():self.assertEqual(group['threshold_logit'],point['threshold_logit'])
        self.assertEqual(sum(g['tp'] for g in groups['by_subject'].values()),point['tp'])

    def test_three_arms_two_seeds_cache_reuse_and_resume(self):
        from training.visual.stability import run_stability
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp);prepared_fixture(root)
            before={str(p):p.stat().st_mtime_ns for folder in ('visual','prepared') for p in (root/folder).rglob('*') if p.is_file()}
            options=dict(prepared_dir=root/'prepared',output_dir=root/'stability',backend='cpu',epochs=1,
                         batch_size=3,seeds=[42,43],patience=2,min_delta=.0001)
            report=run_stability(**options)
            self.assertEqual(report['status'],'complete');self.assertEqual(report['training_runs'],6)
            self.assertEqual(len(report['comparisons']),13)
            self.assertEqual(len({r['window_label_hash'] for r in report['comparisons']}),1)
            self.assertEqual(len(report['aggregate_best']),3)
            for group in report['aggregate_best'].values():self.assertEqual(group['seed_count'],2)
            self.assertEqual(report['paired_best']['C_minus_D']['seed_count'],2)
            best={(r['arm'],r['seed']):r for r in report['comparisons'] if r.get('checkpoint_kind')=='best'}
            for delta in report['paired_best']['C_minus_D']['per_seed']:
                seed=delta['seed']
                self.assertEqual(delta['metrics']['tp'],best[('C_fusion',seed)]['overall']['tp']-best[('D_context_control',seed)]['overall']['tp'])
            for seed in (42,43):
                d=read_json(root/'stability'/f'D_context_control-seed{seed}'/'train-001'/'config.json')
                c=read_json(root/'stability'/f'C_fusion-seed{seed}'/'train-001'/'config.json')
                self.assertEqual(d['training'],c['training'])
                dp=dict(d['model']);cp=dict(c['model']);dp.pop('mode');cp.pop('mode');self.assertEqual(dp,cp)
            self.assertEqual(before,{str(p):p.stat().st_mtime_ns for folder in ('visual','prepared') for p in (root/folder).rglob('*') if p.is_file()})
            checkpoints={str(p):p.stat().st_mtime_ns for p in (root/'stability').rglob('*.pt')}
            again=run_stability(**options,continue_run=True)
            self.assertEqual(again,report)
            self.assertEqual(checkpoints,{str(p):p.stat().st_mtime_ns for p in (root/'stability').rglob('*.pt')})
            with self.assertRaisesRegex(ValueError,'configuration|identity'):
                run_stability(**dict(options,seeds=[42,44]),continue_run=True)

    def test_crash_after_early_stopped_last_checkpoint_finalizes_without_training(self):
        from tests.test_visual_training import ProbeDataset
        from training.visual.suite import DEFAULTS,_train
        from training.common.checkpoint import load_checkpoint
        settings=dict(DEFAULTS,epochs=10,batch_size=3,hidden_dim=8,early_stopping_patience=1,
                      early_stopping_min_delta=10.)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            stage=Path(tmp)/'stage';data=ProbeDataset()
            result=_train(stage,data,data,'fusion',settings,{})
            self.assertEqual(result['completed_epochs'],2);self.assertTrue(result['stopped_early'])
            attempt=stage/'train-001';last=load_checkpoint(attempt/'last.pt')
            stamp=(attempt/'last.pt').stat().st_mtime_ns
            for filename in ('result.json','best.pt','best.pt.sha256.json'):(attempt/filename).unlink()
            with patch('training.visual.suite.train_probe',side_effect=AssertionError('terminal stage must not retrain')):
                recovered=_train(stage,data,data,'fusion',settings,{})
                again=_train(stage,data,data,'fusion',settings,{})
            self.assertEqual(recovered,again);self.assertEqual(recovered['completed_epochs'],2)
            self.assertTrue(recovered['stopped_early']);self.assertEqual((attempt/'last.pt').stat().st_mtime_ns,stamp)
            best=load_checkpoint(attempt/'best.pt')
            for key,value in best['model_state'].items():
                torch.testing.assert_close(value,last['best_model_state'][key],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
