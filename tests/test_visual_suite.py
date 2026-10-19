"""End-to-end three-way feature experiment and fixed-cohort reporting."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_visual_data import prepared_fixture,ROOT


class VisualSuiteTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_unavailable_positive_denominator_and_fpr_ties(self):
        from training.visual.report import metrics,operating_point
        rows=[dict(label_known=True,target=y,available=a,logit=s) for y,a,s in
              [(1,True,1.),(1,False,None),(0,True,1.),(0,True,-1.)]]
        report=metrics(rows)
        self.assertEqual(report['positive'],2)
        self.assertEqual(report['all_positive_detection_rate'],.5)
        self.assertEqual(report['unavailable_positive'],1)
        point=operating_point(rows,0.)
        self.assertEqual(point['tp'],0) # tied positive cannot be taken without FP
        point=operating_point(rows,.5)
        self.assertEqual(point['tp'],1)

    def test_three_arms_resume_and_provenance(self):
        from training.visual.suite import run_suite
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp);prepared_fixture(root)
            settings=dict(epochs=2,batch_size=3,seed=42,learning_rate=.001,weight_decay=.0001,
                          gradient_clip_norm=1.,hidden_dim=8,dropout=.1,backend='cpu',device_index=0)
            report=run_suite(root/'prepared',root/'suite',settings=settings)
            self.assertEqual(report['status'],'complete')
            self.assertEqual(len(report['comparisons']),5)
            baseline=report['comparisons'][0]
            for row in report['comparisons']:
                self.assertEqual(row['window_label_hash'],baseline['window_label_hash'])
                self.assertEqual(row['overall']['positive'],baseline['overall']['positive'])
                self.assertEqual(row['baseline_h_ineligible']['positive'],baseline['baseline_h_ineligible']['positive'])
            stamps={str(p):p.stat().st_mtime_ns for p in (root/'suite').rglob('*.pt')}
            again=run_suite(root/'prepared',root/'suite',settings=settings,continue_run=True)
            self.assertEqual(report,again)
            self.assertEqual(stamps,{str(p):p.stat().st_mtime_ns for p in (root/'suite').rglob('*.pt')})
            changed=dict(settings,seed=43)
            with self.assertRaisesRegex(ValueError,'identity|configuration'):
                run_suite(root/'prepared',root/'suite',settings=changed,continue_run=True)

    def test_resume_after_interrupted_checkpoint_publication(self):
        from training.visual.suite import run_suite
        from training.visual.train import train_probe
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp);prepared_fixture(root)
            settings=dict(epochs=1,batch_size=3,seed=42,learning_rate=.001,weight_decay=.0001,
                          gradient_clip_norm=1.,hidden_dim=8,dropout=.1,backend='cpu',device_index=0)
            def interrupt(train,val,mode,config,output,provenance,resume=None):
                output.mkdir();(output/'last.pt').write_bytes(b'partial checkpoint')
                raise RuntimeError('simulated publication interruption')
            with patch('training.visual.suite.train_probe',side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError,'interruption'):
                    run_suite(root/'prepared',root/'suite',settings=settings)
            report=run_suite(root/'prepared',root/'suite',settings=settings,continue_run=True)
            self.assertEqual(report['status'],'complete')
            self.assertEqual((root/'suite'/'B_rgb'/'train-001'/'last.pt').read_bytes(),b'partial checkpoint')
            self.assertTrue((root/'suite'/'B_rgb'/'train-002'/'result.json').is_file())


if __name__ == '__main__':unittest.main()
