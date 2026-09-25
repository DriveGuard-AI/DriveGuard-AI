"""Orchestration guards and paired-window accounting for warm-start experiments."""
import importlib.util
import contextlib
from copy import deepcopy
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class HeadAdaptationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('training.visual.head_adaptation'))
        from training.visual import head_adaptation
        return head_adaptation

    def rows(self):
        return [dict(record_id='record', start_slot=i*25, end_slot=i*25+99,
                     subject_id='7' if i<3 else '14', target=target, label_known=known,
                     available=True, logit=score)
                for i,(target,known,score) in enumerate([(1,True,-1),(1,True,1),
                                                       (0,True,-1),(0,True,1),(0,False,-1)])]

    def test_paired_changes_preserve_lost_positives_and_ignore_unknown(self):
        module = self.module()
        before = self.rows()
        after = [dict(r) for r in before]
        after[0]['logit'] = 1
        after[1]['available'] = False
        after[1]['logit'] = None
        after[2]['logit'] = 1
        after[3]['logit'] = -1
        after[4]['logit'] = 1
        result = module.paired_changes(after,before)
        self.assertEqual(result['overall'], dict(recovered_positive=1,lost_positive=1,
                                                new_false_positive=1,removed_false_positive=1))
        self.assertEqual(result['by_subject']['7']['lost_positive'],1)
        after[0]['target'] = 0
        with self.assertRaisesRegex(ValueError,'cohort'): module.paired_changes(after,before)

    def test_sparse_replay_requires_masks_labels_and_fixed_predictions(self):
        module = self.module()
        expected = self.rows()
        actual = [dict(r,logit=r['logit']+1e-6) for r in expected]
        report = module.check_sparse_replay(actual,expected)
        self.assertLess(report['max_abs_logit_difference'],1e-4)
        actual[0]['logit'] = .1
        with self.assertRaisesRegex(ValueError,'replay'): module.check_sparse_replay(actual,expected)
        actual = [dict(r) for r in expected]
        actual[1]['available'] = False
        with self.assertRaisesRegex(ValueError,'replay'): module.check_sparse_replay(actual,expected)

    def test_directions_change_only_frame_count_and_encoder_training(self):
        module = self.module()
        self.assertEqual(module.ARMS, {'A_frozen8':(8,False),'B_tune8':(8,True),
                                       'C_frozen32':(32,False),'D_tune32':(32,True)})


class AdaptationReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tests.test_fatigue_head_stability import HeadStabilityTests
        from training.visual.fatigue_head_stability import run
        cls.fixture = HeadStabilityTests
        cls.fixture.setUpClass()
        cls.root = cls.fixture.root
        with contextlib.redirect_stdout(io.StringIO()):
            cls.report = run(spatial_report=cls.fixture.path,output_dir=cls.root/'adaptation-reference',seeds=[42])
        cls.path = cls.root/'adaptation-reference/comparison.json'
        cls.encoder = SimpleNamespace(model=torch.nn.Linear(2,2))

    @classmethod
    def tearDownClass(cls): cls.fixture.tearDownClass()

    def test_actual_selected_checkpoint_is_reproduced(self):
        from training.visual.head_adaptation import _inputs
        with patch('training.visual.encoder.FrozenEncoder',return_value=self.encoder):
            with _inputs(self.path,'C_regularized-seed42-best',self.root/'local-weights','cpu',0) as data:
                self.assertEqual(data['reference']['arm'],'C_regularized')
                self.assertGreater(len(data['rows']),0)
                self.assertEqual(data['payload']['provenance'],self.report['provenance'])

    def test_selected_reference_changes_rejected(self):
        from training.visual.head_adaptation import _inputs
        from training.common.checkpoint import atomic_json
        for key in ('checkpoint_sha256','overall','epoch'):
            report = deepcopy(self.report)
            row = next(r for r in report['comparisons'] if r['variant']=='C_regularized-seed42-best')
            if key=='overall': row[key]['tp']+=1
            elif key=='epoch': row[key]+=1
            else: row[key]='changed'
            path = self.path.parent/'changed.json'
            atomic_json(path,report)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'reference'):
                with _inputs(path,'C_regularized-seed42-best',self.root/'local-weights','cpu',0): pass


if __name__=='__main__': unittest.main()
