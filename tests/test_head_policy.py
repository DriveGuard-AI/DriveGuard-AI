"""Policy experiments must preserve evidence masks and checkpoint semantics."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from training.common.config import validate_config
from training.common.model_factory import build_model
from tests import test_gaze_head as head_tests
from tests.test_gaze_training import gaze_fixture

ROOT = Path(__file__).resolve().parents[1]


class HeadPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def config(self, ratio=.5, gap=0):
        config = head_tests.HeadBranchTests().config()
        config['model'].update(kind='relation_gaze_head_policy_v1',
                               head_policy=dict(min_valid_ratio=ratio, max_gap_steps=gap))
        return config

    def test_policy_validation(self):
        validate_config(self.config())
        for ratio, gap in [(0, 0), (1.1, 0), (True, 0), (.5, -1), (.5, True), (.5, 100), (.5, 1.5)]:
            with self.subTest(ratio=ratio, gap=gap), self.assertRaises(ValueError):
                validate_config(self.config(ratio, gap))
        config = self.config()
        config['model']['kind'] = 'relation_gaze_head_v1'
        with self.assertRaises(ValueError): validate_config(config)
        config = self.config()
        del config['model']['head_policy']
        with self.assertRaises(ValueError): validate_config(config)

    def test_experiment_configs_only_change_policy_and_identity(self):
        from training.gaze.config import load_config
        from training.gaze.head_coverage import POLICIES
        baseline = load_config(ROOT/'configs/training/relation_gaze_head.json')
        for name in ('coverage25', 'gap80'):
            candidate = load_config(ROOT/f'configs/training/relation_gaze_head_{name}.json')
            self.assertEqual(candidate['model'].pop('head_policy'), POLICIES[name])
            self.assertEqual(candidate['model']['kind'], 'relation_gaze_head_policy_v1')
            candidate['model']['kind'] = baseline['model']['kind']
            candidate['experiment'] = baseline['experiment']
            self.assertEqual(candidate, baseline)

    def test_coverage_gap_boundaries_numpy_torch_and_no_fill(self):
        from training.gaze.head_evidence import head_statistics, behavior_availability
        for ratio, gap in [(.25, 0), (.5, 2)]:
            config = self.config(ratio, gap)
            model = build_model(config).eval()
            inputs = head_tests.HeadBranchTests().inputs(6)
            masks = inputs['x'][:, :, 28:31]
            masks.zero_()
            count = int(100 * ratio)
            masks[0, -count:] = 1
            masks[1, -(count-1):] = 1
            end = 100-gap
            masks[2, end-count:end] = 1
            masks[3, end-count-1:end-1] = 1
            masks[5, :, :2] = 1  # No simultaneous three-axis evidence.
            before = inputs['x'].clone()
            out = model(**inputs)
            self.assertEqual(out['head_available'].tolist(), [True, False, True, False, False, False])
            stats = head_statistics(inputs['x'].numpy(), config['model']['head_policy'])
            np.testing.assert_array_equal(stats['eligible'], out['head_available'].numpy())
            np.testing.assert_array_equal(stats['gap_steps'], out['head_gap_steps'].numpy())
            np.testing.assert_array_equal(behavior_availability(
                {k: v.numpy() for k, v in inputs.items()}, config['model']['kind'], config['model']['head_policy']),
                out['task_available'].numpy())
            torch.testing.assert_close(inputs['x'], before, rtol=0, atol=0)
            self.assertEqual(stats['gap_steps'][4], 100)
            changed = deepcopy(inputs)
            changed['x'][:, :, 5:8].masked_fill_(~masks.bool(), 999)
            other = model(**changed)
            torch.testing.assert_close(out['behavior_logits'], other['behavior_logits'], rtol=0, atol=0)

    def test_high_precision_threshold_has_identical_integer_boundary(self):
        from training.gaze.head_evidence import head_statistics
        for ratio, expected in [(.07, True), (.0700000001, False)]:
            config = self.config(ratio, 0)
            model = build_model(config).eval()
            inputs = head_tests.HeadBranchTests().inputs()
            inputs['x'][:, :93, 28:31] = 0
            stats = head_statistics(inputs['x'].numpy(), config['model']['head_policy'])
            self.assertEqual(bool(stats['eligible'][0]), expected)
            self.assertEqual(model(**inputs)['head_available'].item(), expected)

    def test_strict_policy_is_exact_legacy_forward(self):
        torch.manual_seed(34)
        legacy = build_model(head_tests.HeadBranchTests().config()).eval()
        policy = build_model(self.config()).eval()
        policy.load_state_dict(legacy.state_dict(), strict=True)
        inputs = head_tests.HeadBranchTests().inputs(4)
        inputs['eligible'][:, 1] = True
        inputs['quality'].uniform_()
        inputs['x'][1, :51, 28:31] = 0
        inputs['x'][2, -1, 28:31] = 0
        a, b = legacy(**inputs), policy(**inputs)
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_gap_policy_is_causal_and_rejects_old_evidence(self):
        model = build_model(self.config(.5, 2)).eval()
        inputs = head_tests.HeadBranchTests().inputs()
        inputs['x'][:, -2:, 28:31] = 0
        out = model(**inputs)
        self.assertTrue(out['head_available'].item())
        self.assertFalse(out['gaze_available'].item())
        encoded = model.encode_head(inputs['x'])
        inputs['x'][:, 60:, 5:8] += 4
        torch.testing.assert_close(encoded[:, :, :60], model.encode_head(inputs['x'])[:, :, :60], rtol=0, atol=0)
        inputs['x'][:, -3:, 28:31] = 0
        self.assertFalse(model(**inputs)['head_available'].item())

    def test_policy_training_resume_predict_diagnose_and_coverage(self):
        from training.gaze.train import run_training
        from training.gaze.predict import JointPredictor
        from training.gaze.diagnostics import diagnose
        from training.gaze.head_coverage import audit_coverage
        from training.common.checkpoint import load_checkpoint
        config = self.config(.5, 2)
        config['model']['dropout'] = .1
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root = Path(tmp)
            manifest_path, manifest = gaze_fixture(root/'data', config)
            masks = np.zeros((300, 23), np.uint8)
            masks[:, 5:8] = 1
            for end in range(100, 301, 25): masks[end-2:end, 5:8] = 0
            np.save(root/'data/body_masks.npy', masks)
            body = dict(manifest['records'][1], record_id='body', view='body',
                        subset='distraction', masks='body_masks.npy')
            for name in ('values', 'targets', 'label_masks', 'expert_age_ms', 'gaze_targets', 'gaze_label_masks'):
                np.save(root/'data'/('body_'+name+'.npy'), np.load(root/'data'/body[name]))
                body[name] = 'body_'+name+'.npy'
            manifest['records'].append(body)
            manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
            config['runtime'].update(manifest=str(manifest_path), data_root=str(root/'data'), output_dir=str(root/'first'))
            first = run_training(config)
            config['training']['epochs'] = 2
            config['runtime']['output_dir'] = str(root/'resumed')
            resumed = run_training(config, resume=first['last_checkpoint'])
            config['runtime']['output_dir'] = str(root/'full')
            full = run_training(config)
            a, b = [load_checkpoint(r['last_checkpoint']) for r in (resumed, full)]
            torch.testing.assert_close(a['model_state'], b['model_state'], rtol=0, atol=0)
            torch.testing.assert_close(a['optimizer_state'], b['optimizer_state'], rtol=0, atol=0)
            self.assertEqual(a['history'], b['history'])
            predictor = JointPredictor(first['best_checkpoint'])
            prediction = predictor.predict_window(np.load(root/'data/val_values.npy')[:100], masks[:100], np.zeros((100, 3), np.float32))
            self.assertEqual(prediction['distraction_evidence_status'], 'recent_head_supported')
            self.assertEqual(prediction['head_gap_ms'], 80)
            self.assertFalse(prediction['head_current_valid'])
            self.assertTrue(prediction['behavior_available']['distraction'])
            self.assertIsNone(prediction['behavior_probabilities']['fatigue'])
            report = diagnose(first['best_checkpoint'], output_dir=root/'diagnostics', view='body')
            self.assertEqual(report['head_policy'], config['model']['head_policy'])
            self.assertEqual(report['groups']['h_eligibility']['h_eligible']['windows'], 9)
            with (root/'diagnostics/windows.csv').open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(all(r['h_gap_steps']=='2' and r['h_current_valid']=='False' for r in rows))
            audit = audit_coverage(first['best_checkpoint'], output_dir=root/'coverage', view='body')
            self.assertEqual(audit['variants']['strict']['body']['eligible'], 0)
            self.assertEqual(audit['variants']['gap80']['body']['eligible'], 9)
            self.assertEqual(audit['variants']['gap80']['body']['newly_eligible'], 9)
            self.assertEqual(audit['gap_bins']['body']['2'], {'positive': 5, 'negative': 4, 'unknown': 0})
            with self.assertRaises(FileExistsError):
                audit_coverage(first['best_checkpoint'], output_dir=root/'coverage', view='body')
            config['model']['head_policy']['max_gap_steps'] = 1
            config['runtime']['output_dir'] = str(root/'wrong-policy')
            with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                run_training(config, resume=first['last_checkpoint'])


if __name__ == '__main__': unittest.main()
