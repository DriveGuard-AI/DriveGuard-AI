"""Synthetic evidence diagnostics, never a claim about actual DMD accuracy."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import torch

import test_gaze_diagnostics as fixture
from training.common.checkpoint import load_checkpoint, save_checkpoint
from training.common.data import file_sha256
from training.common.model_factory import build_model

ROOT = Path(__file__).resolve().parents[1]


class FatigueDiagnosticTests(unittest.TestCase):
    def setUp(self):
        fixture.GazeDiagnosticTests.setUp(self)
        # Retain the real corpus fixture's low-F, stale-face, unknown, and empty windows.
        manifest = json.loads(self.manifest.read_text(encoding='utf-8'))
        body = manifest['records'][1]
        body['subset'] = 'drowsiness'
        face = dict(body, record_id='val-face', view='face')
        for key in ('values', 'masks', 'targets', 'label_masks', 'expert_age_ms', 'gaze_targets', 'gaze_label_masks'):
            face[key] = 'face_' + body[key]
            array = np.load(self.root / 'data' / body[key])
            if key == 'masks':
                array[:] = 1
            if key == 'expert_age_ms':
                array[:] = 0
            np.save(self.root / 'data' / face[key], array)
        manifest['records'].append(face)
        self.manifest.write_text(json.dumps(manifest), encoding='utf-8')
        for prefix in ('val', 'face_val'):
            labels = np.load(self.root / f'data/{prefix}_label_masks.npy')
            targets = np.load(self.root / f'data/{prefix}_targets.npy')
            labels[124, 0] = 0
            targets[124, 0] = 0
            np.save(self.root / f'data/{prefix}_label_masks.npy', labels)
            np.save(self.root / f'data/{prefix}_targets.npy', targets)
        self.refresh_checkpoint()

    def refresh_checkpoint(self, policy=None):
        from training.gaze.data import FeatureCorpus
        if policy:
            self.config['model'].update(kind='relation_gaze_head_policy_v1', head_policy=policy)
        corpus = FeatureCorpus(self.config, self.manifest, self.root / 'data')
        try:
            normalizer, summary = corpus.fit_normalizer(), corpus.summary
        finally:
            corpus.close()
        payload = load_checkpoint(self.checkpoint)
        model = build_model(self.config)
        with torch.no_grad():
            model.heads[0].weight.zero_()
            model.heads[0].bias.fill_(-1e-8)
        payload.update(config=self.config, normalizer=normalizer, data_summary=summary,
                       model_kind=self.config['model']['kind'], model_state=model.state_dict())
        save_checkpoint(self.checkpoint, payload)

    def run_report(self, name='report', **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('training.gaze.fatigue_diagnostics'), 'fatigue diagnostic missing')
        from training.gaze.fatigue_diagnostics import diagnose
        result = diagnose(self.checkpoint, output_dir=self.root / name, **kwargs)
        with (self.root / name / 'windows.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(result, json.loads((self.root / name / 'summary.json').read_text(encoding='utf-8')))
        return result, rows

    def test_real_masks_unknowns_scores_and_conditional_groups(self):
        summary, rows = self.run_report(batch_size=3)
        self.assertEqual(len(rows), 18)
        body = [r for r in rows if r['view'] == 'body']
        self.assertEqual(body[0]['outcome'], 'FN')
        self.assertEqual(body[0]['prediction'], '0')
        self.assertEqual(float(body[0]['probability']), .5)  # logit decides the boundary
        self.assertEqual(body[0]['f_eligible'], 'False')
        self.assertEqual(float(body[0]['f_fatigue_weight']), 0.)
        self.assertEqual(float(body[0]['effective_ear_left_valid_ratio']), 0.)
        self.assertEqual(body[1]['outcome'], 'unknown')
        self.assertEqual(body[1]['target'], '')
        self.assertEqual(body[-1]['probability'], '')
        self.assertEqual(body[-1]['outcome'], 'unavailable')
        self.assertGreater(float(body[1]['raw_ear_left_valid_ratio']), float(body[1]['effective_ear_left_valid_ratio']))
        self.assertEqual(float(body[1]['effective_perclos_4s_valid_ratio']), 0.)
        self.assertEqual(summary['overall']['unknown'], 2)
        self.assertEqual(summary['overall']['tp'], 0)
        self.assertEqual(summary['groups']['view_label']['body:positive']['positive'], 5)
        face = summary['groups']['view_label']['face:positive']
        self.assertEqual(face['evaluated'], 5)
        self.assertEqual(face['evidence']['f_eligible_below_zero'], 5)
        self.assertEqual(face['scores']['logit']['count'], 5)
        self.assertLess(face['scores']['logit']['max'], 0.)
        self.assertIn('subject_id', summary['groups'])
        self.assertIn('subset', summary['groups'])

    def test_batch_invariance_and_training_metrics(self):
        a, _ = self.run_report('one', batch_size=1)
        b, _ = self.run_report('many', batch_size=7)
        # Batch kernels can differ in the last float32 bit; counts must be exact.
        for key in a['overall']:
            if key != 'scores':
                self.assertEqual(a['overall'][key], b['overall'][key])
        for score in a['overall']['scores']:
            for key, value in a['overall']['scores'][score].items():
                self.assertAlmostEqual(value, b['overall']['scores'][score][key], places=6)
        from torch.utils.data import DataLoader
        from training.common.model_factory import load_model
        from training.gaze.data import FeatureCorpus
        from training.gaze.train import run_epoch
        model, payload = load_model(self.checkpoint)
        corpus = FeatureCorpus(self.config, self.manifest, self.root / 'data')
        try:
            report = run_epoch(model, DataLoader(corpus.dataset('val', payload['normalizer']), batch_size=3), torch.device('cpu'))
        finally:
            corpus.close()
        for key in ('precision', 'recall', 'f1'):
            self.assertEqual(a['overall'][key], report['metrics']['per_task']['fatigue'][key])

    def test_each_checkpoint_own_head_policy(self):
        for name, ratio in (('gap80', .5), ('coverage25', .25)):
            policy = dict(min_valid_ratio=ratio, max_gap_steps=2)
            self.refresh_checkpoint(policy)
            summary, rows = self.run_report(name)
            self.assertEqual(summary['head_policy'], policy)
            self.assertEqual(rows[-1]['view'], 'face')
            self.assertEqual(summary['groups']['view']['body']['known_unavailable'], 4)

    def test_inputs_unchanged_and_bad_fingerprint_rejected(self):
        paths = [self.checkpoint, self.checkpoint.with_suffix('.pt.sha256.json'), self.manifest,
                 *sorted((self.root / 'data').glob('*.npy'))]
        before = {str(p): file_sha256(p) for p in paths}
        self.run_report()
        self.assertEqual(before, {str(p): file_sha256(p) for p in paths})
        from training.gaze.fatigue_diagnostics import diagnose
        with self.assertRaises(FileExistsError):
            diagnose(self.checkpoint, output_dir=self.root / 'report')
        values = np.load(self.root / 'data/val_values.npy')
        values[0, 0] += 1
        np.save(self.root / 'data/val_values.npy', values)
        with self.assertRaisesRegex(ValueError, 'fingerprint|provenance'):
            diagnose(self.checkpoint, output_dir=self.root / 'changed')
        self.assertFalse((self.root / 'changed').exists())

    def test_output_overlap_and_non_validation_rejected(self):
        self.assertIsNotNone(importlib.util.find_spec('training.gaze.fatigue_diagnostics'))
        from training.gaze.fatigue_diagnostics import diagnose
        with self.assertRaisesRegex(ValueError, 'overlap'):
            diagnose(self.checkpoint, output_dir=self.root / 'data/new-report')
        for options in ({'split': 'test'}, {'batch_size': 0}, {'view': 'invalid'}):
            with self.assertRaises(ValueError):
                diagnose(self.checkpoint, output_dir=self.root / 'invalid', **options)

    def test_nonfinite_valid_prediction_never_publishes_summary(self):
        payload = load_checkpoint(self.checkpoint)
        payload['model_state']['heads.0.bias'].fill_(float('nan'))
        save_checkpoint(self.checkpoint, payload)
        self.assertIsNotNone(importlib.util.find_spec('training.gaze.fatigue_diagnostics'))
        from training.gaze.fatigue_diagnostics import diagnose
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            diagnose(self.checkpoint, output_dir=self.root / 'bad')
        self.assertFalse((self.root / 'bad/summary.json').exists())

    def test_cli_default_compares_body_and_face_validation(self):
        result = subprocess.run([sys.executable, '-m', 'training.gaze.fatigue_diagnostics',
                                 '--checkpoint', str(self.checkpoint), '--output-dir', str(self.root / 'cli'),
                                 '--backend', 'cpu', '--batch-size', '4'], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output['overall']['windows'], 18)
        self.assertTrue(Path(output['summary']).is_file())

    def test_input_change_during_evaluation_rejects_complete_report(self):
        from training.gaze.fatigue_diagnostics import diagnose
        def mutate(message):
            if message.startswith('Rechecking'):
                # Manifest is not memory-mapped (Windows locks open .npy files).
                self.manifest.write_text(self.manifest.read_text(encoding='utf-8') + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Input changed'):
            diagnose(self.checkpoint, output_dir=self.root / 'raced', progress=mutate)
        self.assertFalse((self.root / 'raced/summary.json').exists())

    def test_manifest_drowsiness_subset_and_head_only_fatigue_unavailable(self):
        masks = np.load(self.root / 'data/val_masks.npy')
        masks[:100] = 0
        masks[:100, 5:8] = 1
        np.save(self.root / 'data/val_masks.npy', masks)
        self.refresh_checkpoint(dict(min_valid_ratio=.25, max_gap_steps=2))
        summary, rows = self.run_report(subset='drowsiness')
        self.assertEqual(summary['overall']['windows'], 18)
        self.assertEqual(rows[0]['h_eligible'], 'True')
        self.assertEqual(rows[0]['behavior_available'], 'False')
        self.assertEqual(rows[0]['probability'], '')
        self.assertEqual(rows[0]['outcome'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
