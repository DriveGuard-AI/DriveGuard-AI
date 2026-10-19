"""Diagnostic output must explain missing evidence without changing evaluation."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from training.common.checkpoint import save_checkpoint
from training.common.config import TASKS
from training.common.model_factory import build_model
from training.gaze.config import load_config
from training.gaze.data import FeatureCorpus
from test_gaze_training import gaze_fixture

ROOT = Path(__file__).resolve().parents[1]


class GazeDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'runs')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = load_config()
        self.config['model'].update(channels=[4], dilations=[1], projection_dim=4, gate_hidden_dim=4)
        self.manifest, manifest = gaze_fixture(self.root / 'data', self.config)
        manifest['records'][1].update(view='body', subset='distraction')
        self.manifest.write_text(json.dumps(manifest), encoding='utf-8')
        masks = np.ones((300, 23), np.uint8)
        # Window 0: head pose present, but F quality below .5. O/HO remain available.
        masks[:100, :8] = 0
        masks[:100, 12] = 0
        masks[:100, 19:] = 0
        masks[:100, 5:8] = 1
        masks[200:] = 0  # Last window has no input; do not emit sigmoid(0)=.5.
        np.save(self.root / 'data/val_masks.npy', masks)
        ages = np.zeros((300, 3), np.float32)
        ages[100:125, 1] = 81
        np.save(self.root / 'data/val_expert_age_ms.npy', ages)
        labels = np.load(self.root / 'data/val_label_masks.npy')
        targets = np.load(self.root / 'data/val_targets.npy')
        labels[124, 3] = 0
        targets[124, 3] = 0
        np.save(self.root / 'data/val_label_masks.npy', labels)
        np.save(self.root / 'data/val_targets.npy', targets)
        self.config['runtime'].update(manifest=str(self.manifest), data_root=str(self.root / 'data'))
        corpus = FeatureCorpus(self.config, self.manifest, self.root / 'data')
        try:
            normalizer = corpus.fit_normalizer()
            self.summary = corpus.summary
        finally:
            corpus.close()
        model = build_model(self.config)
        with torch.no_grad():
            model.heads[3].weight.zero_()
            model.heads[3].bias.fill_(np.log(4.))  # Predict .8 when any evidence exists.
        self.checkpoint = self.root / 'best.pt'
        payload = dict(format_version=1, kind='weights', epoch=15, config=self.config,
                       normalizer=normalizer, data_summary=self.summary, model_state=model.state_dict(),
                       model_kind='relation_gaze_v1', active_tasks=list(TASKS), task_order=list(TASKS),
                       gaze_classes=self.config['data']['gaze_classes'])
        for key in ('feature_schema', 'extractor_profile', 'label_mapping', 'gaze_mapping',
                    'group_profile', 'group_profile_sha256'):
            payload[key] = manifest[key]
        save_checkpoint(self.checkpoint, payload)

    def diagnose(self, name='report', **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('training.gaze.diagnostics'), 'diagnostic tool missing')
        from training.gaze.diagnostics import diagnose
        output = self.root / name
        summary = diagnose(self.checkpoint, output_dir=output, **kwargs)
        with (output / 'windows.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(summary, json.loads((output / 'summary.json').read_text(encoding='utf-8')))
        return summary, rows

    def test_rows_time_masks_and_unknowns(self):
        summary, rows = self.diagnose(batch_size=4)
        self.assertEqual(len(rows), 9)
        first = rows[0]
        self.assertEqual(first['recording_id'], 'val')
        self.assertEqual(float(first['window_start_s']), 0)
        self.assertEqual(float(first['window_end_s']), 3.96)
        self.assertEqual(float(rows[1]['window_start_s']), 1.)
        self.assertEqual(float(first['raw_head_pose_valid_ratio']), 1.)
        self.assertEqual(float(first['effective_head_pose_valid_ratio']), 1.)
        self.assertEqual(first['f_eligible'], 'False')
        self.assertIn('low_valid_ratio', first['f_block_reasons'])
        self.assertEqual(first['outcome'], 'TP')
        self.assertAlmostEqual(float(first['probability']), .8, places=6)
        self.assertEqual(rows[1]['target'], '')
        self.assertEqual(rows[1]['outcome'], 'unknown')
        self.assertGreater(float(rows[1]['raw_head_pose_valid_ratio']), float(rows[1]['effective_head_pose_valid_ratio']))
        self.assertEqual(rows[-1]['outcome'], 'unavailable')
        self.assertEqual(rows[-1]['probability'], '')
        self.assertEqual(rows[-1]['prediction'], '')
        self.assertEqual(summary['checkpoint_epoch'], 15)
        self.assertEqual(summary['overall']['windows'], 9)
        self.assertEqual(summary['overall']['unknown'], 1)
        self.assertEqual(summary['overall']['known_unavailable'], 4)
        self.assertEqual(summary['overall']['evaluated'], 4)
        self.assertIn('f_ineligible', summary['groups']['f_eligibility'])

    def test_report_matches_training_metric_and_batch_sizes(self):
        from torch.utils.data import DataLoader
        from training.common.model_factory import load_model
        from training.gaze.train import run_epoch
        a, rows_a = self.diagnose('one', batch_size=1)
        b, rows_b = self.diagnose('many', batch_size=7)
        self.assertEqual(a['overall'], b['overall'])
        self.assertEqual([r['outcome'] for r in rows_a], [r['outcome'] for r in rows_b])
        model, payload = load_model(self.checkpoint)
        corpus = FeatureCorpus(self.config, self.manifest, self.root / 'data')
        try:
            report = run_epoch(model, DataLoader(corpus.dataset('val', payload['normalizer']), batch_size=3), torch.device('cpu'))
        finally:
            corpus.close()
        for key in ('precision', 'recall', 'f1'):
            self.assertAlmostEqual(a['overall'][key], report['metrics']['per_task']['distraction'][key])
        self.assertEqual(a['thresholds'][-1]['tp'], 0)
        self.assertEqual(a['thresholds'][-1]['fp'], 0)

    def test_changed_data_and_existing_output_rejected(self):
        self.diagnose()
        from training.gaze.diagnostics import diagnose
        with self.assertRaises(FileExistsError):
            diagnose(self.checkpoint, output_dir=self.root / 'report')
        values = np.load(self.root / 'data/val_values.npy')
        values[0, 0] += 1
        np.save(self.root / 'data/val_values.npy', values)
        with self.assertRaisesRegex(ValueError, 'fingerprint|provenance'):
            diagnose(self.checkpoint, output_dir=self.root / 'changed')
        self.assertFalse((self.root / 'changed').exists())

    def test_empty_selection_and_invalid_options_rejected(self):
        self.assertIsNotNone(importlib.util.find_spec('training.gaze.diagnostics'), 'diagnostic tool missing')
        from training.gaze.diagnostics import diagnose
        for options in ({'view': 'face'}, {'batch_size': 0}, {'threshold': float('nan')}, {'threshold': 2.}):
            with self.assertRaises(ValueError):
                diagnose(self.checkpoint, output_dir=self.root / 'invalid', **options)
        self.assertFalse((self.root / 'invalid').exists())

    def test_cli_uses_saved_paths_and_exports_comparison_views(self):
        result = subprocess.run([
            sys.executable, '-m', 'training.gaze.diagnostics', '--checkpoint', str(self.checkpoint),
            '--output-dir', str(self.root / 'cli'), '--view', 'all', '--subset', 'all',
            '--backend', 'cpu', '--batch-size', '4',
        ], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output['checkpoint_epoch'], 15)
        self.assertEqual(output['overall']['windows'], 9)
        self.assertTrue(Path(output['summary']).is_file())
        self.assertIn('9/9', result.stderr)

    def test_original_files_unchanged(self):
        from training.common.data import file_sha256
        paths = [self.checkpoint, self.checkpoint.with_suffix('.pt.sha256.json'), self.manifest,
                 *sorted((self.root / 'data').glob('*.npy'))]
        before = {str(path): file_sha256(path) for path in paths}
        self.diagnose()
        self.assertEqual(before, {str(path): file_sha256(path) for path in paths})

    def test_near_zero_negative_logit_matches_training_boundary(self):
        from training.common.checkpoint import load_checkpoint
        payload = load_checkpoint(self.checkpoint)
        payload['model_state']['heads.3.bias'].fill_(-1e-8)
        save_checkpoint(self.checkpoint, payload)
        summary, rows = self.diagnose()
        self.assertEqual(float(rows[0]['probability']), .5)
        self.assertEqual(rows[0]['outcome'], 'FN')
        self.assertEqual(rows[0]['prediction'], '0')
        self.assertEqual(summary['overall']['tp'], 0)
        self.assertEqual(summary['thresholds'][4]['tp'], 0)


if __name__ == '__main__':
    unittest.main()
