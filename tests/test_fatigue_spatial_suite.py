"""Spatial comparisons reproduce their references and keep a fixed fatigue cohort."""
import contextlib
from copy import deepcopy
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from training.common.checkpoint import atomic_json, load_checkpoint
from training.common.config import read_json
from training.common.data import file_sha256
from training.visual.suite import DEFAULTS


class FatigueSpatialSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_visual_fatigue import fatigue_fixture
        from tests.test_visual_spatial import head_cache_fixture
        from training.visual.fatigue import run
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'runs')
        cls.root = Path(cls.temp.name)
        cls.settings = dict(DEFAULTS, hidden_dim=4, epochs=1, batch_size=3,
                            early_stopping_patience=4)
        with contextlib.redirect_stdout(io.StringIO()):
            fatigue_fixture(cls.root)
            cls.reference = run(reference_prepared_dir=cls.root/'prepared',
                                output_dir=cls.root/'old', settings=cls.settings, seeds=[42])
        cls.report_path = cls.root/'old/comparison.json'
        prepared = read_json(Path(cls.reference['prepared_dir'])/'prepared.json')
        manifest = Path(prepared['manifest'])
        cls.head = head_cache_fixture(cls.root/'head', manifest, read_json(manifest)['records'])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_end_to_end_and_exact_completed_continuation(self):
        from training.visual.fatigue_spatial_suite import run
        kwargs = dict(fatigue_report=self.report_path, head_cache=self.head,
                      output_dir=self.root/'spatial', settings=self.settings, seeds=[42])
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(**kwargs)
            with patch('training.visual.suite.train_probe', side_effect=AssertionError('must reuse completed runs')):
                continued = run(**kwargs, continue_run=True)
        self.assertEqual(result, continued)
        self.assertEqual(result['schema_version'], 'driveguard-fatigue-spatial-comparison-v1')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['training_runs'], 3)
        self.assertEqual(len(result['comparisons']), 8)
        self.assertEqual(len({r['window_label_hash'] for r in result['comparisons']}), 1)
        self.assertEqual(set(result['aggregate_best']), {'A_whole','B_head','C_whole_head'})
        self.assertEqual(set(result['paired_best_deltas']['42']), {'B_minus_A','C_minus_A'})
        old = next(r for r in result['comparisons'] if r['variant'] == 'B_rgb-seed42-best')
        published = next(r for r in self.reference['comparisons'] if r['variant'] == old['variant'])
        self.assertEqual(old['overall'], published['overall'])
        params = []
        for row in result['comparisons']:
            if row['arm'] not in result['aggregate_best']:
                continue
            payload = load_checkpoint(row['checkpoint'])
            self.assertEqual(payload['settings']['positive_weight'], 1.)
            self.assertEqual(payload['model_params']['visual_dim'], 1024)
            params.append(sum(t.numel() for t in payload['model_state'].values()))
            self.assertIn('fp', row['by_subject'][next(iter(row['by_subject']))])
        self.assertEqual(len(set(params)), 1)
        for delta in result['paired_best_deltas']['42'].values():
            self.assertIn('fp', delta['overall'])
            self.assertEqual(set(delta['by_subject']), set(old['by_subject']))
        with self.assertRaisesRegex(ValueError, 'configuration|identity'):
            run(**dict(kwargs, settings=dict(self.settings, dropout=.2)), continue_run=True)
        with self.assertRaises(FileExistsError):
            run(**kwargs)
        changed = dict(self.reference, notes=['changed immutable input report'])
        changed_path = self.root/'changed-reference.json'
        atomic_json(changed_path, changed)
        with self.assertRaisesRegex(ValueError, 'configuration|identity'):
            run(**dict(kwargs, fatigue_report=changed_path), continue_run=True)

    def test_cross_cohort_preparation_is_rejected(self):
        from training.visual.fatigue_spatial_suite import run
        alternative = self.root/'alternative-prepared'
        shutil.copytree(self.reference['prepared_dir'], alternative)
        preparation = read_json(alternative/'prepared.json')
        val = preparation['splits']['val']
        path = alternative/val['rows']
        rows = read_json(path)
        row = next(r for r in rows if r['label_known'])
        row['target'] = 1-row['target']
        atomic_json(path, rows)
        val['rows_sha256'] = file_sha256(path)
        atomic_json(alternative/'prepared.json', preparation)
        reference = dict(self.reference, prepared_dir=str(alternative),
                         prepared_sha256=file_sha256(alternative/'prepared.json'))
        path = self.root/'cross-cohort-report.json'
        atomic_json(path, reference)
        with self.assertRaisesRegex(ValueError, 'reference|cohort'):
            run(fatigue_report=path, head_cache=self.head, output_dir=self.root/'cross-cohort-output',
                settings=self.settings, seeds=[42])
        self.assertFalse((self.root/'cross-cohort-output').exists())

    def test_reference_replay_keeps_original_batch_size(self):
        from training.visual.data import PreparedData
        from training.visual.fatigue_spatial_suite import _reference
        published = next(r for r in self.reference['comparisons'] if r['variant']=='B_rgb-seed42-best')
        base = PreparedData(self.reference['prepared_dir'])
        try:
            replayed, _ = _reference(published,base.dataset('val'),dict(self.settings,batch_size=1))
            self.assertEqual(replayed['validation_fpr03'],published['validation_fpr03'])
        finally:base.close()

    def test_reference_metrics_cohort_and_checksum_are_reproduced(self):
        from training.visual.fatigue_spatial_suite import run
        for field in ('overall', 'window_label_hash', 'checkpoint_sha256', 'by_subject', 'seed'):
            report = deepcopy(self.reference)
            row = next(r for r in report['comparisons'] if r['variant'] == 'B_rgb-seed42-best')
            if field == 'overall': row[field]['tp'] += 1
            elif field == 'by_subject': row[field][next(iter(row[field]))]['fp'] += 1
            elif field == 'seed': row[field] = 43
            else: row[field] = 'changed'
            path = self.root/f'tampered-{field}.json'
            atomic_json(path, report)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'reference|Reference|checksum|cohort|seed'):
                run(fatigue_report=path, head_cache=self.head, output_dir=self.root/f'bad-{field}',
                    settings=self.settings, seeds=[42])
            self.assertFalse((self.root/f'bad-{field}').exists())

    def test_reject_weight_seed_schema_and_source_overlap(self):
        from training.visual.fatigue_spatial_suite import run
        kwargs = dict(fatigue_report=self.report_path, head_cache=self.head,
                      output_dir=self.root/'invalid', settings=self.settings, seeds=[42])
        for seeds in ([], [42,42], [True], [43]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                run(**dict(kwargs, seeds=seeds))
        with self.assertRaisesRegex(ValueError, 'weight|BCE'):
            run(**dict(kwargs, settings=dict(self.settings, positive_weight=2.)))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            run(**dict(kwargs, output_dir=self.head/'output'))
        for key,value in [('task','distraction'),('split','test'),('status','incomplete'),('schema_version','bad')]:
            report = dict(self.reference, **{key:value})
            path = self.root/'invalid-report.json'
            atomic_json(path, report)
            with self.subTest(key=key), self.assertRaises(ValueError):
                run(**dict(kwargs, fatigue_report=path))


if __name__ == '__main__': unittest.main()
