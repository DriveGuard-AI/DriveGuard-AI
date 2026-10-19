"""Head stability comparisons keep the published cohort and isolate interventions."""
import contextlib
from copy import deepcopy
import csv
import io
import unittest
from unittest.mock import patch

from training.common.checkpoint import atomic_json, load_checkpoint
from training.common.config import read_json


class HeadStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.test_fatigue_spatial_suite import FatigueSpatialSuiteTests
        from training.visual.fatigue_spatial_suite import run
        from training.visual.fatigue import run as prepare_reference
        cls.fixture = FatigueSpatialSuiteTests
        cls.fixture.setUpClass()
        cls.root = cls.fixture.root
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_reference(reference_prepared_dir=cls.root/'prepared', output_dir=cls.root/'old-two-seeds',
                              settings=cls.fixture.settings, seeds=[42,43])
            cls.reference = run(fatigue_report=cls.root/'old-two-seeds/comparison.json', head_cache=cls.fixture.head,
                                output_dir=cls.root/'spatial-reference',
                                settings=cls.fixture.settings, seeds=[42,43])
        cls.path = cls.root/'spatial-reference/comparison.json'

    @classmethod
    def tearDownClass(cls):
        cls.fixture.tearDownClass()

    def test_real_training_report_and_continuation(self):
        from training.visual.fatigue_head_stability import run
        kwargs = dict(spatial_report=self.path, output_dir=self.root/'stability')
        with contextlib.redirect_stdout(io.StringIO()):
            report = run(**kwargs)
            with patch('training.visual.suite.train_probe', side_effect=AssertionError('must reuse')):
                self.assertEqual(report, run(**kwargs, continue_run=True))
        self.assertEqual(report['training_runs'], 8)
        self.assertEqual(len(report['comparisons']), 18)
        self.assertEqual(len({r['window_label_hash'] for r in report['comparisons']}), 1)
        self.assertEqual(len(report['aggregate_best']), 4)
        self.assertEqual(len(report['paired_best_deltas']['42']), 3)
        self.assertEqual(len(report['last_minus_best']), 8)
        self.assertEqual(report['aggregate_best']['B_slow']['overall']['tp']['count'], 2)
        rows = {r['arm']: r for r in report['comparisons'] if r['checkpoint_kind']=='best' and r['seed']==42}
        original = load_checkpoint(rows['A_control']['checkpoint'])['settings']
        for arm, keys in [('B_slow', {'learning_rate'}),
                          ('C_regularized', {'dropout','weight_decay'}),
                          ('D_slow_regularized', {'learning_rate','dropout','weight_decay'})]:
            actual = load_checkpoint(rows[arm]['checkpoint'])['settings']
            self.assertEqual({k for k in original if original[k] != actual[k]}, keys)
        source = next(r for r in self.reference['comparisons'] if r['variant']=='B_head-seed42-best')
        replay = report['references'][0]
        self.assertEqual(replay, source)
        self.assertEqual(rows['A_control']['overall'], source['overall'])
        with (self.root/'stability/by-subject.csv').open(encoding='utf-8', newline='') as handle:
            flat = list(csv.DictReader(handle))
        self.assertEqual(len(flat), sum(len(r['by_subject']) for r in report['comparisons']))
        self.assertTrue(all('fp' in r and 'fn_total' in r for r in flat))
        with self.assertRaises(FileExistsError): run(**kwargs)
        with self.assertRaisesRegex(ValueError, 'identity|configuration'):
            run(**kwargs, epochs=2, continue_run=True)

    def test_reject_reference_tampering_before_output(self):
        from training.visual.fatigue_head_stability import run
        for field in ('overall','checkpoint_sha256','epoch','by_subject','prediction_rows_hash'):
            reference = deepcopy(self.reference)
            row = next(r for r in reference['comparisons'] if r['variant']=='B_head-seed42-best')
            if field=='overall': row[field]['tp'] += 1
            elif field=='by_subject': row[field][next(iter(row[field]))]['fp'] += 1
            elif field=='epoch': row[field] += 1
            else: row[field] = 'tampered'
            path = self.path.parent/f'bad-stability-{field}.json'
            atomic_json(path, reference)
            output = self.root/f'reject-stability-{field}'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'Reference'):
                run(spatial_report=path, output_dir=output, seeds=[42])
            self.assertFalse(output.exists())

    def test_reject_wrong_inputs_and_overlap(self):
        from training.visual.fatigue_head_stability import run
        kwargs = dict(spatial_report=self.path, output_dir=self.root/'invalid-stability', seeds=[42])
        for seeds in ([], [42,42], [True], [44]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                run(**dict(kwargs, seeds=seeds))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            run(**dict(kwargs, output_dir=self.fixture.head/'new'))
        for key,value in [('status','partial'),('task','distraction'),('split','test'),
                          ('head_cache_sha256','bad'),('prepared_sha256','bad')]:
            path = self.path.parent/'bad-stability-input.json'
            atomic_json(path, dict(self.reference, **{key:value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                run(**dict(kwargs, spatial_report=path))
        self.assertFalse(kwargs['output_dir'].exists())

    def test_interruption_reuses_completed_stage(self):
        from training.visual.fatigue_head_stability import run
        from training.visual.suite import _train
        calls = []
        def interrupt(stage, *args):
            calls.append(stage.name)
            if len(calls)==2: raise RuntimeError('simulated interruption')
            return _train(stage, *args)
        output = self.root/'interrupted-stability'
        kwargs = dict(spatial_report=self.path, output_dir=output, seeds=[42])
        with contextlib.redirect_stdout(io.StringIO()):
            with patch('training.visual.fatigue_head_stability._train', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated'): run(**kwargs)
            with patch('training.visual.fatigue_head_stability._train', wraps=_train) as training:
                report = run(**kwargs, continue_run=True)
        self.assertEqual(report['status'], 'complete')
        self.assertEqual(training.call_count, 4)
        self.assertEqual(len(list((output/'A_control-seed42').glob('train-*'))), 1)

    def test_settings_change_is_not_accepted_as_reference(self):
        from training.visual.fatigue_head_stability import run
        path = self.path.parent/'bad-stability-settings.json'
        reference = deepcopy(self.reference)
        reference['settings']['learning_rate'] *= 2
        atomic_json(path, reference)
        with self.assertRaisesRegex(ValueError, 'settings|identity'):
            run(spatial_report=path, output_dir=self.root/'bad-settings-output', seeds=[42])


if __name__ == '__main__': unittest.main()
