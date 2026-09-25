"""Fatigue-specific labels and missing-face recovery contracts, not quality claims."""
from pathlib import Path
from copy import deepcopy
import contextlib
import io
import json
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from tests.test_visual_training import ProbeDataset


def fatigue_fixture(root):
    """Face records keep baseline trainable; body records have no perception observations.

    Body fatigue labels deliberately disagree with distraction and include unknowns.
    """
    from tests.test_gaze_training import gaze_fixture
    from tests.test_visual_data import visual_fixture, ROOT
    from training.gaze.config import load_config
    from training.gaze.train import run_training
    from training.visual.prepare import prepare
    config = load_config(ROOT/'configs/training/relation_gaze_head_coverage25.json')
    config['model'].update(channels=[4], dilations=[1], projection_dim=4, gate_hidden_dim=4,
                           head_policy=dict(min_valid_ratio=.25, max_gap_steps=2))
    config['training'].update(epochs=1, batch_size=3)
    path, manifest = gaze_fixture(root/'features', config)
    originals = list(manifest['records'])
    for original in originals:
        record = deepcopy(original)
        record.update(record_id=original['record_id']+'-body', recording_id=original['recording_id']+'-body', view='body')
        for key in ('values', 'masks', 'targets', 'label_masks', 'expert_age_ms', 'gaze_targets', 'gaze_label_masks'):
            array = np.load(root/'features'/original[key])
            if key == 'masks':
                array[:] = 0
            elif key == 'targets':
                array[:,0] = 1-array[:,3]
                array[124,0] = 0
            elif key == 'label_masks':
                array[124,0] = 0
            record[key] = 'body-'+original[key]
            np.save(root/'features'/record[key], array)
        manifest['records'].append(record)
    path.write_text(json.dumps(manifest), encoding='utf-8')
    config['runtime'].update(manifest=str(path), data_root=str(root/'features'), output_dir=str(root/'baseline'))
    baseline = run_training(config)
    cache = visual_fixture(root/'visual', path, manifest['records'])
    prepare(checkpoint=baseline['best_checkpoint'], visual_cache=cache, output_dir=root/'prepared')
    return baseline, cache


class FatigueVisualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_weight_uses_only_known_rgb_available_training_labels(self):
        from training.visual.fatigue import class_balance
        result = class_balance(ProbeDataset())
        self.assertEqual(result, dict(positive=2, negative=3, positive_weight=1.5))
        empty = ProbeDataset()
        empty.mask[1] = False
        empty.mask[3] = False
        with self.assertRaisesRegex(ValueError, 'positives AND negatives'):
            class_balance(empty)

    def test_balanced_training_resumes_exactly_and_rejects_weight_change(self):
        from training.visual.train import train_probe, _settings
        from training.visual.suite import DEFAULTS
        from training.common.checkpoint import load_checkpoint
        data = ProbeDataset()
        settings = dict(DEFAULTS, hidden_dim=4, epochs=1, batch_size=3, positive_weight=1.5)
        for weight in (0., -1., float('nan'), float('inf'), True):
            with self.assertRaisesRegex(ValueError, 'positive_weight'):
                _settings(dict(settings, positive_weight=weight))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            first = train_probe(data, data, 'visual', settings, root/'first', {})
            settings['epochs'] = 2
            resumed = train_probe(data, data, 'visual', settings, root/'resumed', {}, first['last_checkpoint'])
            full = train_probe(data, data, 'visual', settings, root/'full', {})
            a, b = [load_checkpoint(r['last_checkpoint']) for r in (resumed, full)]
            self.assertEqual(a['history'], b['history'])
            for key in a['model_state']:
                torch.testing.assert_close(a['model_state'][key], b['model_state'][key], atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                train_probe(data, data, 'visual', dict(settings, positive_weight=2.), root/'bad', {}, first['last_checkpoint'])

    def test_weight_changes_positive_gradient_but_not_validation_loss(self):
        from training.visual.train import run_epoch

        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.tensor(0.))

            def forward(self, visual, **kwargs):
                return dict(logits=self.bias.expand(len(visual)))

        loader = DataLoader(ProbeDataset(), batch_size=7)
        model = Constant()
        reference = run_epoch(model, loader, 'cpu')['loss']
        weighted_val = run_epoch(model, loader, 'cpu', positive_weight=3.)['loss']
        self.assertEqual(reference, weighted_val)
        run_epoch(model, loader, 'cpu', torch.optim.SGD(model.parameters(), lr=.1), positive_weight=3.)
        self.assertGreater(model.bias.item(), 0., 'Balanced loss must push against negative-class collapse')

    def test_missing_predictions_remain_false_negatives_and_f_missing_is_fixed(self):
        from training.visual.fatigue import summarize
        rows = [dict(record_id='r', subject_id='s', start_slot=i*25, end_slot=i*25+99,
                     label_known=known, target=target, f_eligible=False,
                     available=available, logit=score)
                for i, (known,target,available,score) in enumerate([
                    (True,1,True,1.), (True,1,False,None), (True,0,True,-1.), (False,0,True,1.)])]
        report = summarize(rows, variant='test')
        self.assertEqual(report['overall']['tp'], 1)
        self.assertEqual(report['overall']['fn_total'], 1)
        self.assertEqual(report['overall']['all_positive_detection_rate'], .5)
        self.assertEqual(report['baseline_f_ineligible']['tp'], 1)
        self.assertEqual(report['overall']['unknown'], 1)

    def test_fatigue_preparation_and_end_to_end_suite(self):
        from training.visual.prepare import prepare
        from training.visual.data import PreparedData
        from training.visual.fatigue import run
        from training.visual.suite import DEFAULTS, run_suite
        root_dir = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root_dir/'runs') as tmp:
            root = Path(tmp)
            baseline, cache = fatigue_fixture(root)
            result = prepare(checkpoint=baseline['best_checkpoint'], visual_cache=cache,
                             output_dir=root/'fatigue', task='fatigue')
            self.assertEqual(result['task'], 'fatigue')
            prepared = PreparedData(root/'fatigue')
            try:
                self.assertEqual(prepared.task, 'fatigue')
                self.assertTrue(all(r['f_eligible'] is False for r in prepared.rows['val']))
                self.assertTrue(all(r['baseline_available'] is False for r in prepared.rows['val']))
                old = PreparedData(root/'prepared')
                try:
                    for row, distraction in zip(prepared.rows['val'], old.rows['val']):
                        if row['label_known']:
                            self.assertEqual(row['target'], 1-distraction['target'])
                        else:
                            self.assertEqual(row['target'], 0)
                    self.assertEqual(sum(not r['label_known'] for r in prepared.rows['val']), 1)
                finally:
                    old.close()
            finally:
                prepared.close()
            with self.assertRaisesRegex(ValueError, 'distraction'):
                run_suite(root/'fatigue', root/'wrong')
            settings = dict(DEFAULTS, epochs=1, batch_size=3, hidden_dim=4, dropout=0.)
            with contextlib.redirect_stdout(io.StringIO()):
                report = run(reference_prepared_dir=root/'prepared', output_dir=root/'run',
                             settings=settings, seeds=[42])
            self.assertEqual(report['task'], 'fatigue')
            self.assertEqual(len(report['comparisons']), 7)
            cohorts = {r['window_label_hash'] for r in report['comparisons']}
            self.assertEqual(len(cohorts), 1)
            self.assertEqual(report['comparisons'][0]['overall']['available'], 0)
            for row in report['comparisons'][1:]:
                self.assertEqual(row['overall']['available'], 8)
                self.assertEqual(row['overall']['unknown'], 1)
                self.assertEqual(row['baseline_f_ineligible']['positive'], 3)
            with contextlib.redirect_stdout(io.StringIO()):
                repeated = run(reference_prepared_dir=root/'prepared', output_dir=root/'run',
                               settings=settings, seeds=[42], continue_run=True)
            self.assertEqual(report, repeated)
            with self.assertRaisesRegex(ValueError, 'configuration/identity'):
                run(reference_prepared_dir=root/'prepared', output_dir=root/'run',
                    settings=settings, seeds=[43], continue_run=True)


if __name__ == '__main__':
    unittest.main()
