"""Early stopping preserves strict best selection and exact epoch resume."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tests.test_visual_training import ProbeDataset
from training.common.checkpoint import load_checkpoint, save_checkpoint
from training.visual import train as trainer


class EarlyStoppingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.data = ProbeDataset()
        self.settings = dict(epochs=9, batch_size=3, seed=51, learning_rate=.002,
                             weight_decay=.01, gradient_clip_norm=1., hidden_dim=8,
                             dropout=.2, backend='cpu', device_index=0,
                             early_stopping_patience=3, early_stopping_min_delta=.0001)

    def train(self, output, losses, settings=None, resume=None):
        real_epoch = trainer.run_epoch
        validation_losses = iter(losses)

        def controlled_epoch(model, loader, device, optimizer=None, gradient_clip_norm=1.):
            report = real_epoch(model, loader, device, optimizer, gradient_clip_norm)
            if optimizer is None:
                report['loss'] = next(validation_losses)
            return report

        with patch.object(trainer, 'run_epoch', side_effect=controlled_epoch):
            return trainer.train_probe(self.data, self.data, 'context', settings or self.settings,
                                       output, {}, resume=resume)

    def assert_tree_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_tree_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_small_strict_improvements_select_best_without_resetting_patience(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.train(Path(temporary)/'run', [1., .99996, .99994, .99992])
            self.assertEqual(result['completed_epochs'], 4)
            self.assertEqual(result['max_epochs'], 9)
            self.assertTrue(result['stopped_early'])
            last = load_checkpoint(result['last_checkpoint'])
            best = load_checkpoint(result['best_checkpoint'])
            self.assertEqual(best['epoch'], 4)
            self.assertEqual(best['best_val_loss'], .99992)
            self.assertEqual(last['early_stopping_state'],
                             dict(best=1., bad_epochs=3, stopped_early=True))
            self.assertTrue(trainer.is_terminal(last))
            with self.assertRaisesRegex(ValueError, 'stopped'):
                self.train(Path(temporary)/'extend', [], dict(self.settings, epochs=12),
                           resume=result['last_checkpoint'])
            self.assertFalse((Path(temporary)/'extend').exists())

    def test_significant_improvement_resets_counter(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.train(Path(temporary)/'run', [1., .8, .9, .7, .8, .9, 1.])
            self.assertEqual(result['completed_epochs'], 7)
            self.assertEqual(result['best_epoch'], 4)
            last = load_checkpoint(result['last_checkpoint'])
            self.assertEqual(last['early_stopping_state'],
                             dict(best=.7, bad_epochs=3, stopped_early=True))

    def test_interrupted_save_resumes_exact_model_optimizer_history_and_rng(self):
        losses = [1., .99996, .99994, .99992]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = self.train(root/'full', losses)

            def interrupt_after_last(path, payload):
                save_checkpoint(path, payload)
                if Path(path).name == 'last.pt' and payload['epoch'] == 2:
                    raise RuntimeError('simulated process interruption')

            with patch.object(trainer, 'save_checkpoint', side_effect=interrupt_after_last):
                with self.assertRaisesRegex(RuntimeError, 'interruption'):
                    self.train(root/'interrupted', losses)
            interrupted = load_checkpoint(root/'interrupted'/'last.pt')
            self.assertFalse(trainer.is_terminal(interrupted))
            self.assertEqual(interrupted['early_stopping_state']['bad_epochs'], 1)
            resumed = self.train(root/'resumed', losses[2:], resume=root/'interrupted'/'last.pt')
            self.assert_tree_equal(load_checkpoint(full['last_checkpoint']),
                                   load_checkpoint(resumed['last_checkpoint']))
            self.assert_tree_equal(load_checkpoint(full['best_checkpoint']),
                                   load_checkpoint(resumed['best_checkpoint']))

    def test_terminal_marker_is_saved_before_other_epoch_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def interrupt_terminal_save(path, payload):
                save_checkpoint(path, payload)
                if Path(path).name == 'last.pt' and payload['epoch'] == 4:
                    raise RuntimeError('simulated terminal interruption')

            with patch.object(trainer, 'save_checkpoint', side_effect=interrupt_terminal_save):
                with self.assertRaisesRegex(RuntimeError, 'terminal interruption'):
                    self.train(root/'run', [1., 1., 1., 1.])
            payload = load_checkpoint(root/'run'/'last.pt')
            self.assertTrue(trainer.is_terminal(payload))
            self.assertTrue(payload['early_stopping_state']['stopped_early'])
            self.assertEqual(payload['epoch'], 4)
            self.assertFalse((root/'run'/'result.json').exists())

    def test_legacy_settings_and_max_epoch_checkpoints_remain_extendable(self):
        legacy = {key: value for key, value in self.settings.items()
                  if not key.startswith('early_stopping_')}
        legacy['epochs'] = 2
        before = deepcopy(legacy)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.train(root/'legacy', [1., 1.], legacy)
            payload = load_checkpoint(result['last_checkpoint'])
            self.assertEqual(legacy, before)
            self.assertEqual(payload['settings'], before)
            self.assertNotIn('early_stopping_state', payload)
            self.assertFalse(result['stopped_early'])
            self.assertTrue(trainer.is_terminal(payload))
            self.assertFalse(trainer.is_terminal(payload, dict(legacy, epochs=3)))
            resumed = self.train(root/'extended', [1.], dict(legacy, epochs=3),
                                 resume=result['last_checkpoint'])
            self.assertEqual(resumed['completed_epochs'], 3)

    def test_optional_settings_are_validated_without_inserting_defaults(self):
        legacy = {key: value for key, value in self.settings.items()
                  if not key.startswith('early_stopping_')}
        only_patience = dict(legacy, early_stopping_patience=3)
        only_delta = dict(legacy, early_stopping_min_delta=.0001)
        for settings in (legacy, only_patience, only_delta, self.settings):
            self.assertEqual(trainer._settings(settings), settings)
        for key, values in [('early_stopping_patience', (0, -1, True, 1.5, None)),
                            ('early_stopping_min_delta', (-.1, True, float('nan'), float('inf')))]:
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        trainer._settings(dict(legacy, **{key: value}))

    def test_omitted_delta_is_zero_and_omitted_patience_disables_stopping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            only_patience = dict(self.settings, epochs=3, early_stopping_patience=1)
            only_patience.pop('early_stopping_min_delta')
            result = self.train(root/'zero_delta', [1., .99996, .99994], only_patience)
            self.assertEqual(result['completed_epochs'], 3)
            self.assertFalse(result['stopped_early'])
            payload = load_checkpoint(result['last_checkpoint'])
            self.assertNotIn('early_stopping_min_delta', payload['settings'])
            self.assertEqual(payload['early_stopping_state']['best'], .99994)
            only_delta = dict(self.settings, epochs=5)
            only_delta.pop('early_stopping_patience')
            result = self.train(root/'disabled', [1.] * 5, only_delta)
            self.assertEqual(result['completed_epochs'], 5)
            self.assertFalse(result['stopped_early'])

    def test_improvement_equal_to_delta_does_not_reset_patience(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = dict(self.settings, early_stopping_min_delta=.125, early_stopping_patience=2)
            result = self.train(Path(temporary)/'run', [1., .875, .9], settings)
            self.assertEqual(result['completed_epochs'], 3)
            self.assertEqual(result['best_epoch'], 2)
            payload = load_checkpoint(result['last_checkpoint'])
            self.assertEqual(payload['early_stopping_state']['best'], 1.)

    def test_enabled_checkpoint_requires_complete_valid_stopping_state_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.train(root/'original', [1.], dict(self.settings, epochs=1))
            original = load_checkpoint(result['last_checkpoint'])
            malformed = [None, {}, dict(best=1., bad_epochs=-1, stopped_early=False),
                         dict(best=float('nan'), bad_epochs=0, stopped_early=False),
                         dict(best=1., bad_epochs=3, stopped_early=False)]
            for index, state in enumerate(malformed):
                with self.subTest(state=state):
                    payload = deepcopy(original)
                    if state is None:
                        payload.pop('early_stopping_state')
                    else:
                        payload['early_stopping_state'] = state
                    save_checkpoint(root/'invalid.pt', payload)
                    with self.assertRaisesRegex(ValueError, 'early_stopping_state'):
                        self.train(root/f'resume-{index}', [], resume=root/'invalid.pt')
                    self.assertFalse((root/f'resume-{index}').exists())


if __name__ == '__main__':
    unittest.main()
