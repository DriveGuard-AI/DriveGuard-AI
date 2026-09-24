"""Frozen RGB probe contracts; synthetic inputs make no quality claims."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from training.common.checkpoint import load_checkpoint, save_checkpoint


class ProbeDataset(Dataset):
    visual_dim = 512
    context_dim = 6

    def __init__(self):
        generator = torch.Generator().manual_seed(81)
        self.visual = torch.randn(7, 8, 512, generator=generator)
        self.context = torch.randn(7, 6, generator=generator)
        self.mask = torch.ones(7, 8, dtype=torch.bool)
        self.mask[5, 3] = False
        self.targets = torch.tensor([0., 1., 0., 1., 0., 1., float('nan')])
        self.known = torch.tensor([True] * 6 + [False])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return ({'visual': self.visual[index], 'context': self.context[index],
                 'visual_mask': self.mask[index]}, self.targets[index], self.known[index])


class VisualTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def model_type(self):
        self.assertIsNotNone(importlib.util.find_spec('training.visual.model'), 'visual probe model missing')
        from training.visual.model import VisualProbe
        return VisualProbe

    def trainer(self):
        self.assertIsNotNone(importlib.util.find_spec('training.visual.train'), 'visual probe trainer missing')
        from training.visual.train import train_probe
        return train_probe

    def settings(self, epochs=1):
        return dict(epochs=epochs, batch_size=3, seed=51, learning_rate=.002,
                    weight_decay=.01, gradient_clip_norm=1., hidden_dim=8,
                    dropout=.2, backend='cpu', device_index=0)

    def test_visual_ignores_context_and_requires_all_eight_frames(self):
        model = self.model_type()(512, 6, hidden_dim=8).eval()
        data = ProbeDataset()
        out = model(data.visual, data.context, data.mask)
        changed = model(data.visual, torch.full_like(data.context, float('nan')), data.mask)
        torch.testing.assert_close(out['logits'], changed['logits'], atol=0, rtol=0)
        self.assertEqual(out['available'].tolist(), [True] * 5 + [False, True])
        self.assertEqual(tuple(out['logits'].shape), (7,))
        self.assertTrue(torch.isfinite(out['logits']).all())

    def test_fusion_context_receives_gradients_only_for_available_samples(self):
        model = self.model_type()(512, 6, hidden_dim=8, dropout=0, mode='fusion')
        data = ProbeDataset()
        context = data.context.clone().requires_grad_()
        visual = data.visual.clone().requires_grad_()
        out = model(visual, context, data.mask)
        out['logits'].sum().backward()
        self.assertGreater(float(context.grad[:5].abs().sum()), 0)
        self.assertGreater(float(visual.grad[:5].abs().sum()), 0)
        self.assertEqual(float(context.grad[5].abs().sum()), 0)
        self.assertEqual(float(visual.grad[5].abs().sum()), 0)

    def test_unknown_and_unavailable_rows_do_not_change_updates_or_rng(self):
        model = self.model_type()(512, 6, hidden_dim=8, dropout=.2, mode='fusion')
        self.trainer()
        from training.visual.train import run_epoch
        data = ProbeDataset()
        reference = deepcopy(model)
        def run(candidate, indices):
            torch.manual_seed(91)
            loader = DataLoader([data[i] for i in indices], batch_size=7,
                                generator=torch.Generator().manual_seed(12))
            optimizer = torch.optim.Adam(candidate.parameters(), lr=.002, weight_decay=.01)
            report = run_epoch(candidate, loader, torch.device('cpu'), optimizer=optimizer)
            return report, torch.get_rng_state()
        report, rng = run(model, range(7))
        other, other_rng = run(reference, range(5))
        self.assertEqual(report['known_count'], 5)
        self.assertEqual(report['label_coverage'], {'positive': 3, 'negative': 3, 'unknown': 1})
        self.assertEqual(report['unavailable_positive'], 1)
        self.assertEqual(report['unavailable_windows'], 1)
        self.assertEqual(report['loss'], other['loss'])
        torch.testing.assert_close(rng, other_rng, atol=0, rtol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[key], atol=0, rtol=0)

    def test_exact_epoch_resume_reload_and_provenance_rejection(self):
        train = self.trainer()
        from training.visual.train import load_probe
        data = ProbeDataset()
        provenance = {'manifest_sha256': 'synthetic', 'normalization': {'train_only': True}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = train(data, data, 'fusion', self.settings(), root/'first', provenance)
            resumed = train(data, data, 'fusion', self.settings(2), root/'resumed', provenance,
                            resume=first['last_checkpoint'])
            full = train(data, data, 'fusion', self.settings(2), root/'full', provenance)
            a, b = [load_checkpoint(item['last_checkpoint']) for item in (resumed, full)]
            self.assertEqual(a['history'], b['history'])
            for key, value in a['model_state'].items():
                torch.testing.assert_close(value, b['model_state'][key], atol=0, rtol=0)
                self.assertEqual(value.device.type, 'cpu')
            model, payload = load_probe(resumed['best_checkpoint'])
            self.assertFalse(model.training)
            self.assertEqual(payload['selection_metric'], 'binary_validation_bce')
            self.assertTrue(torch.isfinite(model(data.visual, data.context, data.mask)['logits']).all())
            with self.assertRaisesRegex(ValueError, 'provenance'):
                train(data, data, 'fusion', self.settings(2), root/'bad', {'different': True},
                      resume=first['last_checkpoint'])
            self.assertFalse((root/'bad').exists())
            with self.assertRaisesRegex(ValueError, 'configuration'):
                train(data, data, 'fusion', dict(self.settings(2), seed=12), root/'bad_config', provenance,
                      resume=first['last_checkpoint'])
            broken = deepcopy(payload)
            broken['provenance']['manifest_sha256'] = 'changed'
            save_checkpoint(root/'broken.pt', broken)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                load_probe(root/'broken.pt')
            with open(full['best_checkpoint'], 'ab') as handle:
                handle.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_probe(full['best_checkpoint'])

    def test_missing_supervision_fails_before_output_creation(self):
        train = self.trainer()
        data = ProbeDataset()
        empty = deepcopy(data)
        empty.mask[:] = False
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'out'
            for datasets in ((empty, data), (data, empty)):
                with self.assertRaisesRegex(ValueError, 'supervision'):
                    train(*datasets, 'visual', self.settings(), path, {})
                self.assertFalse(path.exists())

    def test_entire_unsupervised_batch_consumes_no_dropout_or_optimizer_state(self):
        model = self.model_type()(512, 6, hidden_dim=8, mode='fusion')
        self.trainer()
        from training.visual.train import run_epoch
        data = ProbeDataset()
        before = deepcopy(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=.1)
        loader = DataLoader([data[5], data[6]], batch_size=2,
                            generator=torch.Generator().manual_seed(21))
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, 'supervision'):
            run_epoch(model, loader, torch.device('cpu'), optimizer=optimizer)
        self.assertFalse(optimizer.state)
        torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
