"""Adaptation contracts with real encoder gradients and lightweight resume tests."""
from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from training.common.checkpoint import load_checkpoint
from training.visual.encoder import ResNet18
from training.visual.model import VisualProbe


def probe():
    model = VisualProbe(1024, 0, hidden_dim=4, dropout=.2)
    return dict(model_params=model.model_params, model_state=model.state_dict(),
                provenance={'normalization': {'head': {'mean': [0.] * 512, 'std': [1.] * 512}}})


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Sequential(nn.Conv2d(3, 512, 1), nn.BatchNorm2d(512))

    def forward(self, x):
        return self.layer4(x.mean((2, 3), keepdim=True)).flatten(1)


class Pixels(Dataset):
    frames = 8
    def __init__(self, extra=False):
        self.labels = [(0., True), (1., True), (1., True)]
        if extra:
            self.labels += [(float('nan'), False)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        y, known = self.labels[i]
        return dict(images=torch.full((8, 3, 224, 224), 30 + i, dtype=torch.uint8),
                    visual_mask=torch.ones(8, dtype=torch.bool)), y, known

    def metadata(self, i):
        y, known = self.labels[i]
        return dict(record_id=str(i), target=y if known else 0., label_known=known)


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('training.visual.head_adapt_train'),
                             'adaptation trainer missing')
        return importlib.import_module('training.visual.head_adapt_train')

    def test_real_encoder_warm_start_gradients_and_bn_frozen(self):
        api = self.module()
        payload = probe()
        model = api.AdaptModel(payload, ResNet18().state_dict(), 8, True, 4)
        images = torch.randint(0, 256, (1, 8, 3, 224, 224), dtype=torch.uint8)
        mask = torch.ones(1, 8, dtype=torch.bool)
        model.eval()
        with torch.no_grad():
            encoded = model.encoder((images.flatten(0, 1).float() / 255 - torch.tensor([.485,.456,.406])[None,:,None,None]) / torch.tensor([.229,.224,.225])[None,:,None,None]).reshape(1,8,512).clamp(-5,5)
            old = VisualProbe(**payload['model_params']).eval()
            old.load_state_dict(payload['model_state'])
            expected = old(torch.cat((torch.zeros_like(encoded), encoded), -1), None, mask)['logits']
            torch.testing.assert_close(model(images, mask)['logits'], expected, atol=1e-6, rtol=1e-6)
        model.train()
        before = {k: v.clone() for k,v in model.encoder.state_dict().items() if 'running_' in k}
        model(images, mask)['logits'].sum().backward()
        self.assertNotEqual(float(model.gate.grad), 0.)
        for name, param in model.encoder.named_parameters():
            if param.requires_grad:
                self.assertTrue(name.startswith('layer4.') and ('conv' in name or 'downsample.0.' in name))
                self.assertIsNotNone(param.grad)
            else:
                self.assertIsNone(param.grad)
        for name, value in before.items():
            torch.testing.assert_close(model.encoder.state_dict()[name], value, atol=0, rtol=0)
        dense = api.AdaptModel(payload, model.encoder.state_dict(), 32, False)
        self.assertEqual({k:v.shape for k,v in model.state_dict().items()}, {k:v.shape for k,v in dense.state_dict().items()})

    def test_resume_exact_unknown_invariance_and_self_contained_checkpoints(self):
        api = self.module()
        payload, state = probe(), TinyEncoder().state_dict()
        settings = dict(api.DEFAULTS, epochs=2, batch_size=2, accumulation_steps=2, early_stopping_patience=10)
        with tempfile.TemporaryDirectory() as tmp, patch.object(api, 'ResNet18', TinyEncoder):
            root = Path(tmp)
            def run(path, config, data=None, resume=None):
                return api.train_adaptation(data or Pixels(), Pixels(), payload, state, config, root/path,
                                            {'fixture': 'test'}, 8, True, resume=resume)
            full = run('full', settings)
            one = run('one', dict(settings, epochs=1))
            resumed = run('resumed', settings, resume=one['last_checkpoint'])
            unknown = run('unknown', settings, Pixels(extra=True))
            a = load_checkpoint(full['last_checkpoint'])
            for result in (resumed, unknown):
                b = load_checkpoint(result['last_checkpoint'])
                for key,value in a['model_state'].items():
                    torch.testing.assert_close(value, b['model_state'][key], atol=0, rtol=0)
                self.assertEqual(a['history'], b['history'])
            rows, loaded = api.evaluate_adaptation(Pixels(extra=True), full['best_checkpoint'])
            self.assertEqual(len(rows), 4)
            self.assertEqual(loaded['model_kind'], 'head_adaptation_v1')
            terminal = run('terminal', settings, resume=full['last_checkpoint'])
            self.assertEqual(terminal['global_step'], full['global_step'])
            initial = load_checkpoint(terminal['initial_checkpoint'])
            self.assertEqual(initial['epoch'], 0)
            for key,value in initial['model_state'].items():
                torch.testing.assert_close(value, a['initial_model_state'][key], atol=0, rtol=0)

    def test_accumulation_normalizes_by_rows_not_microbatches(self):
        api = self.module()
        class Scalar(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(.3))
            def forward(self, images, visual_mask):
                return {'logits': self.weight.expand(len(images)), 'available': visual_mask.all(1)}
        a, b = Scalar(), Scalar()
        for model, batch, accumulation in ((a, 2, 2), (b, 3, 1)):
            api.run_epoch(model, DataLoader(Pixels(), batch_size=batch), torch.device('cpu'),
                          torch.optim.SGD(model.parameters(), lr=.2), accumulation_steps=accumulation)
        torch.testing.assert_close(a.weight, b.weight, atol=1e-7, rtol=0)

    def test_epoch_zero_remains_best_and_invalid_supervision_rejected(self):
        api = self.module()
        payload, state = probe(), TinyEncoder().state_dict()
        config = dict(api.DEFAULTS, epochs=2)
        with tempfile.TemporaryDirectory() as tmp, patch.object(api, 'ResNet18', TinyEncoder):
            reports = [dict(loss=.1), dict(loss=.4,updates=1), dict(loss=.5), dict(loss=.4,updates=1), dict(loss=.6)]
            with patch.object(api, 'run_epoch', side_effect=reports):
                result = api.train_adaptation(Pixels(), Pixels(), payload, state, config, Path(tmp)/'run', {}, 8, False)
            self.assertEqual(result['best_epoch'], 0)
            empty = Pixels(); empty.labels = [(float('nan'), False)]
            with self.assertRaisesRegex(ValueError, 'supervision'):
                api.train_adaptation(empty, Pixels(), payload, state, config, Path(tmp)/'empty', {}, 8, False)

    def test_normalization_low_bits_and_checkpoint_metadata_consistency(self):
        api = self.module()
        payload = probe()
        payload['provenance']['normalization']['head'] = dict(mean=[10000.12500025]*512, std=[1e-6]*512)
        with patch.object(api, 'ResNet18', TinyEncoder):
            model = api.AdaptModel(payload, TinyEncoder().state_dict(), 8, False)
            captured = []
            handle = model.probe.visual_projection.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
            with patch.object(model.encoder, 'forward', side_effect=lambda x: x.new_full((len(x),512),10000.125)):
                model(**{k:v.unsqueeze(0) for k,v in Pixels()[0][0].items()})
            handle.remove()
            import numpy as np
            expected = float((np.float64(np.float32(10000.125))-10000.12500025)/1e-6)
            torch.testing.assert_close(captured[0][...,512:], torch.full((1,8,512),expected), atol=1e-7, rtol=1e-7)
            with tempfile.TemporaryDirectory() as tmp:
                result = api.train_adaptation(Pixels(), Pixels(), payload, TinyEncoder().state_dict(),
                    dict(api.DEFAULTS, epochs=1), Path(tmp)/'run', {}, 8, False)
                data = load_checkpoint(result['last_checkpoint'])
                altered = deepcopy(data)
                altered['normalizer']['mean'][0] += 1
                with self.assertRaisesRegex(ValueError, 'normaliz'):
                    api.validate_adaptation(altered)
                altered = deepcopy(data)
                del altered['model_state']['head_std']
                with self.assertRaisesRegex(ValueError, 'normaliz'):
                    api.validate_adaptation(altered)
                altered = deepcopy(data)
                altered['epoch'] = 100
                with self.assertRaisesRegex(ValueError, 'epoch|history'):
                    api.validate_adaptation(altered)

    def test_dense_gate_reaches_first_frame_and_chunks_are_bounded(self):
        api = self.module()
        with patch.object(api, 'ResNet18', TinyEncoder):
            model = api.AdaptModel(probe(), TinyEncoder().state_dict(), 32, False, frame_chunk=3).eval()
            with torch.no_grad():
                model.probe.visual_projection.weight.zero_()
                model.probe.visual_projection.weight[:,512] = .1
                model.probe.visual_projection.bias.fill_(.3)
                for layer in model.probe.temporal:
                    layer.weight.fill_(.1); layer.bias.zero_()
                model.extra.weight.fill_(.01); model.extra.bias.zero_()
                model.probe.head.weight.fill_(1.); model.probe.head.bias.zero_()
            chunks = []
            def encode(x):
                chunks.append(len(x))
                return x.mean((1,2,3))[:,None].expand(-1,512)
            images = torch.full((1,32,3,224,224), 100, dtype=torch.uint8)
            other = images.clone(); other[:,0] = 200
            mask = torch.ones(1,32,dtype=torch.bool)
            with patch.object(model.encoder, 'forward', side_effect=encode):
                torch.testing.assert_close(model(images,mask)['logits'], model(other,mask)['logits'], atol=0, rtol=0)
                with torch.no_grad(): model.gate.fill_(1.)
                self.assertGreater(float((model(images,mask)['logits']-model(other,mask)['logits']).abs().detach()), 1e-6)
            self.assertLessEqual(max(chunks),3)
            self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))

    def test_dataset_identity_checked_and_supervision_scan_uses_metadata(self):
        api = self.module()
        class MetadataOnly(Pixels):
            def availability(self, index):
                return True
            def __getitem__(self, index):
                raise AssertionError('Scan must not read image pixels')
        self.assertEqual(api._active_indices(MetadataOnly(extra=True), 'train'), [0,1,2])
        with tempfile.TemporaryDirectory() as tmp, patch.object(api, 'ResNet18', TinyEncoder):
            data = Pixels()
            data.cache = SimpleNamespace(root=Path(tmp), manifest_sha256='pixels')
            data.prepared = SimpleNamespace(provenance={'prepared_sha256':'prepared','preparation':{'x':1}})
            provenance = dict(frame_cache=str(Path(tmp)), frame_cache_sha256='pixels',
                              source=dict(data.prepared.provenance, normalization=probe()['provenance']['normalization']))
            result = api.train_adaptation(data, data, probe(), TinyEncoder().state_dict(),
                dict(api.DEFAULTS, epochs=1), Path(tmp)/'run', provenance, 8, False)
            data.cache.manifest_sha256 = 'different'
            with self.assertRaisesRegex(ValueError, 'provenance'):
                api.evaluate_adaptation(data, result['best_checkpoint'])


if __name__ == '__main__':
    unittest.main()
