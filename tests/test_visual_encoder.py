"""Frozen backbone contracts, with authored weights (no accuracy claims)."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from inference.experts.assets import file_sha256


class VisualEncoderTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('training.visual.encoder'), 'encoder missing')
        from training.visual import encoder
        return encoder

    def test_full_roi_letterbox_rgb_mean_std(self):
        api = self.api()
        bgr = np.zeros((10, 24, 3), np.uint8)
        bgr[:, 2:22] = [10, 20, 240]
        value = api.preprocess(bgr, [2, 0, 22, 10])
        self.assertEqual(value.shape, (3, 224, 224))
        np.testing.assert_allclose(value[:, 112, 112], (np.array([240,20,10])/255-api.MEAN)/api.STD, atol=1e-6)
        np.testing.assert_allclose(value[:, 0, 0], -api.MEAN/api.STD, atol=1e-6)
        np.testing.assert_allclose(value[:, 56, 0], value[:, 112, 112], atol=1e-6)
        for roi in ([0,0,30,10], [2,0,2,10], [0,float('nan'),4,4]):
            with self.assertRaises(ValueError): api.preprocess(bgr, roi)

    def test_strict_cpu_verified_load_frozen_features(self):
        api = self.api()
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'authored.pth'
            model = api.ResNet18()
            self.assertEqual(tuple(model.state_dict()['layer4.0.downsample.0.weight'].shape), (512,256,1,1))
            self.assertEqual(tuple(model.state_dict()['fc.weight'].shape), (1000,512))
            torch.save(model.state_dict(), path)
            encoder = api.FrozenEncoder(path, file_sha256(path))
            self.assertFalse(encoder.model.training)
            self.assertTrue(all(not p.requires_grad for p in encoder.model.parameters()))
            encoded = encoder.encode(np.zeros((2,3,224,224), np.float32))
            self.assertEqual(encoded.shape, (2,512))
            self.assertEqual(encoded.dtype, np.float32)
            self.assertTrue(np.isfinite(encoded).all())
            with patch('torch.load', side_effect=AssertionError('must verify first')):
                with self.assertRaisesRegex(ValueError, 'SHA256'): api.FrozenEncoder(path, '0'*64)
            torch.save({'fc.weight': model.fc.weight}, path)
            with self.assertRaises(RuntimeError): api.FrozenEncoder(path, file_sha256(path))


if __name__ == '__main__': unittest.main()
