"""Contract tests for the causal temporal classifier and masked multilabel loss."""

import importlib
import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F


class TemporalCoreTests(unittest.TestCase):
    def model_type(self):
        try:
            return importlib.import_module("training.temporal.model").TemporalTCN
        except ModuleNotFoundError:
            self.fail("TemporalTCN implementation is missing")

    def loss_function(self):
        try:
            return importlib.import_module("training.temporal.losses").masked_bce_with_logits
        except ModuleNotFoundError:
            self.fail("masked_bce_with_logits implementation is missing")

    @staticmethod
    def inputs(batch=2, steps=100, channels=24):
        return torch.cat((torch.randn(batch, steps, channels // 2),
                          torch.randint(0, 2, (batch, steps, channels // 2)).float()), dim=-1)

    def test_default_parameter_count_and_last_timestep_raw_logits(self):
        model = self.model_type()().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 31204)
        x = self.inputs()
        features = model.forward_features(x)
        self.assertEqual(features.shape, (2, 32, 100))
        self.assertEqual(model(x).shape, (2, 4))
        heads = [m for m in model.modules() if isinstance(m, nn.Linear)]
        self.assertEqual(len(heads), 1)
        torch.testing.assert_close(model(x), heads[0](features[:, :, -1]))
        with torch.no_grad():
            heads[0].weight.zero_()
            heads[0].bias.copy_(torch.tensor([-3., 2., 5., -2.]))
        torch.testing.assert_close(model(x), torch.tensor([[-3., 2., 5., -2.]]).expand(2, -1))

    def test_alternate_feature_profile(self):
        model = self.model_type()(input_channels=40, channels=(8, 12), dilations=(1, 3), window_steps=17)
        x = self.inputs(3, 17, 40)
        self.assertEqual(model.forward_features(x).shape, (3, 12, 17))
        self.assertEqual(model(x).shape, (3, 4))

    def test_changing_future_values_and_masks_preserves_past_features(self):
        torch.manual_seed(120)
        model = self.model_type()().eval()
        x = self.inputs()
        changed = x.clone()
        changed[:, 41:, :12] = torch.randn_like(changed[:, 41:, :12]) * 10
        changed[:, 41:, 12:] = 1 - changed[:, 41:, 12:]
        a, b = model.forward_features(x), model.forward_features(changed)
        torch.testing.assert_close(a[:, :, :41], b[:, :, :41], rtol=0, atol=0)
        self.assertFalse(torch.equal(a[:, :, 41:], b[:, :, 41:]))

    def test_invalid_model_configuration(self):
        invalid = [dict(input_channels=0), dict(input_channels=3), dict(input_channels=True),
                   dict(input_channels=24.0), dict(channels=()), dict(channels=(0,)),
                   dict(channels=(True,)), dict(channels=(3.5,)), dict(dilations=(0,) * 5),
                   dict(dilations=(True,) * 5), dict(dilations=(1.0,) * 5),
                   dict(channels=(8, 8), dilations=(1,)), dict(kernel_size=0),
                   dict(kernel_size=True), dict(kernel_size=3.0), dict(dropout=-.1),
                   dict(dropout=1.0), dict(dropout=1.1), dict(dropout=math.nan), dict(dropout=True),
                   dict(window_steps=0), dict(window_steps=True), dict(window_steps=10.0)]
        cls = self.model_type()
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises((ValueError, TypeError)):
                cls(**kwargs)

    def test_invalid_model_inputs(self):
        model = self.model_type()()
        invalid = [torch.zeros(100, 24), torch.zeros(0, 100, 24), torch.zeros(1, 0, 24),
                   torch.zeros(1, 99, 24), torch.zeros(1, 100, 22),
                   torch.zeros(1, 100, 24, dtype=torch.float64), torch.zeros(1, 100, 24, dtype=torch.int64)]
        for value in (math.nan, math.inf):
            x = self.inputs(1)
            x[0, 0, 0] = value
            invalid.append(x)
        for value in (-1., .5, 2.):
            x = self.inputs(1)
            x[0, 0, 12] = value
            invalid.append(x)
        for index, x in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises((ValueError, TypeError)):
                model(x)

    def test_multilabel_gradient_and_class_balanced_reduction(self):
        loss_fn = self.loss_function()
        logits = torch.tensor([[1., -1., .5, -2.], [2., 1., -.5, 3.], [-1., 4., 1., 0.]], requires_grad=True)
        targets = torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 0.], [1., 0., 0., 0.]])
        mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
        loss, sums, counts = loss_fn(logits, targets, mask)
        elements = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        expected_sums = (elements * mask).sum(0)
        expected = (expected_sums[:3] / torch.tensor([3., 1., 1.])).mean()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(sums, expected_sums.detach())
        torch.testing.assert_close(counts.to(torch.float32), torch.tensor([3., 1., 1., 0.]))
        self.assertFalse(sums.requires_grad)
        self.assertFalse(counts.requires_grad)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[~mask] == 0))
        self.assertLess(logits.grad[0, 0].item(), 0)
        self.assertLess(logits.grad[0, 1].item(), 0)
        self.assertGreater(logits.grad[1, 0].item(), 0)

    def test_active_task_mask_excludes_loss_counts_and_gradient(self):
        logits = torch.zeros(2, 4, requires_grad=True)
        targets = torch.ones_like(logits)
        loss, sums, counts = self.loss_function()(logits, targets, torch.ones_like(logits), torch.tensor([1, 0, 1, 0]))
        torch.testing.assert_close(loss, torch.tensor(math.log(2)))
        torch.testing.assert_close(counts.float(), torch.tensor([2., 0., 2., 0.]))
        torch.testing.assert_close(sums, torch.tensor([2 * math.log(2), 0., 2 * math.log(2), 0.]))
        loss.backward()
        self.assertTrue(torch.all(logits.grad[:, [1, 3]] == 0))
        self.assertTrue(torch.all(logits.grad[:, [0, 2]] < 0))

    def test_all_unknown_or_inactive_returns_none_and_zero_statistics(self):
        fn = self.loss_function()
        for mask, active in ((torch.zeros(2, 4), None), (torch.ones(2, 4), torch.zeros(4))):
            logits = torch.zeros(2, 4, requires_grad=True)
            loss, sums, counts = fn(logits, torch.zeros(2, 4), mask, active)
            self.assertIsNone(loss)
            self.assertIsNone(logits.grad)
            self.assertTrue(torch.all(sums == 0))
            self.assertTrue(torch.all(counts == 0))

    def test_invalid_loss_inputs(self):
        fn = self.loss_function()
        base = [torch.zeros(2, 4), torch.zeros(2, 4), torch.ones(2, 4), torch.ones(4)]
        invalid = [(0, torch.zeros(2, 3)), (0, torch.zeros(2, 4, 1)),
                   (1, torch.zeros(1, 4)), (2, torch.ones(4)), (3, torch.ones(1, 4)),
                   (0, torch.full((2, 4), math.nan)), (0, torch.full((2, 4), math.inf)),
                   (1, torch.full((2, 4), .5)), (1, torch.full((2, 4), math.nan)),
                   (2, torch.full((2, 4), .5)), (2, torch.full((2, 4), math.inf)),
                   (3, torch.full((4,), -1.)), (3, torch.full((4,), math.nan))]
        for index, value in invalid:
            args = base.copy()
            args[index] = value
            with self.subTest(index=index, value=value), self.assertRaises((ValueError, TypeError)):
                fn(*args)
        with self.assertRaises(ValueError):
            fn(torch.zeros(2, 4), torch.ones(2, 4), torch.zeros(2, 4))


if __name__ == "__main__":
    unittest.main()
