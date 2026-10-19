"""Behavioral contracts for the shared, quality-gated causal relation model."""

import importlib
import io
import unittest

import torch
from torch import nn


class RelationModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)
        self.assertIsNotNone(
            importlib.util.find_spec("training.relation"),
            "The relation model package must exist",
        )
        self.model_type = importlib.import_module("training.relation.model").RelationGatedTCN

    def inputs(self, batch=3, steps=100):
        x = torch.randn(batch, steps, 38)
        x[:, :, 19:] = 1
        return x, torch.rand(batch, 4, 2), torch.ones(batch, 4, dtype=torch.bool)

    def test_default_architecture_is_shared_and_under_budget(self):
        model = self.model_type()
        convs = [module for module in model.modules() if isinstance(module, nn.Conv1d)]
        self.assertEqual(len(convs), 10)
        self.assertEqual([conv.dilation[0] for conv in convs], [1, 1, 2, 2, 4, 4, 8, 8, 16, 16])
        self.assertEqual(len(model.projections), 4)
        self.assertEqual([projection[0].in_features for projection in model.projections], [18, 4, 12, 4])
        self.assertEqual(len(model.gates), 4)
        self.assertEqual([gate[0].in_features for gate in model.gates], [72] * 4)
        self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 100_000)
        x, quality, eligible = self.inputs()
        self.assertEqual(model(x, quality, eligible).shape, (3, 4))
        self.assertEqual(model.encode(x).shape, (3, 4, 16, 100))

    def test_encode_is_causal(self):
        model = self.model_type(dropout=0).eval()
        x, _, _ = self.inputs(batch=1)
        changed = x.clone()
        changed[:, 50:, :19] += 100
        torch.testing.assert_close(model.encode(x)[..., :50], model.encode(changed)[..., :50], rtol=0, atol=0)

    def test_invalid_values_are_zeroed_before_projection(self):
        model = self.model_type(dropout=0).eval()
        x, quality, eligible = self.inputs()
        x[:, :, 19::2] = 0
        changed = x.clone()
        changed[:, :, :19:2] = 12345
        torch.testing.assert_close(model.encode(x), model.encode(changed), rtol=0, atol=0)
        torch.testing.assert_close(model(x, quality, eligible), model(changed, quality, eligible), rtol=0, atol=0)

    def test_ineligible_group_values_do_not_affect_eligible_gates_or_logits(self):
        model = self.model_type(dropout=0).eval()
        x, quality, eligible = self.inputs()
        eligible[:, 0] = False
        changed = x.clone()
        changed[:, :, list(model.group_indices[0])] += 100
        original = model(x, quality, eligible, return_aux=True)
        perturbed = model(changed, quality, eligible, return_aux=True)
        torch.testing.assert_close(original["weights"], perturbed["weights"], rtol=0, atol=0)
        torch.testing.assert_close(original["logits"], perturbed["logits"], rtol=0, atol=0)

    def test_ineligible_group_inputs_have_zero_gradients(self):
        model = self.model_type(dropout=0).eval()
        x, quality, eligible = self.inputs()
        eligible[:, 0] = False
        x.requires_grad_()
        model(x, quality, eligible).square().sum().backward()
        group = list(model.group_indices[0])
        ineligible_channels = group + [index + 19 for index in group]
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(torch.count_nonzero(x.grad[:, :, ineligible_channels]).item(), 0)
        self.assertGreater(torch.count_nonzero(x.grad).item(), 0)

    def test_mixed_missing_rows_have_zero_weights_and_finite_gradients(self):
        model = self.model_type(dropout=0)
        x, quality, eligible = self.inputs()
        eligible[0] = False
        eligible[1] = torch.tensor([False, True, False, False])
        eligible[2] = torch.tensor([True, False, True, False])
        x.requires_grad_()
        quality.requires_grad_()
        result = model(x, quality, eligible, return_aux=True)
        self.assertEqual(set(result), {"logits", "weights", "available"})
        self.assertEqual(result["weights"].shape, (3, 4, 4))
        self.assertEqual(result["available"].dtype, torch.bool)
        self.assertEqual(result["available"].tolist(), [False, True, True])
        self.assertTrue(torch.isfinite(result["logits"]).all())
        self.assertTrue(torch.isfinite(result["weights"]).all())
        self.assertEqual(torch.count_nonzero(result["logits"][0]).item(), 0)
        self.assertEqual(torch.count_nonzero(result["weights"].masked_select(~eligible[:, None, :])).item(), 0)
        torch.testing.assert_close(result["weights"].sum(-1), torch.tensor([[0.] * 4, [1.] * 4, [1.] * 4]))
        torch.testing.assert_close(result["weights"][1, :, 1], torch.ones(4), rtol=0, atol=0)
        result["logits"].sum().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(torch.isfinite(quality.grad).all())
        self.assertEqual(torch.count_nonzero(x.grad[0]).item(), 0)

    def test_state_dict_round_trip(self):
        model = self.model_type().eval()
        restored = self.model_type().eval()
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        inputs = self.inputs()
        torch.testing.assert_close(model(*inputs), restored(*inputs), rtol=0, atol=0)

    def test_nondefault_widths_and_steps(self):
        model = self.model_type(projection_dim=5, gate_hidden_dim=7, channels=(8, 12), dilations=(1, 3), window_steps=20)
        inputs = self.inputs(steps=20)
        self.assertEqual(model.encode(inputs[0]).shape, (3, 4, 12, 20))
        self.assertEqual(model(*inputs).shape, (3, 4))

    def test_invalid_configuration_is_rejected(self):
        invalid = [
            {"input_channels": 37}, {"input_channels": True}, {"projection_dim": 0},
            {"gate_hidden_dim": False}, {"channels": ()}, {"channels": (0,)},
            {"channels": (16,), "dilations": (1, 2)}, {"dilations": (1, 2, 4, 8, 0)},
            {"kernel_size": 0}, {"window_steps": 0}, {"dropout": float("nan")},
            {"dropout": 1}, {"dropout": True}, {"group_indices": [[0], [1], [2]]},
            {"group_indices": [[0], [1], [2], []]},
            {"group_indices": [[0], [1], [2], list(range(2, 19))]},
            {"group_indices": [[0], [1], [2], list(range(4, 19))]},
            {"group_indices": [[False], [1], [2], list(range(3, 19))]},
            {"group_indices": [[0], [1], [2], list(range(3, 20))]},
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model_type(**kwargs)

    def test_input_and_quality_validation(self):
        model = self.model_type()
        x, quality, eligible = self.inputs()
        for bad_x in (x[:, :-1], x[:, :, :-1], x[:0], torch.full_like(x, float("nan"))):
            with self.subTest(shape=bad_x.shape), self.assertRaises(ValueError):
                model(bad_x, quality, eligible)
        fractional_masks = x.clone()
        fractional_masks[:, :, 19] = 0.5
        with self.assertRaises(ValueError):
            model(fractional_masks, quality, eligible)
        for bad_quality in (quality[:, :, :1], quality[:1], torch.full_like(quality, -0.1), torch.full_like(quality, 1.1), torch.full_like(quality, float("nan"))):
            with self.subTest(quality=bad_quality), self.assertRaises(ValueError):
                model(x, bad_quality, eligible)
        for bad_eligible in (eligible[:1], eligible[:, :3]):
            with self.assertRaises(ValueError):
                model(x, quality, bad_eligible)
        for bad_inputs in ((x.double(), quality, eligible), (x, quality.double(), eligible), (x, quality, eligible.float())):
            with self.assertRaises(TypeError):
                model(*bad_inputs)


if __name__ == "__main__":
    unittest.main()
