"""Cost measurements must keep cached inference separate from deployment cost."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


class EvaluationCostTests(unittest.TestCase):
    def test_tensor_accounting_includes_buffers_without_claiming_device_peak(self):
        from training.visual.evaluation_cost import model_footprint, modelstate_footprint
        model = torch.nn.Linear(3, 2)
        model.register_buffer('count', torch.tensor(7, dtype=torch.int64))
        info = model_footprint(model)
        self.assertEqual(info['parameter_count'], 8)
        self.assertEqual(info['parameter_bytes'], 32)
        self.assertEqual(info['buffer_bytes'], 8)
        self.assertEqual(info['tensor_bytes'], 40)
        self.assertEqual(modelstate_footprint(model.state_dict())['tensor_bytes'], 40)

    def test_cpu_measurement_returns_result_and_no_fake_accelerator_memory(self):
        from training.visual.evaluation_cost import measure_pass
        with patch('training.visual.evaluation_cost.perf_counter', side_effect=[10., 12.]):
            result, cost = measure_pass(lambda: [1, 2], 'cpu')
        self.assertEqual(result, [1, 2])
        self.assertEqual(cost['wall_seconds'], 2.)
        self.assertIsNone(cost['peak_device_allocated_bytes'])
        self.assertFalse(cost['includes_rgb_extraction'])
        self.assertFalse(cost['deployment_latency'])

    def test_accelerator_boundaries_synchronize_and_measure_requested_device(self):
        from training.visual.evaluation_cost import measure_pass
        events = []
        api = SimpleNamespace(
            synchronize=lambda device: events.append(('sync', device)),
            reset_peak_memory_stats=lambda device: events.append(('reset', device)),
            max_memory_allocated=lambda device: 700,
            memory_allocated=lambda device: 100)
        with patch.object(torch, 'npu', api, create=True):
            result, cost = measure_pass(lambda: events.append(('run',)) or 3, 'npu', 2)
        self.assertEqual(result, 3)
        self.assertEqual(events, [('sync', 2), ('reset', 2), ('run',), ('sync', 2)])
        self.assertEqual(cost['peak_device_allocated_bytes'], 700)
        self.assertEqual(cost['starting_device_allocated_bytes'], 100)

    def test_invalid_options_and_operation_errors_are_not_hidden(self):
        from training.visual.evaluation_cost import measure_pass
        for backend, index in [('bad', 0), ('cpu', 1), ('npu', -1)]:
            with self.assertRaises(ValueError):
                measure_pass(lambda: 1, backend, index)
        def broken():
            raise RuntimeError('real failure')
        with self.assertRaisesRegex(RuntimeError, 'real failure'):
            measure_pass(broken, 'cpu')


if __name__ == '__main__':
    unittest.main()
