import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('training.common.config'),
                             'configuration API not implemented')
        from training.common import config
        return config

    def test_default_contract(self):
        config = self.api().load_config()
        self.assertEqual(config['model']['input_channels'], 24)
        self.assertEqual(config['data']['window_steps'], 100)
        self.assertEqual(config['training']['active_tasks'],
                         ['fatigue', 'phone_use', 'drinking', 'distraction'])

    def test_invalid_values_and_unknown_keys(self):
        api = self.api()
        for section, key, value in [('training', 'epochs', True),
                                    ('training', 'batch_size', 0),
                                    ('training', 'learning_rate', float('nan')),
                                    ('model', 'input_channels', 25),
                                    ('training', 'active_tasks', ['fatigue', 'fatigue']),
                                    ('training', 'surprise', 1)]:
            with self.subTest(key=key):
                config = api.load_config()
                config[section][key] = value
                with self.assertRaises(ValueError):
                    api.validate_config(config)

    def test_platform_cannot_change_architecture(self):
        api = self.api()
        with self.assertRaises(ValueError):
            api.apply_platform(api.load_config(), {'model': {'input_channels': 10}})
        resolved = api.apply_platform(api.load_config(), {'backend': 'npu', 'device_index': 0})
        self.assertEqual(resolved['runtime']['backend'], 'npu')

    def test_gpu_and_npu_precheck_without_site_packages(self):
        for backend in ('gpu', 'npu'):
            result = subprocess.run([sys.executable, '-S', '-m',
                                     f'training.temporal.{backend}_train', '--check-config'],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload['status'], 'config_valid')
            self.assertFalse(payload['device_verified'])
            self.assertEqual(payload['config']['runtime']['backend'],
                             'cuda' if backend == 'gpu' else 'npu')

    def test_explicit_requested_device_does_not_fallback(self):
        self.assertIsNotNone(importlib.util.find_spec('training.common.device'),
                             'device adapter not implemented')
        from training.common.device import select_device
        with self.assertRaises(ValueError):
            select_device('automatic', 0)
        import torch
        if not torch.cuda.is_available():
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                select_device('cuda', 0)


if __name__ == '__main__':
    unittest.main()
