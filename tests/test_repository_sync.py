"""Regression checks for files needed by a fresh server checkout (requires Git)."""
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositorySyncTests(unittest.TestCase):
    def assert_ignored(self, path, expected):
        result = subprocess.run(
            ['git', '-c', f'core.excludesFile={os.devnull}',
             '-c', 'core.ignorecase=false', 'check-ignore', '--no-index', '--quiet', path],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertIn(result.returncode, (0, 1), result.stderr)
        self.assertEqual(result.returncode == 0, expected, path)

    def test_source_tests_and_templates_are_not_ignored(self):
        for path in (
            'tests/test_relation_model.py', 'tests/helpers.py',
            'training/relation/smoke.py', 'training/relation/npu_train.py',
            'requirements-npu.txt', 'requirements.txt',
            'configs/training/relation_gated.json',
            'configs/perception/runtime.example.json',
        ):
            with self.subTest(path=path):
                self.assert_ignored(path, False)

    def test_machine_data_and_artifacts_remain_ignored_on_linux(self):
        for path in (
            'Dataset/distraction/example.mp4', 'datasets/example.mp4',
            'data/features/values.npy', 'runs/smoke/best.pt',
            '.venv-training/bin/python', 'tests/__pycache__/test_example.pyc',
            'configs/perception/runtime.local.json',
            'configs/training/server.local.json', 'models/local/yolox_nano.pth',
        ):
            with self.subTest(path=path):
                self.assert_ignored(path, True)

    def test_selected_release_weights_can_be_versioned(self):
        self.assert_ignored('models/releases/example/best.pt', False)


if __name__ == '__main__':
    unittest.main()
