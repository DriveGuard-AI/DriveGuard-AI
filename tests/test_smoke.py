import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SmokeTests(unittest.TestCase):
    def test_explicit_cpu_smoke_and_existing_run_protection(self):
        self.assertIsNotNone(importlib.util.find_spec('training.temporal.smoke'),
                             'explicit synthetic smoke command not implemented')
        (ROOT / 'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as temporary:
            output = Path(temporary) / 'smoke'
            command = [sys.executable, '-B', '-m', 'training.temporal.smoke', '--output-dir', str(output)]
            result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report['synthetic_only'])
            self.assertEqual(report['initial']['completed_epochs'], 2)
            self.assertEqual(report['resumed']['completed_epochs'], 3)
            self.assertTrue((output / 'smoke_report.json').is_file())
            retry = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(retry.returncode, 0)


if __name__ == '__main__':
    unittest.main()
