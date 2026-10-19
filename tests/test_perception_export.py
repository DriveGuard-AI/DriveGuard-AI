import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.perception_helpers import packet
from training.common.config import TASKS, load_config, canonical_hash
from training.common.data import FeatureCorpus, file_sha256
from training.temporal.model import TemporalTCN
from inference.perception.export import export_recording, read_packets
from inference.perception.profile import load_profile

ROOT = Path(__file__).resolve().parents[1]


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'runs')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.profile = load_profile()

    def test_streamed_arrays_and_unknown_labels(self):
        output = self.root / 'recording'
        report = export_recording((packet(i) for i in range(4101)), output, self.profile)
        self.assertEqual(report['slots'], 4101)
        self.assertEqual(report['profile_sha256'], canonical_hash(self.profile))
        self.assertEqual(report['status'], 'complete_unlabeled')
        for name, columns, dtype in [('values', 19, np.float32), ('masks', 19, np.uint8),
                                      ('targets', 4, np.float32), ('label_masks', 4, np.uint8)]:
            array = np.load(output / f'{name}.npy', allow_pickle=False)
            self.assertEqual(array.shape, (4101, columns))
            self.assertEqual(array.dtype, dtype)
            self.assertTrue(np.isfinite(array).all())
            self.assertEqual(report['array_sha256'][name], file_sha256(output / f'{name}.npy'))
            if name in ('targets', 'label_masks'):
                self.assertFalse(array.any())
        self.assertEqual(json.loads((output / 'record.json').read_text()), report)
        self.assertFalse((output / 'incomplete.json').exists())
        self.assertFalse(list(output.glob('*.tmp')))

    def test_existing_output_never_overwritten_and_invalid_stream_incomplete(self):
        output = self.root / 'existing'
        output.mkdir()
        marker = output / 'keep.txt'
        marker.write_text('user data')
        with self.assertRaises(FileExistsError):
            export_recording(iter(()), output, self.profile)
        self.assertEqual(marker.read_text(), 'user data')
        for case, stream in [('short', (packet(i) for i in range(2))),
                             ('gap', iter([packet(0), packet(2)]))]:
            target = self.root / case
            with self.assertRaises(ValueError):
                export_recording(stream, target, self.profile)
            self.assertFalse((target / 'record.json').exists())
            self.assertTrue((target / 'incomplete.json').exists())
            self.assertFalse(list(target.glob('*.tmp')))

    def test_jsonl_rejects_duplicate_keys_nonfinite_and_blank_lines(self):
        path = self.root / 'observations.jsonl'
        for text in ['{"a":1,"a":2}\n', '{"a":NaN}\n', '\n', '[]\n']:
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                list(read_packets(path))

    def test_cleanup_failure_cannot_publish_success(self):
        original_unlink = Path.unlink
        for suffix in ('.tmp', 'incomplete.json'):
            def fail_one(path, *args, **kwargs):
                if str(path).endswith(suffix):
                    raise OSError('simulated locked output file')
                return original_unlink(path, *args, **kwargs)
            target = self.root / ('tmp-failure' if suffix == '.tmp' else 'marker-failure')
            with patch.object(Path, 'unlink', fail_one), self.assertRaises(OSError):
                export_recording((packet(i) for i in range(100)), target, self.profile)
            self.assertFalse((target / 'record.json').exists())
            self.assertTrue((target / 'incomplete.json').exists())

    def test_export_metadata_does_not_alias_caller_packet(self):
        first = packet(0)
        def stream():
            yield first
            for i in range(1, 100):
                yield packet(i)
            first['driver_roi'][:] = [1, 1, 2, 2]
        target = self.root / 'snapshot'
        result = export_recording(stream(), target, self.profile)
        self.assertEqual(result['driver_roi'], [0, 0, 800, 500])
        self.assertEqual(json.loads((target / 'record.json').read_text())['driver_roi'], [0, 0, 800, 500])

    def test_cli_and_profile_precheck_do_not_load_accelerators(self):
        path = self.root / 'observations.jsonl'
        with path.open('w') as handle:
            for i in range(100):
                handle.write(json.dumps(packet(i)) + '\n')
        result = subprocess.run([sys.executable, '-B', '-m', 'inference.perception.export',
                                 '--input', str(path), '--output-dir', str(self.root / 'cli')],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['slots'], 100)

    def test_exports_enter_existing_corpus_and_tcn_without_invented_supervision(self):
        config = load_config(ROOT / 'configs/training/temporal_interaction.json')
        records = []
        for index in range(4):
            def stream():
                for i in range(100):
                    value = packet(i, closed=i in (2, 3))
                    value['recording_id'] = f'video-{index}'
                    yield value
            export_recording(stream(), self.root / f'r{index}', self.profile)
            records.append({'record_id': f'r{index}', 'subject_id': f's{index}',
                            'recording_id': f'video-{index}', 'view': 'body', 'subset': 'synthetic',
                            'split': 'train' if index < 2 else 'val',
                            **{name: f'r{index}/{name}.npy' for name in ('values', 'masks', 'targets', 'label_masks')}})
        manifest = {'schema_version': 'driveguard-feature-manifest-v1', 'dataset_version': 'synthetic-bridge-v1',
                    'feature_schema': {'version': self.profile['feature_schema_version'],
                                       'names': self.profile['feature_names'], 'sample_rate_hz': 25},
                    'extractor_profile': {**self.profile, 'synthetic': True},
                    'label_mapping': {'version': 'driveguard-behavior-v1', 'tasks': list(TASKS),
                                      'rules': {'all': 'unknown; no behavior labels inferred from geometry'}},
                    'records': records}
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(manifest))
        corpus = FeatureCorpus(config, path, str(self.root))
        try:
            with self.assertRaisesRegex(ValueError, 'requires known positive and negative'):
                corpus.check_supervision(list(TASKS))
            x, labels, mask = corpus.dataset('train', corpus.fit_normalizer())[0]
            self.assertEqual(x.shape, (100, 38))
            self.assertFalse(mask.any())
            model = TemporalTCN(**config['model'], window_steps=100).eval()
            with torch.no_grad():
                logits = model(torch.from_numpy(x).unsqueeze(0))
            self.assertEqual(tuple(logits.shape), (1, 4))
            self.assertTrue(torch.isfinite(logits).all())
        finally:
            corpus.close()


if __name__ == '__main__':
    unittest.main()
