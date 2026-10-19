import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from training.common.config import load_config
from tests.helpers import write_fixture, write_manifest

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def setUp(self):
        (ROOT / 'runs').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'runs')
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.config = load_config()
        self.path, self.manifest = write_fixture(self.root, self.config)

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('training.common.data'),
                             'feature data loader not implemented')
        from training.common import data
        return data

    def corpus(self):
        corpus = self.api().FeatureCorpus(self.config, self.path, self.root)
        self.addCleanup(corpus.close)
        return corpus

    def test_windows_use_end_slot_and_do_not_duplicate_storage(self):
        target = np.load(self.root / 'r0_targets.npy')
        target[99] = 1
        np.save(self.root / 'r0_targets.npy', target)
        corpus = self.corpus()
        normalizer = corpus.fit_normalizer()
        dataset = corpus.dataset('train', normalizer)
        self.assertEqual(len(dataset), 4)
        x, y, m = dataset[0]
        self.assertEqual(x.shape, (100, 24))
        np.testing.assert_array_equal(y, np.ones(4))
        np.testing.assert_array_equal(m, np.ones(4))
        self.assertEqual(len(list(self.root.glob('*.npy'))), 16)

    def test_subject_and_recording_leakage_rejected(self):
        api = self.api()
        for key in ('subject_id', 'recording_id'):
            original = self.manifest['records'][2][key]
            self.manifest['records'][2][key] = self.manifest['records'][0][key]
            write_manifest(self.path, self.manifest)
            with self.assertRaisesRegex(ValueError, 'split|subject|recording'):
                api.FeatureCorpus(self.config, self.path, self.root)
            self.manifest['records'][2][key] = original

    def test_paths_reject_escape_absolute_and_windows_style(self):
        api = self.api()
        for path in ('../outside.npy', '/absolute.npy', 'C:/data.npy', 'a\\b.npy'):
            self.manifest['records'][0]['values'] = path
            write_manifest(self.path, self.manifest)
            with self.subTest(path=path), self.assertRaises(ValueError):
                api.FeatureCorpus(self.config, self.path, self.root)

    def test_nonfinite_and_invalid_mask_rejected(self):
        api = self.api()
        for name, value in [('values', float('nan')), ('masks', 2), ('targets', 0.3)]:
            file = self.root / f'r0_{name}.npy'
            original = np.load(file)
            changed = original.copy()
            changed[0, 0] = value
            np.save(file, changed)
            with self.subTest(name=name), self.assertRaises(ValueError):
                api.FeatureCorpus(self.config, self.path, self.root)
            np.save(file, original)

    def test_normalizer_only_train_unique_slots_and_unsupported_channel(self):
        for i in range(4):
            values = np.full((125, 12), 2 if i == 0 else 4 if i == 1 else 1000, dtype='float32')
            masks = np.ones((125, 12), dtype='uint8')
            if i < 2:
                masks[:, 0] = 0
            np.save(self.root / f'r{i}_values.npy', values)
            np.save(self.root / f'r{i}_masks.npy', masks)
        corpus = self.corpus()
        normalizer = corpus.fit_normalizer()
        self.assertEqual(normalizer['mean'][1], 3)
        self.assertEqual(normalizer['std'][1], 1)
        self.assertFalse(normalizer['supported'][0])
        self.assertEqual(normalizer['count'][1], 250)
        x, _, _ = corpus.dataset('val', normalizer)[0]
        self.assertTrue(np.all(x[:, 1:12] == 5))
        self.assertTrue(np.all(x[:, 0] == 0))
        self.assertTrue(np.all(x[:, 12] == 0))

    def test_active_task_requires_both_classes_per_train_and_val(self):
        np.save(self.root / 'r3_targets.npy', np.zeros((125, 4), dtype='float32'))
        corpus = self.corpus()
        with self.assertRaisesRegex(ValueError, 'val.*positive|positive.*val'):
            corpus.check_supervision(self.config['training']['active_tasks'])

    def test_data_root_cli_precedence(self):
        api = self.api()
        with patch.dict(os.environ, {'DRIVEGUARD_DATA_ROOT': str(self.root / 'env')}):
            self.assertEqual(api.resolve_data_root(str(self.root)), self.root.resolve())
            self.assertEqual(api.resolve_data_root(None), (self.root / 'env').resolve())


if __name__ == '__main__':
    unittest.main()
