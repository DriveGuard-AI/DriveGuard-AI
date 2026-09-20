from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from training.common.config import canonical_hash, load_config, validate_config
from training.common.data import FeatureCorpus
from tests.helpers import write_fixture, write_manifest

ROOT = Path(__file__).resolve().parents[1]


def relation_fixture(root, config, frames=125):
    path, manifest = write_fixture(root, config, frames)
    manifest['schema_version'] = 'driveguard-feature-manifest-v2'
    manifest['group_profile'] = deepcopy(config['data']['group_profile'])
    manifest['group_profile_sha256'] = canonical_hash(manifest['group_profile'])
    for record in manifest['records']:
        record['expert_age_ms'] = record['record_id'] + '_ages.npy'
        np.save(Path(root) / record['expert_age_ms'], np.zeros((frames, 3), np.float32))
    write_manifest(path, manifest)
    return path, manifest


class RelationDataTests(unittest.TestCase):
    def api(self):
        self.assertTrue((ROOT / 'configs/training/relation_gated.json').is_file(), 'v2 config not implemented')
        return load_config(ROOT / 'configs/training/relation_gated.json')

    def test_v2_config_and_strict_v1_compatibility(self):
        config = self.api()
        self.assertEqual(config['model']['kind'], 'relation_gated_v1')
        old = load_config()
        old['model']['kind'] = 'relation_gated_v1'
        with self.assertRaises(ValueError):
            validate_config(old)
        for mutate in (lambda c: c['data']['feature_names'].reverse(),
                       lambda c: c['data']['group_profile'].update(max_age_ms=0),
                       lambda c: c['model'].update(channels=[32]*5)):
            wrong = deepcopy(config)
            mutate(wrong)
            with self.assertRaises(ValueError):
                validate_config(wrong)

    def test_v2_dataset_shapes_and_provenance(self):
        config = self.api()
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as root:
            path, _ = relation_fixture(root, config)
            corpus = FeatureCorpus(config, path, root)
            try:
                normalizer = corpus.fit_normalizer()
                x, target, labels = corpus.dataset('train', normalizer)[0]
                self.assertEqual(x['x'].shape, (100, 38))
                self.assertEqual(x['quality'].shape, (4, 2))
                self.assertEqual(x['eligible'].tolist(), [True]*4)
                self.assertEqual(target.shape, (4,))
                self.assertIn('expert_age_ms', corpus.summary['array_sha256']['r0'])
                self.assertEqual(corpus.summary['group_profile_sha256'], canonical_hash(config['data']['group_profile']))
            finally:
                corpus.close()

    def test_rejects_schema_hash_ages_and_split_errors(self):
        config = self.api()
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as root:
            path, manifest = relation_fixture(root, config)
            for mutate in (lambda m: m.update(schema_version='driveguard-feature-manifest-v1'),
                           lambda m: m.update(group_profile_sha256='wrong'),
                           lambda m: m['records'][2].update(subject_id='s0')):
                wrong = deepcopy(manifest)
                mutate(wrong)
                write_manifest(path, wrong)
                with self.assertRaises(ValueError):
                    FeatureCorpus(config, path, root)
            write_manifest(path, manifest)
            for value in (-1., float('nan'), 82.):
                np.save(Path(root)/manifest['records'][0]['expert_age_ms'], np.full((125,3), value, np.float32))
                with self.assertRaises(ValueError):
                    FeatureCorpus(config, path, root)

    def test_missing_dependencies_quality_and_unsupported_fields(self):
        config = self.api()
        from training.relation.preprocessing import prepare_window, dependency_masks
        from training.common.normalization import fit_normalizer
        p = config['data']['group_profile']
        values = np.ones((125,19), np.float32)
        masks = np.ones((125,19), np.uint8)
        masks[:,5:8] = 0
        ages = np.zeros((125,3), np.float32)
        ages[-1,2] = 81
        filtered = dependency_masks(masks, ages, p)
        self.assertFalse(filtered[-1,13:].any())
        self.assertTrue(filtered[-1,[0,8,10]].all())
        normalizer = fit_normalizer([(values,filtered)], config['data']['feature_names'])
        prepared = prepare_window(values[-100:], filtered[-100:], ages[-100:], normalizer, p)
        self.assertEqual(prepared['eligible'].tolist(), [True,True,False,True])
        self.assertTrue((prepared['x'][:,24:27] == 0).all())
        ages[-1,0] = 81
        filtered = dependency_masks(masks, ages, p)
        prepared = prepare_window(values[-100:],filtered[-100:],ages[-100:],normalizer,p)
        self.assertEqual(prepared['eligible'].tolist(), [True,False,False,False])
        ages[:] = 81
        filtered = dependency_masks(masks, ages, p)
        prepared = prepare_window(values[-100:],filtered[-100:],ages[-100:],normalizer,p)
        self.assertFalse(prepared['eligible'].any())
        self.assertTrue(np.isfinite(prepared['quality']).all())

    def test_history_is_invalidated_after_a_gap_and_expiry_boundary(self):
        config = self.api()
        from training.relation.preprocessing import dependency_masks, prepare_window
        from training.common.normalization import fit_normalizer
        p = config['data']['group_profile']
        masks = np.ones((201,19), np.uint8)
        ages = np.zeros((201,3), np.float32)
        ages[50,2] = 81
        filtered = dependency_masks(masks,ages,p)
        self.assertFalse(filtered[50:150,15].any())
        self.assertTrue(filtered[150,15])
        self.assertTrue(filtered[51,13])
        ages[:] = 80
        v = np.zeros((201,19),np.float32)
        norm = fit_normalizer([(v,masks)],config['data']['feature_names'])
        prepared = prepare_window(v[-100:],masks[-100:],ages[-100:],norm,p)
        self.assertTrue(prepared['eligible'].all())
        self.assertTrue((prepared['quality'][:,1] == 0).all())

    def test_positive_labels_with_no_input_do_not_satisfy_training_coverage(self):
        config = self.api()
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as root:
            path,manifest = relation_fixture(root,config)
            # All positive training windows are unusable; negative ones remain valid.
            record = manifest['records'][1]
            np.save(Path(root)/record['expert_age_ms'],np.full((125,3),81,np.float32))
            corpus = FeatureCorpus(config,path,root)
            try:
                with self.assertRaisesRegex(ValueError,'positive|negative|usable'):
                    corpus.check_supervision(config['training']['active_tasks'])
            finally:
                corpus.close()


if __name__ == '__main__':
    unittest.main()
