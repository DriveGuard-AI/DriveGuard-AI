"""Authored array fixtures exercise the real training manifest gate."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from inference.perception.profile import load_profile
from training.common.config import TASKS, canonical_hash, load_config, read_json
from training.common.data import FeatureCorpus, file_sha256

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def cache_fixture(root, split='train', identity='1', *, positive=True, available=True,
                  actual_export=False):
    """Entirely authored data, never copied from DMD."""
    config = load_config(ROOT / 'configs/training/relation_gated.json')
    attempt = Path(root) / 'cache' / identity
    features = attempt / 'features'
    features.mkdir(parents=True)
    slots = 125
    for name, array in {
        'values': np.full((slots, 19), 2 if split == 'train' else 100, np.float32),
        'masks': np.full((slots, 19), available, np.uint8),
        'expert_age_ms': np.zeros((slots, 3), np.float32),
        'targets': np.zeros((slots, 4), np.float32),
        'label_masks': np.zeros((slots, 4), np.uint8),
    }.items():
        np.save(features / (name + '.npy'), array)
    profile = load_profile()
    raw_paths = {name: name + '.npy' for name in
                 ('values', 'masks', 'targets', 'label_masks', 'expert_age_ms')}
    feature = {
        'schema_version': 'driveguard-feature-record-v2', 'status': 'complete_unlabeled',
        'slots': slots, 'recording_id': identity, 'view': 'body',
        'first_timestamp_ms': 0, 'last_timestamp_ms': (slots - 1) * 40,
        'image_size': [640, 480], 'driver_roi': [0, 0, 640, 480],
        'expert_models': {'face': 'authored', 'objects': 'authored', 'hands': 'authored'},
        'feature_schema': {'version': 'driveguard-interaction-v1',
                           'names': config['data']['feature_names'], 'sample_rate_hz': 25},
        'extractor_profile': profile, 'profile_sha256': canonical_hash(profile),
        'group_profile': config['data']['group_profile'],
        'group_profile_sha256': canonical_hash(config['data']['group_profile']),
        'arrays': raw_paths,
        'array_sha256': {name: file_sha256(features / path) for name, path in raw_paths.items()},
        'labels': 'All unknown',
    }
    save_json(features / 'record.json', feature)
    if actual_export:
        from inference.perception.export import export_recording
        from tests.perception_helpers import packet
        def packets():
            for index in range(slots):
                value = packet(index)
                value['recording_id'] = identity
                yield value
        exported = attempt / 'exported'
        feature = export_recording(packets(), exported, profile,
                                   group_profile=config['data']['group_profile'])
        # Replace only this fixture's authored files with a genuine exporter result.
        for source in exported.iterdir():
            source.replace(features / source.name)
        exported.rmdir()
    targets = np.zeros((slots, 4), np.float32)
    targets[124] = int(positive)
    np.save(attempt / 'targets.npy', targets)
    np.save(attempt / 'label_masks.npy', np.ones((slots, 4), np.uint8))
    np.save(attempt / 'source_pts_ms.npy', np.arange(slots, dtype=np.int64) * 40)
    paths = {'values': 'features/values.npy', 'masks': 'features/masks.npy',
             'expert_age_ms': 'features/expert_age_ms.npy', 'targets': 'targets.npy',
             'label_masks': 'label_masks.npy', 'source_pts_ms': 'source_pts_ms.npy'}
    report = {
        'schema_version': 'driveguard-dmd-complete-v1', 'fingerprint': canonical_hash(identity),
        'job': {'job_id': identity, 'subject_id': identity, 'recording_id': identity,
                'view': 'body', 'subset': 'distraction', 'split': split},
        'feature_record': feature, 'arrays': paths,
        'array_sha256': {name: file_sha256(attempt / path) for name, path in paths.items()},
        'label_summary': {'authored': True},
    }
    save_json(attempt / 'complete.json', report)
    return attempt, report


class DmdCacheTests(unittest.TestCase):
    def api(self):
        self.assertTrue((ROOT / 'training/dmd/cache.py').is_file(), 'Cache API not implemented')
        return importlib.import_module('training.dmd.cache')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = load_config(ROOT / 'configs/training/relation_gated.json')
        self.policy = {'closed_eye_min_ms': 1000, 'meaning': 'authored event proxy'}

    def test_json_writer_exclusively_creates_and_rejects_nonfinite(self):
        api = self.api()
        target = self.root / 'new.json'
        api.write_json(target, {'valid': 1})
        with self.assertRaises(FileExistsError):
            api.write_json(target, {'replacement': 2})
        self.assertEqual(read_json(target), {'valid': 1})
        with self.assertRaises(ValueError):
            api.write_json(self.root / 'nan.json', {'invalid': float('nan')})

    def test_valid_completion_and_corrupted_array(self):
        api = self.api()
        attempt, report = cache_fixture(self.root)
        self.assertEqual(api.validate_complete(attempt, report['fingerprint']), report)
        self.assertIsNone(api.validate_complete(attempt, 'f' * 64))
        with (attempt / 'targets.npy').open('ab') as handle:
            handle.write(b'corruption')
        self.assertIsNone(api.validate_complete(attempt, report['fingerprint']))

    def test_failure_marker_invalidates_an_otherwise_complete_attempt(self):
        api = self.api()
        attempt, report = cache_fixture(self.root)
        save_json(attempt / 'failure.json', {'status': 'failed'})
        self.assertIsNone(api.validate_complete(attempt, report['fingerprint']))
        with self.assertRaises(ValueError):
            api.publish_manifest(self.root, [(attempt, report)], self.config, self.policy)

    def test_invalid_completion_metadata_paths_and_source_pts(self):
        api = self.api()
        attempt, report = cache_fixture(self.root)
        mutations = [
            lambda r: r['arrays'].update(values='../outside.npy'),
            lambda r: r['arrays'].update(targets='features/targets.npy'),
            lambda r: r['job'].update(recording_id='other'),
            lambda r: r['feature_record'].update(status='incomplete'),
            lambda r: r['feature_record'].update(slots=126),
            lambda r: r.update(label_summary=[]),
            lambda r: r['arrays'].update(extra='extra.npy'),
        ]
        for mutate in mutations:
            broken = deepcopy(report)
            mutate(broken)
            save_json(attempt / 'complete.json', broken)
            self.assertIsNone(api.validate_complete(attempt, report['fingerprint']))
        for pts in (np.arange(125, dtype=np.int64), np.zeros(125, np.int64),
                    np.arange(125, dtype=np.float64) * 40):
            np.save(attempt / 'source_pts_ms.npy', pts)
            report['array_sha256']['source_pts_ms'] = file_sha256(attempt / 'source_pts_ms.npy')
            save_json(attempt / 'complete.json', report)
            self.assertIsNone(api.validate_complete(attempt, report['fingerprint']))

    def test_raw_export_arrays_stay_unknown_and_hash_checked(self):
        api = self.api()
        attempt, report = cache_fixture(self.root)
        np.save(attempt / 'features/targets.npy', np.ones((125, 4), np.float32))
        self.assertIsNone(api.validate_complete(attempt, report['fingerprint']))
        report['feature_record']['array_sha256']['targets'] = file_sha256(attempt / 'features/targets.npy')
        save_json(attempt / 'features/record.json', report['feature_record'])
        save_json(attempt / 'complete.json', report)
        self.assertIsNone(api.validate_complete(attempt, report['fingerprint']))

    def test_actual_feature_export_is_accepted(self):
        api = self.api()
        attempt, report = cache_fixture(self.root, actual_export=True)
        self.assertEqual(api.validate_complete(attempt, report['fingerprint']), report)

    def test_missing_export_metadata_is_rejected(self):
        api = self.api()
        attempt, report = cache_fixture(self.root)
        for key in ('image_size', 'driver_roi', 'expert_models', 'labels'):
            broken = deepcopy(report)
            del broken['feature_record'][key]
            save_json(attempt / 'features/record.json', broken['feature_record'])
            save_json(attempt / 'complete.json', broken)
            self.assertIsNone(api.validate_complete(attempt, report['fingerprint']))

    def test_disguised_npz_is_rejected(self):
        api = self.api()
        attempt, report = cache_fixture(self.root)
        with (attempt / 'targets.npy').open('wb') as handle:
            np.savez(handle, data=np.zeros((125, 4), np.float32))
        report['array_sha256']['targets'] = file_sha256(attempt / 'targets.npy')
        save_json(attempt / 'complete.json', report)
        self.assertIsNone(api.validate_complete(attempt, report['fingerprint']))

    def test_symlink_attempt_is_not_a_reusable_cache(self):
        api = self.api()
        attempt, report = cache_fixture(self.root)
        linked = self.root / 'linked'
        try:
            linked.symlink_to(attempt, target_is_directory=True)
        except OSError:
            self.skipTest('Creating symlinks requires Windows developer mode or privilege')
        self.assertIsNone(api.validate_complete(linked, report['fingerprint']))

    def test_publish_real_corpus_with_train_only_normalizer_and_provenance(self):
        api = self.api()
        completed = [cache_fixture(self.root, 'train', '1'), cache_fixture(self.root, 'val', '2')]
        result = api.publish_manifest(self.root, completed, self.config, self.policy)
        manifest_path = Path(result['manifest'])
        self.assertTrue(manifest_path.is_absolute())
        self.assertTrue(manifest_path.is_file())
        self.assertFalse(manifest_path.with_name('candidate.json').exists())
        corpus = FeatureCorpus(self.config, manifest_path, self.root)
        try:
            normalizer = corpus.fit_normalizer()
            self.assertEqual(normalizer['mean'], [2] * 19)
            self.assertEqual(corpus.check_supervision(list(TASKS), normalizer), result['coverage'])
            self.assertEqual(corpus.manifest['label_mapping']['rules'], self.policy)
        finally:
            corpus.close()
        publication = read_json(manifest_path.with_name('report.json'))
        self.assertEqual(publication['status'], 'complete')
        self.assertEqual([entry['job_id'] for entry in publication['sources']], ['1', '2'])
        self.assertEqual(publication['sources'][0]['fingerprint'], completed[0][1]['fingerprint'])
        # Re-publication gets a new directory and preserves the previous artifact.
        again = api.publish_manifest(self.root, completed, self.config, self.policy)
        self.assertNotEqual(again['manifest'], result['manifest'])
        self.assertTrue(manifest_path.exists())

    def test_insufficient_or_unavailable_supervision_never_publishes(self):
        api = self.api()
        for number, options in enumerate(({'positive': False}, {'available': False})):
            output = self.root / str(number)
            completed = [cache_fixture(output, 'train', '1'), cache_fixture(output, 'val', '2', **options)]
            with self.assertRaisesRegex(ValueError, 'supervision|positive|negative') as failure:
                api.publish_manifest(output, completed, self.config, self.policy)
            self.assertFalse(list(output.rglob('manifest.json')))
            reports = list((output / 'manifests').glob('*/report.json'))
            self.assertEqual(len(reports), 1)
            self.assertIn(str(reports[0].resolve()), str(failure.exception))
            self.assertIsInstance(failure.exception.__cause__, ValueError)
            diagnostic = read_json(reports[0])
            self.assertEqual(diagnostic['status'], 'insufficient_supervision')
            self.assertEqual(diagnostic['coverage']['val']['positive'], [0] * 4)

    def test_rejects_profile_disagreement_duplicates_empty_and_untrusted_report(self):
        api = self.api()
        first = cache_fixture(self.root, 'train', '1')
        second = cache_fixture(self.root, 'val', '2')
        for completed in ([], [first, first], [(first[0], {})]):
            with self.assertRaises(ValueError):
                api.publish_manifest(self.root, completed, self.config, self.policy)
        report = second[1]
        report['feature_record']['extractor_profile']['name'] = 'different valid profile'
        report['feature_record']['profile_sha256'] = canonical_hash(report['feature_record']['extractor_profile'])
        save_json(second[0] / 'features/record.json', report['feature_record'])
        save_json(second[0] / 'complete.json', report)
        with self.assertRaisesRegex(ValueError, 'profile'):
            api.publish_manifest(self.root, [first, second], self.config, self.policy)
        self.assertFalse(list(self.root.rglob('manifest.json')))


if __name__ == '__main__':
    unittest.main()
