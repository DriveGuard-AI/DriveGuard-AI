"""Batch integration with authored annotations and synthetic perception only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from training.common.config import canonical_hash, read_json, load_config
from training.dmd.prepare import prepare_dataset
from training.dmd.audit import audit_dataset
from tests.test_dmd_labels import document

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'runs')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.dataset = self.base / 'Dataset'
        for subject in ('1', '7'):
            for subset, session in [('distraction', 's1'), ('drowsiness', 's5'), ('gaze', 's6')]:
                prefix = f'gA_{subject}_{session}_2026-09-21T00;00;00'
                folder = self.dataset / subset / 'dmd/gA' / subject / session
                folder.mkdir(parents=True)
                actions = [('driver_actions/safe_drive', 0, 99),
                    ('driver_actions/texting_left', 100, 199), ('driver_actions/drinking', 100, 199),
                    ('gaze_on_road/looking_road', 0, 99), ('gaze_on_road/not_looking_road', 100, 199)]
                if subset == 'drowsiness':
                    actions = [('eyes_state/open', 0, 99), ('yawning/Yawning without hand', 100, 199)]
                if subset == 'gaze':
                    actions = [('gaze_zone/front', 0, 199)]
                doc = document(*actions, count=200)
                doc['openlabel']['metadata']['name'] = prefix + '_' + subset
                doc['openlabel']['objects'] = {'0': {'type': 'driver', 'name': subject}}
                for view in ('face', 'body', 'hands'):
                    name = f'{prefix}_rgb_{view}.mp4'
                    doc['openlabel']['streams'][view + '_camera']['uri'] = f'gA/{subject}/{session}/{name}'
                    (folder / name).write_bytes(b'synthetic-video')
                write(folder / f'{prefix}_rgb_ann_{subset}.json', doc)
        audit_dataset(self.dataset, self.base / 'audit')
        self.prep = self.base / 'prepared/preparation.json'
        split = self.base / 'split.json'
        write(split, {'schema_version': 'driveguard-dmd-split-v1', 'train': ['1'], 'val': ['7']})
        prepare_dataset(audit_path=self.base / 'audit/audit.json', dataset_root=self.dataset,
                        output_dir=self.prep.parent, split_path=split)
        self.config_path = self.base / 'batch.local.json'
        self.output = self.base / 'cache'

    def api(self):
        self.assertTrue((ROOT / 'training/dmd/batch_config.py').is_file(), 'Batch config implementation missing')
        from training.dmd.batch_config import init_config, load_batch
        return init_config, load_batch

    def configure(self, confirm=True):
        self.api()[0](self.prep, self.config_path)
        config = read_json(self.config_path)
        for setting in config['jobs'].values():
            setting['confirmed_exclusive'] = confirm
        write(self.config_path, config)
        return config

    def test_init_selects_each_split_subset_view_and_excludes_gaze(self):
        config = self.configure(False)
        self.assertEqual(len(config['jobs']), 8)
        self.assertEqual(config['closed_eye_min_ms'], 1000)
        self.assertTrue(all(not j['confirmed_exclusive'] for j in config['jobs'].values()))
        loaded = self.api()[1](self.config_path)
        self.assertEqual({j['subset'] for j in loaded['selected']}, {'distraction', 'drowsiness'})
        self.assertEqual({j['split'] for j in loaded['selected']}, {'train', 'val'})
        with self.assertRaises(FileExistsError):
            self.api()[0](self.prep, self.config_path)

    def test_rejects_roi_invalid_and_unknown_job(self):
        config = self.configure()
        jid = next(iter(config['jobs']))
        for roi in ([0, 0, 2, 1], [0.8, 0, 0.2, 1], [False, 0, 1, 1]):
            with self.subTest(roi=roi):
                bad = copy.deepcopy(config)
                bad['jobs'][jid]['normalized_roi'] = roi
                write(self.config_path, bad)
                with self.assertRaises(ValueError):
                    self.api()[1](self.config_path)
        bad = copy.deepcopy(config)
        bad['jobs']['unknown'] = bad['jobs'].pop(jid)
        write(self.config_path, bad)
        with self.assertRaises(ValueError):
            self.api()[1](self.config_path)

    def test_rejects_changed_preparation(self):
        self.configure()
        prep = read_json(self.prep)
        prep['jobs'][0]['split'] = 'val'
        write(self.prep, prep)
        with self.assertRaisesRegex(ValueError, 'preparation|Preparation'):
            self.api()[1](self.config_path)

    def batch_api(self):
        self.assertTrue((ROOT / 'training/dmd/batch.py').is_file(), 'Batch runner implementation missing')
        from training.dmd.batch import run_batch, publish_batch
        return run_batch, publish_batch

    def test_unconfirmed_roi_blocks_before_loading_models(self):
        self.configure(False)
        with self.assertRaisesRegex(ValueError, 'ROI|exclusive'):
            self.batch_api()[0](self.config_path, self.dataset, self.output)
        self.assertFalse(self.output.exists())

    def fake_extract(self, source, output, runtime, *, group_profile, frame_observer):
        from inference.perception.export import export_recording
        from inference.perception.profile import load_profile
        from inference.perception.video_source import VideoFrame
        from tests.perception_helpers import packet
        def packets():
            for index in range(200):
                frame_observer(VideoFrame(index * 40, np.zeros((500, 1000, 3), dtype=np.uint8)))
                value = packet(index)
                value.update(recording_id=runtime['recording_id'], view=runtime['view'],
                             driver_roi=runtime['driver_roi'])
                yield value
        profile = load_profile()
        # Synthetic origin is explicit in the provenance, not a claimed real video test.
        return export_recording(packets(), output, profile, group_profile=group_profile,
                                provenance={'synthetic_test_only': True})

    def patches(self):
        from contextlib import ExitStack
        from inference.perception.video_source import VideoFrame
        stack = ExitStack()
        stack.enter_context(patch('training.dmd.batch.verify_asset', side_effect=lambda a: a))
        stack.enter_context(patch('training.dmd.batch.decode_video', side_effect=lambda p: iter([
            VideoFrame(0, np.zeros((500, 1000, 3), dtype=np.uint8))])))
        self.extract_mock = stack.enter_context(patch('training.dmd.batch.run_video', side_effect=self.fake_extract))
        return stack

    def test_run_resume_publish_and_cpu_training(self):
        self.configure()
        run, publish = self.batch_api()
        with self.patches():
            first = run(self.config_path, self.dataset, self.output)
            self.assertEqual(first['completed'], 8)
            self.assertEqual(first['failed'], 0)
            again = run(self.config_path, self.dataset, self.output)
            self.assertEqual(again['reused'], 8)
            self.assertEqual(self.extract_mock.call_count, 8)
            release = publish(self.config_path, self.dataset, self.output, ROOT / 'configs/training/relation_gated.json')
        from training.common.data import FeatureCorpus
        config = load_config(ROOT / 'configs/training/relation_gated.json')
        corpus = FeatureCorpus(config, release['manifest'], self.output)
        try:
            coverage = corpus.check_supervision(config['training']['active_tasks'])
            self.assertTrue(all(v > 0 for v in coverage['train']['positive']))
        finally:
            corpus.close()
        # One small CPU training epoch verifies the real preparation -> training contract.
        from training.temporal.train import run_training
        config['training']['epochs'] = 1
        config['runtime'].update(backend='cpu', manifest=release['manifest'],
                                 data_root=str(self.output), output_dir=str(self.base / 'train'))
        result = run_training(config)
        self.assertGreater(result['global_step'], 0)
        self.assertTrue(Path(result['best_checkpoint']).is_file())

    def test_failure_keeps_attempt_and_next_run_recovers(self):
        self.configure()
        run, _ = self.batch_api()
        with self.patches():
            self.extract_mock.side_effect = RuntimeError('simulated decoder failure')
            failed = run(self.config_path, self.dataset, self.output)
            self.assertEqual(failed['failed'], 8)
            self.assertFalse(list(self.output.glob('records/*/*/complete.json')))
            self.assertEqual(len(list(self.output.glob('records/*/*/failure.json'))), 8)
            self.extract_mock.side_effect = self.fake_extract
            passed = run(self.config_path, self.dataset, self.output)
            self.assertEqual(passed['completed'], 8)
            self.assertEqual(len(list(self.output.glob('records/*/*/failure.json'))), 8)

    def test_source_or_threshold_change_never_reuses_cache(self):
        self.configure()
        run, _ = self.batch_api()
        with self.patches():
            run(self.config_path, self.dataset, self.output)
            config = read_json(self.config_path)
            config['closed_eye_min_ms'] = 1200
            write(self.config_path, config)
            changed = run(self.config_path, self.dataset, self.output)
            self.assertEqual(changed['reused'], 0)
            self.assertEqual(changed['completed'], 8)

    def test_never_writes_inside_dataset(self):
        self.configure()
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.batch_api()[0](self.config_path, self.dataset, self.dataset / 'output')

    def test_completion_identity_edit_invalidates_reuse(self):
        self.configure()
        run, _ = self.batch_api()
        with self.patches():
            first = run(self.config_path, self.dataset, self.output)
            path = self.output / first['jobs'][0]['attempt'] / 'complete.json'
            report = read_json(path)
            report['job']['subject_id'] = '99'
            write(path, report)
            again = run(self.config_path, self.dataset, self.output)
            self.assertEqual(again['reused'], 7)
            self.assertEqual(again['completed'], 1)

    def test_same_size_video_change_reextracts_only_affected_job(self):
        self.configure()
        run, _ = self.batch_api()
        with self.patches():
            run(self.config_path, self.dataset, self.output)
            selected = self.api()[1](self.config_path)['selected'][0]
            video = self.dataset / selected['video']
            video.write_bytes(b'x' * video.stat().st_size)
            again = run(self.config_path, self.dataset, self.output)
            self.assertEqual((again['reused'], again['completed']), (7, 1))

    def test_does_not_steal_existing_lock(self):
        self.configure()
        run, _ = self.batch_api()
        self.output.mkdir()
        lock = self.output / '.batch.lock'
        lock.write_text('another process', encoding='utf-8')
        with self.patches(), self.assertRaises(FileExistsError):
            run(self.config_path, self.dataset, self.output)
        self.assertEqual(lock.read_text(encoding='utf-8'), 'another process')

    def test_legacy_null_threshold_is_explicitly_replaced_by_batch_default(self):
        prep = read_json(self.prep)
        prep['label_policy']['closed_eye_min_ms'] = None
        prep['label_policy_sha256'] = canonical_hash(prep['label_policy'])
        write(self.prep, prep)
        config = self.configure()
        self.assertEqual(config['closed_eye_min_ms'], 1000)
        self.assertEqual(self.api()[1](self.config_path)['label_policy']['closed_eye_min_ms'], 1000)

    def test_model_index_error_is_logged_and_other_jobs_continue(self):
        self.configure()
        run, _ = self.batch_api()
        with self.patches():
            self.extract_mock.side_effect = IndexError('decoder or model failure')
            result = run(self.config_path, self.dataset, self.output)
            self.assertEqual(result['failed'], 8)
            self.assertEqual(len(list(self.output.glob('records/*/*/failure.json'))), 8)

    def test_init_preserves_explicit_preparation_threshold(self):
        prep = read_json(self.prep)
        prep['label_policy']['closed_eye_min_ms'] = 2500
        prep['label_policy_sha256'] = canonical_hash(prep['label_policy'])
        write(self.prep, prep)
        config = self.configure()
        self.assertEqual(config['closed_eye_min_ms'], 2500)
        self.assertEqual(self.api()[1](self.config_path)['label_policy']['closed_eye_min_ms'], 2500)


if __name__ == '__main__':
    unittest.main()
