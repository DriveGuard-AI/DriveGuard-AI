"""Spatial rejection must precede eye features, without changing the baseline."""
from copy import deepcopy
import json
import shutil
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from inference.experts.assets import file_sha256
from inference.experts.mediapipe_adapter import face_observation
from tests.test_face_diagnostic_replay import replay_fixture
from tests.test_face_experiments import capture_tasks
from tests.test_fatigue_replay import diagnostic
from training.dmd.face_experiment_config import BASELINE, validate_config


def region_config(source, roi=None):
    baseline = dict(BASELINE, head_region_mode='none')
    return dict(schema_version='driveguard-face-experiments-v2',
                variants=[baseline, dict(baseline, name='head_gate', head_region_mode='gate'),
                          dict(baseline, name='head_crop', head_region_mode='crop')],
                head_regions=[dict(record_id='job1', video_sha256=file_sha256(source),
                                   image_size=[1000, 500], head_roi_xyxy=roi or [.05, .05, .2, .2])])


class FaceHeadRegionTests(unittest.TestCase):
    def test_multiple_records_use_their_own_regions_and_missing_scope_fails(self):
        from training.common.config import canonical_hash
        from training.dmd.face_experiments import run_experiments
        from training.dmd.fatigue_recovery import run
        with tempfile.TemporaryDirectory(dir='runs') as tmp, replay_fixture(tmp) as (kwargs, _, source):
            root = Path(tmp); cache = kwargs['data_root']
            second = cache/'records/job2/attempt1'
            shutil.copytree(cache/'records/job1/attempt1', second)
            inputs = json.loads((second/'input.json').read_text()); inputs['job']['job_id'] = 'job2'
            (second/'input.json').write_text(json.dumps(inputs))
            complete = json.loads((second/'complete.json').read_text())
            complete.update(job=inputs['job'], fingerprint=canonical_hash(inputs))
            (second/'complete.json').write_text(json.dumps(complete))
            manifest = json.loads(kwargs['manifest'].read_text())
            record = deepcopy(manifest['records'][0]); record['record_id'] = 'job2'
            for key, value in record.items():
                if isinstance(value, str): record[key] = value.replace('records/job1/', 'records/job2/')
            manifest['records'].append(record); kwargs['manifest'].write_text(json.dumps(manifest))
            config = region_config(source)
            with self.assertRaisesRegex(ValueError, 'scope'):
                run_experiments(**kwargs, experiment_config=config, output_dir=root/'bad', samples_per_reason=0)
            self.assertFalse((root/'bad').exists())
            config['head_regions'].append(dict(config['head_regions'][0], record_id='job2',
                                                head_roi_xyxy=[.1, .05, .7, .9]))
            evidence = root/'diagnostic.json'; evidence.write_text(json.dumps(diagnostic(kwargs)))
            with capture_tasks() as tasks:
                report = run(diagnostic_summary=evidence, dataset_root=kwargs['dataset_root'],
                             experiment_config=config, output_dir=root/'report', samples_per_reason=0)
            self.assertEqual(len(tasks), 6)
            gates = [r for r in report['records'] if r['variant']['name'] == 'head_gate']
            self.assertEqual([r['fatigue_evidence']['frames']['all']['variant_eye_pair'] for r in gates], [0, 100])
            self.assertEqual(report['variants']['head_gate']['fatigue_evidence']['windows']['all']['windows'], 2)

    def test_spatial_gate_boundary_and_ambiguity_are_not_relaxed(self):
        head = [SimpleNamespace(x=.2, y=.2)] * 478
        boundary = face_observation([head], 1000, 500, [0, 0, 1000, 500], True, 0, 'id',
                                    head_roi=[200, 100, 400, 250])
        self.assertIsNotNone(boundary['landmarks'])
        ambiguous = face_observation([head, head], 1000, 500, [0, 0, 1000, 500], True, 0, 'id',
                                     head_roi=[100, 50, 400, 250])
        self.assertIsNone(ambiguous['landmarks'])
        for invalid in ([0, 0, float('nan'), 200], [0, 0, 1100, 500], [10, 20, 1, 2], [0, 0, 1]):
            with self.subTest(region=invalid), self.assertRaises(ValueError):
                face_observation([head], 1000, 500, [0, 0, 1000, 500], True, 0, 'id', head_roi=invalid)

    def test_prepare_selects_positive_records_and_builds_bound_three_arm_config(self):
        from training.dmd.fatigue_head_regions import prepare, build
        with tempfile.TemporaryDirectory(dir='runs') as tmp, replay_fixture(tmp) as (kwargs, closed, source):
            root = Path(tmp)
            evidence = root/'diagnostic.json'; evidence.write_text(json.dumps(diagnostic(kwargs)))
            recovery = dict(schema_version='driveguard-fatigue-recovery-report-v1', status='complete', partial=False,
                manifest=str(kwargs['manifest']), fatigue_settings=dict(diagnostic_summary=str(evidence),
                diagnostic_sha256=file_sha256(evidence)), records=[dict(record_id='job1', view='body',
                variant=dict(BASELINE), fatigue_evidence=dict(baseline_fatigue_reproduced=True,
                windows=dict(known_positive=dict(windows=1))))])
            previous = root/'recovery.json'; previous.write_text(json.dumps(recovery))
            tasks_before = closed.count('task')
            with patch('training.dmd.face_crop_preview._save_frame', lambda p, f: p.write_bytes(b'preview')):
                result = prepare(recovery_summary=previous, dataset_root=kwargs['dataset_root'],
                                 output_dir=root/'previews', samples=4)
            self.assertEqual(closed.count('task'), tasks_before)
            self.assertEqual(result['record_ids'], ['job1'])
            picker = root/'previews/record-001/index.html'
            self.assertTrue(picker.is_file())
            scope = json.loads((picker.parent/'summary.json').read_text())['crop_scope']
            export = dict(schema_version='driveguard-face-experiments-v1', crop_scope=scope,
                          variants=[dict(BASELINE), dict(BASELINE, name='head_crop', crop_xyxy=[.1, .05, .7, .9])])
            crop_file = root/'reviewed.json'; crop_file.write_text(json.dumps(export))
            assembled = build(crop_configs=[crop_file], output=root/'config.json')
            self.assertEqual([v['head_region_mode'] for v in assembled['variants']], ['none', 'gate', 'crop'])
            self.assertEqual(assembled['head_regions'][0]['head_roi_xyxy'], [.1, .05, .7, .9])
            self.assertEqual(validate_config(assembled), assembled)
            with self.assertRaises(FileExistsError): build(crop_configs=[crop_file], output=root/'config.json')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                build(crop_configs=[crop_file, crop_file], output=root/'bad.json')
            self.assertFalse((root/'bad.json').exists())

    def test_prepare_rejects_unreproduced_or_partial_source(self):
        from training.dmd.fatigue_head_regions import prepare
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root = Path(tmp); source = root/'report.json'
            source.write_text(json.dumps(dict(schema_version='driveguard-fatigue-recovery-report-v1',
                                              status='complete', partial=True)))
            with self.assertRaisesRegex(ValueError, 'complete|partial'):
                prepare(recovery_summary=source, dataset_root=root, output_dir=root/'out')
            self.assertFalse((root/'out').exists())
            source.write_text(json.dumps(dict(schema_version='driveguard-fatigue-recovery-report-v1',
                status='complete', partial=False, records=[dict(record_id='job1', view='body',
                variant=dict(BASELINE), fatigue_evidence=dict(baseline_fatigue_reproduced=False,
                windows=dict(known_positive=dict(windows=1))))])))
            with self.assertRaisesRegex(ValueError, 'reproduced'):
                prepare(recovery_summary=source, dataset_root=root, output_dir=root/'out')
            self.assertFalse((root/'out').exists())

    def test_region_preserves_valid_features_without_crop_and_no_input_mutation(self):
        from training.dmd.fatigue_recovery import run
        with tempfile.TemporaryDirectory(dir='runs') as tmp, replay_fixture(tmp) as (kwargs, _, source):
            evidence = Path(tmp)/'diagnostic.json'; evidence.write_text(json.dumps(diagnostic(kwargs)))
            before = {str(p): file_sha256(p) for p in Path(tmp).rglob('*') if p.is_file()}
            config = region_config(source, [.1, .05, .7, .9])
            with capture_tasks():
                report = run(diagnostic_summary=evidence, dataset_root=kwargs['dataset_root'],
                             experiment_config=config, output_dir=Path(tmp)/'report', samples_per_reason=0)
            gate = report['records'][1]['fatigue_evidence']
            self.assertEqual(gate['raw_mask_mismatches'], 0)
            self.assertEqual(gate['raw_value_mismatches'], 0)
            self.assertEqual(gate['frames']['all']['variant_eye_pair'], 100)
            self.assertEqual(gate['frames']['all']['eye_pair_outside_head_region'], 0)
            self.assertEqual(before, {p: file_sha256(p) for p in before})

    def test_hand_rejected_head_kept_and_matrix_index_preserved(self):
        hand = [SimpleNamespace(x=.8, y=.8)] * 478
        head = [SimpleNamespace(x=.2, y=.2)] * 478
        matrices = [np.eye(4).tolist(), np.eye(4).tolist()]
        old = face_observation([hand], 1000, 500, [0, 0, 1000, 500], True, 0, 'id', matrices=matrices[:1])
        self.assertIsNotNone(old['landmarks'])  # Reproduce the existing acceptance hole.
        rejected = face_observation([hand], 1000, 500, [0, 0, 1000, 500], True, 0, 'id',
                                    matrices=matrices[:1], head_roi=[100, 50, 400, 250])
        self.assertIsNone(rejected['landmarks'])
        self.assertIsNone(rejected['head_pose'])
        self.assertFalse(rejected['driver_confirmed'])
        selected = face_observation([hand, head], 1000, 500, [0, 0, 1000, 500], True, 0, 'id',
                                   matrices=[None, matrices[1]], head_roi=[100, 50, 400, 250])
        self.assertEqual(selected['landmarks'][0], [200., 100.])
        self.assertIsNotNone(selected['head_pose'])

    def test_scope_checks_before_creating_output(self):
        from training.dmd.face_experiments import run_experiments
        with tempfile.TemporaryDirectory(dir='runs') as tmp, replay_fixture(tmp) as (kwargs, _, source):
            config = region_config(source)
            self.assertEqual(validate_config(config), config)
            for key, value in [('record_id', 'wrong'), ('video_sha256', '0'*64), ('image_size', [99, 50])]:
                bad = deepcopy(config); bad['head_regions'][0][key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'scope|region'):
                    run_experiments(**kwargs, experiment_config=bad, output_dir=Path(tmp)/key, samples_per_reason=0)
                self.assertFalse((Path(tmp)/key).exists())
            for change in ('duplicate', 'bounds', 'mixed'):
                bad = deepcopy(config)
                if change == 'duplicate': bad['head_regions'] *= 2
                if change == 'bounds': bad['head_regions'][0]['head_roi_xyxy'] = [0, 0, 2, 1]
                if change == 'mixed': bad['variants'][1]['detection'] = .3
                with self.subTest(change=change), self.assertRaises(ValueError): validate_config(bad)

    def test_full_replay_gate_blocks_numeric_eye_features_and_crop_remaps(self):
        from training.dmd.fatigue_recovery import run
        with tempfile.TemporaryDirectory(dir='runs') as tmp, replay_fixture(tmp) as (kwargs, _, source):
            evidence = Path(tmp)/'diagnostic.json'; evidence.write_text(json.dumps(diagnostic(kwargs)))
            with capture_tasks() as tasks:
                report = run(diagnostic_summary=evidence, dataset_root=kwargs['dataset_root'],
                             experiment_config=region_config(source), output_dir=Path(tmp)/'report', samples_per_reason=0)
            baseline, gate, crop = report['records']
            self.assertTrue(baseline['fatigue_evidence']['baseline_fatigue_reproduced'])
            self.assertEqual(gate['reason_counts'], {'outside_head_region': 100})
            for row in (baseline, gate, crop):
                self.assertEqual(row['fatigue_evidence']['windows']['all']['windows'], 1)
            b = baseline['fatigue_evidence']['frames']['known_negative']
            g = gate['fatigue_evidence']['frames']['known_negative']
            self.assertEqual(b['eye_pair_outside_head_region'], 100)
            self.assertEqual(g['variant_eye_pair'], 0)
            self.assertEqual(g['head_region_rejected_slots'], 100)
            self.assertEqual(gate['fatigue_evidence']['windows']['all']['variant_f_eligible'], 0)
            self.assertEqual(tasks[0][1][0][2], tasks[1][1][0][2])
            self.assertEqual(tasks[2][1][0][2], (75, 150, 3))
            self.assertEqual(crop['pose_coordinate_frame'], 'crop-experimental')
            self.assertFalse(crop['training_cache_compatible'])
            candidates = Path(tmp)/'report'/crop['directory']/'candidates.jsonl'
            detail = json.loads(candidates.read_text().splitlines()[0])
            self.assertEqual(detail['head_region_pixels'], [50, 25, 200, 100])
            self.assertTrue(detail['candidates'][0]['inside_head_region'])
            self.assertIn('eye_pair_outside_head_region', (Path(tmp)/'report/fatigue_comparison.csv').read_text())


if __name__ == '__main__': unittest.main()
