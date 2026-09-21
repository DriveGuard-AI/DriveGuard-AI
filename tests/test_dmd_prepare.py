"""Authored tiny DMD layout; preparation must not decode or import ML."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tests.test_dmd_audit import document, PREFIX
from training.dmd.audit import audit_dataset

ROOT = Path(__file__).resolve().parents[1]


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.dataset = self.base / 'Dataset'
        self.folder = self.dataset / 'gaze/dmd/gA/1/s6'
        self.folder.mkdir(parents=True)
        doc = document()
        for view, shift in [('body', 1), ('hands', 2)]:
            doc['openlabel']['streams'][view + '_camera'] = {
                'uri': f'gA/1/s6/{PREFIX}_rgb_{view}.mp4',
                'stream_properties': {'total_frames': 4 - shift, 'sync': {'frame_shift': shift}}}
        self.annotation = self.folder / f'{PREFIX}_rgb_ann_gaze.json'
        self.annotation.write_text(json.dumps(doc), encoding='utf-8')
        # Deliberately not decodable videos: preparation reads only stat and annotations.
        for view in ('face', 'body'):
            (self.folder / f'{PREFIX}_rgb_{view}.mp4').write_bytes(b'not-a-real-video')
        audit_dataset(self.dataset, self.base / 'audit')
        self.audit = self.base / 'audit/audit.json'
        self.split = {'schema_version': 'driveguard-dmd-split-v1', 'train': ['1'], 'val': ['7']}
        self.split_path = self.base / 'split.json'
        self.split_path.write_text(json.dumps(self.split), encoding='utf-8')

    def api(self):
        self.assertTrue((ROOT / 'training/dmd/prepare.py').exists(), 'Preparation CLI is not implemented')
        from training.dmd.prepare import prepare_dataset
        return prepare_dataset

    def run_prepare(self, **kwargs):
        options = dict(audit_path=self.audit, dataset_root=self.dataset,
                       output_dir=self.base / 'prepared', split_path=self.split_path)
        options.update(kwargs)
        return self.api()(**options)

    def change_split(self, **values):
        self.split.update(values)
        self.split_path.write_text(json.dumps(self.split), encoding='utf-8')

    def change_audit(self, change):
        report = json.loads(self.audit.read_text(encoding='utf-8'))
        change(report)
        self.audit.write_text(json.dumps(report), encoding='utf-8')

    def test_portable_jobs_are_not_training_manifest(self):
        result = self.run_prepare()
        self.assertEqual(result['schema_version'], 'driveguard-dmd-preparation-v1')
        self.assertFalse(result['training_ready'])
        self.assertFalse(result['video_decoded'])
        self.assertEqual(len(result['jobs']), 2)
        self.assertEqual({j['view'] for j in result['jobs']}, {'face', 'body'})
        for job in result['jobs']:
            self.assertEqual(job['split'], 'train')
            self.assertEqual(job['subject_id'], '1')
            self.assertFalse(Path(job['video']).is_absolute())
            self.assertIsNone(job['driver_roi'])
            self.assertFalse(job['driver_region_confirmed_exclusive'])
            self.assertEqual(job['supervised_tasks'], [])
            self.assertEqual(len(job['annotation_sha256']), 64)
        body = next(j for j in result['jobs'] if j['view'] == 'body')
        self.assertEqual(body['annotation_frame_offset'], 1)
        self.assertTrue((self.base / 'prepared/README.md').exists())
        self.assertFalse((self.base / 'prepared/manifest.json').exists())
        self.assertIsNone(result['label_policy']['closed_eye_min_ms'])
        self.assertEqual(result['split_summary']['configured_but_absent'], ['7'])

    def test_rejects_changed_annotation_without_publishing(self):
        self.annotation.write_text(self.annotation.read_text(encoding='utf-8') + ' ', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed|SHA|hash'):
            self.run_prepare()
        self.assertFalse((self.base / 'prepared').exists())

    def test_rejects_error_audit(self):
        self.change_audit(lambda r: r.update(status='errors'))
        with self.assertRaisesRegex(ValueError, 'audit|Audit'):
            self.run_prepare()

    def test_rejects_unassigned_or_duplicate_subjects(self):
        for train, val in [(['2'], ['7']), (['1'], ['1']), (['1', '1'], ['7'])]:
            with self.subTest(train=train, val=val):
                self.change_split(train=train, val=val)
                with self.assertRaises(ValueError):
                    self.run_prepare()

    def test_new_output_only_and_never_write_dataset(self):
        self.run_prepare()
        with self.assertRaises(FileExistsError):
            self.run_prepare()
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.run_prepare(output_dir=self.dataset / 'generated')

    def test_missing_existing_video_is_explicitly_skipped(self):
        (self.folder / f'{PREFIX}_rgb_body.mp4').unlink()
        result = self.run_prepare()
        self.assertEqual([j['view'] for j in result['jobs']], ['face'])
        self.assertTrue(any(s['reason'] == 'missing_video' and s['view'] == 'body'
                            for s in result['skipped']))

    def test_rejects_audit_path_escape(self):
        self.change_audit(lambda r: r['recordings'][0]['annotations'][0].update(path='../escape.json'))
        with self.assertRaisesRegex(ValueError, 'path|Path|relative'):
            self.run_prepare()

    def test_rejects_forged_identity(self):
        self.change_audit(lambda r: r['recordings'][0].update(subject_id='7'))
        with self.assertRaisesRegex(ValueError, 'identity|subject|Identity'):
            self.run_prepare()

    def test_cli_has_no_site_dependency(self):
        self.api()
        result = subprocess.run([sys.executable, '-S', '-m', 'training.dmd.prepare',
            '--audit', str(self.audit), '--dataset-root', str(self.dataset),
            '--split', str(self.split_path), '--output-dir', str(self.base / 'cli')],
            cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)['training_ready'])

    def test_distraction_views_have_independent_task_scope(self):
        # Change fixture subset/kind, retaining source labels to exercise planning only.
        target = self.dataset / 'distraction/dmd/gA/1/s6'
        target.mkdir(parents=True)
        doc = json.loads(self.annotation.read_text(encoding='utf-8'))
        doc['openlabel']['metadata']['name'] = PREFIX + '_distraction'
        (target / f'{PREFIX}_rgb_ann_distraction.json').write_text(json.dumps(doc), encoding='utf-8')
        for view in ('face', 'body'):
            (target / f'{PREFIX}_rgb_{view}.mp4').write_bytes(b'not-a-real-video')
        audit_dataset(self.dataset, self.base / 'audit2')
        result = self.run_prepare(audit_path=self.base / 'audit2/audit.json')
        self.assertEqual(len(result['jobs']), 4)
        self.assertEqual({j['split'] for j in result['jobs']}, {'train'})
        self.assertEqual(len({j['job_id'] for j in result['jobs']}), 4)
        for job in result['jobs']:
            if job['subset'] == 'distraction':
                self.assertEqual(job['supervised_tasks'], ['phone_use', 'drinking', 'distraction'])

    def test_threshold_must_be_explicit_positive_integer(self):
        for threshold in (0, -1, True, 0.5):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.run_prepare(closed_eye_min_ms=threshold)
        result = self.run_prepare(closed_eye_min_ms=1000)
        self.assertEqual(result['label_policy']['closed_eye_min_ms'], 1000)


if __name__ == '__main__':
    unittest.main()
