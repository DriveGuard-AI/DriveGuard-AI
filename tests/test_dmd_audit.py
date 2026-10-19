"""Small authored fixtures; never commit user datasets or require ML dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'gA_1_s6_2019-03-08T09;15;15+01;00'


def document():
    return {'openlabel': {
        'metadata': {'schema_version': '1.0.0', 'name': PREFIX + '_gaze'},
        'frame_intervals': [{'frame_start': 0, 'frame_end': 3}],
        'objects': {'0': {'type': 'driver', 'name': '1',
                          'object_data': {'num': [{'name': 'age', 'val': 47}]}}},
        'actions': {'0': {'type': 'gaze_zone/front', 'frame_intervals': [
            {'frame_start': 0, 'frame_end': 2}, {'frame_start': 2, 'frame_end': 3}]}},
        'frames': {str(i): {'actions': {'0': {}}} for i in range(4)},
        'streams': {'face_camera': {'uri': f'gA/1/s6/{PREFIX}_rgb_face.mp4',
            'stream_properties': {'total_frames': 4, 'sync': {'frame_shift': 0}}}},
    }}


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'Dataset'
        self.root.mkdir()

    def api(self):
        self.assertTrue((ROOT / 'training/dmd/audit.py').is_file(),
                        'DMD audit implementation is not present')
        from training.dmd.audit import audit_dataset
        from training.dmd.annotations import parse_annotation
        return audit_dataset, parse_annotation

    def annotation(self, doc=None, kind='gaze', subset='gaze', prefix=PREFIX):
        path = self.root / subset / 'dmd/gA/1/s6' / f'{prefix}_rgb_ann_{kind}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document() if doc is None else doc), encoding='utf-8')
        return path

    def scan(self):
        return self.api()[0](self.root, self.base / 'report')

    def codes(self, result):
        return {issue['code'] for issue in result['issues']}

    def test_inclusive_union_and_privacy(self):
        result = self.api()[1](self.annotation())
        self.assertEqual(result['labels'][0]['covered_frames'], 4)
        self.assertEqual(result['labels'][0]['interval_count'], 2)
        self.assertEqual(result['annotation_frames'], 4)
        self.assertEqual(len(result['sha256']), 64)
        self.assertNotIn('object_data', json.dumps(result))
        self.assertNotIn('age', result)
        self.assertEqual(result['driver_ids'], ['1'])
        self.assertNotIn('FRAME_ACTION_MISMATCH', self.codes(result))

    def test_invalid_intervals_and_unknown_ids(self):
        for interval in ({'frame_start': True, 'frame_end': 2},
                         {'frame_start': 3, 'frame_end': 1},
                         {'frame_start': -1, 'frame_end': 1},
                         {'frame_start': 0, 'frame_end': 1.5}):
            doc = document()
            doc['openlabel']['actions']['0']['frame_intervals'] = [interval]
            with self.subTest(interval=interval):
                result = self.api()[1](self.annotation(doc))
                self.assertIn('INVALID_INTERVALS', self.codes(result))
        doc = document()
        doc['openlabel']['frames']['0']['actions']['999'] = {}
        self.assertIn('UNDEFINED_ACTION', self.codes(self.api()[1](self.annotation(doc))))

    def test_frame_and_top_level_disagreements(self):
        doc = document()
        del doc['openlabel']['frames']['1']['actions']['0']
        doc['openlabel']['frame_intervals'][0]['frame_end'] = 2
        result = self.api()[1](self.annotation(doc))
        self.assertIn('FRAME_ACTION_MISMATCH', self.codes(result))
        self.assertIn('OUTSIDE_TIMELINE', self.codes(result))

    def test_multiple_label_axes_are_not_conflicts(self):
        doc = document()
        doc['openlabel']['actions']['1'] = {'type': 'blinks/blinking',
            'frame_intervals': [{'frame_start': 1, 'frame_end': 1}]}
        doc['openlabel']['frames']['1']['actions']['1'] = {}
        result = self.api()[1](self.annotation(doc))
        self.assertEqual(len(result['labels']), 2)
        self.assertFalse(any(i['severity'] == 'error' for i in result['issues']))

    def test_duplicate_keys_empty_and_invalid_documents(self):
        path = self.annotation()
        for content in ('', '{"openlabel":{},"openlabel":{}}', '{',
                        '{"openlabel":NaN}', '[]', '{"openlabel":{}}'):
            path.write_text(content, encoding='utf-8')
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.api()[1](path)

    def test_not_valid_preserved_without_binary_targets(self):
        doc = document()
        doc['openlabel']['actions']['0']['type'] = 'gaze_zone/not_valid'
        result = self.api()[1](self.annotation(doc))
        self.assertEqual(result['labels'][0]['type'], 'gaze_zone/not_valid')
        self.assertEqual(result['labels'][0]['covered_frames'], 4)
        self.assertNotIn('targets', result)

    def test_overflow_numeric_does_not_abort_other_reports(self):
        self.annotation()
        path = self.annotation(kind='hands')
        raw = path.read_text(encoding='utf-8').replace('"val": 47', '"val": 1e400')
        path.write_text(raw, encoding='utf-8')
        report = self.scan()
        self.assertEqual(report['status'], 'errors')
        self.assertEqual(report['counts']['parsed_annotations'], 1)
        self.assertTrue((self.base / 'report/audit.json').is_file())

    def test_metadata_name_must_be_text(self):
        doc = document()
        doc['openlabel']['metadata']['name'] = {'not': 'a name'}
        with self.assertRaises(ValueError):
            self.api()[1](self.annotation(doc))

    def test_unpaired_surrogate_is_a_per_file_error(self):
        self.annotation()
        doc = document()
        doc['openlabel']['metadata']['name'] = '\ud800'
        self.annotation(doc, kind='hands')
        report = self.scan()
        self.assertEqual(report['status'], 'errors')
        self.assertEqual(report['counts']['parsed_annotations'], 1)
        self.assertTrue((self.base / 'report/audit.md').is_file())

    def test_symlink_directory_is_skipped_without_walk(self):
        self.api()
        from training.dmd.audit import _local_files
        issues = []
        # Emulate an inaccessible symlink directory without Windows link privileges.
        with patch('training.dmd.audit.os.walk', return_value=iter([(str(self.root), ['linked'], [])])), \
                patch.object(Path, 'is_symlink', return_value=True):
            self.assertEqual(list(_local_files(self.root, issues)), [])
        self.assertEqual(issues[0]['code'], 'SYMLINK_SKIPPED')

    def test_looping_camera_reference_is_reported(self):
        self.api()
        from training.dmd.audit import _check_streams
        record = dict(subset='gaze', group='gA', subject_id='1', session='s6', recording_id=PREFIX)
        annotation = {'streams': [{'name': 'face_camera', 'uri': 'gA/1/s6/loop.mp4'}]}
        issues = []
        # Windows cannot always create symlinks; emulate Path.resolve's documented failure.
        with patch.object(Path, 'resolve', side_effect=RuntimeError('Symlink loop')):
            _check_streams(annotation, record, self.root, set(), issues, 'sample.json')
        self.assertIn('UNSAFE_URI', {i['code'] for i in issues})

    def test_duplicate_type_ids_share_union_coverage(self):
        doc = document()
        doc['openlabel']['actions']['1'] = {'type': 'gaze_zone/front',
            'frame_intervals': [{'frame_start': 1, 'frame_end': 2}]}
        for index in ('1', '2'):
            doc['openlabel']['frames'][index]['actions']['1'] = {}
        result = self.api()[1](self.annotation(doc))
        self.assertEqual(len(result['labels']), 1)
        self.assertEqual(result['labels'][0]['covered_frames'], 4)

    def test_unreviewed_type_is_preserved_and_flagged(self):
        doc = document()
        doc['openlabel']['actions']['0']['type'] = 'new_axis/new_action'
        result = self.api()[1](self.annotation(doc))
        self.assertIn('UNREVIEWED_LABEL_TYPE', self.codes(result))
        self.assertEqual(result['labels'][0]['covered_frames'], 4)

    def test_negative_shift_is_preserved_and_bad_stream_reported(self):
        doc = document()
        doc['openlabel']['streams']['face_camera']['stream_properties']['sync']['frame_shift'] = -3
        result = self.api()[1](self.annotation(doc))
        self.assertEqual(result['streams'][0]['frame_shift'], -3)
        doc['openlabel']['streams']['face_camera']['stream_properties']['sync']['frame_shift'] = True
        self.assertIn('INVALID_STREAM', self.codes(self.api()[1](self.annotation(doc))))

    def test_declared_frame_count_is_only_diagnostic(self):
        doc = document()
        doc['openlabel']['streams']['face_camera']['stream_properties']['total_frames'] = 3
        result = self.api()[1](self.annotation(doc))
        self.assertIn('FRAME_COUNT_DECLARATION', self.codes(result))
        self.assertEqual(result['streams'][0]['total_frames'], 3)
        self.assertFalse(result['video_decoded'])

    def test_pairs_annotations_and_multiple_recordings(self):
        self.annotation()
        self.annotation(kind='hands')
        other = PREFIX.replace('09;15;15', '10;15;15')
        self.annotation(prefix=other)
        report = self.scan()
        self.assertEqual(report['counts']['recordings'], 2)
        self.assertEqual(report['counts']['annotation_files'], 3)
        first = next(r for r in report['recordings'] if r['recording_id'] == PREFIX)
        self.assertEqual(len(first['annotations']), 2)
        self.assertEqual(report['subjects'][0]['subject_id'], '1')
        self.assertIn('MISSING_VIDEO', self.codes(report))
        self.assertTrue((self.base / 'report/audit.md').is_file())

    def test_subject_aggregation_across_subsets(self):
        self.annotation()
        self.annotation(subset='distraction', kind='distraction')
        report = self.scan()
        self.assertEqual(report['counts']['subjects'], 1)
        self.assertEqual(report['subjects'][0]['recordings'], 2)
        self.assertEqual(set(report['subjects'][0]['subsets']), {'gaze', 'distraction'})

    def test_bad_file_continues_and_output_protected(self):
        self.annotation()
        bad = self.annotation(kind='hands')
        bad.write_text('{', encoding='utf-8')
        report = self.scan()
        self.assertEqual(report['status'], 'errors')
        self.assertEqual(report['counts']['parsed_annotations'], 1)
        with self.assertRaises(FileExistsError):
            self.scan()

    def test_unrecognized_layout_and_empty_input(self):
        path = self.root / 'misplaced.json'
        path.write_text('{}', encoding='utf-8')
        report = self.scan()
        self.assertIn('UNRECOGNIZED_LAYOUT', self.codes(report))
        self.assertIn('NO_ANNOTATIONS', self.codes(report))

    def test_output_inside_dataset_is_rejected(self):
        self.annotation()
        with self.assertRaises(ValueError):
            self.api()[0](self.root, self.root / 'report')
        self.assertFalse((self.root / 'report').exists())

    def test_unsafe_uri_and_mismatched_subject(self):
        doc = document()
        doc['openlabel']['streams']['face_camera']['uri'] = '../../outside.mp4'
        doc['openlabel']['objects']['0']['name'] = '2'
        self.annotation(doc)
        report = self.scan()
        self.assertIn('UNSAFE_URI', self.codes(report))
        self.assertIn('SUBJECT_MISMATCH', self.codes(report))

    def test_wrong_annotation_kind_is_visible(self):
        self.annotation(kind='drowsiness', subset='gaze')
        self.assertIn('ANNOTATION_KIND_MISMATCH', self.codes(self.scan()))

    def test_conflicting_stream_declarations_are_visible(self):
        self.annotation()
        doc = document()
        doc['openlabel']['streams']['face_camera']['stream_properties']['sync']['frame_shift'] = 5
        self.annotation(doc, kind='hands')
        self.assertIn('STREAM_DECLARATION_MISMATCH', self.codes(self.scan()))

    def test_camera_view_must_match_uri(self):
        doc = document()
        doc['openlabel']['streams']['face_camera']['uri'] = f'gA/1/s6/{PREFIX}_rgb_hands.mp4'
        self.annotation(doc)
        self.assertIn('STREAM_VIEW_MISMATCH', self.codes(self.scan()))

    def test_declared_missing_hands_is_not_fatal(self):
        self.annotation(subset='drowsiness', kind='drowsiness')
        report = self.scan()
        self.assertEqual(report['status'], 'ok')

    def test_missing_hands_without_stream_declaration_is_visible(self):
        self.annotation()
        report = self.scan()
        self.assertTrue(any(i['code'] == 'MISSING_VIDEO' and 'hands.mp4' in i['message']
                            for i in report['issues']))

    def test_absolute_uri_is_not_followed(self):
        for uri in ('/etc/passwd', 'C:/private/video.mp4', 'https://example.com/video.mp4'):
            with self.subTest(uri=uri):
                doc = document()
                doc['openlabel']['streams']['face_camera']['uri'] = uri
                self.annotation(doc)
                output = self.base / ('report-' + str(len(list(self.base.glob('report-*')))))
                report = self.api()[0](self.root, output)
                self.assertIn('UNSAFE_URI', self.codes(report))

    def test_symlink_not_followed(self):
        target = self.base / 'outside.json'
        target.write_text(json.dumps(document()), encoding='utf-8')
        link = self.root / 'link.json'
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest('symlink privilege unavailable')
        report = self.scan()
        self.assertIn('SYMLINK_SKIPPED', self.codes(report))
        self.assertEqual(report['counts']['parsed_annotations'], 0)

    def test_cli_without_site_packages_and_error_exit(self):
        self.api()
        self.annotation()
        command = [sys.executable, '-S', '-B', '-m', 'training.dmd.audit',
                   '--dataset-root', str(self.root), '--output-dir', str(self.base / 'cli')]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'ok')
        self.annotation(kind='hands').write_text('', encoding='utf-8')
        command[-1] = str(self.base / 'cli-errors')
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertTrue((self.base / 'cli-errors/audit.json').is_file())


if __name__ == '__main__':
    unittest.main()
