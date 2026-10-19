"""Mechanical fatigue review exports; synthetic checks make no quality claim."""
import builtins
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_face_diagnostic_replay import replay_fixture


def rows_fixture():
    return [dict(record_id='r'+subject, subject_id=subject, start_slot=i*25,
                 end_slot=i*25+99, label_known=True, target=target,
                 f_eligible=False, available=True, logit=score)
            for subject, targets in [('7', [(1, -2.)]*35+[(0, -2.)]*10),
                                     ('14', [(1, 1.)]*7+[(0, 1.)]*6+[(0, -1.)]*10)]
            for i, (target, score) in enumerate(targets)]


class FatigueAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # runtime_fixture temporarily replaces sys.modules. Initialize Torch's
        # optimizer registration before that boundary so it is not registered twice.
        import torch._dynamo

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('training.visual.fatigue_audit'), 'audit missing')
        from training.visual import fatigue_audit
        return fatigue_audit

    def test_subject_false_negatives_tie_auc_and_bounded_selection(self):
        audit = self.api()
        rows = rows_fixture()
        result = audit.subject_summaries(rows)
        self.assertEqual(result['7']['metrics']['fn_total'], 35)
        self.assertEqual(result['14']['metrics']['tp'], 7)
        self.assertEqual(result['7']['ranking_auc_available'], .5)
        self.assertEqual(result['7']['positive_logits']['median'], -2.)
        selected = audit.select_windows(rows)
        self.assertEqual(sum(r['target'] == 1 for r in selected), 42)
        for subject in ('7', '14'):
            self.assertEqual(sum(r['target'] == 0 and r['subject_id'] == subject for r in selected), 8)
        self.assertEqual(selected, audit.select_windows(list(reversed(rows))))

    def test_unavailable_and_unknown_scores_are_not_ranked(self):
        audit = self.api()
        rows = rows_fixture()[:2]
        rows[0].update(available=False, logit=None)
        rows[1].update(label_known=False)
        result = audit.subject_summaries(rows)['7']
        self.assertEqual(result['metrics']['fn_total'], 1)
        self.assertEqual(result['positive_logits']['count'], 0)
        self.assertIsNone(result['ranking_auc_available'])
        self.assertEqual(len(audit.select_windows(rows)), 1)

    def test_original_window_end_labels_reject_self_consistent_prepared_corruption(self):
        audit = self.api()
        from types import SimpleNamespace
        from training.visual.extract import _prepare
        from training.visual.fatigue import summarize
        with tempfile.TemporaryDirectory(dir='runs') as tmp, replay_fixture(tmp) as (kwargs, _, _):
            record = json.loads(Path(kwargs['manifest']).read_text())['records'][0]
            item = _prepare(record, Path(kwargs['data_root']).resolve(), Path(kwargs['dataset_root']).resolve())
            row = dict(record_id='job1',subject_id='14',start_slot=0,end_slot=99,
                       label_known=True,target=0,f_eligible=False,available=True,logit=-1.)
            self.assertEqual(audit.verify_window_end_labels(item,[row]),dict(windows=1,known=1,unknown=0))
            corrupt = dict(row,target=1)
            # Both publication and preparation agree on the wrong target: old check passes.
            audit._validate_comparison({'prepared_sha256':'same'},summarize([corrupt]),
                SimpleNamespace(task='fatigue',provenance={'prepared_sha256':'same'}),[corrupt])
            with self.assertRaisesRegex(ValueError,'window-end fatigue label'):
                audit.verify_window_end_labels(item,[corrupt])
            with self.assertRaisesRegex(ValueError,'window-end fatigue label'):
                audit.verify_window_end_labels(item,[dict(row,label_known=False)])
            for changes in ({'end_slot':98},{'start_slot':25,'end_slot':124},{'start_slot':-25,'end_slot':74}):
                with self.subTest(changes=changes),self.assertRaisesRegex(ValueError,'window bounds'):
                    audit.verify_window_end_labels(item,[dict(row,**changes)])

    def test_replay_without_landmarks_checks_full_pts_and_causal_samples(self):
        audit = self.api()
        from inference.perception.video_source import VideoFrame
        from training.visual.extract import _prepare
        from training.visual.spatial import crop_policy
        from tests.test_visual_extract import VisualExtractTests
        with tempfile.TemporaryDirectory(dir='runs') as tmp, replay_fixture(tmp) as (kwargs, _, _):
            root = Path(tmp)
            visual = VisualExtractTests().run_extract(kwargs, root/'visual')
            record = json.loads(Path(kwargs['manifest']).read_text())['records'][0]
            item = _prepare(record, Path(kwargs['data_root']).resolve(), Path(kwargs['dataset_root']).resolve())
            entry = visual['records'][0]
            arrays = {key: np.load(root/'visual'/value) for key, value in entry['arrays'].items()}
            original_import = builtins.__import__
            def no_landmarks(name, *args, **options):
                if name.startswith('mediapipe') or name.endswith('mediapipe_adapter'):
                    raise AssertionError('audit must not import landmarks')
                return original_import(name, *args, **options)
            with patch('builtins.__import__', no_landmarks):
                frames = audit.record_frames(item, arrays, [12,49,99], crop_policy((.05,0.,.65,.65)))
            self.assertEqual(sorted(frames), [12,49,99])
            self.assertEqual(frames[49]['source_pts_ms'], 1960)
            self.assertTrue(frames[49]['jpeg'].startswith(b'\xff\xd8'))
            def wrong_decode(*args):
                for i in range(100):
                    yield VideoFrame(i*40+(1 if i == 1 else 0), np.zeros((500,1000,3), np.uint8))
            with patch('inference.perception.video_source.decode_video', wrong_decode):
                with self.assertRaisesRegex(ValueError, 'PTS'):
                    audit.record_frames(item, arrays, [99], crop_policy((.05,0.,.65,.65)))
            arrays['source_pts_ms'][0] -= 1
            with self.assertRaisesRegex(ValueError, 'PTS'):
                audit.record_frames(item, arrays, [99], crop_policy((.05,0.,.65,.65)))

    def test_complete_export_matches_evaluation_and_rejects_changed_report(self):
        audit = self.api()
        import torch
        from training.common.data import file_sha256
        from training.visual.data import PreparedData, METADATA
        from training.visual.fatigue import summarize
        from training.visual.report import evaluate
        from training.visual.suite import DEFAULTS
        from training.visual.train import train_probe
        from tests.test_visual_extract import VisualExtractTests
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir='runs') as tmp, replay_fixture(tmp) as (kwargs, _, _):
            root = Path(tmp).resolve()
            # Set the synthetic immutable source label before extracting its cache.
            attempt = root/'cache/records/job1/attempt1'
            targets = np.load(attempt/'targets.npy')
            targets[99,0] = 1.
            np.save(attempt/'targets.npy',targets)
            complete = json.loads((attempt/'complete.json').read_text())
            complete['array_sha256']['targets'] = file_sha256(attempt/'targets.npy')
            (attempt/'complete.json').write_text(json.dumps(complete))
            visual = VisualExtractTests().run_extract(kwargs, root/'visual')
            entry = visual['records'][0]
            prep = root/'prepared'
            prep.mkdir()
            rows = [dict(**{k: entry[k] for k in METADATA}, start_slot=0, end_slot=99,
                         visual_start=0, label_known=True, target=1, f_eligible=False,
                         baseline_available=False, baseline_logit=None)]
            (prep/'rows.json').write_text(json.dumps(rows))
            np.save(prep/'context.npy', np.full((1,2),6.,np.float32))
            prepared_report = dict(schema_version='driveguard-visual-prepared-v2',status='complete',task='fatigue',
                manifest=str(Path(kwargs['manifest']).resolve()),data_root=str(Path(kwargs['data_root']).resolve()),
                visual_cache=str(root/'visual'), visual_cache_sha256=file_sha256(root/'visual/manifest.json'),context_dim=2,
                normalization=dict(visual=dict(mean=[0.]*512,std=[1.]*512),context=dict(mean=[0.]*2,std=[1.]*2)),
                splits=dict(val=dict(rows='rows.json',rows_sha256=file_sha256(prep/'rows.json'),
                                     context='context.npy',context_sha256=file_sha256(prep/'context.npy'))))
            (prep/'prepared.json').write_text(json.dumps(prepared_report))
            prepared = PreparedData(prep)
            try:
                trained = train_probe(prepared.dataset('val'),prepared.dataset('val'),'visual',
                    dict(DEFAULTS,epochs=1,hidden_dim=4,batch_size=1),root/'probe',prepared.provenance)
                checkpoint = trained['best_checkpoint']
                evaluated, _ = evaluate(prepared.dataset('val'),checkpoint)
                comparison = summarize(evaluated,variant='B_rgb-seed43-best',checkpoint=checkpoint,
                                       checkpoint_sha256=file_sha256(checkpoint))
            finally:
                prepared.close()
            report = dict(schema_version='driveguard-fatigue-visual-comparison-v1',status='complete',task='fatigue',
                          split='val',prepared_dir=str(prep),prepared_sha256=file_sha256(prep/'prepared.json'),
                          comparisons=[comparison])
            report_path = root/'comparison.json'
            report_path.write_text(json.dumps(report))
            before = {p:file_sha256(p) for p in root.rglob('*') if p.is_file()}
            result = audit.run(fatigue_report=report_path,dataset_root=kwargs['dataset_root'],output_dir=root/'audit')
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['known_windows'],1)
            self.assertEqual(result['positive_contact_sheets'],1)
            self.assertTrue(result['mechanical_checks']['original_window_end_fatigue_labels_match'])
            self.assertEqual(result['original_label_check_scope']['windows'],1)
            self.assertEqual(result['by_subject']['14']['normalization_clipping']['context']['fraction'],1.)
            self.assertEqual(result['manual_review'],dict(status='pending',label_semantics_verified=False,head_coverage_verified=False))
            self.assertIn('WINDOW END',(root/'audit/index.html').read_text(encoding='utf-8'))
            with (root/'audit/windows.csv').open(newline='') as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(csv_rows[0]['end_ms'],'3960')
            self.assertTrue((root/'audit'/csv_rows[0]['contact_sheet']).is_file())
            self.assertEqual(before,{p:file_sha256(p) for p in before})
            report['comparisons'][0]['overall']['tp'] += 1
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'overall'):
                audit.run(fatigue_report=report_path,dataset_root=kwargs['dataset_root'],output_dir=root/'bad')
            self.assertFalse((root/'bad/summary.json').exists())


if __name__ == '__main__':
    unittest.main()
