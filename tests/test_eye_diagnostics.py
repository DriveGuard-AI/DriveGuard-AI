"""Ranking and causal event diagnostics, without retraining or label leakage."""
import unittest
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


class ScoreTests(unittest.TestCase):
    def test_perfect_ranking_below_zero_and_unknown_nan(self):
        from training.visual.eye_scores import score_summary
        result = score_summary([-4., -3., -2., -1., np.nan], [0, 0, 1, 1, 0], [1, 1, 1, 1, 0])
        self.assertEqual(result['fixed_zero']['tp'], 0)
        self.assertEqual(result['roc_auc'], 1.)
        self.assertEqual(result['average_precision'], 1.)
        self.assertEqual(result['positive_scores']['median'], -1.5)
        self.assertEqual(result['fpr03_threshold'], -2.)
        self.assertEqual(result['balanced_accuracy_threshold'], -2.)
        self.assertEqual(result['unknown_or_invalid'], 1)

    def test_ties_cannot_be_split_and_undefined_metrics_are_null(self):
        from training.visual.eye_scores import score_summary
        result = score_summary([1., 1., 1., 1.], [0, 0, 1, 1], [1]*4)
        self.assertEqual(result['roc_auc'], .5)
        self.assertEqual(result['average_precision'], .5)
        self.assertIsNone(result['fpr03_threshold'])
        self.assertEqual(len(result['curve']), 2)
        one = score_summary([1.], [1], [1])
        self.assertIsNone(one['roc_auc'])
        self.assertIsNone(one['fpr03_threshold'])
        self.assertIsNone(one['balanced_accuracy_threshold'])
        empty = score_summary([np.nan], [0], [0])
        self.assertIsNone(empty['average_precision'])
        with self.assertRaises(ValueError): score_summary([np.nan], [1], [1])
        with self.assertRaises(ValueError): score_summary([0.], [2], [1])


def event_fixture():
    from training.visual.head_frames import selected_slots
    slots = selected_slots(100)
    arrays = dict(slot_indices=slots, source_pts_ms=slots*40, valid=np.ones(32, bool))
    cache = SimpleNamespace(arrays=lambda rid: arrays)
    rows = [dict(record_id='a', start_slot=0, end_slot=99, target=1, label_known=True)]
    predictions = {'a': np.tile([-1., -10.], (32, 1))}
    resets = {'a': np.r_[True, np.zeros(31, bool)]}
    return cache, rows, predictions, resets, arrays


class ReplayTests(unittest.TestCase):
    def test_eye_threshold_recomputes_duration_not_final_score(self):
        from training.visual.action_eval import replay_events
        cache, rows, pred, resets, arrays = event_fixture()
        self.assertLess(replay_events(cache, pred, rows, resets, 500)[0]['logit'], 0)
        self.assertGreaterEqual(replay_events(cache, pred, rows, resets, 500, eye_threshold=-2.)[0]['logit'], 0)
        resets['a'][-2] = True
        self.assertLess(replay_events(cache, pred, rows, resets, 500, eye_threshold=-2.)[0]['logit'], 0)
        arrays['valid'][-1] = False
        self.assertFalse(replay_events(cache, pred, rows, resets, 500)[0]['available'])

    def test_branch_attribution_and_labels_do_not_affect_inference(self):
        from training.visual.action_eval import replay_events
        cache, rows, pred, resets, arrays = event_fixture()
        pred['a'][:, 1] = 2.
        self.assertEqual(replay_events(cache, pred, rows, resets, 500, branches='yawn')[0]['logit'], 2.)
        self.assertLess(replay_events(cache, pred, rows, resets, 500, branches='eye')[0]['logit'], 0)
        before = replay_events(cache, pred, rows, resets, 500)
        after = replay_events(cache, pred, [dict(rows[0], target=0, label_known=False)], resets, 500)
        self.assertEqual(before[0]['logit'], after[0]['logit'])
        self.assertEqual(replay_events(cache, pred, rows, resets, 500, eye_threshold=None)[0]['logit'], 2.)
        with self.assertRaises(ValueError): replay_events(cache, pred, rows, resets, 500, eye_threshold=float('nan'))


class PipelineTests(unittest.TestCase):
    def test_replay_export_and_training_thresholds_are_independent_of_validation_targets(self):
        from training.visual.eye_diagnostics import run
        from training.visual.action_eval import evaluate_events, action_metrics
        from training.visual.action_labels import ActionLabels
        from training.visual.fatigue_spatial_suite import _summary
        from training.visual.head_frames import selected_slots
        from training.common.config import read_json
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(()))
            def frame_actions(self, images):
                eye = images[:, 0, 0, 0].float()/10-4
                return torch.stack((eye, torch.full_like(eye, -5.)), -1)
        model = Model()
        arrays, metadata, records, rows = {}, {}, {}, {}
        slots = selected_slots(150); n = len(slots)
        for split, rid, subject in [('train', 't', '1'), ('val', 'v', '7')]:
            values = np.r_[np.full(n//2, 10), np.full(n-n//2, 20)].astype(np.uint8)
            arrays[rid] = dict(slot_indices=slots, source_pts_ms=slots*40, valid=np.ones(n, bool),
                               images=values[:, None, None, None])
            metadata[rid] = dict(record_id=rid, subject_id=subject, split=split)
            targets = np.zeros((n, 2)); targets[n//2:, 0] = 1
            records[rid] = dict(targets=targets, masks=np.ones((n, 2), bool),
                                subtypes=np.full(150, 'closed_eye'), reset=np.r_[True, np.zeros(n-1, bool)])
            rows[split] = [dict(metadata[rid], start_slot=i*25, end_slot=i*25+99, label_known=i!=2,
                                target=i%2, f_eligible=False) for i in range(3)]
        cache = SimpleNamespace(records=metadata, arrays=lambda rid: arrays[rid])
        labels = ActionLabels.__new__(ActionLabels)
        labels.records = records; labels.closed_eye_min_ms = 500
        resets = {rid: r['reset'] for rid, r in records.items()}
        baseline, predictions = evaluate_events(cache, model, rows['val'], resets, 500)
        variant = 'B_action_supervision-seed42-best-event'
        selected = _summary(baseline, variant=variant, arm='B_action_supervision_event', seed=42,
                            checkpoint_kind='best', checkpoint='fixture.pt', checkpoint_sha256='fixture', epoch=4)
        report = dict(action_metrics={variant.removesuffix('-event'): action_metrics(predictions, labels, cache)})
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root = Path(tmp).resolve(); source = root/'source'; source.mkdir()
            report_path = source/'comparison.json'; report_path.write_text('{}')
            @contextmanager
            def inputs(*args):
                yield dict(base=SimpleNamespace(rows=rows), cache=cache, labels=labels, model=model,
                           selected=selected, report=report, provenance={'fixture':True}, sources=[source])
            with patch('training.visual.eye_diagnostics._inputs', inputs), redirect_stdout(io.StringIO()):
                first = run(action_report=report_path, dataset_root=source, output_dir=root/'out')
                self.assertEqual(first, read_json(root/'out/summary.json'))
                self.assertTrue(first['reference_reproduced'])
                self.assertEqual(first['training_runs'], 0)
                self.assertEqual(len(first['comparisons']), 5)
                self.assertTrue((root/'out/frames.csv').exists())
                self.assertTrue((root/'out/score-curves.csv').exists())
                self.assertEqual(first['scores']['val']['overall']['eye']['roc_auc'], 1.)
                self.assertEqual(first['train_selected_thresholds']['fpr03'], -2.)
                for item in first['comparisons']:
                    self.assertEqual(item['window_label_hash'], selected['window_label_hash'])
                with self.assertRaises(FileExistsError):
                    run(action_report=report_path, dataset_root=source, output_dir=root/'out')
                with self.assertRaisesRegex(ValueError, 'overlaps source'):
                    run(action_report=report_path, dataset_root=source, output_dir=source/'nested')
                self.assertFalse((source/'nested').exists())
                saved = deepcopy(report)
                records['v']['targets'][:, 0] = 0
                report['action_metrics'][variant.removesuffix('-event')] = action_metrics(predictions, labels, cache)
                second = run(action_report=report_path, dataset_root=source, output_dir=root/'other')
                self.assertEqual(first['train_selected_thresholds'], second['train_selected_thresholds'])
                self.assertEqual([r['prediction_rows_hash'] for r in first['comparisons']],
                                 [r['prediction_rows_hash'] for r in second['comparisons']])
                report.update(saved)
                with self.assertRaisesRegex(ValueError, 'action metrics'):
                    run(action_report=report_path, dataset_root=source, output_dir=root/'wrong-actions')
                self.assertFalse((root/'wrong-actions/summary.json').exists())
                selected['overall']['tp'] += 1
                with self.assertRaisesRegex(ValueError, 'published event'):
                    run(action_report=report_path, dataset_root=source, output_dir=root/'wrong-events')
                self.assertFalse((root/'wrong-events/summary.json').exists())

    def test_selection_rejects_wrong_policy_provenance_and_direct_readout(self):
        from training.visual.eye_diagnostics import validate_selection
        provenance = {'fixture':True}
        selected = dict(arm='B_action_supervision_event', checkpoint_kind='best', epoch=4, seed=42,
                        selection_metric='action_validation_bce', variant='B_action_supervision-seed42-best-event')
        report = dict(schema_version='driveguard-fatigue-actions-comparison-v1', status='complete',
                      split='val', task='fatigue', comparisons=[selected],
                      plan=dict(provenance=provenance, event_policy=dict(action_logit_threshold=0.,
                                closed_eye_min_ms=500, reset_on_source_gap_ms=80, threshold_search=False)))
        payload = dict(kind='weights', epoch=4, settings={'seed':42}, auxiliary=True,
                       checkpoint_role='best', selection_metric='action_validation_bce', provenance=provenance)
        validate_selection(report, selected, payload, 500)
        for path in ('provenance', 'duration', 'direct', 'checkpoint_role'):
            r, s, p = deepcopy(report), deepcopy(selected), deepcopy(payload)
            if path=='provenance': p['provenance'] = {'wrong':1}
            if path=='duration': r['plan']['event_policy']['closed_eye_min_ms'] = 100
            if path=='direct': s['arm'] = 'B_action_supervision_direct'
            if path=='checkpoint_role': p['checkpoint_role'] = 'fatigue-best'
            with self.assertRaises(ValueError): validate_selection(r, s, p, 500)


if __name__ == '__main__': unittest.main()
