"""Coverage25 outputs composed with the unchanged gap80-context C branch."""
from copy import deepcopy
import csv
import json
import unittest
from unittest.mock import patch

import torch

from tests import test_visual_all_tasks as fixtures
from tests.test_visual_all_tasks import joint_row, probe_row
from training.common.checkpoint import load_checkpoint, save_checkpoint
from training.common.config import canonical_hash
from training.common.data import file_sha256


class OutputCompositionTests(unittest.TestCase):
    def test_alternate_outputs_preserve_c_branch_fallback_and_gap80_groups(self):
        from training.visual import all_tasks
        self.assertTrue(hasattr(all_tasks, 'replace_output_base'), 'Output-base composition is missing')
        original = [joint_row(), joint_row('face'), joint_row(start=25)]
        c_rows, _ = all_tasks.compose(original, [probe_row(original[0]), probe_row(original[2], False)], 1.)
        coverage = deepcopy(original)
        for row in coverage:
            row.update(behavior_logits=[4., 3., 2., 1.], task_available=[False]*4,
                       thresholds=[.2]*4, gaze_logits=list(range(9)), gaze_available=False,
                       h_eligible=True)
        before = deepcopy((c_rows, coverage))
        rows, audit = all_tasks.replace_output_base(c_rows, list(reversed(coverage)))
        self.assertEqual((c_rows, coverage), before)
        self.assertTrue(audit['passed'])
        for index, row in enumerate(rows):
            self.assertFalse(row['h_eligible'])
            self.assertTrue(row['output_base_h_eligible'])
            self.assertEqual(row['gaze_logits'], coverage[index]['gaze_logits'])
            self.assertFalse(row['gaze_available'])
            columns = range(3) if row['view'] == 'body' else range(4)
            for column in columns:
                for field in ('behavior_logits', 'task_available', 'thresholds'):
                    self.assertEqual(row[field][column], coverage[index][field][column])
            if row['view'] == 'body':
                for field in ('behavior_logits', 'task_available', 'thresholds'):
                    self.assertEqual(row[field][3], c_rows[index][field][3])
                for field in ('body_probe_available', 'body_probe_logit', 'body_probe_threshold', 'body_distraction_source'):
                    self.assertEqual(row[field], c_rows[index][field])
        self.assertEqual(rows[2]['body_distraction_source'], 'baseline_rgb_unavailable')
        self.assertEqual(all_tasks.paired_errors(original, rows), all_tasks.paired_errors(original, c_rows))

    def test_reject_full_identity_labels_metadata_and_nonfinite_outputs(self):
        from training.visual import all_tasks
        self.assertTrue(hasattr(all_tasks, 'replace_output_base'), 'Output-base composition is missing')
        row = joint_row()
        for changes in (dict(targets=[0, 0, 1, 1]), dict(label_known=[False]*4),
                        dict(gaze_target=1), dict(gaze_label_known=False), dict(subject_id='other'),
                        dict(behavior_logits=[float('nan')]*4), dict(gaze_logits=[float('inf')]*9)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                all_tasks.replace_output_base([row], [dict(row, **changes)])
        for invalid in ([], [row, row], [dict(row, start_slot=25)]):
            with self.assertRaises(ValueError):
                all_tasks.replace_output_base([row], invalid)


class OutputBaseIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.IntegrationTests.setUpClass.__func__(cls)
        from training.gaze.train import run_training
        baseline = load_checkpoint(cls.root/'baseline/best.pt')
        config = deepcopy(baseline['config'])
        config['model']['head_policy']['max_gap_steps'] = 0
        config['runtime']['output_dir'] = str(cls.root/'coverage25')
        result = run_training(config)
        cls.coverage_path = cls.root/'coverage25/best.pt'
        payload = load_checkpoint(result['best_checkpoint'])
        # Distinct valid preprocessing makes accidental reuse of gap80 normalization observable.
        payload['normalizer']['mean'] = [value + .3 for value in payload['normalizer']['mean']]
        save_checkpoint(cls.coverage_path, payload)
        cls.coverage_payload = payload

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_inference_own_normalizer_eight_exports_and_two_probe_passes(self):
        from training.visual import all_tasks
        from training.common.model_factory import load_model
        from training.gaze.data import FeatureCorpus
        output = self.root/'coverage-evaluation'
        before = {p: file_sha256(p) for p in self.root.rglob('*') if p.is_file()}
        original_evaluate = all_tasks.evaluate
        with patch.object(all_tasks, 'evaluate', wraps=original_evaluate) as calls:
            report = all_tasks.run(prepared_dir=self.root/'prepared', stability_report=self.source_path,
                output_dir=output, output_base_checkpoint=self.coverage_path, batch_size=3)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual(len(report['composites']), 6)
        self.assertEqual(report['output_base']['checkpoint_sha256'], file_sha256(self.coverage_path))
        self.assertEqual(report['output_base']['checkpoint_epoch'], self.coverage_payload['epoch'])
        self.assertEqual(report['output_base']['normalizer_sha256'], canonical_hash(self.coverage_payload['normalizer']))
        self.assertEqual(before, {p: file_sha256(p) for p in before})
        rows = [json.loads(line) for line in (output/'predictions.jsonl').read_text().splitlines()]
        self.assertEqual(len({r['candidate'] for r in rows}), 8)
        model, payload = load_model(self.coverage_path)
        corpus = FeatureCorpus(payload['config'], payload['config']['runtime']['manifest'], payload['config']['runtime']['data_root'])
        try:
            expected = all_tasks._baseline_rows(model, corpus.dataset('val', payload['normalizer']), 3)
        finally:
            corpus.close()
        actual = [{k: v for k, v in r.items() if k != 'candidate'} for r in rows if r['candidate'] == report['output_base']['name']]
        self.assertEqual(actual, expected)
        with (output/'metrics.csv').open(newline='', encoding='utf-8') as handle:
            metrics = list(csv.DictReader(handle))
        self.assertEqual(len({r['candidate'] for r in metrics if r['task'] == 'gaze'}), 8)
        self.assertIn('evaluation_cost', report)
        for value in report['evaluation_cost']['model_inventory'].values():
            self.assertIn('parameter_count', value)
            self.assertGreater(value['parameter_count'], 0)
            self.assertGreater(value['tensor_bytes'], 0)
        alternates = [r for r in report['composites'] if r.get('output_base') == 'A_coverage25']
        self.assertEqual(len(alternates), 3)
        for item in alternates:
            self.assertTrue(item['equality_audit']['passed'])
            self.assertEqual(len(item['model_dependencies']), 3)
        with self.assertRaises(FileExistsError):
            all_tasks.run(prepared_dir=self.root/'prepared', stability_report=self.source_path,
                output_dir=output, output_base_checkpoint=self.coverage_path)

    def test_reject_wrong_policy_fingerprint_and_overlap(self):
        from training.visual.all_tasks import run
        for reason in ('policy', 'fingerprint'):
            payload = deepcopy(self.coverage_payload)
            if reason == 'policy':
                payload['config']['model']['head_policy']['max_gap_steps'] = 2
            else:
                payload['data_summary']['fingerprint'] = '0'*64
            directory = self.root/f'wrong-{reason}'
            directory.mkdir()
            path = directory/'best.pt'
            save_checkpoint(path, payload)
            with self.assertRaisesRegex(ValueError, 'coverage25|fingerprint'):
                run(prepared_dir=self.root/'prepared', stability_report=self.source_path,
                    output_dir=self.root/f'rejected-{reason}', output_base_checkpoint=path)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            run(prepared_dir=self.root/'prepared', stability_report=self.source_path,
                output_dir=self.coverage_path.parent/'overlap', output_base_checkpoint=self.coverage_path)

    def test_input_changes_during_evaluation_reject_without_output(self):
        from training.visual import all_tasks
        prepared = json.loads((self.root/'prepared/prepared.json').read_text())
        paths = [self.coverage_path, self.root/'prepared'/prepared['splits']['val']['rows']]
        for index, path in enumerate(paths):
            original = path.read_bytes()
            output = self.root/f'changed-input-{index}'
            evaluate = all_tasks.evaluate
            def mutate_after_read(*args, **kwargs):
                result = evaluate(*args, **kwargs)
                if args[1] == self.root/'probe43/best.pt':
                    with path.open('ab') as handle:
                        handle.write(b' ')
                return result
            try:
                with patch.object(all_tasks, 'evaluate', side_effect=mutate_after_read):
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        all_tasks.run(prepared_dir=self.root/'prepared', stability_report=self.source_path,
                            output_dir=output, output_base_checkpoint=self.coverage_path)
                self.assertFalse(output.exists())
            finally:
                path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
