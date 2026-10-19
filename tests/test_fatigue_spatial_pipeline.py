"""Pipeline ordering, audit failure barrier, and immutable stage resume."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.common.checkpoint import atomic_json
from training.common.data import file_sha256


class SpatialPipelineTests(unittest.TestCase):
    def test_audit_precedes_extraction_and_is_verified_on_resume(self):
        from training.visual.fatigue_spatial_pipeline import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root/'Dataset'; dataset.mkdir()
            weights = root/'weights.pt'; weights.write_bytes(b'local-weights')
            report = root/'source.json'; report.write_text('{}')
            manifest = root/'manifest.json'; manifest.write_text('{}')
            data_root = root/'features'; data_root.mkdir()
            prepared = root/'prepared'; prepared.mkdir()
            cache = root/'cache'; cache.mkdir()
            reference = dict(manifest=str(manifest), data_root=str(data_root), prepared_dir=str(prepared),
                             prepared_sha256='prepared-sha', visual_cache=str(cache),
                             weights_sha256=file_sha256(weights))
            calls = []
            def audit(**kwargs):
                calls.append('audit')
                out = Path(kwargs['output_dir']); out.mkdir()
                atomic_json(out/'summary.json',dict(status='complete'))
                (out/'index.html').write_text('evidence')
                return dict(status='complete')
            def extract(**kwargs):
                calls.append('extract')
                out = Path(kwargs['output_dir']); out.mkdir(exist_ok=True)
                atomic_json(out/'manifest.json',dict(status='complete'))
            def compare(**kwargs):
                calls.append('compare')
                out = Path(kwargs['output_dir']); out.mkdir(exist_ok=True)
                atomic_json(out/'comparison.json',dict(status='complete'))
                return dict(status='complete')
            options = dict(fatigue_report=report, dataset_root=dataset, weights=weights, output_dir=root/'output')
            with patch('training.visual.fatigue_spatial_pipeline._reference', return_value=reference), \
                 patch('training.visual.fatigue_spatial_pipeline._audit',side_effect=audit), \
                 patch('training.visual.fatigue_spatial_pipeline._extract',side_effect=extract), \
                 patch('training.visual.fatigue_spatial_pipeline._compare',side_effect=compare):
                first = run(**options, audit_only=True)
                self.assertEqual(calls,['audit'])
                self.assertEqual(first['status'],'audit_complete')
                run(**options,continue_run=True)
                self.assertEqual(calls,['audit','extract','compare'])
                evidence = next((root/'output').glob('audit-*/index.html'))
                evidence.write_text('changed')
                with self.assertRaisesRegex(ValueError,'audit|Audit'):
                    run(**options,continue_run=True)
                self.assertEqual(calls,['audit','extract','compare'])

    def test_audit_failure_stops_before_encoder_and_training(self):
        from training.visual.fatigue_spatial_pipeline import _run_audit
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('training.visual.fatigue_spatial_pipeline._audit',side_effect=ValueError('PTS mismatch')):
                with self.assertRaisesRegex(ValueError,'PTS mismatch'):
                    _run_audit(root,{'id':'fixed'},dict())
            self.assertFalse((root/'audit-complete.json').exists())


if __name__ == '__main__':unittest.main()
