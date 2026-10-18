import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from inference.live.bundle import verified_path, select_reports, load_bundle, code_identity, SCHEMA, ROLES


class BundleTests(unittest.TestCase):
    def test_bundle_binds_code_assets_and_event_policy(self):
        from training.visual.eye_tracking import POLICY
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'weights.pt').write_bytes(b'test artifact')
            asset=dict(path='weights.pt',sha256=hashlib.sha256(b'test artifact').hexdigest())
            report=dict(schema_version=SCHEMA,status='complete',models={r:asset for r in ROLES},
                files={'weights.pt':asset},code_identity=code_identity(),eye_tracking=POLICY,
                event_policy=dict(closed_eye_min_ms=1000,threshold_logit=0.,missing='reset-eye-only'),
                head_crop=dict(version='driver-relative-rect-v1',xyxy=[.05,0,.65,.65]))
            path=root/'bundle.json'
            def write():path.write_text(json.dumps(report),encoding='utf-8')
            write();self.assertEqual(set(load_bundle(path)[1]),set(ROLES))
            report['code_identity']={};write()
            with self.assertRaisesRegex(ValueError,'code identity'):load_bundle(path)
            report['code_identity']=code_identity();report['event_policy']['closed_eye_min_ms']=500;write()
            with self.assertRaisesRegex(ValueError,'policy'):load_bundle(path)

    def test_asset_hash_and_containment(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'a.pt').write_bytes(b'weights')
            entry=dict(path='a.pt',sha256=hashlib.sha256(b'weights').hexdigest())
            self.assertEqual(verified_path(root,entry),root/'a.pt')
            (root/'a.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):verified_path(root,entry)
            with self.assertRaises(ValueError):verified_path(root,dict(entry,path='../a.pt'))
            with self.assertRaises(ValueError):verified_path(root,dict(entry,path=str(root/'a.pt')))

    def test_fixed_report_selection_no_best_search(self):
        full=dict(schema_version='driveguard-visual-all-tasks-v1',status='complete',split='val',
                  composites=[dict(name='C_fusion_seed43_validation_fpr03_coverage25_outputs',
                                   body_threshold_logit=-.6933670043945312)])
        eye=dict(schema_version='driveguard-eye-recovery-comparison-v1',status='complete',split='val',
                 reference_reproduced=True,frozen_weights_unchanged=True,training_runs=0,
                 comparisons=[dict(variant='dense_track-combined')])
        selected=select_reports(full,eye)
        self.assertEqual(selected[0]['body_threshold_logit'],-.6933670043945312)
        full['composites']*=2
        with self.assertRaises(ValueError):select_reports(full,eye)


if __name__=='__main__':unittest.main()
