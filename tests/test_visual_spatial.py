"""Geometry, cache identity and spatial ablations; no model quality claims."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from training.common.checkpoint import atomic_json
from training.common.config import canonical_hash, read_json
from training.common.data import file_sha256
from training.visual.spatial import crop_policy, crop_roi


def head_cache_fixture(root, manifest, records):
    from tests.test_visual_data import visual_fixture
    cache = visual_fixture(root, manifest, records, {'val':1000})
    run = read_json(cache/'input.json')
    run.update(schema_version='driveguard-visual-extract-v2', crop=crop_policy())
    atomic_json(cache/'input.json', run)
    report = read_json(cache/'manifest.json')
    report.update(schema_version='driveguard-visual-cache-v2', crop=run['crop'])
    for row in report['records']:
        identity = {k:row[k] for k in ('record_id','recording_id','subject_id','view','subset','split','slots','source')}
        row['fingerprint'] = canonical_hash(dict(run=run, record=identity))
        atomic_json(cache/row['directory']/'record.json', row)
    atomic_json(cache/'manifest.json', report)
    return cache


class SpatialTests(unittest.TestCase):
    def test_crop_is_relative_to_original_driver_roi(self):
        self.assertEqual(crop_roi([100,50,900,650], crop_policy([.25,0,.75,.5])), [300,50,700,350])
        for box in ([0,0,0,1],[-.1,0,.5,1],[0,0,2,1],[0,0,float('nan'),1],[False,0,1,1]):
            with self.assertRaises(ValueError): crop_policy(box)

    def test_extract_crop_preserves_original_source_and_changes_run_identity(self):
        from tests.test_face_diagnostic_replay import replay_fixture
        from tests.test_visual_extract import FakeEncoder
        from training.visual.extract import extract
        from training.visual.data import VisualCache
        from training.visual.encoder import preprocess
        with tempfile.TemporaryDirectory(dir='runs') as tmp, replay_fixture(tmp) as (kwargs,_,_):
            output = Path(tmp)/'head'
            weights = Path(kwargs['dataset_root'])/'synthetic.pth'
            calls = []
            def capture(image, roi):
                calls.append(roi)
                return preprocess(image, roi)
            with patch('training.visual.encoder.FrozenEncoder', FakeEncoder), patch('training.visual.encoder.preprocess', capture):
                report = extract(**kwargs, output_dir=output, weights=weights,
                                 weights_sha256=file_sha256(weights), crop=crop_policy())
            row = report['records'][0]
            self.assertEqual(report['schema_version'], 'driveguard-visual-cache-v2')
            self.assertEqual(calls, [crop_roi(row['source']['driver_roi'], crop_policy())]*8)
            cache = VisualCache(output)
            cache.close()
            with patch('inference.perception.video_source.decode_video', side_effect=AssertionError('resume must not decode')):
                self.assertEqual(report, extract(**kwargs, output_dir=output, weights=weights,
                    weights_sha256=file_sha256(weights), crop=crop_policy(), continue_run=True))
            with self.assertRaisesRegex(ValueError, 'configuration/provenance'):
                extract(**kwargs, output_dir=output, weights=weights, weights_sha256=file_sha256(weights),
                        crop=crop_policy([0,0,.5,.5]), continue_run=True)

    def test_paired_data_ablation_normalization_and_mismatch(self):
        import torch
        from tests.test_visual_fatigue import fatigue_fixture
        from training.visual.prepare import prepare
        from training.visual.data import PreparedData
        from training.visual.spatial import SpatialData
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            root = Path(tmp).resolve()
            baseline, cache = fatigue_fixture(root)
            prepared_report = prepare(checkpoint=baseline['best_checkpoint'], visual_cache=cache,
                                      output_dir=root/'fatigue', task='fatigue')
            manifest = Path(prepared_report['manifest'])
            head = head_cache_fixture(root/'head', manifest, read_json(manifest)['records'])
            base = PreparedData(root/'fatigue')
            try:
                spatial = SpatialData(base, head)
                try:
                    whole, local, both = [spatial.dataset('val', arm) for arm in ('whole','head','both')]
                    a,b,c = [d[0][0] for d in (whole,local,both)]
                    self.assertEqual(a['visual'].shape, (8,1024))
                    self.assertTrue((a['visual'][:,512:] == 0).all())
                    self.assertTrue((b['visual'][:,:512] == 0).all())
                    np.testing.assert_array_equal(a['visual'][:,:512],c['visual'][:,:512])
                    np.testing.assert_array_equal(b['visual'][:,512:],c['visual'][:,512:])
                    self.assertTrue((b['visual'][:,512:] == 5).all(), 'Val extremes must not affect train-only normalization')
                    self.assertTrue(a['visual_mask'].all())
                    self.assertFalse(whole.metadata(0)['f_eligible'])
                    self.assertNotEqual(whole.prepared.provenance, local.prepared.provenance)
                    for i in range(len(whole)):
                        self.assertEqual(whole.metadata(i),local.metadata(i))
                        self.assertEqual(whole[i][1:],local[i][1:])
                finally: spatial.close()
                report = read_json(head/'manifest.json')
                row = report['records'][0]
                path = head/row['arrays']['source_pts_ms']
                values = np.load(path); values[:] -= 1; np.save(path, values)
                row['sha256']['source_pts_ms'] = file_sha256(path)
                atomic_json(head/row['directory']/'record.json',row)
                atomic_json(head/'manifest.json',report)
                with self.assertRaisesRegex(ValueError,'PTS|sampling'):
                    SpatialData(base,head)
            finally: base.close()


if __name__ == '__main__': unittest.main()
