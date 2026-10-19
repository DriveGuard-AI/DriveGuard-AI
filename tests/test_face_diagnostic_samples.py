"""Real image write/read roundtrip, bounded sampling, and escaped local gallery."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from inference.perception.video_source import VideoFrame
from training.dmd.face_diagnostic_samples import Samples, write_gallery


class SampleTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'Requires perception environment OpenCV')
    def test_real_jpeg_bounded_samples_do_not_mutate_frames(self):
        import cv2
        detail=dict(reason='ok',iris_reason='ok',detected_faces=1,
                    candidates=[dict(bbox=[20,20,80,80],invalid_reason=None,inside_roi=True)])
        original=deepcopy(detail)
        bgr=np.zeros((100,160,3),dtype=np.uint8)
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            samples=Samples(Path(tmp),2)
            for timestamp in (0,40,1000,2000,3000):
                samples.add(VideoFrame(timestamp,bgr),detail,[0,0,100,99],
                            landmarks=[(30,30),(60,40)],crop=[10,10,110,90])
            self.assertEqual([i['source_pts_ms'] for i in samples.items],[0,1000])
            self.assertEqual(len(list(Path(tmp).glob('*.jpg'))),2)
            image=cv2.imread(str(Path(tmp)/samples.items[0]['file']))
            self.assertEqual(image.shape,bgr.shape)
            self.assertTrue(image.any())
            self.assertFalse(bgr.any())
            self.assertEqual(original,detail)
            disabled=Samples(Path(tmp),0)
            disabled.add(VideoFrame(4000,bgr),detail,[0,0,100,99])
            self.assertFalse(disabled.items)

    def test_gallery_escapes_ids_and_handles_no_samples(self):
        record=dict(record_id='<script>alert(1)</script>',view='body',slots=25,partial=True,
                    head_mask_mismatches=3,iris_mask_mismatches=2,directory='record-001',samples=[])
        with tempfile.TemporaryDirectory(dir='runs') as tmp:
            write_gallery(Path(tmp),[record])
            html=(Path(tmp)/'index.html').read_text(encoding='utf-8')
            self.assertNotIn('<script>',html)
            self.assertIn('&lt;script&gt;',html)
            self.assertIn('record-001/frames.csv',html)


if __name__ == '__main__': unittest.main()
