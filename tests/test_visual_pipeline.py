"""Orchestration integration with synthetic RGB cache; real baseline/probe training."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from tests.test_visual_data import ROOT,prepared_fixture,visual_fixture
from training.common.config import read_json


class VisualPipelineTests(unittest.TestCase):
    def test_pipeline_completion_reuse_and_config_rejection(self):
        from training.visual.pipeline import run
        from training.visual.suite import DEFAULTS
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp);prepared,baseline,_=prepared_fixture(root)
            dataset=root/'dataset';dataset.mkdir()
            assets=root/'assets';assets.mkdir();weights=assets/'weights.pth';weights.write_bytes(b'synthetic')
            manifest=Path(prepared['manifest']);records=read_json(manifest)['records']
            def extract(source,data_root,dataset_root,weights,sha,output,**kwargs):
                if output.exists():return read_json(output/'manifest.json')
                visual_fixture(output,source,records)
                return read_json(output/'manifest.json')
            settings=dict(DEFAULTS,epochs=1,batch_size=3,hidden_dim=8)
            options=dict(manifest=manifest,data_root=prepared['data_root'],dataset_root=dataset,
                         checkpoint=baseline['best_checkpoint'],weights=weights,output_dir=root/'pipeline',settings=settings)
            with patch.dict('sys.modules',{'av':types.ModuleType('av')}),patch(
                    'training.visual.weights.ensure',return_value=weights),patch('training.visual.extract.extract',side_effect=extract):
                report=run(**options)
                self.assertEqual(report['status'],'complete')
                result=read_json(root/'pipeline'/'result.json')
                self.assertEqual(read_json(result['comparison']),report)
                stamps={str(p):p.stat().st_mtime_ns for p in (root/'pipeline').rglob('*.npy')}
                self.assertEqual(run(**options,continue_run=True),report)
                self.assertEqual(stamps,{str(p):p.stat().st_mtime_ns for p in (root/'pipeline').rglob('*.npy')})
                with self.assertRaisesRegex(ValueError,'identity'):
                    run(**dict(options,settings=dict(settings,seed=43)),continue_run=True)


if __name__=='__main__':unittest.main()
