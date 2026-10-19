from copy import deepcopy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.common.config import load_config
from training.common.checkpoint import load_checkpoint
from tests.test_relation_data import relation_fixture

ROOT = Path(__file__).resolve().parents[1]


class RelationTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT/'runs')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = load_config(ROOT/'configs/training/relation_gated.json')
        self.config['training'].update(epochs=2,batch_size=2)
        self.path,self.manifest = relation_fixture(self.root/'data',self.config)
        self.config['runtime'].update(backend='cpu',manifest=str(self.path),data_root=str(self.root/'data'),output_dir=str(self.root/'initial'))

    def factory(self):
        self.assertIsNotNone(importlib.util.find_spec('training.common.model_factory'), 'model factory missing')
        from training.common import model_factory
        return model_factory

    def test_train_resume_reload_match_continuous(self):
        factory = self.factory()
        from training.temporal.train import run_training
        from training.common.data import FeatureCorpus
        first = run_training(self.config)
        c = deepcopy(self.config)
        c['training']['epochs'] = 3
        c['runtime']['output_dir'] = str(self.root/'resumed')
        resumed = run_training(c,resume=first['last_checkpoint'])
        c['runtime']['output_dir'] = str(self.root/'continuous')
        continuous = run_training(c)
        a,b = [load_checkpoint(r['last_checkpoint']) for r in (resumed,continuous)]
        self.assertEqual(a['history'],b['history'])
        self.assertEqual(a['model_kind'],'relation_gated_v1')
        self.assertEqual(a['group_profile'],self.config['data']['group_profile'])
        for name in a['model_state']:
            torch.testing.assert_close(a['model_state'][name],b['model_state'][name],atol=0,rtol=0)
            self.assertEqual(a['model_state'][name].device.type,'cpu')
        model,payload = factory.load_model(resumed['best_checkpoint'])
        corpus = FeatureCorpus(self.config,self.path,self.root/'data')
        try:
            x,_,_ = corpus.dataset('val',payload['normalizer'])[0]
            with torch.no_grad():
                out = model(**{key:torch.as_tensor(v)[None] for key,v in x.items()},return_aux=True)
            self.assertTrue(torch.isfinite(out['logits']).all())
            self.assertTrue(out['available'].all())
        finally:
            corpus.close()

    def test_missing_inputs_skip_updates_and_preserve_label_coverage(self):
        factory = self.factory()
        from training.temporal.train import run_epoch
        model = factory.build_model(self.config)
        optimizer = torch.optim.Adam(model.parameters(),weight_decay=.01)
        item = ({'x':torch.zeros(100,38),'quality':torch.zeros(4,2),'eligible':torch.zeros(4,dtype=torch.bool)},torch.ones(4),torch.ones(4,dtype=torch.bool))
        before = deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError,'supervision|updates'):
            run_epoch(model,DataLoader([item,item],batch_size=1),torch.device('cpu'),torch.ones(4,dtype=torch.bool),optimizer)
        self.assertFalse(optimizer.state)
        for name in before:
            torch.testing.assert_close(before[name],model.state_dict()[name],atol=0,rtol=0)
        good = deepcopy(item)
        good[0]['x'][:,19:] = 1
        good[0]['quality'][:] = 1
        good[0]['eligible'][:] = True
        report = run_epoch(model,DataLoader([item,good],batch_size=2),torch.device('cpu'),torch.ones(4,dtype=torch.bool),optimizer)
        self.assertEqual(report['unavailable_windows'],1)
        self.assertEqual(report['unavailable_known_positive'],[1]*4)
        self.assertEqual(report['label_coverage']['positive'],[2]*4)
        self.assertEqual(report['known_counts'],[1]*4)

    def test_age_and_profile_changes_reject_resume(self):
        self.factory()
        from training.temporal.train import run_training
        result = run_training(self.config)
        c = deepcopy(self.config)
        c['training']['epochs'] = 3
        c['runtime']['output_dir'] = str(self.root/'bad')
        age_path = self.root/'data'/self.manifest['records'][0]['expert_age_ms']
        ages = np.load(age_path)
        ages[0,0] = 40
        np.save(age_path,ages)
        with self.assertRaisesRegex(ValueError,'fingerprint|data'):
            run_training(c,resume=result['last_checkpoint'])

    def test_all_platform_entrypoints_precheck_without_site_packages(self):
        self.factory()
        for platform in ('cpu','gpu','npu'):
            p = subprocess.run([sys.executable,'-B','-S','-m',f'training.relation.{platform}_train','--check-config'],cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertIn('relation_gated_v1',p.stdout)
            self.assertIn('"device_verified": false',p.stdout)

    def test_synthetic_smoke_proves_resume_and_reload(self):
        self.assertIsNotNone(importlib.util.find_spec('training.relation.smoke'),'relation smoke missing')
        from training.relation.smoke import run_smoke
        report = run_smoke(self.root/'smoke')
        self.assertTrue(report['synthetic_only'])
        self.assertTrue(report['resume_matches_continuous'])
        self.assertTrue(report['reload_finite'])
        self.assertLess(report['parameter_count'],100000)
        self.assertGreater(report['unavailable_windows'],0)


if __name__ == '__main__':
    unittest.main()
