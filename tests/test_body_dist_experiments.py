"""Controlled training interventions, without claims about real-data accuracy."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from tests.test_gaze_training import gaze_fixture
from training.gaze.config import load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


class BodyDistExperimentTests(unittest.TestCase):
    def test_sampling_excludes_unknown_unavailable_and_face(self):
        from types import SimpleNamespace
        from training.gaze.body_dist import audit_dataset
        config=self.config()
        class Dataset:
            corpus=SimpleNamespace(config=config)
            records=[dict(split='train')]
            def __len__(self): return 4
            def metadata(self,index): return dict(view='face' if index==3 else 'body')
            def __getitem__(self,index):
                x=np.zeros((100,46),np.float32)
                inputs=dict(x=x,quality=np.ones((4,2),np.float32),eligible=np.array([False,index!=2,False,False]))
                labels=np.ones(4,bool); labels[3]=index!=1
                return inputs,np.ones(4),labels,0,False
        audit,weights=audit_dataset(Dataset(),3.)
        self.assertEqual(weights.tolist(),[3.,1.,1.,1.])
        self.assertEqual(audit['weighted_windows'],1)

    def test_neutral_intervention_preserves_baseline_training(self):
        from training.gaze.train import run_training
        from training.common.checkpoint import load_checkpoint
        config=self.config(); config['training'].pop('body_dist_experiment')
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temporary:
            root=Path(temporary); path,_=gaze_fixture(root/'data',config)
            config['runtime'].update(manifest=str(path),data_root=str(root/'data'),output_dir=str(root/'legacy'))
            legacy=run_training(config)
            config['runtime']['output_dir']=str(root/'neutral')
            config['training']['body_dist_experiment']=dict(missing_head_positive_multiplier=1.,face_dropout_probability=0.)
            neutral=run_training(config)
            a,b=[load_checkpoint(item['last_checkpoint']) for item in (legacy,neutral)]
            self.assertEqual(a['history'],b['history'])
            for key in a['model_state']: torch.testing.assert_close(a['model_state'][key],b['model_state'][key],rtol=0,atol=0)

    def setUp(self):
        torch.set_num_threads(1)

    def config(self):
        config = load_config(ROOT/'configs/training/relation_gaze_head_coverage25.json')
        config['model'].update(channels=[4], dilations=[1], projection_dim=4, gate_hidden_dim=4)
        config['training'].update(epochs=1, batch_size=3)
        config['training']['body_dist_experiment'] = dict(missing_head_positive_multiplier=3., face_dropout_probability=.3)
        return config

    def test_optional_config_is_strict_and_legacy_unchanged(self):
        config = self.config()
        self.assertEqual(validate_config(config), config)
        for key, value in [('face_dropout_probability', 1.1), ('face_dropout_probability', True),
                           ('missing_head_positive_multiplier', .5), ('missing_head_positive_multiplier', float('nan'))]:
            bad = deepcopy(config); bad['training']['body_dist_experiment'][key] = value
            with self.assertRaises(ValueError): validate_config(bad)
        bad = deepcopy(config); bad['model']['kind'] = 'relation_gaze_head_v1'; bad['model'].pop('head_policy')
        with self.assertRaises(ValueError): validate_config(bad)
        config['training'].pop('body_dist_experiment')
        self.assertEqual(validate_config(config), config)

    def test_dropout_removes_all_face_dependencies_without_mutation(self):
        from training.gaze.body_dist import drop_face_evidence
        from training.gaze.profile import GROUP_INDICES
        from training.gaze.head_evidence import head_statistics
        from training.gaze.preprocessing import gaze_available
        inputs = dict(x=np.ones((100,46), np.float32), quality=np.ones((4,2), np.float32), eligible=np.ones(4,bool))
        result = drop_face_evidence(inputs)
        face = list(GROUP_INDICES[0]+GROUP_INDICES[3])
        self.assertFalse(result['x'][:,face].any())
        self.assertFalse(result['x'][:,[23+i for i in face]].any())
        self.assertEqual(result['eligible'].tolist(), [False,True,True,False])
        self.assertFalse(result['quality'][[0,3]].any())
        self.assertFalse(bool(head_statistics(result['x'], dict(min_valid_ratio=.25,max_gap_steps=0))['eligible']))
        self.assertFalse(gaze_available(result['x'][:,23:], result['eligible']))
        np.testing.assert_array_equal(result['x'][:,GROUP_INDICES[1]+GROUP_INDICES[2]], 1.)
        self.assertTrue(inputs['x'].all()); self.assertTrue(inputs['eligible'].all())

    def test_audit_sampling_and_augmentation_are_train_only(self):
        from training.gaze.body_dist import audit_dataset, TrainingDataset
        from training.gaze.data import FeatureCorpus
        config = self.config()
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temporary:
            root = Path(temporary); path, manifest = gaze_fixture(root/'data', config)
            for record in manifest['records']: record['view'] = 'body'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            masks = np.load(root/'data/train_masks.npy'); masks[150:,5:8] = 0
            np.save(root/'data/train_masks.npy', masks)
            corpus = FeatureCorpus(config,path,root/'data')
            try:
                norm = corpus.fit_normalizer(); train = corpus.dataset('train',norm)
                audit, weights = audit_dataset(train, multiplier=3.)
                self.assertEqual(audit['windows'], len(train))
                self.assertGreater(audit['weighted_windows'], 0)
                for i, weight in enumerate(weights):
                    inputs, target, known, *_ = train[i]
                    from training.gaze.head_evidence import head_statistics
                    wanted = known[3] and target[3] == 1 and not head_statistics(inputs['x'],config['model']['head_policy'])['eligible'] and inputs['eligible'].any()
                    self.assertEqual(weight, 3. if wanted else 1.)
                augmented = TrainingDataset(train,1.)
                self.assertFalse(augmented[0][0]['eligible'][0])
                self.assertTrue(train[0][0]['eligible'][0])
                val = corpus.dataset('val',norm)
                with self.assertRaisesRegex(ValueError,'train'): TrainingDataset(val,1.)
                with self.assertRaisesRegex(ValueError,'train'): audit_dataset(val,multiplier=3.)
            finally: corpus.close()

    def test_resume_exact_with_sampling_and_dropout(self):
        from training.gaze.train import run_training
        from training.common.checkpoint import load_checkpoint
        config = self.config()
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temporary:
            root = Path(temporary); path, manifest = gaze_fixture(root/'data',config)
            # Keep face recording for complete gaze coverage; add a body recording
            # referencing independent arrays, with the same subject in each split.
            for record in list(manifest['records']):
                body = dict(record, record_id=record['record_id']+'-body', view='body')
                for name in ('values','masks','targets','label_masks','expert_age_ms','gaze_targets','gaze_label_masks'):
                    data = np.load(root/'data'/record[name]).copy()
                    if name == 'masks': data[150:,5:8] = 0
                    body[name] = 'body_'+record[name]; np.save(root/'data'/body[name],data)
                manifest['records'].append(body)
            path.write_text(json.dumps(manifest),encoding='utf-8')
            config['runtime'].update(manifest=str(path),data_root=str(root/'data'),output_dir=str(root/'first'))
            first = run_training(config)
            config['training']['epochs'] = 2; config['runtime']['output_dir'] = str(root/'resume')
            resumed = run_training(config,resume=first['last_checkpoint'])
            config['runtime']['output_dir'] = str(root/'full'); full = run_training(config)
            a,b = [load_checkpoint(result['last_checkpoint']) for result in (resumed,full)]
            self.assertEqual(a['history'],b['history'])
            for key in a['model_state']: torch.testing.assert_close(a['model_state'][key],b['model_state'][key],rtol=0,atol=0)
            self.assertTrue((root/'full/body_dist_audit.json').is_file())
            self.assertGreater(a['history'][-1]['train']['face_dropout']['applied'],0)
            self.assertNotIn('face_dropout',a['history'][-1]['val'])


if __name__ == '__main__': unittest.main()
