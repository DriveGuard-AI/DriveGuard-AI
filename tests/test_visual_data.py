"""Visual cache/window alignment, frozen context, and train-only normalization."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tests.test_gaze_training import gaze_fixture
from training.common.config import read_json,canonical_hash
from training.common.data import file_sha256
from training.gaze.config import load_config

ROOT=Path(__file__).resolve().parents[1]


def visual_fixture(root, manifest, records, value_by_split=None):
    from training.visual.encoder import encoder_metadata
    root=Path(root); root.mkdir()
    encoder=encoder_metadata(canonical_hash('synthetic-test-weights'))
    run=dict(schema_version='driveguard-visual-extract-v1',manifest_sha256=file_sha256(manifest),
             encoder=encoder,backend='cpu',device_index=0,batch_size=3)
    (root/'input.json').write_text(json.dumps(run),encoding='utf-8')
    metadata=[]
    for record in records:
        if record['view'] != 'body': continue
        directory=root/record['record_id']; directory.mkdir()
        slots=np.array([i for i in range(300) if i%25 in (12,24)],np.int64)
        rng=np.random.default_rng(10)
        features=rng.normal(size=(len(slots),512)).astype(np.float32)
        features+= (value_by_split or {}).get(record['split'],0)
        arrays=dict(embeddings=features,slot_indices=slots,source_pts_ms=slots*40,valid=np.ones(len(slots),bool))
        paths={}
        for name,array in arrays.items():
            path=directory/(name+'.npy');np.save(path,array);paths[name]=path.relative_to(root).as_posix()
        identity={k:record[k] for k in ('record_id','recording_id','subject_id','view','subset','split')}
        identity.update(slots=300,source=dict(video='synthetic/'+record['record_id']+'.mp4',
                        sha256=canonical_hash(record['record_id']),size_bytes=100,
                        cache_fingerprint=canonical_hash(record),driver_roi=[0,0,1000,500],image_size=[1000,500]))
        fingerprint=canonical_hash(dict(run=run,record=identity))
        entry=dict(identity,fingerprint=fingerprint,status='complete',sample_count=len(slots),directory=directory.name,
                   arrays=paths,sha256={k:file_sha256(root/p) for k,p in paths.items()})
        metadata.append(entry)
        (directory/'record.json').write_text(json.dumps(entry),encoding='utf-8')
    report=dict(schema_version='driveguard-visual-cache-v1',status='complete',manifest_sha256=file_sha256(manifest),
                encoder=encoder,records=metadata)
    (root/'manifest.json').write_text(json.dumps(report),encoding='utf-8')
    return root


def prepared_fixture(root):
    from training.gaze.train import run_training
    from training.visual.prepare import prepare
    config=load_config(ROOT/'configs/training/relation_gaze_head_coverage25.json')
    config['model'].update(channels=[4],dilations=[1],projection_dim=4,gate_hidden_dim=4,
                           head_policy=dict(min_valid_ratio=.25,max_gap_steps=2))
    config['training'].update(epochs=1,batch_size=3)
    path,manifest=gaze_fixture(root/'features',config)
    for record in manifest['records']: record['view']='body'
    path.write_text(json.dumps(manifest),encoding='utf-8')
    config['runtime'].update(manifest=str(path),data_root=str(root/'features'),output_dir=str(root/'baseline'))
    baseline=run_training(config)
    cache=visual_fixture(root/'visual',path,manifest['records'],{'val':1000})
    result=prepare(checkpoint=baseline['best_checkpoint'],visual_cache=cache,output_dir=root/'prepared',backend='cpu')
    return result,baseline,cache


class VisualDataTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_cache_rejects_future_pts_and_bad_checksum(self):
        from training.visual.data import VisualCache
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp);manifest=root/'source.json';manifest.write_text('{}',encoding='utf-8')
            record=dict(record_id='r',recording_id='r',subject_id='s',view='body',subset='distraction',split='train')
            cache=visual_fixture(root/'cache',manifest,[record])
            report=read_json(cache/'manifest.json');entry=report['records'][0]
            pts=cache/entry['arrays']['source_pts_ms']; values=np.load(pts);values[0]+=1;np.save(pts,values)
            with self.assertRaisesRegex(ValueError,'checksum|SHA'): VisualCache(cache)
            entry['sha256']['source_pts_ms']=file_sha256(pts)
            (cache/'manifest.json').write_text(json.dumps(report),encoding='utf-8')
            (cache/entry['directory']/'record.json').write_text(json.dumps(entry),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'future|PTS'): VisualCache(cache)

    def test_cache_requires_encoder_source_and_completion_identity(self):
        from training.visual.data import VisualCache
        changes=('encoder','run','source','fingerprint','marker','failed','array_path')
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp);manifest=root/'source.json';manifest.write_text('{}',encoding='utf-8')
            record=dict(record_id='r',recording_id='r',subject_id='s',view='body',subset='distraction',split='train')
            for change in changes:
                with self.subTest(change=change):
                    cache=visual_fixture(root/change,manifest,[record])
                    report=read_json(cache/'manifest.json');entry=report['records'][0]
                    marker=cache/entry['directory']/'record.json'
                    if change=='encoder': report['encoder']['preprocessing']['color']='BGR'
                    if change=='run':
                        run=read_json(cache/'input.json');run['manifest_sha256']='0'*64
                        (cache/'input.json').write_text(json.dumps(run),encoding='utf-8')
                    if change=='source': entry.pop('source')
                    if change=='fingerprint': entry['fingerprint']='0'*64
                    if change=='marker': marker.write_text('{}',encoding='utf-8')
                    if change=='failed': (marker.parent/'failure.json').write_text('{}',encoding='utf-8')
                    if change=='array_path': entry['arrays']['embeddings']='r/./embeddings.npy'
                    if change not in ('marker','failed'): marker.write_text(json.dumps(entry),encoding='utf-8')
                    (cache/'manifest.json').write_text(json.dumps(report),encoding='utf-8')
                    with self.assertRaisesRegex(ValueError,'encoder|run|source|identity|fingerprint|marker|path'):
                        value=VisualCache(cache)
                        value.close()

    def test_context_finiteness_validation_is_chunked(self):
        from training.visual.data import PreparedData
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp);prepared_fixture(root)
            report=read_json(root/'prepared/prepared.json')
            entry=report['splits']['train']
            rows=read_json(root/'prepared'/entry['rows'])
            # A large valid temporal grid is not needed: the finite scan occurs
            # before row identity validation and must remain bounded either way.
            count=4097
            rows=[deepcopy(rows[0]) for _ in range(count)]
            (root/'prepared'/entry['rows']).write_text(json.dumps(rows),encoding='utf-8')
            np.save(root/'prepared'/entry['context'],np.ones((count,report['context_dim']),np.float32))
            entry['rows_sha256']=file_sha256(root/'prepared'/entry['rows'])
            entry['context_sha256']=file_sha256(root/'prepared'/entry['context'])
            (root/'prepared/prepared.json').write_text(json.dumps(report),encoding='utf-8')
            original=np.isfinite
            def bounded(value,*args,**kwargs):
                if isinstance(value,np.ndarray) and value.ndim==2:
                    self.assertLessEqual(len(value),2048,'finiteness scan must be bounded')
                return original(value,*args,**kwargs)
            with patch('training.visual.data.np.isfinite',bounded):
                with self.assertRaisesRegex(ValueError,'window identity'):
                    PreparedData(root/'prepared')

    def test_reader_accepts_actual_extractor_completion_metadata(self):
        from tests.test_face_diagnostic_replay import replay_fixture
        from tests.test_visual_extract import FakeEncoder
        from training.visual.extract import extract
        from training.visual.data import VisualCache
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp,replay_fixture(tmp) as (kwargs,_,_):
            output=Path(tmp)/'visual'
            weights=Path(kwargs['dataset_root'])/'synthetic.pth'
            with patch('training.visual.encoder.FrozenEncoder',FakeEncoder):
                report=extract(**kwargs,weights=weights,weights_sha256=file_sha256(weights),
                               output_dir=output,batch_size=3)
            cache=VisualCache(output)
            try:
                self.assertEqual(list(cache.records.values()),report['records'])
            finally: cache.close()

    def test_preparation_alignment_normalization_and_immutability(self):
        from training.visual.data import PreparedData
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            root=Path(tmp);result,baseline,cache=prepared_fixture(root)
            prepared=PreparedData(root/'prepared')
            try:
                train,val=prepared.dataset('train'),prepared.dataset('val')
                self.assertEqual(len(train),9);self.assertEqual(len(val),9)
                inputs,target,known=train[0]
                self.assertEqual(inputs['visual'].shape,(8,512))
                self.assertEqual(inputs['context'].shape,(36,))
                self.assertTrue(inputs['visual_mask'].all())
                self.assertEqual(train.metadata(0)['end_slot'],99)
                self.assertEqual(train.metadata(0)['visual_start'],0)
                self.assertEqual(train.metadata(1)['visual_start'],2)
                self.assertLess(max(abs(x) for x in prepared.normalizer['visual']['mean']),1.)
                self.assertGreater(float(val[0][0]['visual'].mean()),3.)
                self.assertEqual(result['baseline_checkpoint_sha256'],file_sha256(baseline['best_checkpoint']))
            finally: prepared.close()
            with self.assertRaises(FileExistsError):
                from training.visual.prepare import prepare
                prepare(checkpoint=baseline['best_checkpoint'],visual_cache=cache,output_dir=root/'prepared')

    def test_context_is_masked_and_baseline_not_modified(self):
        from training.visual.prepare import frozen_context
        from training.common.model_factory import build_model
        config=load_config(ROOT/'configs/training/relation_gaze_head_coverage25.json')
        model=build_model(config).eval()
        x=torch.ones(2,100,46);x[0,:,23:]=0
        inputs=dict(x=x,quality=torch.ones(2,4,2),eligible=torch.tensor([[False]*4,[True]*4]))
        before={k:v.clone() for k,v in model.state_dict().items()}
        context,outputs=frozen_context(model,inputs)
        self.assertEqual(context.shape,(2,96))
        self.assertFalse(context.requires_grad)
        self.assertFalse(context[0,:80].any())
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
        self.assertFalse(outputs['task_available'][0].any())


if __name__ == '__main__': unittest.main()
