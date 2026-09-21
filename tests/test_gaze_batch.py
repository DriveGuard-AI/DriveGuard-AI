from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_dmd_batch as legacy
from tests.test_dmd_batch import write, ROOT
from tests.test_gaze_perception import gaze_packet
from training.common.config import read_json


class GazeBatchTests(unittest.TestCase):
    setUp = legacy.BatchTests.setUp
    patches = legacy.BatchTests.patches

    def configure(self, confirmed=True):
        from training.dmd.gaze_batch import init_config
        init_config(self.prep,self.config_path)
        config = read_json(self.config_path)
        for item in config['jobs'].values():
            item['confirmed_exclusive'] = confirmed
        write(self.config_path,config)
        return config

    def fake_extract(self, source, output, runtime, *, group_profile, frame_observer, feature_profile):
        from inference.perception.export import export_recording
        from inference.perception.video_source import VideoFrame
        def packets():
            for index in range(325):
                frame_observer(VideoFrame(index*40,np.zeros((500,1000,3),np.uint8)))
                value = gaze_packet(index)
                value.update(recording_id=runtime['recording_id'],view=runtime['view'],driver_roi=runtime['driver_roi'])
                yield value
        return export_recording(packets(),output,feature_profile,group_profile=group_profile,
                                provenance={'synthetic_test_only':True})

    def test_all_selection_includes_gaze_face_and_copies_only_exact_reviewed_roi(self):
        from training.dmd.batch_config import init_config as old_init
        from training.dmd.gaze_batch import init_config, load_batch
        previous = self.base/'old.json'
        old_init(self.prep,previous)
        old = read_json(previous)
        reviewed = next(iter(old['jobs']))
        old['jobs'][reviewed]['confirmed_exclusive'] = True
        write(previous,old)
        original = self.prep.read_bytes()
        init_config(self.prep,self.config_path,roi_from=[previous])
        context = load_batch(self.config_path)
        self.assertEqual(len(context['selected']),10)
        self.assertEqual(sum(j['subset']=='gaze' for j in context['selected']),2)
        self.assertTrue(all(j['view']=='face' for j in context['selected'] if j['subset']=='gaze'))
        self.assertTrue(context['config']['jobs'][reviewed]['confirmed_exclusive'])
        self.assertEqual(sum(s['confirmed_exclusive'] for s in context['config']['jobs'].values()),1)
        self.assertEqual(self.prep.read_bytes(),original)
        with self.assertRaises(FileExistsError):
            init_config(self.prep,self.config_path)

    def test_unconfirmed_rejected_and_legacy_runtime_not_accepted(self):
        from training.dmd.gaze_batch import run_batch, load_batch
        self.configure(False)
        with self.assertRaisesRegex(ValueError,'ROI'):
            run_batch(self.config_path,self.dataset,self.output)
        config=read_json(self.config_path)
        config['runtime_config']=str(ROOT/'configs/perception/dmd_cpu.json')
        write(self.config_path,config)
        with self.assertRaisesRegex(ValueError,'runtime'):
            load_batch(self.config_path)

    def test_extract_cache_publish_coverage_failure_is_diagnostic(self):
        from training.dmd.gaze_batch import run_batch,publish_batch
        self.configure()
        with self.patches():
            result=run_batch(self.config_path,self.dataset,self.output)
            self.assertEqual(result['failed'],0,result)
            self.assertEqual(result['completed'],10)
            repeat=run_batch(self.config_path,self.dataset,self.output)
            self.assertEqual(repeat['reused'],10)
            with self.assertRaisesRegex(ValueError,'report.json'):
                publish_batch(self.config_path,self.dataset,self.output,ROOT/'configs/training/relation_gaze.json')
        self.assertFalse(list(self.output.glob('manifests/*/manifest.json')))
        reports=list(self.output.glob('manifests/*/report.json'))
        self.assertTrue(reports)
        self.assertIn('coverage',read_json(reports[0]))

    def test_full_coverage_publishes_and_cpu_trains(self):
        from training.dmd.gaze_batch import run_batch,publish_batch
        from training.dmd.audit import audit_dataset
        from training.dmd.prepare import prepare_dataset
        from training.gaze.profile import GAZE_CLASSES
        for path in self.dataset.rglob('*ann_*.json'):
            doc=read_json(path)
            root=doc['openlabel']
            root['frame_intervals']=[{'frame_start':0,'frame_end':324}]
            for stream in root['streams'].values():
                stream['stream_properties']['total_frames']=325
            if 'ann_gaze' in path.name:
                root['actions']={str(i):{'type':'gaze_zone/'+name,'frame_intervals':[
                    {'frame_start':0 if i==0 else 100+(i-1)*25,
                     'frame_end':99 if i==0 else (324 if i==8 else 99+i*25)}]}
                    for i,name in enumerate(GAZE_CLASSES)}
            write(path,doc)
        audit_dataset(self.dataset,self.base/'audit-full')
        prepare_dataset(audit_path=self.base/'audit-full/audit.json',dataset_root=self.dataset,
                        output_dir=self.base/'prepared-full',split_path=self.base/'split.json')
        self.prep=self.base/'prepared-full/preparation.json'
        self.configure()
        with self.patches():
            result=run_batch(self.config_path,self.dataset,self.output)
            self.assertEqual(result['failed'],0,result)
            published=publish_batch(self.config_path,self.dataset,self.output,ROOT/'configs/training/relation_gaze.json')
        manifest=read_json(published['manifest'])
        self.assertEqual(manifest['schema_version'],'driveguard-feature-manifest-v3')
        self.assertEqual(len(manifest['records']),10)
        from training.gaze.config import load_config
        from training.gaze.train import run_training
        config=load_config(ROOT/'configs/training/relation_gaze.json')
        config['training']['epochs']=1
        config['runtime'].update(backend='cpu',manifest=published['manifest'],data_root=str(self.output),
                                 output_dir=str(self.base/'joint-train'))
        trained=run_training(config)
        self.assertGreater(trained['global_step'],0)
        self.assertTrue(Path(trained['best_checkpoint']).is_file())

    def test_corrupted_gaze_array_forces_only_its_record_to_reextract(self):
        from training.dmd.gaze_batch import run_batch
        self.configure()
        with self.patches():
            result=run_batch(self.config_path,self.dataset,self.output)
            target=self.output/result['jobs'][0]['attempt']/'gaze_targets.npy'
            with target.open('ab') as handle:
                handle.write(b'corrupt')
            repeat=run_batch(self.config_path,self.dataset,self.output)
            self.assertEqual(repeat['completed'],1)
            self.assertEqual(repeat['reused'],9)

    def test_fingerprint_covers_joint_training_implementation(self):
        from training.dmd.batch import _environment
        from inference.experts.assets import file_sha256
        original=_environment()['implementation_sha256']
        def changed(path):
            return '0'*64 if 'training/gaze/' in Path(path).as_posix() else file_sha256(path)
        with patch('training.dmd.batch.file_sha256',side_effect=changed):
            self.assertNotEqual(_environment()['implementation_sha256'],original)
