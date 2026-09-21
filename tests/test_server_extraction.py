from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys
import json

from training.common.config import read_json
from inference.perception.runtime import load_runtime_config, runtime_signature


class ServerExtractionTests(unittest.TestCase):
    def test_status_detects_dead_or_reused_pid(self):
        from training.dmd.server_runner import session_status
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)
            (base/'owner.json').write_text(json.dumps({'pid':123,'process_start':'old'}))
            (base/'status.json').write_text(json.dumps({'status':'extracting'}))
            for actual,expected in ((None,False),('new',False),('old',True)):
                with patch('training.dmd.server_runner.process_start',return_value=actual):
                    result=session_status(base)
                self.assertEqual(result['supervisor_alive'],expected)
                self.assertEqual(result['status'],'extracting' if expected else 'supervisor_not_running')

    def exercise_runner(self, mode):
        from training.dmd.server_setup import configure
        from training.dmd.server_runner import supervise
        from tests import test_gaze_batch as fixtures
        fixture=fixtures.GazeBatchTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.configure(); weights=fixture.base/'yolo11s.pt'; weights.write_bytes(b'fixture')
        new=fixture.base/'yolo.local.json'; configure(fixture.config_path,new,weights)
        session=fixture.base/'session'; calls=[]; state={'service_dead':False}
        args=SimpleNamespace(config=str(new),dataset_root=str(fixture.dataset),output_dir=str(fixture.output),
            session_dir=str(session),npu_python=sys.executable,cpu_python=sys.executable,
            workers_dist=1,workers_drows=1,workers_gaze=1,threads=1,ready_timeout=1,resume=False)
        if mode=='locked':
            fixture.output.mkdir(parents=True,exist_ok=True)
            (fixture.output/'.supervisor.lock').write_text('existing owner')
        class Child:
            def __init__(self,kind): self.kind=kind; self.pid=100+len(calls); self.returncode=None; self.count=0
            def poll(self):
                self.count+=1
                if self.kind=='service': return 1 if state['service_dead'] else self.returncode
                if self.kind=='extract':
                    if mode=='crash' and self.count==1:
                        state['service_dead']=True
                        return None
                    self.returncode=0
                if self.kind=='publish' and self.count>3: self.returncode=0
                return self.returncode
            def terminate(self):
                if self.kind=='publish': raise AssertionError('Must drain publisher to release its cache lock')
                self.returncode=-15
            def kill(self): self.returncode=-9
            def wait(self,timeout=None):
                if self.returncode is None: self.returncode=0
                return self.returncode
        def launch(command,**kwargs):
            kind='service' if 'inference.experts.detector_service' in command else ('publish' if 'publish' in command else 'extract')
            calls.append(kind)
            if kind=='service': (session/'detector-ready.json').write_text('{}',encoding='utf-8')
            if kind=='publish': (session/'STOP').touch()
            return Child(kind)
        with patch('training.dmd.server_runner.sys.platform','linux'), patch('training.dmd.server_runner.subprocess.Popen',side_effect=launch), patch('training.dmd.server_runner.time.sleep'):
            if mode=='locked':
                with self.assertRaises(FileExistsError): supervise(args)
                self.assertEqual(calls,[])
                self.assertEqual((fixture.output/'.supervisor.lock').read_text(),'existing owner')
                return
            if mode=='crash':
                with self.assertRaisesRegex(RuntimeError,'Detector'): supervise(args)
            else:
                self.assertEqual(supervise(args),2)
        report=read_json(session/'status.json')
        self.assertEqual(report['status'],'failed' if mode=='crash' else 'stopped')
        self.assertFalse((fixture.output/'.supervisor.lock').exists())

    def test_detector_crash_is_failure_not_user_stop(self):
        self.exercise_runner('crash')

    def test_second_supervisor_rejected_before_model_process(self):
        self.exercise_runner('locked')

    def test_stop_during_publish_is_honored(self):
        self.exercise_runner('stop')

    def test_service_identity_change_cannot_publish_a_record(self):
        from tests import test_gaze_batch as fixtures
        from training.dmd.gaze_batch import run_batch
        fixture=fixtures.GazeBatchTests(); fixture.setUp()
        try:
            fixture.configure()
            with fixture.patches(), patch('training.dmd.batch._environment',return_value={'detector_service_model_id':'expected'}):
                result=run_batch(fixture.config_path,fixture.dataset,fixture.output)
            self.assertEqual(result['completed'],0)
            self.assertEqual(result['failed'],10)
            self.assertTrue(all('identity' in j['error'] for j in result['jobs']))
        finally: fixture.doCleanups()

    def test_runner_commands_keep_environments_separate(self):
        from training.dmd.server_runner import commands, perception_env
        c=commands('npu-python','cpu-python','batch.json','runtime.json','Dataset','cache','session', (6,3,3),1)
        self.assertEqual(c['service'][0],'npu-python')
        self.assertEqual(c['extract'][0],'cpu-python')
        self.assertIn('training.dmd.gaze_batch',c['extract'])
        self.assertIn('--workers-dist',c['extract'])
        self.assertIn('publish',c['publish'])
        env=perception_env({'PYTHONPATH':'CANN','PYTHONHOME':'bad','OTHER':'keep'})
        self.assertNotIn('PYTHONPATH',env)
        self.assertNotIn('PYTHONHOME',env)
        self.assertEqual(env['OTHER'],'keep')
        self.assertEqual(env['PYTHONNOUSERSITE'],'1')

    def config(self):
        c=read_json('configs/perception/dmd_gaze_cpu.json')
        c['objects'].update(variant='yolo11s',backend='npu',input_size=640,
            service=dict(host='127.0.0.1',port=18761,token_file='token'))
        return c

    def test_yolo11_runtime_validation_and_portable_identity(self):
        c=self.config()
        with patch('inference.perception.runtime.read_json',return_value=c):
            result=load_runtime_config('configs/perception/example.json')
        self.assertTrue(Path(result['objects']['service']['token_file']).is_absolute())
        other=deepcopy(result); other['objects']['service'].update(port=18762,token_file='/tmp/other-token')
        self.assertEqual(runtime_signature(result),runtime_signature(other))

    def test_remote_runtime_rejects_external_host_or_bad_port(self):
        for key,value in (('host','0.0.0.0'),('port',True),('port',0),('token_file','')):
            c=self.config(); c['objects']['service'][key]=value
            with patch('inference.perception.runtime.read_json',return_value=c):
                with self.assertRaises(ValueError): load_runtime_config('unused.json')

    def test_configuration_migration_preserves_roi_and_refuses_overwrite(self):
        from training.dmd.server_setup import configure
        from tests import test_gaze_batch as fixtures
        fixture=fixtures.GazeBatchTests(); fixture.setUp()
        try:
            original=fixture.configure(); before=fixture.config_path.read_bytes()
            weights=fixture.base/'yolo11s.pt'; weights.write_bytes(b'fixture-not-a-model')
            new=fixture.base/'full-yolo.local.json'
            result=configure(fixture.config_path,new,weights,18761)
            created=read_json(new)
            self.assertEqual(created['jobs'],original['jobs'])
            self.assertEqual(created['preparation_sha256'],original['preparation_sha256'])
            self.assertEqual(fixture.config_path.read_bytes(),before)
            runtime=load_runtime_config(new.parent/created['runtime_config'])
            self.assertEqual(runtime['objects']['variant'],'yolo11s')
            self.assertEqual(runtime['objects']['input_size'],640)
            self.assertTrue(Path(runtime['objects']['service']['token_file']).is_file())
            with self.assertRaises(FileExistsError): configure(fixture.config_path,new,weights,18761)
        finally:
            fixture.doCleanups()
