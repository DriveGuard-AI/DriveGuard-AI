"""Suite identity, comparable report denominators, and small end-to-end execution."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from training.gaze.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class BodyDistSuiteTests(unittest.TestCase):
    def test_failed_retry_preserves_resume_checkpoint(self):
        from training.gaze.body_dist_suite import run_suite, _hash
        config=load_config(ROOT/'configs/training/relation_gaze_head_coverage25.json')
        config['training']['epochs']=2
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temporary:
            root=Path(temporary); manifest=root/'manifest.json'; manifest.write_text('{}',encoding='utf-8')
            config['runtime'].update(manifest=str(manifest),data_root=str(root/'data'),backend='cpu')
            commands=[]
            def failed_worker(command,log):
                commands.append(command)
                if len(commands)==1:
                    directory=Path(command[command.index('--output-dir')+1]);directory.mkdir()
                    saved=json.loads(Path(command[command.index('--config')+1]).read_text(encoding='utf-8'))
                    saved['runtime']['output_dir']=str(directory)
                    (directory/'config.json').write_text(json.dumps(saved),encoding='utf-8')
                    (directory/'last.pt').write_bytes(b'checkpoint placeholder for orchestration test')
                    (directory/'last.pt.sha256.json').write_text(json.dumps(dict(sha256=_hash(directory/'last.pt'))),encoding='utf-8')
                raise RuntimeError('simulated interrupted child')
            with patch('training.gaze.body_dist_suite._worker',side_effect=failed_worker),patch('training.common.checkpoint.load_checkpoint',return_value={'epoch':1}):
                for attempt in range(3):
                    with self.assertRaisesRegex(RuntimeError,'interrupted'):
                        run_suite(config,root/'suite',variants=['baseline'],continue_run=attempt>0)
            self.assertIn('--resume',commands[1])
            self.assertIn('--resume',commands[2])
            self.assertEqual(commands[1][-1],commands[2][-1])

    def test_arms_change_one_factor_and_precheck_without_torch(self):
        from training.gaze.body_dist_suite import build_variants
        base = load_config(ROOT/'configs/training/relation_gaze_head_coverage25.json')
        original = deepcopy(base)
        variants = build_variants(base)
        self.assertEqual(base,original)
        self.assertEqual(list(variants),['baseline','sampling','face_dropout','gap80'])
        for name, config in variants.items():
            self.assertEqual(config['data'],base['data'])
            self.assertEqual(config['training']['seed'],base['training']['seed'])
            self.assertEqual(config['model']['head_policy']['min_valid_ratio'],.25)
        self.assertEqual(variants['sampling']['training']['body_dist_experiment']['missing_head_positive_multiplier'],3.)
        self.assertEqual(variants['face_dropout']['training']['body_dist_experiment']['face_dropout_probability'],.3)
        self.assertEqual(variants['gap80']['model']['head_policy']['max_gap_steps'],2)
        result = subprocess.run([sys.executable,'-S','-m','training.gaze.body_dist_suite','--check-config'],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('face_dropout',result.stdout)

    def make_report(self,root,name, fingerprint='abc', changed_label=False):
        path=root/name; path.mkdir()
        summary=dict(status='complete',data_fingerprint=fingerprint,checkpoint_epoch=1,
                     selection=dict(split='val',view='all',subset='all'),threshold=.5)
        (path/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
        rows=[]
        for index,(target,logit,available) in enumerate([(1,1.,True),(1,None,False),(0,1.,True),(0,-1.,True)]):
            rows.append(dict(record_id='r',start_slot=index*25,end_slot=index*25+99,view='body',subset='distraction',
                             label_known=True,target=target,logit=logit,probability=.73 if logit==1. else (.27 if logit==-1. else None),
                             behavior_available=available,h_eligible=False,f_eligible=False,o_eligible=available,ho_eligible=False,fo_eligible=False,
                             f_dist_weight=0.,o_dist_weight=1. if available else 0.,ho_dist_weight=0.,fo_dist_weight=0.,h_dist_weight=0.))
        if changed_label: rows[0]['target']=0
        with (path/'windows.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        return path

    def test_report_preserves_unavailable_denominator_and_checks_identity(self):
        from training.gaze.body_dist_report import compare_reports
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temporary:
            root=Path(temporary); a=self.make_report(root,'a'); b=self.make_report(root,'b')
            result=compare_reports([dict(variant='baseline',checkpoint='best',directory=str(a)),dict(variant='sampling',checkpoint='best',directory=str(b))])
            row=result['comparisons'][0]
            self.assertEqual(row['body']['positive'],2)
            self.assertEqual(row['body']['unavailable_positive'],1)
            self.assertEqual(row['body']['recall'],1.)
            self.assertEqual(row['body']['all_positive_detection_rate'],.5)
            self.assertEqual(row['body']['false_positive_rate'],.5)
            self.assertEqual(row['body_h_ineligible']['tp'],1)
            bad=self.make_report(root,'bad',changed_label=True)
            with self.assertRaisesRegex(ValueError,'window|label'):
                compare_reports([dict(variant='baseline',checkpoint='best',directory=str(a)),dict(variant='bad',checkpoint='best',directory=str(bad))])
            bad=self.make_report(root,'bad_data',fingerprint='different')
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                compare_reports([dict(variant='baseline',checkpoint='best',directory=str(a)),dict(variant='bad',checkpoint='best',directory=str(bad))])

    def test_synthetic_suite_and_idempotent_continue(self):
        import numpy as np
        from tests.test_gaze_training import gaze_fixture
        from training.gaze.body_dist_suite import run_suite
        config=load_config(ROOT/'configs/training/relation_gaze_head_coverage25.json')
        config['model'].update(channels=[4],dilations=[1],projection_dim=4,gate_hidden_dim=4)
        config['training'].update(epochs=1,batch_size=3)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as temporary:
            root=Path(temporary); path,manifest=gaze_fixture(root/'data',config)
            for original in list(manifest['records']):
                body=dict(original,record_id=original['record_id']+'body',view='body')
                for key in ('values','masks','targets','label_masks','expert_age_ms','gaze_targets','gaze_label_masks'):
                    value=np.load(root/'data'/original[key]); body[key]='body_'+original[key]
                    if key == 'masks': value[150:,5:8]=0
                    np.save(root/'data'/body[key],value)
                manifest['records'].append(body)
            path.write_text(json.dumps(manifest),encoding='utf-8')
            config['runtime'].update(manifest=str(path),data_root=str(root/'data'),backend='cpu')
            arms=['baseline','sampling','face_dropout','gap80']
            result=run_suite(config,root/'suite',variants=arms)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(result['comparisons']),8)
            stamps={str(p):p.stat().st_mtime_ns for p in (root/'suite').rglob('*.pt')}
            again=run_suite(config,root/'suite',variants=arms,continue_run=True)
            self.assertEqual(result,again)
            self.assertEqual(stamps,{str(p):p.stat().st_mtime_ns for p in (root/'suite').rglob('*.pt')})
            # A later comparison rejection must keep the validated prefix and mark failure.
            from training.gaze.body_dist_report import compare_reports
            def reject_second(entries):
                if len(entries)>1: raise ValueError('simulated window identity mismatch')
                return compare_reports(entries)
            with patch('training.gaze.body_dist_suite.compare_reports',side_effect=reject_second):
                with self.assertRaisesRegex(ValueError,'window identity'):
                    run_suite(config,root/'suite',variants=arms,continue_run=True)
            failed=json.loads((root/'suite/comparison.json').read_text(encoding='utf-8'))
            self.assertEqual(failed['status'],'failed')
            self.assertEqual(len(failed['comparisons']),1)
            changed=deepcopy(config); changed['training']['seed']+=1
            with self.assertRaisesRegex(ValueError,'configuration|identity'):
                run_suite(changed,root/'suite',variants=arms,continue_run=True)


if __name__ == '__main__': unittest.main()
