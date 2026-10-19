"""Read-only all-task composition and real synthetic checkpoint integration."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import torch

from training.common.config import read_json
from training.common.data import file_sha256
from tests.test_visual_data import prepared_fixture

ROOT = Path(__file__).resolve().parents[1]


def joint_row(view='body', start=0):
    return dict(record_id=view, recording_id=view, subject_id='s', view=view,
                subset='synthetic', split='val', start_slot=start, end_slot=start+99,
                targets=[1, 0, 1, 1], label_known=[True]*4,
                behavior_logits=[.2, -.2, .5, -.8], task_available=[True]*4,
                thresholds=[0.]*4, gaze_logits=[.1]*9, gaze_target=0,
                gaze_label_known=True, gaze_available=True, h_eligible=False)


def probe_row(row, available=True):
    result = {k: row[k] for k in ('record_id','recording_id','subject_id','view','subset','split',
                                'start_slot','end_slot','h_eligible')}
    result.update(target=row['targets'][3], label_known=row['label_known'][3],
                  available=available, logit=2. if available else None)
    return result


class CompositionTests(unittest.TestCase):
    def test_body_only_replacement_fallback_and_pairing(self):
        from training.visual.all_tasks import compose, summarize_all, paired_errors
        rows=[joint_row(),joint_row('face'),joint_row(start=25)]
        before=deepcopy(rows)
        out,audit=compose(rows,[probe_row(rows[0]),probe_row(rows[2],False)],1.)
        self.assertEqual(rows,before)
        self.assertEqual(out[1],rows[1])
        self.assertEqual(out[2]['behavior_logits'],rows[2]['behavior_logits'])
        self.assertEqual(out[0]['behavior_logits'],[.2,-.2,.5,2.])
        self.assertEqual(out[0]['thresholds'],[0.,0.,0.,1.])
        self.assertTrue(audit['passed'])
        self.assertEqual(audit['fallback_body_windows'],1)
        report=summarize_all(out)
        self.assertEqual(report['overall']['behavior']['distraction']['tp'],1)
        self.assertEqual(report['by_view']['face']['gaze']['raw_known'],1)
        self.assertEqual(paired_errors(rows,out)['overall']['rescued_positive'],1)

    def test_unknown_unavailable_and_strict_identity(self):
        from training.visual.all_tasks import compose, summarize_all
        a=joint_row();b=joint_row(start=25);c=joint_row(start=50)
        b['task_available'][3]=False;c['label_known'][3]=False;c['targets'][3]=0
        b['gaze_available']=False;c['gaze_label_known']=False
        report=summarize_all([a,b,c])['overall']
        metric=report['behavior']['distraction']
        self.assertEqual((metric['known'],metric['unknown'],metric['available']),(2,1,1))
        self.assertEqual(metric['unavailable_positive'],1)
        self.assertEqual(report['gaze']['raw_known'],2)
        self.assertEqual(report['gaze']['coverage'],.5)
        probes=[probe_row(a),probe_row(b),probe_row(c)]
        for invalid in (probes[:-1],probes+[probes[0]], [dict(probes[0],target=0),*probes[1:]],
                        [dict(probes[0],logit=float('nan')),*probes[1:]]):
            with self.assertRaises(ValueError): compose([a,b,c],invalid,0.)

    def test_null_threshold_rejects_probe_but_keeps_baseline_fallback(self):
        from training.visual.all_tasks import compose,summarize_all
        a=joint_row();b=joint_row(start=25);b['behavior_logits'][3]=1.
        rows,audit=compose([a,b],[probe_row(a),probe_row(b,False)],None)
        self.assertEqual(rows[0]['thresholds'][3],None)
        self.assertEqual(rows[1]['thresholds'][3],0.)
        self.assertEqual(summarize_all(rows)['overall']['behavior']['distraction']['tp'],1)
        self.assertEqual(audit['fallback_body_windows'],1)

    def test_cached_baseline_requires_identical_labels_masks_and_bounded_drift(self):
        from training.visual.all_tasks import _verify_baseline
        row=joint_row();cached=probe_row(row)
        cached.update(baseline_available=True,baseline_logit=row['behavior_logits'][3]+.00005)
        self.assertLess(_verify_baseline([row],[cached])['max_absolute_logit_difference'],.0001)
        for change in (dict(baseline_logit=8.),dict(baseline_available=False),dict(target=0),
                       dict(h_eligible=True),dict(subject_id='different'),dict(baseline_logit=float('nan'))):
            with self.assertRaises(ValueError): _verify_baseline([row],[dict(cached,**change)])
        with self.assertRaises(ValueError): _verify_baseline([row],[])

    def test_unsupported_task_metrics_are_null(self):
        from training.visual.all_tasks import summarize_all
        row=joint_row();row['label_known'][0]=False;row['targets'][0]=0
        row['task_available'][1]=False;row['gaze_available']=False
        summary=summarize_all([row])['overall']
        self.assertIsNone(summary['behavior']['fatigue']['all_positive_f1'])
        self.assertIsNone(summary['behavior']['phone_use']['effective_f1'])
        self.assertIsNone(summary['behavior']['phone_use']['precision'])
        self.assertIsNone(summary['gaze']['accuracy'])
        self.assertIsNone(summary['gaze']['macro_f1'])


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp=tempfile.TemporaryDirectory(dir=ROOT/'runs');cls.root=Path(cls.tmp.name)
        prepared_fixture(cls.root)
        from training.visual.data import PreparedData
        from training.visual.train import train_probe
        from training.visual.suite import DEFAULTS
        from training.visual.report import evaluate,summarize
        prepared=PreparedData(cls.root/'prepared');comparisons=[]
        try:
            for seed in (42,43):
                settings=dict(DEFAULTS,backend='cpu',epochs=1,batch_size=3,seed=seed,hidden_dim=4)
                result=train_probe(prepared.dataset('train'),prepared.dataset('val'),'fusion',settings,
                                   cls.root/f'probe{seed}',prepared.provenance)
                path=result['best_checkpoint'];rows,payload=evaluate(prepared.dataset('val'),path)
                comparisons.append(summarize(rows,arm='C_fusion',seed=seed,checkpoint_kind='best',
                    checkpoint=path,checkpoint_sha256=file_sha256(path),epoch=payload['epoch']))
        finally: prepared.close()
        cls.source=dict(schema_version='driveguard-body-visual-stability-v1',status='complete',split='val',comparisons=comparisons)
        cls.source_path=cls.root/'stability.json'
        cls.source_path.write_text(json.dumps(cls.source),encoding='utf-8')

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_real_synthetic_inference_and_immutable_sources(self):
        from training.visual.all_tasks import run
        paths=[p for p in self.root.rglob('*') if p.is_file()]
        before={p:file_sha256(p) for p in paths}
        report=run(prepared_dir=self.root/'prepared',stability_report=self.source_path,
                   output_dir=self.root/'evaluation',batch_size=3)
        self.assertEqual(report['status'],'complete')
        self.assertEqual(len(report['composites']),3)
        self.assertEqual(report['baseline']['overall']['windows'],9)
        for item in report['composites']:
            self.assertTrue(item['equality_audit']['passed'])
            self.assertEqual(item['overall']['gaze'],report['baseline']['overall']['gaze'])
        self.assertEqual(before,{p:file_sha256(p) for p in paths})
        self.assertTrue((self.root/'evaluation/predictions.jsonl').is_file())
        self.assertEqual(read_json(self.root/'evaluation/report.json'),report)
        with (self.root/'evaluation/metrics.csv').open(encoding='utf-8',newline='') as handle:
            rows=list(csv.DictReader(handle))
        self.assertEqual({r['task'] for r in rows},{'fatigue','phone_use','drinking','distraction','gaze'})
        gaze=next(r for r in rows if r['task']=='gaze' and r['group']=='overall')
        self.assertEqual(int(gaze['raw_known']),9)
        self.assertIn('macro_f1',gaze)
        self.assertEqual(gaze['tp'],'')

    def test_reject_report_tampering_and_test_selection(self):
        from training.visual.all_tasks import run
        for field in ('checkpoint_sha256','window_label_hash','threshold','metric'):
            bad=deepcopy(self.source);entry=bad['comparisons'][1]
            if field=='threshold': entry['validation_fpr03']['threshold_logit']=999.
            elif field=='metric': entry['overall']['tp']+=1
            else: entry[field]='0'*64
            path=self.root/(field+'.json');path.write_text(json.dumps(bad),encoding='utf-8')
            output=self.root/('bad-'+field)
            with self.assertRaises(ValueError):
                run(prepared_dir=self.root/'prepared',stability_report=path,output_dir=output)
            self.assertFalse((output/'report.json').exists())
        with self.assertRaisesRegex(ValueError,'val|test'):
            run(prepared_dir=self.root/'prepared',stability_report=self.source_path,
                output_dir=self.root/'test',split='test')

    def test_reject_prepared_identity_and_cached_prediction_tampering(self):
        from training.visual.all_tasks import run
        path=self.root/'prepared/prepared.json';original=path.read_bytes()
        reference=read_json(path);rows_path=self.root/'prepared'/reference['splits']['val']['rows']
        original_rows=rows_path.read_bytes()
        for field in ('baseline_checkpoint_sha256','data_fingerprint','baseline_logit','context'):
            output=self.root/('bad-prepared-'+field)
            try:
                report=deepcopy(reference)
                if field=='baseline_logit':
                    rows=json.loads(original_rows);rows[0]['baseline_logit']+=1
                    rows_path.write_text(json.dumps(rows),encoding='utf-8')
                    report['splits']['val']['rows_sha256']=file_sha256(rows_path)
                elif field=='context': report['notes'].append('altered preparation provenance')
                else: report[field]='0'*64
                path.write_text(json.dumps(report),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'hash|fingerprint|drift|provenance'):
                    run(prepared_dir=path.parent,stability_report=self.source_path,output_dir=output)
                self.assertFalse((output/'report.json').exists())
            finally:
                rows_path.write_bytes(original_rows);path.write_bytes(original)


if __name__=='__main__': unittest.main()
