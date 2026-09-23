"""Compare matching diagnostic windows using only the Python standard library."""
import csv
import math
from pathlib import Path

from training.common.config import canonical_hash, read_json


def _bool(value):
    if value not in ('True','False'): raise ValueError('Invalid diagnostic boolean')
    return value == 'True'


class Counts:
    def __init__(self):
        self.counts = dict(windows=0,positive=0,negative=0,unknown=0,tp=0,fp=0,tn=0,fn=0,
                           unavailable_positive=0,unavailable_negative=0)
        self.sums = {}; self.samples = {}

    def add(self,row,threshold):
        c=self.counts; c['windows']+=1
        for key,value in row.items():
            if value != '' and (key.endswith('_dist_weight') or key.endswith('_valid_ratio') or key == 'probability'):
                number=float(value)
                if not math.isfinite(number): raise ValueError('Nonfinite diagnostic statistic')
                self.sums[key]=self.sums.get(key,0.)+number
                self.samples[key]=self.samples.get(key,0)+1
        if not _bool(row['label_known']): c['unknown']+=1; return
        if row['target'] not in ('0','1'): raise ValueError('Invalid known target')
        target=int(row['target']); category='positive' if target else 'negative'
        c[category]+=1
        if not _bool(row['behavior_available']): c['unavailable_'+category]+=1; return
        logit=float(row['logit'])
        if not math.isfinite(logit): raise ValueError('Invalid prediction logit')
        prediction=logit >= math.log(threshold/(1-threshold))
        c[('t' if prediction == bool(target) else 'f')+('p' if prediction else 'n')]+=1

    def report(self):
        c=dict(self.counts); tp,fp,tn,fn=(c[name] for name in ('tp','fp','tn','fn'))
        ratio=lambda n,d: n/d if d else None
        c.update(precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),f1=ratio(2*tp,2*tp+fp+fn),
                 false_positive_rate=ratio(fp,fp+tn),all_positive_detection_rate=ratio(tp,c['positive']),
                 known_input_coverage=ratio(tp+fp+tn+fn,c['positive']+c['negative']),
                 means={key:self.sums[key]/self.samples[key] for key in self.sums})
        return c


def read_report(directory):
    directory=Path(directory); summary=read_json(directory/'summary.json')
    if summary['status'] != 'complete': raise ValueError('Diagnostic report incomplete')
    threshold=summary['threshold']
    if not 0 < threshold < 1: raise ValueError('Invalid diagnostic threshold')
    cohorts={name:Counts() for name in ('body','face','body_h_eligible','body_h_ineligible')}
    strata={}; identities={}
    with (directory/'windows.csv').open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            key=(row['record_id'],int(row['start_slot']),int(row['end_slot']))
            if key in identities: raise ValueError('Duplicate diagnostic window')
            identities[key]=(row['view'],row['subset'],row['label_known'],row['target'])
            view=row['view']
            if view in ('body','face'): cohorts[view].add(row,threshold)
            head=_bool(row['h_eligible'])
            if view == 'body': cohorts['body_h_eligible' if head else 'body_h_ineligible'].add(row,threshold)
            pattern=''.join(str(int(_bool(row[name+'_eligible']))) for name in ('f','o','ho','fo'))
            stratum=f'{view}:H{int(head)}:F_O_HO_FO={pattern}'
            strata.setdefault(stratum,Counts()).add(row,threshold)
    if not identities: raise ValueError('Empty diagnostic windows')
    identity=canonical_hash([(list(key),list(identities[key])) for key in sorted(identities)])
    return summary,identity,{name:value.report() for name,value in cohorts.items()}, {name:value.report() for name,value in strata.items()}


def compare_reports(entries):
    comparisons=[]; expected=None
    for entry in entries:
        summary,identity,cohorts,strata=read_report(entry['directory'])
        provenance=(summary['data_fingerprint'],summary['selection'],summary['threshold'],identity)
        if expected is None: expected=provenance
        if provenance[0] != expected[0]: raise ValueError('Diagnostic data fingerprint mismatch')
        if provenance[1:3] != expected[1:3]: raise ValueError('Diagnostic selection/threshold mismatch')
        if provenance[3] != expected[3]: raise ValueError('Diagnostic window/label identity mismatch')
        row=dict(variant=entry['variant'],checkpoint=entry['checkpoint'],epoch=summary['checkpoint_epoch'],
                 diagnostic_directory=str(entry['directory']),**cohorts,strata=strata)
        if 'train_dir' in entry:
            history=read_json(Path(entry['train_dir'])/'history.json')
            epoch=next(item for item in history if item['epoch'] == summary['checkpoint_epoch'])
            row.update(val_loss=epoch['val']['loss'],gaze=epoch['val']['gaze_metrics'],
                       behavior_tasks=epoch['val']['metrics']['per_task'])
            row['input_audit']=str(Path(entry['train_dir'])/'body_dist_audit.json')
        comparisons.append(row)
    return dict(schema_version='driveguard-body-dist-comparison-v1',comparisons=comparisons,
                data_fingerprint=expected[0] if expected else None,window_label_hash=expected[3] if expected else None,
                threshold=expected[2] if expected else .5,
                notes=['Fixed-threshold development results, not held-out generalization or event metrics.',
                       'Recall excludes unavailable positives; all_positive_detection_rate includes them.',
                       'H eligibility can change with head policy; H strata are not necessarily identical cohorts across variants.',
                       'Mean gate weights/feature coverage are descriptive, not causal importance.',
                       'No automatic winner or deployment. Compare body gains against false positives and face/gaze regressions.'])


def write_csv(path,report):
    fields=['variant','checkpoint','epoch','val_loss','gaze_accuracy','gaze_macro_f1',
            'fatigue_f1','phone_use_f1','drinking_f1']
    metrics=('positive','negative','tp','fp','tn','fn','unavailable_positive','precision','recall','f1',
             'false_positive_rate','all_positive_detection_rate','known_input_coverage')
    for cohort in ('body','face','body_h_eligible','body_h_ineligible'):
        fields.extend(cohort+'_'+metric for metric in metrics)
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for entry in report['comparisons']:
            row={name:entry.get(name) for name in fields[:4]}
            for key in ('accuracy','macro_f1'): row['gaze_'+key]=entry.get('gaze',{}).get(key)
            for task in ('fatigue','phone_use','drinking'):
                row[task+'_f1']=entry.get('behavior_tasks',{}).get(task,{}).get('f1')
            for cohort in ('body','face','body_h_eligible','body_h_ineligible'):
                row.update({cohort+'_'+metric:entry[cohort][metric] for metric in metrics})
            writer.writerow(row)
