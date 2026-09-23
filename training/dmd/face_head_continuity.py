"""Streaming fresh-mask statistics, not trained-model eligibility or accuracy."""
from collections import Counter, deque
from copy import deepcopy

POLICIES = dict(strict=(50,0), coverage25=(25,0), gap80=(50,2))
WINDOW_FIELDS = ['start_slot','end_slot','window_start_s','window_end_s','dist_label_known','dist_target']
for _name in ('cached','variant'):
    WINDOW_FIELDS += [_name+'_valid_ratio',_name+'_gap_steps'] + [_name+'_'+p+'_eligible' for p in POLICIES]


def _bin(length):
    return '1-2' if length<=2 else '3-5' if length<=5 else '6-25' if length<=25 else '>25'


def _windows():
    return {policy:{label:dict(windows=0,cached_eligible=0,variant_eligible=0,gained=0,lost=0)
                    for label in ('all','known_positive','known_negative','unknown')} for policy in POLICIES}


class HeadContinuity:
    def __init__(self):
        self.slots=0
        self.streams={name:dict(history=deque(maxlen=100),missing_slots=0,run=0,
                               longest=0,runs=Counter()) for name in ('cached','variant')}
        self.windows=_windows()

    def update(self,index,cached,variant,known,target):
        if index!=self.slots:
            raise ValueError('Head continuity slots must be consecutive from zero')
        self.slots+=1
        for name,valid in (('cached',cached),('variant',variant)):
            s=self.streams[name]; s['history'].append(bool(valid))
            if not valid:
                s['run']+=1; s['missing_slots']+=1
                s['longest']=max(s['longest'],s['run'])
            elif s['run']:
                s['runs'][_bin(s['run'])]+=1; s['run']=0
        if self.slots<100 or (self.slots-100)%25:
            return None
        row=dict(start_slot=index-99,end_slot=index,window_start_s=(index-99)/25,
                 window_end_s=index/25,dist_label_known=bool(known),dist_target=int(target) if known else '')
        for name,s in self.streams.items():
            count=sum(s['history']); gap=s['run'] if count else None
            row[name+'_valid_ratio']=count/100
            row[name+'_gap_steps']=gap
            for policy,(minimum,max_gap) in POLICIES.items():
                row[name+'_'+policy+'_eligible']=count>=minimum and gap is not None and gap<=max_gap
        label='unknown' if not known else 'known_positive' if target else 'known_negative'
        for policy in POLICIES:
            old,new=row['cached_'+policy+'_eligible'],row['variant_'+policy+'_eligible']
            for group in ('all',label):
                counts=self.windows[policy][group]
                counts['windows']+=1; counts['cached_eligible']+=int(old); counts['variant_eligible']+=int(new)
                counts['gained']+=int(new and not old); counts['lost']+=int(old and not new)
        return row

    def report(self):
        result=dict(slots=self.slots,window_steps=100,stride_steps=25,sample_rate_hz=25,windows=deepcopy(self.windows),
                    interpretation='Fresh raw head-mask window proxies only; no normalizer, classifier or pose calibration. Counts are not recall.')
        for name,s in self.streams.items():
            runs=s['runs'].copy()
            if s['run']: runs[_bin(s['run'])]+=1
            result[name]=dict(missing_slots=s['missing_slots'],missing_runs=sum(runs.values()),
                              longest_missing_slots=s['longest'],longest_missing_ms=s['longest']*40,
                              missing_run_bins=dict(runs))
        return result


def aggregate_continuity(reports):
    result=HeadContinuity().report()
    result['slots']=sum(r['slots'] for r in reports)
    for name in ('cached','variant'):
        rows=[r[name] for r in reports]
        for key in ('missing_slots','missing_runs'): result[name][key]=sum(r[key] for r in rows)
        for key in ('longest_missing_slots','longest_missing_ms'): result[name][key]=max((r[key] for r in rows),default=0)
        bins=Counter()
        for row in rows: bins.update(row['missing_run_bins'])
        result[name]['missing_run_bins']=dict(bins)
    for policy,groups in result['windows'].items():
        for label,counts in groups.items():
            for key in counts: counts[key]=sum(r['windows'][policy][label][key] for r in reports)
    return result
