"""Masked action supervision from immutable annotations; never inference inputs."""
from pathlib import Path
import numpy as np

from training.common.config import read_json, canonical_hash
from training.common.data import file_sha256
from training.dmd.labels import (compile_labels, _root, _streams, _annotations,
                                _active_axes, YAWN_TYPES, MAX_SOURCE_AGE_MS)
from training.dmd.annotations import _contains
from training.visual.extract import _contained, _prepare
from training.visual.head_frames import FrameDataset


def compile_actions(document, view, source_times, closed_eye_min_ms, subset):
    """Independent frame targets; missing/conflicting axes retain mask=False.

    Yawn absence is negative only with explicitly open eyes in covered drowsiness
    annotations, matching the existing fatigue-negative evidence rule. No mere
    absence, conflicting/undefined state, or distraction/gaze clip is a negative.
    """
    original = compile_labels(document, view, source_times,
                              closed_eye_min_ms=closed_eye_min_ms, subset=subset)
    root = _root(document)
    streams = _streams(root)
    offset = streams[view]['shift'] - min(v['shift'] for v in streams.values())
    timeline, annotations = _annotations(root)
    targets = np.zeros((len(source_times),2),np.float32)
    masks = np.zeros_like(targets,dtype=np.bool_)
    subtypes = np.full(len(source_times),'unknown',dtype='<U12')
    closed_since = None
    for i,t in enumerate(source_times):
        if i and t-source_times[i-1] > MAX_SOURCE_AGE_MS: closed_since = None
        frame = i+offset
        if subset!='drowsiness' or not _contains(timeline,frame):
            closed_since = None
            continue
        axes = _active_axes(annotations,frame)
        eyes,yawns = axes['eyes_state'],axes['yawning']
        if eyes in ({'open'},{'close'}):
            targets[i,0],masks[i,0] = float(eyes=={'close'}),True
        yawn = bool(yawns) and yawns <= YAWN_TYPES
        if yawn:
            targets[i,1],masks[i,1] = 1.,True
        elif eyes=={'open'} and not yawns:
            masks[i,1] = True
        if eyes=={'close'}:
            if closed_since is None: closed_since = t
        else: closed_since = None
        closed = closed_since is not None and t-closed_since>=closed_eye_min_ms
        if yawn and closed: subtypes[i]='overlap'
        elif yawn: subtypes[i]='yawn'
        elif closed: subtypes[i]='closed_eye'
        elif eyes=={'open'} and not yawns: subtypes[i]='negative'
    indices = np.asarray(original['source_indices'])
    fresh = np.asarray(original['slots_ms'])-np.asarray(source_times)[indices]<=MAX_SOURCE_AGE_MS
    action_targets,action_masks = targets[indices].copy(),masks[indices].copy()
    action_masks &= fresh[:,None]
    action_targets[~action_masks] = 0
    types = subtypes[indices].copy(); types[~fresh]='unknown'
    return dict(targets=action_targets,masks=action_masks,subtypes=types,
                fatigue_targets=np.asarray(original['targets'],np.float32)[:,0],
                fatigue_masks=np.asarray(original['label_masks'],np.bool_)[:,0])


def pixel_resets(source_times, sampled_times):
    """Only timestamp evidence: true if an original >80ms gap was crossed."""
    source_times=np.asarray(source_times)
    indices=np.searchsorted(source_times,sampled_times)
    if np.any(indices>=len(source_times)) or not np.array_equal(source_times[indices],sampled_times):
        raise ValueError('Action pixels do not match original source PTS')
    gaps=np.r_[0,np.cumsum(np.diff(source_times)>MAX_SOURCE_AGE_MS)]
    return np.r_[True,np.diff(gaps[indices])>0]


class ActionLabels:
    def __init__(self,prepared,cache,dataset_root):
        self.records={}; identities=[]; thresholds=set()
        manifest=read_json(prepared.report['manifest'])
        records={r['record_id']:r for r in manifest['records']}
        dataset_root=Path(dataset_root).resolve(strict=True)
        for rid,pixels in cache.records.items():
            print(f'Compiling eye/yawn labels: {rid}',flush=True)
            item=_prepare(records[rid],Path(prepared.report['data_root']),dataset_root)
            complete=item['complete']; job=complete['job']
            inputs=read_json(item['attempt']/'input.json')
            annotation=_contained(dataset_root,job['annotation'])
            digest=file_sha256(annotation)
            if digest!=job['annotation_sha256'] or digest!=inputs['job']['annotation_sha256']:
                raise ValueError('Action annotation checksum mismatch')
            for source in job['annotation_sources']:
                if file_sha256(_contained(dataset_root,source['path']))!=source['sha256']:
                    raise ValueError('Action auxiliary annotation checksum mismatch')
            doc=read_json(annotation)
            if file_sha256(annotation)!=digest: raise ValueError('Action annotation changed during read')
            threshold=inputs['label_policy']['closed_eye_min_ms']; thresholds.add(threshold)
            source_times=np.load(_contained(item['attempt'],complete['arrays']['source_pts_ms']),allow_pickle=False)
            result=compile_actions(doc,job['view'],source_times.tolist(),threshold,job['subset'])
            for key,col in (('targets','fatigue_targets'),('label_masks','fatigue_masks')):
                old=np.load(_contained(item['attempt'],complete['arrays'][key]),allow_pickle=False)
                if not np.array_equal(old[:,0],result[col]):
                    raise ValueError('Recompiled fatigue labels differ from original cache')
            arrays=cache.arrays(rid)
            slots=arrays['slot_indices']
            labels=dict(targets=result['targets'][slots],masks=result['masks'][slots],
                        subtypes=result['subtypes'],reset=pixel_resets(source_times,arrays['source_pts_ms']))
            for row in prepared.rows[job['split']]:
                if row['record_id']!=rid: continue
                end=row['end_slot']
                if (row['label_known']!=bool(result['fatigue_masks'][end]) or
                        row['target']!=int(result['fatigue_targets'][end])):
                    raise ValueError('Action window-end labels differ from preparation')
            self.records[rid]=labels
            identities.append(dict(record_id=rid,annotation=str(annotation),annotation_sha256=digest,
                                   cache_fingerprint=complete['fingerprint'],closed_eye_min_ms=threshold))
        if len(thresholds)!=1: raise ValueError('Expected one original closed-eye duration policy')
        self.closed_eye_min_ms=thresholds.pop()
        self.provenance=dict(schema_version='driveguard-action-labels-v1',records=identities,
             policy='explicit-eye; yawn-positive; yawn-negative-only-open-eye; unknown-masked')
        self.digest=canonical_hash(self.provenance)

    def coverage(self,cache):
        rows=[]
        for rid,labels in self.records.items():
            valid=cache.arrays(rid)['valid']
            row=dict(record_id=rid,split=cache.records[rid]['split'],subject_id=cache.records[rid]['subject_id'])
            for i,name in enumerate(('eye','yawn')):
                mask=labels['masks'][:,i]&valid
                row[name]=dict(positive=int((mask&(labels['targets'][:,i]==1)).sum()),
                               negative=int((mask&(labels['targets'][:,i]==0)).sum()),
                               unknown_or_invalid=int((~mask).sum()))
            rows.append(row)
        return rows


class ActionDataset(FrameDataset):
    def __init__(self,prepared,cache,split,labels):
        super().__init__(prepared,cache,split,32)
        self.labels=labels

    def supervision(self,index):
        row=self.metadata(index); start=row['start_slot']//25*8
        labels=self.labels.records[row['record_id']]
        return labels['targets'][start:start+32].copy(),labels['masks'][start:start+32].copy()

    def __getitem__(self,index):
        inputs,target,known=super().__getitem__(index)
        actions,masks=self.supervision(index)
        inputs.update(action_targets=actions,action_masks=masks)
        return inputs,target,known
