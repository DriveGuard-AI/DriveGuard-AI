"""Verified visual caches and bounded, causal window loading."""
from collections import OrderedDict
from pathlib import Path, PurePosixPath
import math
import re

import numpy as np

from training.common.config import read_json,canonical_hash
from training.common.data import file_sha256

OFFSETS=np.array([12,24,37,49,62,74,87,99],np.int64)
METADATA=('record_id','recording_id','subject_id','view','subset','split')


def contained(root,relative):
    if not isinstance(relative,str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Invalid relative cache path')
    part=PurePosixPath(relative)
    if part.is_absolute() or '..' in part.parts: raise ValueError('Cache path escapes root')
    path=(root/relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file(): raise ValueError('Cache path escapes root')
    return path


def close_arrays(arrays):
    for value in arrays.values():
        if getattr(value,'_mmap',None) is not None: value._mmap.close()


def _sha256(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _relative(value):
    return (isinstance(value,str) and bool(value) and '\\' not in value and ':' not in value
            and not PurePosixPath(value).is_absolute() and '..' not in PurePosixPath(value).parts
            and PurePosixPath(value).as_posix()==value and value!='.')


def _validate_source(source):
    expected={'video','sha256','size_bytes','cache_fingerprint','driver_roi','image_size'}
    if not isinstance(source,dict) or set(source)!=expected:
        raise ValueError('Invalid visual source provenance')
    if (not _relative(source['video']) or not _sha256(source['sha256'])
            or not _sha256(source['cache_fingerprint'])
            or type(source['size_bytes'])!=int or source['size_bytes']<1):
        raise ValueError('Invalid visual source identity')
    size,roi=source['image_size'],source['driver_roi']
    if (not isinstance(size,list) or len(size)!=2 or any(type(v)!=int or v<1 for v in size)
            or not isinstance(roi,list) or len(roi)!=4
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in roi)
            or not (0<=roi[0]<roi[2]<=size[0] and 0<=roi[1]<roi[3]<=size[1])):
        raise ValueError('Invalid visual source image/ROI provenance')


def _validate_run(root,report):
    from training.visual.encoder import encoder_metadata
    from training.visual.extract import _validate_options
    try:
        run=read_json(contained(root,'input.json'))
    except (OSError,ValueError) as error:
        raise ValueError('Missing or invalid visual run identity') from error
    required={'schema_version','manifest_sha256','encoder','backend','device_index','batch_size'}
    cropped=report.get('schema_version')=='driveguard-visual-cache-v2'
    if cropped:required.add('crop')
    if (not isinstance(run,dict) or set(run)!=required
            or run['schema_version']!=('driveguard-visual-extract-v2' if cropped else 'driveguard-visual-extract-v1')
            or not _sha256(run['manifest_sha256'])
            or run['manifest_sha256']!=report.get('manifest_sha256')):
        raise ValueError('Visual run/manifest identity mismatch')
    if cropped:
        from training.visual.spatial import validate_crop
        validate_crop(run['crop'])
        if report.get('crop')!=run['crop']:raise ValueError('Visual crop identity mismatch')
    elif 'crop' in report:raise ValueError('Legacy visual cache cannot contain spatial crop metadata')
    encoder=run['encoder']
    if not isinstance(encoder,dict) or not _sha256(encoder.get('weights_sha256')):
        raise ValueError('Invalid visual encoder provenance')
    if encoder!=encoder_metadata(encoder['weights_sha256']) or encoder!=report.get('encoder'):
        raise ValueError('Visual encoder metadata mismatch')
    _validate_options(encoder['weights_sha256'],run['backend'],run['device_index'],run['batch_size'])
    return run


class VisualCache:
    def __init__(self,root):
        self.root=Path(root).resolve(strict=True); self.report=read_json(self.root/'manifest.json')
        self.manifest_sha256=file_sha256(self.root/'manifest.json')
        self._opened=OrderedDict();self.records={}
        if self.report.get('schema_version') not in ('driveguard-visual-cache-v1','driveguard-visual-cache-v2') or self.report.get('status') != 'complete':
            raise ValueError('Visual cache incomplete or unsupported')
        try:
            run=_validate_run(self.root,self.report)
            for entry in self.report['records']:
                expected_keys=set(METADATA)|{'slots','source','fingerprint','status','sample_count','directory','arrays','sha256'}
                if (not isinstance(entry,dict) or set(entry)!=expected_keys
                        or any(not isinstance(entry[k],str) or not entry[k].strip() for k in METADATA)):
                    raise ValueError('Invalid visual record source/identity metadata')
                rid=entry['record_id']
                if rid in self.records or entry['view'] != 'body' or entry['split'] not in ('train','val','test'):
                    raise ValueError('Duplicate or invalid visual record')
                if type(entry['slots']) is not int or entry['slots']<100: raise ValueError('Invalid visual slot count')
                _validate_source(entry['source'])
                identity={key:entry[key] for key in (*METADATA,'slots','source')}
                if entry['fingerprint']!=canonical_hash(dict(run=run,record=identity)):
                    raise ValueError('Visual record fingerprint mismatch')
                if entry['status']!='complete' or not _relative(entry['directory']):
                    raise ValueError('Invalid visual completion marker/path')
                try:
                    marker=contained(self.root,entry['directory']+'/record.json')
                    if read_json(marker)!=entry or (marker.parent/'failure.json').exists():
                        raise ValueError('Invalid visual completion marker')
                except (OSError,ValueError) as error:
                    raise ValueError('Missing or invalid visual completion marker') from error
                if set(entry['arrays']) != {'embeddings','slot_indices','source_pts_ms','valid'} or set(entry['sha256']) != set(entry['arrays']):
                    raise ValueError('Invalid visual array layout')
                for key,relative in entry['arrays'].items():
                    if relative!=entry['directory']+'/'+key+'.npy': raise ValueError('Unexpected visual array path')
                    if file_sha256(contained(self.root,relative)) != entry['sha256'][key]:
                        raise ValueError('Visual array checksum mismatch')
                self.records[rid]=entry
                arrays=self.arrays(rid);count=entry['sample_count']
                expected=np.array([i for i in range(entry['slots']) if i%25 in (12,24)],np.int64)
                for key,dtype,shape in (('embeddings',np.float32,(count,512)),('slot_indices',np.int64,(count,)),
                                        ('source_pts_ms',np.int64,(count,)),('valid',np.bool_,(count,))):
                    if arrays[key].dtype != dtype or arrays[key].shape != shape: raise ValueError('Visual array dtype/shape mismatch')
                if not np.array_equal(arrays['slot_indices'],expected): raise ValueError('Visual causal slot grid mismatch')
                age=expected*40-arrays['source_pts_ms']
                if (age<0).any() or (arrays['source_pts_ms']<0).any() or (np.diff(arrays['source_pts_ms'])<0).any():
                    raise ValueError('Visual source PTS uses future or unordered frames')
                if not np.array_equal(arrays['valid'],age<=80): raise ValueError('Visual freshness mask mismatch')
                for start in range(0,count,2048):
                    values=arrays['embeddings'][start:start+2048]; valid=arrays['valid'][start:start+2048]
                    if not np.isfinite(values).all() or np.any(values[~valid] != 0): raise ValueError('Invalid visual embeddings')
            if not self.records: raise ValueError('Empty visual cache')
        except Exception:
            self.close();raise

    def arrays(self,rid):
        if rid in self._opened: self._opened.move_to_end(rid);return self._opened[rid]
        if len(self._opened)>=4: close_arrays(self._opened.popitem(last=False)[1])
        arrays={}
        try:
            for key,relative in self.records[rid]['arrays'].items():
                value=np.load(contained(self.root,relative),mmap_mode='r',allow_pickle=False)
                if not isinstance(value,np.ndarray): value.close();raise ValueError('Expected NPY array')
                arrays[key]=value
        except Exception: close_arrays(arrays);raise
        self._opened[rid]=arrays;return arrays

    def close(self):
        for arrays in self._opened.values(): close_arrays(arrays)
        self._opened.clear()


def scale(values,state):
    mean=np.asarray(state['mean'],np.float64);std=np.asarray(state['std'],np.float64)
    if mean.shape != (values.shape[-1],) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
        raise ValueError('Invalid visual normalization')
    return np.clip((values-mean)/std,-5,5).astype(np.float32)


class PreparedData:
    def __init__(self,root):
        self.root=Path(root).resolve(strict=True);self.report=read_json(self.root/'prepared.json')
        schema=self.report.get('schema_version')
        if schema not in ('driveguard-visual-prepared-v1','driveguard-visual-prepared-v2') or self.report.get('status') != 'complete':
            raise ValueError('Prepared visual data incomplete')
        self.task=self.report.get('task','distraction')
        if (schema=='driveguard-visual-prepared-v1' and self.task!='distraction') or (schema=='driveguard-visual-prepared-v2' and self.task!='fatigue'):
            raise ValueError('Invalid prepared task/schema identity')
        self.cache=VisualCache(self.report['visual_cache']);self.contexts={};self.rows={}
        self.normalizer=self.report['normalization']
        try:
            if self.cache.manifest_sha256 != self.report['visual_cache_sha256']:
                raise ValueError('Prepared visual-cache identity mismatch')
            for split,entry in self.report['splits'].items():
                for key in ('context','rows'):
                    if file_sha256(contained(self.root,entry[key])) != entry[key+'_sha256']: raise ValueError('Prepared checksum mismatch')
                context=np.load(contained(self.root,entry['context']),mmap_mode='r',allow_pickle=False)
                self.contexts[split]=context
                rows=read_json(contained(self.root,entry['rows']));self.rows[split]=rows
                if not isinstance(context,np.ndarray): context.close();raise ValueError('Expected prepared NPY context')
                if context.dtype!=np.float32 or context.shape!=(len(rows),self.report['context_dim']):
                    raise ValueError('Invalid prepared context')
                for start in range(0,len(context),2048):
                    if not np.isfinite(context[start:start+2048]).all(): raise ValueError('Invalid prepared context')
                seen=set()
                for row in rows:
                    if self.task=='fatigue' and type(row.get('f_eligible'))!=bool:raise ValueError('Missing baseline F eligibility')
                    record=self.cache.records[row['record_id']]
                    if any(row[k]!=record[k] for k in METADATA) or row['split'] != split: raise ValueError('Prepared split/metadata mismatch')
                    start=row['start_slot'];key=(row['record_id'],start)
                    if key in seen or type(start)!=int or start<0 or start%25 or row['end_slot']!=start+99 or start+99>=record['slots']:
                        raise ValueError('Prepared window identity mismatch')
                    seen.add(key)
                    if row['visual_start'] != start//25*2: raise ValueError('Prepared visual offset mismatch')
                    if type(row['label_known'])!=bool or row['target'] not in (0,1) or (not row['label_known'] and row['target']!=0):
                        raise ValueError('Invalid prepared label')
            identity=dict(self.report)
            self.provenance=dict(prepared_sha256=file_sha256(self.root/'prepared.json'),
                                 preparation=identity,normalization=self.normalizer)
        except Exception: self.close();raise

    def dataset(self,split):
        if split not in self.rows: raise ValueError(f'No prepared {split} split')
        return WindowDataset(self,split)

    def close(self):
        close_arrays(self.contexts);self.contexts.clear();self.cache.close()


class WindowDataset:
    visual_dim=512
    def __init__(self,prepared,split):
        self.prepared=prepared;self.split=split
        self.context_dim=prepared.report['context_dim']
    def __len__(self): return len(self.prepared.rows[self.split])
    def metadata(self,index): return dict(self.prepared.rows[self.split][index])
    def __getitem__(self,index):
        row=self.prepared.rows[self.split][index]
        arrays=self.prepared.cache.arrays(row['record_id']);start=row['visual_start']
        if not np.array_equal(arrays['slot_indices'][start:start+8],row['start_slot']+OFFSETS):
            raise ValueError('Visual window misaligned')
        mask=arrays['valid'][start:start+8].copy()
        visual=scale(arrays['embeddings'][start:start+8],self.prepared.normalizer['visual'])
        visual[~mask]=0
        context=scale(self.prepared.contexts[self.split][index],self.prepared.normalizer['context'])
        return dict(visual=visual,context=context,visual_mask=mask),np.float32(row['target']),bool(row['label_known'])
