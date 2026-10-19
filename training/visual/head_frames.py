"""Immutable, bounded uint8 RGB cache for warm-start head adaptation.

Dense frames retain the original 25 Hz clock. The eight legacy samples are an
exact subset; no windows, labels, source checkpoints or source caches are edited.
"""
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import numpy as np

from inference.experts.assets import file_sha256
from training.common.config import canonical_hash, read_json
from training.dmd.cache import _unlinked, validate_complete
from training.visual.data import METADATA, OFFSETS, PreparedData, VisualCache, contained
from training.visual.extract import _prepare, _identity, _stamp, write_json
from training.visual.spatial import crop_roi, validate_crop

DENSE_OFFSETS = (2,5,8,12,15,18,21,24)
SCHEMA = 'driveguard-head-pixels-v1'


def selected_slots(slots):
    if type(slots) is not int or slots < 1:
        raise ValueError('Invalid original slot count')
    return np.flatnonzero(np.isin(np.arange(slots)%25,DENSE_OFFSETS)).astype(np.int64)


def _inputs(prepared, crop, dataset_root):
    validate_crop(crop)
    if prepared.task != 'fatigue':
        raise ValueError('Head adaptation requires fatigue preparation')
    prepared_path = prepared.root/'prepared.json'
    if (file_sha256(prepared_path) != prepared.provenance['prepared_sha256']
            or read_json(prepared_path) != prepared.report):
        raise ValueError('Original preparation changed')
    for entry in prepared.report['splits'].values():
        for key in ('context','rows'):
            if file_sha256(contained(prepared.root,entry[key])) != entry[key+'_sha256']:
                raise ValueError('Original preparation arrays changed')
    manifest = _unlinked(prepared.report['manifest'])
    digest = file_sha256(manifest)
    if (digest != prepared.cache.report['manifest_sha256']
            or file_sha256(prepared.cache.root/'manifest.json') != prepared.report['visual_cache_sha256']):
        raise ValueError('Original manifest or visual cache changed')
    source_manifest = read_json(manifest)
    if source_manifest.get('schema_version') != 'driveguard-feature-manifest-v3':
        raise ValueError('Original immutable v3 manifest required')
    records = [r for r in source_manifest['records'] if r['view']=='body' and r['split'] in ('train','val')]
    if not records or len({r['record_id'] for r in records}) != len(records):
        raise ValueError('No train/val body records or duplicate record identities')
    root,dataset = _unlinked(prepared.report['data_root']),_unlinked(dataset_root)
    run = dict(schema_version=SCHEMA,prepared_root=str(prepared.root),
               prepared_sha256=prepared.provenance['prepared_sha256'],manifest=str(manifest),
               manifest_sha256=digest,data_root=str(root),dataset_root=str(dataset),crop=crop,
               preprocessing=dict(version='driver-roi-letterbox-v1',shape=[3,224,224],dtype='uint8',color='RGB'),
               sampling=dict(hz=25,offsets=list(DENSE_OFFSETS),max_age_ms=80))
    items = [_prepare(record,root,dataset) for record in records]
    for item in items:
        original = prepared.cache.records.get(item['record']['record_id'])
        if (original is None or original['source'] != item['source']
                or original['slots'] != item['complete']['feature_record']['slots']
                or any(original[k] != item['record'][k] for k in METADATA)):
            raise ValueError('Prepared visual source identity differs from original cache')
    return run,items


def _validate_record(output, row, identity, item):
    expected_keys = set(identity)|{'status','sample_count','directory','arrays','sha256'}
    if (not isinstance(row,dict) or set(row) != expected_keys
            or any(row.get(k) != v for k,v in identity.items()) or row['status'] != 'complete'):
        raise ValueError('Pixel record configuration/provenance mismatch')
    marker = contained(output,row['directory']+'/record.json')
    if read_json(marker) != row or (marker.parent/'failure.json').exists():
        raise ValueError('Invalid pixel completion marker')
    indices = selected_slots(identity['slots']); count = len(indices)
    specs = dict(images=((count,3,224,224),np.uint8),slot_indices=((count,),np.int64),
                 source_pts_ms=((count,),np.int64),valid=((count,),np.bool_))
    if type(row['sample_count']) is not int or row['sample_count'] != count or set(row['arrays']) != set(specs) or set(row['sha256']) != set(specs):
        raise ValueError('Invalid pixel array layout')
    with ExitStack() as stack:
        arrays = {}
        for key,(shape,dtype) in specs.items():
            if row['arrays'][key] != row['directory']+'/'+key+'.npy':
                raise ValueError('Unexpected pixel array path')
            path = contained(output,row['arrays'][key])
            if file_sha256(path) != row['sha256'][key]:
                raise ValueError('Pixel array checksum mismatch')
            array = np.load(path,mmap_mode='r',allow_pickle=False)
            stack.callback(array._mmap.close)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError('Pixel array dimensions or dtype mismatch')
            arrays[key] = array
        original = np.load(item['attempt']/item['complete']['arrays']['source_pts_ms'],mmap_mode='r',allow_pickle=False)
        stack.callback(original._mmap.close)
        for start in range(0,count,16):
            selected = indices[start:start+16]
            locations = np.searchsorted(original,selected*40,side='right')-1
            if np.any(locations < 0):raise ValueError('Original PTS missing causal frame')
            pts = original[locations]; valid = selected*40-pts <= 80
            if (not np.array_equal(arrays['slot_indices'][start:start+16],selected)
                    or not np.array_equal(arrays['source_pts_ms'][start:start+16],pts)
                    or not np.array_equal(arrays['valid'][start:start+16],valid)):
                raise ValueError('Pixel sampling or PTS differs from original cache')
            if np.any(arrays['images'][start:start+16][~valid] != 0):
                raise ValueError('Invalid stale pixel values')
    return row


def _extract_record(output, directory, item, identity, crop):
    from inference.perception.video_source import decode_video, sample_frames
    from training.visual.encoder import preprocess_pixels
    indices = selected_slots(identity['slots']); count = len(indices)
    roi = crop_roi(item['source']['driver_roi'],crop)
    arrays = {}
    try:
        with ExitStack() as stack:
            for key,shape,dtype in (('images',(count,3,224,224),np.uint8),('slot_indices',(count,),np.int64),
                                    ('source_pts_ms',(count,),np.int64),('valid',(count,),np.bool_)):
                array = np.lib.format.open_memmap(directory/(key+'.npy'),mode='w+',dtype=dtype,shape=shape)
                stack.callback(array._mmap.close); arrays[key] = array
            arrays['slot_indices'][:] = indices
            original = np.load(item['attempt']/item['complete']['arrays']['source_pts_ms'],mmap_mode='r',allow_pickle=False)
            stack.callback(original._mmap.close)
            if _stamp(item['source_path']) != item['source_stamp']:
                raise ValueError('Source video changed after verification')
            frames = decode_video(item['source_path']); stack.callback(frames.close)
            def checked_frames():
                decoded = 0
                for frame in frames:
                    if decoded >= len(original) or frame.timestamp_ms != int(original[decoded]):
                        raise ValueError('Decoded source PTS differs from original cache')
                    if list(reversed(frame.bgr.shape[:2])) != item['source']['image_size']:
                        raise ValueError('Decoded image dimensions differ from original cache')
                    decoded += 1
                    yield frame
                if decoded != len(original):raise ValueError('Decoded source frame count differs from original cache')
            checked = checked_frames(); stack.callback(checked.close)
            samples = sample_frames(checked); stack.callback(samples.close)
            total = selected = 0
            for slot,frame in samples:
                if total >= identity['slots'] or slot != total*40:
                    raise ValueError('Decoded slot count/grid differs from original cache')
                location = int(np.searchsorted(original,slot,side='right'))-1
                if location < 0 or frame.timestamp_ms != int(original[location]) or frame.timestamp_ms > slot:
                    raise ValueError('Slot PTS differs from original causal cache')
                if total%25 in DENSE_OFFSETS:
                    fresh = slot-frame.timestamp_ms <= 80
                    arrays['source_pts_ms'][selected] = frame.timestamp_ms
                    arrays['valid'][selected] = fresh
                    arrays['images'][selected] = preprocess_pixels(frame.bgr,roi) if fresh else 0
                    selected += 1
                total += 1
            if total != identity['slots'] or selected != count:
                raise ValueError('Decoded slot count differs from original cache')
            if _stamp(item['source_path']) != item['source_stamp']:
                raise ValueError('Source video changed during extraction')
            if validate_complete(item['attempt'],item['complete']['fingerprint']) != item['complete']:
                raise ValueError('Original cache changed during extraction')
            for array in arrays.values():array.flush()
        relative = directory.relative_to(output).as_posix()
        paths = {key:relative+'/'+key+'.npy' for key in arrays}
        row = dict(identity,status='complete',sample_count=count,directory=relative,arrays=paths,
                   sha256={key:file_sha256(output/path) for key,path in paths.items()})
        write_json(directory/'record.json',row)
        return row
    except BaseException as error:
        write_json(directory/'failure.json',dict(status='failed',error=str(error),error_type=type(error).__name__))
        raise


def _report(run, records):
    return dict(schema_version=SCHEMA,status='complete',run=run,crop=run['crop'],
                prepared_sha256=run['prepared_sha256'],records=records)


def build_frame_cache(prepared: PreparedData, crop, dataset_root, output_dir, continue_run=False):
    """Verify and cache only train/validation records, resuming complete records."""
    output = Path(output_dir).absolute()
    roots = (prepared.root,prepared.cache.root,Path(prepared.report['data_root']).resolve(),Path(dataset_root).resolve())
    if any(output.resolve().is_relative_to(root) or root.is_relative_to(output.resolve()) for root in roots):
        raise ValueError('Output must be separate from preparation, original cache and dataset')
    if output.exists() and not continue_run:
        raise FileExistsError('Pixel output exists; explicit continue_run required')
    run,items = _inputs(prepared,crop,dataset_root)
    if output.exists():
        output = _unlinked(output)
        if read_json(contained(output,'input.json')) != run:
            raise ValueError('Pixel resume configuration/provenance mismatch')
        if (output/'manifest.json').exists():
            report = read_json(contained(output,'manifest.json'))
            if not isinstance(report,dict) or report != _report(run,report.get('records',[])) or len(report['records']) != len(items):
                raise ValueError('Invalid completed pixel manifest')
            for item,row in zip(items,report['records']):
                _validate_record(output,row,_identity(item,run),item)
            return report
    if not output.exists():
        output.mkdir(parents=True); output = _unlinked(output)
        write_json(output/'input.json',run)
    estimated = sum(len(selected_slots(item['complete']['feature_record']['slots']))*(3*224*224+17)+4096 for item in items)
    print(f'Head RGB cache estimated total: {estimated/1024**3:.3f} GiB (uint8, 8 frames/s)',file=sys.stderr,flush=True)
    completed = []
    for item in items:
        identity = _identity(item,run)
        record_dir = output/('record-'+canonical_hash(identity['record_id']))
        record_dir.mkdir(exist_ok=True); _unlinked(record_dir)
        reusable = None
        for marker in sorted(record_dir.glob('attempt-*/record.json')):
            try:
                reusable = _validate_record(output,read_json(_unlinked(marker)),identity,item)
                break
            except (OSError,ValueError,KeyError,TypeError,EOFError):continue
        if reusable is None:
            needed = len(selected_slots(identity['slots']))*(3*224*224+17)+4096
            if shutil.disk_usage(output).free < needed+16*1024**2:
                raise OSError(f'Insufficient disk space for pixel record: need {needed} bytes plus 16 MiB reserve')
            print('Caching head RGB: '+identity['record_id'],file=sys.stderr,flush=True)
            directory = record_dir/('attempt-'+uuid4().hex); directory.mkdir()
            reusable = _extract_record(output,directory,item,identity,crop)
        completed.append(reusable)
    if file_sha256(run['manifest']) != run['manifest_sha256'] or file_sha256(prepared.root/'prepared.json') != run['prepared_sha256']:
        raise ValueError('Original manifest/preparation changed during pixel extraction')
    report = _report(run,completed)
    write_json(output/'manifest.json',report)
    return report


class FrameCache(VisualCache):
    """Checksum/source verification with at most four lazily opened mmap records."""
    def __init__(self,root):
        self.root = _unlinked(root); self._opened = OrderedDict(); self.records = {}
        self.report = read_json(contained(self.root,'manifest.json'))
        self.manifest_sha256 = file_sha256(self.root/'manifest.json')
        run = read_json(contained(self.root,'input.json'))
        if (not isinstance(run,dict) or run.get('schema_version') != SCHEMA
                or not isinstance(self.report,dict) or self.report != _report(run,self.report.get('records',[]))):
            raise ValueError('Incomplete or invalid pixel manifest identity')
        prepared = PreparedData(run['prepared_root'])
        try:
            expected,items = _inputs(prepared,run['crop'],run['dataset_root'])
            if run != expected or len(items) != len(self.report['records']):
                raise ValueError('Pixel preparation/source configuration mismatch')
            for item,row in zip(items,self.report['records']):
                _validate_record(self.root,row,_identity(item,run),item)
                self.records[row['record_id']] = row
        finally:prepared.close()


class FrameDataset:
    """Original rows/labels with 8 legacy or 32 dense causal head frames."""
    def __init__(self,prepared,cache,split,frames):
        if type(frames) is not int or frames not in (8,32):raise ValueError('Frames must be 8 or 32')
        if split not in ('train','val') or split not in prepared.rows:raise ValueError('Expected prepared train/val split')
        if cache.report['prepared_sha256'] != prepared.provenance['prepared_sha256']:
            raise ValueError('Pixel cache preparation identity mismatch')
        self.prepared,self.cache,self.split,self.frames = prepared,cache,split,frames
        for row in prepared.rows[split]:
            record = cache.records.get(row['record_id'])
            if record is None or any(record[k] != row[k] for k in METADATA):
                raise ValueError('Pixel dataset record metadata mismatch')

    def __len__(self):return len(self.prepared.rows[self.split])

    def metadata(self,index):return dict(self.prepared.rows[self.split][index])

    def availability(self,index):
        """Inspect masks without copying megabytes of pixels during supervision scans."""
        row = self.prepared.rows[self.split][index]
        arrays = self.cache.arrays(row['record_id'])
        start = row['start_slot']//25*8
        positions = start+np.arange(32) if self.frames==32 else start+np.arange(3,32,4)
        return bool(arrays['valid'][positions].all())

    def __getitem__(self,index):
        row = self.prepared.rows[self.split][index]
        arrays = self.cache.arrays(row['record_id'])
        start = row['start_slot']//25*8
        offsets = selected_slots(100) if self.frames == 32 else OFFSETS
        positions = start+np.arange(32) if self.frames == 32 else start+np.arange(3,32,4)
        if not np.array_equal(arrays['slot_indices'][positions],row['start_slot']+offsets):
            raise ValueError('Pixel window sampling misaligned')
        return dict(images=arrays['images'][positions].copy(),visual_mask=arrays['valid'][positions].copy()),np.float32(row['target']),bool(row['label_known'])
