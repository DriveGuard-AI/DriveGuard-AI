"""Bounded, immutable native eye pixels aligned to the original 8 Hz head cache.

Landmark crops are candidate eye regions, not verified visibility annotations.
Missing regions are unavailable; this module never reads action labels.
"""
import argparse
from collections import Counter, OrderedDict
from contextlib import ExitStack
import math
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import numpy as np
from PIL import Image

from inference.experts.assets import file_sha256, verify_asset
from training.common.config import canonical_hash, read_json
from training.dmd.cache import _unlinked, validate_complete
from training.dmd.face_experiment_config import BASELINE, validate_rectangle
from training.visual.data import PreparedData, VisualCache
from training.visual.extract import _contained, _identity, _stamp, write_json
from training.visual.head_frames import FrameCache, _inputs as head_inputs, _validate_record as validate_head
from training.visual.spatial import crop_roi

SCHEMA = 'driveguard-native-eye-pixels-v1'
POLICY = dict(version='larger-eye-corners-v1',corners=[[33,133],[362,263]],
              min_native_span=6.,square_side_per_span=2.,selection='largest_projected_euclidean_span',
              containment='all_face_landmarks_and_padded_eye_inside_head_and_image_no_clipping',
              missing='unavailable_never_negative',visibility='candidate_not_human_verified',
              detector='FaceExperimentAdapter-head-crop-VIDEO-confidence-0.5',
              inference='unique_fresh_sampled_source_pts_only',label_access=False)


def eye_geometry(observation, head_roi, image_size):
    """Return (square xyxy, corner span, reason), never gate on eyelid opening."""
    if observation.get('status') != 'ok' or observation.get('driver_confirmed') is not True:
        return None,0.,'unconfirmed_or_missing_face'
    try:points=np.asarray(observation.get('landmarks'),dtype=np.float64)
    except (ValueError,TypeError):return None,0.,'invalid_landmarks'
    if points.shape not in ((468,2),(478,2)) or not np.isfinite(points).all():
        return None,0.,'invalid_landmarks'
    x0,y0,x1,y1=head_roi; w,h=image_size
    if (np.any(points[:,0]<max(0,x0)) or np.any(points[:,0]>min(w,x1))
            or np.any(points[:,1]<max(0,y0)) or np.any(points[:,1]>min(h,y1))):
        return None,0.,'face_outside_head'
    pairs=[points[[a,b]] for a,b in POLICY['corners']]
    spans=[float(np.linalg.norm(pair[1]-pair[0])) for pair in pairs]
    index=int(np.argmax(spans));span=spans[index]
    if span<POLICY['min_native_span']:return None,0.,'tiny_eye'
    center=pairs[index].mean(axis=0);half=span*POLICY['square_side_per_span']/2
    roi=[float(center[0]-half),float(center[1]-half),float(center[0]+half),float(center[1]+half)]
    if not (max(0,x0)<=roi[0]<roi[2]<=min(w,x1) and max(0,y0)<=roi[1]<roi[3]<=min(h,y1)):
        return None,0.,'padded_eye_outside_head'
    return roi,span,'accepted'


def _pixels(bgr, roi):
    """Torch-free driver-roi-letterbox-v1 pixel preprocessing."""
    x0,y0,x1,y1=roi
    crop=bgr[math.floor(y0):math.ceil(y1),math.floor(x0):math.ceil(x1),::-1]
    height,width=crop.shape[:2];scale=224/max(height,width)
    size=max(1,round(width*scale)),max(1,round(height*scale))
    rgb=Image.fromarray(np.ascontiguousarray(crop)).resize(size,Image.Resampling.BILINEAR)
    canvas=np.zeros((224,224,3),np.uint8);left,top=(224-size[0])//2,(224-size[1])//2
    canvas[top:top+size[1],left:left+size[0]]=np.asarray(rgb)
    return np.ascontiguousarray(canvas.transpose(2,0,1))


def recovery_policy(mode=None,split='all'):
    if mode is None:
        if split!='all':raise ValueError('A restricted split requires an explicit recovery policy')
        return None
    if mode not in ('sampled8','dense','dense_track') or split not in ('all','val'):
        raise ValueError('Invalid recovery policy/split')
    tracking_policy=None
    if mode=='dense_track':
        from training.visual.eye_tracking import POLICY as tracking_policy
    return dict(version='eye-continuity-v1',mode=mode,split=split,
                detector_cadence='sampled-source' if mode=='sampled8' else 'every-decoded-source-frame',
                tracking=tracking_policy if mode=='dense_track' else None)


def _inputs(head,dataset_root,detector_runtime,recovery=None):
    runtime_keys={'mediapipe_version'}|({'opencv_version'} if recovery and recovery['mode']=='dense_track' else set())
    if (not isinstance(detector_runtime,dict) or set(detector_runtime)!=runtime_keys
            or not isinstance(detector_runtime['mediapipe_version'],str) or not detector_runtime['mediapipe_version']):
        raise ValueError('Invalid detector runtime identity')
    if recovery is not None and recovery!=recovery_policy(recovery['mode'],recovery['split']):
        raise ValueError('Recovery policy identity mismatch')
    if 'opencv_version' in runtime_keys and (not isinstance(detector_runtime['opencv_version'],str) or not detector_runtime['opencv_version']):
        raise ValueError('Invalid optical flow runtime identity')
    if (file_sha256(_contained(head.root,'manifest.json')) != head.manifest_sha256
            or read_json(_contained(head.root,'manifest.json')) != head.report):
        raise ValueError('Head cache manifest changed')
    original=head.report['run']
    prepared=PreparedData(original['prepared_root'])
    try:expected,items=head_inputs(prepared,original['crop'],dataset_root)
    finally:prepared.close()
    if original != expected or read_json(_contained(head.root,'input.json')) != original:
        raise ValueError('Head cache source configuration mismatch')
    if len(items)!=len(head.records):raise ValueError('Head cache record identity mismatch')
    if recovery and recovery['split']!='all':
        items=[item for item in items if item['record']['split']==recovery['split']]
        if not items:raise ValueError('No records in requested recovery split')
    detectors={}
    for item in items:
        rid=item['record']['record_id']
        validate_head(head.root,head.records[rid],_identity(item,original),item)
        config=item['complete']['feature_record']['provenance']['config']
        asset=verify_asset(config['face_asset'])
        roi=crop_roi(item['source']['driver_roi'],original['crop'])
        w,h=item['source']['image_size'];normalized=[roi[0]/w,roi[1]/h,roi[2]/w,roi[3]/h]
        validate_rectangle(normalized)
        item.update(head_roi=roi,config=config,face_asset=asset,
                    variant=dict(BASELINE,name='native_eye_head_crop',crop_xyxy=normalized))
        if recovery is not None:item['recovery']=recovery
        detectors[rid]=dict(config_sha256=canonical_hash(config),face_asset=asset,
                            driver_region_confirmed_exclusive=config['driver_region_confirmed_exclusive'],
                            head_roi=roi,variant=item['variant'])
    run=dict(schema_version=SCHEMA,head_cache=str(head.root),head_cache_sha256=head.manifest_sha256,
             prepared_sha256=head.report['prepared_sha256'],dataset_root=str(_unlinked(dataset_root)),
             policy=POLICY,detectors=detectors,detector_runtime=detector_runtime,
             preprocessing=dict(version='driver-roi-letterbox-v1',shape=[3,224,224],dtype='uint8',color='RGB'),
             sampling=original['sampling'])
    if recovery is not None:run['recovery']=recovery
    return run,items


def _specs(count,recovery=False):
    result=dict(images=((count,3,224,224),np.uint8),slot_indices=((count,),np.int64),
                source_pts_ms=((count,),np.int64),valid=((count,),np.bool_),
                roi_xyxy=((count,4),np.float32),native_eye_width=((count,),np.float32))
    if recovery:result.update(origin=((count,),np.uint8),track_age_ms=((count,),np.int64))
    return result


def _audit(valid,width,reasons,localization=None):
    selected=width[valid]
    result=dict(sample_count=len(valid),available=int(valid.sum()),missing=int((~valid).sum()),
                native_eye_width=dict(min=float(selected.min()),median=float(np.median(selected)),max=float(selected.max())) if len(selected) else None,
                reasons=dict(sorted(reasons.items())))
    if localization is not None:result['localization']=localization
    return result


def _validate_record(output,row,identity,item,head):
    if (not isinstance(row,dict) or set(row)!=set(identity)|{'status','sample_count','directory','arrays','sha256','audit'}
            or any(row.get(k)!=v for k,v in identity.items()) or row['status']!='complete'):
        raise ValueError('Eye record configuration/provenance mismatch')
    marker=_contained(output,row['directory']+'/record.json')
    if read_json(marker)!=row or (marker.parent/'failure.json').exists():raise ValueError('Invalid eye completion marker')
    original=head.arrays(identity['record_id']);count=len(original['valid']);specs=_specs(count,'recovery' in item)
    if type(row['sample_count']) is not int or row['sample_count']!=count or set(row['arrays'])!=set(specs) or set(row['sha256'])!=set(specs):
        raise ValueError('Invalid eye array layout')
    with ExitStack() as stack:
        arrays={}
        for key,(shape,dtype) in specs.items():
            if row['arrays'][key]!=row['directory']+'/'+key+'.npy':raise ValueError('Unexpected eye array path')
            path=_contained(output,row['arrays'][key])
            if file_sha256(path)!=row['sha256'][key]:raise ValueError('Eye array checksum mismatch')
            array=np.load(path,mmap_mode='r',allow_pickle=False);stack.callback(array._mmap.close)
            if array.shape!=shape or array.dtype!=dtype:raise ValueError('Eye array dimensions or dtype mismatch')
            arrays[key]=array
        if any(not np.array_equal(arrays[k],original[k]) for k in ('slot_indices','source_pts_ms')):
            raise ValueError('Eye sampling or PTS differs from head cache')
        x0,y0,x1,y1=item['head_roi']
        for start in range(0,count,16):
            sl=slice(start,start+16);valid=arrays['valid'][sl];roi=arrays['roi_xyxy'][sl];width=arrays['native_eye_width'][sl]
            if np.any(valid & ~original['valid'][sl]):raise ValueError('Eye validity exceeds head freshness')
            if (not np.isfinite(roi).all() or not np.isfinite(width).all()
                    or np.any(roi[~valid]!=0) or np.any(width[~valid]!=0) or np.any(arrays['images'][sl][~valid]!=0)):
                raise ValueError('Invalid eye pixel or geometry values')
            r=roi[valid];s=width[valid]
            if (np.any(s<POLICY['min_native_span']) or np.any(r[:,0]<x0) or np.any(r[:,1]<y0)
                    or np.any(r[:,2]>x1) or np.any(r[:,3]>y1)
                    or not np.allclose(r[:,2]-r[:,0],s*2,rtol=1e-5,atol=1e-4)
                    or not np.allclose(r[:,3]-r[:,1],s*2,rtol=1e-5,atol=1e-4)):
                raise ValueError('Eye geometry outside head or inconsistent square/span')
        localization=row['audit'].get('localization')
        if 'recovery' in item:
            origin=arrays['origin'];age=arrays['track_age_ms']
            max_age=item['recovery']['tracking']['max_age_ms'] if item['recovery']['tracking'] else 0
            if (np.any(origin>2) or not np.array_equal(origin>0,arrays['valid']) or np.any(age<0)
                    or np.any(age[origin!=2]!=0) or np.any(age[origin==2]<=0) or np.any(age>max_age)):
                raise ValueError('Invalid localization origin or tracking age')
            if (not isinstance(localization,dict) or set(localization)!={'detector_calls','sampled_detector_reasons','origins'}
                    or localization['origins']!={name:int((origin==i).sum()) for i,name in enumerate(('missing','detected','tracked'))}):
                raise ValueError('Invalid localization audit')
            for key in ('detector_calls','sampled_detector_reasons'):
                values=localization[key]
                if not isinstance(values,dict) or any(type(v) is not int or v<0 for v in values.values()):
                    raise ValueError('Invalid detector reason counts')
            if sum(localization['sampled_detector_reasons'].values())!=count:
                raise ValueError('Detector sample count mismatch')
        elif localization is not None:raise ValueError('Unexpected recovery audit in legacy cache')
        reasons=row['audit'].get('reasons',{})
        if (not isinstance(reasons,dict) or any(type(v) is not int or v<0 for v in reasons.values())
                or sum(reasons.values())!=count or reasons.get('accepted',0)!=int(arrays['valid'].sum())
                or row['audit']!=_audit(arrays['valid'],arrays['native_eye_width'],reasons,localization)):
            raise ValueError('Eye availability audit mismatch')
    return row


def _extract_record(output,directory,item,identity,head):
    from inference.experts.face_experiment_adapter import FaceExperimentAdapter
    from inference.perception.video_source import decode_video, sample_frames
    original=head.arrays(identity['record_id']);count=len(original['valid']);arrays={};reasons=Counter()
    recovery=item.get('recovery');detector_calls=Counter();sampled_reasons=Counter()
    try:
        with ExitStack() as stack:
            for key,(shape,dtype) in _specs(count,recovery is not None).items():
                arrays[key]=np.lib.format.open_memmap(directory/(key+'.npy'),mode='w+',dtype=dtype,shape=shape)
                stack.callback(arrays[key]._mmap.close);arrays[key][:]=0
            arrays['slot_indices'][:]=original['slot_indices'];arrays['source_pts_ms'][:]=original['source_pts_ms']
            source=np.load(item['attempt']/item['complete']['arrays']['source_pts_ms'],mmap_mode='r',allow_pickle=False)
            stack.callback(source._mmap.close)
            adapter=FaceExperimentAdapter(item['face_asset'],driver_roi=item['source']['driver_roi'],
                driver_region_confirmed_exclusive=item['config']['driver_region_confirmed_exclusive'],variant=item['variant'])
            stack.callback(adapter.close)
            locator=None
            if recovery is not None:
                from training.visual.eye_tracking import EyeLocator
                locator=EyeLocator(adapter,item['head_roi'],item['source']['image_size'],tracking=recovery['mode']=='dense_track')
            snapshots=OrderedDict()
            def locate(frame):
                result=locator.step(frame.bgr,int(frame.timestamp_ms))
                detector_calls[result['detector_reason']]+=1
                return result
            if _stamp(item['source_path'])!=item['source_stamp']:raise ValueError('Source video changed after verification')
            frames=decode_video(item['source_path']);stack.callback(frames.close)
            def checked_frames():
                decoded=0
                for frame in frames:
                    if decoded>=len(source) or frame.timestamp_ms!=int(source[decoded]):raise ValueError('Decoded source PTS mismatch')
                    if list(reversed(frame.bgr.shape[:2]))!=item['source']['image_size']:raise ValueError('Decoded image dimensions mismatch')
                    if recovery and recovery['mode']!='sampled8':
                        snapshots[frame.timestamp_ms]=locate(frame)
                        # Resampling can decode one future frame. Retain the
                        # PREVIOUS frame's snapshot; never use latest state.
                        while len(snapshots)>2:snapshots.popitem(last=False)
                    decoded+=1;yield frame
                if decoded!=len(source):raise ValueError('Decoded source frame count mismatch')
            checked=checked_frames();stack.callback(checked.close)
            samples=sample_frames(checked);stack.callback(samples.close)
            total=selected=0;previous_pts=None;geometry=None;pixels=None
            for slot,frame in samples:
                if total>=identity['slots'] or slot!=total*40:raise ValueError('Decoded slot count/grid mismatch')
                if selected<count and total==int(original['slot_indices'][selected]):
                    if frame.timestamp_ms!=int(original['source_pts_ms'][selected]):raise ValueError('Sampled source PTS mismatch')
                    if bool(original['valid'][selected]):
                        if previous_pts!=frame.timestamp_ms:
                            if recovery:
                                localized=locate(frame) if recovery['mode']=='sampled8' else snapshots[frame.timestamp_ms]
                                reason=localized['reason']
                                if reason=='unconfirmed_or_missing_face':reason=localized['detector_reason']
                                geometry=localized['roi'],localized['width'],reason
                            else:
                                observation=adapter.predict(frame.bgr,int(frame.timestamp_ms))
                                geometry=eye_geometry(observation,item['head_roi'],item['source']['image_size'])
                            roi,width,reason=geometry
                            pixels=_pixels(frame.bgr,roi) if roi is not None else None
                            previous_pts=frame.timestamp_ms
                        roi,width,reason=geometry
                        if roi is not None:
                            arrays['valid'][selected]=True;arrays['roi_xyxy'][selected]=roi
                            arrays['native_eye_width'][selected]=width;arrays['images'][selected]=pixels
                        if recovery:
                            sampled_reasons[localized['detector_reason']]+=1
                            arrays['origin'][selected]={'missing':0,'detected':1,'tracked':2}[localized['origin']]
                            arrays['track_age_ms'][selected]=localized['track_age_ms']
                    else:
                        reason='stale_source'
                        if recovery:sampled_reasons['stale_source']+=1
                    reasons[reason]+=1;selected+=1
                total+=1
            if total!=identity['slots'] or selected!=count:raise ValueError('Decoded slot count mismatch')
            if _stamp(item['source_path'])!=item['source_stamp']:raise ValueError('Source video changed during extraction')
            if validate_complete(item['attempt'],item['complete']['fingerprint'])!=item['complete']:raise ValueError('Original source cache changed')
            verify_asset(item['face_asset'])
            localization=None if not recovery else dict(detector_calls=dict(sorted(detector_calls.items())),
                sampled_detector_reasons=dict(sorted(sampled_reasons.items())),
                origins={name:int((arrays['origin']==i).sum()) for i,name in enumerate(('missing','detected','tracked'))})
            audit=_audit(arrays['valid'],arrays['native_eye_width'],reasons,localization)
            for array in arrays.values():array.flush()
        relative=directory.relative_to(output).as_posix();paths={key:relative+'/'+key+'.npy' for key in arrays}
        row=dict(identity,status='complete',sample_count=count,directory=relative,arrays=paths,
                 sha256={key:file_sha256(output/path) for key,path in paths.items()},audit=audit)
        write_json(directory/'record.json',row);return row
    except BaseException as error:
        write_json(directory/'failure.json',dict(status='failed',error=str(error),error_type=type(error).__name__))
        raise


def _report(run,records):
    return dict(schema_version=SCHEMA,status='complete',run=run,prepared_sha256=run['prepared_sha256'],records=records)


def build_eye_cache(head_cache,dataset_root,output_dir,continue_run=False,recovery=None,split='all'):
    """Extract from original decoded pixels, resuming verified complete records."""
    head=head_cache;output=Path(output_dir).absolute();original=head.report['run']
    roots=[head.root,Path(original['prepared_root']),Path(original['data_root']),Path(dataset_root)]
    prepared_report=read_json(Path(original['prepared_root'])/'prepared.json')
    roots.append(Path(prepared_report['visual_cache']))
    if any(output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()) for root in roots):
        raise ValueError('Eye output must be separate from head cache, preparation, original cache and dataset')
    # Check existing ancestors before mkdir so a linked parent cannot redirect writes.
    parent=output
    while not parent.exists():parent=parent.parent
    _unlinked(parent)
    if output.exists() and not continue_run:raise FileExistsError('Eye output exists; explicit continue_run required')
    from importlib import import_module
    runtime=dict(mediapipe_version=import_module('mediapipe').__version__)
    policy=recovery_policy(recovery,split)
    if recovery=='dense_track':runtime['opencv_version']=import_module('cv2').__version__
    run,items=_inputs(head,dataset_root,runtime,policy)
    if output.exists():
        output=_unlinked(output)
        if read_json(_contained(output,'input.json'))!=run:raise ValueError('Eye resume configuration/provenance mismatch')
        if (output/'manifest.json').exists():
            report=read_json(_contained(output,'manifest.json'))
            if not isinstance(report,dict) or report!=_report(run,report.get('records',[])) or len(report['records'])!=len(items):
                raise ValueError('Invalid completed eye manifest')
            for item,row in zip(items,report['records']):_validate_record(output,row,_identity(item,run),item,head)
            return report
    else:
        output.mkdir(parents=True);output=_unlinked(output);write_json(output/'input.json',run)
    completed=[]
    for item in items:
        identity=_identity(item,run);record_dir=output/('record-'+canonical_hash(identity['record_id']))
        record_dir.mkdir(exist_ok=True);_unlinked(record_dir);reusable=None
        for marker in sorted(record_dir.glob('attempt-*/record.json')):
            try:reusable=_validate_record(output,read_json(_unlinked(marker)),identity,item,head);break
            except (OSError,ValueError,KeyError,TypeError,EOFError):continue
        if reusable is None:
            needed=head.records[identity['record_id']]['sample_count']*(3*224*224+37)+4096
            if shutil.disk_usage(output).free<needed+16*1024**2:raise OSError('Insufficient disk space for native eye record')
            print('Caching native eye RGB: '+identity['record_id'],file=sys.stderr,flush=True)
            directory=record_dir/('attempt-'+uuid4().hex);directory.mkdir()
            reusable=_extract_record(output,directory,item,identity,head)
        completed.append(reusable)
    if file_sha256(_contained(head.root,'manifest.json'))!=head.manifest_sha256:raise ValueError('Head manifest changed during extraction')
    report=_report(run,completed);write_json(output/'manifest.json',report);return report


class EyeCache(VisualCache):
    """Source/hash validated cache; inherited mmap LRU opens at most four records."""
    def __init__(self,root,head_cache):
        self.root=_unlinked(root);self._opened=OrderedDict();self.records={}
        self.report=read_json(_contained(self.root,'manifest.json'));self.manifest_sha256=file_sha256(self.root/'manifest.json')
        stored=read_json(_contained(self.root,'input.json'))
        run,items=_inputs(head_cache,head_cache.report['run']['dataset_root'],stored.get('detector_runtime'),stored.get('recovery'))
        if (read_json(_contained(self.root,'input.json'))!=run or not isinstance(self.report,dict)
                or self.report!=_report(run,self.report.get('records',[])) or len(self.report['records'])!=len(items)):
            raise ValueError('Eye cache source/configuration mismatch')
        for item,row in zip(items,self.report['records']):
            _validate_record(self.root,row,_identity(item,run),item,head_cache)
            self.records[row['record_id']]=row


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('head-cache','dataset-root','output-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--continue-run',action='store_true')
    parser.add_argument('--recovery',choices=['sampled8','dense','dense_track'])
    parser.add_argument('--split',choices=['all','val'],default='all')
    options=vars(parser.parse_args(argv));head=FrameCache(options.pop('head_cache'))
    try:build_eye_cache(head,**options)
    finally:head.close()


if __name__=='__main__':main()
