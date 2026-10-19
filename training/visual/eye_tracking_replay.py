"""CPU-only, verified source replay of eye locator stop diagnostics."""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from importlib import import_module
import json
from pathlib import Path
import re
import sys

import numpy as np

from inference.experts.assets import file_sha256,verify_asset
from training.common.config import canonical_hash,read_json
from training.dmd.cache import _unlinked,validate_complete
from training.visual.extract import _stamp,write_json
from training.visual.eye_frames import EyeCache,_inputs,_pixels,recovery_policy
from training.visual.head_frames import FrameCache


FIELDS=('record_id','source_pts_ms','origin','detector_reason','geometry_reason','tracking_status','tracking_reason',
        'seed_age_ms','track_age_ms','source_gap_ms','last_stop_reason','last_stop_pts_ms','sampled_indices','trace_json')


def _replay(item,head,eyes,windows,trace_path):
    from inference.experts.face_experiment_adapter import FaceExperimentAdapter
    from inference.perception.video_source import decode_video
    from training.visual.eye_tracking import EyeLocator
    rid=item['record']['record_id'];a=head.arrays(rid);b=eyes.arrays(rid)
    indexes=defaultdict(list)
    for i,t in enumerate(a['source_pts_ms']):
        if a['valid'][i]:indexes[int(t)].append(i)
    verified=0;decoded=0;counts=Counter();window_counts=Counter();unique_counts=Counter();last_stop=None
    with ExitStack() as stack:
        source=np.load(item['attempt']/item['complete']['arrays']['source_pts_ms'],mmap_mode='r',allow_pickle=False)
        stack.callback(source._mmap.close)
        adapter=FaceExperimentAdapter(item['face_asset'],driver_roi=item['source']['driver_roi'],
            driver_region_confirmed_exclusive=item['config']['driver_region_confirmed_exclusive'],variant=item['variant'])
        stack.callback(adapter.close)
        locator=EyeLocator(adapter,item['head_roi'],item['source']['image_size'],tracking=True)
        if _stamp(item['source_path'])!=item['source_stamp']:raise ValueError('Source video changed before replay')
        frames=decode_video(item['source_path']);stack.callback(frames.close)
        handle=stack.enter_context(trace_path.open('w',encoding='utf-8',newline=''))
        writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader()
        for frame in frames:
            t=int(frame.timestamp_ms)
            if decoded>=len(source) or t!=int(source[decoded]):raise ValueError('Decoded source PTS mismatch')
            if list(reversed(frame.bgr.shape[:2]))!=item['source']['image_size']:raise ValueError('Decoded image dimensions mismatch')
            result=locator.step(frame.bgr,t);trace=locator.last_trace
            status,reason=trace['tracking_status'],trace['tracking_reason'];counts[status,reason]+=1
            if status=='detected':last_stop=None
            if status=='stopped' or reason=='seed_insufficient_texture':last_stop=(reason,t)
            covering=[w for w in windows if w['start_pts_ms']<=t<=w['end_pts_ms']]
            if covering:unique_counts[status,reason]+=1
            for window in covering:window_counts[window['window_id'],status,reason]+=1
            pixels=None
            for i in indexes.get(t,[]):
                valid=result['roi'] is not None;origin={'missing':0,'detected':1,'tracked':2}[result['origin']]
                if bool(b['valid'][i])!=valid or int(b['origin'][i])!=origin or int(b['track_age_ms'][i])!=result['track_age_ms']:
                    raise ValueError(f'Cannot reproduce cached localization: {rid} PTS {t}')
                if valid:
                    if (not np.array_equal(b['roi_xyxy'][i],np.asarray(result['roi'],np.float32))
                            or b['native_eye_width'][i]!=np.float32(result['width'])):
                        raise ValueError(f'Cannot reproduce cached geometry: {rid} PTS {t}')
                    if pixels is None:pixels=_pixels(frame.bgr,result['roi'])
                    if not np.array_equal(b['images'][i],pixels):raise ValueError(f'Cannot reproduce cached pixels: {rid} PTS {t}')
                verified+=1
            writer.writerow(dict(record_id=rid,source_pts_ms=t,origin=result['origin'],detector_reason=result['detector_reason'],
                geometry_reason=result['reason'],tracking_status=status,tracking_reason=reason,seed_age_ms=trace['seed_age_ms'],
                track_age_ms=result['track_age_ms'],source_gap_ms=trace['source_gap_ms'],
                last_stop_reason=last_stop[0] if last_stop else None,last_stop_pts_ms=last_stop[1] if last_stop else None,
                sampled_indices=json.dumps(indexes.get(t,[])),trace_json=json.dumps(trace,allow_nan=False,sort_keys=True)))
            decoded+=1
        if decoded!=len(source) or verified!=int(a['valid'].sum()):raise ValueError('Replay source/sample counts mismatch')
    if _stamp(item['source_path'])!=item['source_stamp']:raise ValueError('Source video changed during replay')
    if validate_complete(item['attempt'],item['complete']['fingerprint'])!=item['complete']:raise ValueError('Source cache changed')
    verify_asset(item['face_asset'])
    counters=[]
    for scope,counter in [('record_all_frames',counts),('requested_intervals_unique_frames',unique_counts)]:
        counters.extend(dict(record_id=rid,window_id='',scope=scope,tracking_status=k[0],reason=k[1],count=v) for k,v in sorted(counter.items()))
    counters.extend(dict(record_id=rid,window_id=k[0],scope='window_with_preroll',tracking_status=k[1],reason=k[2],count=v)
                    for k,v in sorted(window_counts.items()))
    return dict(record_id=rid,source_frames=decoded,verified_samples=verified,cache_reproduced=True,counters=counters)


def run(*,request,dataset_root,output_dir,continue_run=False):
    request_path=_unlinked(request);request_hash=file_sha256(request_path);spec=read_json(request_path)
    if spec.get('schema_version')!='driveguard-eye-breakpoint-replay-request-v1' or spec.get('status')!='complete':
        raise ValueError('Expected complete breakpoint replay request')
    source=_unlinked(spec['source_report'])
    if file_sha256(source)!=spec['source_report_sha256']:raise ValueError('Source report checksum mismatch')
    published=read_json(source)
    if published.get('schema_version')!='driveguard-eye-recovery-comparison-v1' or published.get('status')!='complete':
        raise ValueError('Expected complete recovery source report')
    binding=published['plan']['caches']['dense_track'];output=Path(output_dir).resolve()
    if output.exists() and not continue_run:raise FileExistsError('Use a NEW output_dir or --continue-run')
    with ExitStack() as stack:
        head=FrameCache(spec['head_cache']);stack.callback(head.close)
        eyes=EyeCache(spec['eye_cache'],head);stack.callback(eyes.close)
        if (head.manifest_sha256!=spec['head_cache_sha256'] or eyes.manifest_sha256!=spec['eye_cache_sha256']
                or eyes.manifest_sha256!=binding['manifest_sha256'] or eyes.root.resolve()!=Path(binding['path']).resolve()
                or eyes.report['run'].get('recovery')!=recovery_policy('dense_track','val')):
            raise ValueError('Replay cache identity/policy mismatch')
        runtime=dict(mediapipe_version=import_module('mediapipe').__version__,opencv_version=import_module('cv2').__version__)
        if runtime!=eyes.report['run']['detector_runtime']:raise ValueError('Replay runtime differs from original cache runtime')
        run_config,items=_inputs(head,dataset_root,runtime,recovery_policy('dense_track','val'))
        if run_config!=eyes.report['run']:raise ValueError('Replay source configuration mismatch')
        by_id={item['record']['record_id']:item for item in items};wanted={}
        for entry in spec['records']:
            rid=entry['record_id']
            if rid not in by_id or rid in wanted:raise ValueError('Unknown or duplicated replay record')
            ids=set()
            for w in entry['windows']:
                if (not re.fullmatch('[0-9a-f]{24}',w['window_id']) or w['window_id'] in ids
                        or type(w['start_pts_ms']) is not int or type(w['end_pts_ms']) is not int
                        or not 0<=w['start_pts_ms']<=w['end_pts_ms']<=int(head.arrays(rid)['source_pts_ms'][-1])):
                    raise ValueError('Invalid replay window')
                ids.add(w['window_id'])
            if not ids:raise ValueError('Replay record has no windows')
            wanted[rid]=entry['windows']
        roots=[request_path.parent,source.parent,head.root,eyes.root,Path(dataset_root),
               Path(head.report['run']['prepared_root']),Path(head.report['run']['data_root'])]
        for path in roots:
            path=path.resolve()
            if output.is_relative_to(path) or path.is_relative_to(output):raise ValueError('Replay output overlaps input')
        from training.visual.eye_tracking import POLICY
        plan=dict(schema_version='driveguard-eye-source-replay-plan-v1',request=str(request_path),request_sha256=request_hash,
            dataset_root=str(Path(dataset_root).resolve()),runtime=runtime,tracking_policy=POLICY,diagnostic_version='eye-tracking-trace-v1')
        if output.exists():
            if read_json(output/'plan.json')!=plan:raise ValueError('Source replay resume plan mismatch')
        else:output.mkdir(parents=True);write_json(output/'plan.json',plan)
        records=[];counters=[]
        for rid,windows in sorted(wanted.items()):
            token=canonical_hash(rid)[:24];trace_file=f'{token}-source.csv';marker=output/f'{token}.json'
            if marker.exists():
                record=read_json(marker)
                if (record.get('record_id')!=rid or record.get('plan_hash')!=canonical_hash(plan) or record.get('trace_file')!=trace_file
                        or not record.get('cache_reproduced') or file_sha256(output/trace_file)!=record.get('trace_sha256')):
                    raise ValueError('Completed source trace checksum/provenance mismatch')
            else:
                print(json.dumps(dict(stage='replay_eye_locator',record_id=rid)),flush=True)
                record=_replay(by_id[rid],head,eyes,windows,output/trace_file)
                record.update(plan_hash=canonical_hash(plan),trace_file=trace_file,trace_sha256=file_sha256(output/trace_file))
                write_json(marker,record)
            records.append(record);counters.extend(record['counters'])
        if file_sha256(request_path)!=request_hash or file_sha256(source)!=spec['source_report_sha256']:
            raise ValueError('Replay inputs changed')
        report=dict(schema_version='driveguard-eye-source-replay-v1',status='complete',cache_reproduced=True,training_runs=0,
            plan=plan,source_frames=sum(r['source_frames'] for r in records),records=records,
            notes=['Trace unit is decoded source frames; counts differ from 8Hz cached samples.',
                   'stopped is the first terminating gate; later no_seed frames are not additional termination events.',
                   'Window scopes include diagnostic preroll and can overlap; use unique scope for deduplicated counts.',
                   'Geometry and current pixels reproduce every fresh cached sample of each requested recording.',
                   'No new training, threshold, locator policy, or cache changes.'])
        if (output/'summary.json').exists():
            previous=read_json(output/'summary.json')
            if file_sha256(output/'stop-reasons.csv')!=previous.get('stop_reasons_sha256'):
                raise ValueError('Completed stop reason table checksum mismatch')
            report['stop_reasons_sha256']=previous['stop_reasons_sha256']
            if read_json(output/'summary.json')!=report:raise ValueError('Completed source replay differs')
        else:
            with (output/'stop-reasons.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=('record_id','window_id','scope','tracking_status','reason','count'))
                writer.writeheader();writer.writerows(counters)
            report['stop_reasons_sha256']=file_sha256(output/'stop-reasons.csv')
            write_json(output/'summary.json',report)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('request','dataset-root','output-dir'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--continue-run',action='store_true')
    try:
        result=run(**vars(parser.parse_args(argv)));print(json.dumps(dict(status=result['status'],source_frames=result['source_frames'],cache_reproduced=True)));return 0
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        print(f'error: {error}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
