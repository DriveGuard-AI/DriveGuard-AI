"""Explain fixed dense-track eye events; cached inference only, zero training."""
import argparse
from collections import Counter, defaultdict
from html import escape
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from training.common.checkpoint import atomic_json
from training.common.config import canonical_hash, read_json
from training.common.data import file_sha256
from training.visual.action_eval import evaluate_events
from training.visual.action_model import EventDecoder
from training.visual.eye_composition import compose_events, infer_eyes, frame_metrics
from training.visual.eye_frames import EyeCache
from training.visual.extract import _contained
from training.visual.eye_recovery import _reference, _reproduce, VARIANT
from training.visual.fatigue_eyes import _state_hash
from training.visual.report import write_rows
from training.visual.spatial import crop_roi


def trace_record(head, eye, scores, resets, duration):
    """Use the production decoder and recording-wide history, never labels."""
    n=len(head['valid']);scores=np.asarray(scores)
    if scores.shape!=(n,) or len(eye['valid'])!=n or len(resets)!=n:
        raise ValueError('Trace arrays do not align')
    decoder=EventDecoder(duration);support=EventDecoder(duration);result=[]
    previous_time=None;previous_slot=None;last_break=None
    for i,(slot,pts) in enumerate(zip(head['slot_indices'],head['source_pts_ms'])):
        t=int(pts);slot=int(slot);hv=bool(head['valid'][i]);ev=bool(eye['valid'][i]);valid=hv and ev
        if previous_slot is not None and slot<=previous_slot:raise ValueError('Unordered trace slots')
        if previous_time is not None and t<previous_time:raise ValueError('Unordered trace PTS')
        if valid and not np.isfinite(scores[i]):raise ValueError('Nonfinite observed eye score')
        score=float(scores[i]) if valid else None;reset=bool(resets[i])
        reason=('missing_head' if not hv else 'missing_eye' if not ev else
                'source_reset' if reset else 'below_threshold' if score<0 else 'none')
        previous_since=decoder.since
        stopped_ms=previous_time-previous_since if previous_since is not None and reason!='none' else None
        if reason!='none':last_break=dict(slot=slot,pts_ms=t,reason=reason)
        # Missing eye is the same negative sentinel used by compose_events.
        event=decoder.step(t,score if valid else -1e6,-1e6,hv,reset=reset)
        hypothetical=support.step(t,1.,-1e6,valid,reset=reset)
        result.append(dict(sample_index=i,slot_index=slot,source_pts_ms=t,head_valid=hv,eye_valid=ev,
            observed=valid,source_reset=reset,eye_logit=score,
            origin=int(eye['origin'][i]),track_age_ms=int(eye['track_age_ms'][i]),
            closed_since_ms=decoder.since,closed_run_ms=t-decoder.since if decoder.since is not None else 0,
            closure_margin=decoder.margin,eye_event=bool(event is not None and event>=0),
            input_run_ms=t-support.since if support.since is not None else 0,
            input_duration_eligible=bool(hypothetical is not None and hypothetical>=0),
            break_reason=reason,interrupted_closed_run_ms=stopped_ms,
            last_break_slot=last_break['slot'] if last_break else None,
            last_break_pts_ms=last_break['pts_ms'] if last_break else None,
            last_break_reason=last_break['reason'] if last_break else 'record_start'))
        previous_time=t;previous_slot=slot
    return result


def describe_window(row, subtype, trace):
    selected=[r for r in trace if row['start_slot']<=r['slot_index']<=row['end_slot']]
    if not selected or selected[-1]['slot_index']!=row['end_slot']:
        raise ValueError('Window endpoint missing from trace')
    end=selected[-1];head_ok=all(r['head_valid'] for r in selected)
    support=head_ok and end['input_duration_eligible']
    detected=head_ok and end['observed'] and end['eye_event']
    reason=('head_window_unavailable' if not head_ok else 'missing_eye_endpoint' if not end['observed'] else
            'input_duration_short' if not support else 'eye_detected' if detected else
            'predicted_open_endpoint' if end['eye_logit']<0 else 'predicted_closure_short')
    missing_start=None;max_gap=0;max_gap_samples=0;gap_count=0
    for r in selected:
        if not r['observed']:
            if missing_start is None:missing_start=r['slot_index'];gap_count=0
            gap_count+=1;max_gap=max(max_gap,(r['slot_index']-missing_start)*40)
            max_gap_samples=max(max_gap_samples,gap_count)
        else:missing_start=None
    observed=[r for r in selected if r['observed']]
    values=[r['eye_logit'] for r in observed]
    return dict(window_id=canonical_hash([row['record_id'],row['start_slot'],row['end_slot']])[:24],
        record_id=row['record_id'],subject_id=row['subject_id'],start_slot=row['start_slot'],end_slot=row['end_slot'],
        start_pts_ms=selected[0]['source_pts_ms'],end_pts_ms=end['source_pts_ms'],
        label_known=bool(row['label_known']),target=row['target'],subtype=subtype,
        combined_available=bool(row['available']),combined_logit=row['logit'],
        false_negative=bool(row['label_known'] and row['target']==1 and not(row['available'] and row['logit']>=0)),
        samples=len(selected),observed_samples=len(observed),missing_samples=len(selected)-len(observed),
        detected_samples=sum(r['observed'] and r['origin']==1 for r in selected),
        tracked_samples=sum(r['observed'] and r['origin']==2 for r in selected),
        longest_missing_sample_span_ms=max_gap,longest_missing_sample_count=max_gap_samples,
        longest_input_run_seen_ms=max(r['input_run_ms'] for r in selected),
        longest_closed_run_seen_ms=max(r['closed_run_ms'] for r in selected),
        endpoint_input_run_ms=end['input_run_ms'],endpoint_closed_run_ms=end['closed_run_ms'],
        eye_endpoint_available=end['observed'],input_duration_eligible=support,eye_event=detected,reason=reason,
        eye_logit_min=min(values) if values else None,eye_logit_max=max(values) if values else None,
        below_threshold_samples=sum(v<0 for v in values),
        last_break_slot=end['last_break_slot'],last_break_pts_ms=end['last_break_pts_ms'],
        last_break_reason=end['last_break_reason'])


def write_playback(output, window_id, head, eye, trace, first, last, head_roi):
    directory=output/window_id;directory.mkdir(parents=True,exist_ok=True)
    frames=[];times=[];rows=[]
    x0,y0,x1,y1=head_roi;scale=224/max(x1-x0,y1-y0)
    w,h=max(1,round((x1-x0)*scale)),max(1,round((y1-y0)*scale))
    for j,i in enumerate(range(first,last+1)):
        r=trace[i];canvas=Image.new('RGB',(448,340),'#111827');draw=ImageDraw.Draw(canvas)
        left=Image.fromarray(head['images'][i].transpose(1,2,0).copy())
        if r['observed']:
            roi=eye['roi_xyxy'][i]
            box=[(roi[0]-x0)*w/(x1-x0)+(224-w)//2,(roi[1]-y0)*h/(y1-y0)+(224-h)//2,
                 (roi[2]-x0)*w/(x1-x0)+(224-w)//2,(roi[3]-y0)*h/(y1-y0)+(224-h)//2]
            ImageDraw.Draw(left).rectangle(box,outline='yellow',width=2)
            right=Image.fromarray(eye['images'][i].transpose(1,2,0).copy())
        else:
            right=Image.new('RGB',(224,224),'#202020')
            ImageDraw.Draw(right).text((50,104),'MISSING EYE',fill='white')
        canvas.paste(left,(0,76));canvas.paste(right,(224,76))
        score='missing' if r['eye_logit'] is None else f"{r['eye_logit']:.3f}"
        lines=[f"PTS {r['source_pts_ms']} ms | slot {r['slot_index']}",
               f"Eye score {score} | origin {r['origin']} | age {r['track_age_ms']} ms",
               f"Closed run {r['closed_run_ms']} ms | input run {r['input_run_ms']} ms",
               f"Event {r['eye_event']} | break {r['break_reason']}"]
        for k,line in enumerate(lines):draw.text((5,3+k*18),line,fill='white')
        draw.text((5,306),'Cached head + eye; yellow = candidate eye ROI',fill='white')
        draw.text((5,322),'0 missing / 1 detected / 2 tracked; score >= 0 closed',fill='white')
        file=f'frame-{j:03d}.jpg';canvas.save(directory/file,quality=85)
        frames.append(canvas);times.append(r['slot_index']*40);rows.append(dict(r,file=file))
    durations=[max(20,times[i+1]-times[i]) for i in range(len(times)-1)]+[120]
    frames[0].save(directory/'replay.gif',save_all=True,append_images=frames[1:],duration=durations,loop=0)
    payload=json.dumps(rows,allow_nan=False).replace('<','\\u003c')
    html='''<!doctype html><meta charset="utf-8"><title>Eye breakpoint replay</title>
<style>body{font:16px sans-serif;max-width:920px;margin:24px auto;background:#f3f4f6}img{max-width:100%}pre{white-space:pre-wrap}input{width:90%}</style>
<h1>眼区与事件断点回放</h1><p>左右为缓存头部、眼部输入；黄色框为候选眼区。缺失不能解释为睁眼或闭眼。</p>
<button id="play">播放 / 暂停</button> <a href="replay.gif">动图</a><p><input id="seek" type="range" min="0" value="0"></p>
<img id="frame"><pre id="info"></pre><script>
const rows=PAYLOAD;const seek=document.getElementById('seek');seek.max=rows.length-1;
const frame=document.getElementById('frame'),info=document.getElementById('info');
let playing=false,timer;function show(){let r=rows[Number(seek.value)];frame.src=r.file;info.textContent=JSON.stringify(r,null,2);}
function tick(){if(!playing)return;let i=Number(seek.value);seek.value=(i+1)%rows.length;show();
let j=Number(seek.value);timer=setTimeout(tick,j+1<rows.length?Math.max(20,(rows[j+1].slot_index-rows[j].slot_index)*40):120);}
document.getElementById('play').onclick=()=>{playing=!playing;clearTimeout(timer);if(playing)tick();};
seek.oninput=()=>{playing=false;clearTimeout(timer);show();};show();</script>'''.replace('PAYLOAD',payload)
    (directory/'index.html').write_text(html,encoding='utf-8')
    return f'{window_id}/index.html'


def run(*,recovery_report,dataset_root,output_dir,backend='cpu',device_index=0,continue_run=False):
    source=Path(recovery_report).resolve(strict=True);digest=file_sha256(source);published=read_json(source)
    if (published.get('schema_version')!='driveguard-eye-recovery-comparison-v1' or published.get('status')!='complete'
            or published.get('split')!='val' or not published.get('reference_reproduced')
            or published.get('training_runs')!=0 or not published.get('frozen_weights_unchanged')):
        raise ValueError('Expected completed frozen recovery comparison')
    original_plan=published['plan'];output=Path(output_dir).resolve()
    if output.exists() and not continue_run:raise FileExistsError('Use a NEW output_dir or --continue-run')
    if original_plan['event_policy']!=dict(closed_eye_min_ms=1000,threshold_logit=0.,missing='reset-eye-only'):
        raise ValueError('Expected unchanged 1000ms event policy')
    with _reference(original_plan['provenance']['eye_report'],VARIANT,dataset_root,backend,device_index) as data:
        if data['provenance']!=original_plan['provenance']:raise ValueError('Recovery reference provenance mismatch')
        head=data['cache'];labels=data['labels'];binding=original_plan['caches']['dense_track']
        eyes=EyeCache(binding['path'],head)
        try:
            if (eyes.manifest_sha256!=binding['manifest_sha256'] or eyes.report['run'].get('recovery')!=binding['policy']
                    or binding['policy'].get('mode')!='dense_track' or binding['policy'].get('split')!='val'
                    or set(eyes.records)!={rid for rid,r in head.records.items() if r['split']=='val'}):
                raise ValueError('Dense tracking cache identity/scope mismatch')
            for path in [source.parent,eyes.root,*data['sources']]:
                path=Path(path).resolve()
                if output.is_relative_to(path) or path.is_relative_to(output):raise ValueError('Output overlaps input')
            plan=dict(schema_version='driveguard-eye-breakpoints-plan-v1',source_report=str(source),source_report_sha256=digest,
                backend=backend,device_index=device_index,provenance=data['provenance'],eye_cache=binding,
                event_policy=original_plan['event_policy'],training_runs=0)
            if output.exists():
                if read_json(output/'plan.json')!=plan:raise ValueError('Breakpoint resume plan mismatch')
            else:output.mkdir(parents=True);atomic_json(output/'plan.json',plan)
            if (output/'summary.json').exists():
                completed=read_json(output/'summary.json')
                if (completed.get('schema_version')!='driveguard-eye-breakpoints-v1' or completed.get('status')!='complete'
                        or completed.get('plan')!=plan or not completed.get('artifact_sha256')):
                    raise ValueError('Completed breakpoint report provenance mismatch')
                for name,checksum in completed['artifact_sha256'].items():
                    if file_sha256(_contained(output,name))!=checksum:raise ValueError('Completed breakpoint artifact checksum mismatch')
                return completed
            states={name:_state_hash(data[name].state_dict()) for name in ('model','eye_model')}
            resets={rid:r['reset'] for rid,r in labels.records.items()}
            baseline,actions=evaluate_events(head,data['model'],data['base'].rows['val'],resets,1000)
            _reproduce(baseline,data['selected'],'Original action')
            predictions=infer_eyes(head,eyes,data['eye_model'],baseline,'eye')
            combined,eye_only=compose_events(head,eyes,actions,predictions,baseline,resets,1000)
            for name,rows in [('combined',combined),('eye_only',eye_only)]:
                matches=[r for r in published['comparisons'] if r['variant']=='dense_track-'+name]
                if len(matches)!=1:raise ValueError('Recovery readout missing or ambiguous')
                _reproduce(rows,matches[0],'Dense track '+name)
            if frame_metrics(predictions,head,eyes,labels)!=published['frame_metrics']['dense_track']:
                raise ValueError('Dense track frame predictions differ')
            traces={rid:trace_record(head.arrays(rid),eyes.arrays(rid),scores,resets[rid],1000) for rid,scores in predictions.items()}
            write_rows(output/'frames.csv',[dict(record_id=rid,subject_id=head.records[rid]['subject_id'],**r)
                                           for rid,trace in traces.items() for r in trace])
            details=[];request_records=defaultdict(list);cards=[];support=Counter()
            for row,eye_row in zip(combined,eye_only):
                rid=row['record_id'];subtype=str(labels.records[rid]['subtypes'][row['end_slot']])
                detail=describe_window(row,subtype,traces[rid])
                if detail['eye_event']!=bool(eye_row['available'] and eye_row['logit']>=0):
                    raise ValueError('Trace event differs from production readout')
                detail['playback']=''
                if row['label_known'] and row['target']==1 and subtype=='closed_eye':
                    support['positive']+=1;support['duration_eligible']+=int(detail['input_duration_eligible'])
                    support['endpoint_available']+=int(detail['eye_endpoint_available'])
                    trace=traces[rid];pts=np.array([r['source_pts_ms'] for r in trace])
                    first=max(0,int(np.searchsorted(pts,max(0,detail['start_pts_ms']-1000))))
                    last=next(i for i,r in enumerate(trace) if r['slot_index']==row['end_slot'])
                    roi=crop_roi(head.records[rid]['source']['driver_roi'],head.report['crop'])
                    roi=[math.floor(roi[0]),math.floor(roi[1]),math.ceil(roi[2]),math.ceil(roi[3])]
                    detail['playback']='playback/'+write_playback(output/'playback',detail['window_id'],head.arrays(rid),
                        eyes.arrays(rid),trace,first,last,roi)
                    request_records[rid].append(dict(window_id=detail['window_id'],
                        start_pts_ms=trace[first]['source_pts_ms'],end_pts_ms=trace[last]['source_pts_ms']))
                    cards.append(f'<li><a href="{detail["playback"]}">subject {escape(str(row["subject_id"]))} / '
                                 f'{detail["start_pts_ms"]}–{detail["end_pts_ms"]}ms</a>: {detail["reason"]}</li>')
                details.append(detail)
            expected=published['input_support']['dense_track']['by_subtype']['closed_eye']
            if {k:support[k] for k in ('positive','duration_eligible','endpoint_available')}!=expected:
                raise ValueError('Trace support differs from published input audit')
            if any(_state_hash(data[name].state_dict())!=state for name,state in states.items()):
                raise ValueError('Frozen weights changed')
            if file_sha256(source)!=digest:raise ValueError('Source report changed')
            write_rows(output/'windows.csv',details)
            pure=[r for r in details if r['label_known'] and r['target']==1 and r['subtype']=='closed_eye']
            misses=[r for r in pure if r['false_negative']]
            write_rows(output/'pure-eye-windows.csv',pure)
            request=dict(schema_version='driveguard-eye-breakpoint-replay-request-v1',status='complete',
                head_cache=str(head.root),head_cache_sha256=head.manifest_sha256,eye_cache=str(eyes.root),
                eye_cache_sha256=eyes.manifest_sha256,source_report=str(source),source_report_sha256=digest,
                records=[dict(record_id=rid,windows=windows) for rid,windows in sorted(request_records.items())])
            atomic_json(output/'replay-request.json',request)
            report=dict(schema_version='driveguard-eye-breakpoints-v1',status='complete',phase='cached_inference',
                training_runs=0,reference_reproduced=True,frozen_weights_unchanged=True,plan=plan,
                windows=len(details),pure_eye_positive=len(pure),pure_eye_false_negative=len(misses),
                miss_reasons=dict(Counter(r['reason'] for r in misses)),
                by_subject={s:dict(positive=sum(r['subject_id']==s for r in pure),
                    false_negative=sum(r['subject_id']==s for r in misses),
                    reasons=dict(Counter(r['reason'] for r in misses if r['subject_id']==s))) for s in sorted({r['subject_id'] for r in pure})},
                pure_eye_windows=pure,source_replay_required=bool(request_records),
                notes=['Run times carry recording history; maximum seen run may start before the window.',
                       'Missing sample span is last minus first missing grid time; one missing sample has span zero.',
                       'Window outcome is evaluated at its endpoint, not any earlier event.',
                       'Stop reasons are unknown in existing caches: use verified source replay results.',
                       'Labels select diagnostic views only; inference retains all validation windows.'])
            (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>纯闭眼窗口断点</h1>'
                '<p>包含漏检及成功检出对照。全部验证窗口见 windows.csv；完整逐帧状态见 frames.csv。</p><ul>'+''.join(cards)+'</ul>',encoding='utf-8')
            report['artifact_sha256']={path.relative_to(output).as_posix():file_sha256(path)
                for path in sorted(output.rglob('*')) if path.is_file() and path.name not in ('plan.json','summary.json')}
            atomic_json(output/'summary.json',report)
            return report
        finally:eyes.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('recovery-report','dataset-root','output-dir'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--backend',choices=['cpu','cuda','npu'],default='cpu');parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--continue-run',action='store_true');args=vars(parser.parse_args(argv))
    try:
        report=run(**args);print(json.dumps({k:report[k] for k in ('status','training_runs','pure_eye_false_negative','miss_reasons')}));return 0
    except (OSError,ValueError,KeyError,TypeError,RuntimeError) as error:
        print(f'error: {error}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
