"""Replay verified body-fatigue windows for manual label and crop review.

This exports diagnostics, not a new decision rule or a quality claim. Labels are
attached to the WINDOW END; the sampled images do not carry frame-level labels.
"""
import argparse
from contextlib import ExitStack
from html import escape
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from training.common.config import read_json
from training.common.data import file_sha256
from training.visual.data import PreparedData
from training.visual.extract import _prepare, _stamp, write_json
from training.visual.fatigue import fatigue_metrics, summarize
from training.visual.report import evaluate, write_rows
from training.visual.spatial import crop_policy, crop_roi


SAMPLE_OFFSETS = (12, 49, 99)
LABEL_NOTE = ('Target is the fatigue label at the WINDOW END, not a label for each '
              'sampled image. Label semantics and head coverage require manual review.')


def _key(row):
    return row['subject_id'], row['record_id'], row['start_slot']


def select_windows(rows, negatives_per_subject=8):
    """All known positives plus high scoring and time-spread negatives per person."""
    selected = []
    for subject in sorted({r['subject_id'] for r in rows}):
        known = sorted((r for r in rows if r['subject_id'] == subject and r['label_known']), key=_key)
        selected.extend(r for r in known if r['target'] == 1)
        negative = [r for r in known if r['target'] == 0]
        high = sorted(negative, key=lambda r: (-(r['logit'] if r['available'] else -math.inf), _key(r)))
        chosen = high[:min(4, negatives_per_subject)]
        chosen_keys = {_key(r) for r in chosen}
        remaining = [r for r in negative if _key(r) not in chosen_keys]
        count = min(negatives_per_subject-len(chosen), len(remaining))
        if count:
            chosen.extend(remaining[i] for i in np.linspace(0, len(remaining)-1, count, dtype=int))
        selected.extend(chosen)
    return sorted(selected, key=_key)


def _score_summary(scores):
    return dict(count=len(scores), min=min(scores) if scores else None,
                median=float(np.median(scores)) if scores else None, max=max(scores) if scores else None)


def subject_summaries(rows):
    result = {}
    for subject in sorted({r['subject_id'] for r in rows}):
        group = [r for r in rows if r['subject_id'] == subject]
        positives, negatives = [[float(r['logit']) for r in group
            if r['label_known'] and r['available'] and r['target'] == target] for target in (1,0)]
        # Pairwise rank probability, half credit for ties; excludes unavailable scores.
        auc = (sum(sum(p > n for n in negatives)+.5*sum(p == n for n in negatives)
                   for p in positives)/(len(positives)*len(negatives)) if positives and negatives else None)
        result[subject] = dict(metrics=fatigue_metrics(group), positive_logits=_score_summary(positives),
                               negative_logits=_score_summary(negatives), ranking_auc_available=auc)
    return result


def normalization_summary(prepared, rows):
    """Fraction beyond +/-5 before clipping, counted per window (overlap repeats)."""
    result = {}
    for row_index, row in enumerate(rows):
        if not row['label_known']:
            continue
        group = result.setdefault(row['subject_id'], {k: dict(clipped=0, elements=0)
                                                      for k in ('visual', 'context')})
        arrays = prepared.cache.arrays(row['record_id'])
        start = row['visual_start']
        visual = arrays['embeddings'][start:start+8][arrays['valid'][start:start+8]]
        for name, values in (('visual', visual), ('context', prepared.contexts['val'][row_index])):
            state = prepared.normalizer[name]
            z = (values-np.asarray(state['mean']))/np.asarray(state['std'])
            group[name]['clipped'] += int((np.abs(z)>5).sum())
            group[name]['elements'] += int(z.size)
    for group in result.values():
        for counts in group.values():
            counts['fraction'] = counts['clipped']/counts['elements'] if counts['elements'] else None
    return result


def verify_window_end_labels(item, rows):
    """Compare prepared rows to the verified immutable fatigue labels at end_slot."""
    from training.common.config import TASKS
    task_index = TASKS.index('fatigue')
    if task_index != 0:
        raise ValueError('Original fatigue task index must be zero')
    slots = item['complete']['feature_record']['slots']
    counts = dict(windows=0,known=0,unknown=0)
    with ExitStack() as stack:
        arrays = {}
        for name in ('targets','label_masks'):
            value = np.load(item['attempt']/item['complete']['arrays'][name],mmap_mode='r',allow_pickle=False)
            stack.callback(value._mmap.close)
            if value.shape != (slots,len(TASKS)):
                raise ValueError('Original fatigue label array dimensions differ')
            arrays[name] = value
        for row in rows:
            start,end = row['start_slot'],row['end_slot']
            if (row['record_id'] != item['record']['record_id'] or type(start) != int or type(end) != int
                    or start < 0 or start % 25 or end != start+99 or end >= slots):
                raise ValueError('Prepared window bounds differ from original cache')
            known = bool(arrays['label_masks'][end,task_index])
            target = int(arrays['targets'][end,task_index]) if known else 0
            if type(row['label_known']) != bool or row['label_known'] != known or row['target'] != target:
                raise ValueError('Prepared window-end fatigue label differs from original immutable source: '
                                 f"{row['record_id']} end_slot={end}")
            counts['windows'] += 1
            counts['known' if known else 'unknown'] += 1
    return counts


def _thumbnail(bgr, roi, policy):
    """Full source with original ROI/crop outlines, beside the proposed crop."""
    rgb = Image.fromarray(np.ascontiguousarray(bgr[:,:,::-1]))
    rectangle = crop_roi(roi, policy)
    x0,y0,x1,y1 = rectangle
    cropped = rgb.crop((math.floor(x0),math.floor(y0),math.ceil(x1),math.ceil(y1)))
    draw = ImageDraw.Draw(rgb)
    draw.rectangle(roi, outline='#58c4ff', width=3)
    draw.rectangle(rectangle, outline='#ffcc33', width=3)
    canvas = Image.new('RGB',(640,180),'#161b22')
    canvas.paste(ImageOps.pad(rgb,(400,180),color='#161b22'),(0,0))
    canvas.paste(ImageOps.pad(cropped,(240,180),color='#161b22'),(400,0))
    buffer = BytesIO()
    canvas.save(buffer,format='JPEG',quality=85)
    return buffer.getvalue()


def record_frames(item, arrays, wanted_slots, policy):
    """Decode a record once, verify ALL PTS, and retain only small JPEG samples."""
    from inference.perception.video_source import decode_video, sample_frames
    from training.dmd.cache import validate_complete
    wanted = set(wanted_slots)
    result = {}
    with ExitStack() as stack:
        original = np.load(item['attempt']/item['complete']['arrays']['source_pts_ms'],
                           mmap_mode='r',allow_pickle=False)
        stack.callback(original._mmap.close)
        if _stamp(item['source_path']) != item['source_stamp']:
            raise ValueError('Source video changed before audit decode')
        frames = decode_video(item['source_path'])
        stack.callback(frames.close)
        def checked_frames():
            decoded = 0
            for frame in frames:
                if decoded >= len(original) or frame.timestamp_ms != int(original[decoded]):
                    raise ValueError('Decoded source PTS differs from original cache')
                if list(reversed(frame.bgr.shape[:2])) != item['source']['image_size']:
                    raise ValueError('Decoded dimensions differ from original cache')
                decoded += 1
                yield frame
            if decoded != len(original):
                raise ValueError('Decoded source frame count differs from original cache')
        checked = checked_frames()
        stack.callback(checked.close)
        sampled = sample_frames(checked)
        stack.callback(sampled.close)
        slots = visual_index = 0
        for slot_ms, frame in sampled:
            slot = slots
            if slot_ms != slot*40 or slot >= item['complete']['feature_record']['slots']:
                raise ValueError('Decoded slot grid differs from original cache')
            expected_index = int(np.searchsorted(original,slot_ms,side='right'))-1
            if expected_index < 0 or frame.timestamp_ms != int(original[expected_index]) or frame.timestamp_ms > slot_ms:
                raise ValueError('Sample source PTS differs from original causal cache')
            if slot % 25 in (12,24):
                if (visual_index >= len(arrays['slot_indices'])
                        or int(arrays['slot_indices'][visual_index]) != slot
                        or int(arrays['source_pts_ms'][visual_index]) != frame.timestamp_ms
                        or bool(arrays['valid'][visual_index]) != (slot_ms-frame.timestamp_ms <= 80)):
                    raise ValueError('Stored visual slot/PTS differs from decoded source')
                visual_index += 1
            if slot in wanted:
                result[slot] = dict(source_pts_ms=frame.timestamp_ms, age_ms=slot_ms-frame.timestamp_ms,
                                    jpeg=_thumbnail(frame.bgr,item['source']['driver_roi'],policy))
            slots += 1
        if (slots != item['complete']['feature_record']['slots']
                or visual_index != len(arrays['slot_indices']) or set(result) != wanted):
            raise ValueError('Decoded slot/sample count differs from original cache')
        if _stamp(item['source_path']) != item['source_stamp'] or file_sha256(item['source_path']) != item['source']['sha256']:
            raise ValueError('Source video changed during audit decode')
        if validate_complete(item['attempt'],item['complete']['fingerprint']) != item['complete']:
            raise ValueError('Original cache changed during audit decode')
    return result


def _outcome(row):
    if not row['available']:
        return 'FN-unavailable' if row['target'] else 'negative-unavailable'
    return ('TP' if row['logit'] >= 0 else 'FN') if row['target'] else ('FP' if row['logit'] >= 0 else 'TN')


def _render_sheet(output, row, frames, number):
    measure = ImageDraw.Draw(Image.new('RGB',(1,1)))
    lines = []
    for text in (f"Subject {row['subject_id']} | {_outcome(row)} | END target={row['target']}",
                 f"Record: {row['record_id']}",
                 f"Window slots {row['start_slot']}..{row['end_slot']}",
                 f"Window times {row['start_slot']*40}..{row['end_slot']*40} ms | logit={row['logit']}",
                 'Full source: blue=driver ROI, yellow=proposed crop | right=proposed crop'):
        line = ''
        for character in text:
            if measure.textlength(line+character) > 620:
                lines.append(line)
                line = ''
            line += character
        lines.append(line)
    header_height = 16+18*len(lines)
    sheet = Image.new('RGB',(640,header_height+639),'#ffffff')
    draw = ImageDraw.Draw(sheet)
    for index,line in enumerate(lines):
        draw.text((10,8+18*index),line,fill='black')
    for position, offset in enumerate(SAMPLE_OFFSETS):
        slot = row['start_slot']+offset
        sample = frames[slot]
        y = header_height+position*205
        with Image.open(BytesIO(sample['jpeg'])) as tile:
            sheet.paste(tile,(0,y))
        draw.text((10,y+184),f"slot={slot} ({slot*40} ms); source PTS offset={sample['source_pts_ms']} ms; "
                  f"age={sample['age_ms']} ms",fill='black')
    draw.text((10,header_height+618),'WINDOW END label only. Manual label/crop review pending.',fill='black')
    relative = f'images/window-{number:05d}.jpg'
    sheet.save(output/relative,quality=90)
    return relative


def _validate_comparison(report, comparison, prepared, rows):
    if prepared.task != 'fatigue' or report.get('prepared_sha256') != prepared.provenance['prepared_sha256']:
        raise ValueError('Fatigue preparation/checksum differs from comparison')
    current = summarize(rows)
    for key in ('window_label_hash', 'overall', 'by_subject'):
        if current[key] != comparison.get(key):
            raise ValueError('Re-evaluated fatigue '+key+' differs from comparison')
    return current


def run(*, fatigue_report, dataset_root, output_dir, variant='B_rgb-seed43-best',
        backend='cpu', device_index=0, batch_size=64, crop_xyxy=(.05,0.,.65,.65)):
    policy = crop_policy(crop_xyxy)
    source_report = Path(fatigue_report).resolve(strict=True)
    report_sha = file_sha256(source_report)
    report = read_json(source_report)
    if (report.get('schema_version') != 'driveguard-fatigue-visual-comparison-v1'
            or report.get('status') != 'complete' or report.get('task') != 'fatigue' or report.get('split') != 'val'):
        raise ValueError('Expected complete fatigue validation comparison')
    matches = [r for r in report['comparisons'] if r['variant'] == variant]
    if len(matches) != 1:
        raise ValueError('Expected exactly one fatigue comparison variant')
    comparison = matches[0]
    checkpoint = Path(comparison['checkpoint']).resolve(strict=True)
    if file_sha256(checkpoint) != comparison['checkpoint_sha256']:
        raise ValueError('Fatigue checkpoint checksum differs from comparison')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Fatigue audit requires a NEW output_dir')
    dataset = Path(dataset_root).resolve(strict=True)
    prepared = PreparedData(report['prepared_dir'])
    try:
        for source in (prepared.root, prepared.cache.root, Path(prepared.report['data_root']).resolve(), dataset):
            if output.is_relative_to(source) or source.is_relative_to(output):
                raise ValueError('Audit output overlaps immutable inputs')
        if type(batch_size) != int or batch_size < 1:
            raise ValueError('batch_size must be positive')
        rows, _ = evaluate(prepared.dataset('val'),checkpoint,backend,device_index,batch_size)
        current = _validate_comparison(report,comparison,prepared,rows)
        selected = select_windows(rows)
        manifest_path = Path(prepared.report['manifest']).resolve(strict=True)
        manifest_sha = file_sha256(manifest_path)
        if manifest_sha != prepared.cache.report['manifest_sha256']:
            raise ValueError('Original manifest checksum differs from visual cache')
        manifest = read_json(manifest_path)
        if manifest.get('schema_version') != 'driveguard-feature-manifest-v3':
            raise ValueError('Original immutable v3 manifest required')
        originals = {r['record_id']: r for r in manifest['records']}
        items = {}
        label_scope = dict(records=0,windows=0,known=0,unknown=0,
                           task='fatigue',task_index=0,split='val',
                           scope='All prepared windows, including unknowns, in every validation record with known labels')
        for rid in sorted({r['record_id'] for r in rows if r['label_known']}):
            item = _prepare(originals[rid],Path(prepared.report['data_root']).resolve(),dataset)
            if item['source'] != prepared.cache.records[rid]['source']:
                raise ValueError('Visual source identity differs from original cache')
            counts = verify_window_end_labels(item,[r for r in rows if r['record_id'] == rid])
            label_scope['records'] += 1
            for name,count in counts.items():
                label_scope[name] += count
            items[rid] = item
        identity = dict(fatigue_report=str(source_report),fatigue_report_sha256=report_sha,
                        variant=variant,prepared_dir=str(prepared.root),
                        prepared_sha256=prepared.provenance['prepared_sha256'],
                        checkpoint=str(checkpoint),checkpoint_sha256=comparison['checkpoint_sha256'],
                        window_label_hash=current['window_label_hash'],dataset_root=str(dataset),crop_policy=policy)
        by_subject = subject_summaries(rows)
        clipping = normalization_summary(prepared, rows)
        for subject, value in by_subject.items():
            value['normalization_clipping'] = clipping.get(subject)
        output.mkdir(parents=True)
        (output/'images').mkdir()
        write_json(output/'input.json', identity)
        exported = []
        for rid, item in items.items():
            record_rows = [r for r in selected if r['record_id'] == rid]
            if not record_rows:
                continue
            wanted = {r['start_slot']+offset for r in record_rows for offset in SAMPLE_OFFSETS}
            print(json.dumps(dict(stage='fatigue_audit_decode',record_id=rid,windows=len(record_rows))),flush=True)
            frames = record_frames(item,prepared.cache.arrays(rid),wanted,policy)
            for row in record_rows:
                image_path = _render_sheet(output,row,frames,len(exported)+1)
                exported.append(dict(subject_id=row['subject_id'],record_id=rid,start_slot=row['start_slot'],
                                     end_slot=row['end_slot'],target=row['target'],outcome=_outcome(row),image=image_path))
        images = {(r['record_id'],r['start_slot']):r['image'] for r in exported}
        csv_rows = [dict(row,start_ms=row['start_slot']*40,end_ms=row['end_slot']*40,
                         outcome=_outcome(row),contact_sheet=images.get((row['record_id'],row['start_slot']),''))
                    for row in sorted(rows,key=_key) if row['label_known']]
        write_rows(output/'windows.csv',csv_rows)
        cards = '\n'.join(f'<figure><figcaption>Subject {escape(r["subject_id"])} | '
            f'{escape(r["record_id"])} | slots {r["start_slot"]}–{r["end_slot"]} | '
            f'{r["outcome"]} | window END target {r["target"]}</figcaption>'
            f'<a href="{r["image"]}"><img loading="lazy" src="{r["image"]}" width="640"></a></figure>'
            for r in exported)
        html = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Fatigue window audit</title>'
                '<style>body{font:16px sans-serif;margin:24px;background:#edf0f4}figure{margin:24px 0}'
                'img{max-width:100%;height:auto}pre{white-space:pre-wrap}</style><h1>Fatigue window audit</h1>'
                f'<p>{escape(LABEL_NOTE)}</p><p>Fixed decision threshold: logit 0. '
                'Source PTS offsets are relative to the first decoded video frame. '
                'Every known positive and up to 8 deterministic negatives per subject are shown. '
                'Three causal RGB samples per window; no landmark inference.</p>'
                '<p><a href="windows.csv">All known validation windows</a> | '
                '<a href="summary.json">Verified inputs and per-subject summaries</a></p>'
                f'<pre>{escape(json.dumps(by_subject,indent=2))}</pre>{cards}</html>')
        (output/'index.html').write_text(html,encoding='utf-8')
        # Publication requires a second source/immutable-cache check after rendering.
        for rid, item in items.items():
            rechecked = _prepare(originals[rid],Path(prepared.report['data_root']).resolve(),dataset)
            if rechecked['source'] != item['source'] or rechecked['complete'] != item['complete']:
                raise ValueError('Source/cache changed before audit publication')
        for path, digest in ((source_report,report_sha),(checkpoint,comparison['checkpoint_sha256']),
                             (prepared.root/'prepared.json',identity['prepared_sha256']),
                             (manifest_path,manifest_sha),
                             (prepared.cache.root/'manifest.json',prepared.cache.manifest_sha256)):
            if file_sha256(path) != digest:
                raise ValueError('Audit input changed before publication')
        result = dict(schema_version='driveguard-fatigue-audit-v1',status='complete',**identity,
                      overall=current['overall'],by_subject=by_subject,contact_sheets=exported,
                      known_windows=len(csv_rows),positive_contact_sheets=sum(r['target'] == 1 for r in exported),
                      sampled_offsets=list(SAMPLE_OFFSETS),
                      original_label_check_scope=label_scope,
                      mechanical_checks=dict(checkpoint_prepared_cohort_metrics_match=True,
                          original_window_end_fatigue_labels_match=True,
                          source_hashes_full_pts_dimensions_causal_visual_samples_match=True),
                      manual_review=dict(status='pending',label_semantics_verified=False,head_coverage_verified=False),
                      notes=[LABEL_NOTE,'AUC uses available known scores and gives ties half credit.',
                          'Normalization clip fractions count repeated window samples; visual excludes stale samples.',
                          'Negative selection: up to four highest scores plus time-spread remaining negatives per subject.',
                          'No per-person threshold, label edits, held-out metrics, or automatic head-coverage claim.'])
        result['artifact_sha256'] = {p.relative_to(output).as_posix():file_sha256(p)
                                     for p in sorted(output.rglob('*')) if p.is_file()}
        write_json(output/'summary.json',result)
        return result
    finally:
        prepared.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fatigue-report','dataset-root','output-dir'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--variant',default='B_rgb-seed43-best')
    parser.add_argument('--backend',default='cpu',choices=('cpu','cuda','npu'))
    parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--crop-xyxy',nargs=4,type=float,default=(.05,0.,.65,.65))
    try:
        result = run(**vars(parser.parse_args(argv)))
        print(json.dumps(dict(status=result['status'],known_windows=result['known_windows'])),flush=True)
    except (OSError,ValueError,RuntimeError,ImportError) as error:
        parser.exit(2,f'error: {error}\n')


if __name__ == '__main__':
    main()
