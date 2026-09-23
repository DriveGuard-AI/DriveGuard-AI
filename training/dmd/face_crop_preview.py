"""Create a portable, single-video crop picker without running perception models."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

from training.common.config import read_json
from training.dmd.face_experiment_config import BASELINE


def _save_frame(path, frame):
    import cv2
    if not cv2.imwrite(str(path), frame.bgr):
        raise OSError('Could not write crop preview image')


def write_picker(output, report):
    template=Path(__file__).with_name('face_crop_picker.html').read_text(encoding='utf-8')
    data=json.dumps(dict(report=report,baseline=BASELINE),ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    (output/'index.html').write_text(template.replace('__CROP_DATA__',data),encoding='utf-8')


def preview_crop(*,manifest,data_root,dataset_root,output_dir,record_id,samples=12):
    from inference.experts.assets import file_sha256
    from inference.perception.video_source import decode_video,sample_frames
    from training.dmd.cache import write_json
    from training.dmd.face_diagnostics import _prepare,_stamp

    if type(samples) is not int or not 2<=samples<=48:
        raise ValueError('samples must be an integer between 2 and 48')
    output=Path(output_dir).resolve()
    if output.exists(): raise FileExistsError('Choose a NEW output directory')
    root,dataset=Path(data_root).resolve(strict=True),Path(dataset_root).resolve(strict=True)
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (root,dataset)):
        raise ValueError('Output must be separate from original cache and dataset')
    manifest=Path(manifest).resolve(strict=True)
    manifest_stamp=_stamp(manifest); manifest_hash=file_sha256(manifest)
    data=read_json(manifest)
    if data.get('schema_version')!='driveguard-feature-manifest-v3':
        raise ValueError('Original v3 feature manifest required')
    selected=[r for r in data['records'] if r['record_id']==record_id]
    if len(selected)!=1 or selected[0]['view']!='body':
        raise ValueError('Select exactly one body record from the manifest')
    item=_prepare(selected[0],root,dataset)
    feature=item['complete']['feature_record']; expected=feature['slots']
    stamps={p:_stamp(p) for p in item['attempt'].rglob('*') if p.is_file()}
    stamps.update({manifest:manifest_stamp,item['source']:item['source_stamp']})
    def unchanged():
        if any(_stamp(p)!=s for p,s in stamps.items()):
            raise ValueError('Original manifest/cache/video changed during preview')
    unchanged()
    # Include both ends; source PTS is recorded separately from requested slot.
    indices=sorted({round(i*(expected-1)/(samples-1)) for i in range(samples)})
    wanted=set(indices)
    output.mkdir(parents=True,exist_ok=False)
    images=[]; clock={}; count=0
    with ExitStack() as stack:
        frames=decode_video(item['source'],clock); stack.callback(frames.close)
        slots=sample_frames(frames); stack.callback(slots.close)
        for index,(slot,frame) in enumerate(slots):
            if index>=expected or list(reversed(frame.bgr.shape[:2]))!=feature['image_size']:
                raise ValueError('Preview decode differs from original slot count/image size')
            if index in wanted:
                filename=f'sample-{len(images)+1:03d}.jpg'
                _save_frame(output/filename,frame)
                images.append(dict(file=filename,slot_index=index,slot_ms=slot,source_pts_ms=frame.timestamp_ms))
            count+=1
    if count!=expected or len(images)!=len(indices):
        raise ValueError('Preview decode does not cover original cached duration')
    unchanged()
    scope=dict(record_id=record_id,video_sha256=item['video_sha256'],image_size=feature['image_size'])
    report=dict(schema_version='driveguard-face-crop-preview-v1',status='complete',crop_scope=scope,
                manifest_sha256=manifest_hash,slots=count,duration_grid_seconds=count/25,
                samples=images,clock=clock,
                notes=['Uniform samples do not prove all head motion is covered. Inspect full replay before using results.',
                       'No crop is guessed or approved by this tool. Select and review in index.html.',
                       'No perception model was run and no input files were changed.'])
    write_picker(output,report)
    write_json(output/'summary.pending.json',report)
    unchanged()
    (output/'summary.pending.json').replace(output/'summary.json')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','data-root','dataset-root','output-dir','record-id'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--samples',type=int,default=12)
    args=vars(parser.parse_args(argv))
    try:
        report=preview_crop(**args)
        print(json.dumps(dict(status='complete',samples=len(report['samples']),index=str(Path(args['output_dir']).resolve()/'index.html'))))
        return 0
    except (ValueError,OSError,RuntimeError,ImportError,KeyError,TypeError) as exc:
        print(f'Crop preview failed: {exc}',file=sys.stderr); return 2


if __name__=='__main__': raise SystemExit(main())
