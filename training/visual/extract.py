"""Bounded frozen RGB extraction from immutable DMD v3 records.

Only complete record markers can be resumed. Attempts and published manifests
are immutable; a failed/interrupted record always gets a new attempt directory.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from uuid import uuid4


def write_json(path,value):
    """Publish a complete JSON file exclusively; interruption cannot truncate a marker."""
    path=Path(path)
    encoded=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    descriptor,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.partial',dir=path.parent)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8',newline='\n') as handle:handle.write(encoded)
        os.link(name,path)
    finally:Path(name).unlink(missing_ok=True)


def selected_slots(slots):
    import numpy as np
    if type(slots) is not int or slots < 1:
        raise ValueError('Invalid original slot count')
    return np.array([n for n in range(slots) if n % 25 in (12,24)],dtype=np.int64)


def _contained(root, relative):
    from training.dmd.cache import _unlinked
    if not isinstance(relative,str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Expected contained relative path')
    path = _unlinked(root/relative)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Path outside root or not a file')
    return path


def _stamp(path):
    stat = path.stat()
    return stat.st_size,stat.st_mtime_ns,stat.st_ino


def _prepare(record, root, dataset):
    """Verify original cache and source; RGB needs no landmark model assets."""
    from inference.experts.assets import file_sha256
    from training.common.config import canonical_hash, read_json
    from training.dmd.cache import METADATA, validate_complete
    values = _contained(root,record['values'])
    attempt = values.parent.parent
    inputs = read_json(_contained(attempt,'input.json'))
    complete = validate_complete(attempt,canonical_hash(inputs))
    if complete is None:
        raise ValueError('Invalid original cache: checksums, provenance or dimensions')
    job,feature = complete['job'],complete['feature_record']
    if record['record_id'] != job['job_id'] or any(record[k] != job[k] for k in METADATA):
        raise ValueError('Manifest metadata differs from original cache')
    for name,relative in complete['arrays'].items():
        if name != 'source_pts_ms' and _contained(root,record[name]) != _contained(attempt,relative):
            raise ValueError('Manifest arrays differ from original cache')
    if feature['schema_version'] != 'driveguard-feature-record-v3':
        raise ValueError('Original immutable v3 feature cache required')
    provenance = feature['provenance']
    config = provenance['config']
    if (config['schema_version'] != 'driveguard-runtime-v2' or config['driver_roi'] != feature['driver_roi']
            or any(config[k] != job[k] for k in ('recording_id','view'))):
        raise ValueError('Original runtime config differs from cache')
    source = _contained(dataset,job['video'])
    stamp = _stamp(source)
    digest = file_sha256(source)
    if stamp != _stamp(source):
        raise ValueError('Source video changed during verification')
    if digest != inputs.get('video_sha256') or digest != provenance['source_video']['sha256']:
        raise ValueError('Source video checksum differs from original cache')
    if stamp[0] != provenance['source_video']['size_bytes']:
        raise ValueError('Source video size differs from original cache')
    return dict(record=record,complete=complete,attempt=attempt,source_path=source,source_stamp=stamp,
                source=dict(video=job['video'],sha256=digest,size_bytes=stamp[0],
                    cache_fingerprint=complete['fingerprint'],driver_roi=feature['driver_roi'],
                    image_size=feature['image_size']))


def _identity(item, run):
    from training.common.config import canonical_hash
    from training.dmd.cache import METADATA
    result = {k:item['record'][k] for k in ('record_id',*METADATA)}
    result.update(slots=item['complete']['feature_record']['slots'],source=item['source'])
    result['fingerprint'] = canonical_hash(dict(run=run,record=result))
    return result


def _validate_record(output, row, identity, item):
    import numpy as np
    from inference.experts.assets import file_sha256
    from training.common.config import read_json
    if (not isinstance(row,dict) or any(row.get(key) != value for key,value in identity.items())
            or row.get('status') != 'complete'):
        raise ValueError('Record configuration/provenance mismatch')
    marker = _contained(output,row['directory']+'/record.json')
    if marker.parent.joinpath('failure.json').exists() or read_json(marker) != row:
        raise ValueError('Invalid record completion marker')
    indices = selected_slots(identity['slots'])
    count = len(indices)
    keys = {'embeddings','slot_indices','source_pts_ms','valid'}
    if row['sample_count'] != count or set(row['arrays']) != keys or set(row['sha256']) != keys:
        raise ValueError('Invalid visual array layout')
    expected = {'embeddings':((count,512),np.float32),'slot_indices':((count,),np.int64),
                'source_pts_ms':((count,),np.int64),'valid':((count,),np.bool_)}
    with ExitStack() as stack:
        arrays = {}
        for name,(shape,dtype) in expected.items():
            if row['arrays'][name] != row['directory']+'/'+name+'.npy':
                raise ValueError('Unexpected relative visual path')
            path = _contained(output,row['arrays'][name])
            if file_sha256(path) != row['sha256'][name]:
                raise ValueError('Visual array checksum mismatch')
            array = np.load(path,mmap_mode='r',allow_pickle=False)
            stack.callback(array._mmap.close)
            if array.shape != shape or array.dtype != dtype:
                raise ValueError('Invalid visual array dimensions or dtype')
            arrays[name] = array
        original = np.load(item['attempt']/item['complete']['arrays']['source_pts_ms'],mmap_mode='r',allow_pickle=False)
        stack.callback(original._mmap.close)
        for start in range(0,count,512):
            selected = indices[start:start+512]
            pts = original[np.searchsorted(original,selected*40,side='right')-1]
            valid = selected*40-pts <= 80
            if (not np.array_equal(arrays['slot_indices'][start:start+512],selected)
                    or not np.array_equal(arrays['source_pts_ms'][start:start+512],pts)
                    or not np.array_equal(arrays['valid'][start:start+512],valid)):
                raise ValueError('Visual sampling or PTS differs from original cache')
            features = arrays['embeddings'][start:start+512]
            if not np.isfinite(features).all() or np.any(features[~valid] != 0):
                raise ValueError('Invalid visual feature values')
    return row


def _extract_record(output, directory, item, identity, encoder, batch_size):
    import numpy as np
    from inference.experts.assets import file_sha256
    from inference.perception.video_source import decode_video, sample_frames
    from training.dmd.cache import validate_complete
    from training.visual.encoder import preprocess
    indices = selected_slots(identity['slots'])
    count = len(indices)
    arrays = {}
    try:
        with ExitStack() as stack:
            for name,shape,dtype in (('embeddings',(count,512),np.float32),('slot_indices',(count,),np.int64),
                                    ('source_pts_ms',(count,),np.int64),('valid',(count,),np.bool_)):
                array = np.lib.format.open_memmap(directory/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
                stack.callback(array._mmap.close)
                arrays[name] = array
            arrays['slot_indices'][:] = indices
            arrays['embeddings'][:] = 0
            original = np.load(item['attempt']/item['complete']['arrays']['source_pts_ms'],mmap_mode='r',allow_pickle=False)
            stack.callback(original._mmap.close)
            if _stamp(item['source_path']) != item['source_stamp']:
                raise ValueError('Source video changed after verification')
            frames = decode_video(item['source_path'])
            stack.callback(frames.close)
            def checked_frames():
                decoded = 0
                for frame in frames:
                    if decoded >= len(original) or frame.timestamp_ms != int(original[decoded]):
                        raise ValueError('Decoded source PTS differs from original cache')
                    if list(reversed(frame.bgr.shape[:2])) != item['source']['image_size']:
                        raise ValueError('Decoded image dimensions differ from original cache')
                    decoded += 1
                    yield frame
                if decoded != len(original):
                    raise ValueError('Decoded source frame count differs from original cache')
            checked = checked_frames()
            stack.callback(checked.close)
            samples = sample_frames(checked)
            stack.callback(samples.close)
            pending,positions = [],[]
            def flush():
                if pending:
                    features = encoder.encode(np.stack(pending))
                    if features.shape != (len(pending),512) or features.dtype != np.float32 or not np.isfinite(features).all():
                        raise ValueError('Invalid encoder output')
                    arrays['embeddings'][positions] = features
                    pending.clear(); positions.clear()
            total = selected = 0
            for slot,frame in samples:
                if total >= identity['slots'] or slot != total*40:
                    raise ValueError('Decoded slot count/grid differs from original cache')
                pts_index = int(np.searchsorted(original,slot,side='right'))-1
                if pts_index < 0 or frame.timestamp_ms != int(original[pts_index]) or frame.timestamp_ms > slot:
                    raise ValueError('Slot PTS differs from original causal cache')
                if total % 25 in (12,24):
                    fresh = slot-frame.timestamp_ms <= 80
                    arrays['source_pts_ms'][selected] = frame.timestamp_ms
                    arrays['valid'][selected] = fresh
                    if fresh:
                        pending.append(preprocess(frame.bgr,item['source']['driver_roi']))
                        positions.append(selected)
                        if len(pending) == batch_size:
                            flush()
                    selected += 1
                total += 1
            if total != identity['slots'] or selected != count:
                raise ValueError('Decoded slot count differs from original cache')
            flush()
            if _stamp(item['source_path']) != item['source_stamp']:
                raise ValueError('Source video changed during extraction')
            if validate_complete(item['attempt'],item['complete']['fingerprint']) != item['complete']:
                raise ValueError('Original cache changed during extraction')
            for array in arrays.values(): array.flush()
        relative = directory.relative_to(output).as_posix()
        paths = {name:relative+'/'+name+'.npy' for name in arrays}
        row = dict(identity,status='complete',sample_count=count,directory=relative,arrays=paths,
                   sha256={name:file_sha256(output/path) for name,path in paths.items()})
        write_json(directory/'record.json',row)
        return row
    except BaseException as error:
        write_json(directory/'failure.json',dict(status='failed',error=str(error),error_type=type(error).__name__))
        raise


def extract(manifest, data_root, dataset_root, weights, weights_sha256, output_dir,
            backend='cpu', device_index=0, batch_size=32, continue_run=False):
    from inference.experts.assets import file_sha256
    from training.common.config import canonical_hash, read_json
    from training.dmd.cache import _unlinked
    from training.visual.encoder import FrozenEncoder, encoder_metadata
    _validate_options(weights_sha256,backend,device_index,batch_size)
    root,dataset = _unlinked(data_root),_unlinked(dataset_root)
    manifest = _unlinked(manifest)
    output = Path(output_dir).absolute()
    if any(output.resolve().is_relative_to(p) or p.is_relative_to(output.resolve()) for p in (root,dataset)):
        raise ValueError('Output must be separate from original cache and dataset')
    if output.exists() and not continue_run:
        raise FileExistsError('Output exists; explicit continue_run required')
    manifest_stamp = _stamp(manifest)
    data = read_json(manifest)
    digest = file_sha256(manifest)
    if _stamp(manifest) != manifest_stamp:
        raise ValueError('Manifest changed during verification')
    if data['schema_version'] != 'driveguard-feature-manifest-v3':
        raise ValueError('Original immutable v3 manifest required')
    records = [row for row in data['records'] if row['view'] == 'body' and row['split'] in ('train','val','test')]
    if not records or len({r['record_id'] for r in records}) != len(records):
        raise ValueError('No body records or duplicate record identities')
    run = dict(schema_version='driveguard-visual-extract-v1',manifest_sha256=digest,
               encoder=encoder_metadata(weights_sha256),backend=backend,device_index=device_index,batch_size=batch_size)
    if output.exists():
        output = _unlinked(output)
        if read_json(_contained(output,'input.json')) != run:
            raise ValueError('Resume configuration/provenance mismatch')
    items = [_prepare(record,root,dataset) for record in records]
    if output.exists() and (output/'manifest.json').exists():
        report = read_json(_contained(output,'manifest.json'))
        if (report.get('schema_version') != 'driveguard-visual-cache-v1' or report.get('status') != 'complete'
                or report.get('manifest_sha256') != digest or report.get('encoder') != run['encoder']
                or len(report.get('records',[])) != len(items)):
            raise ValueError('Invalid completed visual manifest')
        try:
            for item,row in zip(items,report['records']):
                _validate_record(output,row,_identity(item,run),item)
        except (OSError,ValueError,KeyError,TypeError,EOFError) as error:
            raise ValueError('Invalid completed visual cache: '+str(error)) from error
        return report
    encoder = FrozenEncoder(weights,weights_sha256,backend=backend,device_index=device_index)
    if not output.exists():
        output.mkdir(parents=True)
        output = _unlinked(output)
        write_json(output/'input.json',run)
    completed = []
    for item in items:
        identity = _identity(item,run)
        record_dir = output/('record-'+canonical_hash(identity['record_id']))
        record_dir.mkdir(exist_ok=True)
        _unlinked(record_dir)
        reusable = None
        for marker in sorted(record_dir.glob('attempt-*/record.json')):
            try:
                row = read_json(_unlinked(marker))
                reusable = _validate_record(output,row,identity,item)
                break
            except (OSError,ValueError,KeyError,TypeError,EOFError):
                continue
        if reusable is None:
            print('Extracting RGB: '+identity['record_id'],file=sys.stderr,flush=True)
            directory = record_dir/('attempt-'+uuid4().hex)
            directory.mkdir()
            reusable = _extract_record(output,directory,item,identity,encoder,batch_size)
        completed.append(reusable)
    if _stamp(manifest) != manifest_stamp or file_sha256(manifest) != digest:
        raise ValueError('Original manifest changed during extraction')
    report = dict(schema_version='driveguard-visual-cache-v1',status='complete',manifest_sha256=digest,
                  encoder=run['encoder'],records=completed)
    write_json(output/'manifest.json',report)
    return report


def _validate_options(weights_sha256,backend,device_index,batch_size):
    if not isinstance(weights_sha256,str) or re.fullmatch('[0-9a-f]{64}',weights_sha256) is None:
        raise ValueError('Expected lowercase weights SHA256')
    if backend not in ('cpu','cuda','npu') or type(device_index) is not int or device_index < 0 or (backend == 'cpu' and device_index != 0):
        raise ValueError('Invalid backend/device index')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','data-root','dataset-root','weights','weights-sha256','output-dir'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--backend',default='cpu',choices=('cpu','cuda','npu'))
    parser.add_argument('--device-index',type=int,default=0)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--continue-run',action='store_true')
    parser.add_argument('--check-config',action='store_true')
    options = vars(parser.parse_args(argv))
    check = options.pop('check_config')
    _validate_options(options['weights_sha256'],options['backend'],options['device_index'],options['batch_size'])
    if check:
        print(json.dumps(dict(status='valid_options',**options)))
        return
    result = extract(**options)
    print(json.dumps(dict(status=result['status'],records=len(result['records']))))


if __name__ == '__main__': main()
