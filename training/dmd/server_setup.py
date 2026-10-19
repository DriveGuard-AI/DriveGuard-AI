"""Server-only asset retrieval and non-destructive configuration migration."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import secrets
import sys
import urllib.request

from inference.experts.assets import file_sha256
from training.common.config import read_json
from training.dmd.batch_config import _relative
from training.dmd.gaze_batch import load_batch

SOURCE = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt'


def download_weight(destination):
    """TLS official-source download, locally computed receipt (not vendor signature)."""
    destination = Path(destination).resolve()
    receipt = destination.with_suffix('.receipt.json')
    if destination.exists():
        if not receipt.exists():
            raise ValueError('Existing weight has no trusted download receipt; use a new destination')
        previous = read_json(receipt)
        if previous.get('source') != SOURCE or previous.get('sha256') != file_sha256(destination):
            raise ValueError('Existing weight/receipt mismatch; never overwrite')
        return
    if receipt.exists(): raise FileExistsError(receipt)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix('.download')
    # An interrupted download requires operator inspection; do not overwrite it.
    with partial.open('xb') as handle, urllib.request.urlopen(SOURCE, timeout=120) as response:
        total = 0
        while block := response.read(1024*1024):
            total += len(block)
            if total > 100*1024*1024: raise ValueError('YOLO11s asset exceeds 100 MiB download cap')
            handle.write(block)
    if total < 1024*1024: raise ValueError('Downloaded asset unexpectedly small')
    # Exclusive final target; no overwrite if another operator created it.
    with destination.open('xb') as target, partial.open('rb') as source:
        import shutil
        shutil.copyfileobj(source, target)
    partial.unlink()
    _write(receipt, dict(source=SOURCE, sha256=file_sha256(destination), license='AGPL-3.0',
                         verification='local SHA256 after HTTPS download; not upstream signature'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def configure(batch_path, output_path, weights_path, port=18761):
    loaded = load_batch(batch_path)
    output = Path(output_path).resolve()
    runtime_path = output.with_name(output.stem+'.runtime.json')
    token = output.with_name(output.stem+'.token')
    for path in (output, runtime_path, token):
        if path.exists(): raise FileExistsError(path)
    if type(port) is not int or not 1024 <= port <= 65535: raise ValueError('Invalid service port')
    weights = Path(weights_path).resolve(strict=True)
    if not weights.is_file() or weights.stat().st_size == 0: raise ValueError('Missing local model asset')
    runtime = deepcopy(loaded['runtime'])
    runtime['objects'].update(variant='yolo11s', backend='npu', device_index=0, input_size=640,
        asset=dict(path=str(weights), sha256=file_sha256(weights), source=SOURCE, license='AGPL-3.0'),
        service=dict(host='127.0.0.1', port=port, token_file=str(token)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(token, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), 'w') as handle:
        handle.write(secrets.token_hex(32)+'\n')
    _write(runtime_path, runtime)
    batch = deepcopy(loaded['config'])
    source_parent = Path(batch_path).resolve().parent
    batch['preparation'] = _relative((source_parent/batch['preparation']).resolve(), output.parent)
    batch['runtime_config'] = _relative(runtime_path, output.parent)
    _write(output, batch)
    return dict(config=str(output), runtime=str(runtime_path), jobs=len(batch['jobs']),
                confirmed=sum(j['confirmed_exclusive'] for j in batch['jobs'].values()))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--weights',default='models/local/yolo11s.pt')
    parser.add_argument('--port',type=int,default=18761)
    args=parser.parse_args(argv)
    try:
        if sys.platform!='linux': raise ValueError('Setup runs on Linux server only; no local installation/download')
        if Path(args.output).exists(): raise FileExistsError(args.output)
        # Parse source before doing any download.
        load_batch(args.config)
        download_weight(args.weights)
        print(json.dumps(configure(args.config,args.output,args.weights,args.port),ensure_ascii=False))
        return 0
    except (OSError,ValueError,KeyError,TypeError) as error:
        print(f'Server setup failed: {error}',file=sys.stderr)
        return 2


if __name__=='__main__': raise SystemExit(main())
