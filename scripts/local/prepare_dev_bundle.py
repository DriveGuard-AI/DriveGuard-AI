"""Create a local development copy of a released live-composite bundle.

A released bundle pins the exact source tree (``code_identity``) and the
accelerator that was used when it was exported. A workstation that only has a
CPU, or that carries commits newer than the release, therefore cannot load it.
This tool writes a NEW directory (normally under ``runs/``) containing:

* the same weight and asset files, hardlinked when the filesystem allows it and
  never modified in place;
* ``runtime.json`` with an explicit local backend and detector endpoint;
* ``bundle.json`` with the current ``code_identity`` and the updated
  ``runtime.json`` digest.

The release directory is never touched and an existing output is refused, so a
stale development bundle cannot be mistaken for a fresh one. The result is a
development artifact: it keeps the candidate qualification of the release and
must not be published or described as validated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

# Running this file directly puts scripts/local on sys.path, not the repository
# root, so make the package imports below independent of the launch directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            value.update(chunk)
    return value.hexdigest()


def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    if digest(target) != digest(source):
        raise ValueError('Development copy differs from source: ' + target.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='models/releases/live-composite-v1')
    parser.add_argument('--output', required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda', 'npu'), default='cpu')
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--detector-host', default='127.0.0.1')
    parser.add_argument('--detector-port', type=int, default=18761)
    parser.add_argument('--detector-token-file', required=True)
    args = parser.parse_args(argv)

    from inference.live.bundle import code_identity, load_bundle

    source = Path(args.source).resolve(strict=True)
    output = Path(args.output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise SystemExit('error: --output must be outside the release directory')
    if output.exists():
        raise SystemExit(f'error: output already exists: {output}; choose a new directory')
    token_file = Path(args.detector_token_file).resolve()
    if not token_file.is_file():
        raise SystemExit(f'error: detector token file not found: {token_file}')
    if args.backend == 'cpu' and args.device_index != 0:
        raise SystemExit('error: cpu requires --device-index 0')

    report = json.loads((source / 'bundle.json').read_text(encoding='utf-8'))
    runtime_name = report['runtime']
    runtime_entry = report['files'].get(runtime_name)
    if runtime_entry is None:
        raise SystemExit(f'error: bundle does not record {runtime_name}')
    for name, entry in report['files'].items():
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise SystemExit(f'error: bundle entry escapes the release: {name}')
        payload = source / relative
        if not payload.is_file() or digest(payload) != entry['sha256']:
            raise SystemExit(f'error: release file missing or tampered: {entry["path"]}')
        # runtime.json is rewritten below, and a hard link would share the inode
        # with the release, so writing the copy would corrupt the release.
        if entry['path'] == runtime_name:
            continue
        link_or_copy(payload, output / relative)

    runtime_path = source / runtime_name
    runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
    runtime['objects']['backend'] = args.backend
    runtime['objects']['device_index'] = args.device_index
    runtime['objects']['service'] = {'host': args.detector_host, 'port': args.detector_port,
                                     'token_file': str(token_file)}
    written = output / runtime_name
    written.write_text(json.dumps(runtime, indent=2) + '\n', encoding='utf-8')
    if digest(runtime_path) != runtime_entry['sha256']:
        raise SystemExit('error: the release runtime.json was modified while preparing the copy')
    report['files'][runtime_name] = {'path': runtime_name, 'sha256': digest(written)}

    # The pinned source identity is what a release update would refresh; the
    # development copy records the checkout it will actually run against.
    report['code_identity'] = code_identity()
    report['development_copy'] = {'backend': args.backend, 'device_index': args.device_index,
                                  'detector_endpoint': f'{args.detector_host}:{args.detector_port}'}
    (output / 'bundle.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    load_bundle(output / 'bundle.json')
    print(json.dumps({'bundle': str(output / 'bundle.json'), 'backend': args.backend,
                      'detector': f'{args.detector_host}:{args.detector_port}',
                      'qualification': report.get('qualification')}, sort_keys=True))


if __name__ == '__main__':
    main()
