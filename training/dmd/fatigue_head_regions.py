"""Prepare per-video head-region pickers and assemble a three-arm fatigue experiment."""
import argparse
from html import escape
import json
from pathlib import Path
import sys

from training.common.config import read_json
from training.dmd.face_experiment_config import BASELINE, load_config, validate_config


def build(*, crop_configs, output):
    """Use explicitly selected rectangles only; never infer or approve a region."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a NEW experiment configuration path')
    regions = []
    for path in crop_configs:
        config = load_config(path)
        crops = [v['crop_xyxy'] for v in config['variants'] if v['crop_xyxy'] is not None]
        if config['schema_version'] != 'driveguard-face-experiments-v1' or 'crop_scope' not in config or len(crops) != 1:
            raise ValueError('Each picker export must contain one crop and its record/video/image-size scope')
        regions.append(dict(config['crop_scope'], head_roi_xyxy=crops[0]))
    baseline = dict(BASELINE, head_region_mode='none')
    config = validate_config(dict(schema_version='driveguard-face-experiments-v2', head_regions=regions,
        variants=[baseline, dict(baseline, name='head_gate', head_region_mode='gate'),
                  dict(baseline, name='head_crop', head_region_mode='crop')]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return config


def prepare(*, recovery_summary, dataset_root, output_dir, samples=24, record_ids=None):
    from inference.experts.assets import file_sha256
    from training.dmd.face_crop_preview import preview_crop
    from training.dmd.cache import write_json

    source = Path(recovery_summary).resolve(strict=True)
    digest = file_sha256(source)
    previous = read_json(source)
    if (previous.get('schema_version') != 'driveguard-fatigue-recovery-report-v1' or
            previous.get('status') != 'complete' or previous.get('partial') is not False):
        raise ValueError('A complete nonpartial fatigue recovery report is required')
    rows = [r for r in previous['records'] if r['variant']['name'] == 'baseline' and r['view'] == 'body']
    if record_ids is None:
        rows = [r for r in rows if r['fatigue_evidence']['windows']['known_positive']['windows'] > 0]
    else:
        rows = [r for r in rows if r['record_id'] in record_ids]
    ids = [r['record_id'] for r in rows]
    if not ids or len(set(ids)) != len(ids) or (record_ids is not None and set(record_ids) != set(ids)):
        raise ValueError('No positive records, duplicate records, or unmatched explicit record IDs')
    if any(r['fatigue_evidence']['baseline_fatigue_reproduced'] is not True for r in rows):
        raise ValueError('Selected records must have reproduced the original baseline fatigue evidence')
    settings = previous['fatigue_settings']
    diagnostic = Path(settings['diagnostic_summary']).resolve(strict=True)
    if file_sha256(diagnostic) != settings['diagnostic_sha256']:
        raise ValueError('Original fatigue diagnostic checksum changed')
    saved = read_json(diagnostic)
    manifest = Path(previous['manifest']).resolve(strict=True)
    manifest_digest = file_sha256(manifest)
    if previous.get('manifest_sha256') and manifest_digest != previous['manifest_sha256']:
        raise ValueError('Original manifest checksum changed')
    root, dataset = Path(saved['data_root']).resolve(strict=True), Path(dataset_root).resolve(strict=True)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a NEW preview output directory')
    if (any(output.is_relative_to(p) or p.is_relative_to(output) for p in (root, dataset)) or
            any(p.is_relative_to(output) for p in (source, diagnostic, manifest))):
        raise ValueError('Preview output overlaps original inputs')
    if type(samples) is not int or not 2 <= samples <= 48:
        raise ValueError('samples must be an integer between 2 and 48')
    output.mkdir(parents=True, exist_ok=False)
    previews = []
    links = ['<!doctype html><meta charset="utf-8"><title>疲劳头部区域选择</title>',
             '<h1>逐视频选择头部活动范围</h1><p>覆盖低头、前倾、转头并留出余量。'
             '结合完整原视频检查；均匀抽样不能证明覆盖全部动作。每个页面导出一个配置，'
             '按记录分别保存，然后用 build 合并。仅选择区域，不自动确认人脸或眼部准确。</p><ul>']
    for index, record_id in enumerate(ids, 1):
        directory = f'record-{index:03d}'
        result = preview_crop(manifest=manifest, data_root=root, dataset_root=dataset,
                              output_dir=output/directory, record_id=record_id, samples=samples)
        previews.append(dict(directory=directory, crop_scope=result['crop_scope']))
        links.append(f'<li><a href="{directory}/index.html">{escape(record_id)}</a></li>')
    if (file_sha256(source) != digest or file_sha256(diagnostic) != settings['diagnostic_sha256'] or
            file_sha256(manifest) != manifest_digest):
        raise ValueError('Source reports changed while preparing previews')
    links.append('</ul>')
    (output/'index.html').write_text('\n'.join(links), encoding='utf-8')
    report = dict(schema_version='driveguard-fatigue-head-region-preview-v1', status='complete',
                  recovery_summary=str(source), recovery_sha256=digest, record_ids=ids, previews=previews,
                  region_selection='No region guessed; each picker requires an explicit selection.')
    write_json(output/'summary.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    preview = commands.add_parser('prepare', help='Preview baseline records containing fatigue-positive windows')
    for name in ('recovery-summary', 'dataset-root', 'output-dir'):
        preview.add_argument('--'+name, required=True)
    preview.add_argument('--samples', type=int, default=24)
    preview.add_argument('--record-id', dest='record_ids', action='append')
    builder = commands.add_parser('build', help='Combine scoped picker exports into baseline/gate/crop arms')
    builder.add_argument('--crop-config', dest='crop_configs', required=True, action='append')
    builder.add_argument('--output', required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop('command')
    try:
        if command == 'prepare':
            result = prepare(**args)
            print(json.dumps(dict(status=result['status'], records=len(result['record_ids']),
                                  index=str(Path(args['output_dir']).resolve()/'index.html'))))
        else:
            result = build(**args)
            print(json.dumps(dict(records=len(result['head_regions']), variants=len(result['variants']),
                                  record_variants=len(result['head_regions'])*len(result['variants']), output=args['output'])))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f'Fatigue head regions failed: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
