"""Compare fatigue input recovery on original body validation videos, without training."""
import argparse
import json
from pathlib import Path
import sys

from training.common.config import read_json
from training.gaze.profile import FEATURE_NAMES, validate_group_profile


def run(*, diagnostic_summary, dataset_root, output_dir,
        experiment_config='configs/perception/face_experiments.json',
        manifest=None, data_root=None, window_steps=None, stride_steps=None,
        subset='all', record_ids=None, max_seconds=0, samples_per_reason=2):
    from inference.experts.assets import file_sha256
    from training.dmd.face_experiments import run_experiments
    from training.dmd.face_experiment_config import load_config, validate_config
    experiment = validate_config(experiment_config) if isinstance(experiment_config, dict) else load_config(experiment_config)
    if experiment.get('head_regions') and record_ids is None:
        record_ids = [region['record_id'] for region in experiment['head_regions']]
    summary_path = Path(diagnostic_summary).resolve(strict=True)
    digest = file_sha256(summary_path)
    summary = read_json(summary_path)
    if (summary.get('schema_version') != 'driveguard-fatigue-diagnostics-v1'
            or summary.get('status') != 'complete' or summary.get('original_inputs_unchanged') is not True
            or summary.get('selection', {}).get('split') != 'val'):
        raise ValueError('A complete unchanged-input validation fatigue diagnostic summary is required')
    profile = validate_group_profile(summary['group_profile'], FEATURE_NAMES)
    supported = summary['normalizer_supported']
    if set(supported) != set(FEATURE_NAMES) or any(type(v) is not bool for v in supported.values()):
        raise ValueError('Diagnostic normalizer support does not match the canonical feature schema')
    dimensions = {}
    for key, value in (('window_steps', window_steps), ('stride_steps', stride_steps)):
        saved = summary.get(key)
        if value is not None and saved is not None and value != saved:
            raise ValueError(f'{key} differs from the saved fatigue diagnostic')
        value = saved if value is None else value
        if type(value) is not int or value <= 0:
            raise ValueError(f'Older summaries require explicit --{key.replace("_", "-")} from training config')
        dimensions[key] = value
    manifest = Path(manifest or summary['manifest']).resolve(strict=True)
    data_root = Path(data_root or summary['data_root']).resolve(strict=True)
    if summary.get('manifest_file_sha256') and file_sha256(manifest) != summary['manifest_file_sha256']:
        raise ValueError('Manifest checksum differs from fatigue diagnostic')
    data = read_json(manifest)
    if 'group_profile' in data and data['group_profile'] != profile:
        raise ValueError('Manifest group profile differs from fatigue diagnostic')
    if 'extractor_profile' in data and data['extractor_profile']['feature_definitions'] != summary.get('feature_definitions'):
        raise ValueError('Manifest feature definitions differ from fatigue diagnostic')
    # Existing caches may be relocated explicitly, but never silently select a different manifest.
    if manifest != Path(summary['manifest']).resolve():
        original = Path(summary['manifest'])
        if not original.is_file() or file_sha256(original) != file_sha256(manifest):
            raise ValueError('Manifest override must match the original diagnostic manifest checksum')
    output = Path(output_dir).resolve()
    if summary_path == output or summary_path.is_relative_to(output):
        raise ValueError('Output overlaps diagnostic input')
    settings = dict(group_profile=profile, normalizer_supported=supported, **dimensions,
                    diagnostic_summary=str(summary_path), diagnostic_sha256=digest)
    original_body = summary.get('groups', {}).get('view', {}).get('body')
    if original_body and summary['selection'].get('subset') == 'all' and subset == 'all' and not record_ids and not max_seconds:
        settings['expected_cached_windows'] = {
            'all': original_body['windows'], 'known_positive': original_body['positive'],
            'known_negative': original_body['negative'], 'unknown': original_body['unknown']}
        settings['expected_cached_f_eligible'] = original_body['evidence']['f_eligible']
    return run_experiments(manifest=manifest, data_root=data_root, dataset_root=dataset_root,
                           output_dir=output, experiment_config=experiment_config,
                           view='body', split='val', subset=subset, record_ids=record_ids,
                           max_seconds=max_seconds, samples_per_reason=samples_per_reason,
                           fatigue_settings=settings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic-summary', 'dataset-root', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--experiment-config', default='configs/perception/face_experiments.json')
    for name in ('manifest', 'data-root'):
        parser.add_argument('--'+name)
    for name in ('window-steps', 'stride-steps'):
        parser.add_argument('--'+name, type=int)
    parser.add_argument('--subset', choices=('all', 'drowsiness', 'distraction', 'gaze'), default='all')
    parser.add_argument('--record-id', dest='record_ids', action='append')
    parser.add_argument('--max-seconds', type=float, default=0)
    parser.add_argument('--samples-per-reason', type=int, default=2)
    args = vars(parser.parse_args(argv))
    try:
        report = run(**args)
        print(json.dumps(dict(status=report['status'], partial=report['partial'],
                              record_variants=len(report['records']), output_dir=args['output_dir'])))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f'Fatigue recovery failed: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
