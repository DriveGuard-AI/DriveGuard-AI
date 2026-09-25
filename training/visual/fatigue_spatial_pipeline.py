"""Audit fatigue misses, extract head RGB once, then compare spatial inputs."""
import argparse
import json
from pathlib import Path

from training.common.checkpoint import atomic_json
from training.common.config import canonical_hash, read_json
from training.common.data import file_sha256
from training.visual.data import PreparedData, contained
from training.visual.spatial import DEFAULT_CROP, crop_policy
from training.visual.suite import DEFAULTS
from training.visual.train import _settings


def _reference(path):
    report = read_json(path)
    if (report.get('schema_version') != 'driveguard-fatigue-visual-comparison-v1'
            or report.get('status') != 'complete' or report.get('task') != 'fatigue'
            or report.get('split') != 'val'):
        raise ValueError('Expected completed fatigue visual validation report')
    prepared = PreparedData(report['prepared_dir'])
    try:
        if prepared.task != 'fatigue' or 'crop' in prepared.cache.report:
            raise ValueError('Reference must use whole-frame fatigue RGB')
        if prepared.provenance['prepared_sha256'] != report['prepared_sha256']:
            raise ValueError('Reference preparation checksum mismatch')
        for row in report['comparisons']:
            if row.get('arm') == 'B_rgb' and row.get('checkpoint_kind') == 'best':
                if file_sha256(row['checkpoint']) != row['checkpoint_sha256']:
                    raise ValueError('Reference RGB checkpoint checksum mismatch')
        baseline = prepared.report
        if file_sha256(baseline['baseline_checkpoint']) != baseline['baseline_checkpoint_sha256']:
            raise ValueError('Reference baseline checkpoint checksum mismatch')
        if file_sha256(baseline['manifest']) != prepared.cache.report['manifest_sha256']:
            raise ValueError('Reference manifest checksum mismatch')
        return dict(manifest=baseline['manifest'], data_root=baseline['data_root'],
                    prepared_dir=str(prepared.root), prepared_sha256=report['prepared_sha256'],
                    visual_cache=baseline['visual_cache'],
                    weights_sha256=prepared.cache.report['encoder']['weights_sha256'])
    finally:
        prepared.close()


def _audit(**kwargs):
    from training.visual.fatigue_audit import run
    return run(**kwargs)


def _extract(**kwargs):
    from training.visual.extract import extract
    return extract(**kwargs)


def _compare(**kwargs):
    from training.visual.fatigue_spatial_suite import run
    return run(**kwargs)


def _run_audit(output, identity, kwargs):
    """Reuse only a complete audit with unchanged evidence files and inputs."""
    marker = output/'audit-complete.json'
    if marker.exists():
        receipt = read_json(marker)
        if receipt.get('identity') != identity:
            raise ValueError('Audit input identity mismatch')
        directory = receipt.get('directory')
        if not isinstance(directory,str) or not directory.startswith('audit-') or '/' in directory or '\\' in directory or ':' in directory:
            raise ValueError('Invalid audit directory')
        root = output/directory
        expected = receipt['artifacts']
        actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        if not expected or set(expected) != actual:
            raise ValueError('Audit evidence layout changed')
        for relative, digest in expected.items():
            if file_sha256(contained(root,relative)) != digest:
                raise ValueError('Audit evidence checksum mismatch')
        if read_json(root/'summary.json').get('status') != 'complete':
            raise ValueError('Audit incomplete')
        return root
    attempts = [p for p in output.glob('audit-*') if p.is_dir()]
    root = output/f'audit-{len(attempts)+1:03d}'
    result = _audit(output_dir=root, **kwargs)
    if result.get('status') != 'complete' or read_json(root/'summary.json').get('status') != 'complete':
        raise ValueError('Audit did not complete; extraction/training stopped')
    artifacts = {p.relative_to(root).as_posix():file_sha256(p) for p in root.rglob('*') if p.is_file()}
    atomic_json(marker, dict(identity=identity,directory=root.name,artifacts=artifacts))
    return root


def run(*, fatigue_report, dataset_root, weights, output_dir, settings=None, seeds=(42,43),
        crop_xyxy=DEFAULT_CROP, variant='B_rgb-seed43-best', rgb_backend='cpu', rgb_device_index=0,
        rgb_batch_size=32, continue_run=False, audit_only=False):
    settings = _settings(dict(DEFAULTS,early_stopping_patience=4) if settings is None else dict(settings))
    if settings.get('positive_weight',1.) != 1.:
        raise ValueError('Spatial comparison fixes ordinary BCE; positive weighting is not allowed')
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):raise ValueError('Distinct seeds required')
    for seed in seeds:_settings(dict(settings,seed=seed))
    policy = crop_policy(crop_xyxy)
    source = Path(fatigue_report).resolve(strict=True)
    dataset = Path(dataset_root).resolve(strict=True)
    if not dataset.is_dir():raise ValueError('Dataset root must be a directory')
    weights = Path(weights).resolve()
    output = Path(output_dir).resolve()
    reference = _reference(source)
    from training.visual.extract import _validate_options
    _validate_options(reference['weights_sha256'],rgb_backend,rgb_device_index,rgb_batch_size)
    inputs = [source, dataset, weights, *(Path(reference[k]).resolve() for k in
              ('manifest','data_root','prepared_dir','visual_cache'))]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in inputs):
        raise ValueError('Pipeline output overlaps input artifacts')
    plan = dict(schema_version='driveguard-fatigue-spatial-pipeline-v1',
                fatigue_report=str(source), fatigue_report_sha256=file_sha256(source), reference=reference,
                dataset_root=str(dataset), weights=str(weights), crop=policy, settings=settings,seeds=seeds,
                variant=variant,rgb_backend=rgb_backend,rgb_device_index=rgb_device_index,rgb_batch_size=rgb_batch_size)
    if output.exists():
        if not continue_run:raise FileExistsError('Use a NEW output_dir or --continue-run')
        if read_json(output/'plan.json') != plan:raise ValueError('Pipeline configuration/identity mismatch')
    else:
        output.mkdir(parents=True)
        atomic_json(output/'plan.json',plan)
    # Audit-only never needs the image encoder weights. Full runs never download.
    if not audit_only and (not weights.is_file() or file_sha256(weights) != reference['weights_sha256']):
        raise ValueError('Local RGB weights missing or checksum mismatch; no automatic download')
    print('Phase 1/3: reproduce fatigue scores and export verified window evidence',flush=True)
    audit = _run_audit(output,dict(plan_sha256=canonical_hash(plan)),dict(fatigue_report=source,
                       dataset_root=dataset,variant=variant,backend=settings['backend'],
                       device_index=settings['device_index'],batch_size=settings['batch_size'],crop_xyxy=list(crop_xyxy)))
    if audit_only:
        return dict(status='audit_complete',audit_dir=str(audit))
    print('Phase 2/3: extract head RGB on the unchanged causal grid (local weights)',flush=True)
    head = output/'head-rgb-cache'
    _extract(manifest=reference['manifest'],data_root=reference['data_root'],dataset_root=dataset,
             weights=weights,weights_sha256=reference['weights_sha256'],output_dir=head,
             backend=rgb_backend,device_index=rgb_device_index,batch_size=rgb_batch_size,
             crop=policy,continue_run=head.exists())
    print('Phase 3/3: whole/head/both, ordinary BCE, matched model size and seeds',flush=True)
    comparison = output/'comparison'
    report = _compare(fatigue_report=source,head_cache=head,output_dir=comparison,
                      settings=settings,seeds=seeds,continue_run=comparison.exists())
    if report.get('status') != 'complete':raise ValueError('Spatial comparison incomplete')
    result = dict(status='complete',training_runs=3*len(seeds),audit_dir=str(audit),
                  comparison=str(comparison/'comparison.json'),manual_crop_review='pending')
    atomic_json(output/'result.json',result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fatigue-report','dataset-root','weights','output-dir'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    parser.add_argument('--crop',type=float,nargs=4,default=list(DEFAULT_CROP),dest='crop_xyxy')
    parser.add_argument('--variant',default='B_rgb-seed43-best')
    parser.add_argument('--rgb-backend',choices=('cpu','npu','cuda'),default='cpu')
    parser.add_argument('--rgb-device-index',type=int,default=0)
    parser.add_argument('--rgb-batch-size',type=int,default=32)
    parser.add_argument('--patience',type=int,default=4)
    parser.add_argument('--continue-run',action='store_true')
    parser.add_argument('--audit-only',action='store_true')
    for key,value in DEFAULTS.items():
        parser.add_argument('--'+key.replace('_','-'),type=type(value),default=value)
    args = vars(parser.parse_args(argv))
    settings = {k:args.pop(k) for k in DEFAULTS}
    settings['early_stopping_patience'] = args.pop('patience')
    try:
        result = run(settings=settings,**args)
        print(json.dumps(result),flush=True)
    except (OSError,ValueError,RuntimeError,ImportError) as error:
        parser.exit(2,f'error: {error}\n')


if __name__ == '__main__':main()
