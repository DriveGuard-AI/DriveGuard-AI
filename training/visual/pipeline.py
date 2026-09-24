"""Resumable extraction, frozen-context preparation and three-arm comparison."""
import argparse
import json
from pathlib import Path


def run(*,manifest,data_root,dataset_root,checkpoint,weights,output_dir,settings,
        rgb_backend='cpu',rgb_device_index=0,rgb_batch_size=32,continue_run=False):
    import av
    from PIL import Image
    from training.common.checkpoint import atomic_json
    from training.common.config import read_json
    from training.common.data import file_sha256
    from training.common.device import select_device
    from training.common.model_factory import load_model
    from training.visual.weights import ensure,SHA256
    from training.visual.extract import extract
    from training.visual.prepare import prepare
    from training.visual.data import PreparedData
    from training.visual.suite import run_suite
    from training.visual.train import _settings
    settings=_settings(settings)
    weights=ensure(weights)
    manifest=Path(manifest).resolve(strict=True);data_root=Path(data_root).resolve(strict=True)
    dataset_root=Path(dataset_root).resolve(strict=True);checkpoint=Path(checkpoint).resolve(strict=True)
    output=Path(output_dir).resolve()
    for source in (data_root,dataset_root,checkpoint.parent,weights.parent):
        if output.is_relative_to(source) or source.is_relative_to(output):raise ValueError('Pipeline output overlaps inputs')
    plan=dict(schema_version='driveguard-visual-pipeline-v1',manifest=str(manifest),manifest_sha256=file_sha256(manifest),
              data_root=str(data_root),dataset_root=str(dataset_root),checkpoint=str(checkpoint),
              checkpoint_sha256=file_sha256(checkpoint),weights=str(weights),weights_sha256=SHA256,
              settings=settings,rgb_backend=rgb_backend,rgb_device_index=rgb_device_index,rgb_batch_size=rgb_batch_size)
    if output.exists():
        if not continue_run:raise FileExistsError('Use a NEW output_dir or --continue-run')
        if read_json(output/'pipeline.json')!=plan:raise ValueError('Pipeline configuration/identity mismatch')
    # Fail before expensive extraction if baseline or accelerator is incompatible.
    model,payload=load_model(checkpoint,settings['backend'],settings['device_index'])
    if (payload['config']['model']['kind']!='relation_gaze_head_policy_v1'
        or payload['config']['model']['head_policy']!=dict(min_valid_ratio=.25,max_gap_steps=2)):
        raise ValueError('Expected coverage25+gap80 reference checkpoint')
    del model,payload
    select_device(rgb_backend,rgb_device_index)
    if not output.exists():output.mkdir(parents=True);atomic_json(output/'pipeline.json',plan)
    print('Phase 1/3: verify/extract frozen RGB features',flush=True)
    extract(manifest,data_root,dataset_root,weights,SHA256,output/'rgb-cache',
            backend=rgb_backend,device_index=rgb_device_index,batch_size=rgb_batch_size,
            continue_run=(output/'rgb-cache').exists())
    print('Phase 2/3: prepare frozen gap80 context and train-only normalization',flush=True)
    prepared_root=None
    for attempt in sorted(output.glob('prepared-*')):
        if not (attempt/'prepared.json').exists():continue
        data=PreparedData(attempt)
        try:
            report=data.report
            if (report['baseline_checkpoint_sha256']!=plan['checkpoint_sha256'] or report['manifest']!=str(manifest)
                or report['data_root']!=str(data_root) or report['visual_cache']!=str(output/'rgb-cache')):
                raise ValueError('Prepared context identity mismatch')
        finally:data.close()
        prepared_root=attempt;break
    if prepared_root is None:
        prepared_root=output/f"prepared-{len(list(output.glob('prepared-*')))+1:03d}"
        prepare(checkpoint=checkpoint,visual_cache=output/'rgb-cache',output_dir=prepared_root,
                manifest=manifest,data_root=data_root,backend=settings['backend'],
                device_index=settings['device_index'],batch_size=settings['batch_size'])
    print('Phase 3/3: train B/C and compare with A on fixed validation windows',flush=True)
    report=run_suite(prepared_root,output/'comparison',settings=settings,continue_run=(output/'comparison').exists())
    atomic_json(output/'result.json',dict(status='complete',comparison=str(output/'comparison'/'comparison.json')))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','data-root','dataset-root','checkpoint','output-dir'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--weights',default='models/local/resnet18-f37072fd.pth')
    parser.add_argument('--continue-run',action='store_true')
    parser.add_argument('--check-config',action='store_true',help='Syntax-only check; does not verify dependencies or input artifacts')
    parser.add_argument('--rgb-backend',choices=('cpu','cuda','npu'),default='cpu')
    parser.add_argument('--rgb-device-index',type=int,default=0);parser.add_argument('--rgb-batch-size',type=int,default=32)
    parser.add_argument('--backend',choices=('cpu','cuda','npu'),default='npu')
    parser.add_argument('--device-index',type=int,default=0);parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=64);parser.add_argument('--seed',type=int,default=42)
    args=vars(parser.parse_args(argv));check=args.pop('check_config')
    for key in ('epochs','batch_size','rgb_batch_size'):
        if args[key]<1:parser.error(key+' must be positive')
    if min(args['device_index'],args['rgb_device_index'],args['seed'])<0:parser.error('Indices/seed must be nonnegative')
    if check:print(json.dumps(dict(status='valid_options',**args)));return
    from training.visual.suite import DEFAULTS
    settings=dict(DEFAULTS)
    for key in ('backend','device_index','epochs','batch_size','seed'):settings[key]=args.pop(key)
    try:
        report=run(settings=settings,**args)
        print(json.dumps(dict(status=report['status'],comparisons=len(report['comparisons']))))
    except (OSError,ValueError,RuntimeError,ImportError) as exc:parser.exit(2,f'error: {exc}\n')


if __name__=='__main__':main()
