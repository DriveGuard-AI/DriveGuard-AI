"""Export immutable, local deployment artifacts from already selected reports."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

SCHEMA = 'driveguard-live-composite-v1'
CANDIDATE = 'C_fusion_seed43_validation_fpr03_coverage25_outputs'
ROLES = ('coverage25','gap80','distraction','actions','eye','rgb_encoder')


def code_identity():
    root=Path(__file__).resolve().parents[2]
    paths=[]
    for directory in ('inference','training/common','training/gaze','training/relation','training/temporal','training/visual','training/dmd'):
        paths.extend((root/directory).rglob('*.py'))
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for p in sorted(paths)}


def expert_identity(runtime):
    result=deepcopy({key:runtime[key] for key in ('objects','face_asset','hands_asset')})
    result['objects'].pop('service',None)
    for asset in (result['objects']['asset'],result['face_asset'],result['hands_asset']):asset.pop('path',None)
    return result


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verified_path(root, entry):
    root=Path(root).resolve();relative=Path(entry['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Bundle asset must be a contained relative file')
    path=(root/relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or digest(path)!=entry['sha256']:
        raise ValueError('Bundle asset checksum/containment mismatch: '+entry['path'])
    return path


def select_reports(full, recovery):
    if (full.get('schema_version')!='driveguard-visual-all-tasks-v1' or
            recovery.get('schema_version')!='driveguard-eye-recovery-comparison-v1' or
            any(r.get('status')!='complete' or r.get('split')!='val' for r in (full,recovery)) or
            not recovery.get('reference_reproduced') or not recovery.get('frozen_weights_unchanged') or
            recovery.get('training_runs')!=0):
        raise ValueError('Expected completed all-task and fixed-eye recovery reports')
    selected=[r for r in full['composites'] if r['name']==CANDIDATE]
    fatigue=[r for r in recovery['comparisons'] if r['variant']=='dense_track-combined']
    if len(selected)!=1 or len(fatigue)!=1:
        raise ValueError('Expected exactly one fixed composite and dense_track-combined entry')
    return selected[0],fatigue[0]


def load_bundle(path):
    path=Path(path).resolve(strict=True);root=path.parent;report=read(path)
    if (report.get('schema_version')!=SCHEMA or report.get('status')!='complete' or
            set(report.get('models',{}))!=set(ROLES)):
        raise ValueError('Incomplete live model bundle')
    if report.get('code_identity')!=code_identity():
        raise ValueError('Bundle code identity differs; use matching source or export a new bundle')
    for entry in report['files'].values():verified_path(root,entry)
    models={name:verified_path(root,entry) for name,entry in report['models'].items()}
    from training.visual.eye_tracking import POLICY
    if report['eye_tracking']!=POLICY or report['event_policy']!={
            'closed_eye_min_ms':1000,'threshold_logit':0.,'missing':'reset-eye-only'}:
        raise ValueError('Bundle event/tracking policy differs from implementation')
    from training.visual.spatial import validate_crop
    validate_crop(report['head_crop'])
    return report,models


def export(*, all_tasks_report, recovery_report, rgb_weights, runtime_config, output_dir):
    from inference.perception.runtime import load_runtime_config
    from training.common.checkpoint import load_checkpoint
    from training.visual.weights import SHA256
    from training.visual.eye_tracking import POLICY
    full,recovery=read(all_tasks_report),read(recovery_report)
    selected,fatigue=select_reports(full,recovery)
    output=Path(output_dir).resolve()
    if output.exists():raise FileExistsError('Use a NEW output directory')
    provenance=recovery['plan']['provenance'];action=provenance['source_reference']
    references=dict(coverage25=(full['output_base']['checkpoint'],full['output_base']['checkpoint_sha256']),
        gap80=(full['provenance']['baseline_checkpoint'],full['provenance']['baseline_checkpoint_sha256']),
        distraction=(selected['checkpoint'],selected['checkpoint_sha256']),
        actions=(action['checkpoint'],action['checkpoint_sha256']),
        eye=(provenance['eye_checkpoint'],provenance['eye_checkpoint_sha256']),rgb_encoder=(rgb_weights,SHA256))
    payloads={}
    for role,(path,sha) in references.items():
        if digest(path)!=sha:raise ValueError('Selected weight checksum mismatch: '+role)
        if role!='rgb_encoder':payloads[role]=load_checkpoint(path)
    probe=payloads['distraction'];eye=payloads['eye']
    if (probe['provenance']['preparation']['baseline_checkpoint_sha256']!=references['gap80'][1] or
            eye['provenance']['source_reference']!=action):
        raise ValueError('Specialist dependency identity mismatch')
    head_path=Path(eye['provenance']['head_cache'])/'manifest.json'
    if digest(head_path)!=eye['provenance']['head_cache_sha256']:
        raise ValueError('Head crop provenance checksum mismatch')
    crop=read(head_path)['run']['crop']
    cache_path=Path(probe['provenance']['preparation']['visual_cache'])/'manifest.json'
    if digest(cache_path)!=probe['provenance']['preparation']['visual_cache_sha256']:
        raise ValueError('RGB encoder provenance checksum mismatch')
    from training.visual.encoder_metadata import encoder_metadata
    cache_report=read(cache_path)
    if cache_report['encoder']!=encoder_metadata(SHA256):
        raise ValueError('RGB encoder identity mismatch')
    runtime=load_runtime_config(runtime_config)
    if runtime['schema_version']!='driveguard-runtime-v2':raise ValueError('Gaze runtime v2 required')
    # Latest feature corpus was extracted with YOLO11. A different detector changes
    # model inputs; do not silently substitute the older local YOLOX example.
    if runtime['objects']['variant']!='yolo11s':raise ValueError('Latest composite requires original YOLO11 runtime')
    preparation=probe['provenance']['preparation']
    if digest(preparation['manifest'])!=cache_report['manifest_sha256']:
        raise ValueError('Source feature manifest checksum mismatch')
    records={r['record_id']:r for r in read(preparation['manifest'])['records']}
    source=next(r for r in cache_report['records'] if r['view']=='body')
    value_path=(Path(preparation['data_root'])/records[source['record_id']]['values']).resolve(strict=True)
    complete=read(value_path.parent.parent/'complete.json')
    if complete['fingerprint']!=source['source']['cache_fingerprint']:
        raise ValueError('Original feature extraction identity mismatch')
    original_feature=complete['feature_record']
    from inference.perception.gaze_profile import gaze_profile
    if (original_feature['extractor_profile']!=gaze_profile() or
            expert_identity(original_feature['provenance']['config'])!=expert_identity(runtime)):
        raise ValueError('Live perception settings differ from original trained feature inputs')
    from inference.experts.assets import verify_asset
    for asset in (runtime['objects']['asset'],runtime['face_asset'],runtime['hands_asset']):verify_asset(asset)
    if recovery['plan']['caches']['dense_track']['policy']['tracking']!=POLICY:
        raise ValueError('Tracking report differs from running code')
    output.mkdir(parents=True);(output/'weights').mkdir();(output/'assets').mkdir()
    files={};models={}
    def copy(source,name,expected=None):
        target=output/name;before=digest(source)
        if expected is not None and before!=expected:raise ValueError('Source changed before packaging')
        shutil.copyfile(source,target)
        if digest(target)!=before or digest(source)!=before:raise ValueError('Source changed during packaging')
        entry=dict(path=name,sha256=before);files[name]=entry;return entry
    for role,(path,sha) in references.items():
        name='weights/'+role+Path(path).suffix
        models[role]=copy(path,name,sha)
        if role!='rgb_encoder':copy(str(path)+'.sha256.json',name+'.sha256.json')
    for role,asset in (('objects',runtime['objects']['asset']),('face',runtime['face_asset']),('hands',runtime['hands_asset'])):
        name='assets/'+role+Path(asset['path']).suffix
        copy(asset['path'],name,asset['sha256']);asset['path']=name
    # Keep the existing detector endpoint/token location: no secret is exported.
    runtime['driver_region_confirmed_exclusive']=False
    runtime_path=output/'runtime.json'
    runtime_path.write_text(json.dumps(runtime,indent=2)+'\n',encoding='utf-8')
    files['runtime.json']=dict(path='runtime.json',sha256=digest(runtime_path))
    report=dict(schema_version=SCHEMA,status='complete',models=models,files=files,code_identity=code_identity(),
        candidate=CANDIDATE,distraction_threshold=selected['body_threshold_logit'],head_crop=crop,
        eye_tracking=POLICY,event_policy=recovery['plan']['event_policy'],
        reference_metrics=dict(all_tasks=selected,body_fatigue=fatigue),
        source_reports=dict(all_tasks_sha256=digest(all_tasks_report),recovery_sha256=digest(recovery_report)),
        runtime='runtime.json',qualification='webcam_integration_candidate_not_live_validated')
    # This completion marker is written last; no partial package is loadable.
    pending=output/'bundle.pending.json'
    pending.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    load_bundle(pending)
    pending.rename(output/'bundle.json')
    return output/'bundle.json'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',help='Verify an existing bundle without loading model runtimes')
    for name in ('all-tasks-report','recovery-report','rgb-weights','runtime-config','output-dir'):
        parser.add_argument('--'+name)
    args=vars(parser.parse_args(argv));check=args.pop('check')
    try:
        if check:load_bundle(check);print('Bundle checksums and policies verified');return
        if not all(args.values()):parser.error('All export arguments are required without --check')
        print(export(**args))
    except (OSError,ValueError,KeyError,TypeError) as error:parser.exit(2,f'error: {error}\n')


if __name__=='__main__':main()
