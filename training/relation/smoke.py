"""Reproducible CPU-only synthetic relation training; never opens real datasets."""
import argparse
from copy import deepcopy
import json
from pathlib import Path


def run_smoke(output_dir):
    import numpy as np
    import torch
    from training.common.checkpoint import atomic_json, load_checkpoint
    from training.common.config import TASKS, canonical_hash, load_config
    from training.common.data import FeatureCorpus
    from training.common.model_factory import load_model
    from training.temporal.train import run_training

    output = Path(output_dir).resolve()
    output.mkdir(parents=True,exist_ok=False)
    root = output/'synthetic_data'
    root.mkdir()
    config = load_config(Path(__file__).resolve().parents[2]/'configs/training/relation_gated.json')
    names = config['data']['feature_names']
    rng = np.random.default_rng(17)
    records = []
    for i,split in enumerate(('train','train','val','val')):
        frames = 225
        record = {'record_id':f'synthetic-{i}','subject_id':f'synthetic-subject-{i}',
                  'recording_id':f'synthetic-recording-{i}','view':'body','subset':'synthetic-only','split':split}
        values = rng.normal(size=(frames,19)).astype(np.float32)
        masks = np.ones((frames,19),np.uint8)
        masks[:,5:8] = 0  # Pose was never available during training.
        ages = np.zeros((frames,3),np.float32)
        ages[120:125] = 81  # Includes one unavailable labelled window endpoint.
        ages[175:180,2] = 81  # Hand-only interruption, other evidence survives.
        targets = np.full((frames,4),i%2,np.float32)
        labels = np.ones((frames,4),np.uint8)
        labels[:10,2:] = 0
        labels[149,3] = 0
        targets[labels == 0] = 0
        for name,array in (('values',values),('masks',masks),('expert_age_ms',ages),('targets',targets),('label_masks',labels)):
            record[name] = f'record-{i}-{name}.npy'
            np.save(root/record[name],array,allow_pickle=False)
        records.append(record)
    profile = config['data']['group_profile']
    manifest = {'schema_version':'driveguard-feature-manifest-v2','dataset_version':'synthetic-relation-smoke-v1',
                'feature_schema':{'version':config['data']['feature_schema_version'],'names':names,'sample_rate_hz':25},
                'extractor_profile':{'name':'synthetic-only','synthetic':True,
                                     'feature_definitions':{name:'Random synthetic feature; NOT a physical measurement' for name in names}},
                'group_profile':profile,'group_profile_sha256':canonical_hash(profile),
                'label_mapping':{'version':'driveguard-behavior-v1','tasks':list(TASKS),
                                 'rules':{'synthetic':'Record parity labels with explicit unknown slots; not DMD truth'}},
                'records':records}
    path = root/'manifest.json'
    atomic_json(path,manifest)
    config['training'].update(epochs=2,batch_size=4)
    config['runtime'].update(backend='cpu',data_root=str(root),manifest=str(path),output_dir=str(output/'initial'))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        initial = run_training(config)
        config['training']['epochs'] = 3
        config['runtime']['output_dir'] = str(output/'resumed')
        resumed = run_training(config,resume=initial['last_checkpoint'])
        config['runtime']['output_dir'] = str(output/'continuous')
        continuous = run_training(config)
        a,b = (load_checkpoint(r['last_checkpoint']) for r in (resumed,continuous))
        equal = a['history'] == b['history'] and a['global_step'] == b['global_step']
        equal = equal and all(torch.equal(a['model_state'][key],b['model_state'][key]) for key in a['model_state'])
        if not equal:
            raise AssertionError('CPU resumed training differs from continuous run')
        model,payload = load_model(resumed['best_checkpoint'])
        corpus = FeatureCorpus(config,path,root)
        try:
            x,_,_ = corpus.dataset('val',payload['normalizer'])[0]
            with torch.no_grad():
                result = model(**{key:torch.as_tensor(value)[None] for key,value in x.items()},return_aux=True)
            finite = bool(torch.isfinite(result['logits']).all())
            if not finite or not bool(result['available'].all()):
                raise AssertionError('Reloaded model failed finite available-input inference')
        finally:
            corpus.close()
        report = {'synthetic_only':True,'purpose':'Engineering only; NOT DMD scores or GPU/NPU validation',
                  'initial':initial,'resumed':resumed,'continuous':continuous,
                  'resume_matches_continuous':equal,'reload_finite':finite,
                  'parameter_count':sum(p.numel() for p in model.parameters()),
                  'unavailable_windows':a['history'][-1]['train']['unavailable_windows'],
                  'unsupported_features':[name for name,supported in zip(names,payload['normalizer']['supported']) if not supported],
                  'torch':str(torch.__version__)}
        atomic_json(output/'smoke_report.json',report)
        return report
    finally:
        torch.set_num_threads(old_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,help='NEW directory under ignored runs/')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_smoke(args.output_dir),ensure_ascii=False,allow_nan=False))
        return 0
    except (OSError,ValueError,RuntimeError,ImportError,AssertionError) as exc:
        parser.exit(2,f'error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
