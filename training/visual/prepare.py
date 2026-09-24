"""Bind frozen gap80 context and RGB windows without changing original caches."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training.common.checkpoint import atomic_json
from training.common.data import file_sha256
from training.common.model_factory import load_model
from training.gaze.data import FeatureCorpus
from training.visual.data import VisualCache,METADATA


@torch.no_grad()
def frozen_context(model,inputs):
    if model.training: raise ValueError('Reference model must be frozen in eval mode')
    output=model(**inputs)
    embeddings=model.checked_embeddings(**inputs)
    eligible,ratio,gap=model.head_statistics(inputs['x'])
    head=model.encode_head(inputs['x'])[:,:,-1].masked_fill(~eligible[:,None],0)
    freshness=torch.where(gap==0,inputs['quality'][:,0,1],(1-gap.float()/(model.head_policy['max_gap_steps']+1)).clamp_min(0))
    head_quality=torch.stack((ratio,freshness),dim=-1).masked_fill(~eligible[:,None],0)
    quality=inputs['quality'].masked_fill(~inputs['eligible'][:,:,None],0)
    return torch.cat((embeddings.flatten(1),head,quality.flatten(1),head_quality,
                      inputs['eligible'].float(),eligible[:,None].float(),output['behavior_logits'][:,3:4]),dim=1),output


def fit_statistics(chunks,size):
    count=0;mean=np.zeros(size,np.float64);m2=np.zeros(size,np.float64)
    for values in chunks:
        values=np.asarray(values,np.float64)
        if not len(values):continue
        if values.ndim!=2 or values.shape[1]!=size or not np.isfinite(values).all(): raise ValueError('Invalid normalization input')
        n=len(values);batch=values.mean(0);delta=batch-mean
        m2+=((values-batch)**2).sum(0)+delta**2*count*n/(count+n)
        mean+=delta*n/(count+n);count+=n
    if not count: raise ValueError('No usable training features for normalization')
    return dict(count=count,mean=mean.tolist(),std=np.maximum(np.sqrt(np.maximum(m2,0)/count),1e-6).tolist())


def prepare(*,checkpoint,visual_cache,output_dir,manifest=None,data_root=None,backend='cpu',device_index=0,batch_size=64):
    if type(batch_size)!=int or batch_size<1: raise ValueError('Invalid preparation batch size')
    output=Path(output_dir).resolve()
    if output.exists():raise FileExistsError('Use a NEW preparation output_dir')
    checkpoint=Path(checkpoint).resolve(strict=True);checkpoint_hash=file_sha256(checkpoint)
    model,payload=load_model(checkpoint,backend,device_index)
    config=payload['config']
    if config['model']['kind']!='relation_gaze_head_policy_v1' or config['model']['head_policy']!=dict(min_valid_ratio=.25,max_gap_steps=2):
        raise ValueError('Visual comparison requires frozen coverage25+gap80 checkpoint')
    model.requires_grad_(False)
    manifest=Path(manifest or config['runtime']['manifest']).resolve(strict=True)
    data_root=Path(data_root or config['runtime']['data_root']).resolve(strict=True)
    visual_cache=Path(visual_cache).resolve(strict=True)
    for source in (visual_cache,data_root,checkpoint.parent):
        if output.is_relative_to(source) or source.is_relative_to(output): raise ValueError('Preparation output must be separate from input artifacts')
    cache=VisualCache(visual_cache);corpus=None
    try:
        if cache.report['manifest_sha256']!=file_sha256(manifest):raise ValueError('Visual/source manifest mismatch')
        corpus=FeatureCorpus(config,manifest,data_root)
        if corpus.summary!=payload['data_summary']:raise ValueError('Baseline data fingerprint mismatch')
        original={r['record_id']:r for r in corpus.records if r['view']=='body'}
        if set(original)!=set(cache.records):raise ValueError('Visual cache must cover every original body record')
        for rid,record in original.items():
            entry=cache.records[rid]
            if any(record[k]!=entry[k] for k in METADATA) or entry['slots']!=record['frames']:
                raise ValueError('Visual metadata/split/count differs from original corpus')
        output.mkdir(parents=True,exist_ok=False)
        dimension=5*config['model']['projection_dim']+16;split_report={}
        device=next(model.parameters()).device
        for split in ('train','val','test'):
            dataset=corpus.dataset(split,payload['normalizer'])
            indices=[i for i in range(len(dataset)) if dataset.metadata(i)['view']=='body']
            if not indices: continue
            rows=[];destination=output/(split+'-context.npy')
            context=np.lib.format.open_memmap(destination,mode='w+',dtype=np.float32,shape=(len(indices),dimension))
            try:
                for start in range(0,len(indices),batch_size):
                    items=[dataset[i] for i in indices[start:start+batch_size]]
                    tensors={k:torch.as_tensor(np.stack([item[0][k] for item in items]),device=device) for k in items[0][0]}
                    encoded,predictions=frozen_context(model,tensors)
                    context[start:start+len(items)]=encoded.cpu().numpy()
                    logits=predictions['behavior_logits'][:,3].cpu().tolist()
                    available=predictions['task_available'][:,3].cpu().tolist()
                    heads=predictions['head_available'].cpu().tolist()
                    for offset,(inputs,target,known,*_) in enumerate(items):
                        record,slot=dataset.locate(indices[start+offset])
                        rows.append(dict(**{k:record[k] for k in METADATA},start_slot=slot,end_slot=slot+99,
                                         visual_start=slot//25*2,label_known=bool(known[3]),target=int(target[3]),
                                         baseline_available=bool(available[offset]),
                                         baseline_logit=float(logits[offset]) if available[offset] else None,
                                         h_eligible=bool(heads[offset])))
                context.flush()
            finally: context._mmap.close()
            row_path=output/(split+'-windows.json');atomic_json(row_path,rows)
            split_report[split]=dict(context=destination.name,context_sha256=file_sha256(destination),
                                     rows=row_path.name,rows_sha256=file_sha256(row_path),windows=len(rows))
        if not {'train','val'}<=set(split_report):raise ValueError('Body train and val records required')
        def visual_chunks():
            for rid,entry in cache.records.items():
                if entry['split']!='train':continue
                arrays=cache.arrays(rid)
                for start in range(0,entry['sample_count'],2048):
                    yield arrays['embeddings'][start:start+2048][arrays['valid'][start:start+2048]]
        context=np.load(output/split_report['train']['context'],mmap_mode='r',allow_pickle=False)
        try:
            context_state=fit_statistics((context[i:i+2048] for i in range(0,len(context),2048)),dimension)
        finally: context._mmap.close()
        normalization=dict(visual=fit_statistics(visual_chunks(),512),context=context_state)
        if file_sha256(checkpoint)!=checkpoint_hash:raise ValueError('Reference checkpoint changed during preparation')
        report=dict(schema_version='driveguard-visual-prepared-v1',status='complete',
                    baseline_checkpoint=str(checkpoint),baseline_checkpoint_sha256=checkpoint_hash,
                    baseline_epoch=payload['epoch'],data_fingerprint=corpus.summary['fingerprint'],
                    manifest=str(manifest),data_root=str(data_root),visual_cache=str(visual_cache),
                    visual_cache_sha256=cache.manifest_sha256,context_dim=dimension,
                    normalization=normalization,splits=split_report,
                    notes=['Reference model frozen; only body distraction probes are trained.',
                           'RGB normalization uses unique train samples; context normalization uses train windows only.',
                           'Reference epoch selected by original joint validation loss; probes use binary validation BCE.'])
        atomic_json(output/'prepared.json',report)
        return report
    finally:
        cache.close()
        if corpus is not None:corpus.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','visual-cache','output-dir'):parser.add_argument('--'+name,required=True)
    for name in ('manifest','data-root'):parser.add_argument('--'+name)
    parser.add_argument('--backend',choices=('cpu','cuda','npu'),default='cpu')
    parser.add_argument('--device-index',type=int,default=0);parser.add_argument('--batch-size',type=int,default=64)
    try:
        args=parser.parse_args(argv);result=prepare(**vars(args))
        print(json.dumps(dict(status=result['status'],output_dir=args.output_dir)));return 0
    except (ValueError,RuntimeError,OSError,ImportError) as exc:parser.exit(2,f'error: {exc}\n')


if __name__ == '__main__':raise SystemExit(main())
