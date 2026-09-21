"""Single-device FP32 joint training with deterministic epoch resume."""
from copy import deepcopy
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from training.common.checkpoint import (atomic_json,capture_rng,cpu_tree,load_checkpoint,
    restore_rng,save_checkpoint,seed_everything,validate_resume)
from training.common.config import TASKS,canonical_hash
from training.common.device import select_device
from training.common.model_factory import build_model
from training.temporal.train import environment_info
from training.temporal.metrics import ClassificationMetrics
from training.gaze.config import validate_config
from training.gaze.data import FeatureCorpus
from training.gaze.losses import joint_loss
from training.gaze.metrics import GazeMetrics
from training.gaze.model import RelationGazeTCN
from training.gaze.checkpoint import validate_metadata


def run_epoch(model,loader,device,task_weights=None,optimizer=None,gradient_clip_norm=1.):
    training=optimizer is not None
    model.train(training)
    active=torch.ones(4,dtype=torch.bool)
    behavior=ClassificationMetrics(active); gaze=GazeMetrics(); label_coverage=ClassificationMetrics(active)
    groups={key:{} for key in ('subject_id','view','subset')}
    sums=torch.zeros(5,dtype=torch.float64); counts=torch.zeros(5,dtype=torch.int64)
    weights=torch.tensor(task_weights or [1.]*5,dtype=torch.float64)
    updates=skipped=batches=offset=unavailable=gaze_unavailable=gaze_raw_known=0
    with torch.set_grad_enabled(training):
        for inputs,target,labels,gaze_target,gaze_labels in loader:
            batches+=1
            inputs={key:value.to(device) for key,value in inputs.items()}
            target,labels,gaze_target,gaze_labels=[value.to(device) for value in (target,labels,gaze_target,gaze_labels)]
            available=inputs['eligible'].any(-1)
            gaze_valid=RelationGazeTCN.gaze_availability(inputs['x'],inputs['eligible'])
            label_coverage.update_coverage(target,labels)
            gaze_raw_known+=int(gaze_labels.sum())
            unavailable+=int((~available).sum()); gaze_unavailable+=int((~gaze_valid).sum())
            effective=labels.bool() & available[:,None]
            gaze_effective=gaze_labels.bool() & gaze_valid
            target=target.masked_fill(~effective,0)
            gaze_target=gaze_target.masked_fill(~gaze_effective,0)
            # Fully unsupervised batches consume neither dropout RNG nor Adam steps.
            if training and not (effective.any() or gaze_effective.any()):
                behavior.update_coverage(target,effective)
                gaze.unknown+=int((~gaze_effective).sum())
                skipped+=1; offset+=len(target)
                continue
            if training: optimizer.zero_grad(set_to_none=True)
            outputs=model(**inputs)
            loss,batch_sums,batch_counts=joint_loss(outputs['behavior_logits'],outputs['gaze_logits'],target,effective,gaze_target,gaze_effective,task_weights)
            sums+=batch_sums.detach().cpu().double(); counts+=batch_counts.cpu()
            if loss is None: skipped+=1
            elif training:
                if not torch.isfinite(loss): raise ValueError('Nonfinite joint loss')
                loss.backward()
                # Concatenation creates zero gradients for unused behavior heads.
                # None also prevents Adam momentum/weight decay from updating them.
                for task in range(4):
                    if not batch_counts[task]:
                        for module in (model.heads[task],model.gates[task]):
                            for parameter in module.parameters(): parameter.grad=None
                if not batch_counts[4]:
                    for parameter in model.gaze_head.parameters(): parameter.grad=None
                torch.nn.utils.clip_grad_norm_(model.parameters(),gradient_clip_norm,error_if_nonfinite=True)
                optimizer.step(); updates+=1
            behavior.update(outputs['behavior_logits'],target,effective)
            gaze.update(outputs['gaze_logits'],gaze_target,gaze_effective)
            if not training and hasattr(loader.dataset,'metadata'):
                for i in range(len(target)):
                    metadata=loader.dataset.metadata(offset+i)
                    for key in groups:
                        metrics=groups[key].setdefault(metadata[key],(ClassificationMetrics(active),GazeMetrics()))
                        metrics[0].update(outputs['behavior_logits'][i:i+1],target[i:i+1],effective[i:i+1])
                        metrics[1].update(outputs['gaze_logits'][i:i+1],gaze_target[i:i+1],gaze_effective[i:i+1])
            offset+=len(target)
    supported=counts>0
    if not supported.any() or (training and not updates): raise ValueError('Epoch has no usable supervision/updates')
    loss=float(((sums[supported]/counts[supported])*weights[supported]).sum()/weights[supported].sum())
    return {'loss':loss,'loss_sums':sums.tolist(),'known_counts':counts.tolist(),'metrics':behavior.report(),
        'gaze_metrics':gaze.report(),'updates':updates,'skipped_batches':skipped,'batches':batches,
        'unavailable_windows':unavailable,'gaze_unavailable_windows':gaze_unavailable,'gaze_raw_known':gaze_raw_known,
        'label_coverage':{'positive':label_coverage.positive.tolist(),'negative':label_coverage.negative.tolist(),'unknown':label_coverage.unknown.tolist()},
        'groups':{key:{name:{'behavior':value[0].report(),'gaze':value[1].report()} for name,value in entries.items()} for key,entries in groups.items()}}


def run_training(config,resume=None):
    config=deepcopy(validate_config(config)); settings=config['training']; runtime=config['runtime']
    if not runtime['output_dir'] or not runtime['manifest']: raise ValueError('Training requires manifest and output_dir')
    output=Path(runtime['output_dir']).resolve()
    if output.exists(): raise FileExistsError('Use a NEW output_dir; existing artifacts are never overwritten')
    device=select_device(runtime['backend'],runtime['device_index'])
    corpus=FeatureCorpus(config,runtime['manifest'],runtime['data_root'])
    try:
        normalizer=corpus.fit_normalizer(); coverage=corpus.check_supervision(settings['active_tasks'],normalizer)
        previous=load_checkpoint(resume) if resume else None
        if previous is not None:
            validate_metadata(previous)
            validate_resume(previous,config,corpus.summary,normalizer)
        seed_everything(settings['seed'],device)
        model=build_model(config).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'],foreach=False)
        generator=torch.Generator().manual_seed(settings['seed'])
        history=[]; start=step=best_epoch=0; best_loss=float('inf'); best_state=best_snapshot=None
        if previous is not None:
            model.load_state_dict(previous['model_state'],strict=True); optimizer.load_state_dict(previous['optimizer_state'])
            history=deepcopy(previous['history']); start=previous['epoch']; step=previous['global_step']
            best_epoch=previous['best_epoch']; best_loss=previous['best_val_loss']; best_state=previous['best_model_state']
            best_snapshot=previous['best_snapshot']
            restore_rng(previous['rng_state'],device,generator)
        train=DataLoader(corpus.dataset('train',normalizer),batch_size=settings['batch_size'],shuffle=True,generator=generator,num_workers=0)
        val=DataLoader(corpus.dataset('val',normalizer),batch_size=settings['batch_size'],shuffle=False,
            generator=torch.Generator().manual_seed(settings['seed']),num_workers=0)
        environment=environment_info(device)
        output.mkdir(parents=True,exist_ok=False)
        atomic_json(output/'config.json',config); atomic_json(output/'data_summary.json',{'data':corpus.summary,'supervision':coverage})
        atomic_json(output/'environment.json',environment)
        base={'format_version':1,'config':config,'normalizer':normalizer,'data_summary':corpus.summary,'environment':environment,
            'active_tasks':list(TASKS),'task_order':list(TASKS),'gaze_classes':config['data']['gaze_classes'],
            'feature_schema':corpus.manifest['feature_schema'],'extractor_profile':corpus.manifest['extractor_profile'],
            'label_mapping':corpus.manifest['label_mapping'],'gaze_mapping':corpus.manifest['gaze_mapping'],
            'model_kind':'relation_gaze_v1','group_profile':config['data']['group_profile'],
            'group_profile_sha256':canonical_hash(config['data']['group_profile'])}
        for epoch in range(start+1,settings['epochs']+1):
            train_report=run_epoch(model,train,device,settings['task_weights'],optimizer,settings['gradient_clip_norm'])
            val_report=run_epoch(model,val,device,settings['task_weights'])
            step+=train_report['updates']
            history.append({'epoch':epoch,'global_step':step,'train':train_report,'val':val_report})
            rng=capture_rng(device,generator)
            if val_report['loss'] < best_loss:
                best_loss=val_report['loss']; best_epoch=epoch; best_state=cpu_tree(model.state_dict())
                best_snapshot=cpu_tree({'optimizer_state':optimizer.state_dict(),'rng_state':rng,'global_step':step})
            payload={**base,'kind':'training','epoch':epoch,'global_step':step,'model_state':model.state_dict(),
                'optimizer_state':optimizer.state_dict(),'best_val_loss':best_loss,'best_epoch':best_epoch,'best_model_state':best_state,
                'best_snapshot':best_snapshot,'rng_state':rng,'history':history}
            save_checkpoint(output/'last.pt',payload)
            save_checkpoint(output/'best.pt',{**base,**best_snapshot,'kind':'weights','epoch':best_epoch,'model_state':best_state,'best_val_loss':best_loss})
            atomic_json(output/'history.json',history)
        result={'completed_epochs':settings['epochs'],'global_step':step,'best_epoch':best_epoch,'best_val_loss':best_loss,
            'best_checkpoint':str(output/'best.pt'),'last_checkpoint':str(output/'last.pt'),'device':str(device),
            'synthetic':corpus.manifest['extractor_profile'].get('synthetic',False)}
        atomic_json(output/'result.json',result)
        return result
    finally: corpus.close()
