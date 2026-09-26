"""Matched action-supervised/control training with exact epoch resume."""
from copy import deepcopy
from itertools import islice
from pathlib import Path
import math
import torch
from torch.utils.data import DataLoader, Subset

from training.common.checkpoint import (atomic_json,save_checkpoint,load_checkpoint,cpu_tree,
                                       seed_everything,capture_rng,restore_rng)
from training.common.config import canonical_hash
from training.common.device import select_device
from training.visual.action_model import ActionModel,loss_terms
from training.visual.head_adapt_train import AdaptModel,validate_settings


def run_epoch(model,loader,device,auxiliary,optimizer=None,accumulation_steps=1,clip=1.):
    training=optimizer is not None
    model.train(training)
    report={name:dict(loss_sum=0.,count=0) for name in ('fatigue','eye','yawn')}
    updates=0
    iterator=iter(loader)
    while True:
        batches=list(islice(iterator,accumulation_steps if training else 1))
        if not batches: break
        totals=dict(fatigue=0,eye=0,yawn=0)
        for inputs,target,known in batches:
            available=inputs['visual_mask'].all(1)
            totals['fatigue']+=int((known&available).sum())
            for col,name in enumerate(('eye','yawn')):
                totals[name]+=int((inputs['action_masks'][:,:,col]&available[:,None]).sum())
        if training: optimizer.zero_grad(set_to_none=True)
        backward=False
        with torch.set_grad_enabled(training):
            for inputs,target,known in batches:
                output=model(inputs['images'].to(device),inputs['visual_mask'].to(device))
                terms=loss_terms(output,target.to(device).float(),known.to(device),
                                 inputs['action_targets'].to(device),inputs['action_masks'].to(device))
                objective=None
                for name,(loss,count) in terms.items():
                    report[name]['loss_sum']+=float(loss.detach())
                    report[name]['count']+=count
                    weight=1. if name=='fatigue' else (.5 if auxiliary else 0.)
                    if training and count and weight:
                        term=weight*loss/totals[name]
                        objective=term if objective is None else objective+term
                if objective is not None:
                    objective.backward(); backward=True
        if training and backward:
            torch.nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=True)
            optimizer.step(); updates+=1
    for value in report.values():
        value['loss']=value['loss_sum']/value['count'] if value['count'] else None
    action_losses=[report[n]['loss'] for n in ('eye','yawn') if report[n]['count']]
    report.update(action_loss=sum(action_losses)/len(action_losses) if action_losses else None,updates=updates)
    return report


def active_indices(data):
    # Both arms see the same ordered/shuffled union; control simply has no
    # auxiliary loss. Unknown-only samples never generate pseudo-negatives.
    result=[]; counts=dict(fatigue=0,eye=0,yawn=0)
    positives=dict(eye=0,yawn=0); negatives=dict(eye=0,yawn=0)
    for i in range(len(data)):
        if not data.availability(i): continue
        row=data.metadata(i); actions,masks=data.supervision(i)
        if row['label_known'] or masks.any(): result.append(i)
        counts['fatigue']+=int(row['label_known'])
        for col,name in enumerate(('eye','yawn')):
            counts[name]+=int(masks[:,col].sum())
            positives[name]+=int((masks[:,col]&(actions[:,col]==1)).sum())
            negatives[name]+=int((masks[:,col]&(actions[:,col]==0)).sum())
    if not counts['fatigue'] or any(not positives[n] or not negatives[n] for n in positives):
        raise ValueError('Action training requires known fatigue and positive/negative eye and yawn labels')
    return result,dict(counts=counts,positive=positives,negative=negatives,unit='window-frame occurrences')


def _backbone(params,normalizer,state):
    probe=dict(model_params=params['probe_params'],provenance={'normalization':{'head':normalizer}},
               model_state={k[6:]:v for k,v in state.items() if k.startswith('probe.')})
    encoder={k[8:]:v for k,v in state.items() if k.startswith('encoder.')}
    model=AdaptModel(probe,encoder,params['frames'],params['tune_encoder'],params['frame_chunk'])
    model.load_state_dict(state,strict=True)
    return model


def load_actions(path,backend='cpu',device_index=0):
    payload=load_checkpoint(path)
    if (payload.get('model_kind')!='fatigue_actions_v1'
            or canonical_hash(payload['provenance'])!=payload['provenance_sha256']
            or payload['config']!=dict(model=payload['model_params'],training=payload['settings'],auxiliary=payload['auxiliary'])
            or type(payload['auxiliary'])!=bool
            or payload['selection_metric']!=('action_validation_bce' if payload['auxiliary'] else 'fatigue_validation_bce')):
        raise ValueError('Invalid action checkpoint identity')
    validate_settings(payload['settings'])
    if type(payload['epoch'])!=int or not 0<=payload['epoch']<=payload['settings']['epochs']:
        raise ValueError('Invalid action checkpoint epoch')
    if payload['model_params']['frames']!=32 or payload['model_params']['frame_chunk']!=payload['settings']['frame_chunk']:
        raise ValueError('Invalid action model geometry')
    state={k[len('backbone.'):]:v for k,v in payload['model_state'].items() if k.startswith('backbone.')}
    model=ActionModel(_backbone(payload['model_params'],payload['normalizer'],state))
    model.load_state_dict(payload['model_state'],strict=True)
    mean=torch.tensor(payload['normalizer']['mean'],dtype=torch.float64)
    std=torch.tensor(payload['normalizer']['std'],dtype=torch.float32)
    if mean.shape!=(512,) or std.shape!=(512,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std>0).all():
        raise ValueError('Invalid action normalization statistics')
    for name,expected in (('head_mean_hi',mean.float()),('head_mean_lo',(mean-mean.float().double()).float()),
                          ('head_std',std)):
        if not torch.equal(state[name],expected): raise ValueError('Action normalization mismatch')
    return model.to(select_device(backend,device_index)).eval(),payload


def train_actions(train,val,backbone,settings,output_dir,provenance,auxiliary,resume=None):
    settings=validate_settings(settings)
    if type(auxiliary)!=bool: raise ValueError('Expected auxiliary boolean')
    output=Path(output_dir).resolve()
    if output.exists(): raise FileExistsError('Use a NEW action training directory')
    active,coverage=active_indices(train)
    device=select_device(settings['backend'],settings['device_index'])
    seed_everything(settings['seed'],device)
    model=ActionModel(backbone).to(device)
    backbone.frame_chunk=settings['frame_chunk']
    backbone.model_params['frame_chunk']=settings['frame_chunk']
    groups=[]
    for is_encoder,lr,wd in ((False,settings['learning_rate'],settings['weight_decay']),
                             (True,settings['encoder_learning_rate'],settings['encoder_weight_decay'])):
        params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('backbone.encoder.')==is_encoder]
        if params: groups.append(dict(params=params,lr=lr,weight_decay=wd))
    optimizer=torch.optim.Adam(groups,foreach=False)
    generator=torch.Generator().manual_seed(settings['seed'])
    val_generator=torch.Generator().manual_seed(settings['seed'])
    train_loader=DataLoader(Subset(train,active),batch_size=settings['batch_size'],shuffle=True,generator=generator,num_workers=0)
    val_loader=DataLoader(val,batch_size=settings['batch_size'],shuffle=False,generator=val_generator,num_workers=0)
    metadata=dict(format_version=1,model_kind='fatigue_actions_v1',model_params=deepcopy(backbone.model_params),
                  settings=settings,auxiliary=auxiliary,normalizer=deepcopy(backbone.normalization),
                  provenance=deepcopy(provenance),provenance_sha256=canonical_hash(provenance),data_summary=coverage,
                  selection_metric='action_validation_bce' if auxiliary else 'fatigue_validation_bce')
    metadata['config']=dict(model=metadata['model_params'],training=settings,auxiliary=auxiliary)
    start=updates=best_epoch=fatigue_epoch=bad=0; history=[]
    def score(report):
        value=report['action_loss'] if auxiliary else report['fatigue']['loss']
        if value is None or not math.isfinite(value): raise ValueError('No finite supervised validation score')
        return value
    if resume is None:
        initial_report=run_epoch(model,val_loader,device,auxiliary)
        if initial_report['fatigue']['loss'] is None:
            raise ValueError('Action comparison requires known validation fatigue labels')
        initial_state=cpu_tree(model.state_dict())
        best_state=fatigue_state=initial_state
        best_loss=score(initial_report); fatigue_loss=initial_report['fatigue']['loss']
    else:
        _,previous=load_actions(resume)
        old=deepcopy(previous['settings']); current=deepcopy(settings)
        old_epochs=old.pop('epochs'); current_epochs=current.pop('epochs')
        if (previous['kind']!='training' or old!=current or current_epochs<old_epochs
                or previous['provenance']!=provenance or previous['auxiliary']!=auxiliary
                or previous['model_params']!=metadata['model_params'] or previous['data_summary']!=coverage
                or previous['normalizer']!=metadata['normalizer']):
            raise ValueError('Action resume configuration/provenance mismatch')
        if previous['bad_epochs']>=settings['early_stopping_patience'] and current_epochs!=old_epochs:
            raise ValueError('Cannot extend early-stopped action run')
        model.load_state_dict(previous['model_state'],strict=True)
        optimizer.load_state_dict(previous['optimizer_state'])
        initial_report,initial_state=previous['initial_report'],previous['initial_state']
        best_state,best_loss,best_epoch=previous['best_state'],previous['best_loss'],previous['best_epoch']
        fatigue_state,fatigue_loss,fatigue_epoch=previous['fatigue_state'],previous['fatigue_loss'],previous['fatigue_epoch']
        start,updates,bad,history=previous['epoch'],previous['updates'],previous['bad_epochs'],previous['history']
        if len(history)!=start or [r['epoch'] for r in history]!=list(range(1,start+1)):
            raise ValueError('Action resume history mismatch')
        restore_rng(previous['rng_state'],device,generator)
        val_generator.set_state(previous['rng_state']['val_loader'])
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'config.json',metadata['config'])
    atomic_json(output/'provenance.json',provenance)
    def weights(name,state,epoch,loss):
        selection='fatigue_validation_bce' if name=='fatigue-best' else metadata['selection_metric']
        save_checkpoint(output/(name+'.pt'),dict(metadata,kind='weights',model_state=state,epoch=epoch,selected_loss=loss,
                                                checkpoint_role=name,checkpoint_selection_metric=selection))
    weights('initial',initial_state,0,score(initial_report))
    def persist(epoch):
        rng=capture_rng(device,generator); rng['val_loader']=val_generator.get_state()
        save_checkpoint(output/'last.pt',dict(metadata,kind='training',model_state=model.state_dict(),epoch=epoch,
             optimizer_state=optimizer.state_dict(),rng_state=rng,initial_state=initial_state,initial_report=initial_report,
             best_state=best_state,best_loss=best_loss,best_epoch=best_epoch,fatigue_state=fatigue_state,
             fatigue_loss=fatigue_loss,fatigue_epoch=fatigue_epoch,updates=updates,bad_epochs=bad,history=history))
        weights('best',best_state,best_epoch,best_loss)
        weights('fatigue-best',fatigue_state,fatigue_epoch,fatigue_loss)
        atomic_json(output/'history.json',history)
    epoch=start
    persist(epoch)
    if bad<settings['early_stopping_patience']:
        for epoch in range(start+1,settings['epochs']+1):
            tr=run_epoch(model,train_loader,device,auxiliary,optimizer,settings['accumulation_steps'],settings['gradient_clip_norm'])
            va=run_epoch(model,val_loader,device,auxiliary)
            updates+=tr['updates']; value=score(va)
            history.append(dict(epoch=epoch,train=tr,val=va,selection_score=value))
            improved=value<best_loss-settings['early_stopping_min_delta']
            if improved: best_state,best_loss,best_epoch=cpu_tree(model.state_dict()),value,epoch
            bad=0 if improved else bad+1
            if va['fatigue']['loss']<fatigue_loss:
                fatigue_state,fatigue_loss,fatigue_epoch=cpu_tree(model.state_dict()),va['fatigue']['loss'],epoch
            persist(epoch)
            print(f"actions auxiliary={auxiliary} epoch={epoch}/{settings['epochs']} "
                  f"fatigue_bce={va['fatigue']['loss']:.6f} action_bce={va['action_loss']:.6f}",flush=True)
            if bad>=settings['early_stopping_patience']: break
    result=dict(completed_epochs=epoch,best_epoch=best_epoch,best_loss=best_loss,
                fatigue_best_epoch=fatigue_epoch,selection_metric=metadata['selection_metric'],
                stopped_early=bad>=settings['early_stopping_patience'],device=str(device))
    for name in ('initial','best','last','fatigue-best'): result[name.replace('-','_')+'_checkpoint']=str(output/(name+'.pt'))
    atomic_json(output/'result.json',result)
    return result
