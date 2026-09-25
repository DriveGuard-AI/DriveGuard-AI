"""Warm-start RGB head adaptation with frozen BN and exact epoch continuation."""
from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from training.common.checkpoint import (atomic_json, capture_rng, cpu_tree, load_checkpoint,
                                       restore_rng, save_checkpoint, seed_everything)
from training.common.config import canonical_hash
from training.common.device import select_device
from training.visual.encoder import ResNet18
from training.visual.model import VisualProbe


DEFAULTS = dict(epochs=10, batch_size=2, accumulation_steps=8, frame_chunk=16, seed=42,
                learning_rate=1e-4, encoder_learning_rate=1e-5, weight_decay=.01,
                encoder_weight_decay=1e-4, gradient_clip_norm=1., backend='cpu', device_index=0,
                early_stopping_patience=3, early_stopping_min_delta=0.)


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) - set(DEFAULTS):
        raise ValueError('Unknown adaptation settings')
    result = dict(DEFAULTS, **settings)
    for key in ('epochs', 'batch_size', 'accumulation_steps', 'frame_chunk', 'early_stopping_patience'):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('seed', 'device_index'):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if result['seed'] >= 2**32:
        raise ValueError('seed must be below 2**32')
    for key in ('learning_rate', 'encoder_learning_rate', 'weight_decay', 'encoder_weight_decay',
                'gradient_clip_norm', 'early_stopping_min_delta'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid {key}')
        if key in ('learning_rate', 'encoder_learning_rate', 'gradient_clip_norm') and value == 0:
            raise ValueError(f'{key} must be positive')
    if result['backend'] not in ('cpu', 'cuda', 'npu'):
        raise ValueError('Unknown backend')
    return result


class AdaptModel(nn.Module):
    def __init__(self, probe_payload, encoder_state, frames, tune_encoder, frame_chunk=16):
        super().__init__()
        if type(frames) is not int or frames not in (8, 32) or type(tune_encoder) is not bool:
            raise ValueError('Adaptation requires 8/32 frames and boolean tune_encoder')
        if type(frame_chunk) is not int or frame_chunk <= 0:
            raise ValueError('frame_chunk must be positive')
        params = deepcopy(probe_payload['model_params'])
        if params['visual_dim'] != 1024 or params['mode'] != 'visual':
            raise ValueError('Adaptation requires the 1024-D head-only visual probe')
        normalization = deepcopy(probe_payload['provenance']['normalization']['head'])
        mean, std = (np.asarray(normalization[k], np.float64) for k in ('mean', 'std'))
        if (mean.shape != (512,) or std.shape != (512,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError('Invalid frozen head normalization')
        self.normalization = normalization
        self.frames, self.tune_encoder, self.frame_chunk = frames, tune_encoder, frame_chunk
        self.model_params = dict(frames=frames, tune_encoder=tune_encoder, frame_chunk=frame_chunk,
                                 probe_params=params)
        self.encoder = ResNet18()
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.encoder.requires_grad_(False)
        if tune_encoder:
            for module in self.encoder.layer4.modules():
                if isinstance(module, nn.Conv2d):
                    module.requires_grad_(True)
        self.probe = VisualProbe(**params)
        self.probe.load_state_dict(probe_payload['model_state'], strict=True)
        hidden = params['hidden_dim']
        self.extra = nn.Conv1d(hidden, hidden, 3, dilation=9)
        self.gate = nn.Parameter(torch.zeros(()))
        hi = mean.astype(np.float32)
        self.register_buffer('head_mean_hi', torch.from_numpy(hi))
        self.register_buffer('head_mean_lo', torch.from_numpy((mean-hi.astype(np.float64)).astype(np.float32)))
        self.register_buffer('head_std', torch.from_numpy(std.astype(np.float32)))
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406]).reshape(1,3,1,1))
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225]).reshape(1,3,1,1))
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        # Frozen statistics AND affine parameters, including the tuned layer4 BN.
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                module.requires_grad_(False)
        return self

    def forward(self, images, visual_mask):
        if images.dtype != torch.uint8 or images.ndim != 5 or tuple(images.shape[1:]) != (self.frames,3,224,224):
            raise ValueError('Expected uint8 images [batch,frames,3,224,224]')
        if visual_mask.dtype != torch.bool or tuple(visual_mask.shape) != tuple(images.shape[:2]):
            raise ValueError('Expected boolean visual_mask [batch,frames]')
        available = visual_mask.all(1)
        selected = images[available].flatten(0,1)
        if not len(selected):
            return dict(logits=self.gate.new_zeros(len(images)), available=available)
        encoded = []
        for chunk in selected.split(self.frame_chunk):
            pixels = (chunk.float() / 255. - self.rgb_mean) / self.rgb_std
            encoded.append(self.encoder(pixels))
        feature = torch.cat(encoded).reshape(-1, self.frames, 512)
        feature = (((feature-self.head_mean_hi)-self.head_mean_lo)/self.head_std).clamp(-5,5)
        visual = torch.cat((torch.zeros_like(feature), feature), -1)
        hidden = F.relu(self.probe.visual_projection(visual)).transpose(1,2)
        for layer in self.probe.temporal:
            hidden = hidden + self.probe.dropout(F.relu(layer(F.pad(hidden, (2*layer.dilation[0],0)))))
        hidden = hidden + self.gate * torch.tanh(self.extra(F.pad(hidden, (18,0))))
        active_logits = self.probe.head(self.probe.dropout(hidden[:,:,-1])).squeeze(-1)
        logits = active_logits.new_zeros(len(images)).masked_scatter(available, active_logits)
        return dict(logits=logits, available=available)


def _active_indices(dataset, split):
    active = []
    for i in range(len(dataset)):
        if hasattr(dataset, 'metadata') and hasattr(dataset, 'availability'):
            metadata = dataset.metadata(i)
            target, known = metadata['target'], metadata['label_known']
            available = bool(dataset.availability(i)) if known else False
        else:
            inputs, target, known = dataset[i]
            available = bool(torch.as_tensor(inputs['visual_mask']).all())
        if bool(known):
            if float(target) not in (0., 1.):
                raise ValueError('Known binary target must be 0 or 1')
            if available:
                active.append(i)
    if not active:
        raise ValueError(f'{split} has no usable supervision')
    return active


def run_epoch(model, loader, device, optimizer=None, gradient_clip_norm=1., accumulation_steps=1):
    training = optimizer is not None
    model.train(training)
    report = dict(windows=0, known_count=0, loss_sum=0., updates=0, batches=0, skipped_batches=0,
                  unavailable_windows=0, unavailable_positive=0,
                  label_coverage=dict(positive=0, negative=0, unknown=0))
    tn = fp = fn = tp = pending_rows = pending_batches = 0
    if training:
        optimizer.zero_grad(set_to_none=True)

    def step(rows):
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(rows)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        report['updates'] += 1

    with torch.set_grad_enabled(training):
        for inputs, target, known in loader:
            target = target.to(device).float()
            known = known.to(device).bool()
            available = inputs['visual_mask'].to(device).bool().all(1)
            active = known & available
            report['batches'] += 1
            report['windows'] += len(target)
            report['unavailable_windows'] += int((~available).sum())
            report['unavailable_positive'] += int((known & (target == 1) & ~available).sum())
            report['label_coverage']['positive'] += int((known & (target == 1)).sum())
            report['label_coverage']['negative'] += int((known & (target == 0)).sum())
            report['label_coverage']['unknown'] += int((~known).sum())
            if not bool(active.any()):
                report['skipped_batches'] += 1
                continue
            target = target[active]
            if not bool(((target == 0) | (target == 1)).all()):
                raise ValueError('Known binary target must be 0 or 1')
            selected = {k:v.to(device)[active] for k,v in inputs.items()}
            logits = model(**selected)['logits']
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction='sum')
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite adaptation loss')
            report['loss_sum'] += float(loss.detach())
            report['known_count'] += len(target)
            if training:
                loss.backward()
                pending_rows += len(target)
                pending_batches += 1
                if pending_batches == accumulation_steps:
                    step(pending_rows)
                    pending_rows = pending_batches = 0
            positive, predicted = target == 1, logits.detach() >= 0
            tp += int((positive & predicted).sum()); fp += int((~positive & predicted).sum())
            fn += int((positive & ~predicted).sum()); tn += int((~positive & ~predicted).sum())
    if training and pending_rows:
        step(pending_rows)
    if not report['known_count']:
        raise ValueError('Epoch has no usable supervision')
    report['loss'] = report['loss_sum']/report['known_count']
    report['metrics'] = dict(confusion_matrix=[[tn,fp],[fn,tp]], threshold=.5)
    return report


def validate_adaptation(payload):
    required = {'model_kind','model_params','settings','provenance','provenance_sha256',
                'normalizer','data_summary','config','selection_metric'}
    if not required <= payload.keys() or payload['model_kind'] != 'head_adaptation_v1':
        raise ValueError('Invalid adaptation checkpoint metadata')
    if (canonical_hash(payload['provenance']) != payload['provenance_sha256']
            or payload['data_summary'] != payload['provenance']):
        raise ValueError('Adaptation provenance mismatch')
    if payload['selection_metric'] != 'binary_validation_bce':
        raise ValueError('Invalid adaptation selection metric')
    if (validate_settings(payload['settings']) != payload['settings'] or
            payload['config'] != dict(model=payload['model_params'], training=payload['settings'])):
        raise ValueError('Adaptation configuration mismatch')
    if payload['model_params']['frame_chunk'] != payload['settings']['frame_chunk']:
        raise ValueError('Adaptation frame chunk mismatch')
    params = payload['model_params']
    if (set(params) != {'frames','tune_encoder','frame_chunk','probe_params'} or
            type(params['frames']) is not int or params['frames'] not in (8,32) or
            type(params['tune_encoder']) is not bool or params['probe_params']['visual_dim'] != 1024 or
            params['probe_params']['mode'] != 'visual'):
        raise ValueError('Invalid adaptation model parameters')
    mean = np.asarray(payload['normalizer']['mean'], np.float64)
    std = np.asarray(payload['normalizer']['std'], np.float64)
    if (mean.shape != (512,) or std.shape != (512,) or not np.isfinite(mean).all() or
            not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('Invalid adaptation normalization')
    hi = mean.astype(np.float32)
    expected_buffers = dict(head_mean_hi=torch.from_numpy(hi),
                            head_mean_lo=torch.from_numpy((mean-hi.astype(np.float64)).astype(np.float32)),
                            head_std=torch.from_numpy(std.astype(np.float32)))
    for key, expected in expected_buffers.items():
        for state_key in ('model_state','initial_model_state','best_model_state'):
            if state_key in payload:
                actual = payload[state_key].get(key)
                if not isinstance(actual, torch.Tensor) or not torch.equal(actual, expected):
                    raise ValueError('Adaptation normalization state mismatch')
    source = payload['provenance'].get('source')
    if source is not None and (not isinstance(source, dict) or
            payload['normalizer'] != source.get('normalization', {}).get('head')):
        raise ValueError('Adaptation source normalization mismatch')
    epoch = payload['epoch']
    if type(epoch) is not int or not 0 <= epoch <= payload['settings']['epochs']:
        raise ValueError('Invalid adaptation checkpoint epoch')
    if payload['kind'] == 'training':
        required = {'optimizer_state','global_step','rng_state','best_model_state','best_epoch',
                    'best_val_loss','initial_model_state','initial_val_report','history','early_stopping_state'}
        if not required <= payload.keys():
            raise ValueError('Incomplete adaptation training state')
        if (len(payload['history']) != epoch or
                [row['epoch'] for row in payload['history']] != list(range(1,epoch+1)) or
                type(payload['best_epoch']) is not int or not 0 <= payload['best_epoch'] <= epoch or
                type(payload['global_step']) is not int or payload['global_step'] < 0 or
                'val_loader' not in payload['rng_state']):
            raise ValueError('Invalid adaptation epoch/history/RNG state')
        state = payload['early_stopping_state']
        if (set(state) != {'best','bad_epochs','stopped_early'} or
                not math.isfinite(state['best']) or state['best'] < 0 or
                type(state['bad_epochs']) is not int or not 0 <= state['bad_epochs'] <= payload['epoch'] or
                type(state['stopped_early']) is not bool or
                state['stopped_early'] != (state['bad_epochs'] >= payload['settings']['early_stopping_patience'])):
            raise ValueError('Invalid adaptation early stopping state')
    return payload


def load_adaptation(path, backend='cpu', device_index=0):
    payload = validate_adaptation(load_checkpoint(path))
    state = payload['model_state']
    params = payload['model_params']
    probe_payload = dict(model_params=params['probe_params'],
                         model_state={k[6:]:v for k,v in state.items() if k.startswith('probe.')},
                         provenance={'normalization': {'head': payload['normalizer']}})
    encoder = {k[8:]:v for k,v in state.items() if k.startswith('encoder.')}
    model = AdaptModel(probe_payload, encoder, params['frames'], params['tune_encoder'], params['frame_chunk'])
    model.load_state_dict(state, strict=True)
    model.to(select_device(backend, device_index)).eval()
    return model, payload


def evaluate_model(dataset, model, batch_size=2):
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if dataset.frames != model.frames:
        raise ValueError('Evaluation frame count mismatch')
    rows = [dict(dataset.metadata(i)) for i in range(len(dataset))]
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            items = [dataset[i][0] for i in range(start, min(start+batch_size, len(dataset)))]
            inputs = {k:torch.stack([torch.as_tensor(item[k]) for item in items]).to(device) for k in items[0]}
            output = model(**inputs)
            for i, (logit, available) in enumerate(zip(output['logits'].cpu().tolist(), output['available'].cpu().tolist())):
                if available and not math.isfinite(logit):
                    raise ValueError('Nonfinite adaptation logit')
                rows[start+i].update(logit=float(logit) if available else None, available=bool(available))
    return rows


def evaluate_adaptation(dataset, checkpoint, backend='cpu', device_index=0, batch_size=2):
    model, payload = load_adaptation(checkpoint, backend, device_index)
    _verify_dataset_identity(dataset, payload['provenance'])
    return evaluate_model(dataset, model, batch_size), payload


def _verify_dataset_identity(dataset, provenance):
    if hasattr(dataset, 'cache'):
        if (provenance.get('frame_cache_sha256') != dataset.cache.manifest_sha256 or
                provenance.get('frame_cache') is None or
                Path(provenance['frame_cache']).resolve() != Path(dataset.cache.root).resolve()):
            raise ValueError('Adaptation pixel cache provenance mismatch')
    if hasattr(dataset, 'prepared'):
        source = provenance.get('source', {})
        expected = dataset.prepared.provenance
        if any(source.get(key) != expected.get(key) for key in ('prepared_sha256','preparation')):
            raise ValueError('Adaptation prepared provenance mismatch')


def train_adaptation(train, val, probe_payload, encoder_state, settings, output_dir, provenance,
                     frames, tune_encoder, resume=None):
    settings = validate_settings(settings)
    provenance = deepcopy(provenance)
    if train.frames != frames or val.frames != frames:
        raise ValueError('Dataset frame count mismatch')
    _verify_dataset_identity(train, provenance)
    _verify_dataset_identity(val, provenance)
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a NEW output_dir; existing artifacts are never overwritten')
    active = _active_indices(train, 'train')
    _active_indices(val, 'val')
    previous = validate_adaptation(load_checkpoint(resume)) if resume else None
    device = select_device(settings['backend'], settings['device_index'])
    seed_everything(settings['seed'], device)
    model = AdaptModel(probe_payload, encoder_state, frames, tune_encoder, settings['frame_chunk']).to(device)
    params = model.model_params
    if previous is not None:
        if previous['kind'] != 'training':
            raise ValueError('Resume requires a training checkpoint')
        old, current = dict(previous['settings']), dict(settings)
        old_epochs, new_epochs = old.pop('epochs'), current.pop('epochs')
        if (old != current or previous['model_params'] != params or previous['provenance'] != provenance
                or previous['normalizer'] != model.normalization):
            raise ValueError('Resume configuration/provenance mismatch')
        if new_epochs < old_epochs or new_epochs < previous['epoch']:
            raise ValueError('Resume epochs cannot shrink')
        if previous['early_stopping_state']['stopped_early'] and new_epochs != old_epochs:
            raise ValueError('Cannot extend an early-stopped checkpoint')
    head_parameters = [p for name,p in model.named_parameters() if p.requires_grad and not name.startswith('encoder.')]
    groups = [dict(params=head_parameters, lr=settings['learning_rate'], weight_decay=settings['weight_decay'])]
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    if encoder_parameters:
        groups.append(dict(params=encoder_parameters, lr=settings['encoder_learning_rate'], weight_decay=settings['encoder_weight_decay']))
    optimizer = torch.optim.Adam(groups, foreach=False)
    generator = torch.Generator().manual_seed(settings['seed'])
    val_generator = torch.Generator().manual_seed(settings['seed'])
    # Filter BEFORE shuffling, so unsupported rows cannot change order or dropout.
    train_loader = DataLoader(Subset(train, active), batch_size=settings['batch_size'], shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val, batch_size=settings['batch_size'], shuffle=False, generator=val_generator, num_workers=0)
    start = global_step = best_epoch = 0
    history = []
    if previous is None:
        initial_report = run_epoch(model, val_loader, device)
        initial_state = cpu_tree(model.state_dict())
        best_state, best_loss = initial_state, initial_report['loss']
        stopping = dict(best=best_loss, bad_epochs=0, stopped_early=False)
    else:
        model.load_state_dict(previous['model_state'], strict=True)
        optimizer.load_state_dict(previous['optimizer_state'])
        initial_state, initial_report = previous['initial_model_state'], previous['initial_val_report']
        best_state, best_loss, best_epoch = previous['best_model_state'], previous['best_val_loss'], previous['best_epoch']
        start, global_step, history = previous['epoch'], previous['global_step'], deepcopy(previous['history'])
        stopping = deepcopy(previous['early_stopping_state'])
        restore_rng(previous['rng_state'], device, generator)
        val_generator.set_state(previous['rng_state']['val_loader'])
    base = dict(format_version=1, model_kind='head_adaptation_v1', model_params=params,
                config=dict(model=params, training=settings), settings=settings, normalizer=model.normalization,
                data_summary=provenance, provenance=provenance, provenance_sha256=canonical_hash(provenance),
                selection_metric='binary_validation_bce')
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'config.json', base['config'])
    atomic_json(output/'provenance.json', provenance)
    save_checkpoint(output/'initial.pt', dict(base, kind='weights', epoch=0, model_state=initial_state,
                                             best_val_loss=initial_report['loss']))

    def persist(epoch):
        rng = capture_rng(device, generator)
        rng['val_loader'] = val_generator.get_state()
        save_checkpoint(output/'last.pt', dict(base, kind='training', epoch=epoch, global_step=global_step,
                        model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                        initial_model_state=initial_state, initial_val_report=initial_report,
                        best_model_state=best_state, best_val_loss=best_loss, best_epoch=best_epoch,
                        history=history, early_stopping_state=stopping, rng_state=rng))
        save_checkpoint(output/'best.pt', dict(base, kind='weights', epoch=best_epoch,
                        model_state=best_state, best_val_loss=best_loss))
        atomic_json(output/'history.json', history)

    epoch = start
    persist(epoch)
    if not stopping['stopped_early']:
        for epoch in range(start+1, settings['epochs']+1):
            train_report = run_epoch(model, train_loader, device, optimizer, settings['gradient_clip_norm'], settings['accumulation_steps'])
            val_report = run_epoch(model, val_loader, device)
            global_step += train_report['updates']
            history.append(dict(epoch=epoch, global_step=global_step, train=train_report, val=val_report))
            if val_report['loss'] < best_loss:
                best_state, best_loss, best_epoch = cpu_tree(model.state_dict()), val_report['loss'], epoch
            if val_report['loss'] < stopping['best'] - settings['early_stopping_min_delta']:
                stopping.update(best=val_report['loss'], bad_epochs=0)
            else:
                stopping['bad_epochs'] += 1
            stopping['stopped_early'] = stopping['bad_epochs'] >= settings['early_stopping_patience']
            persist(epoch)
            print(f"head-adapt frames={frames} tune={tune_encoder} epoch={epoch}/{settings['epochs']} "
                  f"train_bce={train_report['loss']:.6f} val_bce={val_report['loss']:.6f}", flush=True)
            if stopping['stopped_early']:
                break
    result = dict(completed_epochs=epoch, max_epochs=settings['epochs'], stopped_early=stopping['stopped_early'],
                  global_step=global_step, best_epoch=best_epoch, best_val_loss=best_loss,
                  initial_val_loss=initial_report['loss'], selection_metric='binary_validation_bce',
                  initial_checkpoint=str(output/'initial.pt'), best_checkpoint=str(output/'best.pt'),
                  last_checkpoint=str(output/'last.pt'), device=str(device))
    atomic_json(output/'result.json', result)
    return result
