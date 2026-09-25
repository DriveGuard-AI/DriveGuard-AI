"""Binary frozen-feature probe training with integrity and exact epoch resume."""
from copy import deepcopy
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from training.common.checkpoint import (atomic_json, capture_rng, cpu_tree, load_checkpoint,
                                      restore_rng, save_checkpoint, seed_everything)
from training.common.config import canonical_hash
from training.common.device import select_device
from training.visual.model import VisualProbe


def _settings(settings):
    settings = deepcopy(settings)
    required = {'epochs', 'batch_size', 'seed', 'learning_rate', 'weight_decay',
                'gradient_clip_norm', 'hidden_dim', 'dropout', 'backend', 'device_index'}
    optional = {'early_stopping_patience', 'early_stopping_min_delta', 'positive_weight'}
    if not required <= settings.keys() or not settings.keys() <= required | optional:
        raise ValueError('Probe settings must contain only the documented training settings')
    for key in ('epochs', 'batch_size', 'hidden_dim'):
        if type(settings[key]) is not int or settings[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('seed', 'device_index'):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if settings['seed'] >= 2**32:
        raise ValueError('seed must be below 2**32')
    for key in ('learning_rate', 'weight_decay', 'gradient_clip_norm', 'dropout'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
    if settings['learning_rate'] <= 0 or settings['weight_decay'] < 0 or settings['gradient_clip_norm'] <= 0:
        raise ValueError('Invalid learning rate, weight decay or gradient clipping')
    if not 0 <= settings['dropout'] < 1:
        raise ValueError('dropout must be in [0,1)')
    if settings['backend'] not in ('cpu', 'cuda', 'npu'):
        raise ValueError('Unknown backend')
    if 'early_stopping_patience' in settings:
        patience = settings['early_stopping_patience']
        if type(patience) is not int or patience <= 0:
            raise ValueError('early_stopping_patience must be a positive integer')
    if 'early_stopping_min_delta' in settings:
        delta = settings['early_stopping_min_delta']
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or delta < 0:
            raise ValueError('early_stopping_min_delta must be finite and nonnegative')
    weight = settings.get('positive_weight', 1.)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError('positive_weight must be finite and positive')
    return settings


def is_terminal(payload, settings=None):
    """Recognize a completed stage even if the process died just after last.pt."""
    settings = payload['settings'] if settings is None else settings
    return (bool(payload.get('early_stopping_state', {}).get('stopped_early', False))
            or payload['epoch'] >= settings['epochs'])


def _check_supervision(dataset, split):
    for index in range(len(dataset)):
        inputs, target, known = dataset[index]
        if bool(known) and bool(torch.as_tensor(inputs['visual_mask']).all()):
            if float(target) not in (0., 1.):
                raise ValueError('Known binary target must be 0 or 1')
            return
    raise ValueError(f'{split} has no usable supervision')


def run_epoch(model, loader, device, optimizer=None, gradient_clip_norm=1., positive_weight=1.):
    """Supervise known, available rows; retain all rows in coverage reporting."""
    training = optimizer is not None
    model.train(training)
    report = dict(windows=0, known_count=0, loss_sum=0., unavailable_windows=0,
                  unavailable_positive=0, updates=0, skipped_batches=0, batches=0,
                  label_coverage=dict(positive=0, negative=0, unknown=0))
    tn = fp = fn = tp = 0
    with torch.set_grad_enabled(training):
        for inputs, target, known in loader:
            report['batches'] += 1
            target, known = target.to(device), known.to(device).bool()
            available = inputs['visual_mask'].to(device).bool().all(dim=1)
            effective = known & available
            report['windows'] += len(target)
            report['unavailable_windows'] += int((~available).sum())
            report['unavailable_positive'] += int((known & (target == 1) & ~available).sum())
            report['label_coverage']['positive'] += int((known & (target == 1)).sum())
            report['label_coverage']['negative'] += int((known & (target == 0)).sum())
            report['label_coverage']['unknown'] += int((~known).sum())
            if not bool(effective.any()):
                report['skipped_batches'] += 1
                continue
            selected_target = target[effective]
            if not bool(((selected_target == 0) | (selected_target == 1)).all()):
                raise ValueError('Known binary target must be 0 or 1')
            # Selection before forward keeps unsupported rows out of dropout RNG.
            selected = {key: value.to(device)[effective] for key, value in inputs.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(**selected)
            logits = outputs['logits']
            # Validation remains unweighted so checkpoint selection is comparable.
            weight = logits.new_tensor(positive_weight) if training and positive_weight != 1. else None
            losses = F.binary_cross_entropy_with_logits(logits, selected_target, reduction='none', pos_weight=weight)
            loss = losses.mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite binary probe loss')
            report['loss_sum'] += float(losses.detach().double().sum())
            report['known_count'] += len(losses)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
                optimizer.step()
                report['updates'] += 1
            prediction = logits.detach() >= 0
            positive = selected_target == 1
            tp += int((prediction & positive).sum())
            fp += int((prediction & ~positive).sum())
            fn += int((~prediction & positive).sum())
            tn += int((~prediction & ~positive).sum())
    if not report['known_count'] or (training and not report['updates']):
        raise ValueError('Epoch has no usable supervision/updates')
    report['loss'] = report['loss_sum'] / report['known_count']
    report['metrics'] = dict(confusion_matrix=[[tn, fp], [fn, tp]], threshold=.5,
                            precision=tp / (tp + fp) if tp + fp else None,
                            recall=tp / (tp + fn) if tp + fn else None,
                            f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
    return report


def _validate_payload(payload):
    required = {'model_params', 'model_kind', 'settings', 'provenance', 'provenance_sha256',
                'selection_metric'}
    if not required <= payload.keys() or payload['model_kind'] != 'visual_probe_v1':
        raise ValueError('Invalid visual probe checkpoint metadata')
    if payload['selection_metric'] != 'binary_validation_bce':
        raise ValueError('Invalid visual probe selection metric')
    if canonical_hash(payload['provenance']) != payload['provenance_sha256']:
        raise ValueError('Checkpoint provenance metadata mismatch')
    if payload['data_summary'] != payload['provenance']:
        raise ValueError('Checkpoint provenance data summary mismatch')
    if payload['normalizer'] != payload['provenance'].get('normalization', {}):
        raise ValueError('Checkpoint provenance normalizer mismatch')
    if payload['config'] != {'model': payload['model_params'], 'training': payload['settings']}:
        raise ValueError('Checkpoint configuration metadata mismatch')
    patience = payload['settings'].get('early_stopping_patience')
    if payload['kind'] == 'training' and patience is not None:
        state = payload.get('early_stopping_state')
        if not isinstance(state, dict) or set(state) != {'best', 'bad_epochs', 'stopped_early'}:
            raise ValueError('Missing or invalid early_stopping_state')
        best, bad, stopped = state['best'], state['bad_epochs'], state['stopped_early']
        if (isinstance(best, bool) or not isinstance(best, (int, float)) or not math.isfinite(best)
                or best < 0 or type(bad) is not int or not 0 <= bad <= payload['epoch']
                or type(stopped) is not bool or stopped != (bad >= patience)):
            raise ValueError('Invalid early_stopping_state')


def load_probe(path, backend='cpu', device_index=0):
    """Load only verified CPU checkpoint data, then move the head to the device."""
    payload = load_checkpoint(path)
    _validate_payload(payload)
    model = VisualProbe(**payload['model_params'])
    model.load_state_dict(payload['model_state'], strict=True)
    model.to(select_device(backend, device_index)).eval()
    return model, payload


def train_probe(train_dataset, val_dataset, mode, settings, output_dir, provenance, resume=None):
    settings = _settings(settings)
    provenance = deepcopy(provenance)
    provenance_hash = canonical_hash(provenance)
    params = dict(visual_dim=train_dataset.visual_dim, context_dim=train_dataset.context_dim,
                  hidden_dim=settings['hidden_dim'], dropout=settings['dropout'], mode=mode)
    if (val_dataset.visual_dim, val_dataset.context_dim) != (params['visual_dim'], params['context_dim']):
        raise ValueError('Train/validation feature dimension mismatch')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a NEW output_dir; existing artifacts are never overwritten')
    _check_supervision(train_dataset, 'train')
    _check_supervision(val_dataset, 'val')
    previous = load_checkpoint(resume) if resume is not None else None
    if previous is not None:
        _validate_payload(previous)
        required = {'optimizer_state', 'global_step', 'rng_state', 'best_model_state',
                    'best_epoch', 'best_val_loss', 'history'}
        if previous['kind'] != 'training' or not required <= previous.keys():
            raise ValueError('Resume requires a complete last.pt training checkpoint')
        if previous['provenance'] != provenance:
            raise ValueError('Resume provenance mismatch')
        old = dict(previous['settings'])
        old_epochs = old.pop('epochs')
        current = dict(settings)
        current.pop('epochs')
        if old != current or previous['model_params'] != params:
            raise ValueError('Resume configuration mismatch; only total epochs/output may change')
        if previous.get('early_stopping_state', {}).get('stopped_early', False):
            raise ValueError('Cannot resume an early-stopped checkpoint')
        if settings['epochs'] < old_epochs or settings['epochs'] <= previous['epoch']:
            raise ValueError('Resume total epochs must increase past the completed epoch')
        if 'val_loader' not in previous['rng_state']:
            raise ValueError('Resume requires validation loader RNG state')
    device = select_device(settings['backend'], settings['device_index'])
    seed_everything(settings['seed'], device)
    model = VisualProbe(**params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'], foreach=False)
    generator = torch.Generator().manual_seed(settings['seed'])
    val_generator = torch.Generator().manual_seed(settings['seed'])
    history = []
    start = step = best_epoch = 0
    best_loss = float('inf')
    best_state = None
    patience = settings.get('early_stopping_patience')
    min_delta = settings.get('early_stopping_min_delta', 0.)
    early_stopping = dict(best=float('inf'), bad_epochs=0, stopped_early=False) if patience else None
    if previous is not None:
        model.load_state_dict(previous['model_state'], strict=True)
        optimizer.load_state_dict(previous['optimizer_state'])
        history = deepcopy(previous['history'])
        start, step = previous['epoch'], previous['global_step']
        best_epoch, best_loss = previous['best_epoch'], previous['best_val_loss']
        best_state = previous['best_model_state']
        if patience:
            early_stopping = deepcopy(previous['early_stopping_state'])
        restore_rng(previous['rng_state'], device, generator)
        val_generator.set_state(previous['rng_state']['val_loader'])
    train_loader = DataLoader(train_dataset, batch_size=settings['batch_size'], shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=settings['batch_size'], shuffle=False,
                            generator=val_generator, num_workers=0)
    base = dict(format_version=1, model_kind='visual_probe_v1', model_params=params,
                config={'model': params, 'training': settings}, settings=settings,
                normalizer=provenance.get('normalization', {}), data_summary=provenance,
                provenance=provenance, provenance_sha256=provenance_hash,
                selection_metric='binary_validation_bce')
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'config.json', base['config'])
    atomic_json(output/'provenance.json', provenance)
    for epoch in range(start + 1, settings['epochs'] + 1):
        weight_options = ({'positive_weight': settings['positive_weight']}
                          if settings.get('positive_weight', 1.) != 1. else {})
        train_report = run_epoch(model, train_loader, device, optimizer, settings['gradient_clip_norm'],
                                 **weight_options)
        val_report = run_epoch(model, val_loader, device)
        step += train_report['updates']
        history.append(dict(epoch=epoch, global_step=step, train=train_report, val=val_report))
        if val_report['loss'] < best_loss:
            best_loss, best_epoch = val_report['loss'], epoch
            best_state = cpu_tree(model.state_dict())
        if early_stopping is not None:
            if val_report['loss'] < early_stopping['best'] - min_delta:
                early_stopping.update(best=val_report['loss'], bad_epochs=0)
            else:
                early_stopping['bad_epochs'] += 1
            early_stopping['stopped_early'] = early_stopping['bad_epochs'] >= patience
        rng = capture_rng(device, generator)
        rng['val_loader'] = val_generator.get_state()
        stopping_metadata = {'early_stopping_state': early_stopping} if early_stopping is not None else {}
        save_checkpoint(output/'last.pt', dict(base, kind='training', epoch=epoch, global_step=step,
                        model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                        best_val_loss=best_loss, best_epoch=best_epoch, best_model_state=best_state,
                        rng_state=rng, history=history, **stopping_metadata))
        save_checkpoint(output/'best.pt', dict(base, kind='weights', epoch=best_epoch,
                        model_state=best_state, best_val_loss=best_loss))
        atomic_json(output/'history.json', history)
        print(f"{mode} epoch {epoch}/{settings['epochs']} step={step} "
              f"train_bce={train_report['loss']:.6f} val_bce={val_report['loss']:.6f}", flush=True)
        if early_stopping is not None and early_stopping['stopped_early']:
            break
    result = dict(completed_epochs=epoch, max_epochs=settings['epochs'],
                  stopped_early=bool(early_stopping and early_stopping['stopped_early']),
                  global_step=step, mode=mode,
                  best_epoch=best_epoch, best_val_loss=best_loss, selection_metric='binary_validation_bce',
                  best_checkpoint=str(output/'best.pt'), last_checkpoint=str(output/'last.pt'), device=str(device))
    atomic_json(output/'result.json', result)
    return result
