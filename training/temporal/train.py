"""Single-device FP32 training shared by CPU smoke, CUDA and Ascend entrypoints."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.common.checkpoint import (atomic_json, capture_rng, cpu_tree, load_checkpoint,
                                       restore_rng, save_checkpoint, seed_everything, validate_resume)
from training.common.config import TASKS, canonical_hash, validate_config
from training.common.data import FeatureCorpus
from training.common.device import select_device
from training.temporal.losses import masked_bce_with_logits
from training.temporal.metrics import ClassificationMetrics
from training.common.model_factory import build_model, model_kind


def run_epoch(model, loader, device, active_mask, optimizer=None, gradient_clip_norm=1.0):
    training = optimizer is not None
    model.train(training)
    active_mask = active_mask.to(device)
    sums = torch.zeros(4, dtype=torch.float64)
    counts = torch.zeros(4, dtype=torch.int64)
    metrics = ClassificationMetrics(active_mask)
    groups = {key: {} for key in ('subject_id', 'subset', 'view')}
    skipped = updates = batches = offset = 0
    unavailable_windows = 0
    unavailable_positive = torch.zeros(4,dtype=torch.int64)
    label_coverage = ClassificationMetrics(active_mask)
    with torch.set_grad_enabled(training):
        for x, target, mask in loader:
            batches += 1
            target, mask = target.to(device), mask.to(device)
            batch_size = len(target)
            label_coverage.update_coverage(target,mask)
            if isinstance(x,dict):
                x = {key:value.to(device) for key,value in x.items()}
                available = x['eligible'].any(dim=-1)
                unavailable_windows += int((~available).sum().item())
                unavailable_positive += ((target == 1) & mask.bool() & ~available[:,None]).sum(0).cpu()
                effective_mask = mask.bool() & available[:,None]
                # Original labels are retained in coverage, not relabeled as negatives.
                target = target.masked_fill(~effective_mask,0)
                mask = effective_mask
            else:
                x = x.to(device)
            # Do not even consume dropout RNG or step Adam on unsupervised training batches.
            if training and not (mask.bool() & active_mask).any():
                metrics.update_coverage(target, mask)
                skipped += 1
                offset += batch_size
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(**x) if isinstance(x,dict) else model(x)
            loss, class_sums, class_counts = masked_bce_with_logits(logits, target, mask, active_mask)
            if loss is None:
                skipped += 1
            elif training:
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
            sums += class_sums.cpu().double()
            counts += class_counts.cpu()
            metrics.update(logits, target, mask)
            if not training and hasattr(loader.dataset, 'metadata'):
                for i in range(batch_size):
                    metadata = loader.dataset.metadata(offset + i)
                    for key in groups:
                        group = groups[key].setdefault(metadata[key], ClassificationMetrics(active_mask))
                        group.update(logits[i:i+1], target[i:i+1], mask[i:i+1])
            offset += batch_size
    supported = counts > 0
    if not supported.any() or (training and updates == 0):
        raise ValueError('Epoch has no valid supervision/updates')
    return {'loss': float((sums[supported] / counts[supported]).mean()),
            'loss_sums': sums.tolist(), 'known_counts': counts.tolist(), 'metrics': metrics.report(),
            'updates': updates, 'skipped_batches': skipped, 'batches': batches,
            'unavailable_windows': unavailable_windows,
            'unavailable_known_positive': unavailable_positive.tolist(),
            'label_coverage': {'positive':label_coverage.positive.tolist(),
                               'negative':label_coverage.negative.tolist(),
                               'unknown':label_coverage.unknown.tolist()},
            'groups': {key: {name: value.report() for name, value in entries.items()}
                       for key, entries in groups.items()}}


def environment_info(device):
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, timeout=5).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True, timeout=5))
    except (OSError, subprocess.SubprocessError):
        commit, dirty = 'unavailable', None
    info = {'python': platform.python_version(), 'torch': str(torch.__version__),
            'numpy': str(np.__version__), 'platform': platform.platform(), 'device': str(device),
            'git_revision': commit, 'git_dirty': dirty, 'recorded_at': datetime.now(timezone.utc).isoformat()}
    if device.type in ('cuda', 'npu'):
        info['device_name'] = getattr(torch, device.type).get_device_name(device)
    if device.type == 'cuda':
        info['cuda'] = str(torch.version.cuda)
    if device.type == 'npu':
        import torch_npu
        info['torch_npu'] = str(torch_npu.__version__)
    return info


def run_training(config, resume=None):
    config = deepcopy(validate_config(config))
    runtime, settings = config['runtime'], config['training']
    if not runtime['output_dir'] or not runtime['manifest']:
        raise ValueError('Training requires manifest and output_dir')
    output = Path(runtime['output_dir']).resolve()
    if output.exists():
        raise FileExistsError('Use a NEW output_dir; existing run artifacts are never overwritten')
    device = select_device(runtime['backend'], runtime['device_index'])
    corpus = FeatureCorpus(config, runtime['manifest'], runtime['data_root'])
    try:
        normalizer = corpus.fit_normalizer()
        coverage = corpus.check_supervision(settings['active_tasks'],normalizer)
        previous = load_checkpoint(resume) if resume else None
        if previous is not None:
            validate_resume(previous, config, corpus.summary, normalizer)
        seed_everything(settings['seed'], device)
        model = build_model(config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'],
                                     weight_decay=settings['weight_decay'], foreach=False)
        generator = torch.Generator().manual_seed(settings['seed'])
        active = torch.tensor([name in settings['active_tasks'] for name in TASKS], dtype=torch.bool, device=device)
        history, start_epoch, step = [], 0, 0
        best_loss, best_epoch, best_state = float('inf'), 0, None
        if previous is not None:
            model.load_state_dict(previous['model_state'], strict=True)
            optimizer.load_state_dict(previous['optimizer_state'])
            history = deepcopy(previous['history'])
            start_epoch, step = previous['epoch'], previous['global_step']
            best_loss, best_epoch, best_state = previous['best_val_loss'], previous['best_epoch'], previous['best_model_state']
            restore_rng(previous['rng_state'], device, generator)
        train_data, val_data = corpus.dataset('train', normalizer), corpus.dataset('val', normalizer)
        train_loader = DataLoader(train_data, batch_size=settings['batch_size'], shuffle=True, generator=generator, num_workers=0)
        # Validation's iterator RNG must not perturb training's dropout RNG.
        val_generator = torch.Generator().manual_seed(settings['seed'])
        val_loader = DataLoader(val_data, batch_size=settings['batch_size'], shuffle=False, generator=val_generator, num_workers=0)
        environment = environment_info(device)
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / 'config.json', config)
        atomic_json(output / 'data_summary.json', {'data': corpus.summary, 'supervision': coverage})
        atomic_json(output / 'environment.json', environment)
        base = {'format_version': 1, 'config': config, 'normalizer': normalizer,
                'data_summary': corpus.summary, 'environment': environment,
                'active_tasks': list(settings['active_tasks']), 'task_order': list(TASKS),
                'feature_schema': corpus.manifest['feature_schema'],
                'extractor_profile': corpus.manifest['extractor_profile'],
                'label_mapping': corpus.manifest['label_mapping']}
        base['model_kind'] = model_kind(config)
        if corpus.relation:
            base['group_profile'] = config['data']['group_profile']
            base['group_profile_sha256'] = canonical_hash(base['group_profile'])
        for epoch in range(start_epoch + 1, settings['epochs'] + 1):
            train_report = run_epoch(model, train_loader, device, active, optimizer, settings['gradient_clip_norm'])
            val_report = run_epoch(model, val_loader, device, active)
            step += train_report['updates']
            if val_report['loss'] < best_loss:
                best_loss, best_epoch, best_state = val_report['loss'], epoch, cpu_tree(model.state_dict())
            history.append({'epoch': epoch, 'global_step': step, 'train': train_report, 'val': val_report})
            payload = {**base, 'kind': 'training', 'epoch': epoch, 'global_step': step,
                       'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                       'best_val_loss': best_loss, 'best_epoch': best_epoch, 'best_model_state': best_state,
                       'rng_state': capture_rng(device, generator), 'history': history}
            save_checkpoint(output / 'last.pt', payload)
            save_checkpoint(output / 'best.pt', {**base, 'kind': 'weights', 'epoch': best_epoch,
                                                'model_state': best_state, 'best_val_loss': best_loss})
            atomic_json(output / 'history.json', history)
        result = {'completed_epochs': settings['epochs'], 'global_step': step, 'best_epoch': best_epoch,
                  'best_val_loss': best_loss, 'best_checkpoint': str(output / 'best.pt'),
                  'last_checkpoint': str(output / 'last.pt'), 'device': str(device),
                  'synthetic': corpus.manifest['extractor_profile'].get('synthetic', False)}
        atomic_json(output / 'result.json', result)
        return result
    finally:
        corpus.close()
