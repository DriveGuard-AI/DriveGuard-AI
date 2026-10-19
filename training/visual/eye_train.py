"""Private eye encoder, unique masked frames and exact epoch continuation."""
from collections import Counter
from copy import deepcopy
from itertools import islice
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from training.common.checkpoint import (atomic_json, capture_rng, cpu_tree, load_checkpoint,
                                       restore_rng, save_checkpoint, seed_everything)
from training.common.config import canonical_hash
from training.common.device import select_device
from training.visual.encoder import ResNet18
from training.visual.head_adapt_train import DEFAULTS, validate_settings


SELECTION = 'subject_balanced_eye_bce'


class EyeDataset:
    """One item per fresh, ROI-valid, explicitly known sampled eye frame.

    Cohort identity intentionally excludes input mode. Metadata is copied while
    reading each record; no mmap views outlive the cache's bounded record lease.
    """
    def __init__(self, head_cache, eye_cache, labels, split, mode='head'):
        if split not in ('train', 'val') or mode not in ('head', 'eye'):
            raise ValueError('Expected train/val split and head/eye mode')
        self.head_cache, self.eye_cache, self.labels = head_cache, eye_cache, labels
        self.cache = head_cache if mode == 'head' else eye_cache
        self.split, self.mode = split, mode
        self.indices, self._metadata = [], []
        for rid, record in sorted(head_cache.records.items()):
            if record['split'] != split:
                continue
            other = eye_cache.records.get(rid)
            if other is None or any(other[k] != record[k] for k in ('subject_id', 'split')):
                raise ValueError('Eye/head record identity mismatch')
            head, eye = head_cache.arrays(rid), eye_cache.arrays(rid)
            if any(not np.array_equal(head[k], eye[k]) for k in ('slot_indices', 'source_pts_ms')):
                raise ValueError('Eye/head clock alignment mismatch')
            target, known = labels.records[rid]['targets'], labels.records[rid]['masks']
            n = len(head['valid'])
            if target.shape != (n, 2) or known.shape != (n, 2):
                raise ValueError('Eye label alignment mismatch')
            mask = head['valid'] & eye['valid'] & known[:, 0]
            for i in np.flatnonzero(mask):
                y = float(target[i, 0])
                if y not in (0., 1.):
                    raise ValueError('Known eye target must be binary')
                self.indices.append((rid, int(i)))
                self._metadata.append(dict(record_id=rid, subject_id=record['subject_id'], split=split,
                    slot_index=int(head['slot_indices'][i]), source_pts_ms=int(head['source_pts_ms'][i]), target=y))
        if len({(r['record_id'], r['slot_index']) for r in self._metadata}) != len(self.indices):
            raise ValueError('Duplicate sampled eye frames')
        self.summary = _summary(self._metadata)

    def __len__(self):
        return len(self.indices)

    def metadata(self, index):
        return dict(self._metadata[index])

    def __getitem__(self, index):
        rid, position = self.indices[index]
        pixels = self.cache.arrays(rid)['images'][position].copy()
        return torch.from_numpy(pixels), np.float32(self._metadata[index]['target'])


def _summary(rows):
    by_subject = {}
    for row in rows:
        counts = by_subject.setdefault(row['subject_id'], dict(count=0, positive=0, negative=0))
        counts['count'] += 1
        counts['positive' if row['target'] else 'negative'] += 1
    positive = sum(r['target'] == 1 for r in rows)
    return dict(count=len(rows), positive=positive, negative=len(rows)-positive, by_subject=by_subject,
                unit='unique sampled frames', frame_cohort_sha256=canonical_hash(rows))


def subject_weights(dataset):
    """Training-only inverse subject frequency; equal expected subject draws."""
    if dataset.split != 'train':
        raise ValueError('Subject sampling requires training split')
    subjects = [dataset.metadata(i)['subject_id'] for i in range(len(dataset))]
    counts = Counter(subjects)
    return [1. / counts[subject] for subject in subjects]


class EyeModel(nn.Module):
    def __init__(self, reference_model=None, frame_chunk=16, normalizer=None):
        super().__init__()
        if type(frame_chunk) is not int or frame_chunk <= 0:
            raise ValueError('Invalid eye frame_chunk')
        self.frame_chunk = frame_chunk
        if reference_model is not None:
            base = reference_model.backbone
            self.encoder = deepcopy(base.encoder)
            self.normalization = deepcopy(base.normalization)
            for key in ('rgb_mean', 'rgb_std', 'head_mean_hi', 'head_mean_lo', 'head_std'):
                self.register_buffer(key, getattr(base, key).detach().clone())
            self.eye = nn.Linear(512, 1).to(reference_model.actions.weight.device)
            with torch.no_grad():
                self.eye.weight.copy_(reference_model.actions.weight[:1])
                self.eye.bias.copy_(reference_model.actions.bias[:1])
        else:
            self.encoder = ResNet18()
            self.normalization = deepcopy(normalizer)
            mean = torch.tensor(normalizer['mean'], dtype=torch.float64)
            self.register_buffer('head_mean_hi', mean.float())
            self.register_buffer('head_mean_lo', (mean-mean.float().double()).float())
            self.register_buffer('head_std', torch.tensor(normalizer['std'], dtype=torch.float32))
            self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406]).reshape(1,3,1,1))
            self.register_buffer('rgb_std', torch.tensor([.229, .224, .225]).reshape(1,3,1,1))
            self.eye = nn.Linear(512, 1)
        self.encoder.requires_grad_(False)
        for module in self.encoder.layer4.modules():
            if isinstance(module, nn.Conv2d):
                module.requires_grad_(True)
        self.model_params = dict(frame_chunk=frame_chunk, features=512, tune_encoder='layer4_conv')
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
                module.requires_grad_(False)
        return self

    def forward(self, images):
        if images.dtype != torch.uint8 or images.ndim != 4 or tuple(images.shape[1:]) != (3,224,224):
            raise ValueError('Expected uint8 eye images [N,3,224,224]')
        if not len(images):
            return self.eye.weight.new_empty(0)
        features = []
        for chunk in images.split(self.frame_chunk):
            pixels = (chunk.float()/255.-self.rgb_mean)/self.rgb_std
            feature = self.encoder(pixels)
            features.append((((feature-self.head_mean_hi)-self.head_mean_lo)/self.head_std).clamp(-5,5))
        # Keep the reference GEMM output width for bit-identical initial row 0.
        # The padding row is zero, has no parameters and is never an output.
        weight = torch.cat((self.eye.weight, torch.zeros_like(self.eye.weight)))
        bias = torch.cat((self.eye.bias, torch.zeros_like(self.eye.bias)))
        return F.linear(torch.cat(features), weight, bias)[:, 0]


def run_epoch(model, loader, device, pos_weight, optimizer=None, accumulation_steps=1, clip=1.):
    training = optimizer is not None
    model.train(training)
    iterator = iter(loader)
    loss_sum = weight_sum = 0.
    count = updates = 0
    while True:
        batches = list(islice(iterator, accumulation_steps if training else 1))
        if not batches:
            break
        denominator = sum(float(torch.where(y == 1, pos_weight, 1.).sum()) for _, y in batches)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            for images, target in batches:
                target = target.to(device).float()
                prediction = model(images.to(device))
                if not torch.isfinite(prediction).all() or not ((target == 0) | (target == 1)).all():
                    raise ValueError('Invalid supervised eye values')
                weights = torch.where(target == 1, pos_weight, 1.)
                loss = (F.binary_cross_entropy_with_logits(prediction, target, reduction='none')*weights).sum()
                loss_sum += float(loss.detach()); weight_sum += float(weights.sum()); count += len(target)
                if training:
                    (loss/denominator).backward()
        if training:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            optimizer.step(); updates += 1
    return dict(loss=loss_sum/weight_sum if weight_sum else None, loss_sum=loss_sum,
                weight_sum=weight_sum, count=count, updates=updates)


def evaluate_eye(model, loader, device):
    """Unweighted class means per subject, then equal eligible-subject mean.

    Single-class subjects are reported and excluded, never imputed a class.
    The loader must preserve dataset order (the trainer uses shuffle=False).
    """
    model.eval()
    subjects = {}; position = 0
    with torch.no_grad():
        for images, targets in loader:
            predictions = model(images.to(device))
            if not torch.isfinite(predictions).all():
                raise ValueError('Nonfinite validation eye logits')
            losses = F.binary_cross_entropy_with_logits(predictions, targets.to(device).float(), reduction='none').cpu()
            for target, loss in zip(targets.tolist(), losses.tolist()):
                row = loader.dataset.metadata(position); position += 1
                if target not in (0., 1.) or target != row['target']:
                    raise ValueError('Validation eye supervision/order mismatch')
                subject = subjects.setdefault(row['subject_id'], dict(positive_sum=0., negative_sum=0., positive=0, negative=0))
                name = 'positive' if target else 'negative'
                subject[name] += 1; subject[name+'_sum'] += loss
    scores = []; excluded = []
    for subject_id, subject in subjects.items():
        if subject['positive'] and subject['negative']:
            subject['loss'] = (subject['positive_sum']/subject['positive'] + subject['negative_sum']/subject['negative'])/2
            scores.append(subject['loss'])
        else:
            subject['loss'] = None; excluded.append(subject_id)
    if not scores:
        raise ValueError('No validation subjects have both positive and negative eye supervision')
    return dict(loss=sum(scores)/len(scores), count=position, by_subject=subjects,
                excluded_single_class_subjects=excluded, single_class_policy='exclude-and-report',
                selection_metric=SELECTION)


def _verify_data(train, val, provenance):
    if train.split != 'train' or val.split != 'val' or train.mode != val.mode:
        raise ValueError('Eye training/validation split or input mode mismatch')
    if set(train.summary['by_subject']) & set(val.summary['by_subject']):
        raise ValueError('Eye training/validation subjects overlap')
    for name in ('head_cache', 'eye_cache'):
        a, b = getattr(train, name), getattr(val, name)
        if a.root != b.root or a.manifest_sha256 != b.manifest_sha256:
            raise ValueError('Eye dataset cache identity mismatch')
        if name in provenance and (Path(provenance[name]).resolve() != a.root.resolve()
                or provenance.get(name+'_sha256') != a.manifest_sha256):
            raise ValueError('Eye cache provenance mismatch')
    if train.labels.digest != val.labels.digest:
        raise ValueError('Eye label identity mismatch')
    if 'action_labels_sha256' in provenance and provenance['action_labels_sha256'] != train.labels.digest:
        raise ValueError('Eye labels provenance mismatch')
    cohort = {s: d.summary['frame_cohort_sha256'] for s, d in (('train', train), ('val', val))}
    if 'matched_cohort' in provenance and provenance['matched_cohort'] != cohort:
        raise ValueError('Eye cohort provenance mismatch')


def load_eye(path, backend='cpu', device_index=0):
    payload = load_checkpoint(path)
    try:
        if (payload['model_kind'] != 'eye_v1' or type(payload['subject_balanced']) is not bool
                or payload['input_mode'] not in ('head', 'eye')
                or canonical_hash(payload['provenance']) != payload['provenance_sha256']
                or payload['selection_metric'] != SELECTION
                or payload['config'] != dict(model=payload['model_params'], training=payload['settings'],
                    subject_balanced=payload['subject_balanced'], input_mode=payload['input_mode'])):
            raise ValueError('Invalid eye checkpoint identity')
        settings = validate_settings(payload['settings'])
        if type(payload['epoch']) is not int or not 0 <= payload['epoch'] <= settings['epochs']:
            raise ValueError('Invalid eye checkpoint epoch')
        if payload['model_params'] != dict(frame_chunk=settings['frame_chunk'], features=512, tune_encoder='layer4_conv'):
            raise ValueError('Invalid eye model geometry')
        summary = payload['data_summary']
        for split in ('train', 'val'):
            counts = summary[split]
            if (any(type(counts[k]) is not int or counts[k] < 0 for k in ('count', 'positive', 'negative'))
                    or counts['count'] != counts['positive']+counts['negative']
                    or counts['unit'] != 'unique sampled frames'
                    or not isinstance(counts['frame_cohort_sha256'], str) or len(counts['frame_cohort_sha256']) != 64):
                raise ValueError('Invalid eye data summary metadata')
            for key in ('count', 'positive', 'negative'):
                if sum(r[key] for r in counts['by_subject'].values()) != counts[key]:
                    raise ValueError('Invalid eye subject summary metadata')
        if (not summary['train']['positive'] or not summary['train']['negative']
                or set(summary['train']['by_subject']) & set(summary['val']['by_subject'])
                or summary['class_weighting'] != dict(positive=summary['train']['negative']/summary['train']['positive'],
                    negative=1., source='train', normalization='sum-of-observed-weights')):
            raise ValueError('Invalid eye class weighting or subject metadata')
        if payload['sampling'] != dict(policy='inverse-subject-frequency' if payload['subject_balanced'] else 'uniform-permutation',
                epoch_draws=summary['train']['count'], replacement=payload['subject_balanced']):
            raise ValueError('Invalid eye sampling metadata')
        if ('matched_cohort' in payload['provenance'] and payload['provenance']['matched_cohort'] !=
                {s: summary[s]['frame_cohort_sha256'] for s in ('train', 'val')}):
            raise ValueError('Eye cohort provenance metadata mismatch')
        mean = torch.tensor(payload['normalizer']['mean'], dtype=torch.float64)
        std = torch.tensor(payload['normalizer']['std'], dtype=torch.float32)
        if mean.shape != (512,) or std.shape != (512,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError('Invalid eye normalization statistics')
        expected = dict(head_mean_hi=mean.float(), head_mean_lo=(mean-mean.float().double()).float(), head_std=std,
            rgb_mean=torch.tensor([.485,.456,.406]).reshape(1,3,1,1), rgb_std=torch.tensor([.229,.224,.225]).reshape(1,3,1,1))
        states = [payload['model_state']]
        if payload['kind'] == 'training':
            if not {'optimizer_state', 'rng_state', 'initial_report', 'updates', 'initial_state', 'best_state',
                    'best_epoch', 'best_loss', 'stopping_loss', 'bad_epochs', 'history'} <= payload.keys():
                raise ValueError('Incomplete eye training checkpoint metadata')
            states.extend([payload['initial_state'], payload['best_state']])
            if (len(payload['history']) != payload['epoch']
                    or [r['epoch'] for r in payload['history']] != list(range(1,payload['epoch']+1))
                    or type(payload['best_epoch']) is not int or not 0 <= payload['best_epoch'] <= payload['epoch']
                    or type(payload['bad_epochs']) is not int or not 0 <= payload['bad_epochs'] <= payload['epoch']
                    or not math.isfinite(payload['best_loss']) or not math.isfinite(payload['stopping_loss'])
                    or payload['stopping_loss'] < payload['best_loss']):
                raise ValueError('Invalid eye checkpoint epoch/history')
        elif (payload['checkpoint_role'] not in ('initial', 'best')
                or payload['checkpoint_selection_metric'] != SELECTION or not math.isfinite(payload['selected_loss'])
                or (payload['checkpoint_role'] == 'initial' and payload['epoch'] != 0)):
            raise ValueError('Invalid eye checkpoint selection metadata')
        for state in states:
            if any(k not in state or not torch.equal(state[k], v) for k, v in expected.items()):
                raise ValueError('Eye normalization mismatch')
        model = EyeModel(frame_chunk=settings['frame_chunk'], normalizer=payload['normalizer'])
        for state in states:
            model.load_state_dict(state, strict=True)
            if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
                raise ValueError('Nonfinite eye checkpoint state')
        model.load_state_dict(payload['model_state'], strict=True)
    except (KeyError, TypeError, RuntimeError) as exc:
        raise ValueError('Incomplete or invalid eye checkpoint identity/state') from exc
    return model.to(select_device(backend, device_index)).eval(), payload


def train_eye(train, val, reference_model, settings, output_dir, provenance, subject_balanced=False, resume=None):
    settings = validate_settings(settings)
    if type(subject_balanced) is not bool:
        raise ValueError('Expected subject_balanced boolean')
    _verify_data(train, val, provenance)
    coverage = dict(train=deepcopy(train.summary), val=deepcopy(val.summary))
    if not coverage['train']['positive'] or not coverage['train']['negative']:
        raise ValueError('Training requires positive and negative eye supervision')
    pos_weight = coverage['train']['negative']/coverage['train']['positive']
    coverage['class_weighting'] = dict(positive=pos_weight, negative=1., source='train', normalization='sum-of-observed-weights')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a NEW eye training directory')
    device = select_device(settings['backend'], settings['device_index'])
    seed_everything(settings['seed'], device)
    model = EyeModel(reference_model, settings['frame_chunk']).to(device)
    groups = [dict(params=list(model.eye.parameters()), lr=settings['learning_rate'], weight_decay=settings['weight_decay']),
              dict(params=[p for p in model.encoder.parameters() if p.requires_grad],
                   lr=settings['encoder_learning_rate'], weight_decay=settings['encoder_weight_decay'])]
    optimizer = torch.optim.Adam(groups, foreach=False)
    generator = torch.Generator().manual_seed(settings['seed'])
    val_generator = torch.Generator().manual_seed(settings['seed'])
    sampler = WeightedRandomSampler(subject_weights(train), len(train), replacement=True, generator=generator) if subject_balanced else None
    train_loader = DataLoader(train, batch_size=settings['batch_size'], shuffle=not subject_balanced,
                              sampler=sampler, generator=generator, num_workers=0)
    val_loader = DataLoader(val, batch_size=settings['batch_size'], shuffle=False, generator=val_generator, num_workers=0)
    metadata = dict(format_version=1, model_kind='eye_v1', model_params=model.model_params, settings=settings,
        input_mode=train.mode, subject_balanced=subject_balanced, normalizer=deepcopy(model.normalization),
        provenance=deepcopy(provenance), provenance_sha256=canonical_hash(provenance), data_summary=coverage,
        selection_metric=SELECTION, sampling=dict(policy='inverse-subject-frequency' if subject_balanced else 'uniform-permutation',
            epoch_draws=len(train), replacement=subject_balanced))
    metadata['config'] = dict(model=metadata['model_params'], training=settings, subject_balanced=subject_balanced, input_mode=train.mode)
    start = updates = best_epoch = bad = 0
    history = []
    if resume is None:
        initial_report = evaluate_eye(model, val_loader, device)
        initial_state = cpu_tree(model.state_dict())
        best_state, best_loss = initial_state, initial_report['loss']
        stopping_loss = best_loss
    else:
        _, previous = load_eye(resume)
        old, current = deepcopy(previous['settings']), deepcopy(settings)
        old_epochs, new_epochs = old.pop('epochs'), current.pop('epochs')
        if (previous['kind'] != 'training' or old != current or new_epochs < old_epochs
                or any(previous[k] != metadata[k] for k in ('provenance', 'subject_balanced', 'input_mode', 'model_params', 'normalizer', 'data_summary', 'sampling'))):
            raise ValueError('Eye resume configuration/provenance mismatch')
        if previous['bad_epochs'] >= settings['early_stopping_patience'] and old_epochs != new_epochs:
            raise ValueError('Cannot extend early-stopped eye run')
        current_initial = model.state_dict()
        if any(not torch.equal(v.cpu(), previous['initial_state'][k]) for k, v in current_initial.items()):
            raise ValueError('Eye resume reference initialization mismatch')
        model.load_state_dict(previous['model_state'], strict=True)
        optimizer.load_state_dict(previous['optimizer_state'])
        initial_report, initial_state = previous['initial_report'], previous['initial_state']
        best_state, best_loss, best_epoch = previous['best_state'], previous['best_loss'], previous['best_epoch']
        stopping_loss = previous['stopping_loss']
        start, updates, bad, history = previous['epoch'], previous['updates'], previous['bad_epochs'], previous['history']
        restore_rng(previous['rng_state'], device, generator)
        val_generator.set_state(previous['rng_state']['val_loader'])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'config.json', metadata['config'])
    atomic_json(output/'provenance.json', provenance)
    def weights(name, state, epoch, loss):
        save_checkpoint(output/(name+'.pt'), dict(metadata, kind='weights', model_state=state, epoch=epoch,
            selected_loss=loss, checkpoint_role=name, checkpoint_selection_metric=SELECTION))
    weights('initial', initial_state, 0, initial_report['loss'])
    def persist(epoch):
        rng = capture_rng(device, generator); rng['val_loader'] = val_generator.get_state()
        save_checkpoint(output/'last.pt', dict(metadata, kind='training', model_state=model.state_dict(), epoch=epoch,
            optimizer_state=optimizer.state_dict(), rng_state=rng, initial_state=initial_state, initial_report=initial_report,
            best_state=best_state, best_loss=best_loss, best_epoch=best_epoch, stopping_loss=stopping_loss,
            updates=updates, bad_epochs=bad, history=history))
        weights('best', best_state, best_epoch, best_loss)
        atomic_json(output/'history.json', history)
    epoch = start
    persist(epoch)
    if bad < settings['early_stopping_patience']:
        for epoch in range(start+1, settings['epochs']+1):
            tr = run_epoch(model, train_loader, device, pos_weight, optimizer,
                           settings['accumulation_steps'], settings['gradient_clip_norm'])
            va = evaluate_eye(model, val_loader, device)
            updates += tr['updates']; value = va['loss']
            history.append(dict(epoch=epoch, train=tr, val=va, selection_score=value))
            improved = value < stopping_loss-settings['early_stopping_min_delta']
            if value < best_loss:
                best_state, best_loss, best_epoch = cpu_tree(model.state_dict()), value, epoch
            if improved:
                stopping_loss = value
            bad = 0 if improved else bad+1
            persist(epoch)
            print(f'eye mode={train.mode} balanced={subject_balanced} epoch={epoch}/{settings["epochs"]} subject_bce={value:.6f}', flush=True)
            if bad >= settings['early_stopping_patience']:
                break
    result = dict(completed_epochs=epoch, best_epoch=best_epoch, best_loss=best_loss, history=history,
                  selection_metric=SELECTION, stopped_early=bad >= settings['early_stopping_patience'], device=str(device))
    for name in ('initial', 'best', 'last'):
        result[name+'_checkpoint'] = str(output/(name+'.pt'))
    atomic_json(output/'result.json', result)
    return result
