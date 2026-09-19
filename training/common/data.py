"""Validated, bounded-mmap feature recordings; no raw video or DMD assumptions."""
from bisect import bisect_right
from collections import OrderedDict
from copy import deepcopy
import hashlib
import os
from pathlib import Path, PurePosixPath

import numpy as np

from training.common.config import TASKS, canonical_hash, read_json, validate_config
from training.common.normalization import fit_normalizer, normalize

ARRAY_NAMES = ('values', 'masks', 'targets', 'label_masks')
CHUNK_SIZE = 65536


def resolve_data_root(explicit=None):
    value = explicit or os.environ.get('DRIVEGUARD_DATA_ROOT')
    if not value:
        raise ValueError('Set --data-root or DRIVEGUARD_DATA_ROOT')
    return Path(value).expanduser().resolve()


def safe_data_path(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative:
        raise ValueError('Data paths must use relative POSIX spelling')
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or path.suffix != '.npy':
        raise ValueError('Data path must be a contained relative .npy file')
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Data path escapes data root (including symlinks)')
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _close_arrays(arrays):
    for array in arrays.values():
        mmap = getattr(array, '_mmap', None)
        if mmap is not None:
            mmap.close()


class FeatureCorpus:
    """Immutable data during a run; at most four recordings have open mmap handles."""

    def __init__(self, config, manifest_path, data_root=None):
        self.config = deepcopy(validate_config(config))
        self.relation = self.config['schema_version'] == 'driveguard-training-v2'
        self.array_names = ARRAY_NAMES + (('expert_age_ms',) if self.relation else ())
        self.root = resolve_data_root(data_root)
        self.manifest = read_json(manifest_path)
        self._cache = OrderedDict()
        self.records = []
        try:
            self._validate_manifest()
        except Exception:
            self.close()
            raise

    def _validate_manifest(self):
        manifest, data = self.manifest, self.config['data']
        keys = {'schema_version', 'dataset_version', 'feature_schema', 'extractor_profile', 'label_mapping', 'records'}
        if self.relation:
            keys |= {'group_profile', 'group_profile_sha256'}
        if not isinstance(manifest, dict) or set(manifest) != keys:
            raise ValueError('Invalid manifest keys')
        if manifest['schema_version'] != ('driveguard-feature-manifest-v2' if self.relation else 'driveguard-feature-manifest-v1'):
            raise ValueError('Unsupported manifest schema')
        if self.relation:
            group_profile = data['group_profile']
            if manifest['group_profile'] != group_profile or manifest['group_profile_sha256'] != canonical_hash(group_profile):
                raise ValueError('group_profile/hash mismatch')
        if not isinstance(manifest['dataset_version'], str) or not manifest['dataset_version'].strip():
            raise ValueError('dataset_version must be nonempty')
        schema = {'version': data['feature_schema_version'], 'names': data['feature_names'],
                  'sample_rate_hz': data['sample_rate_hz']}
        if manifest['feature_schema'] != schema:
            raise ValueError('Feature schema mismatch')
        profile, labels = manifest['extractor_profile'], manifest['label_mapping']
        if not isinstance(profile, dict) or not isinstance(profile.get('name'), str):
            raise ValueError('extractor_profile requires a name')
        definitions = profile.get('feature_definitions')
        if not isinstance(definitions, dict) or set(definitions) != set(data['feature_names']) or not all(definitions.values()):
            raise ValueError('extractor_profile must define every feature')
        if not isinstance(labels, dict) or labels.get('version') != data['label_schema_version'] or labels.get('tasks') != list(TASKS):
            raise ValueError('Label mapping/schema mismatch')
        if not isinstance(labels.get('rules'), dict) or not labels['rules']:
            raise ValueError('Label mapping must record positive/negative/unknown rules')
        records = manifest['records']
        if not isinstance(records, list) or not records:
            raise ValueError('Manifest records must be nonempty')
        record_keys = {'record_id', 'subject_id', 'recording_id', 'view', 'subset', 'split', *self.array_names}
        ids, subjects, recordings, views, value_paths = set(), {}, {}, set(), set()
        hashes = {}
        for entry in records:
            if not isinstance(entry, dict) or set(entry) != record_keys:
                raise ValueError('Invalid record fields')
            if any(not isinstance(v, str) or not v.strip() for v in entry.values()):
                raise ValueError('Record fields must be nonempty strings')
            rid, split = entry['record_id'], entry['split']
            if rid in ids or split not in ('train', 'val', 'test'):
                raise ValueError('Duplicate record or invalid split')
            ids.add(rid)
            subject, recording = entry['subject_id'], entry['recording_id']
            if subjects.setdefault(subject, split) != split:
                raise ValueError('subject leaks across splits')
            if recordings.setdefault(recording, (subject, split)) != (subject, split):
                raise ValueError('recording has inconsistent subject/split')
            view_key = (recording, entry['view'])
            if view_key in views:
                raise ValueError('Duplicate recording/view: use one full recording, not copied windows')
            views.add(view_key)
            record = dict(entry)
            record['paths'] = {name: safe_data_path(self.root, entry[name]) for name in self.array_names}
            if record['paths']['values'] in value_paths:
                raise ValueError('Duplicate feature file referenced by recordings')
            value_paths.add(record['paths']['values'])
            self.records.append(record)
            arrays = self.arrays(record)
            self._validate_arrays(record, arrays)
            hashes[rid] = {name: file_sha256(path) for name, path in record['paths'].items()}
        self.summary = {
            'manifest_sha256': canonical_hash(manifest), 'array_sha256': hashes,
            'dataset_version': manifest['dataset_version'],
            'subjects': {split: sorted(s for s, v in subjects.items() if v == split)
                         for split in ('train', 'val', 'test')},
            'extractor_profile_sha256': canonical_hash(profile),
            'label_mapping_sha256': canonical_hash(labels),
        }
        if self.relation:
            self.summary['group_profile_sha256'] = manifest['group_profile_sha256']
        self.summary['fingerprint'] = canonical_hash(self.summary)

    def arrays(self, record):
        rid = record['record_id']
        if rid in self._cache:
            self._cache.move_to_end(rid)
            return self._cache[rid]
        arrays = {}
        try:
            for name, path in record['paths'].items():
                arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
        except Exception:
            _close_arrays(arrays)
            raise
        self._cache[rid] = arrays
        while len(self._cache) > 4:
            _, expired = self._cache.popitem(last=False)
            _close_arrays(expired)
        return arrays

    def _validate_arrays(self, record, arrays):
        n_features = len(self.config['data']['feature_names'])
        values = arrays['values']
        if values.ndim != 2 or values.shape[1] != n_features or values.dtype != np.float32:
            raise ValueError('values must be float32 [slots,F]')
        frames = values.shape[0]
        if frames < self.config['data']['window_steps']:
            raise ValueError('recording shorter than window_steps')
        for name in ARRAY_NAMES:
            expected = (frames, n_features if name in ('values', 'masks') else 4)
            if arrays[name].shape != expected:
                raise ValueError(f'{name} shape mismatch')
        if arrays['targets'].dtype != np.float32:
            raise ValueError('targets must be float32')
        if self.relation:
            ages = arrays['expert_age_ms']
            if ages.dtype != np.float32 or ages.shape != (frames,3):
                raise ValueError('expert_age_ms must be float32 [slots,3]')
        for name in ('masks', 'label_masks'):
            if arrays[name].dtype not in (np.dtype('bool'), np.dtype('uint8'), np.dtype('float32')):
                raise ValueError(f'{name} must be bool/uint8/float32 binary')
        for start in range(0, frames, CHUNK_SIZE):
            chunks = {name: value[start:start + CHUNK_SIZE] for name, value in arrays.items()}
            if any(not np.isfinite(chunk).all() for chunk in chunks.values()):
                raise ValueError('Nonfinite array value')
            if self.relation and ((chunks['expert_age_ms'] < 0).any() or
                                 (chunks['expert_age_ms'] > self.config['data']['group_profile']['max_age_ms']+1).any()):
                raise ValueError('expert_age_ms outside clipped nonnegative range')
            for name in ('masks', 'targets', 'label_masks'):
                if not np.isin(chunks[name], [0, 1]).all():
                    raise ValueError(f'{name} must be binary')
            if np.any(chunks['targets'][chunks['label_masks'] == 0] != 0):
                raise ValueError('Unknown labels must have zero placeholder targets')
        record['frames'] = frames

    def effective_masks(self, arrays, start, stop):
        if not self.relation:
            return arrays['masks'][start:stop]
        from training.relation.preprocessing import dependency_masks
        profile = self.config['data']['group_profile']
        first = max(0,start-profile['history_steps']+1)
        filtered = dependency_masks(arrays['masks'][first:stop], arrays['expert_age_ms'][first:stop], profile)
        return filtered[start-first:]

    def fit_normalizer(self):
        def chunks():
            for record in self.records:
                if record['split'] != 'train':
                    continue
                arrays = self.arrays(record)
                for start in range(0, record['frames'], CHUNK_SIZE):
                    yield arrays['values'][start:start+CHUNK_SIZE], self.effective_masks(arrays,start,start+CHUNK_SIZE)
        return fit_normalizer(chunks(), self.config['data']['feature_names'])

    def dataset(self, split, normalizer):
        if normalizer['feature_names'] != self.config['data']['feature_names']:
            raise ValueError('Normalizer feature order mismatch')
        return WindowDataset(self, split, normalizer)

    def check_supervision(self, active_tasks, normalizer=None):
        if self.relation:
            return self._check_relation_supervision(active_tasks,normalizer or self.fit_normalizer())
        result = {}
        for split in ('train', 'val'):
            positive = np.zeros(4, dtype=np.int64)
            negative = np.zeros(4, dtype=np.int64)
            unknown = np.zeros(4, dtype=np.int64)
            for record in self.records:
                if record['split'] != split:
                    continue
                arrays = self.arrays(record)
                last = self.config['data']['window_steps'] - 1
                stride = self.config['data']['stride_steps']
                targets = arrays['targets'][last::stride]
                known = arrays['label_masks'][last::stride].astype(bool)
                positive += ((targets == 1) & known).sum(axis=0)
                negative += ((targets == 0) & known).sum(axis=0)
                unknown += (~known).sum(axis=0)
            for task in active_tasks:
                c = TASKS.index(task)
                if positive[c] == 0 or negative[c] == 0:
                    raise ValueError(f'{split} {task} requires known positive and negative windows')
            result[split] = {'positive': positive.tolist(), 'negative': negative.tolist(), 'unknown': unknown.tolist()}
        return result

    def _check_relation_supervision(self, active_tasks, normalizer):
        result = {}
        for split in ('train','val'):
            positive = np.zeros(4,dtype=np.int64)
            negative = np.zeros(4,dtype=np.int64)
            unknown = np.zeros(4,dtype=np.int64)
            raw_positive = np.zeros(4,dtype=np.int64)
            raw_negative = np.zeros(4,dtype=np.int64)
            unavailable = 0
            dataset = self.dataset(split,normalizer)
            for i in range(len(dataset)):
                inputs,target,labels = dataset[i]
                available = bool(inputs['eligible'].any())
                unavailable += int(not available)
                raw_positive += (target == 1) & labels
                raw_negative += (target == 0) & labels
                positive += (target == 1) & labels & available
                negative += (target == 0) & labels & available
                unknown += ~labels
            for task in active_tasks:
                c = TASKS.index(task)
                if positive[c] == 0 or negative[c] == 0:
                    raise ValueError(f'{split} {task} requires usable known positive and negative windows')
            result[split] = {'positive':positive.tolist(),'negative':negative.tolist(),'unknown':unknown.tolist(),
                             'raw_positive':raw_positive.tolist(),'raw_negative':raw_negative.tolist(),
                             'unavailable_windows':unavailable}
        return result

    def close(self):
        for arrays in self._cache.values():
            _close_arrays(arrays)
        self._cache.clear()


class WindowDataset:
    def __init__(self, corpus, split, normalizer):
        self.corpus, self.normalizer = corpus, normalizer
        self.records = [r for r in corpus.records if r['split'] == split]
        data = corpus.config['data']
        self.steps, self.stride = data['window_steps'], data['stride_steps']
        self.ends = []
        total = 0
        for record in self.records:
            total += (record['frames'] - self.steps) // self.stride + 1
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        ri = bisect_right(self.ends, index)
        offset = index - (self.ends[ri - 1] if ri else 0)
        return self.records[ri], offset * self.stride

    def metadata(self, index):
        record, _ = self.locate(index)
        return {key: record[key] for key in ('record_id', 'subject_id', 'subset', 'view')}

    def __getitem__(self, index):
        record, start = self.locate(index)
        arrays = self.corpus.arrays(record)
        stop = start + self.steps
        masks = self.corpus.effective_masks(arrays,start,stop)
        if self.corpus.relation:
            from training.relation.preprocessing import prepare_window
            x = prepare_window(arrays['values'][start:stop],masks,arrays['expert_age_ms'][start:stop],
                               self.normalizer,self.corpus.config['data']['group_profile'])
        else:
            x = normalize(arrays['values'][start:stop], masks, self.normalizer)
        return x, arrays['targets'][stop-1].copy(), arrays['label_masks'][stop-1].astype(bool, copy=True)
