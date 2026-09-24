"""Bounded, canonical fatigue feature replay; coverage is not classification."""
from collections import Counter, deque
import csv

import numpy as np

from inference.perception.sequence import FeatureExtractor
from training.gaze.preprocessing import dependency_masks
from training.gaze.profile import FEATURE_NAMES, GROUP_INDICES

FEATURES = ('ear_left', 'ear_right', 'mar', 'perclos_4s',
            'closed_duration_s', 'blink_rate_per_min')
INDICES = [FEATURE_NAMES.index(name) for name in FEATURES]
LABELS = ('all', 'known_positive', 'known_negative', 'unknown')


def _groups():
    return {label: Counter() for label in LABELS}


def _rates(groups, denominator):
    result = {}
    for label, counts in groups.items():
        values = dict(counts)
        values.setdefault(denominator, 0)
        n = values[denominator]
        values['rates'] = {k: v/n if n else None for k, v in values.items() if k != denominator}
        result[label] = values
    return result


def aggregate_fatigue(records):
    """Merge integer counts, not unweighted recording percentages."""
    frames, windows = _groups(), _groups()
    reasons = {label: Counter() for label in LABELS}
    for record in records:
        for target, key in ((frames, 'frames'), (windows, 'windows')):
            for label in LABELS:
                target[label].update({k: v for k, v in record[key][label].items() if k != 'rates'})
        for label in LABELS:
            reasons[label].update(record['eye_failure_reasons'][label])
    parity = [r['baseline_fatigue_reproduced'] for r in records]
    return dict(frames=_rates(frames, 'slots'), windows=_rates(windows, 'windows'),
                eye_failure_reasons={k: dict(v) for k, v in reasons.items()},
                raw_mask_mismatches=sum(r['raw_mask_mismatches'] for r in records),
                raw_value_mismatches=sum(r['raw_value_mismatches'] for r in records),
                baseline_fatigue_reproduced=all(parity) if all(v is not None for v in parity) else None,
                longest_missing_eye_pair_ms={side: max(r['longest_missing_eye_pair_ms'][side] for r in records)
                                             for side in ('cached', 'variant')})


def write_comparison(output, report):
    rows = []
    for name, variant in report['variants'].items():
        audit = variant['fatigue_evidence']
        for label in LABELS:
            frames, windows = audit['frames'][label], audit['windows'][label]
            row = dict(variant=name, label_class=label, partial=report['partial'],
                       baseline_reproduced=audit['baseline_fatigue_reproduced'],
                       slots=frames['slots'], windows=windows['windows'])
            for side in ('cached', 'variant'):
                row[f'{side}_eye_pair_slot_ratio'] = frames['rates'].get(f'{side}_eye_pair')
                row[f'{side}_f_eligible_windows'] = windows.get(f'{side}_f_eligible', 0)
                row[f'{side}_f_eligible_window_ratio'] = windows['rates'].get(f'{side}_f_eligible')
                for feature in FEATURES:
                    for level in ('raw', 'effective'):
                        key = f'{side}_{feature}_{level}'
                        row[key+'_slot_ratio'] = frames['rates'].get(key)
            row.update(f_recovered=windows.get('f_recovered', 0), f_lost=windows.get('f_lost', 0))
            rows.append(row)
    with (output/'fatigue_comparison.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


class FatigueReplay:
    def __init__(self, item, settings, directory, stack):
        self.item, self.settings = item, settings
        feature = item['complete']['feature_record']
        self.extractor = FeatureExtractor(feature['extractor_profile'])
        self.profile = settings['group_profile']
        self.supported = np.array([settings['normalizer_supported'][n] for n in FEATURE_NAMES], bool)
        self.steps, self.stride = settings['window_steps'], settings['stride_steps']
        history = self.profile['history_steps']
        self.raw = {s: deque(maxlen=history) for s in ('cached', 'variant')}
        self.ages = {s: deque(maxlen=history) for s in self.raw}
        self.effective = {s: deque(maxlen=self.steps) for s in self.raw}
        self.frames, self.windows = _groups(), _groups()
        self.reasons = {label: Counter() for label in LABELS}
        self.mask_diff = self.value_diff = 0
        self.missing_run, self.longest = Counter(), Counter()
        self.writers = {}
        self.handles = {key: stack.enter_context((directory/f'fatigue_{key}.csv').open(
            'x', encoding='utf-8-sig', newline='')) for key in ('frames', 'windows')}
        fields = ['start_slot', 'end_slot', 'end_ms', 'fatigue_label_known', 'fatigue_target']
        for side in ('cached', 'variant'):
            fields += [f'{side}_f_valid_ratio', f'{side}_f_eligible', f'{side}_eye_pair_ratio']
            fields += [f'{side}_{name}_effective_ratio' for name in FEATURES]
        fields += ['f_recovered', 'f_lost']
        self.writers['windows'] = csv.DictWriter(self.handles['windows'], fieldnames=fields)
        self.writers['windows'].writeheader()

    def _write(self, key, row):
        if key not in self.writers:
            writer = csv.DictWriter(self.handles[key], fieldnames=list(row))
            writer.writeheader()
            self.writers[key] = writer
        self.writers[key].writerow(row)

    def step(self, index, slot, frame, observation, detail, arrays):
        config = self.item['config']
        height, width = frame.bgr.shape[:2]
        missing = dict(status='missing', timestamp_ms=frame.timestamp_ms,
                       model_id='fatigue-replay-unused-expert', items=[])
        packet = dict(schema_version='driveguard-experts-v1', recording_id=config['recording_id'],
                      view=config['view'], frame_id=str(index), timestamp_ms=slot,
                      width=width, height=height, driver_roi=config['driver_roi'],
                      face=observation, objects=dict(missing), hands=dict(missing))
        derived = self.extractor.step(packet)
        variant_mask = derived['masks'].astype(bool)
        cached_mask = arrays['masks'][index].astype(bool)
        current = dict(cached=cached_mask, variant=variant_mask)
        max_age = self.profile['max_age_ms']
        age = min(slot - observation['timestamp_ms'], max_age+1)
        variant_ages = np.array([max_age+1, age, max_age+1], np.float32)
        cached_ages = arrays['expert_age_ms'][index]
        for side, ages in (('cached', cached_ages), ('variant', variant_ages)):
            self.raw[side].append(current[side])
            self.ages[side].append(ages)
            filtered = dependency_masks(np.asarray(self.raw[side]), np.asarray(self.ages[side]), self.profile)[-1]
            self.effective[side].append(filtered & self.supported)
            self.missing_run[side] = 0 if current[side][:2].all() else self.missing_run[side]+1
            self.longest[side] = max(self.longest[side], self.missing_run[side])
        known = bool(arrays['label_masks'][index, 0])
        target = int(arrays['targets'][index, 0]) if known else None
        label = ('known_positive' if target else 'known_negative') if known else 'unknown'
        cm, vm = cached_mask[INDICES], variant_mask[INDICES]
        same_mask = bool(np.array_equal(cm, vm))
        shared = cm & vm
        close = np.isclose(arrays['values'][index, INDICES], derived['values'][INDICES], rtol=1e-5, atol=1e-6)
        same_value = bool(close[shared].all())
        self.mask_diff += int(not same_mask)
        self.value_diff += int(not same_value)
        # Eye validity is independent of iris validity and matrix/head success.
        pair = bool(vm[:2].all())
        if pair:
            reason = 'eye_pair_available'
        elif age > max_age:
            reason = 'stale_face'
        elif observation['landmarks'] is None:
            reason = detail['reason'] if detail['reason'] != 'ok' else 'face_landmarks_missing'
        elif not observation['driver_confirmed']:
            reason = 'driver_unconfirmed'
        else:
            reason = 'eye_geometry_or_roi_invalid'
        row = dict(slot_index=index, slot_ms=slot, source_pts_ms=frame.timestamp_ms,
                   fatigue_label_known=known, fatigue_target=target, eye_reason=reason,
                   face_reason=detail['reason'], raw_masks_match=same_mask, raw_values_match=same_value)
        for side, masks in current.items():
            values = arrays['values'][index] if side == 'cached' else derived['values']
            for name, col in zip(FEATURES, INDICES):
                row[f'{side}_{name}_valid'] = bool(masks[col])
                row[f'{side}_{name}_value'] = float(values[col]) if masks[col] else None
                row[f'{side}_{name}_effective'] = bool(self.effective[side][-1][col])
        self._write('frames', row)
        for group in ('all', label):
            counts = self.frames[group]; counts['slots'] += 1
            self.reasons[group][reason] += 1
            for side, masks in current.items():
                counts[f'{side}_eye_pair'] += int(masks[:2].all())
                for name, col in zip(FEATURES, INDICES):
                    counts[f'{side}_{name}_raw'] += int(masks[col])
                    counts[f'{side}_{name}_effective'] += int(self.effective[side][-1][col])
        if index >= self.steps-1 and (index-self.steps+1) % self.stride == 0:
            self._window(index, label, target)
        return label, reason

    def _window(self, index, label, target):
        row = dict(start_slot=index-self.steps+1, end_slot=index, end_ms=index*40,
                   fatigue_label_known=target is not None, fatigue_target=target)
        flags = {}
        for side in ('cached', 'variant'):
            masks = np.asarray(self.effective[side])
            supported_count = int(self.supported[list(GROUP_INDICES[0])].sum())
            ratio = float(masks[:, GROUP_INDICES[0]].sum()/(self.steps*supported_count)) if supported_count else 0.
            eligible = bool(supported_count and masks[-1, GROUP_INDICES[0]].any()
                            and ratio >= self.profile['min_valid_ratio']
                            and self.ages[side][-1][1] <= self.profile['max_age_ms'])
            eye_ratio = float(masks[:, :2].all(1).mean())
            row.update({f'{side}_f_valid_ratio': ratio, f'{side}_f_eligible': eligible,
                        f'{side}_eye_pair_ratio': eye_ratio})
            flags[f'{side}_f_eligible'] = eligible
            flags[f'{side}_eye_pair_coverage80'] = eye_ratio >= .8
            flags[f'{side}_fatigue_any'] = bool(masks[:, INDICES].any())
            for name, col in zip(FEATURES, INDICES):
                row[f'{side}_{name}_effective_ratio'] = float(masks[:, col].mean())
                flags[f'{side}_{name}_current'] = bool(masks[-1, col])
        flags['f_recovered'] = not flags['cached_f_eligible'] and flags['variant_f_eligible']
        flags['f_lost'] = flags['cached_f_eligible'] and not flags['variant_f_eligible']
        row.update(f_recovered=flags['f_recovered'], f_lost=flags['f_lost'])
        self._write('windows', row)
        for group in ('all', label):
            self.windows[group]['windows'] += 1
            self.windows[group].update({k: int(v) for k, v in flags.items()})

    def report(self, baseline):
        return dict(frames=_rates(self.frames, 'slots'), windows=_rates(self.windows, 'windows'),
                    raw_mask_mismatches=self.mask_diff, raw_value_mismatches=self.value_diff,
                    baseline_fatigue_reproduced=(self.mask_diff == self.value_diff == 0) if baseline else None,
                    longest_missing_eye_pair_ms={s: self.longest[s]*40 for s in ('cached', 'variant')},
                    eye_failure_reasons={k: dict(v) for k, v in self.reasons.items()},
                    window_steps=self.steps, stride_steps=self.stride,
                    interpretation='F eligibility uses saved supported columns and group policy; it is not fatigue prediction. '
                    'Eye coverage80 is a diagnostic count, not a changed model gate. Values use original units. '
                    'Fatigue labels use the final slot; unknowns are never negatives. Value parity compares jointly valid cells.')
