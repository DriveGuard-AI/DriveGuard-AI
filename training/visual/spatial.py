"""Shared geometric crop and matched spatial RGB input ablations."""
import math


DEFAULT_CROP = (.05, 0., .65, .65)


def crop_policy(xyxy=DEFAULT_CROP):
    if (not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in xyxy)
            or not (0 <= xyxy[0] < xyxy[2] <= 1 and 0 <= xyxy[1] < xyxy[3] <= 1)):
        raise ValueError('Crop must be normalized driver-relative xyxy within [0,1]')
    return dict(version='driver-relative-rect-v1', xyxy=list(xyxy))


def validate_crop(policy):
    if not isinstance(policy, dict) or set(policy) != {'version', 'xyxy'} or policy != crop_policy(policy['xyxy']):
        raise ValueError('Invalid spatial crop policy')
    return policy


def crop_roi(driver_roi, policy):
    validate_crop(policy)
    if (not isinstance(driver_roi, (list, tuple)) or len(driver_roi) != 4
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in driver_roi)
            or not (0 <= driver_roi[0] < driver_roi[2] and 0 <= driver_roi[1] < driver_roi[3])):
        raise ValueError('Invalid original driver ROI')
    x, y, right, bottom = driver_roi
    a, b, c, d = policy['xyxy']
    return [x+a*(right-x), y+b*(bottom-y), x+c*(right-x), y+d*(bottom-y)]


class SpatialData:
    """Verify head/whole RGB have identical source, windows, PTS and validity."""
    def __init__(self, base, head_cache):
        import numpy as np
        from training.visual.data import VisualCache, METADATA
        from training.visual.prepare import fit_statistics
        if base.task != 'fatigue' or 'crop' in base.cache.report:
            raise ValueError('Expected original whole-frame fatigue preparation')
        self.base = base
        self.head = VisualCache(head_cache)
        try:
            a,b = base.cache,self.head
            if (b.report['schema_version'] != 'driveguard-visual-cache-v2'
                    or a.report['manifest_sha256'] != b.report['manifest_sha256']
                    or a.report['encoder'] != b.report['encoder'] or set(a.records) != set(b.records)):
                raise ValueError('Spatial cache source/encoder/record mismatch')
            for rid, entry in a.records.items():
                other = b.records[rid]
                if any(entry[k] != other[k] for k in (*METADATA,'source','slots','sample_count')):
                    raise ValueError('Spatial cache source identity mismatch')
                original, head = a.arrays(rid), b.arrays(rid)
                if any(not np.array_equal(original[k],head[k]) for k in ('slot_indices','source_pts_ms','valid')):
                    raise ValueError('Spatial cache PTS/sampling differs from whole-frame cache')
            def chunks():
                for rid,entry in b.records.items():
                    if entry['split'] != 'train':continue
                    arrays = b.arrays(rid)
                    for start in range(0,entry['sample_count'],2048):
                        yield arrays['embeddings'][start:start+2048][arrays['valid'][start:start+2048]]
            self.head_normalizer = fit_statistics(chunks(),512)
        except Exception:
            self.close()
            raise

    def dataset(self, split, arm):
        return SpatialDataset(self,split,arm)

    def close(self):
        self.head.close()


class SpatialDataset:
    """Matched 1024-D inputs: zero an unused half instead of shrinking the model."""
    visual_dim = 1024

    def __init__(self, spatial, split, arm):
        from copy import deepcopy
        from types import SimpleNamespace
        if arm not in ('whole','head','both'):raise ValueError('Unknown spatial arm')
        self.spatial,self.arm = spatial,arm
        self.base = spatial.base.dataset(split)
        self.context_dim = self.base.context_dim
        provenance = deepcopy(spatial.base.provenance)
        provenance.update(spatial=dict(schema_version='driveguard-spatial-input-v1',arm=arm,
                          head_cache=str(spatial.head.root),head_cache_sha256=spatial.head.manifest_sha256,
                          crop=deepcopy(spatial.head.report['crop']),mask_policy='identical_source_pts_and_validity',
                          input_layout=['whole_512','head_512']))
        provenance['normalization']['head'] = deepcopy(spatial.head_normalizer)
        self.prepared = SimpleNamespace(provenance=provenance)

    def __len__(self):return len(self.base)

    def metadata(self,index):return self.base.metadata(index)

    def __getitem__(self,index):
        import numpy as np
        from training.visual.data import OFFSETS,scale
        inputs,target,known = self.base[index]
        row = self.metadata(index)
        arrays = self.spatial.head.arrays(row['record_id'])
        start = row['visual_start']
        if not np.array_equal(arrays['slot_indices'][start:start+8],row['start_slot']+OFFSETS):
            raise ValueError('Spatial window sampling misaligned')
        head = scale(arrays['embeddings'][start:start+8],self.spatial.head_normalizer)
        head[~inputs['visual_mask']] = 0
        whole = inputs['visual']
        inputs['visual'] = np.concatenate((np.zeros_like(whole) if self.arm=='head' else whole,
                                          np.zeros_like(head) if self.arm=='whole' else head),axis=-1)
        return inputs,target,known
