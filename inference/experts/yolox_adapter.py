"""Bounded YOLOX 0.3.0 COCO nano/tiny object expert, without tracking.

Only explicitly supplied, hash-verified local checkpoints are loaded. Network
execution stays on the requested device; decoded tensors explicitly transfer
to CPU for NumPy validation/filtering/class-aware NMS. That postprocessing
boundary is intentional and is not an accelerator inference fallback.

The nonlegacy BGR transform follows the official 0.3.0 implementation:
https://github.com/Megvii-BaseDetection/YOLOX/blob/0.3.0/yolox/data/data_augment.py
https://github.com/Megvii-BaseDetection/YOLOX/blob/0.3.0/tools/demo.py
"""

import math


_IMPLEMENTATION = 'driveguard-yolox-v1'
_COCO_TARGETS = {39: ('bottle', 'bottle'), 41: ('cup', 'cup'), 67: ('cell phone', 'phone')}
_MAX_FRAME_SIDE = 16384
_MAX_FRAME_PIXELS = 64 * 1024 * 1024


class YOLOXAdapter:
    """An explicit backend, float32 YOLOX detector with a canonical object result.

    ``input_size`` is a square multiple of 32 in [32, 1280]. Frames are uint8
    HWC BGR, at most 16384 pixels per side and 64 Mi pixels in total. Decoded
    output cannot exceed the three feature-grid sizes implied by input_size.
    Equal-score NMS ties retain the earlier decoded prediction. Instances are
    sequential resources; ``close`` is idempotent and prevents future use.
    """

    def __init__(self, asset, *, variant='yolox-nano', backend='cpu', device_index=0,
                 input_size=416, score_threshold=0.25, nms_threshold=0.45):
        if variant not in ('yolox-nano', 'yolox-tiny'):
            raise ValueError('variant must be yolox-nano or yolox-tiny')
        if type(input_size) is not int or not 32 <= input_size <= 1280 or input_size % 32:
            raise ValueError('input_size must be a multiple of 32 within [32, 1280]')
        for name, value in (('score_threshold', score_threshold), ('nms_threshold', nms_threshold)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f'{name} must be a finite probability')
        if backend not in ('cpu', 'cuda', 'npu'):
            raise ValueError('backend must be cpu, cuda, or npu')
        if type(device_index) is not int or device_index < 0 or (backend == 'cpu' and device_index != 0):
            raise ValueError('device_index must be nonnegative, and zero for CPU')

        from inference.experts.assets import model_identity, verify_asset
        verified = verify_asset(asset)
        # Optional packages must never be imported by configuration-only users.
        import cv2
        import numpy as np
        import torch
        import yolox
        from yolox.data.datasets import COCO_CLASSES
        from yolox.exp import get_exp
        from training.common.device import select_device

        if getattr(yolox, '__version__', None) != '0.3.0':
            raise ValueError('only YOLOX 0.3.0 is supported')
        if (len(COCO_CLASSES) != 80 or len(set(COCO_CLASSES)) != 80
                or any(COCO_CLASSES[index] != names[0] for index, names in _COCO_TARGETS.items())):
            raise ValueError('COCO80 class names/order do not match the supported mapping')
        self._device = select_device(backend, device_index)
        experiment = get_exp(None, variant)
        if experiment.num_classes != 80:
            raise ValueError('YOLOX experiment must have 80 classes')
        experiment.test_size = (input_size, input_size)
        model = experiment.get_model()
        checkpoint = torch.load(verified['path'], map_location='cpu', weights_only=True)
        if (not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('model'), dict)
                or not all(isinstance(k, str) and torch.is_tensor(v)
                           for k, v in checkpoint['model'].items())):
            raise ValueError('checkpoint must contain a model tensor state dictionary')
        model.load_state_dict(checkpoint['model'], strict=True)
        model.head.decode_in_inference = True
        model.eval()
        self._model = model.to(device=self._device, dtype=torch.float32)
        self._torch, self._np, self._cv2 = torch, np, cv2
        self._input_size = input_size
        self._score_threshold = float(score_threshold)
        self._nms_threshold = float(nms_threshold)
        self._max_predictions = sum((input_size // stride) ** 2 for stride in (8, 16, 32))
        self.model_id = model_identity('yolox', verified, {
            'implementation': _IMPLEMENTATION, 'yolox_version': yolox.__version__,
            'torch_version': str(torch.__version__), 'opencv_version': cv2.__version__,
            'numpy_version': np.__version__, 'variant': variant,
            'backend': backend, 'device_index': device_index, 'input_size': input_size,
            'score_threshold': self._score_threshold, 'nms_threshold': self._nms_threshold,
            'preprocess': 'bgr-top-left-pad114-chw-float32-nonlegacy',
            'postprocess': 'cpu-numpy-class-aware-nms-stable-input-order',
            'class_mapping': {str(k): list(v) for k, v in _COCO_TARGETS.items()},
        })

    def predict(self, bgr_uint8_HWC, timestamp_ms):
        """Return a fresh canonical objects expert payload in original pixels."""
        if self._model is None:
            raise RuntimeError('YOLOX adapter is closed')
        np = self._np
        frame = bgr_uint8_HWC
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3
                or not 0 < frame.shape[0] <= _MAX_FRAME_SIDE
                or not 0 < frame.shape[1] <= _MAX_FRAME_SIDE
                or frame.shape[0] * frame.shape[1] > _MAX_FRAME_PIXELS):
            raise ValueError('frame must be a bounded, nonempty uint8 HWC BGR array')
        if type(timestamp_ms) is not int or not 0 <= timestamp_ms <= 2**63 - 1:
            raise ValueError('timestamp_ms must be a nonnegative signed-64-bit integer')
        height, width = frame.shape[:2]
        ratio = min(self._input_size / height, self._input_size / width)
        resized_width, resized_height = int(width * ratio), int(height * ratio)
        if not resized_width or not resized_height:
            raise ValueError('frame aspect ratio would produce a zero-sized YOLOX input')
        canvas = np.full((self._input_size, self._input_size, 3), 114, dtype=np.uint8)
        canvas[:resized_height, :resized_width] = self._cv2.resize(
            frame, (resized_width, resized_height), interpolation=self._cv2.INTER_LINEAR)
        image = np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32)
        tensor = self._torch.from_numpy(image).unsqueeze(0).to(self._device)
        with self._torch.inference_mode():
            output = self._model(tensor)
        if (not self._torch.is_tensor(output) or output.ndim != 3 or output.shape[0] != 1
                or output.shape[2] != 85 or output.shape[1] > self._max_predictions
                or not output.is_floating_point()):
            raise ValueError('YOLOX output must be a bounded floating [1, N, 85] tensor')
        # Explicit CPU postprocessing, common to all three network backends.
        rows = output.detach().to(device='cpu', dtype=self._torch.float64).numpy()[0]
        if (not np.isfinite(rows).all() or (rows[:, 2:4] < 0).any()
                or (rows[:, 4:] < 0).any() or (rows[:, 4:] > 1).any()):
            raise ValueError('YOLOX output contains invalid geometry or probabilities')
        items = self._postprocess(rows, ratio, width, height)
        return {'status': 'ok', 'timestamp_ms': timestamp_ms, 'model_id': self.model_id, 'items': items}

    def _postprocess(self, rows, ratio, width, height):
        np = self._np
        if not len(rows):
            return []
        classes = rows[:, 5:].argmax(axis=1)
        scores = rows[:, 4] * rows[np.arange(len(rows)), 5 + classes]
        eligible = (scores >= self._score_threshold) & np.isin(classes, list(_COCO_TARGETS))
        rows, classes, scores = rows[eligible], classes[eligible], scores[eligible]
        boxes = np.concatenate((rows[:, :2] - rows[:, 2:4] / 2,
                                rows[:, :2] + rows[:, 2:4] / 2), axis=1) / ratio
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)
        valid = ((boxes[:, 2:] - boxes[:, :2]) >= 1e-6).all(axis=1)
        boxes, classes, scores = boxes[valid], classes[valid], scores[valid]
        pending = np.argsort(-scores, kind='stable')
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        items = []
        while len(pending):
            current = pending[0]
            items.append({'category': _COCO_TARGETS[int(classes[current])][1],
                          'box': boxes[current].tolist(), 'score': float(scores[current]), 'track_id': None})
            others = pending[1:]
            overlap = np.maximum(0, np.minimum(boxes[current, 2:], boxes[others, 2:])
                                 - np.maximum(boxes[current, :2], boxes[others, :2]))
            intersection = overlap[:, 0] * overlap[:, 1]
            iou = intersection / (areas[current] + areas[others] - intersection)
            pending = others[(classes[others] != classes[current]) | (iou <= self._nms_threshold)]
        return items

    def close(self):
        """Release this adapter's model reference; never reset a shared device."""
        self._model = None
