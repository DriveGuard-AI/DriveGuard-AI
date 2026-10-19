"""Hash-verified official YOLO11s; direct float32 forward, CPU NumPy NMS.

Ultralytics checkpoints use pickle. Only the separately downloaded, trusted
official asset declared in configuration is accepted; verification happens
before vendor construction. This module never downloads models.
"""
import importlib.metadata

from inference.experts.protocol import ULTRALYTICS_VERSION, settings_identity, validate_frame

_TARGETS = {39: ('bottle', 'bottle'), 41: ('cup', 'cup'), 67: ('cell phone', 'phone')}


class YOLO11Adapter:
    def __init__(self, asset, *, variant='yolo11s', backend='npu', device_index=0,
                 input_size=640, score_threshold=.25, nms_threshold=.45):
        request_identity = settings_identity(asset, variant=variant, backend=backend,
            device_index=device_index, input_size=input_size, score_threshold=score_threshold,
            nms_threshold=nms_threshold)
        from inference.experts.assets import model_identity, verify_asset
        verified = verify_asset(asset)
        import cv2
        import numpy as np
        import torch
        import ultralytics
        from training.common.device import select_device
        if ultralytics.__version__ != ULTRALYTICS_VERSION:
            raise ValueError('only Ultralytics ' + ULTRALYTICS_VERSION + ' is supported')
        self._device = select_device(backend, device_index)
        model = ultralytics.YOLO(verified['path']).model
        names = model.names
        if isinstance(names, dict):
            if set(names) != set(range(80)):
                raise ValueError('YOLO11 requires COCO80 class names')
            names = [names[i] for i in range(80)]
        if (not isinstance(names, (list, tuple)) or len(names) != 80 or len(set(names)) != 80
                or any(names[i] != pair[0] for i, pair in _TARGETS.items())):
            raise ValueError('COCO80 class order does not match supported targets')
        self._model = model.eval().float().to(self._device)
        # CPU convolutions are measurably faster from a channels_last layout and
        # the result is bit-identical (max abs difference 0.0 on the released
        # yolo11s asset). Accelerator backends keep their own validated layout.
        self._channels_last = self._device.type == 'cpu'
        if self._channels_last:
            self._model = self._model.to(memory_format=torch.channels_last)
        self._torch, self._np, self._cv2 = torch, np, cv2
        self._input_size, self._score_threshold, self._nms_threshold = input_size, score_threshold, nms_threshold
        self._max_predictions = sum((input_size // stride) ** 2 for stride in (8, 16, 32))
        packages = {}
        for package in ('torch-npu', 'ultralytics', 'torch', 'numpy', 'opencv-python', 'opencv-python-headless'):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        device_name = 'cpu'
        if backend != 'cpu':
            device_name = str(getattr(torch, backend).get_device_name(device_index))
        self.identity = dict(request_identity=request_identity, backend=backend,
            device=str(self._device), device_name=device_name, packages=packages,
            ultralytics_version=ultralytics.__version__, torch_version=str(torch.__version__),
            numpy_version=np.__version__, opencv_version=cv2.__version__,
            preprocess='rgb-centered-letterbox114-float32-div255',
            postprocess='cpu-numpy-class-argmax-class-aware-stable-nms')
        self.model_id = model_identity('yolo11', verified, self.identity)

    def predict(self, bgr_uint8_HWC, timestamp_ms):
        if self._model is None:
            raise RuntimeError('YOLO11 adapter is closed')
        frame = bgr_uint8_HWC
        validate_frame(frame, timestamp_ms)
        np, torch = self._np, self._torch
        height, width = frame.shape[:2]
        ratio = min(self._input_size / height, self._input_size / width)
        resized_width, resized_height = round(width * ratio), round(height * ratio)
        if not resized_width or not resized_height:
            raise ValueError('frame aspect ratio produces zero-sized input')
        left = round((self._input_size - resized_width) / 2 - .1)
        top = round((self._input_size - resized_height) / 2 - .1)
        canvas = np.full((self._input_size, self._input_size, 3), 114, np.uint8)
        canvas[top:top + resized_height, left:left + resized_width] = self._cv2.resize(
            frame, (resized_width, resized_height), interpolation=self._cv2.INTER_LINEAR)
        image = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255
        tensor = torch.from_numpy(image).unsqueeze(0).to(self._device)
        if self._channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        with torch.inference_mode():
            output = self._model(tensor)
        if isinstance(output, tuple) and output:
            output = output[0]
        if (not torch.is_tensor(output) or output.ndim != 3 or output.shape[0] != 1
                or output.shape[1] != 84 or output.shape[2] > self._max_predictions
                or not output.is_floating_point()):
            raise ValueError('YOLO11 output must be bounded floating [1, 84, N]')
        rows = output.detach().to(device='cpu', dtype=torch.float64).numpy()[0].T
        if (not np.isfinite(rows).all() or (rows[:, 2:4] < 0).any()
                or (rows[:, 4:] < 0).any() or (rows[:, 4:] > 1).any()):
            raise ValueError('YOLO11 output contains invalid geometry or probabilities')
        items = self._postprocess(rows, ratio, left, top, width, height)
        return dict(status='ok', timestamp_ms=timestamp_ms, model_id=self.model_id, items=items)

    def _postprocess(self, rows, ratio, left, top, width, height):
        np = self._np
        if not len(rows):
            return []
        classes = rows[:, 4:].argmax(axis=1)
        scores = rows[np.arange(len(rows)), 4 + classes]
        keep = (scores >= self._score_threshold) & np.isin(classes, list(_TARGETS))
        rows, classes, scores = rows[keep], classes[keep], scores[keep]
        boxes = np.concatenate((rows[:, :2] - rows[:, 2:4] / 2,
                                rows[:, :2] + rows[:, 2:4] / 2), axis=1)
        boxes = (boxes - np.array([left, top, left, top])) / ratio
        boxes[:, 0::2] = boxes[:, 0::2].clip(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, height)
        valid = ((boxes[:, 2:] - boxes[:, :2]) >= 1e-6).all(axis=1)
        boxes, classes, scores = boxes[valid], classes[valid], scores[valid]
        pending = np.argsort(-scores, kind='stable')
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        items = []
        while len(pending):
            current, others = pending[0], pending[1:]
            items.append(dict(category=_TARGETS[int(classes[current])][1],
                box=boxes[current].tolist(), score=float(scores[current]), track_id=None))
            overlap = np.maximum(0, np.minimum(boxes[current, 2:], boxes[others, 2:])
                                 - np.maximum(boxes[current, :2], boxes[others, :2]))
            intersection = overlap[:, 0] * overlap[:, 1]
            iou = intersection / (areas[current] + areas[others] - intersection)
            pending = others[(classes[others] != classes[current]) | (iou <= self._nms_threshold)]
        return items

    def close(self):
        self._model = None
