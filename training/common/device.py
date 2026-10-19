"""Accelerators are imported only after configuration-only validation."""
import importlib


def select_device(backend, device_index=0):
    if backend not in ('cpu', 'cuda', 'npu'):
        raise ValueError(f'Unknown backend: {backend}')
    if type(device_index) is not int or device_index < 0:
        raise ValueError('device_index must be a nonnegative integer')
    import torch
    if backend == 'cpu':
        if device_index != 0:
            raise ValueError('CPU device_index must be 0')
        return torch.device('cpu')
    if backend == 'cuda':
        if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
            raise RuntimeError(f'CUDA device {device_index} unavailable; no CPU fallback')
        torch.cuda.set_device(device_index)
        return torch.device(f'cuda:{device_index}')
    try:
        importlib.import_module('torch_npu')
    except (ImportError, OSError) as exc:
        raise RuntimeError('Ascend NPU requires matching torch_npu/CANN; no CPU fallback') from exc
    if not torch.npu.is_available() or device_index >= torch.npu.device_count():
        raise RuntimeError(f'Ascend NPU device {device_index} unavailable; no CPU fallback')
    torch.npu.set_device(device_index)
    return torch.device(f'npu:{device_index}')
