"""Bounded offline evaluation costs, not raw-video deployment benchmarks."""
from time import perf_counter

import torch


def modelstate_footprint(state):
    """Serialized tensor payload size; excludes Python/allocator/runtime overhead."""
    tensors = [v for v in state.values() if torch.is_tensor(v)]
    return dict(state_tensor_count=len(tensors),
                tensor_bytes=sum(v.numel()*v.element_size() for v in tensors),
                scope='state_tensor_payload_excludes_runtime_and_activations')


def model_footprint(model):
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    parameter_bytes = sum(v.numel()*v.element_size() for v in parameters)
    buffer_bytes = sum(v.numel()*v.element_size() for v in buffers)
    return dict(parameter_count=sum(v.numel() for v in parameters),
                parameter_bytes=parameter_bytes, buffer_bytes=buffer_bytes,
                tensor_bytes=parameter_bytes+buffer_bytes,
                scope='model_tensor_payload_excludes_runtime_and_activations')


def measure_pass(operation, backend, device_index=0):
    """Time a caller-labelled cache-based pass; synchronize asynchronous devices.

    No warmup is inserted and no operation is repeated. The caller must state
    whether model loading and cache access are included. Device allocation peaks
    include other tensors still resident in this process; they are not the peak
    of a production composite or system-wide device usage. CPU memory is null.
    """
    if (backend not in ('cpu', 'cuda', 'npu') or type(device_index) is not int
            or device_index < 0 or (backend == 'cpu' and device_index != 0)):
        raise ValueError('Invalid cost measurement backend/device')
    api = None
    starting = peak = None
    if backend != 'cpu':
        api = getattr(torch, backend, None)
        if api is None or not callable(getattr(api, 'synchronize', None)):
            raise RuntimeError(f'{backend} timing requires an initialized accelerator API')
        api.synchronize(device_index)
        memory_api = all(callable(getattr(api, key, None)) for key in (
            'reset_peak_memory_stats', 'max_memory_allocated', 'memory_allocated'))
        if memory_api:
            api.reset_peak_memory_stats(device_index)
            starting = int(api.memory_allocated(device_index))
    begin = perf_counter()
    result = operation()
    if api is not None:
        api.synchronize(device_index)
    elapsed = perf_counter()-begin
    if api is not None and memory_api:
        peak = int(api.max_memory_allocated(device_index))
    return result, dict(wall_seconds=elapsed, backend=backend, device_index=device_index,
        starting_device_allocated_bytes=starting, peak_device_allocated_bytes=peak,
        includes_rgb_extraction=False, deployment_latency=False,
        scope='single_cached_feature_pass_including_host_work_no_warmup',
        memory_scope='process_tensor_allocation_peak_during_stage_not_composite_residency')
