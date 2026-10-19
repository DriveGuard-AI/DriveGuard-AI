"""Bounded recording-level scheduling; the parent exclusively owns the batch lock."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
import multiprocessing
import os
from pathlib import Path
import signal
import uuid

SUBSETS = ('distraction', 'drowsiness', 'gaze')


def validate_quotas(quotas):
    if (not isinstance(quotas, dict) or set(quotas) != set(SUBSETS)
            or any(type(v) is not int or not 1 <= v <= 32 for v in quotas.values())
            or sum(quotas.values()) > 32):
        raise ValueError('Expected positive integer dist/drows/gaze quotas, total at most 32')
    return quotas


def dispatch(jobs, quotas, submit, should_stop=lambda: False, on_done=None):
    validate_quotas(quotas)
    if (len({j['job_id'] for j in jobs}) != len(jobs)
            or any(j['subset'] not in SUBSETS for j in jobs)):
        raise ValueError('Duplicate jobs or unknown subsets')
    queues = {s: deque(sorted((j for j in jobs if j['subset'] == s),
                             key=lambda j: j['video_size_bytes'], reverse=True)) for s in SUBSETS}
    running = {}; active = dict.fromkeys(SUBSETS, 0); results = []
    while running or any(queues.values()):
        for subset in SUBSETS:
            while queues[subset] and active[subset] < quotas[subset] and not should_stop():
                job = queues[subset].popleft()
                running[submit(job)] = job
                active[subset] += 1
        if not running:
            break
        done, _ = wait(running, timeout=1, return_when=FIRST_COMPLETED)
        for future in done:
            job = running.pop(future); active[job['subset']] -= 1
            try:
                result = future.result()
            except Exception as error:
                result = dict(job_id=job['job_id'], status='failed', error=str(error))
            results.append(result)
            if on_done: on_done(result)
    return results, [j['job_id'] for q in queues.values() for j in q]


@contextmanager
def thread_environment(threads):
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError('threads must be an integer in [1,8]')
    keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    previous = {k: os.environ.get(k) for k in keys}
    try:
        for key in keys: os.environ[key] = str(threads)
        yield
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


def _worker_init(threads):
    # Spawned workers inherit thread env before importing NumPy/model packages.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import torch
    import cv2
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(threads)


def _process_job(context, job, log_path):
    from training.dmd.batch import _inputs, _existing, _extract
    # Redirect native MediaPipe diagnostics too; each task has a unique log.
    with open(log_path, 'x', encoding='utf-8', buffering=1) as log:
        saved = [os.dup(fd) for fd in (1, 2)]
        try:
            import sys
            sys.stdout.flush(); sys.stderr.flush()
            for fd in (1, 2): os.dup2(log.fileno(), fd)
            print(f'START {job["video"]}', file=log)
            source, doc, fingerprint, evidence, stat = _inputs(context, job)
            existing = _existing(context, job, fingerprint)
            status = 'reused' if existing else 'completed'
            if existing is None:
                existing = _extract(context, job, source, doc, fingerprint, evidence, stat,
                                    lambda message: print(message, file=log))
            return dict(job_id=job['job_id'], status=status,
                        attempt=existing[0].relative_to(context['output']).as_posix(), log=str(log_path))
        except Exception as error:
            import traceback
            traceback.print_exc(file=log)
            return dict(job_id=job['job_id'], status='failed', error=str(error), log=str(log_path))
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            for fd, copy in zip((1, 2), saved):
                os.dup2(copy, fd); os.close(copy)


def run_parallel(config_path, dataset_root, output_dir, *, quotas, threads=1, progress=None):
    from training.dmd.batch import _context, _lock
    from training.dmd.cache import write_json
    validate_quotas(quotas)
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    context = _context(config_path, dataset_root, output_dir, gaze=True)
    # Caller must clear only its own previous stop request before a new launch.
    stop_file = context['output'] / '.stop-request'
    previous = {}
    with thread_environment(threads), _lock(context['output']):
        run_id = uuid.uuid4().hex
        logs = context['output'] / 'logs' / run_id
        logs.mkdir(parents=True)
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, stop)
            with ProcessPoolExecutor(max_workers=sum(quotas.values()),
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=_worker_init, initargs=(threads,)) as executor:
                def submit(job):
                    if progress: progress(f'START {job["subset"]} {job["job_id"]} {job["video"]}')
                    return executor.submit(_process_job, context, job, logs / (job['job_id']+'.log'))
                def completed(result):
                    if progress: progress(f'{result["status"]}: {result["job_id"]}')
                jobs, pending = dispatch(context['selected'], quotas, submit,
                                         lambda: stopped or stop_file.exists(), completed)
            result = {name:sum(j['status']==name for j in jobs) for name in ('completed','reused','failed')}
            result.update(jobs=jobs, pending=pending, training_ready=False,
                          status='stopped' if pending or stopped or stop_file.exists() else ('errors' if result['failed'] else 'complete'),
                          quotas=quotas, threads=threads, logs=str(logs))
            report = context['output'] / 'reports' / (run_id+'.json')
            report.parent.mkdir(exist_ok=True)
            write_json(report, result)
            result['report'] = str(report)
            return result
        finally:
            for sig, handler in previous.items(): signal.signal(sig, handler)
