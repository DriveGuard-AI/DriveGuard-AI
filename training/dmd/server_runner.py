"""Server supervisor: warmup detector, extract, publish, release owned subprocesses."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from training.common.config import read_json
from training.dmd.gaze_batch import load_batch
from training.dmd.parallel import validate_quotas


def perception_env(source=None):
    env=dict(os.environ if source is None else source)
    for key in ('PYTHONPATH','PYTHONHOME'): env.pop(key,None)
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1')
    return env


def commands(npu_python,cpu_python,config,runtime,dataset,output,session,quotas,threads):
    shared=['--config',str(config),'--dataset-root',str(dataset),'--output-dir',str(output)]
    return dict(service=[str(npu_python),'-u','-m','inference.experts.detector_service',
                          '--config',str(runtime),'--ready-file',str(Path(session)/'detector-ready.json')],
        extract=[str(cpu_python),'-u','-m','training.dmd.gaze_batch','run',*shared,
                 '--workers-dist',str(quotas[0]),'--workers-drows',str(quotas[1]),
                 '--workers-gaze',str(quotas[2]),'--threads',str(threads)],
        publish=[str(cpu_python),'-u','-m','training.dmd.gaze_batch','publish',*shared,
                 '--training-config','configs/training/relation_gaze.json'])


def _state(session, **value):
    value.update(pid=os.getpid(),updated_at=time.time())
    temporary=session/'status.pending.json'
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(session/'status.json')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def process_start(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except (OSError,IndexError):
        return None


def session_status(session):
    session=Path(session)
    owner=read_json(session/'owner.json')
    state=read_json(session/'status.json') if (session/'status.json').exists() else {'status':'starting'}
    start=process_start(owner['pid'])
    live=start is not None and start==owner.get('process_start')
    state['supervisor_alive']=live
    if not live and state.get('status') not in ('complete','failed','stopped'):
        state['last_recorded_status']=state.get('status')
        state['status']='supervisor_not_running'
    return state


def supervise(args):
    if sys.platform!='linux': raise ValueError('Supervisor is Linux server-only')
    config=Path(args.config).resolve(strict=True)
    dataset=Path(args.dataset_root).resolve(strict=True)
    output=Path(args.output_dir).resolve()
    if not dataset.is_dir() or output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError('Feature output must be outside Dataset and not its ancestor')
    loaded=load_batch(config)
    if any(not j['confirmed_exclusive'] for j in loaded['config']['jobs'].values()):
        raise ValueError('Some driver ROIs remain unconfirmed; preserve and complete the existing review')
    if loaded['runtime']['objects']['variant']!='yolo11s': raise ValueError('Expected migrated YOLO11 config')
    quotas=(args.workers_dist,args.workers_drows,args.workers_gaze)
    validate_quotas(dict(zip(('distraction','drowsiness','gaze'),quotas)))
    if type(args.threads) is not int or not 1<=args.threads<=8: raise ValueError('threads must be in [1,8]')
    for interpreter in (args.npu_python,args.cpu_python):
        if not Path(interpreter).is_file(): raise ValueError(f'Interpreter missing: {interpreter}')
    session=Path(args.session_dir).resolve()
    session.mkdir(parents=True,exist_ok=True)
    # This exact session may never be reused. New launches use mktemp directories.
    with (session/'owner.json').open('x',encoding='utf-8') as handle:
        json.dump(dict(pid=os.getpid(),process_start=process_start(os.getpid()),config=str(config),output=str(output)),handle)
    stop_file=session/'STOP'
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    old={sig:signal.signal(sig,stop) for sig in (signal.SIGTERM,signal.SIGINT)}
    service=extract=None
    launch_lock=output/'.supervisor.lock'
    owns_lock=False
    try:
        output.mkdir(parents=True,exist_ok=True)
        with launch_lock.open('x',encoding='utf-8') as handle:
            json.dump(dict(pid=os.getpid(),session=str(session)),handle)
        owns_lock=True
        if (output/'.batch.lock').exists(): raise ValueError('Output has active/stale batch lock; not stealing it')
        if (output/'.stop-request').exists():
            if not args.resume: raise ValueError('Previous stop request found; use --resume after previous run exited')
            (output/'.stop-request').unlink()
        cmds=commands(args.npu_python,args.cpu_python,config,config.parent/loaded['config']['runtime_config'],
                      dataset,output,session,quotas,args.threads)
        env=dict(os.environ, OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                 YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(session/'ultralytics'),PYTHONUNBUFFERED='1')
        with ExitStack() as stack:
            detector_log=stack.enter_context((session/'detector.log').open('x',encoding='utf-8'))
            extraction_log=stack.enter_context((session/'extraction.log').open('x',encoding='utf-8'))
            publish_log=stack.enter_context((session/'publish.log').open('x',encoding='utf-8'))
            service=subprocess.Popen(cmds['service'],env=env,stdin=subprocess.DEVNULL,stdout=detector_log,stderr=subprocess.STDOUT)
            _state(session,status='warming_up',detector_pid=service.pid)
            deadline=time.monotonic()+args.ready_timeout
            ready=session/'detector-ready.json'
            while not ready.exists():
                if service.poll() is not None: raise RuntimeError('Detector failed; inspect detector.log (no CPU fallback)')
                if stopped or stop_file.exists(): return _stopped(session)
                if time.monotonic()>deadline: raise RuntimeError('Detector warmup timeout; inspect detector.log')
                time.sleep(.5)
            if service.poll() is not None: raise RuntimeError('Detector exited after readiness')
            read_json(ready)
            extract=subprocess.Popen(cmds['extract'],env=perception_env(),stdin=subprocess.DEVNULL,
                                     stdout=extraction_log,stderr=subprocess.STDOUT)
            _state(session,status='extracting',detector_pid=service.pid,extractor_pid=extract.pid)
            while extract.poll() is None:
                if service.poll() is not None:
                    raise RuntimeError('Detector process exited during extraction; inspect detector.log')
                if stopped or stop_file.exists():
                    stopped=True
                    output.mkdir(parents=True,exist_ok=True)
                    (output/'.stop-request').touch(exist_ok=True)
                time.sleep(.5)
            if stopped or stop_file.exists(): return _stopped(session)
            if service.poll() is not None: raise RuntimeError('Detector process exited during extraction')
            if extract.returncode: raise RuntimeError('Extraction failed; inspect extraction.log and per-job logs')
            _state(session,status='publishing',detector_pid=service.pid)
            # Keep service available while publish fingerprints the same model.
            published=subprocess.Popen(cmds['publish'],env=perception_env(),stdin=subprocess.DEVNULL,
                                       stdout=publish_log,stderr=subprocess.STDOUT)
            publication_error=None
            try:
                while published.poll() is None:
                    if service.poll() is not None:
                        publication_error='Detector process exited during publication'
                    if (stopped or stop_file.exists()) and not stopped:
                        stopped=True
                        _state(session,status='stopping',phase='waiting_for_publication_lock_release')
                    time.sleep(.5)
            finally:
                # SIGTERM would bypass publisher's Python finally and strand .batch.lock.
                # Drain this finite validation/publication phase before releasing ownership.
                if published.poll() is None: published.wait()
            if publication_error: raise RuntimeError(publication_error)
            if stopped or stop_file.exists(): return _stopped(session)
            if published.returncode: raise RuntimeError('Coverage/cache publication failed; inspect publish.log')
            if service.poll() is not None: raise RuntimeError('Detector process exited during publication')
            _state(session,status='complete',manifest_output=str(session/'publish.log'))
            return 0
    except Exception as error:
        _state(session,status='failed',error=str(error))
        raise
    finally:
        # On a supervisor exception, ask the extractor to drain; keep detector alive.
        if extract is not None and extract.poll() is None:
            output.mkdir(parents=True,exist_ok=True)
            (output/'.stop-request').touch(exist_ok=True)
            extract.wait()
        if service is not None and service.poll() is None:
            service.terminate()
            try: service.wait(timeout=30)
            except subprocess.TimeoutExpired:
                service.kill(); service.wait()
        if owns_lock: launch_lock.unlink()
        for sig, handler in old.items(): signal.signal(sig,handler)


def _stopped(session):
    _state(session,status='stopped')
    return 2


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if len(argv)==2 and argv[0]=='status':
        try:
            print(json.dumps(session_status(argv[1]),ensure_ascii=False,indent=2))
            return 0
        except (OSError,ValueError,KeyError,TypeError) as error:
            print(f'Status unavailable: {error}',file=sys.stderr)
            return 2
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='runs/dmd-full-yolo11.local.json')
    parser.add_argument('--dataset-root',default='Dataset')
    parser.add_argument('--output-dir',default='runs/dmd-full-yolo11-features-001')
    parser.add_argument('--session-dir',required=True)
    parser.add_argument('--npu-python',default='.venv-yolo-npu/bin/python')
    parser.add_argument('--cpu-python',default='.venv-perception-arm64/bin/python')
    parser.add_argument('--workers-dist',type=int,default=6)
    parser.add_argument('--workers-drows',type=int,default=3)
    parser.add_argument('--workers-gaze',type=int,default=3)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--ready-timeout',type=int,default=600)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    try: return supervise(args)
    except (OSError,ValueError,RuntimeError,KeyError,TypeError) as error:
        print(f'Server extraction failed: {error}',file=sys.stderr)
        return 2


if __name__=='__main__': raise SystemExit(main())
