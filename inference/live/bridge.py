"""Bounded request/reply bridge to the isolated perception interpreter."""
import json
import os
from pathlib import Path
from queue import Queue,Empty
import subprocess
from threading import Thread


class PerceptionBridge:
    def __init__(self,python,runtime_config,timeout=30):
        environment=dict(os.environ,PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
        environment.pop('PYTHONHOME',None);environment.pop('PYTHONPATH',None)
        self.process=subprocess.Popen([str(python),'-u','-m','inference.live.perception_worker',
            '--runtime-config',str(runtime_config)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=None,text=True,encoding='utf-8',bufsize=1,env=environment,
            cwd=Path(__file__).resolve().parents[2],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        self.timeout=timeout;self.responses=Queue(maxsize=1)
        def reader():
            try:
                while True:
                    line=self.process.stdout.readline(4*1024*1024+1)
                    if not line or len(line)>4*1024*1024:
                        self.responses.put(None);break
                    self.responses.put(line)
            except (OSError,ValueError):pass
        self.reader=Thread(target=reader,daemon=True);self.reader.start()

    def call(self,request):
        try:
            self.process.stdin.write(json.dumps(request,allow_nan=False)+'\n');self.process.stdin.flush()
            line=self.responses.get(timeout=self.timeout)
            if line is None:raise RuntimeError('Perception worker exited; inspect stderr log')
            response=json.loads(line)
            if not response.get('ok'):raise RuntimeError(response.get('error','Perception failed'))
            return response['result']
        except Empty:
            self.close();raise TimeoutError('Perception timed out; session closed') from None
        except Exception:
            self.close();raise

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=3)
        for stream in (self.process.stdin,self.process.stdout):
            if stream is not None:stream.close()
