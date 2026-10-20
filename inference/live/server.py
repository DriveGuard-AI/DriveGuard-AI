"""Loopback-only webcam integration API; separate from the planned business API."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import signal
from threading import Lock
import time
from urllib.parse import urlsplit
from uuid import uuid4


class ApiError(Exception):
    def __init__(self,status,message):self.status=status;super().__init__(message)


class SessionService:
    def __init__(self,models,runtime_config,perception_python,bridge_factory=None):
        from inference.live.bridge import PerceptionBridge
        self.models=models
        self.bridge_factory=bridge_factory or (lambda:PerceptionBridge(perception_python,runtime_config))
        self.id=None;self.bridge=None;self.stream=None;self.latest=None
        self.last_error=None;self.last_received=None;self.updated=None;self.last_capture=None;self.expires=None

    def status(self):
        if self.last_received is not None and time.monotonic()-self.last_received>60:self.close()
        fresh=self.expires is not None and time.monotonic()<self.expires
        return dict(status='streaming' if self.id and fresh else 'waiting' if self.id else 'idle',
            session_id=self.id,result=self.latest if fresh else None,error=self.last_error,
            models={k:v['sha256'] for k,v in self.models.bundle['models'].items()},
            head_crop=self.models.bundle['head_crop'],input_mode='live_camera',business_risk_enabled=False)

    def open(self,options):
        from inference.live.models import LiveStream
        from inference.live.perception_worker import validate_camera
        self.status()
        if self.id is not None:raise ApiError(409,'A camera session is already active; stop it first')
        validate_camera(options)
        bridge=self.bridge_factory()
        try:bridge.call(dict(operation='open',camera=options,head_crop=self.models.bundle['head_crop']))
        except Exception:bridge.close();raise
        self.bridge=bridge;self.id=uuid4().hex;self.stream=LiveStream(self.models,options['view'])
        self.latest=None;self.updated=None;self.last_error=None;self.last_capture=None
        self.last_received=time.monotonic()
        return dict(session_id=self.id,status='warming',warmup_ms=4000,output_period_ms=1000)

    def frame(self,session,request):
        if session!=self.id or self.id is None:raise ApiError(404,'No matching active camera session')
        try:
            if not isinstance(request,dict) or set(request)!={'timestamp_ms','jpeg'}:
                raise ValueError('Expected timestamp_ms and base64 jpeg')
            timestamp=request['timestamp_ms']
            if (type(timestamp) is not int or timestamp<0 or (self.last_capture is None and timestamp!=0) or
                    (self.last_capture is not None and not 0<timestamp-self.last_capture<=10000)):
                raise ValueError('Camera timestamps must start at zero and strictly increase; restart after >10s gap')
            if not isinstance(request['jpeg'],str) or len(request['jpeg'])>4*1024*1024:
                raise ValueError('Invalid or oversized JPEG')
            started=time.monotonic()
            reply=self.bridge.call(dict(operation='frame',**request))
            result=self.stream.consume(reply['samples'])
            elapsed=(time.monotonic()-started)*1000
            self.last_received=time.monotonic();self.last_capture=timestamp
            if result is not None:self.latest=result;self.updated=time.monotonic()
            age=timestamp-self.latest['window_end_ms']+elapsed if self.latest else None
            fresh=age is not None and age<=1500
            if age is not None:
                deadline=time.monotonic()+max(0.,1500-age)/1000
                self.expires=deadline if result is not None or self.expires is None else min(self.expires,deadline)
            status='ready' if fresh else 'warming' if self.latest is None else 'stale'
            return dict(session_id=session,status=status,result=self.latest if fresh else None,
                capture_timestamp_ms=timestamp,processing_ms=elapsed,result_age_ms=age,
                perception=reply['perception'],feature_slots=self.stream.slot)
        except Exception as error:
            self.last_error=str(error);self.close();raise

    def stop(self,session):
        if session!=self.id or self.id is None:raise ApiError(404,'No matching active camera session')
        self.close();return dict(status='stopped',result=None)

    def close(self):
        if self.bridge is not None:self.bridge.close()
        self.id=self.bridge=self.stream=self.latest=self.updated=self.last_received=self.last_capture=self.expires=None


def create_server(service,token,port=8765,executor=None):
    if not token:raise ValueError('A nonempty bearer token is required')
    lock=Lock();pool=executor;accepted=[float('-inf')]
    class Handler(BaseHTTPRequestHandler):
        # 逐帧调用必须复用连接：HTTP/1.0 默认每帧关闭，公网每帧都会重新握手。
        protocol_version='HTTP/1.1'
        # 逐帧小请求/小响应关闭 Nagle，避免约 40ms 的延迟 ACK。
        disable_nagle_algorithm=True
        def log_message(self,*args):pass
        def reply(self,status,value,kind='application/json; charset=utf-8'):
            encoded=json.dumps(value,ensure_ascii=False,allow_nan=False).encode() if kind.startswith('application/json') else value
            # 错误响应可能没有读完请求体，关闭连接避免下一条请求错位。
            if status>=400:self.close_connection=True
            self.send_response(status);self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(encoded)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; media-src 'self' blob:; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(encoded)
        def body(self):
            if self.headers.get_content_type()!='application/json':raise ApiError(415,'Expected application/json')
            try:length=int(self.headers.get('Content-Length','0'))
            except ValueError:raise ApiError(400,'Invalid content length') from None
            if not 0<length<=5*1024*1024:raise ApiError(413,'Request size must be 1..5 MiB')
            self.connection.settimeout(15)
            data=self.rfile.read(length)
            if len(data)!=length:raise ApiError(400,'Incomplete request')
            try:return json.loads(data,parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            except (ValueError,UnicodeError):raise ApiError(400,'Invalid JSON') from None
        def route(self):
            host=self.headers.get('Host','');origin=self.headers.get('Origin')
            if urlsplit('//'+host).hostname not in ('localhost','127.0.0.1','::1'):
                raise ApiError(403,'Use localhost or an SSH tunnel')
            if origin and urlsplit(origin).netloc!=host:raise ApiError(403,'Cross-origin requests are disabled')
            path=urlsplit(self.path).path
            if self.command=='GET' and path in ('/','/camera.js'):
                filename='camera.html' if path=='/' else 'camera.js'
                self.reply(200,Path(__file__).with_name(filename).read_bytes(),
                    'text/html; charset=utf-8' if path=='/' else 'text/javascript; charset=utf-8');return
            if self.command=='GET' and path=='/api/live/health':
                self.reply(200,dict(status='ok',service='live-composite-v1',models_loaded=True));return
            if not secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                raise ApiError(401,'Bearer token required')
            body=self.body() if self.command=='POST' else None
            stopping=self.command=='DELETE' and path.startswith('/api/live/sessions/')
            acquired=lock.acquire(timeout=35) if stopping else lock.acquire(blocking=False)
            if not acquired:raise ApiError(409,'Inference busy; keep only one frame in flight')
            try:
                def operation():
                    if self.command=='GET' and path=='/api/live/status':return 200,service.status()
                    if self.command=='POST' and path=='/api/live/sessions':return 201,service.open(body)
                    parts=path.strip('/').split('/')
                    if len(parts)==4 and parts[:3]==['api','live','sessions'] and self.command=='DELETE':
                        return 200,service.stop(parts[3])
                    if len(parts)==5 and parts[:3]==['api','live','sessions'] and parts[4]=='frames' and self.command=='POST':
                        return 200,service.frame(parts[3],body)
                    raise ApiError(404,'Unknown live integration endpoint')
                status,result=pool.submit(operation).result() if pool else operation()
            finally:lock.release()
            self.reply(status,result)
        def dispatch(self):
            try:self.route()
            except ApiError as error:self.reply(error.status,dict(error=str(error),result=None))
            except (ValueError,KeyError,TypeError) as error:self.reply(422,dict(error=str(error),result=None))
            except (BrokenPipeError,ConnectionResetError):pass
            except Exception as error:self.reply(503,dict(error=str(error),result=None))
        do_GET=do_POST=do_DELETE=dispatch
    class Server(ThreadingHTTPServer):
        def process_request(self,request,client_address):
            # serve_forever calls service_actions() right after this returns, so
            # record the accept before the sweep can consider itself idle.
            accepted[0]=time.monotonic()
            super().process_request(request,client_address)
        def service_actions(self):
            # serve_forever calls this even when the browser has disappeared, but
            # it also calls it immediately after accepting a connection. The
            # sweep is advisory and /api/live/status performs the same 60 second
            # expiry check, so stand down while requests arrive instead of
            # holding the lock the new request is about to need. One second of
            # quiet is enough: an accepted connection is recorded before this
            # runs, so the post-accept collision is excluded deterministically
            # while a vanished browser still loses its session promptly.
            if time.monotonic()-accepted[0]<1.:return
            if not lock.acquire(blocking=False):return
            try:
                if pool:pool.submit(service.status).result()
                else:service.status()
            finally:lock.release()
    server=Server(('127.0.0.1',port),Handler);server.daemon_threads=True
    return server


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','perception-python','token-file'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--runtime-config',help='Optional same-model detector endpoint override')
    parser.add_argument('--backend',choices=('cpu','cuda','npu'),default='npu')
    parser.add_argument('--device-index',type=int,default=0);parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args(argv)
    from inference.live.bundle import load_bundle
    from inference.perception.runtime import load_runtime_config,runtime_signature
    bundle,_=load_bundle(args.bundle);original=Path(args.bundle).resolve().parent/bundle['runtime']
    runtime=Path(args.runtime_config).resolve() if args.runtime_config else original
    if runtime_signature(load_runtime_config(runtime))!=runtime_signature(load_runtime_config(original)):
        parser.error('Runtime override changes perception identity; only detector endpoint/asset locations may differ')
    token=Path(args.token_file).read_text(encoding='utf-8').strip()
    if len(token)<24:parser.error('Token file must contain at least 24 characters')
    service=server=None
    # All accelerator operations, including loading, happen on this same thread.
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='live-model') as pool:
        previous=signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            def boot():
                from inference.live.models import ModelSet
                return SessionService(ModelSet(args.bundle,args.backend,args.device_index),runtime,args.perception_python)
            service=pool.submit(boot).result()
            server=create_server(service,token,args.port,pool)
            print(json.dumps(dict(status='ready',url=f'http://localhost:{server.server_port}',
                                  candidate=bundle['candidate'],input_mode='live_camera')),flush=True)
            server.serve_forever(poll_interval=.25)
        except KeyboardInterrupt:pass
        finally:
            if server is not None:server.server_close()
            if service is not None:pool.submit(service.close).result()
            signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':main()
