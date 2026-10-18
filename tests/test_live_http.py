import json
from pathlib import Path
import threading
import time
import unittest
from urllib.request import Request,urlopen
from urllib.error import HTTPError

from inference.live.server import create_server,SessionService


class Models:
    bundle=dict(head_crop={'version':'driver-relative-rect-v1','xyxy':[.05,0,.65,.65]},
                models={name:dict(sha256='a'*64) for name in ('coverage25','gap80','distraction','actions','eye','rgb_encoder')})


class Bridge:
    closed=False
    def call(self,request):
        if request['operation']=='open':return {'status':'ready'}
        if request['jpeg']=='fail':raise RuntimeError('test perception failure')
        return dict(samples=[],perception={'eye_origin':'missing'})
    def close(self):self.closed=True


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.service=SessionService(Models(),'runtime.json','python',bridge_factory=lambda:Bridge())
        self.server=create_server(self.service,'test-token',port=0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url='http://127.0.0.1:'+str(self.server.server_port)
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.service.close();self.thread.join()
    def request(self,path,body=None,method=None,token='test-token'):
        data=json.dumps(body).encode() if body is not None else None
        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}
        try:
            with urlopen(Request(self.url+path,data=data,headers=headers,method=method),timeout=5) as reply:
                return reply.status,json.load(reply)
        except HTTPError as error:return error.code,json.load(error)
    def open(self):
        code,result=self.request('/api/live/sessions',dict(view='body',image_size=[640,480],
            driver_roi=[0,0,1,1],driver_region_confirmed_exclusive=False))
        self.assertEqual(code,201);return result['session_id']
    def test_auth_and_health(self):
        self.assertEqual(self.request('/api/live/status',token='wrong')[0],401)
        self.assertEqual(self.request('/api/live/health',token='wrong')[0],200)
    def test_exclusive_session_and_stop(self):
        session=self.open()
        code,_=self.request('/api/live/sessions',{})
        self.assertEqual(code,409)
        code,result=self.request('/api/live/sessions/'+session+'/frames',dict(timestamp_ms=0,jpeg='ok'))
        self.assertEqual(code,200);self.assertEqual(result['status'],'warming')
        self.assertIsNone(result['result'])
        self.assertEqual(self.request('/api/live/sessions/'+session,method='DELETE')[0],200)
        self.assertEqual(self.request('/api/live/sessions/'+session+'/frames',dict(timestamp_ms=40,jpeg='ok'))[0],404)
    def test_failed_frame_closes_session(self):
        session=self.open()
        code,_=self.request('/api/live/sessions/'+session+'/frames',dict(timestamp_ms=0,jpeg='fail'))
        self.assertEqual(code,503)
        _,status=self.request('/api/live/status')
        self.assertIsNone(status['session_id']);self.assertIsNone(status['result'])
    def test_unknown_api_is_not_html(self):
        self.assertEqual(self.request('/api/v1/risk')[0],404)

    def test_stop_waits_for_inflight_frame(self):
        session=self.open();bridge=self.service.bridge
        entered=threading.Event();release=threading.Event();original=bridge.call
        def blocked(request):
            entered.set();release.wait(3);return original(request)
        bridge.call=blocked
        frame=threading.Thread(target=lambda:self.request('/api/live/sessions/'+session+'/frames',dict(timestamp_ms=0,jpeg='ok')))
        frame.start();self.assertTrue(entered.wait(2))
        results=[]
        stop=threading.Thread(target=lambda:results.append(self.request('/api/live/sessions/'+session,method='DELETE')))
        stop.start();time.sleep(.1);release.set();frame.join();stop.join()
        self.assertEqual(results[0][0],200);self.assertTrue(bridge.closed)
        self.assertIsNone(self.service.id)

    def test_idle_session_expires_without_polling(self):
        self.open();bridge=self.service.bridge
        self.service.last_received=time.monotonic()-61
        deadline=time.monotonic()+3
        while not bridge.closed and time.monotonic()<deadline:time.sleep(.05)
        self.assertTrue(bridge.closed);self.assertIsNone(self.service.id)


if __name__=='__main__':unittest.main()
