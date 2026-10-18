// Browser lifecycle contract without a camera, network, or DOM dependency.
const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const path=require('node:path');

function deferred(){let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};}
function harness(fetch){
  const elements={};let now=0;const timers=[];
  function element(id){return elements[id]??={value:id==='view'?'body':id==='roi'?'0,0,1,1':'',
    checked:true,disabled:false,videoWidth:640,videoHeight:480,textContent:'',children:[],listeners:{},
    replaceChildren(){this.children=[];},append(x){this.children.push(x);},
    getContext(){return {clearRect(){},strokeRect(){}};},play:async()=>{},
    cancelVideoFrameCallback(){},requestVideoFrameCallback(){return 1;},
    addEventListener(type,handler){(this.listeners[type]??=[]).push(handler);},
    dispatch(type,event){(this.listeners[type]||[]).forEach(handler=>handler(event));},
    getBoundingClientRect(){return {left:0,top:0,width:640,height:480};}};}
  const context=vm.createContext({document:{getElementById:element,createElement:()=>element(Symbol()),addEventListener(){}},
    window:{isSecureContext:true,addEventListener(){}},navigator:{mediaDevices:{}},
    location:{hash:'',pathname:'/'},history:{replaceState(){}},URLSearchParams,
    performance:{now:()=>now},AbortSignal,fetch,
    setInterval:fn=>timers.push(fn),setTimeout,clearTimeout});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../inference/live/camera.js'),'utf8'),context);
  return {context,element,run:code=>vm.runInContext(code,context),advance:ms=>{now+=ms;timers.forEach(fn=>fn());}};
}
const reply=data=>({ok:true,json:async()=>data});

test('cancel during session creation closes the late session instead of restarting',async()=>{
  const opened=deferred(),posted=deferred(),calls=[];
  const h=harness(async(url,options)=>{
    calls.push([url,options.method]);
    if(url.endsWith('/status'))return reply({head_crop:{xyxy:[.05,0,.65,.65]}});
    if(options.method==='POST'){posted.resolve();return opened.promise;}
    return reply({status:'stopped'});
  });
  h.run('media={getTracks:()=>[{stop(){}}]}');
  const start=h.element('start').onclick();await posted.promise;
  await h.run('stop()');opened.resolve(reply({session_id:'late'}));await start;
  assert.ok(calls.some(([url,method])=>url.endsWith('/late')&&method==='DELETE'));
  assert.equal(h.run('running'),false);assert.equal(h.run('session'),null);
  assert.equal(h.element('preview').disabled,false);
});

test('display expires at result age, even after a recent successful reply',()=>{
  const h=harness(()=>{});
  h.run("running=true;display({status:'ready',processing_ms:1,feature_slots:100,result_age_ms:1400,result:{tasks:{},gaze:{available:false}}})");
  assert.equal(h.element('cards').children.length,1);
  h.advance(110);
  assert.equal(h.element('cards').children.length,0);
  assert.match(h.element('status').textContent,/失效/);
});

test('cancel pending camera permission stops late tracks',async()=>{
  const permission=deferred();let stopped=0;
  const h=harness(()=>{});h.context.navigator.mediaDevices.getUserMedia=()=>permission.promise;
  const preview=h.element('preview').onclick();await h.run('stop()');
  permission.resolve({getTracks:()=>[{stop(){stopped++;}}]});await preview;
  assert.equal(stopped,1);assert.equal(h.run('media'),null);
  assert.equal(h.element('preview').disabled,false);
});

test('dragging on the preview sets a normalized driver ROI',()=>{
  const h=harness(()=>{});
  h.run('media={}');
  const overlay=h.element('overlay');
  const pointer=(x,y)=>({clientX:x,clientY:y,pointerId:1,preventDefault(){}});
  overlay.dispatch('pointerdown',pointer(64,96));
  overlay.dispatch('pointermove',pointer(320,288));
  overlay.dispatch('pointerup',pointer(320,288));
  assert.equal(h.element('roi').value,'0.1000,0.2000,0.5000,0.6000');
  assert.match(h.element('roiInfo').textContent,/已更新驾驶员区域/);
});

test('dragging is ignored while inference is running',()=>{
  const h=harness(()=>{});
  h.run('media={};running=true');
  const overlay=h.element('overlay');
  const pointer=(x,y)=>({clientX:x,clientY:y,pointerId:1,preventDefault(){}});
  overlay.dispatch('pointerdown',pointer(64,96));
  overlay.dispatch('pointermove',pointer(320,288));
  overlay.dispatch('pointerup',pointer(320,288));
  assert.equal(h.element('roi').value,'0,0,1,1');
});

test('a tiny drag is rejected instead of producing a degenerate ROI',()=>{
  const h=harness(()=>{});
  h.run('media={}');
  const overlay=h.element('overlay');
  const pointer=(x,y)=>({clientX:x,clientY:y,pointerId:1,preventDefault(){}});
  overlay.dispatch('pointerdown',pointer(100,100));
  overlay.dispatch('pointermove',pointer(104,104));
  overlay.dispatch('pointerup',pointer(104,104));
  assert.equal(h.element('roi').value,'0,0,1,1');
  assert.match(h.element('roiInfo').textContent,/过小/);
});
