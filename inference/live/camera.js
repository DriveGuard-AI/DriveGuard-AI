'use strict';
const el=id=>document.getElementById(id);
let media=null,session=null,running=false,generation=0,callback=null,firstPts=null,lastPts=-1,lastReply=0,resultDeadline=0;
let starting=false,stopping=false,previewing=false;
let crop=[.05,0,.65,.65];
const labels={fatigue:'疲劳',phone_use:'手机使用',drinking:'饮水',distraction:'分心'};
const fragment=new URLSearchParams(location.hash.slice(1));
el('token').value=fragment.get('token')||'';
if(location.hash)history.replaceState(null,'',location.pathname);
async function api(path,method='GET',body=null){
  const response=await fetch(path,{method,headers:{Authorization:'Bearer '+el('token').value,'Content-Type':'application/json'},
    body:body===null?undefined:JSON.stringify(body),signal:AbortSignal.timeout(45000)});
  const data=await response.json();if(!response.ok)throw new Error(data.error||'服务请求失败');return data;
}
function clearResults(message){el('cards').replaceChildren();el('details').textContent='';el('status').textContent=message;}
function draw(perception=null){
  const v=el('video'),c=el('overlay');c.width=v.videoWidth;c.height=v.videoHeight;
  const ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);
  const r=el('roi').value.split(',').map(Number);if(r.length!==4||!r.every(Number.isFinite))return;
  const box=[r[0]*c.width,r[1]*c.height,r[2]*c.width,r[3]*c.height];
  function rect(b,color){ctx.strokeStyle=color;ctx.lineWidth=3;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);}
  rect(box,'#73e2a7');
  if(el('view').value==='body'){
    rect(perception?.head_roi||[box[0]+crop[0]*(box[2]-box[0]),box[1]+crop[1]*(box[3]-box[1]),
      box[0]+crop[2]*(box[2]-box[0]),box[1]+crop[3]*(box[3]-box[1])],'#e47fff');
    if(perception?.eye_roi)rect(perception.eye_roi,'#5bd9ff');
  }
}
async function stop(){
  if(stopping)return;
  stopping=true;
  running=false;generation++;
  if(callback!==null&&el('video').cancelVideoFrameCallback)el('video').cancelVideoFrameCallback(callback);
  if(callback!==null&&!el('video').cancelVideoFrameCallback)clearTimeout(callback);
  callback=null;const old=session;session=null;
  media?.getTracks().forEach(track=>track.stop());media=null;el('video').srcObject=null;
  el('start').disabled=true;el('preview').disabled=true;el('stop').disabled=true;
  for(const id of ['view','roi','exclusive'])el(id).disabled=false;
  clearResults('已停止，无当前推理结果');
  if(old){try{await api('/api/live/sessions/'+old,'DELETE');}catch(error){el('error').textContent='服务端会话关闭未确认：'+error.message;}}
  stopping=false;el('preview').disabled=starting||previewing;
}
el('preview').onclick=async()=>{
  if(previewing||starting||stopping)return;
  previewing=true;const epoch=++generation;el('preview').disabled=true;el('stop').disabled=false;
  try{
    el('error').textContent='';
    if(!window.isSecureContext||!navigator.mediaDevices)throw new Error('摄像头需 localhost 或 HTTPS；远程服务器请使用 SSH 端口转发。');
    const opened=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},frameRate:{ideal:25}},audio:false});
    if(epoch!==generation){opened.getTracks().forEach(track=>track.stop());return;}
    media=opened;
    el('video').srcObject=media;await el('video').play();draw();
    if(epoch!==generation)return;
    el('start').disabled=false;el('stop').disabled=false;el('preview').disabled=true;
    clearResults('摄像头已打开，请核对视角和驾驶员区域');
    media.getVideoTracks()[0].addEventListener('ended',()=>{stop();el('error').textContent='摄像头已断开';});
  }catch(error){if(epoch===generation){await stop();el('error').textContent=error.message;}}
  finally{previewing=false;el('preview').disabled=!!media||starting||stopping;}
};
el('roi').oninput=()=>draw();el('view').onchange=()=>draw();el('stop').onclick=stop;
function display(data){
  lastReply=performance.now();draw(data.perception);el('details').textContent=JSON.stringify(data,null,2);
  resultDeadline=data.result?lastReply+Math.max(0,1500-data.result_age_ms):0;
  el('cards').replaceChildren();
  const text={warming:'预热中',ready:'推理结果',stale:'结果过期，请检查帧率和服务耗时'};
  el('status').textContent=`${text[data.status]||data.status} · 服务处理 ${data.processing_ms.toFixed(0)} ms · 特征槽 ${data.feature_slots}`;
  if(!data.result)return;
  for(const [name,value] of Object.entries(data.result.tasks)){
    const card=document.createElement('div');card.className='card';
    card.textContent=`${labels[name]}\n${!value.available?'不可用':value.detected?'检出':'未检出'}${value.score===null?'':` · 分数 ${value.score.toFixed(3)}`}\n${value.source}`;
    el('cards').append(card);
  }
  const gaze=document.createElement('div');gaze.className='card';const g=data.result.gaze;
  gaze.textContent='注视方向\n'+(g.available?g.class_name:'不可用')+(g.validation_status==='unvalidated_body_view'?'\nbody 视角未验证':'');el('cards').append(gaze);
}
el('start').onclick=async()=>{
  if(starting||stopping||running||!media)return;
  starting=true;const epoch=++generation;el('start').disabled=true;
  const options={view:el('view').value,image_size:[el('video').videoWidth,el('video').videoHeight],
    driver_roi:el('roi').value.split(',').map(Number),driver_region_confirmed_exclusive:el('exclusive').checked};
  for(const id of ['view','roi','exclusive'])el(id).disabled=true;
  try{
    const status=await api('/api/live/status');if(epoch!==generation)return;
    crop=status.head_crop.xyxy;draw();
    const opened=await api('/api/live/sessions','POST',options);
    if(epoch!==generation){await api('/api/live/sessions/'+opened.session_id,'DELETE');return;}
    session=opened.session_id;running=true;
    firstPts=null;lastPts=-1;lastReply=performance.now();resultDeadline=0;
    const canvas=document.createElement('canvas');[canvas.width,canvas.height]=options.image_size;
    const ctx=canvas.getContext('2d');
    clearResults('模型预热中');
    async function frame(now,metadata){
      if(!running||epoch!==generation)return;
      const pts=metadata?Math.ceil(metadata.mediaTime*1000):Math.ceil(performance.now());
      if(firstPts===null)firstPts=pts;const timestamp=pts-firstPts;
      if(timestamp>lastPts){
        lastPts=timestamp;ctx.drawImage(el('video'),0,0,canvas.width,canvas.height);
        try{
          const jpeg=canvas.toDataURL('image/jpeg',.85).split(',')[1];
          const sent=performance.now();
          const data=await api('/api/live/sessions/'+session+'/frames','POST',{timestamp_ms:timestamp,jpeg});
          if(data.result){
            data.result_age_ms=Math.max(data.result_age_ms,timestamp-data.result.window_end_ms+performance.now()-sent);
            if(data.result_age_ms>1500){data.result=null;data.status='stale';}
          }
          if(!running||epoch!==generation)return;display(data);
        }catch(error){if(epoch===generation){await stop();el('error').textContent=error.message;}return;}
      }
      if(running&&epoch===generation)schedule();
    }
    function schedule(){
      if(el('video').requestVideoFrameCallback)callback=el('video').requestVideoFrameCallback(frame);
      else callback=setTimeout(()=>frame(performance.now(),null),40);
    }
    schedule();
  }catch(error){if(epoch===generation){await stop();el('error').textContent=error.message;}else el('error').textContent=error.message;}
  finally{starting=false;el('preview').disabled=!!media||stopping||previewing;}
};
setInterval(()=>{if(running&&(performance.now()-lastReply>1500||(resultDeadline&&performance.now()>resultDeadline)))clearResults('等待新结果；旧结果已失效');},100);
document.addEventListener('visibilitychange',()=>{if(document.hidden&&(running||media||starting||previewing))stop();});
window.addEventListener('pagehide',()=>{media?.getTracks().forEach(track=>track.stop());});
