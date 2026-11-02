'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const names = ['底座', '肩部', '肘部', '腕部', '腕部', '腕部', '夹爪'];
  $('jointRows').innerHTML = names.map((name, i) => `<tr id="row${i}"><td>J${i+1}<small>${name}</small></td><td class="actual" id="q${i}">—</td><td class="target" id="t${i}">—</td></tr>`).join('');
  let renderer, scene, camera, orbit, robot, ghost, grid, frame, mapping;
  let latest = null, frozen = false, havePose = false, poseSession = null, lastStamp = null;
  let drawn = null, from = null, to = null, transitionAt = 0, transitionDuration = 50;
  let lastSampleAt = 0, lastResponseAt = 0, assetFailed = false;
  const poseState = new TwinPoseState();
  const reduction = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const radiansToDegrees = x => x * 180 / Math.PI;
  const validVector = x => Array.isArray(x) && x.length === 7 && x.every(Number.isFinite);
  const deg = x => Number.isFinite(x) ? (Math.abs(x) < .005 ? 0 : x).toFixed(2) : '—';
  const phases = {ready:'遥操就绪',paused:'保持 / 待恢复',returning:'返回本次起点',at_origin:'已完成返回',disabled:'会话已结束',fault:'故障保持',fault_holding:'故障保持',cleanup_fault:'结束异常',external_poweroff_handover:'外部断电接管'};

  function banner(state, text) {
    document.body.dataset.state = state;
    $('statusBadge').dataset.state = state;
    $('statusText').textContent = text;
  }
  function showModelError(text) {
    assetFailed = true;
    $('modelLoad').hidden = false;
    $('modelLoadText').textContent = text;
    $('retryModel').hidden = false;
    $('loadProgress').hidden = true;
  }
  function setJoints(model, q) {
    if (!model || !validVector(q) || !mapping) return;
    for (let i = 0; i < 6; i++) model.joints[`joint${i+1}`]?.setJointValue(q[i]*mapping.arm_signs[i]+mapping.arm_offsets_rad[i]);
    const travel = Math.max(0, Math.min(1, q[6]/mapping.gripper_full_motor_rad))*mapping.finger_travel_m;
    model.joints.gripper_joint1?.setJointValue(travel);
    model.joints.gripper_joint2?.setJointValue(travel);
  }
  function view(kind, fit = false) {
    if (!camera || !orbit) return;
    if (fit && robot) {
      robot.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(robot);
      const center = box.getCenter(new THREE.Vector3());
      orbit.target.copy(center);
    }
    const directions = {iso:[.72,.48,.9],front:[0,.12,1.15],side:[1.15,.12,0],top:[0,1.25,.001]};
    const d = new THREE.Vector3(...directions[kind]);
    camera.position.copy(orbit.target).add(d);
    orbit.update();
    document.querySelectorAll('.view-presets button').forEach(b => b.classList.remove('selected'));
    $({iso:'viewIso',front:'viewFront',side:'viewSide',top:'viewTop'}[kind]).classList.add('selected');
  }
  async function initModel() {
    try {
      if (!window.THREE || !window.URDFLoader || !THREE.OrbitControls) throw new Error('本地渲染组件未加载，请重新打开页面。');
      const configResponse = await fetch('/model_mapping.json');
      if (!configResponse.ok) throw new Error('显示换算配置读取失败');
      mapping = await configResponse.json();
      scene = new THREE.Scene(); scene.background = new THREE.Color('#121b18');
      camera = new THREE.PerspectiveCamera(38,1,.01,30);
      renderer = new THREE.WebGLRenderer({antialias:true,alpha:false,powerPreference:'low-power'});
      renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
      renderer.outputEncoding = THREE.sRGBEncoding;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.0;
      $('scene').appendChild(renderer.domElement);
      renderer.domElement.addEventListener('webglcontextlost', e => {e.preventDefault();showModelError('图形上下文已丢失，请重新加载模型。');});
      orbit = new THREE.OrbitControls(camera,renderer.domElement);
      orbit.enableDamping = true; orbit.dampingFactor = .09; orbit.minDistance = .3; orbit.maxDistance = 5;
      orbit.target.set(0,.34,0); orbit.maxPolarAngle = Math.PI*.92;
      scene.add(new THREE.HemisphereLight(0xf2fff2,0x667765,.85));
      for (const [color,intensity,position] of [[0xffffff,1.0,[2,3,3]],[0xffffff,.45,[-2,2,-1]],[0xdce9c6,.4,[1,1,-3]]]) {
        const light = new THREE.DirectionalLight(color,intensity);light.position.set(...position);scene.add(light);
      }
      grid = new THREE.GridHelper(2.4,24,0x526b59,0x2b3a31);grid.position.y=-.002;scene.add(grid);
      frame = new THREE.Group();frame.rotation.x=-Math.PI/2;scene.add(frame);
      const axes = new THREE.AxesHelper(.15);frame.add(axes);
      const manager = new THREE.LoadingManager();
      manager.onProgress = (_,loaded,total) => {if(assetFailed)return;const p = Math.round(100*loaded/Math.max(1,total));$('loadProgress').value=p;$('modelLoadText').textContent=`正在加载官方模型 · ${p}%`;};
      manager.onError = url => showModelError('模型资源读取失败：'+url.split('/').pop());
      manager.onLoad = () => {
        if (!robot || assetFailed) return;
        robot.traverse(obj => {
          if (obj.isURDFJoint) obj.ignoreLimits = true;
          if (obj.isMesh) {
            const material = obj.material.clone();
            if ('shininess' in material) material.shininess = 50;
            obj.material = material;
          }
        });
        ghost = robot.clone(true);
        ghost.traverse(obj => {
          if (obj.isURDFJoint) obj.ignoreLimits = true;
          if (obj.isMesh) {obj.material = new THREE.MeshBasicMaterial({color:0x83cbd9,transparent:true,opacity:.18,depthWrite:false});obj.renderOrder=2;}
        });
        frame.add(ghost); ghost.visible = false;
        if (poseState.record) acceptPose(poseState.record,true);
        else setJoints(robot,[0,0,0,0,0,0,0]);
        view('iso',true);
        $('modelLoad').hidden = true;
        $('sceneNote').textContent='官方 RS 模型 · 实测驱动';
        window.twinReady = true;
      };
      const loader = new URDFLoader(manager);
      loader.packages = {rebotarm_bringup: new URL('/vendor',location.href).href};
      loader.parseCollision = false;
      loader.load('/vendor/description/urdf/ReBot_Arm_RS.urdf',r => {robot=r;frame.add(robot);},undefined,e => showModelError('模型加载失败：'+e.message));
      const resize = () => {const {width,height}=$('scene').getBoundingClientRect();renderer.setSize(width,height,false);camera.aspect=width/Math.max(1,height);camera.updateProjectionMatrix();};
      new ResizeObserver(resize).observe($('scene'));resize();view('iso');
      $('scene').addEventListener('keydown', event => {
        const arrows=['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'];
        if (!arrows.includes(event.key) && !['+','=','-','_','Home'].includes(event.key)) return;
        event.preventDefault();
        const offset=camera.position.clone().sub(orbit.target);
        if (event.key==='Home') {view('iso',true);return;}
        if (arrows.includes(event.key) && event.shiftKey) {
          const direction=new THREE.Vector3();
          const horizontal=event.key==='ArrowLeft'||event.key==='ArrowRight';
          direction.setFromMatrixColumn(camera.matrix,horizontal?0:1);
          direction.multiplyScalar(offset.length()*.035*(event.key==='ArrowLeft'||event.key==='ArrowDown'?-1:1));
          orbit.target.add(direction);camera.position.add(direction);
        } else if (arrows.includes(event.key)) {
          const spherical=new THREE.Spherical().setFromVector3(offset);
          spherical.theta+=(event.key==='ArrowLeft'?.10:event.key==='ArrowRight'?-.10:0);
          spherical.phi=Math.max(.03,Math.min(Math.PI*.92,spherical.phi+(event.key==='ArrowUp'?-.08:event.key==='ArrowDown'?.08:0)));
          camera.position.copy(orbit.target).add(new THREE.Vector3().setFromSpherical(spherical));
        } else {
          const distance=Math.max(orbit.minDistance,Math.min(orbit.maxDistance,offset.length()*(event.key==='-'||event.key==='_'?1.10:.90)));
          camera.position.copy(orbit.target).add(offset.setLength(distance));
        }
        orbit.update();
      });
      requestAnimationFrame(animate);
    } catch (error) {showModelError(error.message);}
  }
  function acceptPose(data, force = false) {
    if (!validVector(data.positions) || frozen) return;
    const isNew = data.session_id !== poseSession;
    const now = performance.now();
    if (force || isNew || !drawn || !data.live) {
      drawn=[...data.positions];from=[...drawn];to=[...drawn];
    } else {
      from=[...drawn];to=[...data.positions];
      transitionDuration = Math.max(20,Math.min(90,now-lastSampleAt || 50));
    }
    transitionAt=now;lastSampleAt=now;poseSession=data.session_id;havePose=true;
    if (robot) setJoints(robot,drawn);
    if (ghost && validVector(data.targets)) setJoints(ghost,data.targets);
  }
  function animate(now) {
    requestAnimationFrame(animate);
    if (!renderer) return;
    if (!frozen && latest?.live && lastResponseAt && now-lastResponseAt < 800 && from && to) {
      const t=reduction ? 1 : Math.max(0,Math.min(1,(now-transitionAt)/transitionDuration));
      drawn=from.map((x,i)=>x+(to[i]-x)*t);setJoints(robot,drawn);
    }
    if (ghost) {
      const show=$('showTarget').checked && latest?.live && !frozen && validVector(latest.targets) && validVector(latest.positions);
      const error=show ? Math.max(...latest.targets.map((x,i)=>Math.abs(x-latest.positions[i]))) : 0;
      ghost.visible=show && error>.003;
    }
    orbit.update();renderer.render(scene,camera);
  }
  function updateTelemetry(data) {
    banner(data.state,data.message);
    $('phase').textContent=phases[data.phase] || data.phase || '等待数据';
    $('dataAge').textContent=data.age_ms==null ? '—' : data.age_ms<1000 ? `${data.age_ms} ms 前` : `${(data.age_ms/1000).toFixed(1)} s 前`;
    $('gamepad').textContent=data.gamepad_connected==null ? '—' : data.gamepad_connected ? (data.lb_held ? '已连接 · LB 按住' : '已连接') : '未连接';
    $('speed').textContent=Number.isFinite(data.speed_scale) ? `${Math.round(data.speed_scale*100)}%` : '—';
    $('sendHz').textContent=Number.isFinite(data.send_hz) ? `${data.send_hz.toFixed(1)} Hz` : '—';
    $('temperature').textContent=Number.isFinite(data.temperature_c) ? data.temperature_c>=-40 && data.temperature_c<=200 ? `${data.temperature_c.toFixed(1)} °C` : '回读数值异常' : '—';
    $('sessionId').textContent=data.session_id ? `会话 ${data.session_id.slice(0,12)} · ${data.live?'实测同步':'最后记录'}` : '等待会话';
    const blocks=(data.soft_limit_blocks || []).map((x,i)=>x ? `J${i+1}` : null).filter(Boolean);
    $('softLimit').hidden=!blocks.length;$('softLimit').textContent=blocks.join('、')+' 到达软件余量，仅阻止继续向外点动。';
    $('errorBox').hidden=!data.error;$('errorBox').textContent=data.error || '';
    $('sceneEmpty').hidden=data.live || frozen;
    if (!data.live) $('sceneEmpty').innerHTML=havePose ? '最后一次实测姿态<span>等待新的有效反馈</span>' : '参考零姿态<span>收到实测反馈后自动同步</span>';
    if (!frozen) {
      const shown=poseState.record;
      for(let i=0;i<7;i++) {
        $('q'+i).textContent=validVector(shown?.positions) ? deg(radiansToDegrees(shown.positions[i])) : '—';
        $('t'+i).textContent=validVector(shown?.targets) ? deg(radiansToDegrees(shown.targets[i])) : '—';
        $('row'+i).classList.toggle('blocked',Boolean(data.soft_limit_blocks?.[i]));
      }
      $('poseTime').textContent=data.positions ? data.live ? '显示实测反馈' : '保留最后姿态' : '尚无实测数据';
    }
  }
  async function poll() {
    const started=performance.now();
    try {
      const response=await fetch('/api/state',{cache:'no-store',signal:AbortSignal.timeout(1800)});
      if(!response.ok) throw new Error('状态服务响应异常');
      const data=await response.json();
      latest=data;lastResponseAt=performance.now();
      if (poseState.update(data)) {acceptPose(poseState.record);lastStamp=data.stamp;}
      updateTelemetry(data);
    } catch (error) {
      if(latest) latest={...latest,live:false};
      banner('unavailable','显示服务未连接 · 保留最后姿态');
      $('dataAge').textContent='无法更新';$('sceneEmpty').hidden=false;
      $('sceneEmpty').innerHTML='显示连接已中断<span>正在自动重连，手柄控制独立运行</span>';
    }
    setTimeout(poll,Math.max(15,50-(performance.now()-started)));
  }
  $('retryModel').onclick=()=>location.reload();
  $('viewIso').onclick=()=>view('iso');$('viewFront').onclick=()=>view('front');
  $('viewSide').onclick=()=>view('side');$('viewTop').onclick=()=>view('top');
  $('fitView').onclick=()=>view('iso',true);
  $('showGrid').onchange=()=>{if(grid)grid.visible=$('showGrid').checked;};
  $('showTarget').onchange=()=>{$('targetLegend').hidden=!$('showTarget').checked;};
  $('freezeView').onclick=()=>{
    frozen=!frozen;$('freezeView').textContent=frozen?'继续显示':'暂停画面';$('freezeView').setAttribute('aria-pressed',String(frozen));
    $('displayHint').textContent=frozen?'画面已冻结，实物仍由手柄控制；点击继续显示跟随当前姿态。':'视图操作只影响显示，手柄遥操独立运行。';
    if(frozen){$('poseTime').textContent='画面已暂停';$('sceneEmpty').hidden=true;}
    else if(latest){if(poseState.record)acceptPose(poseState.record,true);updateTelemetry(latest);}
  };
  initModel();poll();
})();
