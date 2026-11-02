"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const rows = [];
  const keyLabels = [
    ["A", "A"], ["B", "B"], ["X", "X"], ["Y", "Y"],
    ["LB", "LB"], ["RB", "RB"], ["DPadLeft", "十字左"],
    ["DPadRight", "十字右"], ["DPadUp", "十字上"], ["DPadDown", "十字下"],
    ["LeftStick", "L3"], ["RightStick", "R3"], ["Start", "Start"], ["Back", "Back"]
  ];
  const keyElements = new Map();
  const gateLabels = {
    ready: "已就绪 · 等待 LB", active: "正在输出预览意图", paused: "预览已暂停",
    disconnected: "手柄未连接", neutral_required: "请先回到中立", input_stale: "手柄输入已过期"
  };
  let current = null;
  let lastReceived = -Infinity;
  let stale = true;
  let actionBusy = false;
  let queuedAction = null;
  let eventSignature = "";
  let stopped = false;
  let polling = false;
  let lifecycleGeneration = 0;

  const finite = (n, fallback = 0) => Number.isFinite(Number(n)) ? Number(n) : fallback;
  const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, finite(n)));
  const signed = (n) => { const v = Math.abs(finite(n)) < .0005 ? 0 : finite(n); return `${v > 0 ? "+" : ""}${v.toFixed(3)}`; };

  for (let i = 0; i < 6; i++) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>J${i + 1}</td><td><div class="speed-cell"><div class="speed-track" aria-hidden="true"><span class="speed-fill"></span></div><output class="number speed-number">0.000</output></div></td><td><output class="number displacement-number">0.000</output></td>`;
    $("joint-rows").append(row);
    rows.push({ row, speed: row.querySelector(".speed-number"), fill: row.querySelector(".speed-fill"), displacement: row.querySelector(".displacement-number") });
  }
  for (const [name, label] of keyLabels) {
    const el = document.createElement("span");
    el.className = "button-state";
    el.textContent = label;
    el.setAttribute("aria-label", `${label} 未按下`);
    $("buttons-readout").append(el);
    keyElements.set(name, el);
  }

  function updateButtons() {
    const fresh = !stale && current !== null;
    $("pause-button").disabled = !fresh || actionBusy;
    $("reset-button").disabled = !fresh || actionBusy;
    $("pause-button").textContent = actionBusy ? "正在处理…" : current?.control?.paused ? "继续预览" : "暂停预览";
  }

  function renderAxes(axes) {
    const x1 = clamp(axes.LX, -1, 1), y1 = clamp(axes.LY, -1, 1);
    const x2 = clamp(axes.RX, -1, 1), y2 = clamp(axes.RY, -1, 1);
    $("left-stick-dot").setAttribute("cx", String(60 + x1 * 45));
    $("left-stick-dot").setAttribute("cy", String(60 - y1 * 45));
    $("right-stick-dot").setAttribute("cx", String(60 + x2 * 45));
    $("right-stick-dot").setAttribute("cy", String(60 - y2 * 45));
    $("left-axis-values").textContent = `X ${x1.toFixed(3)} · Y ${y1.toFixed(3)}`;
    $("right-axis-values").textContent = `X ${x2.toFixed(3)} · Y ${y2.toFixed(3)}`;
    for (const name of ["LT", "RT"]) {
      const value = clamp(axes[name], 0, 1);
      $(name.toLowerCase() + "-value").value = value;
      $(name.toLowerCase() + "-text").textContent = value.toFixed(3);
    }
  }

  function clearMotionVisuals() {
    for (const row of rows) { row.speed.textContent = "0.000"; row.fill.style.width = "0%"; row.fill.style.left = "50%"; }
    renderAxes({});
    for (const [name, el] of keyElements) { el.classList.remove("pressed"); el.setAttribute("aria-label", `${name} 数据不可用`); }
    $("permission-text").textContent = "数据不可用";
    $("permission-text").classList.remove("held");
    $("direction-text").textContent = "显示已过期 · 意图归零";
    $("gripper-intent").textContent = "无有效意图";
  }

  function markStale(message = "未取得新数据，显示已过期。正在重试连接本机服务。") {
    stale = true;
    document.body.classList.add("is-stale");
    $("gate-title").textContent = "预览服务未连接 · 显示已过期";
    $("gate-reason").textContent = message;
    $("gate-dot").className = "status-dot warning";
    $("input-connection").textContent = "数据已过期";
    $("input-connection").className = "connection-indicator";
    $("input-age").textContent = "无有效实时数据";
    $("displacement-title").textContent = "历史虚拟位移（已过期）rad";
    clearMotionVisuals();
    updateButtons();
  }

  function renderEvents(events) {
    const recent = (Array.isArray(events) ? events : []).slice(-5).reverse();
    const signature = JSON.stringify(recent);
    if (signature === eventSignature) return;
    eventSignature = signature;
    $("event-list").replaceChildren();
    if (!recent.length) {
      const el = document.createElement("li"); el.className = "empty-event"; el.textContent = "尚无状态变化。"; $("event-list").append(el); return;
    }
    for (const event of recent) {
      const li = document.createElement("li");
      const time = document.createElement("time");
      const seconds = Math.max(0, finite(event.time));
      time.textContent = `${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
      const message = document.createElement("span"); message.textContent = String(event.message ?? event.kind ?? "状态变化");
      li.append(time, message); $("event-list").append(li);
    }
  }

  function render(state) {
    if (state.mode !== "offline_fake_backend" || state.can_opened !== false || typeof state.session_id !== "string" || !state.session_id || !state.input || !state.control || !state.fake_backend || !Array.isArray(state.control.joint_velocity_rad_s) || state.control.joint_velocity_rad_s.length !== 6) {
      throw new Error("预览数据模式不匹配，已停止显示运动意图。");
    }
    current = state;
    lastReceived = performance.now();
    stale = false;
    document.body.classList.remove("is-stale");
    const input = state.input, control = state.control;
    const selected = Math.round(clamp(control.selected_joint, 1, 6));
    const maxSpeed = Math.max(.001, finite(control.max_speed_rad_s, .08));
    $("gate-title").textContent = gateLabels[control.gate] || "预览状态更新";
    $("gate-reason").textContent = String(control.reason || "按住 LB 并拨动左摇杆，查看所选关节的运动意图。");
    const className = control.gate === "active" ? "active" : control.gate === "ready" ? "ready" : "warning";
    $("gate-dot").className = `status-dot ${className}`;
    $("selected-joint").textContent = `J${selected}`;
    $("speed-limit").textContent = maxSpeed.toFixed(3);
    $("displacement-title").innerHTML = "虚拟累计位移 <span>rad</span>";
    const activeVelocity = finite(control.joint_velocity_rad_s[selected - 1]);
    $("direction-text").textContent = activeVelocity > .00001 ? "原生角增加 +" : activeVelocity < -.00001 ? "原生角减小 −" : "无关节运动意图";
    for (let i = 0; i < rows.length; i++) {
      const velocity = finite(control.joint_velocity_rad_s[i]);
      rows[i].row.classList.toggle("selected", i === selected - 1);
      rows[i].row.setAttribute("aria-label", `J${i + 1}${i === selected - 1 ? " 当前选择" : ""}`);
      rows[i].speed.textContent = signed(velocity);
      const fraction = clamp(velocity / maxSpeed, -1, 1);
      rows[i].fill.style.left = `${fraction < 0 ? 50 + fraction * 50 : 50}%`;
      rows[i].fill.style.width = `${Math.abs(fraction) * 50}%`;
      rows[i].displacement.textContent = signed(state.fake_backend.displacement_rad?.[i]);
    }
    $("input-connection").textContent = input.connected ? "手柄已连接" : "手柄未连接";
    $("input-connection").className = `connection-indicator${input.connected ? " connected" : ""}`;
    $("slot-text").textContent = `XInput · 槽位 ${Math.round(finite(input.slot))}`;
    $("input-age").textContent = input.connected ? `输入 ${Math.max(0, finite(input.age_ms)).toFixed(0)} ms 前` : "等待接入手柄";
    renderAxes(input.connected ? input.axes || {} : {});
    const pressed = new Set(input.connected && Array.isArray(input.buttons) ? input.buttons : []);
    for (const [name, el] of keyElements) { const down = pressed.has(name); el.classList.toggle("pressed", down); el.setAttribute("aria-label", `${name} ${down ? "已按下" : "未按下"}`); }
    $("permission-text").textContent = control.lb_held ? "已按住" : "未按住";
    $("permission-text").classList.toggle("held", Boolean(control.lb_held));
    $("gripper-intent").textContent = ({open: "张开", close: "闭合", hold: "保持"})[control.gripper_intent] || "保持";
    $("sample-count").textContent = `采样 ${Math.max(0, finite(state.stats?.samples)).toLocaleString("zh-CN")} 次`;
    $("change-count").textContent = `状态变化 ${Math.max(0, finite(state.stats?.state_changes)).toLocaleString("zh-CN")} 次`;
    renderEvents(state.events);
    updateButtons();
  }

  async function api(path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 800);
    try {
      const response = await fetch(path, {cache: "no-store", ...options, signal: controller.signal});
      let body;
      try { body = await response.json(); } catch { throw new Error("预览服务返回了无效数据。"); }
      if (!response.ok) throw new Error(typeof body.error === "string" ? body.error : `请求未成功（${response.status}）。`);
      return body;
    } finally { clearTimeout(timeout); }
  }

  function queueAction(action) {
    if (stale || !current || actionBusy) return;
    queuedAction = {session_id: current.session_id, action};
    actionBusy = true;
    $("action-error").hidden = true;
    updateButtons();
  }
  $("pause-button").addEventListener("click", () => queueAction(current?.control?.paused ? "resume" : "pause"));
  $("reset-button").addEventListener("click", () => queueAction("reset"));

  async function poll() {
    if (polling) return;
    polling = true;
    try { while (!stopped) {
      const generation = lifecycleGeneration;
      if (queuedAction) {
        const action = queuedAction; queuedAction = null;
        try {
          await api("/api/action", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(action)});
        } catch (error) {
          if (generation === lifecycleGeneration && !stopped) {
            $("action-error").textContent = `操作未完成：${error.message}`;
            $("action-error").hidden = false;
          }
        } finally { actionBusy = false; updateButtons(); }
      }
      if (stopped || generation !== lifecycleGeneration) continue;
      try {
        const state = await api("/api/state");
        if (!stopped && generation === lifecycleGeneration) render(state);
      }
      catch (error) {
        if (!stopped && generation === lifecycleGeneration) {
          markStale(error.name === "AbortError" ? "本机服务响应超时，显示已过期。正在重试。" : error.message);
        }
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    } } finally {
      polling = false;
      // BFCache restoration may occur while the previous request is settling.
      if (!stopped) poll();
    }
  }
  setInterval(() => { if (!stale && performance.now() - lastReceived > 600) markStale(); }, 100);
  window.addEventListener("pagehide", () => {
    stopped = true;
    lifecycleGeneration++;
    queuedAction = null;
    actionBusy = false;
    markStale("页面已离开，显示已过期；返回后将重新读取本机服务。");
  });
  window.addEventListener("pageshow", (event) => {
    if (!event.persisted) return;
    stopped = false;
    lifecycleGeneration++;
    markStale("页面已恢复，正在重新读取本机预览服务。");
    poll();
  });
  markStale("正在连接本机预览服务，尚无有效实时数据。");
  poll();
})();
