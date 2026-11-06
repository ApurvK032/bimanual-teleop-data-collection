"use strict";

// Data-collection add-on for the teleop repo's Quest page (served only by bimanual_collect/quest_page.py).
// - Recording panel on the page: task text, Start/Stop episode, live status and warnings.
// - Right controller A starts/stops an episode in the headset. Both grips must be released (the grips
//   are the arms' deadman), the same rule the page uses for B (camera view cycling).
// - After A stops an episode, a 6 s prompt in the headset takes A = success, B = failure.
// - Camera panel header: a red "REC m:ss" badge while recording, plus a second line with the success prompt,
//   an arm alert from the collection app (stopping, safety stop), the first warning (camera lost, disk low,
//   arms running but not recording), the arm state ("ARMS LIVE 1:12 left") or the task.
(function collectRecording() {
  const RECORD_URL = `${location.protocol}//${location.hostname}:8444/record`;
  const QUEST_A_BUTTON = 4;
  const QUEST_B_BUTTON = 5;
  const QUEST_X_BUTTON = 4;
  const QUEST_TOUCH_PROFILES = new Set(["meta-quest-touch-plus", "meta-quest-touch-pro", "oculus-touch-v3",
    "oculus-touch-v2", "oculus-touch"]);
  const GRIP_BUTTON = 1;
  const GRIP_RELEASED = 0.2;
  const PROMPT_MS = 6000;
  const state = {
    online: false,
    recording: false,
    busy: false,
    episode: null,
    lastEpisode: null,
    defaults: {task: "", operator: ""},
    warnings: [],
    error: "",
    polledAt: 0,
    prompt: null, // {until, answered}
    arms: {line: "", level: "none"}, // from the collection app, through this page's server (/collect/arms)
  };
  let previousAPressed = false;
  let previousBPressed = false;
  let ui = null;

  function buttonValue(gamepad, index) {
    const button = gamepad?.buttons?.[index];
    return button ? Number(button.value || (button.pressed ? 1 : 0)) : 0;
  }

  window.collectHomeButton = ({left, right}) => {
    const leftGamepad = left?.gamepad;
    if (!left?.profiles?.some((profile) => QUEST_TOUCH_PROFILES.has(profile)) ||
        leftGamepad?.mapping !== "xr-standard" || !leftGamepad.buttons[QUEST_X_BUTTON] ||
        !leftGamepad.buttons[GRIP_BUTTON] ||
        !right?.gamepad?.buttons?.[GRIP_BUTTON]) return undefined;
    return buttonValue(leftGamepad, QUEST_X_BUTTON) >= 0.7;
  };

  function elapsedSeconds() {
    if (!state.recording || !state.episode) return 0;
    return state.episode.elapsed_s + (performance.now() - state.polledAt) / 1000;
  }

  function clock(seconds) {
    const whole = Math.max(0, Math.floor(seconds));
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
  }

  function apply(status) {
    if (!status || typeof status !== "object" || !("recording" in status)) return;
    state.online = true;
    state.recording = Boolean(status.recording);
    state.episode = status.episode || null;
    state.lastEpisode = status.last_episode || null;
    state.defaults = status.defaults || state.defaults;
    state.warnings = status.warnings || [];
    state.polledAt = performance.now();
  }

  async function request(path, method = "GET", body = undefined) {
    const response = await fetch(`${RECORD_URL}${path}`, {
      method,
      cache: "no-store",
      headers: body ? {"Content-Type": "application/json"} : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    apply(payload);
    state.error = "";
    return payload;
  }

  async function poll() {
    try {
      await request("/status");
    } catch (_) {
      state.online = false;
    }
    try {
      const response = await fetch("/collect/arms", {cache: "no-store"});
      if (response.ok) state.arms = await response.json();
    } catch (_) {
      state.arms = {line: "", level: "none"};
    }
    render();
  }

  async function toggle() {
    if (state.busy) return;
    state.busy = true;
    render();
    try {
      const task = ui?.task.value.trim() || "";
      await request("/toggle", "POST", {task});
    } catch (error) {
      state.error = String(error.message || error);
    } finally {
      state.busy = false;
      render();
    }
  }

  async function answerPrompt(outcome) {
    state.prompt = {...state.prompt, answered: outcome, until: performance.now() + 1500};
    try {
      await fetch(`${RECORD_URL}/label`, {
        method: "POST",
        cache: "no-store",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({outcome}),
      });
    } catch (error) {
      state.error = String(error.message || error);
    }
    render();
  }

  function promptActive() {
    if (state.prompt && performance.now() > state.prompt.until) state.prompt = null;
    return Boolean(state.prompt);
  }

  function render() {
    if (!ui) return;
    ui.button.disabled = state.busy || !state.online;
    ui.button.textContent = state.recording ? "Stop episode" : "Start episode";
    ui.button.classList.toggle("secondary", !state.recording);
    ui.task.disabled = state.recording;
    ui.task.placeholder = state.defaults.task ? `Task: ${state.defaults.task}` : "Task (e.g. fold towel in half)";
    let text;
    if (!state.online) {
      text = "Recorder offline: open the collection app on the PC (it starts the recorder)";
    } else if (state.recording && state.episode) {
      text = `● REC episode ${state.episode.index} · ${clock(elapsedSeconds())} · ${state.episode.frames} frames` +
        (state.episode.task ? ` · ${state.episode.task}` : "");
    } else if (state.lastEpisode) {
      const last = state.lastEpisode;
      const outcome = last.labels?.outcome ? `, ${last.labels.outcome}` : "";
      const status = last.status && last.status !== "complete" ? ` (${last.status}: ${last.stop_reason})` : "";
      text = `Ready. Last: episode ${last.episode_index}, ${last.duration_s} s, ${last.frames_recorded} frames` +
        (last.camera_frames_missing ? `, ${last.camera_frames_missing} missing` : "") + outcome + status;
    } else {
      text = "Ready. Right controller A (grips released) or this button starts an episode.";
    }
    if (state.arms.line) text += ` · ${state.arms.line}`;
    if (state.warnings.length) text += ` · ⚠ ${state.warnings.join(" · ")}`;
    if (state.error) text += ` · ${state.error}`;
    ui.status.textContent = text;
    ui.status.classList.toggle("ok", state.online && !state.error && !state.warnings.length);
  }

  function buildPanel() {
    const anchor = document.getElementById("camera-view-mode");
    if (!anchor) return;
    const panel = document.createElement("div");
    panel.className = "camera-setup-actions";
    panel.style.marginTop = "14px";
    panel.innerHTML = `
      <input id="collect-task" type="text" placeholder="Task (e.g. fold towel in half)"
        style="flex:1; min-width:12rem; padding:10px 12px; border-radius:10px; border:1px solid #38536a;
        background:#0b1721; color:#e7eef8; font:inherit;">
      <button id="collect-record" class="secondary compact" type="button">Start episode</button>`;
    const status = document.createElement("p");
    status.className = "camera-status";
    anchor.after(panel, status);
    ui = {task: panel.querySelector("#collect-task"), button: panel.querySelector("#collect-record"), status};
    ui.button.addEventListener("click", toggle);
    render();
  }

  // Called by the page every frame with both controllers. Returns true while the success prompt is using
  // the buttons, so the page doesn't also cycle the camera view on B.
  window.collectHandleRecordButton = (inputSources) => {
    const right = inputSources?.right;
    const aPressed = buttonValue(right?.gamepad, QUEST_A_BUTTON) >= 0.7;
    const bPressed = buttonValue(right?.gamepad, QUEST_B_BUTTON) >= 0.7;
    const aRising = aPressed && !previousAPressed;
    const bRising = bPressed && !previousBPressed;
    previousAPressed = aPressed;
    previousBPressed = bPressed;
    const actuator = right?.gamepad?.hapticActuators?.[0];
    if (promptActive()) {
      if (!state.prompt.answered && (aRising || bRising)) {
        answerPrompt(aRising ? "success" : "failure");
        if (actuator) actuator.pulse(aRising ? 0.7 : 0.35, aRising ? 120 : 60);
      }
      return true;
    }
    if (!aRising) return false;
    const gripsReleased = Object.values(inputSources)
      .filter(Boolean)
      .every((source) => buttonValue(source.gamepad, GRIP_BUTTON) <= GRIP_RELEASED);
    if (!gripsReleased) {
      if (actuator) actuator.pulse(0.25, 70);
      return false;
    }
    const wasRecording = state.recording;
    toggle().then(() => {
      // Long buzz = recording started, short buzz = stopped.
      if (actuator) actuator.pulse(state.recording ? 0.8 : 0.4, state.recording ? 160 : 50);
      if (wasRecording && !state.recording) state.prompt = {until: performance.now() + PROMPT_MS, answered: null};
    });
    return false;
  };

  function secondLine(context, uiWidth, text, colour) {
    context.font = "700 19px system-ui, sans-serif";
    const width = Math.min(uiWidth - 40, context.measureText(text).width + 24);
    const x = (uiWidth - width) / 2;
    context.fillStyle = "rgba(4, 10, 17, .82)";
    context.fillRect(x, 48, width, 28);
    context.fillStyle = colour;
    context.textAlign = "center";
    context.fillText(text, uiWidth / 2, 69, uiWidth - 52);
    context.textAlign = "start";
  }

  window.collectDrawRecordBadge = (context, uiWidth) => {
    if (state.recording) {
      const x = uiWidth / 2 - 62;
      context.beginPath();
      context.arc(x, 29, 9, 0, Math.PI * 2);
      context.fillStyle = "#ff3b3b";
      context.fill();
      context.font = "800 22px system-ui, sans-serif";
      context.fillStyle = "#ffd6d6";
      context.fillText(`REC ${clock(elapsedSeconds())}`, x + 16, 37);
    }
    if (promptActive()) {
      const answered = state.prompt.answered;
      const left = Math.max(0, Math.ceil((state.prompt.until - performance.now()) / 1000));
      secondLine(context, uiWidth, answered ? `Saved: ${answered}` : `Success?  A = yes · B = no  (${left})`,
        answered ? "#baf7d1" : "#ffd49a");
    } else if (state.arms.level === "bad") {
      secondLine(context, uiWidth, state.arms.line, "#ffb4b4");
    } else if (state.arms.homing) {
      secondLine(context, uiWidth, state.arms.line, state.arms.level === "ok" ? "#baf7d1" : "#ffd49a");
    } else if (state.warnings.length) {
      secondLine(context, uiWidth, `⚠ ${state.warnings[0]}`, "#ffd49a");
    } else if (state.arms.line && !state.recording) {
      secondLine(context, uiWidth, state.arms.line, state.arms.level === "ok" ? "#baf7d1" : "#ffd49a");
    } else if (state.recording && state.episode?.task) {
      secondLine(context, uiWidth, state.episode.task, "#dbe7f3");
    }
  };

  window.collectRecording = state;
  // For tests and debugging: the page's camera view (a top-level const in the repo's client.js).
  window.collectCameraView = () => (typeof cameraView === "undefined" ? null : cameraView);
  buildPanel();
  poll();
  setInterval(poll, 1000);
})();
