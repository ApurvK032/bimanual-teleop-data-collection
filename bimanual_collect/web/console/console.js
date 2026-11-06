"use strict";

// Collection app UI. Talks to the app at the same origin (localhost:8440); the app forwards the camera
// service's endpoints (views, recording, episodes, jobs), so every request below is same-origin.
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const OTHER = "__other__";
const WAITING_PHASES = ["waiting_headset", "starting", "checked", "pc_checked", "waiting_controllers", "connected"];
const MOVING_PHASES = ["moving_home", "stopping", "returning"];
const SHUTDOWN_PHASES = ["stopping", "returning", "at_rest"];
const KIND_NAMES = {live: "Live", sim: "Simulation", dry: "Dry run", contact: "Read-only check", guarded: "First-run check"};
const MOTIONS = [
  ["right", "Hand to the right → gripper moves right"],
  ["up", "Hand up → gripper moves up"],
  ["forward", "Hand forward (away from you) → gripper moves forward"],
  ["roll", "Twist the wrist → gripper rolls the same way"],
  ["pitch", "Tilt the hand down → gripper tips down"],
  ["yaw", "Turn the hand left → gripper turns left"],
  ["gripper", "Trigger → gripper closes, release → opens"],
];

const state = {
  tab: "collect",
  app: null,
  appAt: 0,
  offline: false,
  runKind: "live",
  startPending: false,
  stopPending: false,
  checks: {},
  checksAt: {},
  dismissedRun: null,
  // episodes / jobs
  sessions: [], session: null, episodes: [], episode: null, jobs: [], outcomeFor: null, operatorSet: false,
  // bench
  network: null, networkAt: 0, scan: null, scanning: false, identify: null, picture: null, benchCheck: null,
  fixResult: null, openSteps: new Set(), benchRenderedFor: "", motions: {}, validator: "",
  boxes: null, boxName: "sync_roi",
};
let statusRequest = null;
let armRevision = 0;
let previewRequest = null;
let nextPreview = 0;

// ---------------------------------------------------------------- helpers
async function api(path, options = {}) {
  const response = await fetch(path, {
    cache: "no-store",
    headers: options.body ? {"Content-Type": "application/json"} : undefined,
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const text = await response.text();
  let payload = {};
  try { payload = text ? JSON.parse(text) : {}; } catch (_) { payload = {detail: text}; }
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload;
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function toast(message, error = false) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.toggle("error", error);
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (node.hidden = true), error ? 7000 : 2600);
}

function clock(seconds) {
  const whole = Math.max(0, Math.floor(seconds || 0));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function badge(text, kind = text) {
  return el("span", {class: `badge ${String(kind || "neutral").replace(/\s+/g, "_")}`, text: String(text).replace("_", " ")});
}

function tag(text, kind = "") { return el("span", {class: `tag ${kind}`, text}); }
function fileUrl(path) { return `/api/file?path=${encodeURIComponent(path)}`; }
function bench() { return state.app?.bench || null; }
function benchState() { return bench()?.state || {}; }
function currentRun() { return state.app?.arms?.run || null; }
function runActive() { return Boolean(currentRun()?.active); }
function stopInProgress(run) { return run?.stop_sent || SHUTDOWN_PHASES.includes(run?.phase); }
function camera() { return state.app?.camera || null; }

async function act(action, okMessage) {
  try {
    const result = await action();
    if (okMessage) toast(okMessage);
    return result;
  } catch (error) {
    toast(error.message, true);
    return null;
  } finally {
    poll();
  }
}

// ---------------------------------------------------------------- tabs
function showTab(name) {
  state.tab = name;
  $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  if (name !== "collect") clearPreviews();
  if (name === "episodes") loadSessions();
  if (name === "jobs") { loadJobs(); loadExportSessions(); }
  if (name === "bench") {
    loadNetwork();
    renderBench(true);
    if (!state.benchCheck && bench()?.complete) runBenchCheck();  // configuration check: offline, a few seconds
  }
  if (name === "system") loadSystem();
  if (name === "controls" && !state.controlsWired) {
    state.controlsWired = true;
    $("#controls-print").addEventListener("click", () => $("#controls-frame").contentWindow?.print());
  }
  if (name === "collect") loadChecks(true);
}

// ---------------------------------------------------------------- polling
function poll() {
  statusRequest ??= updateStatus().finally(() => { statusRequest = null; });
  return statusRequest;
}

async function updateStatus() {
  const revision = armRevision;
  try {
    const status = await api("/api/app/status", {signal: AbortSignal.timeout(4000)});
    if (revision !== armRevision) return;
    state.app = status;
    state.appAt = performance.now();
    state.offline = false;
  } catch (_) {
    if (revision !== armRevision) return;
    state.offline = true;
  }
  render();
}

function render() {
  renderTop();
  if (!state.app) return;
  renderViews();
  renderArms();
  renderRecord();
  renderQuest();
  if (state.tab === "bench") renderBench(false);
  if (state.tab === "system") renderServices();
}

function recElapsed() {
  const episode = camera()?.episode;
  if (!episode) return 0;
  return episode.elapsed_s + (performance.now() - state.appAt) / 1000;
}

function renderTop() {
  if (state.offline || !state.app) {
    $("#session-name").textContent = "application not reachable";
    $("#pills").replaceChildren(pill("bad", "Application", "offline"));
    return;
  }
  const s = state.app;
  const cam = s.camera;
  const b = s.bench || {};
  const st = b.state || {};
  const armsText = st.arms === 1 ? `1 arm (${st.single_hand})` : "2 arms";
  $("#session-name").textContent = `${b.complete ? armsText : "bench not configured"}` +
    (cam?.session_dir ? ` · ${cam.session_dir.split("/").slice(-2).join("/")}` : "");

  const pills = [];
  const camService = s.services?.camera;
  if (cam) {
    for (const c of cam.cameras_health || []) {
      const ok = c.state === "ok";
      const drops = c.camera_frames_missing || 0;
      const pct = c.frames_captured ? (100 * drops) / (c.frames_captured + drops) : 0;
      pills.push(pill(ok ? (pct > 2 ? "warn" : "ok") : "bad", `Camera ${c.name}`,
        ok ? `${c.mode || ""} ${c.fps_nominal || ""} fps · drops ${pct.toFixed(2)}%` : c.state,
        ok ? `${drops} of ${c.frames_captured + drops} frames dropped by the camera since start` : c.error || ""));
    }
  } else {
    pills.push(pill(camService?.state === "starting" ? "warn" : "bad", "Camera", camService?.state || "off",
      (camService?.log_tail || []).slice(-1)[0] || ""));
  }
  const run = s.arms?.run;
  if (run?.active) {
    const live = run.phase === "live";
    pills.push(pill(live ? "ok" : "warn", `Arms · ${KIND_NAMES[run.kind] || run.kind}`, run.phase_label.split(":")[0].toLowerCase()));
  } else {
    pills.push(pill(b.complete ? "" : "warn", "Arms", b.complete ? "idle" : "not configured"));
  }
  const q = s.quest || {};
  const streaming = q.page?.streaming;
  pills.push(pill(streaming ? "ok" : q.usb === "connected" ? "" : "warn", "Headset",
    streaming ? `${q.page.mode} · ${q.page.rate_hz} Hz` : q.usb === "connected" ? "USB, not streaming" : "not connected"));
  const disk = cam?.disk || {};
  if (disk.free_gb !== undefined && disk.free_gb !== null) {
    const low = disk.minutes_left !== null && disk.minutes_left < 60;
    const hours = disk.minutes_left === null ? "" : ` · ${(disk.minutes_left / 60).toFixed(1)} h`;
    pills.push(pill(low ? "warn" : "ok", "Disk", `${Math.round(disk.free_gb)} GB${hours}`));
  }
  $("#pills").replaceChildren(...pills);

  const warnings = [...(cam?.warnings || [])];
  if (s.services?.quest_page && s.services.quest_page.state !== "running") warnings.push(`Headset page service: ${s.services.quest_page.state}`);
  $("#warnings").hidden = !warnings.length;
  $("#warnings").replaceChildren(...warnings.map((w) => el("div", {}, tag("warn", "warn"), " ", w)));

  const emergency = run?.emergency || (run && !run.active && run.emergency);
  const overdue = run?.stop_overdue && run.motion_started;
  $("#emergency").hidden = !(emergency || overdue);
  if (emergency) $("#emergency").replaceChildren("Cut the arm power now", el("small", {text: run.emergency}));
  else if (overdue) $("#emergency").replaceChildren("Arm launcher not responding to stop",
    el("small", {text: "If the arms are not at rest, cut the arm power at the switch."}));

  $("#rec-indicator").hidden = !cam?.recording;
  $("#rec-time").textContent = clock(recElapsed());
  const stop = $("#global-stop");
  stop.hidden = !run?.active;
  stop.disabled = state.stopPending || Boolean(stopInProgress(run));
  stop.textContent = state.stopPending ? "Sending Stop…" : stopInProgress(run) ? "Stopping…"
    : run?.phase === "waiting_headset" ? "Cancel arms" : "Stop arms";
  $("#bench-attention").hidden = Boolean(b.complete);
  $("#first-run").hidden = Boolean(b.complete);
}

function pill(kind, label, value, title = "") {
  return el("div", {class: `stat ${kind}`, title},
    el("div", {class: "stat-label"}, el("span", {class: "led"}), label), el("div", {class: "stat-value", text: value}));
}

// ---------------------------------------------------------------- camera views
function renderViews() {
  const cam = camera();
  const container = $("#views");
  const missing = $("#no-camera");
  if (!cam) {
    clearPreviews();
    container.replaceChildren();
    const service = state.app.services?.camera || {};
    missing.hidden = false;
    const last = (service.log_tail || []).filter((l) => l.trim()).slice(-3);
    missing.replaceChildren(
      el("div", {class: "panel-title", text: service.state === "starting" ? "Camera starting" : "No camera"}),
      el("p", {text: service.state === "starting" ? "The recorder is opening the camera."
        : "Connect the ZED to a USB 3 port. It is detected automatically within a few seconds."}),
      last.length ? el("pre", {class: "log", text: last.join("\n")}) : "",
      el("p", {class: "hint", text: `Recorder service: ${service.state || "unknown"}` +
        (service.retry_in_s ? ` · next try in ${service.retry_in_s} s` : "")}));
    return;
  }
  missing.hidden = true;
  for (const c of cam.cameras_health || []) {
    let card = container.querySelector(`[data-camera="${c.name}"]`);
    if (!card) {
      card = el("div", {class: "view stale", "data-camera": c.name},
        el("img", {alt: `${c.name} live view`}),
        el("div", {class: "preview-delay", text: "Waiting for a fresh preview"}),
        el("div", {class: "caption"}, el("span", {class: "name"}), el("span", {class: "stats"})));
      container.append(card);
    }
    const lost = c.state !== "ok";
    card.classList.toggle("lost", lost);
    const lostText = card.querySelector(".lost-text");
    if (lost && !lostText) card.append(el("div", {class: "lost-text", text: `${c.name}: ${c.error || c.state}`}));
    if (!lost && lostText) lostText.remove();
    card.querySelector(".name").textContent = `${c.name}${c.primary ? " · primary" : ""}`;
    card.querySelector(".stats").textContent = `${c.mode || ""} · ${c.fps_nominal || "?"} fps · drops ${c.camera_frames_missing || 0}`;
  }
}

function clearPreviews() {
  previewRequest?.abort();
  for (const img of $$("#views img[src]")) {
    URL.revokeObjectURL(img.src);
    img.removeAttribute("src");
    delete img.closest(".view").dataset.previewAt;
  }
}

async function refreshPreview() {
  if (state.tab !== "collect" || document.hidden) return;
  const cards = $$("#views .view:not(.lost)");
  const now = performance.now();
  for (const card of cards) {
    card.classList.toggle("stale", !card.dataset.previewAt || now - Number(card.dataset.previewAt) > 2000);
  }
  if (!cards.length || previewRequest || state.startPending || state.stopPending) return;

  // One small, finite response at a time. The next request samples the latest frame, even on a slow tunnel.
  const card = cards[nextPreview++ % cards.length];
  const controller = new AbortController();
  previewRequest = controller;
  const timer = setTimeout(() => controller.abort(), 1500);
  try {
    const response = await fetch(`/api/view/${encodeURIComponent(card.dataset.camera)}.jpg`,
      {cache: "no-store", signal: controller.signal});
    if (!response.ok) return;
    const jpeg = await response.blob();
    if (controller.signal.aborted || !card.isConnected) return;
    const img = $("img", card);
    if (img.hasAttribute("src")) URL.revokeObjectURL(img.src);
    img.src = URL.createObjectURL(jpeg);
    card.dataset.previewAt = performance.now();
    card.classList.remove("stale");
  } catch (_) {
    // The next tick requests a fresh frame. Keep the stale indication until one arrives.
  } finally {
    clearTimeout(timer);
    previewRequest = null;
  }
}

// ---------------------------------------------------------------- arms card
async function loadChecks(force = false) {
  const kind = state.runKind;
  if (!force && performance.now() - (state.checksAt[kind] || 0) < 5000) return;
  if (runActive() || !bench()?.complete) return;
  state.checksAt[kind] = performance.now();
  try {
    state.checks[kind] = await api(`/api/arms/checks?kind=${kind}`, {signal: AbortSignal.timeout(4000)});
  } catch (error) {
    state.checks[kind] = [{id: "error", label: "Checks unavailable", ok: false, level: "block", detail: error.message}];
  }
  renderArms();
}

function checklist(items) {
  return el("ul", {class: "checklist"}, items.map((c) => el("li", {class: c.level},
    el("span", {class: "mark", text: c.ok ? "ok" : c.level === "warn" ? "warn" : "fail"}),
    el("div", {}, el("div", {text: c.label}),
      c.detail && !c.ok ? el("div", {class: "check-detail", text: c.detail}) : "",
      !c.ok && c.fix ? el("div", {class: "fixline", text: c.fix}) : ""))));
}

function renderArms() {
  const body = $("#arms-body");
  const run = currentRun();
  $$("#run-kind button").forEach((b) => {
    b.classList.toggle("active", b.dataset.kind === state.runKind);
    b.disabled = runActive();
  });
  if (run && (run.active || (run.id !== state.dismissedRun && run.finished && Date.now() / 1000 - run.finished < 600))) {
    body.replaceChildren(...runView(run, true));
    return;
  }
  const b = bench();
  if (!b?.complete) {
    body.replaceChildren(el("div", {class: "message warn"}, el("strong", {text: "Bench not configured"}),
      "Arm addresses, sides and base spacing are required."),
      el("button", {type: "button", class: "big-button", text: "Configure bench", onclick: () => showTab("bench")}));
    return;
  }
  const kind = state.runKind;
  const checks = state.checks[kind];
  const blocking = (checks || []).filter((c) => !c.ok && c.level === "block");
  const needsContact = kind === "live" && (checks || []).some((c) => c.id === "contact" && !c.ok);
  const intro = {
    live: "Hardware run. Requires confirmation that the arms are at rest and the power switch is attended.",
    sim: "MuJoCo simulation of this bench, driven from the headset. No hardware is commanded.",
    dry: "Hardware launcher with simulated arm backends. Exercises the full chain without contacting the arms.",
  }[kind];
  const start = el("button", {type: "button", class: `big-button ${kind === "live" ? "" : "secondary"}`,
    text: {live: "Start arms", sim: "Start simulation", dry: "Start dry run"}[kind],
    disabled: state.startPending || !checks || blocking.length > 0, onclick: () => startRun(kind)});
  body.replaceChildren(
    el("p", {class: "hint", style: "margin-top:0", text: intro}),
    checks ? checklist(checks) : el("p", {class: "hint", text: "Checking…"}),
    needsContact ? el("div", {class: "row", style: "margin-bottom:10px"},
      el("button", {type: "button", class: "secondary", text: "Run read-only arm check",
        onclick: () => startRun("contact")})) : "",
    start,
    el("p", {class: "hint", text: kind === "live" || kind === "dry"
      ? "After start: enter VR with both grips released. The launcher starts once both controllers are tracked."
      : "After start: enter VR, release the grips, then hold a grip to drive that arm."}),
    lastRunNote());
}

function lastRunNote() {
  const last = state.app?.arms?.history?.[0];
  if (!last) return "";
  const when = last.finished ? new Date(last.finished * 1000).toLocaleTimeString() : "";
  return el("p", {class: "hint", text: `Last run: ${last.title} · ${last.outcome}${last.error ? ` (${last.error})` : ""} · ${when}`});
}

function phaseClass(run) {
  if (run.error || run.emergency) return "error";
  if (run.phase === "live" || run.phase === "home") return "live";
  if (MOVING_PHASES.includes(run.phase)) return "moving";
  if (WAITING_PHASES.includes(run.phase)) return "waiting";
  return "";
}

function runView(run, full) {
  const parts = [];
  const sides = Object.entries(run.arms || {}).map(([side, ip]) => `${side} ${ip}`).join(" · ");
  const time = run.phase === "live" && run.time_left_s !== null ? `${clock(run.time_left_s)} left`
    : run.elapsed_s ? clock(run.elapsed_s) : "";
  let sub = run.phase === "waiting_headset" ? run.message : sides;
  if (run.error) sub = run.error;
  parts.push(el("div", {class: `phase ${phaseClass(run)}`}, el("span", {class: "dot"}),
    el("div", {}, `${KIND_NAMES[run.kind] || run.kind}: ${run.phase_label}`, el("span", {class: "sub", text: sub || ""})),
    el("span", {class: "time", text: time})));

  const live = run.live;
  if (full && live && live.rows) {
    parts.push(el("div", {class: "kpis"},
      kpi("Loop rate", live.loop_hz ? `${live.loop_hz} Hz` : "—"),
      kpi("Elapsed", clock(live.elapsed_s)),
      kpi("Collision holds", `${(live.held_ticks / 90).toFixed(1)} s`)));
    parts.push(el("div", {class: "arm-rows"}, Object.entries(live.arms).map(([side, arm]) => armRow(side, arm, run.error_limit_rad))));
  }
  if (run.kind === "contact" && !run.active && Object.keys(run.contact || {}).length) {
    parts.push(contactTable(run.contact));
  }
  if (run.active) {
    if (run.stop_overdue) {
      parts.push(el("div", {class: "message bad"}, el("strong", {text: "No response to stop"}),
        run.motion_started ? "If the arms are not at rest, cut the arm power at the switch. Do not press Stop again."
          : "The launcher is stuck before any motion; force quit is safe."));
    }
    if (run.can_force_quit) {
      parts.push(el("button", {type: "button", class: "secondary", style: "margin-bottom:8px", text: "Force quit",
        onclick: () => act(() => api("/api/arms/force-quit", {method: "POST"}), "Stopped")}));
    }
    const waiting = run.phase === "waiting_headset";
    parts.push(el("button", {type: "button", class: "big-button danger", disabled: state.stopPending || stopInProgress(run),
      text: state.stopPending ? "Sending Stop…" : stopInProgress(run) ? "Stopping: returning to rest"
        : waiting ? "Cancel" : "Stop arms", onclick: stopArms}));
    if (!stopInProgress(run) && run.touches_arms !== false && run.motion_started) {
      parts.push(el("p", {class: "hint", text: "Stop returns the arms to rest. Releasing a grip holds that arm in place."}));
    }
  } else {
    const ok = run.outcome === "done" && !run.error;
    parts.push(el("div", {class: `message ${ok ? "ok" : run.outcome === "cancelled" ? "warn" : "bad"}`},
      el("strong", {text: ok ? (run.kind === "contact" ? "Read-only check passed" : "Run completed") : run.outcome === "cancelled" ? "Cancelled" : "Stopped with an error"}),
      run.error || (run.kind === "contact" ? "All arms responded. No motion commanded." : run.motion_started ? "Arms returned to rest." : "")));
    parts.push(el("button", {type: "button", class: "secondary", text: "Done", onclick: () => { state.dismissedRun = run.id; loadChecks(true); renderArms(); }}));
  }
  if (full) {
    parts.push(el("details", {class: "log"}, el("summary", {text: "Launcher output"}),
      el("pre", {text: (run.log || []).join("\n") || "(nothing yet)"}),
      el("a", {href: "/api/arms/log", target: "_blank", class: "hint", text: "Full log"})));
  }
  return parts;
}

function kpi(k, v) { return el("div", {class: "kpi"}, el("div", {class: "k", text: k}), el("div", {class: "v", text: v})); }

function armRow(side, arm, limit) {
  const error = arm.error_rad ?? 0;
  const fraction = limit ? Math.min(1, error / limit) : 0;
  const errClass = fraction > 0.9 ? "over" : fraction > 0.6 ? "high" : "";
  return el("div", {class: "arm-row"},
    el("div", {class: "side", text: side}),
    el("div", {},
      el("div", {class: "row", style: "gap:4px;margin-bottom:4px"},
        tag(arm.tracked ? "tracked" : "lost", arm.tracked ? "ok" : "bad"),
        tag(arm.clutch ? "engaged" : "holding", arm.clutch ? "ok" : ""),
        arm.held ? tag("collision hold", "warn") : "",
        arm.ik_failures ? tag(`IK miss ${arm.ik_failures}`, "warn") : ""),
      el("div", {class: "meters"},
        el("span", {class: "hint", style: "margin:0", text: "grip"}),
        el("div", {class: "meter"}, el("span", {style: `width:${Math.round((arm.grip || 0) * 100)}%`})),
        el("span", {class: "num", text: (arm.grip || 0).toFixed(2)}),
        el("span", {class: "hint", style: "margin:0", text: "track err"}),
        el("div", {class: `meter err ${errClass}`}, el("span", {style: `width:${Math.round(fraction * 100)}%`})),
        el("span", {class: "num", text: limit ? `${error.toFixed(3)}/${limit} rad` : `${error.toFixed(3)} rad`}))));
}

function contactTable(contact) {
  const rows = Object.entries(contact).filter(([k]) => !k.startsWith("_"));
  return el("table", {class: "data-table"},
    el("tr", {}, ["Arm", "Address", "Driver / firmware", "Gripper", "Joints [rad]"].map((h) => el("th", {text: h}))),
    rows.map(([side, info]) => el("tr", {},
      el("td", {text: side}), el("td", {text: info.ip || "—"}),
      el("td", {text: `${info.driver || "?"} / ${info.firmware || "?"}`}),
      el("td", {class: "num", text: info.gripper_m !== undefined ? `${(info.gripper_m * 1000).toFixed(1)} mm` : "—"}),
      el("td", {class: "num", text: (info.q_rad || []).map((q) => q.toFixed(3)).join(" ") || "—"}))));
}

async function stopArms() {
  const run = currentRun();
  if (run?.active && !stopInProgress(run)) await sendArmCommand("stop", {run_id: run.id});
}

async function sendArmCommand(command, body) {
  const pending = command === "stop" ? "stopPending" : "startPending";
  if (state[pending]) return null;
  state[pending] = true;
  previewRequest?.abort();
  renderTop();
  renderArms();
  if (command === "start") {
    toast("Sending Start…");
    // Use the bench clock from the last poll, advanced by browser elapsed time.
    body.expires_at = state.app.time + (performance.now() - state.appAt) / 1000 + 8;
  }
  try {
    const result = await api(`/api/arms/${command}`, {method: "POST", body,
      signal: AbortSignal.timeout(command === "stop" ? 4000 : 10000)});
    armRevision++;
    const current = currentRun();
    if (command === "start" ? current?.id !== result.id : current?.id === result.id && current.active) {
      state.app.arms.run = result;
    }
    if (command === "stop") toast(result.active ? "Arms are stopping" : "Arm start cancelled");
    return result;
  } catch (error) {
    const message = command === "stop" ? "Stop was not confirmed. For an immediate stop, use the arm power switch."
      : "Start was not confirmed. Check the arm status before trying again.";
    toast(error.name === "TimeoutError" || error instanceof TypeError ? message : error.message, true);
    return null;
  } finally {
    state[pending] = false;
    renderTop();
    renderArms();
    poll();
  }
}

async function startRun(kind, side = null) {
  if (state.startPending || runActive()) return;
  if (["live", "guarded", "contact"].includes(kind)) {
    previewRequest?.abort();
    let plan;
    try {
      plan = await api(`/api/arms/plan?kind=${kind}${side ? `&side=${side}` : ""}`, {signal: AbortSignal.timeout(4000)});
    } catch (error) { toast(error.message, true); return; }
    confirmDialog(plan, side);
    return;
  }
  if (await sendArmCommand("start", {kind})) showTab("collect");
}

function confirmDialog(plan, side) {
  const dialog = $("#confirm-dialog");
  const st = benchState();
  const moves = plan.kind === "contact" ? "No motion. Connects to each controller, reads joint state and versions, disconnects."
    : plan.kind === "guarded" ? `Only the ${side} arm moves, at the 0.40 response. The other arm is connected read-only and screens every command.`
    : "The arms move from rest to home, then follow the controllers while a grip is held.";
  const rows = Object.entries(plan.arms).map(([s, ip]) => el("tr", {}, el("td", {text: `${s[0].toUpperCase()}${s.slice(1)} arm`}), el("td", {text: ip})));
  if (plan.kind === "live") {
    rows.push(el("tr", {}, el("td", {text: "Response (move / turn)"}), el("td", {text: bench().responses[st.response].split(":").slice(1).join(":").trim()})));
    rows.push(el("tr", {}, el("td", {text: "Tracking stop"}), el("td", {text: `${plan.error_limit_rad} rad`})));
  }
  if (plan.cap_s) rows.push(el("tr", {}, el("td", {text: "Run time limit"}), el("td", {text: plan.cap_s >= 120 ? `${Math.round(plan.cap_s / 6) / 10} min` : `${plan.cap_s} s`})));
  const form = el("form", {method: "dialog"},
    el("h3", {text: plan.title}),
    el("p", {class: "hint", style: "margin-top:0", text: moves}),
    el("table", {class: "arms-table"}, rows),
    el("label", {class: "confirm"}, el("input", {type: "checkbox", name: "resting"}),
      el("span", {text: "Arms are at rest and firmly mounted; nothing and nobody is within reach."})),
    el("label", {class: "confirm"}, el("input", {type: "checkbox", name: "switch"}),
      el("span", {text: "The arm power switch is attended and can be cut immediately."})),
    plan.token ? el("div", {class: "token", text: `Live token: ${plan.token}`}) : "",
    el("div", {class: "actions"},
      el("button", {type: "button", class: "secondary", text: "Cancel", onclick: () => dialog.close()}),
      el("button", {type: "submit", value: "go", class: plan.kind === "contact" ? "" : "danger-btn", text: plan.kind === "contact" ? "Run check" : "Start", disabled: true})));
  const boxes = $$("input[type=checkbox]", form);
  const go = $("button[type=submit]", form);
  boxes.forEach((b) => b.addEventListener("change", () => { go.disabled = !boxes.every((x) => x.checked); }));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    dialog.close();
    const body = {kind: plan.kind, side, token: plan.token, confirm: {resting: true, switch: true}};
    const result = await sendArmCommand("start", body);
    if (result) {
      toast(plan.kind === "contact" ? "Reading arm state" : "Waiting for the headset: enter VR with both grips released");
      if (plan.kind !== "guarded") showTab("collect");
    }
  });
  dialog.replaceChildren(form);
  dialog.showModal();
}

// ---------------------------------------------------------------- recording card
function currentTask() {
  const select = $("#task-select");
  return select.value === OTHER ? $("#task-custom").value.trim() : select.value;
}

function renderRecord() {
  const s = camera();
  const button = $("#record-button");
  if (!s) {
    button.disabled = true;
    button.textContent = "Recorder unavailable";
    $("#outcome").hidden = true;
    $("#last-episode").replaceChildren(el("div", {class: "hint", text: "Recording is available once a camera is connected."}));
    return;
  }
  const select = $("#task-select");
  const presets = s.task_presets || [];
  const wanted = [...presets, OTHER].join("|");
  if (select.dataset.options !== wanted) {
    const current = select.value;
    select.replaceChildren(...presets.map((t) => el("option", {value: t, text: t})), el("option", {value: OTHER, text: "Other…"}));
    select.dataset.options = wanted;
    const pick = current || s.defaults?.task || "";
    if (presets.includes(pick)) select.value = pick;
    else if (pick) { select.value = OTHER; $("#task-custom").value = pick; }
    $("#task-custom").hidden = select.value !== OTHER;
  }
  if (!state.operatorSet && s.defaults?.operator !== undefined) {
    $("#operator").value = s.defaults.operator || "";
    state.operatorSet = true;
  }
  button.textContent = s.recording ? (s.episode?.stopping ? "Stopping…" : `Stop episode · ${clock(recElapsed())}`) : "Start episode";
  button.classList.toggle("recording", s.recording);
  button.disabled = Boolean(s.episode?.stopping) || (!s.recording && Object.values(s.cameras || {}).some((c) => c.state !== "ok"));
  select.disabled = $("#task-custom").disabled = $("#operator").disabled = s.recording;
  const last = s.last_episode;
  if (last && !s.recording && state.outcomeFor !== last.directory) {
    state.outcomeFor = last.directory;
    $("#outcome").hidden = false;
    $("#outcome-note").value = last.labels?.note || "";
  }
  if (s.recording) $("#outcome").hidden = true;
  $$("#outcome [data-outcome]").forEach((b) => b.classList.toggle("selected", last?.labels?.outcome === b.dataset.outcome));
  renderLast(last);
}

function renderLast(last) {
  const box = $("#last-episode");
  if (!last) { box.replaceChildren(el("div", {class: "hint", text: "No episode in this session."})); return; }
  const rows = [
    ["Episode", `${last.episode_index} ${last.labels?.task ? `· ${last.labels.task}` : ""}`],
    ["Status", last.status === "complete" ? "complete" : `${last.status}: ${last.stop_reason}`],
    ["Length", last.duration_s !== undefined ? `${last.duration_s} s` : "—"],
    ["Frames", `${last.frames_recorded ?? "—"}${last.camera_frames_missing ? ` (${last.camera_frames_missing} dropped)` : ""}`],
  ];
  box.replaceChildren(el("strong", {text: "Last episode"}),
    el("dl", {}, rows.flatMap(([k, v]) => [el("dt", {text: k}), el("dd", {text: v})])));
}

async function toggleRecording() {
  const s = camera();
  if (!s) return;
  $("#record-button").disabled = true;
  try {
    if (s.recording) {
      await api("/record/stop", {method: "POST", body: {}});
      toast("Episode saved. Set its outcome.");
    } else {
      const task = currentTask();
      const operator = $("#operator").value.trim();
      await api("/record/defaults", {method: "POST", body: {task, operator}});
      await api("/record/start", {method: "POST", body: {task, operator}});
    }
  } catch (error) { toast(error.message, true); }
  poll();
}

async function setOutcome(outcome) {
  await act(() => api("/record/label", {method: "POST", body: {outcome, note: $("#outcome-note").value || null}}), `Marked ${outcome}`);
}

async function saveDefaults() {
  if (camera()?.recording) return;
  try { await api("/record/defaults", {method: "POST", body: {task: currentTask(), operator: $("#operator").value.trim()}}); } catch (_) { /* next poll shows it */ }
}

// ---------------------------------------------------------------- headset card
function renderQuest() {
  const q = state.app.quest || {};
  const page = q.page || {};
  const st = benchState();
  const mode = st.arms === 1 ? st.single_hand : "bimanual";
  const wantMode = mode === "bimanual" ? "bimanual" : `single ${mode}`;
  const usbOk = q.usb === "connected";
  const pageOpen = (page.pages || []).length > 0 || page.streaming;
  const steps = [
    ["USB connection (developer mode on)", q.usb && q.usb !== "none" && q.usb !== "no_adb"],
    ["USB debugging allowed in the headset", usbOk],
    ["Headset page open", pageOpen],
    [`Streaming: ${mode === "bimanual" ? "bimanual" : `single, ${mode} hand`} (Enter Passthrough/VR)`, page.streaming && page.mode === wantMode],
  ];
  const hands = page.hands || {};
  const handCard = (side) => {
    const h = hands[side];
    return el("div", {class: `hand ${h ? "" : "off"}`}, el("div", {class: "name", text: side}),
      h ? el("div", {}, tag(h.tracked ? "tracked" : "lost", h.tracked ? "ok" : "bad"), " ",
        tag(`grip ${Number(h.grip || 0).toFixed(2)}`, h.grip > 0.2 ? "warn" : ""), " ",
        h.mapping && h.mapping !== "real" ? tag(h.mapping, "warn") : "") : el("div", {class: "hint", style: "margin:0", text: "no data"}));
  };
  $("#quest-body").replaceChildren(
    usbOk ? "" : el("div", {class: "message warn", text: q.usb_message || "checking"}),
    checklist(steps.map(([label, done]) => ({label, ok: Boolean(done), level: done ? "ok" : "warn"}))),
    page.streaming ? el("div", {class: "hint", text: `${page.rate_hz} packets/s${page.from_headset ? "" : " · source is not a headset"}`}) : "",
    el("div", {class: "hands"}, handCard("left"), handCard("right")),
    el("div", {class: "row"},
      el("button", {type: "button", class: "secondary", disabled: !usbOk, text: "Open page in headset",
        onclick: () => act(() => api("/api/quest/open", {method: "POST", body: {mode}}), "Page opened in the headset browser")}),
      el("button", {type: "button", class: "link", disabled: !usbOk, text: "Reconnect ports",
        onclick: () => act(() => api("/api/quest/prepare", {method: "POST"}), "Headset ports set up")})),
    q.error ? el("p", {class: "hint", text: q.error}) : "");
}

// ---------------------------------------------------------------- bench tab
async function loadNetwork() {
  try { state.network = await api("/api/bench/network"); state.networkAt = Date.now(); } catch (error) { toast(error.message, true); }
  renderBench(true);
}

async function saveBench(changes, message) {
  try {
    const result = await api("/api/bench", {method: "POST", body: changes});
    if (message) toast(message);
    if (state.app) state.app.bench.state = result.state;
    state.benchCheck = null;
    await poll();
    renderBench(true);
    return result;
  } catch (error) {
    toast(error.message, true);
    return null;
  }
}

function step(id, number, title, done, summary, body, attention = false) {
  const details = el("details", {class: `step ${done ? "done" : attention ? "attention" : ""}`, "data-step": id},
    el("summary", {}, el("span", {class: "step-num", text: number}), el("span", {class: "title", text: title}),
      el("span", {class: "summary-text", text: summary || ""}),
      done ? tag("done", "ok") : attention ? tag("action", "warn") : tag("open")),
    el("div", {class: "step-body"}, body));
  if (state.openSteps.has(id)) details.open = true;
  details.addEventListener("toggle", () => {
    if (details.open) state.openSteps.add(id); else state.openSteps.delete(id);
    if (details.open && id === "camera" && !state.boxes) loadBoxes();
  });
  return details;
}

function renderBench(force) {
  const b = bench();
  if (!b) return;
  const key = JSON.stringify([b.state, b.contact, state.network, state.scan, state.scanning, state.identify, state.picture,
    state.benchCheck, state.fixResult, currentRun()?.id, currentRun()?.phase, currentRun()?.active, state.motions, camera() ? 1 : 0]);
  if (!force && key === state.benchRenderedFor) return;
  if (document.activeElement && $("#bench-steps").contains(document.activeElement) && document.activeElement.tagName === "INPUT" && !force) return;
  state.benchRenderedFor = key;
  const st = b.state;
  const dual = st.arms === 2;
  const net = state.network;
  const netDone = Boolean(net && net.has_source_ip && net.power_profile.ok && net.ac.ok);
  const armsDone = dual ? Boolean(st.left_ip && st.right_ip) : Boolean(st.single_ip);
  const sidesDone = dual ? Boolean(st.identified) : Boolean(st.single_ip);
  const layoutDone = dual ? Boolean(st.spacing_measured) : true;
  const checksDone = Boolean(state.benchCheck?.profile?.ok && state.benchCheck?.workstation?.ok && b.contact);
  const stepsDone = [netDone, armsDone, sidesDone, layoutDone, b.complete, checksDone, Boolean(st.validated)];
  if (!state.openSteps.size) {
    const first = stepsDone.findIndex((d) => !d);
    state.openSteps.add(["network", "arms", "sides", "layout", "driving", "checks", "first-run"][first >= 0 ? first : 0]);
  }
  $("#bench-steps").replaceChildren(
    step("network", 1, "PC and network", netDone, net ? (netDone ? `${net.source_ip} ready` : "needs a fix") : "checking…", networkStep(net)),
    step("arms", 2, "Arms on the network", armsDone, armsDone ? (dual ? `${st.left_ip}, ${st.right_ip}` : st.single_ip) : "", armsStep(st)),
    step("sides", 3, dual ? "Which arm is which" : "Which hand drives it", sidesDone,
      sidesDone ? (dual ? `left ${st.left_ip} · right ${st.right_ip}` : `${st.single_hand} controller`) : "", sidesStep(st), armsDone && !sidesDone),
    step("layout", 4, "Layout", layoutDone, dual ? `${st.spacing_mm} mm apart` : "one arm", layoutStep(st)),
    step("driving", 5, "Driving settings", b.complete, b.complete ? b.responses[st.response].split(":")[0]
      : b.ready_to_save ? "not saved yet" : "finish steps 2 to 4 first", drivingStep(st, b), b.ready_to_save && !b.complete),
    step("checks", 6, "Checks", checksDone, b.contact ? "read-only check passed" : "", checksStep(st, b), b.complete && !checksDone),
    step("first-run", 7, "First-run check", Boolean(st.validated), st.validated ? `done ${st.validated.at}` : "recommended on a new bench", firstRunStep(st, b)),
    step("camera", 8, "Camera", Boolean(camera()), camera() ? "running" : "not found", cameraStep()),
  );
  if (state.openSteps.has("camera") && !state.boxes) loadBoxes();
  else if (state.openSteps.has("camera") && boxEditor.image) drawBoxes();
}

function line(ok, text, extra = null) {
  return el("li", {class: ok ? "ok" : "block"}, el("span", {class: "mark", text: ok ? "ok" : "fail"}), el("div", {}, text, extra));
}

function networkStep(net) {
  if (!net) return el("p", {class: "hint", text: "Checking the network…"});
  const rows = net.interfaces.map((n) => el("tr", {},
    el("td", {text: n.name + (n.usb ? " (USB)" : "")}),
    el("td", {}, tag(n.carrier ? "cable in" : "no cable", n.carrier ? "ok" : "")),
    el("td", {text: n.addresses.join(", ") || "—"}),
    el("td", {}, n.arm_address ? tag("arm network", "ok") : el("button", {type: "button", class: "secondary", text: "Assign to arm network",
      disabled: !n.carrier, title: n.carrier ? "" : "plug the arm network cable into this port first",
      onclick: async () => {
        state.fixResult = await act(() => api("/api/bench/network/fix", {method: "POST", body: {ifname: n.name}}));
        if (state.fixResult) state.fixResult.ifname = n.name;
        loadNetwork();
      }}))));
  const routes = Object.entries(net.routes || {});
  return el("div", {},
    el("p", {text: `The arm controllers are reached through a wired interface with the static address ${net.source_ip} (no gateway). Connect the arms' network switch to a free wired port; a USB-Ethernet adapter works.`}),
    net.interfaces.length ? el("table", {class: "data-table"}, el("tr", {}, ["Port", "Link", "Address", ""].map((h) => el("th", {text: h}))), rows)
      : el("div", {class: "message warn", text: "No wired network port found. Plug in a USB-Ethernet adapter."}),
    state.fixResult && !state.fixResult.ok ? el("div", {class: "message bad"}, el("strong", {text: "Couldn't change the network"}),
      state.fixResult.detail,
      state.fixResult.ifname ? el("div", {style: "margin-top:8px"}, el("button", {type: "button", text: "Do it with my password",
        onclick: async () => { const r = await runSystemSetup({network_if: state.fixResult.ifname}); if (r?.ok) state.fixResult = null; loadNetwork(); }})) : "") : "",
    el("ul", {class: "checklist"},
      line(net.has_source_ip, `This PC has ${net.source_ip} on a wired port`),
      line(net.power_profile.ok, `Power mode: ${net.power_profile.value || net.power_profile.detail}`,
        net.power_profile.ok ? null : el("button", {type: "button", class: "link", style: "margin-left:8px", text: "Set performance mode",
          onclick: async () => { await act(() => api("/api/bench/power", {method: "POST"}), "Performance mode on"); loadNetwork(); }})),
      line(net.ac.ok, net.ac.detail),
      ...routes.map(([ip, r]) => line(r.ok, `Route to ${ip}: ${r.detail}`))),
    el("button", {type: "button", class: "link", text: "Re-check", onclick: loadNetwork}));
}

function armsStep(st) {
  const dual = st.arms === 2;
  const choice = (value, title, sub) => el("label", {class: `choice ${st.arms === value ? "selected" : ""}`},
    el("input", {type: "radio", name: "arm-count", checked: st.arms === value, onchange: () => saveBench({arms: value})}),
    el("strong", {text: title}), el("small", {text: sub}));
  const found = state.scan;
  const pickers = found ? (found.length ? el("table", {class: "data-table"},
    el("tr", {}, ["Address", "Hardware address", ""].map((h) => el("th", {text: h}))),
    found.map((f) => el("tr", {}, el("td", {text: f.ip}), el("td", {class: "hint", text: f.mac || "—"}),
      el("td", {}, dual ? el("div", {class: "row"},
        el("button", {type: "button", class: "secondary", text: "Add", onclick: () => useArm(f.ip)})) :
        el("button", {type: "button", class: "secondary", text: "Select", onclick: () => saveBench({single_ip: f.ip}, "Saved")})))))
    : el("div", {class: "message warn", text: "No response on the arm network. Check arm power and the cable to this PC's arm port."})) : "";
  const manual = dual ? el("div", {class: "row"},
    el("label", {class: "inline-field"}, "Arm A", el("input", {id: "ip-left", value: st.left_ip || "", placeholder: "192.168.1.x"})),
    el("label", {class: "inline-field"}, "Arm B", el("input", {id: "ip-right", value: st.right_ip || "", placeholder: "192.168.1.x"})),
    el("button", {type: "button", class: "secondary", text: "Save",
      onclick: () => saveBench({left_ip: $("#ip-left").value.trim() || null, right_ip: $("#ip-right").value.trim() || null}, "Saved")}))
    : el("div", {class: "row"},
      el("label", {class: "inline-field"}, "Arm", el("input", {id: "ip-single", value: st.single_ip || "", placeholder: "192.168.1.x"})),
      el("button", {type: "button", class: "secondary", text: "Save", onclick: () => saveBench({single_ip: $("#ip-single").value.trim() || null}, "Saved")}));
  return el("div", {},
    el("div", {class: "choice-grid"}, choice(2, "Two arms", "bimanual: one per controller"), choice(1, "One arm", "driven by one controller")),
    el("p", {text: "Power the arms, then scan the arm network (ICMP ping only; no connection is made to the controllers)."}),
    el("button", {type: "button", disabled: state.scanning, text: state.scanning ? "Scanning…" : "Scan network", onclick: scanArms}),
    pickers,
    el("p", {class: "field-label", text: "Manual entry"}), manual,
    el("p", {class: "hint", text: "Every arm needs its own address. Two arms with the same address can't both answer; change one with Trossen's own tools first."}));
}

async function scanArms() {
  state.scanning = true;
  renderBench(true);
  try { state.scan = await api("/api/bench/scan", {method: "POST"}); } catch (error) { toast(error.message, true); }
  state.scanning = false;
  renderBench(true);
}

function useArm(ip) {
  const st = benchState();
  if (st.left_ip === ip || st.right_ip === ip) return;
  if (!st.left_ip) saveBench({left_ip: ip}, `${ip} added`);
  else saveBench({right_ip: ip}, `${ip} added: now tell which is which (step 3)`);
}

function sidesStep(st) {
  if (st.arms === 1) {
    const choice = (hand) => el("label", {class: `choice ${st.single_hand === hand ? "selected" : ""}`},
      el("input", {type: "radio", name: "single-hand", checked: st.single_hand === hand, onchange: () => saveBench({single_hand: hand})}),
      el("strong", {text: `${hand[0].toUpperCase()}${hand.slice(1)} controller`}),
      el("small", {text: `uses the ${hand}-hand calibration (operator behind the arm)`}));
    return el("div", {}, el("div", {class: "choice-grid"}, choice("left"), choice("right")));
  }
  if (!st.left_ip || !st.right_ip) return el("p", {class: "hint", text: "Add both arms in step 2 first."});
  const id = state.identify;
  const ips = [st.left_ip, st.right_ip];
  const box = (ip) => {
    const up = id?.answers?.[ip];
    return el("div", {class: `ip-state ${up === true ? "up" : up === false ? "down" : ""}`},
      el("div", {class: "ip", text: ip}), el("div", {text: up === undefined ? "…" : up ? "answering" : "not answering"}));
  };
  const intro = el("p", {}, "The side assignment places each arm in the collision model. Standing behind the arms, ",
    "identify the left arm by disconnecting its network cable.");
  if (!id) {
    return el("div", {}, intro,
      st.identified ? el("div", {class: "message ok", text: `Left: ${st.left_ip} · Right: ${st.right_ip}`}) : "",
      el("div", {class: "row"},
        el("button", {type: "button", text: "Start cable test", onclick: startIdentify}),
        el("button", {type: "button", class: "secondary", text: "Swap sides", onclick: () => saveBench({left_ip: st.right_ip, right_ip: st.left_ip, identified: false})}),
        st.identified ? "" : el("button", {type: "button", class: "link", text: `Confirm without test (left = ${st.left_ip})`, onclick: () => saveBench({identified: true}, "Sides saved")})));
  }
  const text = {
    unplug: "Disconnect the network cable of the arm on your LEFT (at the arm or at the switch). Leave the power on.",
    replug: `Left arm identified: ${id.left}. Reconnect its cable.`,
    both: "Both arms stopped responding. Reconnect the cables and restart the test.",
  }[id.stage];
  return el("div", {}, intro, el("div", {class: `message ${id.stage === "both" ? "bad" : "warn"}`, text}),
    el("div", {class: "identify"}, ips.map(box)),
    el("button", {type: "button", class: "secondary", text: "Cancel", onclick: () => { clearInterval(state.identifyTimer); state.identify = null; renderBench(true); }}));
}

function startIdentify() {
  const st = benchState();
  state.identify = {stage: "unplug", answers: {}, downCount: {}, left: null};
  clearInterval(state.identifyTimer);
  state.identifyTimer = setInterval(async () => {
    const id = state.identify;
    if (!id) { clearInterval(state.identifyTimer); return; }
    let answers;
    try { answers = await api("/api/bench/ping", {method: "POST", body: {ips: [st.left_ip, st.right_ip]}}); } catch (_) { return; }
    id.answers = answers;
    const down = Object.entries(answers).filter(([, up]) => !up).map(([ip]) => ip);
    if (id.stage === "unplug" || id.stage === "both") {
      if (down.length === 2) id.stage = "both";
      else if (down.length === 1) {
        id.downCount[down[0]] = (id.downCount[down[0]] || 0) + 1;
        if (id.downCount[down[0]] >= 2) { id.left = down[0]; id.stage = "replug"; }
      } else if (id.stage === "both") id.stage = "unplug";
    } else if (id.stage === "replug" && down.length === 0) {
      clearInterval(state.identifyTimer);
      const right = id.left === st.left_ip ? st.right_ip : st.left_ip;
      state.identify = null;
      await saveBench({left_ip: id.left, right_ip: right, identified: true}, `Left arm ${id.left}, right arm ${right}`);
      return;
    }
    renderBench(true);
  }, 1200);
  renderBench(true);
}

function sketch(st) {
  const dual = st.arms === 2;
  const w = 520;
  const arms = dual ? [["LEFT", st.left_ip, 180], ["RIGHT", st.right_ip, 340]] : [[st.single_hand.toUpperCase(), st.single_ip, 260]];
  // innerHTML on a wrapper: elements made with createElement("svg") would not be in the SVG namespace.
  return el("div", {html: `<svg class="sketch" viewBox="0 0 ${w} 250" role="img" aria-label="Top view of the bench">
      <rect class="table" x="60" y="20" width="400" height="120" rx="2"/>
      <text x="260" y="60" text-anchor="middle" class="muted">table · top view</text>
      ${arms.map(([name, ip, x]) => `<circle class="arm" cx="${x}" cy="140" r="16"/>
        <text x="${x}" y="182" text-anchor="middle">${name}</text>
        <text x="${x}" y="198" text-anchor="middle" class="muted">${ip || "?"}</text>`).join("")}
      ${dual ? `<line class="dim" x1="180" y1="112" x2="340" y2="112"/><line class="dim" x1="180" y1="104" x2="180" y2="120"/>
        <line class="dim" x1="340" y1="104" x2="340" y2="120"/><text x="260" y="104" text-anchor="middle">${st.spacing_mm} mm</text>` : ""}
      <circle class="op" cx="260" cy="226" r="11"/><text x="282" y="231" class="muted">operator</text></svg>`});
}

function layoutStep(st) {
  if (st.arms === 1) return el("div", {}, sketch(st), el("p", {class: "hint", text: "Nothing to measure for one arm."}));
  return el("div", {},
    sketch(st),
    el("p", {text: "Measure from the centre of one arm's base to the centre of the other with a tape. The arms must be parallel, both at the table's edge."}),
    el("div", {class: "row"},
      el("label", {class: "inline-field"}, "Separation", el("input", {id: "spacing", type: "number", min: 350, max: 1500, step: 5, value: st.spacing_mm}), "mm"),
      el("button", {type: "button", text: "Save measurement", onclick: () => saveBench({spacing_mm: Number($("#spacing").value), spacing_measured: true}, "Saved")})),
    st.spacing_measured ? "" : el("p", {class: "hint", text: "Not measured yet: the value above is only the default."}),
    safetyFields(st, bench()),
    el("div", {style: "margin-top:12px"},
      el("button", {type: "button", class: "secondary", text: state.picture === "rendering" ? "Rendering…" : "Render simulated bench",
        disabled: state.picture === "rendering" || !bench().complete, onclick: renderPicture}),
      state.picture && state.picture !== "rendering" ? el("div", {},
        el("img", {class: "picture", src: `/api/bench/picture.png?t=${state.picture}`, alt: "simulated bench"}),
        el("p", {class: "hint", text: "Simulated bench seen from the operator position. The left arm in the render must match the physical left arm."})) : ""));
}

// A drop-down of the usual choices plus "Custom…", which shows a number field (in display units) and a Save button.
// scale converts display units to the stored value (cm -> mm: 10, min -> s: 60); the server checks the range.
function choiceWithCustom(st, b, key, options, label, unit, scale = 1, step = 1) {
  state.customOpen ||= {};
  const value = st[key];
  const custom = state.customOpen[key] || !options.includes(value);
  const [low, high] = (b.ranges || {})[key] || [null, null];
  const id = `custom-${key}`;
  const save = () => {
    const shown = Number($(`#${id}`).value);
    if (!Number.isFinite(shown)) return toast("enter a number", true);
    state.customOpen[key] = false;
    saveBench({[key]: Math.round(shown * scale * 1000) / 1000}, "Saved");
  };
  return el("span", {class: "custom-choice"},
    el("select", {style: "width:auto", onchange: (e) => {
      if (e.target.value === "custom") { state.customOpen[key] = true; renderBench(true); return; }
      state.customOpen[key] = false;
      saveBench({[key]: Number(e.target.value)}, "Saved");
    }},
      options.map((v) => el("option", {value: v, text: label(v), selected: !custom && v === value})),
      el("option", {value: "custom", text: custom && !options.includes(value) ? `Custom: ${label(value)}` : "Custom…", selected: custom})),
    custom ? el("span", {},
      " ", el("input", {id, type: "number", step, value: Math.round(value / scale * 1000) / 1000, style: "width:90px",
        min: low == null ? undefined : low / scale, max: high == null ? undefined : high / scale,
        onkeydown: (e) => { if (e.key === "Enter") save(); }}),
      ` ${unit} `, el("button", {type: "button", class: "secondary", text: "Save", onclick: save}),
      low == null ? "" : el("small", {class: "hint", text: ` ${low / scale}-${high / scale} ${unit}`})) : "");
}

function safetyFields(st, b) {
  const number = (id, key, value, min, max) => el("input", {id, type: "number", min, max, step: 1, value, style: "width:90px"});
  return el("div", {style: "margin-top:14px"},
    el("p", {class: "field-label", text: "Collision check between the arms"}),
    el("div", {class: "row"},
      el("label", {class: "inline-field"}, "Keep the arms at least",
        choiceWithCustom(st, b, "clearance_mm", b.clearances_mm || [30], (v) => `${v / 10} cm${v === 30 ? " (teleop default)" : ""}`, "cm", 10, 0.1),
        "apart")),
    el("p", {class: "hint", text: "A command that would bring any part of one arm closer than this to the other is held until the hands move apart."}),
    el("p", {class: "field-label", style: "margin-top:12px", text: "Table floor"}),
    el("label", {class: "check"}, el("input", {type: "checkbox", checked: st.table_floor,
      onchange: (e) => saveBench({table_floor: e.target.checked}, e.target.checked ? "Table floor on" : "Table floor off")}),
      "Hold an arm before it goes into the table (the gripper can come down to the surface to pick up cloth)"),
    st.table_floor ? el("div", {},
      el("div", {class: "row"},
        el("label", {class: "inline-field"}, "Table width", number("table-width", "table_width_mm", st.table_width_mm, 400, 3000), "mm"),
        el("label", {class: "inline-field"}, "depth", number("table-depth", "table_depth_mm", st.table_depth_mm, 300, 2000), "mm"),
        el("label", {class: "inline-field"}, "Base centre to the operator-side edge", number("table-inset", "table_rear_inset_mm", st.table_rear_inset_mm, 0, 300), "mm"),
        el("button", {type: "button", class: "secondary", text: "Save table", onclick: () => saveBench({
          table_width_mm: Number($("#table-width").value), table_depth_mm: Number($("#table-depth").value),
          table_rear_inset_mm: Number($("#table-inset").value)}, "Table saved")})),
      el("div", {class: "row"},
        el("label", {class: "inline-field"}, "Floor below the real surface",
          choiceWithCustom(st, b, "table_floor_offset_mm", b.floor_offsets_mm || [0], (v) => `${v} mm`, "mm"))),
      el("p", {class: "hint", text: "Width runs along the line of the arms (centred between them), depth away from the operator. A floor 1-3 mm below the surface lets the fingers touch the table gently; 0 stops them at the modelled surface."})) : "");
}

async function renderPicture() {
  state.picture = "rendering";
  renderBench(true);
  const result = await act(() => api("/api/bench/picture", {method: "POST"}));
  state.picture = result?.ok ? Date.now() : null;
  if (result && !result.ok) toast(result.detail || "rendering failed", true);
  renderBench(true);
}

function drivingStep(st, b) {
  const radio = (name, value, current, title, sub, key) => el("label", {class: `choice ${current === value ? "selected" : ""}`},
    el("input", {type: "radio", name, checked: current === value, onchange: () => saveBench({[key]: value}, "Saved")}),
    el("strong", {text: title}), sub ? el("small", {text: sub}) : "");
  return el("div", {},
    el("p", {class: "field-label", text: "Response: how far the gripper moves for a hand movement"}),
    el("div", {class: "choice-grid"}, Object.entries(b.responses).map(([key, label]) =>
      radio("response", key, st.response, label.split(":")[0], label.split(":").slice(1).join(":").trim(), "response"))),
    st.response === "custom" ? el("div", {class: "row"},
      el("label", {class: "inline-field"}, "Move", el("input", {id: "custom-move", type: "number", step: 0.05, min: 0.2, max: 1, value: st.custom_move, style: "width:80px"})),
      el("label", {class: "inline-field"}, "Turn", el("input", {id: "custom-turn", type: "number", step: 0.05, min: 0.2, max: 1, value: st.custom_turn, style: "width:80px"})),
      el("button", {type: "button", class: "secondary", text: "Save", onclick: () => saveBench({custom_move: Number($("#custom-move").value),
        custom_turn: Number($("#custom-turn").value)}, "Saved")}),
      el("small", {class: "hint", text: "0.20-1.00; 0.40 = a 10 cm hand move gives 4 cm at the gripper"})) : "",
    el("p", {class: "field-label", text: "Tracking stop: stop the run when a joint lags its command by more than"}),
    el("div", {class: "row"}, el("label", {class: "inline-field"}, "Stop at",
      choiceWithCustom(st, b, "tracking_stop_rad", b.tracking_stops || [0.08, 0.12],
        (v) => `${v} rad${v === 0.08 ? " (teleop default)" : v === 0.12 ? " (more tolerant)" : ""}`, "rad", 1, 0.01))),
    el("p", {class: "hint", text: "Not a speed setting: a higher value tolerates the lag of fast moves, but notices a blocked or colliding joint later."}),
    el("div", {class: "row", style: "margin-bottom:10px"},
      el("label", {class: "inline-field"}, "Run time limit",
        choiceWithCustom(st, b, "cap_s", b.caps_s, (s) => s >= 120 && s % 60 === 0 ? `${s / 60} min` : `${s} s`, "min", 60, 0.5))),
    el("label", {class: "check"}, el("input", {type: "checkbox", checked: st.gripper, onchange: (e) => saveBench({gripper: e.target.checked}, "Saved")}),
      "Gripper control (trigger)"),
    b.complete ? el("p", {class: "hint", text: `Saved as ${b.profile.split("/").slice(-3).join("/")}.`})
      : b.ready_to_save ? el("div", {style: "margin-top:12px"},
        st.imported_from && !b.saved ? el("p", {class: "hint", text: `These settings were read from ${st.imported_from.split("/").slice(-3).join("/")}. Check them, then save.`}) : "",
        el("button", {type: "button", text: "Save the bench", onclick: () => saveBench({}, "Bench saved")}))
      : el("p", {class: "hint", text: "The arm profile is written once steps 2 to 4 are done."}));
}

function checksStep(st, b) {
  const c = state.benchCheck;
  const profile = c?.profile;
  const pc = c?.workstation;
  return el("div", {},
    el("p", {text: "Configuration check: the teleop launcher's own validators and live gates, offline. Read-only arm check: connects to each controller, reads joint state and versions, disconnects; no motion is commanded."}),
    el("div", {class: "row", style: "margin-bottom:10px"},
      el("button", {type: "button", class: "secondary", disabled: !b.complete, text: c === "running" ? "Checking…" : "Run configuration check",
        onclick: runBenchCheck}),
      el("button", {type: "button", disabled: !b.complete || runActive(), text: "Read-only arm check", onclick: () => startRun("contact")})),
    profile ? el("ul", {class: "checklist"},
      line(profile.ok, profile.ok ? `Configuration valid (${profile.kind === "dual" ? "two arms" : "one arm"}); live gates satisfied` : `Configuration rejected: ${(profile.errors || []).join("; ")}`),
      pc ? line(pc.ok, `PC check: ${pc.detail}`) : "") : "",
    profile?.checks ? el("details", {class: "log"}, el("summary", {text: "Combined-scene path checks"}), el("pre", {text: profile.checks.join("\n")})) : "",
    b.contact ? el("div", {}, el("div", {class: "message ok", text: `Read-only check passed at ${new Date(b.contact.at * 1000).toLocaleTimeString()}`}),
      contactTable(b.contact.arms)) : "",
    currentRun()?.kind === "contact" ? el("div", {}, runView(currentRun(), false)) : "");
}

async function runBenchCheck() {
  state.benchCheck = "running";
  renderBench(true);
  try { state.benchCheck = await api("/api/bench/check?fresh=1"); } catch (error) { state.benchCheck = null; toast(error.message, true); }
  renderBench(true);
}

function firstRunStep(st, b) {
  const dual = st.arms === 2;
  const run = currentRun();
  const guardedRun = run && ["guarded", "live"].includes(run.kind) && (run.active || !run.finished || Date.now() / 1000 - run.finished < 600) ? run : null;
  const sides = dual ? ["left", "right"] : [st.single_hand];
  const boxes = sides.map((side) => el("div", {class: "motion-col"},
    el("div", {class: "field-label", text: dual ? `${side[0].toUpperCase()}${side.slice(1)} arm` : "The arm"}),
    MOTIONS.map(([key, text]) => {
      const id = `${side}:${key}`;
      return el("label", {class: "check"}, el("input", {type: "checkbox", checked: Boolean(state.motions[id]),
        onchange: (e) => { state.motions[id] = e.target.checked; }}), text);
    })));
  return el("div", {},
    el("p", {text: dual
      ? "On a new bench, drive one arm at a time at the gentle 0.40 response while the other stays connected read-only. Check that each motion goes the right way."
      : "On a new bench, do a first live run and check that each motion goes the right way."}),
    st.validated ? el("div", {class: "message ok", text: `Checked on ${st.validated.at}${st.validated.operator ? ` by ${st.validated.operator}` : ""}.`}) : "",
    el("div", {class: "row", style: "margin-bottom:10px"}, dual
      ? ["left", "right"].map((side) => el("button", {type: "button", class: "secondary", disabled: !b.complete || runActive() || !b.contact,
        text: `Start: ${side} arm moves`, onclick: () => startRun("guarded", side)}))
      : el("button", {type: "button", class: "secondary", disabled: !b.complete || runActive() || !b.contact, text: "Start a live run",
        onclick: () => startRun("live")})),
    b.contact ? "" : el("p", {class: "hint", text: "Run the read-only arm check (step 6) first."}),
    guardedRun ? el("div", {}, runView(guardedRun, false)) : "",
    el("div", {class: "motions"}, boxes),
    el("div", {class: "row"},
      el("label", {class: "inline-field"}, "Operator", el("input", {id: "validator", value: state.validator, oninput: (e) => { state.validator = e.target.value; }})),
      el("button", {type: "button", text: "Record validation", onclick: async () => {
        const checklist = Object.fromEntries(sides.flatMap((side) => MOTIONS.map(([key]) => [`${side}:${key}`, Boolean(state.motions[`${side}:${key}`])])));
        await act(() => api("/api/bench/validated", {method: "POST", body: {operator: state.validator, checklist}}), "Validation recorded");
        renderBench(true);
      }})));
}

function cameraStep() {
  const cam = camera();
  if (!cam) return el("div", {}, el("p", {text: "No camera yet. Plug the ZED into a USB 3 port; the app picks it up by itself."}));
  return el("div", {},
    el("p", {text: "Draw the boxes the quality checks use, on a live picture: the timing box around both arms (keep yourself out of it), the review-video crop and the table."}),
    el("div", {class: "row", style: "margin-bottom:8px"},
      el("select", {id: "box-name", style: "width:auto", onchange: (e) => { state.boxName = e.target.value; drawBoxes(); }},
        [["sync_roi", "Timing box (arms, no operator)"], ["video_crop", "Video crop"], ["table", "Table"]].map(([v, t]) =>
          el("option", {value: v, text: t, selected: state.boxName === v}))),
      el("button", {type: "button", class: "secondary", text: "New picture", onclick: loadBoxes}),
      el("button", {type: "button", text: "Save boxes", onclick: saveBoxes})),
    el("div", {class: "box-canvas-wrap"}, boxCanvas()),
    el("p", {class: "hint", id: "box-status", text: state.boxStatus || ""}));
}

// box editor (pixel boxes, saved in 2K pixels by the camera service)
const boxEditor = {image: null, drag: null, canvas: null};
const BOX_COLOURS = {sync_roi: "#22c55e", video_crop: "#3b82f6", table: "#f59e0b"};

function boxCanvas() {
  if (!boxEditor.canvas) {
    const canvas = el("canvas", {id: "box-canvas"});
    canvas.addEventListener("pointerdown", (event) => {
      if (!state.boxes) return;
      boxEditor.drag = canvasPoint(event);
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!boxEditor.drag) return;
      const [x, y] = canvasPoint(event);
      const [sx, sy] = boxEditor.drag;
      state.boxes.boxes[state.boxName] = [Math.min(sx, x), Math.min(sy, y), Math.max(sx, x), Math.max(sy, y)];
      drawBoxes();
    });
    canvas.addEventListener("pointerup", () => { boxEditor.drag = null; state.boxStatus = "Not saved yet."; const s = $("#box-status"); if (s) s.textContent = state.boxStatus; });
    boxEditor.canvas = canvas;
  }
  return boxEditor.canvas;
}

async function loadBoxes() {
  try { state.boxes = await api("/api/setup/boxes"); state.boxStatus = ""; } catch (error) { state.boxStatus = `Boxes unavailable: ${error.message}`; return; }
  const image = new Image();
  image.onload = () => { boxEditor.image = image; drawBoxes(); };
  image.src = `/api/setup/still.jpg?t=${Date.now()}`;
}

function drawBoxes() {
  const canvas = boxEditor.canvas;
  const image = boxEditor.image;
  if (!canvas || !image || !state.boxes) return;
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext("2d");
  context.drawImage(image, 0, 0);
  for (const [name, box] of Object.entries(state.boxes.boxes)) {
    const [x0, y0, x1, y1] = box;
    context.lineWidth = Math.max(2, image.width / 400);
    context.strokeStyle = BOX_COLOURS[name] || "#fff";
    context.setLineDash(name === state.boxName ? [] : [8, 6]);
    context.strokeRect(x0, y0, x1 - x0, y1 - y0);
    context.fillStyle = BOX_COLOURS[name] || "#fff";
    const size = Math.max(14, image.width / 60);
    context.font = `600 ${size}px system-ui`;
    const corner = {sync_roi: [x0 + 6, y0 + size + 4, "start"], video_crop: [x1 - 6, y0 + size + 4, "end"],
      table: [x0 + 6, y1 - 8, "start"]}[name] || [x0 + 6, y0 + size + 4, "start"];
    context.textAlign = corner[2];
    context.fillText(name, corner[0], corner[1]);
    context.textAlign = "start";
  }
}

function canvasPoint(event) {
  const canvas = boxEditor.canvas;
  const rect = canvas.getBoundingClientRect();
  return [Math.round((event.clientX - rect.left) * canvas.width / rect.width), Math.round((event.clientY - rect.top) * canvas.height / rect.height)];
}

async function saveBoxes() {
  const result = await act(() => api("/api/setup/boxes", {method: "POST", body: {boxes: state.boxes.boxes}}), "Boxes saved");
  if (result) { state.boxStatus = `Saved to ${result.saved}.`; const s = $("#box-status"); if (s) s.textContent = state.boxStatus; }
}

// ---------------------------------------------------------------- system tab
function loadSystem() {
  renderServices();
  loadDoctor();
  loadSetup();
  loadSystemSetup();
}

async function loadSystemSetup() {
  let info;
  try { info = await api("/api/system"); } catch (error) { $("#system-setup").replaceChildren(el("p", {class: "hint", text: error.message})); return; }
  const output = state.setupOutput;
  $("#system-setup").replaceChildren(
    info.needs.length ? el("div", {},
      el("p", {text: "Missing on this PC (one-time, requires an administrator password):"}),
      el("ul", {class: "checklist"}, info.needs.map((n) => el("li", {class: "block"}, el("span", {class: "mark", text: "fail"}), el("div", {text: n.label})))),
      el("div", {class: "row"},
        el("button", {type: "button", text: "Install (password)", onclick: () => runSystemSetup({})}),
        el("button", {type: "button", class: "link", text: "Preview actions", onclick: () => runSystemSetup({dry_run: true})})),
      info.can_ask_password ? "" : el("p", {class: "hint", text: `No password dialog from here: on the PC, run ${info.command}`}))
      : el("div", {class: "message ok", text: "Camera access rule and system packages are installed."}),
    el("div", {class: "row", style: "margin-top:12px"},
      info.shortcut ? tag("desktop launcher installed", "ok")
        : el("button", {type: "button", class: "secondary", text: "Install desktop launcher", onclick: async () => {
          await act(() => api("/api/system/shortcut", {method: "POST"}), "Desktop icon added"); loadSystemSetup(); }}),
      el("button", {type: "button", class: "link", text: "Re-run installer", onclick: () =>
        act(() => api("/api/system/installer", {method: "POST"}), "The installer window opens on this PC")})),
    output ? el("div", {class: `message ${output.ok ? "ok" : "bad"}`, style: "margin-top:12px"},
      el("strong", {text: output.ok ? "Done" : output.detail}),
      output.output ? el("pre", {class: "log", text: output.output}) : "",
      !output.ok && output.command ? el("div", {class: "hint", text: `Or run on the PC: ${output.command}`}) : "") : "");
}

async function runSystemSetup(body) {
  toast(body.dry_run ? "Previewing" : "Waiting for the administrator password dialog");
  state.setupOutput = await act(() => api("/api/system/setup", {method: "POST", body}));
  loadSystemSetup();
  loadDoctor();
  if (state.tab === "bench") loadNetwork();
  return state.setupOutput;
}

const FIX_ACTIONS = {
  system_setup: ["Install (password)", () => runSystemSetup({})],
  power: ["Set performance mode", () => act(() => api("/api/bench/power", {method: "POST"}), "Performance mode on").then(loadDoctor)],
  calibration: ["Download it", () => act(() => api("/api/system/calibration", {method: "POST"}), "Calibration downloaded").then(loadDoctor)],
  installer: ["Run the installer", () => act(() => api("/api/system/installer", {method: "POST"}), "The installer window opens on this PC")],
  shortcut: ["Add the icon", () => act(() => api("/api/system/shortcut", {method: "POST"}), "Desktop icon added")],
  bench: ["Open Bench", () => showTab("bench")],
};

function renderServices() {
  const services = state.app?.services || {};
  $("#services").replaceChildren(...Object.values(services).map((s) => el("div", {class: "service"},
    el("div", {class: "service-head"},
      el("strong", {text: s.title}), tag(s.state, s.state === "running" ? "ok" : s.state === "starting" ? "warn" : "bad"),
      el("span", {class: "hint", style: "margin:0", text: `port ${s.port}${s.restarts ? ` · restarted ${s.restarts}×` : ""}`}),
      el("span", {class: "spacer"}),
      el("button", {type: "button", class: "secondary", text: "Restart", onclick: () => act(() => api(`/api/services/${s.name}/restart`, {method: "POST"}), `${s.title} restarted`)}),
      el("a", {href: `/api/services/${s.name}/log`, target: "_blank", class: "hint", text: "log"})),
    s.state !== "running" && s.log_tail?.length ? el("pre", {class: "log", text: s.log_tail.join("\n")}) : "")));
}

async function loadDoctor() {
  const box = $("#doctor");
  box.replaceChildren(el("p", {class: "hint", text: "Checking…"}));
  let checks;
  try { checks = await api("/api/doctor"); } catch (error) { box.replaceChildren(el("p", {class: "hint", text: error.message})); return; }
  box.replaceChildren(...checks.map((c) => el("div", {class: `doc doc-${c.status}`},
    el("span", {class: "doc-mark mark", text: c.status}),
    el("div", {}, el("div", {}, el("strong", {text: c.name}), " ", el("span", {class: "hint", text: c.detail})),
      c.status !== "ok" && c.fix ? el("div", {class: "doc-fix", text: c.fix}) : "",
      c.status !== "ok" && FIX_ACTIONS[c.action] ? el("button", {type: "button", class: "secondary", style: "margin-top:4px;padding:3px 10px",
        text: FIX_ACTIONS[c.action][0], onclick: FIX_ACTIONS[c.action][1]}) : ""))));
}

async function loadSetup() {
  let s;
  try { s = await api("/api/setup"); } catch (error) { $("#setup-info").replaceChildren(el("div", {class: "hint", text: `Unavailable while the camera service is down (${error.message}).`})); return; }
  const rows = [["Rig file", s.rig_file], ["Raw data", s.data_raw], ["Datasets", s.data_lerobot],
    ["Arms", s.arms ?? "auto (from the teleop run)"], ["Tasks", s.tasks.join(", ") || "—"],
    ["LeRobot env", `${s.lerobot_python.path}${s.lerobot_python.exists ? "" : " (missing)"}`],
    ["GPU depth env", `${s.depth_python.path}${s.depth_python.exists ? "" : " (missing)"}`]];
  const cams = s.cameras.flatMap((c) => [el("h4", {text: `Camera ${c.name}${c.primary ? " (primary)" : ""}`}),
    ...[["Model", `${c.info?.model || "?"} ${c.info?.serial || ""}`], ["Mode", `${c.mode} · ${c.fps_nominal} fps`],
      ["USB", c.usb_mbps ? `${c.usb_mbps} Mbps` : "—"], ["State", c.state]].flatMap(([k, v]) => [el("div", {class: "k", text: k}), el("div", {class: "v", text: v})])]);
  $("#setup-info").replaceChildren(...rows.flatMap(([k, v]) => [el("div", {class: "k", text: k}), el("div", {class: "v", text: v})]), ...cams);
}

// ---------------------------------------------------------------- episodes tab
async function loadSessions() {
  try { state.sessions = await api("/api/sessions"); } catch (error) { toast(error.message, true); return; }
  const list = $("#session-list");
  if (!state.sessions.length) { list.replaceChildren(el("li", {class: "empty", text: "No recordings yet."})); return; }
  if (!state.session || !state.sessions.some((x) => x.path === state.session)) state.session = state.sessions[0].path;
  list.replaceChildren(...state.sessions.map((x) => el("li", {
    class: x.path === state.session ? "active" : "",
    onclick: () => { state.session = x.path; state.episode = null; loadSessions(); },
  }, el("div", {}, el("strong", {text: x.date}), " ", x.name.replace("session_", "")),
     el("div", {class: "meta", text: `${x.episodes} episode${x.episodes === 1 ? "" : "s"} · ${x.minutes} min · ${x.passed} pass · ${x.accepted} accepted` +
       (x.processed < x.episodes ? ` · ${x.episodes - x.processed} not checked` : "")}))));
  loadEpisodes();
}

async function loadEpisodes() {
  if (!state.session) return;
  $("#episodes-title").textContent = `Episodes · ${state.session}`;
  try { state.episodes = await api(`/api/episodes?session=${encodeURIComponent(state.session)}`); } catch (error) { toast(error.message, true); return; }
  renderEpisodes();
  if (state.episode) loadEpisode(state.episode);
}

function renderEpisodes() {
  const body = $("#episode-rows");
  if (!state.episodes.length) { body.replaceChildren(el("tr", {}, el("td", {colspan: 10, class: "empty", text: "No episodes."}))); return; }
  body.replaceChildren(...state.episodes.map((row) => {
    const name = row.episode.split("/").pop();
    const sync = row.sync_ms === "" ? "—" : `${Number(row.sync_ms) > 0 ? "+" : ""}${row.sync_ms} ms`;
    return el("tr", {class: `clickable ${state.episode === row.episode ? "active" : ""}`, onclick: () => loadEpisode(row.episode)},
      el("td", {}, row.files?.thumbs ? el("img", {class: "thumb", src: fileUrl(row.files.thumbs), alt: "", loading: "lazy"}) : ""),
      el("td", {class: "num"}, name, row.status && row.status !== "complete" ? el("div", {}, badge(row.status)) : ""),
      el("td", {text: row.task || "—"}),
      el("td", {}, row.outcome ? badge(row.outcome) : "—"),
      el("td", {class: "num", text: row.duration_s ? `${Math.round(row.duration_s)} s` : "—"}),
      el("td", {class: "num", text: row.camera_drop_pct === "" ? "—" : `${row.camera_drop_pct}%`}),
      el("td", {class: "num", text: sync}),
      el("td", {}, badge(row.qc === "not processed" ? "not checked" : row.qc, row.qc === "not processed" ? "neutral" : row.qc)),
      el("td", {}, badge(row.review)),
      el("td", {}, el("div", {class: "row-actions"},
        el("button", {class: "good", type: "button", text: "Accept", onclick: (e) => { e.stopPropagation(); review(row.episode, {status: "accepted"}); }}),
        el("button", {class: "bad", type: "button", text: "Reject", onclick: (e) => { e.stopPropagation(); review(row.episode, {status: "rejected"}); }}))));
  }));
}

async function loadEpisode(path) {
  state.episode = path;
  $$("#episode-rows tr").forEach((tr, i) => tr.classList.toggle("active", state.episodes[i]?.episode === path));
  let detail;
  try { detail = await api(`/api/episode?path=${encodeURIComponent(path)}`); } catch (error) { toast(error.message, true); return; }
  const box = $("#episode-detail");
  box.hidden = false;
  const media = detail.files.preview
    ? el("video", {src: fileUrl(detail.files.preview), controls: true, muted: true, playsinline: true, preload: "metadata"})
    : detail.files.thumbs ? el("img", {src: fileUrl(detail.files.thumbs), alt: "thumbnails"}) : el("p", {class: "empty", text: "Check the session to get previews."});
  const qc = detail.qc;
  const checks = qc ? el("table", {class: "checks"}, qc.checks.map((c) => el("tr", {},
    el("td", {}, tag(c.ok ? "ok" : "fail", c.ok ? "ok" : "bad")),
    el("td", {text: c.name}), el("td", {text: String(c.value)}), el("td", {class: "hint", text: c.limit})))) :
    el("p", {class: "hint", text: "Not checked yet."});
  const labels = detail.labels || {};
  const form = el("form", {class: "labels-form", onsubmit: (e) => {
    e.preventDefault();
    const data = new FormData(e.target);
    review(path, {task: data.get("task"), outcome: data.get("outcome") || null, note: data.get("note")});
  }},
    el("label", {class: "field wide"}, el("span", {text: "Task"}), el("input", {name: "task", value: labels.task || ""})),
    el("label", {class: "field"}, el("span", {text: "Outcome"}), el("select", {name: "outcome"},
      ["", "success", "failure", "unsure"].map((o) => el("option", {value: o, text: o || "—", selected: (labels.outcome || "") === o})))),
    el("label", {class: "field"}, el("span", {text: "Operator"}), el("input", {value: labels.operator || "", disabled: true})),
    el("label", {class: "field wide"}, el("span", {text: "Note"}), el("input", {name: "note", value: labels.note || ""})),
    el("div", {class: "wide toolbar-actions"},
      el("button", {type: "submit", text: "Save labels"}),
      el("button", {type: "button", class: "good", text: "Accept", onclick: () => review(path, {status: "accepted"})}),
      el("button", {type: "button", class: "bad", text: "Reject", onclick: () => review(path, {status: "rejected"})}),
      el("button", {type: "button", class: "secondary", text: "Reset review", onclick: () => review(path, {status: "unreviewed"})})));
  box.replaceChildren(
    el("div", {class: "panel-title"}, el("span", {}, path, " ", badge(detail.row.qc === "not processed" ? "not checked" : detail.row.qc,
      detail.row.qc === "not processed" ? "neutral" : detail.row.qc), " ", badge(detail.row.review)),
      el("button", {class: "link", type: "button", text: "Close", onclick: () => { box.hidden = true; state.episode = null; renderEpisodes(); }})),
    el("div", {class: "detail-grid"},
      el("div", {}, media, detail.files.sync_plot ? el("img", {src: fileUrl(detail.files.sync_plot), alt: "sync plot", style: "margin-top:12px"}) : ""),
      el("div", {}, checks, form)));
}

async function review(path, changes) {
  try {
    await api("/api/review", {method: "POST", body: {path, ...changes}});
    toast(changes.status ? `${changes.status}` : "Labels saved");
    await loadEpisodes();
    if (state.episode === path) loadEpisode(path);
  } catch (error) { toast(error.message, true); }
}

async function processSession() {
  if (!state.session) return;
  await act(() => api("/api/jobs", {method: "POST", body: {kind: "process", paths: [state.session],
    depth: $("#process-depth").value, force: $("#process-force").checked}}), "Checking started (see Jobs)");
  loadJobs();
}

// ---------------------------------------------------------------- jobs tab
async function loadJobs() {
  try { state.jobs = await api("/api/jobs"); } catch (_) { return; }
  const active = state.jobs.filter((j) => j.status === "running" || j.status === "queued").length;
  $("#jobs-count").hidden = !active;
  $("#jobs-count").textContent = active;
  const list = $("#job-list");
  if (!state.jobs.length) { list.replaceChildren(el("p", {class: "empty", text: "No jobs yet."})); return; }
  list.replaceChildren(...state.jobs.map((job) => el("div", {class: "job"},
    el("div", {class: "job-head"},
      el("div", {}, el("span", {class: "job-title", text: `#${job.id} ${job.label}`}), " ",
        badge(job.paused ? "paused" : job.status, job.paused ? "needs_review" : job.status)),
      (job.status === "running" || job.status === "queued")
        ? el("button", {class: "secondary", type: "button", text: "Cancel", onclick: () => cancelJob(job.id)}) : ""),
    job.log.length ? el("pre", {text: job.log.join("\n")}) : "")));
}

async function cancelJob(id) { await act(() => api(`/api/jobs/${id}/cancel`, {method: "POST"})); loadJobs(); }

async function loadExportSessions() {
  try { state.sessions = await api("/api/sessions"); } catch (_) { return; }
  const box = $("#export-sessions");
  const withAccepted = state.sessions.filter((x) => x.accepted > 0);
  if (!withAccepted.length) { box.replaceChildren(el("p", {class: "hint", text: "No accepted episodes yet (Episodes tab)."})); return; }
  box.replaceChildren(...withAccepted.map((x) => el("label", {class: "check"},
    el("input", {type: "checkbox", name: "paths", value: x.path, checked: true}),
    `${x.date} ${x.name.replace("session_", "")} · ${x.accepted} accepted`)));
}

async function startExport(event) {
  event.preventDefault();
  const data = new FormData(event.target);
  const paths = data.getAll("paths");
  if (!paths.length) { toast("Pick at least one session", true); return; }
  await act(() => api("/api/jobs", {method: "POST", body: {kind: "export", paths, repo_id: data.get("repo_id"),
    image_size: data.get("image_size"), eyes: data.get("eyes"), overwrite: data.get("overwrite") === "on"}}), "Export started");
  loadJobs();
}

// ---------------------------------------------------------------- start
function bind() {
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $$("#run-kind button").forEach((b) => b.addEventListener("click", () => {
    if (runActive()) return;
    state.runKind = b.dataset.kind;
    renderArms();
    loadChecks(true);
  }));
  $("#global-stop").addEventListener("click", stopArms);
  $("#first-run-go").addEventListener("click", () => showTab("bench"));
  $("#record-button").addEventListener("click", toggleRecording);
  $("#task-select").addEventListener("change", () => { $("#task-custom").hidden = $("#task-select").value !== OTHER; saveDefaults(); });
  $("#task-custom").addEventListener("change", saveDefaults);
  $("#operator").addEventListener("change", saveDefaults);
  $$("#outcome [data-outcome]").forEach((b) => b.addEventListener("click", () => setOutcome(b.dataset.outcome)));
  $("#outcome-save").addEventListener("click", () => act(() => api("/record/label", {method: "POST", body: {note: $("#outcome-note").value}}), "Note saved"));
  $("#sessions-refresh").addEventListener("click", loadSessions);
  $("#process-session").addEventListener("click", processSession);
  $("#export-form").addEventListener("submit", startExport);
  $("#doctor-run").addEventListener("click", loadDoctor);
  $("#quit-app").addEventListener("click", async () => {
    if (!confirm("Quit the collection app? Recording and the headset page stop too.")) return;
    const result = await act(() => api("/api/app/quit", {method: "POST"}));
    if (result) document.body.replaceChildren(el("main", {}, el("div", {class: "panel", style: "max-width:520px;margin:40px auto"},
      el("div", {class: "panel-title", text: "The collection app has stopped"}), el("p", {text: "Open it again from the desktop icon."}))));
  });
}

bind();
poll().then(() => { if (state.app && !state.app.bench?.complete) showTab("bench"); else loadChecks(true); });
setInterval(poll, 1000);
setInterval(refreshPreview, 250);
setInterval(() => { if (state.tab === "collect" && !runActive()) loadChecks(); }, 2500);
setInterval(() => { if (state.tab === "jobs" || state.jobs.some((j) => j.status === "running")) loadJobs(); }, 2000);
setInterval(() => { if (camera()?.recording) $("#rec-time").textContent = clock(recElapsed()); }, 250);
setInterval(() => { if (state.tab === "bench" && Date.now() - state.networkAt > 15000 && !state.identify) loadNetwork(); }, 5000);
