"use strict";

const statusText = document.getElementById("status");
const statusDot = document.getElementById("dot");
const enterButton = document.getElementById("enter");
const exitButton = document.getElementById("exit");
const calibrateButton = document.getElementById("calibrate");
const capturedText = document.getElementById("captured");
const sentText = document.getElementById("sent");
const coalescedText = document.getElementById("coalesced");
const calibrationText = document.getElementById("calibration");
const controllerDescription = document.getElementById("controller-description");
const handSelect = document.getElementById("hand-select");
const mappingModeSelect = document.getElementById("mapping-mode-select");
const inputModeSelect = document.getElementById("input-mode-select");
const leftMappingModeSelect = document.getElementById("left-mapping-mode-select");
const rightMappingModeSelect = document.getElementById("right-mapping-mode-select");
const singleArmFields = document.getElementById("single-arm-fields");
const bimanualFields = document.getElementById("bimanual-fields");
const applyInputButton = document.getElementById("apply-input");
const inputSelectionText = document.getElementById("input-selection");
const cycleCameraViewButton = document.getElementById("cycle-camera-view");
const cameraStatusText = document.getElementById("camera-status");
const cameraViewModeText = document.getElementById("camera-view-mode");
const cameraSetupStatusText = document.getElementById("camera-setup-status");
const applyCameraSetupButton = document.getElementById("apply-camera-setup");
const refreshCamerasButton = document.getElementById("refresh-cameras");
const canvas = document.getElementById("xr-canvas");
const CAMERA_ROLES = window.WidowXAICameraView.CAMERA_ROLES;
const CAMERA_ROLE_LABELS = window.WidowXAICameraView.ROLE_LABELS;
const CAMERA_SERVICE_URL = `${location.protocol}//${location.hostname}:8444`;
const cameraControls = {
  scene: {
    select: document.getElementById("scene-camera-select"),
    enabled: document.getElementById("scene-camera-enabled"),
    card: document.getElementById("scene-preview-card"),
    caption: document.getElementById("scene-preview-caption"),
  },
  left_wrist: {
    select: document.getElementById("left-wrist-camera-select"),
    enabled: document.getElementById("left-wrist-camera-enabled"),
    card: document.getElementById("left-wrist-preview-card"),
    caption: document.getElementById("left-wrist-preview-caption"),
  },
  right_wrist: {
    select: document.getElementById("right-wrist-camera-select"),
    enabled: document.getElementById("right-wrist-camera-enabled"),
    card: document.getElementById("right-wrist-preview-card"),
    caption: document.getElementById("right-wrist-preview-caption"),
  },
};
let cameraDevices = [];
const cameraView = new window.WidowXAICameraView.QuestCameraView({
  previews: {
    scene: document.getElementById("scene-preview"),
    left_wrist: document.getElementById("left-wrist-preview"),
    right_wrist: document.getElementById("right-wrist-preview"),
  },
  statusText: cameraStatusText,
  viewModeText: cameraViewModeText,
  configurationChanged: () => {
    cycleCameraViewButton.disabled = !cameraView.canCycle();
  },
});

const HANDS = ["left", "right"];
// Touch Plus gamepad button indices: 0 trigger, 1 grip.
const TRIGGER_BUTTON = 0;
const GRIP_BUTTON = 1;

const INPUT_SELECTION_STORAGE_KEY = "widowxai.inputSelection";
const query = new URL(location.href).searchParams;

function savedInputSelection() {
  try {
    const value = JSON.parse(localStorage.getItem(INPUT_SELECTION_STORAGE_KEY) || "null");
    return value && typeof value === "object" ? value : {};
  } catch (_) {
    return {};
  }
}

const savedSelection = savedInputSelection();
const requestedHand = query.get("hand") || savedSelection.hand || "left";
const requestedMappingMode = query.get("mode") || savedSelection.mappingMode || "real";
const requestedInputMode = query.get("input") || savedSelection.inputMode || "single";
let selectedHand = requestedHand === "right" ? "right" : "left";
let selectedMappingMode = requestedMappingMode === "mirror" ? "mirror" : "real";
// "single" keeps the validated one-arm packet. "bimanual" emits schema v2 with
// both controllers captured in the same XRFrame.
let selectedInputMode = requestedInputMode === "bimanual" ? "bimanual" : "single";
let handMappingModes = {
  left: (savedSelection.handMappingModes || {}).left === "mirror" ? "mirror" : "real",
  right: (savedSelection.handMappingModes || {}).right === "mirror" ? "mirror" : "real",
};
let wristOffsetStorageKey = `widowxai.${selectedHand}WristOffset`;
let socket = null;
let session = null;
let sessionMode = "immersive-vr";
let referenceSpace = null;
let gl = null;
let latestPacket = null;
let sequence = 0;
let captured = 0;
let sent = 0;
let coalesced = 0;
let selectedInputSource = null;
// Lock the operator's intended forward direction when passthrough starts.
// Otherwise placing the headset down can rotate the task frame on re-clutch.
let operatorHead = null;
let wristOffset = JSON.parse(localStorage.getItem(wristOffsetStorageKey) || "null");
// Wrist calibration starts only after the operator presses "Calibrate wrist".
// Its right-hand chord (grip + trigger) is also clutch + close-gripper, so an
// automatically armed capture paused pose streaming for 5 s mid-teleop.
let calibrationArmed = false;
let calibrationStartedAt = null;
let calibrationSamples = [];
// Bimanual mode needs both hands' pivots live at once, so it keeps its own
// per-hand store rather than the single-arm page's one active offset.
let wristOffsets = {left: null, right: null};
let calibratingHand = null;

function loadWristOffsets() {
  for (const hand of HANDS) {
    wristOffsets[hand] = JSON.parse(localStorage.getItem(`widowxai.${hand}WristOffset`) || "null");
  }
}

function offsetLabel(offset) {
  return offset ? `${offset.map((v) => v.toFixed(3)).join(", ")} m` : "Not calibrated";
}

function mappingModeLabel(mode = selectedMappingMode) {
  return mode === "mirror"
    ? "Mirrored (left/right, front/back, screw, and nod-no flipped)"
    : "Behind / Parallel (matched motion)";
}

function isBimanual() {
  return selectedInputMode === "bimanual";
}

function refreshInputSelection() {
  wristOffsetStorageKey = `widowxai.${selectedHand}WristOffset`;
  wristOffset = JSON.parse(localStorage.getItem(wristOffsetStorageKey) || "null");
  loadWristOffsets();
  calibrationArmed = false;
  calibrationStartedAt = null;
  calibrationSamples = [];
  calibratingHand = null;
  handSelect.value = selectedHand;
  mappingModeSelect.value = selectedMappingMode;
  inputModeSelect.value = selectedInputMode;
  leftMappingModeSelect.value = handMappingModes.left;
  rightMappingModeSelect.value = handMappingModes.right;
  singleArmFields.hidden = isBimanual();
  bimanualFields.hidden = !isBimanual();

  if (isBimanual()) {
    calibrationText.textContent = `left ${offsetLabel(wristOffsets.left)} · right ${offsetLabel(wristOffsets.right)}`;
    controllerDescription.textContent =
      "This page streams BOTH Meta Quest controllers in one timestamped frame. " +
      "Each grip clutches its own arm; each trigger drives that arm's gripper.";
    inputSelectionText.textContent =
      `Active: bimanual · left ${mappingModeLabel(handMappingModes.left)} · right ${mappingModeLabel(handMappingModes.right)}`;
  } else {
    calibrationText.textContent = offsetLabel(wristOffset);
    controllerDescription.textContent = `This page streams the ${selectedHand} Meta Quest controller using ${mappingModeLabel().toLowerCase()}. Grip is the clutch; trigger controls the gripper.`;
    inputSelectionText.textContent = `Active: ${selectedHand} controller · ${mappingModeLabel()}`;
  }
}

function setInputControlsDisabled(disabled) {
  handSelect.disabled = disabled;
  mappingModeSelect.disabled = disabled;
  inputModeSelect.disabled = disabled;
  leftMappingModeSelect.disabled = disabled;
  rightMappingModeSelect.disabled = disabled;
  applyInputButton.disabled = disabled;
}

function setCameraControlsDisabled(disabled) {
  for (const role of CAMERA_ROLES) {
    cameraControls[role].select.disabled = disabled;
    cameraControls[role].enabled.disabled = disabled || !cameraControls[role].select.value;
  }
  applyCameraSetupButton.disabled = disabled;
  refreshCamerasButton.disabled = disabled;
  cycleCameraViewButton.disabled = disabled || !cameraView.canCycle();
}

function updateCameraChoiceAvailability() {
  const selectedByRole = Object.fromEntries(
    CAMERA_ROLES.map((role) => [role, cameraControls[role].select.value]),
  );
  for (const role of CAMERA_ROLES) {
    const control = cameraControls[role];
    for (const option of Array.from(control.select.options)) {
      if (!option.value) continue;
      option.disabled = CAMERA_ROLES.some(
        (otherRole) => otherRole !== role && selectedByRole[otherRole] === option.value,
      );
    }
    if (!control.select.value) control.enabled.checked = false;
    control.enabled.disabled = Boolean(session) || !control.select.value;
  }
}

function cameraLabel(device) {
  return `${device.product} · ${device.serial}`;
}

function renderCameraConfiguration(state) {
  cameraDevices = Array.isArray(state.devices) ? state.devices : [];
  const deviceBySerial = new Map(cameraDevices.map((device) => [device.serial, device]));
  for (const role of CAMERA_ROLES) {
    const control = cameraControls[role];
    const item = state.roles?.[role] || {serial: null, enabled: false};
    control.select.replaceChildren(new Option("No camera", ""));
    for (const device of cameraDevices) {
      control.select.add(new Option(cameraLabel(device), device.serial));
    }
    if (item.serial && !deviceBySerial.has(item.serial)) {
      control.select.add(new Option(`Disconnected · ${item.serial}`, item.serial));
    }
    control.select.value = item.serial || "";
    control.enabled.checked = Boolean(item.enabled && item.serial);
    const selectedDevice = item.serial ? deviceBySerial.get(item.serial) : null;
    control.card.hidden = !item.serial;
    control.caption.textContent = item.serial
      ? `${CAMERA_ROLE_LABELS[role]} · ${selectedDevice?.product || "camera disconnected"}`
      : `${CAMERA_ROLE_LABELS[role]} · no camera`;
  }
  updateCameraChoiceAvailability();
  cameraView.configure(state.roles || {}, cameraDevices);
  setCameraControlsDisabled(Boolean(session));
  const connected = cameraDevices.length;
  cameraSetupStatusText.textContent = `${connected} compatible camera${connected === 1 ? "" : "s"} found · assignments saved on this PC`;
}

async function refreshCameraConfiguration() {
  cameraSetupStatusText.textContent = "Discovering cameras on this machine…";
  try {
    const response = await fetch(`${CAMERA_SERVICE_URL}/configuration`, {cache: "no-store"});
    if (!response.ok) throw new Error(`camera service returned ${response.status}`);
    renderCameraConfiguration(await response.json());
  } catch (error) {
    cameraSetupStatusText.textContent = `Camera setup unavailable: ${error.message}`;
    cameraStatusText.textContent = "Waiting for camera service on USB port 8444";
    cameraStatusText.classList.remove("ok");
  }
}

async function applyCameraConfiguration() {
  if (session) {
    cameraSetupStatusText.textContent = "Exit passthrough before changing camera setup";
    return;
  }
  const roles = Object.fromEntries(CAMERA_ROLES.map((role) => {
    const control = cameraControls[role];
    const serial = control.select.value || null;
    return [role, {serial, enabled: Boolean(serial && control.enabled.checked)}];
  }));
  applyCameraSetupButton.disabled = true;
  cameraSetupStatusText.textContent = "Applying camera setup…";
  try {
    const response = await fetch(`${CAMERA_SERVICE_URL}/configuration`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({roles}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || `camera service returned ${response.status}`);
    renderCameraConfiguration(result);
    cameraSetupStatusText.textContent = "Camera setup applied and saved";
  } catch (error) {
    cameraSetupStatusText.textContent = `Unable to apply camera setup: ${error.message}`;
    applyCameraSetupButton.disabled = false;
  }
}

refreshInputSelection();

function setStatus(text, ok = false) {
  statusText.textContent = text;
  statusDot.classList.toggle("ok", ok);
}

function connectRelay() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${scheme}://${location.host}/ws`);
  socket.onopen = async () => {
    setStatus("Relay connected", true);
    let arSupported = false;
    let vrSupported = false;
    if (navigator.xr) {
      arSupported = await navigator.xr.isSessionSupported("immersive-ar");
      vrSupported = await navigator.xr.isSessionSupported("immersive-vr");
    }
    sessionMode = arSupported ? "immersive-ar" : "immersive-vr";
    enterButton.textContent = arSupported ? "Enter Passthrough" : "Enter VR";
    enterButton.disabled = !(arSupported || vrSupported);
    socket.send(JSON.stringify({
      type: "clock_ping",
      sequence: 0,
      client_send_monotonic_ms: performance.now(),
      client_send_epoch_ms: Date.now(),
    }));
  };
  socket.onclose = () => {
    setStatus("Relay disconnected; retrying…");
    enterButton.disabled = true;
    setTimeout(connectRelay, 750);
  };
  socket.onerror = () => setStatus("Relay connection error");
  socket.onmessage = (event) => {
    let message;
    try { message = JSON.parse(event.data); } catch (_) { return; }
    if (message.type === "haptic") {
      const requestedHand = message.hand === "left" || message.hand === "right" ? message.hand : null;
      const hapticSource = requestedHand && session
        ? Array.from(session.inputSources).find((source) => source.handedness === requestedHand)
        : selectedInputSource;
      const actuator = hapticSource?.gamepad?.hapticActuators?.[0];
      if (actuator) actuator.pulse(Math.max(0, Math.min(1, Number(message.intensity) || 0)), Number(message.duration_ms) || 40);
    }
    if (message.type === "calibration_prompt" && typeof message.text === "string") {
      setStatus(message.text, true);
    }
  };
}

function buttonValue(gamepad, index) {
  const button = gamepad?.buttons?.[index];
  return button ? Number(button.value || (button.pressed ? 1 : 0)) : 0;
}

function rotationMatrix(transform) {
  const {x, y, z, w} = transform.orientation;
  return [
    1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
    2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
    2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
  ];
}

function shiftedPosition(transform, offset) {
  const r = rotationMatrix(transform);
  return [
    transform.position.x + r[0]*offset[0] + r[1]*offset[1] + r[2]*offset[2],
    transform.position.y + r[3]*offset[0] + r[4]*offset[1] + r[5]*offset[2],
    transform.position.z + r[6]*offset[0] + r[7]*offset[1] + r[8]*offset[2],
  ];
}

function transformObject(transform, offset = [0, 0, 0]) {
  return {
    position: shiftedPosition(transform, offset),
    orientation_xyzw: [transform.orientation.x, transform.orientation.y, transform.orientation.z, transform.orientation.w],
  };
}

function sendLatestPacket() {
  if (!latestPacket || !socket || socket.readyState !== WebSocket.OPEN) return false;
  if (socket.bufferedAmount > 2048) return false;
  const packet = latestPacket;
  latestPacket = null;
  packet.send_monotonic_ms = performance.now();
  socket.send(JSON.stringify(packet));
  sent += 1;
  sentText.textContent = sent;
  return true;
}

function solve3x3(a, b) {
  const m = [
    [a[0], a[1], a[2], b[0]],
    [a[3], a[4], a[5], b[1]],
    [a[6], a[7], a[8], b[2]],
  ];
  for (let col = 0; col < 3; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < 3; row += 1) if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    if (Math.abs(m[pivot][col]) < 1e-9) throw new Error("insufficient wrist rotation");
    [m[col], m[pivot]] = [m[pivot], m[col]];
    const divisor = m[col][col];
    for (let j = col; j < 4; j += 1) m[col][j] /= divisor;
    for (let row = 0; row < 3; row += 1) {
      if (row === col) continue;
      const factor = m[row][col];
      for (let j = col; j < 4; j += 1) m[row][j] -= factor * m[col][j];
    }
  }
  return [m[0][3], m[1][3], m[2][3]];
}

function finishWristCalibration() {
  const n = calibrationSamples.length;
  if (n < 60) throw new Error("too few samples");
  const meanP = [0, 0, 0];
  const meanR = new Array(9).fill(0);
  for (const sample of calibrationSamples) {
    for (let i = 0; i < 3; i += 1) meanP[i] += sample.p[i] / n;
    for (let i = 0; i < 9; i += 1) meanR[i] += sample.r[i] / n;
  }
  const normal = new Array(9).fill(0);
  const rhs = [0, 0, 0];
  for (const sample of calibrationSamples) {
    const dp = sample.p.map((value, i) => value - meanP[i]);
    const dr = sample.r.map((value, i) => value - meanR[i]);
    for (let row = 0; row < 3; row += 1) {
      for (let col = 0; col < 3; col += 1) {
        let sum = 0;
        for (let k = 0; k < 3; k += 1) sum += dr[k*3 + row] * dr[k*3 + col];
        normal[row*3 + col] += sum;
      }
      for (let k = 0; k < 3; k += 1) rhs[row] += dr[k*3 + row] * dp[k];
    }
  }
  const solved = solve3x3(normal, rhs.map((value) => -value));
  const norm = Math.hypot(...solved);
  if (!Number.isFinite(norm) || norm > 0.20) throw new Error("offset failed sanity check");
  const hand = calibratingHand || selectedHand;
  localStorage.setItem(`widowxai.${hand}WristOffset`, JSON.stringify(solved));
  loadWristOffsets();
  if (isBimanual()) {
    calibrationText.textContent = `left ${offsetLabel(wristOffsets.left)} · right ${offsetLabel(wristOffsets.right)}`;
  } else {
    wristOffset = solved;
    calibrationText.textContent = offsetLabel(wristOffset);
  }
  calibrationArmed = false;
  calibrationStartedAt = null;
  calibrationSamples = [];
  calibratingHand = null;
  setStatus(`${hand} wrist calibrated; streaming`, true);
}

function failWristCalibration(error) {
  // Disarm so the next grip + trigger press drives the gripper again instead
  // of silently starting another 5 s capture.
  calibrationArmed = false;
  calibrationStartedAt = null;
  calibrationSamples = [];
  calibratingHand = null;
  setStatus(`Wrist calibration failed: ${error.message}; press Calibrate wrist to retry`);
}

function bimanualControllerBlock(frame, inputSource, hand) {
  // Absent source, absent gripSpace, and a null getPose all mean the same
  // thing to the robot: this controller is not tracked right now. Report it
  // explicitly so the coordinator holds that arm instead of replaying a stale
  // pose it cannot distinguish from a live one.
  if (!inputSource || !inputSource.gripSpace) {
    return {tracked: false, mapping_mode: handMappingModes[hand]};
  }
  const pose = frame.getPose(inputSource.gripSpace, referenceSpace);
  if (!pose || pose.emulatedPosition) {
    return {tracked: false, mapping_mode: handMappingModes[hand]};
  }
  return {
    tracked: true,
    mapping_mode: handMappingModes[hand],
    ...transformObject(pose.transform, wristOffsets[hand] || [0, 0, 0]),
    grip: buttonValue(inputSource.gamepad, GRIP_BUTTON),
    trigger: buttonValue(inputSource.gamepad, TRIGGER_BUTTON),
  };
}

function onBimanualFrame(frameTime, frame, viewerPose, inputSources) {
  // Per-hand wrist calibration uses that hand's own grip+trigger chord. The
  // single-arm both-grips chord would collide with the bimanual clutch.
  if (calibrationArmed && calibrationStartedAt === null) {
    for (const hand of HANDS) {
      const source = inputSources[hand];
      if (!source) continue;
      if (
        buttonValue(source.gamepad, GRIP_BUTTON) >= 0.7 &&
        buttonValue(source.gamepad, TRIGGER_BUTTON) >= 0.7
      ) {
        calibratingHand = hand;
        calibrationStartedAt = frameTime;
        calibrationSamples = [];
        setStatus(`Calibrating ${hand} wrist: rotate it while keeping its pivot still`, true);
        break;
      }
    }
  }
  if (calibrationStartedAt !== null) {
    const source = inputSources[calibratingHand];
    const pose = source && source.gripSpace ? frame.getPose(source.gripSpace, referenceSpace) : null;
    if (pose) {
      calibrationSamples.push({
        p: [pose.transform.position.x, pose.transform.position.y, pose.transform.position.z],
        r: rotationMatrix(pose.transform),
      });
    }
    if (frameTime - calibrationStartedAt >= 5000) {
      try { finishWristCalibration(); }
      catch (error) {
        failWristCalibration(error);
      }
    }
    return;
  }

  const left = bimanualControllerBlock(frame, inputSources.left, "left");
  const right = bimanualControllerBlock(frame, inputSources.right, "right");
  // Unlike the single-arm page this never drops the frame when one controller
  // is missing. One shared sequence number keeps the two arms aligned, and a
  // gap in it is what the relay-loss watchdog is allowed to react to.
  const nextPacket = {
    type: "bimanual_pose",
    schema_version: 2,
    input_mode: "bimanual",
    sequence: sequence++,
    capture_monotonic_ms: frameTime,
    capture_epoch_ms: performance.timeOrigin + frameTime,
    enqueue_monotonic_ms: performance.now(),
    send_monotonic_ms: 0,
    left: left,
    right: right,
    head: viewerPose ? transformObject(viewerPose.transform) : null,
    operator_head: operatorHead,
  };
  if (latestPacket !== null) coalesced += 1;
  latestPacket = nextPacket;
  captured += 1;
  sendLatestPacket();
  capturedText.textContent = captured;
  coalescedText.textContent = coalesced;
}

function onXRFrame(frameTime, frame) {
  session.requestAnimationFrame(onXRFrame);
  gl.bindFramebuffer(gl.FRAMEBUFFER, session.renderState.baseLayer.framebuffer);
  if (sessionMode === "immersive-ar") gl.clearColor(0.0, 0.0, 0.0, 0.0);
  else gl.clearColor(0.025, 0.055, 0.10, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const viewerPose = frame.getViewerPose(referenceSpace);
  if (operatorHead === null && viewerPose) {
    operatorHead = transformObject(viewerPose.transform);
    setStatus("Operator forward locked; streaming", true);
  }

  const leftInputSource = Array.from(session.inputSources).find((source) => source.handedness === "left" && source.gripSpace) || null;
  const rightInputSource = Array.from(session.inputSources).find((source) => source.handedness === "right" && source.gripSpace) || null;
  selectedInputSource = selectedHand === "right" ? rightInputSource : leftInputSource;
  cameraView.handleControllerButtons({left: leftInputSource, right: rightInputSource});
  cameraView.render(session, viewerPose);

  if (isBimanual()) {
    onBimanualFrame(frameTime, frame, viewerPose, {left: leftInputSource, right: rightInputSource});
    return;
  }

  const controllerPose = selectedInputSource ? frame.getPose(selectedInputSource.gripSpace, referenceSpace) : null;
  if (!controllerPose) return;
  const calibrationChord = selectedHand === "right"
    ? buttonValue(selectedInputSource?.gamepad, 1) >= 0.7 && buttonValue(selectedInputSource?.gamepad, 0) >= 0.7
    : buttonValue(leftInputSource?.gamepad, 1) >= 0.7 && buttonValue(rightInputSource?.gamepad, 1) >= 0.7;
  if (calibrationArmed && calibrationStartedAt === null && calibrationChord) {
    calibrationStartedAt = frameTime;
    calibrationSamples = [];
    setStatus(`Calibrating ${selectedHand} wrist: rotate it while keeping its pivot still`, true);
  }
  if (calibrationStartedAt !== null) {
    calibrationSamples.push({
      p: [controllerPose.transform.position.x, controllerPose.transform.position.y, controllerPose.transform.position.z],
      r: rotationMatrix(controllerPose.transform),
    });
    if (frameTime - calibrationStartedAt >= 5000) {
      try { finishWristCalibration(); }
      catch (error) {
        failWristCalibration(error);
      }
    }
    return;
  }
  const nextPacket = {
    type: "pose",
    schema_version: 1,
    selected_hand: selectedHand,
    mapping_mode: selectedMappingMode,
    sequence: sequence++,
    capture_monotonic_ms: frameTime,
    capture_epoch_ms: performance.timeOrigin + frameTime,
    enqueue_monotonic_ms: performance.now(),
    send_monotonic_ms: 0,
    [selectedHand]: {
      ...transformObject(controllerPose.transform, wristOffset || [0, 0, 0]),
      grip: buttonValue(selectedInputSource.gamepad, 1),
      trigger: buttonValue(selectedInputSource.gamepad, 0),
    },
    head: viewerPose ? transformObject(viewerPose.transform) : null,
    operator_head: operatorHead,
  };
  if (latestPacket !== null) coalesced += 1;
  latestPacket = nextPacket;
  captured += 1;
  // Send in the same WebXR callback that produced the pose. A separate 120 Hz
  // timer added an avoidable 0-8.3 ms phase wait before every packet.
  sendLatestPacket();
  capturedText.textContent = captured;
  coalescedText.textContent = coalesced;
}

enterButton.addEventListener("click", async () => {
  try {
    operatorHead = null;
    session = await navigator.xr.requestSession(sessionMode, { requiredFeatures: ["local-floor"] });
    gl = canvas.getContext("webgl", {
      xrCompatible: true,
      antialias: false,
      alpha: sessionMode === "immersive-ar",
      premultipliedAlpha: true,
    });
    await gl.makeXRCompatible();
    cameraView.initializeWebGL(gl);
    session.updateRenderState({
      baseLayer: new XRWebGLLayer(session, gl, { alpha: sessionMode === "immersive-ar" }),
    });
    referenceSpace = await session.requestReferenceSpace("local-floor");
    session.addEventListener("end", () => {
      session = null;
      latestPacket = null;
      operatorHead = null;
      enterButton.disabled = false;
      exitButton.disabled = true;
      setInputControlsDisabled(false);
      setCameraControlsDisabled(false);
      setStatus("Relay connected; VR stopped", true);
    });
    enterButton.disabled = true;
    exitButton.disabled = false;
    setInputControlsDisabled(true);
    setCameraControlsDisabled(true);
    setStatus(
      sessionMode === "immersive-ar"
        ? `Streaming ${selectedHand} · ${mappingModeLabel()} in passthrough`
        : `Streaming ${selectedHand} · ${mappingModeLabel()} in VR`,
      true,
    );
    session.requestAnimationFrame(onXRFrame);
  } catch (error) {
    setStatus(`Unable to enter VR: ${error.message}`);
  }
});

exitButton.addEventListener("click", () => session?.end());
cycleCameraViewButton.addEventListener("click", () => cameraView.cycle());
applyCameraSetupButton.addEventListener("click", applyCameraConfiguration);
refreshCamerasButton.addEventListener("click", refreshCameraConfiguration);
for (const role of CAMERA_ROLES) {
  cameraControls[role].select.addEventListener("change", updateCameraChoiceAvailability);
}
applyInputButton.addEventListener("click", () => {
  if (session) {
    setStatus("Exit VR before changing controller input");
    return;
  }
  selectedHand = handSelect.value === "right" ? "right" : "left";
  selectedMappingMode = mappingModeSelect.value === "mirror" ? "mirror" : "real";
  selectedInputMode = inputModeSelect.value === "bimanual" ? "bimanual" : "single";
  handMappingModes = {
    left: leftMappingModeSelect.value === "mirror" ? "mirror" : "real",
    right: rightMappingModeSelect.value === "mirror" ? "mirror" : "real",
  };
  localStorage.setItem(INPUT_SELECTION_STORAGE_KEY, JSON.stringify({
    hand: selectedHand,
    mappingMode: selectedMappingMode,
    inputMode: selectedInputMode,
    handMappingModes: handMappingModes,
  }));
  refreshInputSelection();
  setStatus(
    isBimanual()
      ? "Input set to bimanual (both controllers); enter passthrough"
      : `Input set to ${selectedHand} · ${mappingModeLabel()}; enter passthrough`,
    true,
  );
});
calibrateButton.addEventListener("click", () => {
  calibrationArmed = true;
  calibrationStartedAt = null;
  calibrationSamples = [];
  calibratingHand = null;
  if (isBimanual()) {
    const chord = "hold one controller's grip + trigger";
    calibrationText.textContent = `Armed for bimanual wrists: ${chord} in VR`;
    setStatus(`Enter VR, ${chord}, then rotate that wrist for 5 seconds; repeat for the other hand`, true);
    return;
  }
  const chord = selectedHand === "right" ? "hold right grip + right trigger" : "squeeze both grip buttons";
  calibrationText.textContent = `Armed for ${selectedHand} wrist: ${chord} in VR`;
  setStatus(`Enter VR, ${chord}, then rotate the ${selectedHand} wrist for 5 seconds`, true);
});
refreshCameraConfiguration();
connectRelay();
