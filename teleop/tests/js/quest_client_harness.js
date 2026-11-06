"use strict";
// Runs the real Quest page script (client.js) inside a Node vm with fake
// browser, WebXR, and WebSocket objects, replays controller frames, and prints
// a JSON summary for pytest. No network, headset, or robot is involved.
//
// Usage: node quest_client_harness.js <client.js> '<scenario json>'
// Scenario: {query, hands, arm_calibration, local_storage, steps: [
//   {frames, left: {grip, trigger}, right: {grip, trigger}}]}

const fs = require("fs");
const vm = require("vm");

const [clientPath, scenarioJson] = process.argv.slice(2);
const scenario = JSON.parse(scenarioJson);
const FRAME_MS = 1000 / 90;

function fakeElement(id) {
  const listeners = {};
  return {
    id,
    textContent: "",
    value: "",
    disabled: false,
    hidden: false,
    checked: false,
    options: [],
    listeners,
    classList: {toggle() {}, add() {}, remove() {}},
    addEventListener(type, fn) {
      (listeners[type] = listeners[type] || []).push(fn);
    },
    replaceChildren() {},
    add() {},
    getContext() {
      return fakeGl;
    },
  };
}

const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, fakeElement(id));
    return elements.get(id);
  },
};
const fakeGl = {
  FRAMEBUFFER: 0,
  COLOR_BUFFER_BIT: 0,
  DEPTH_BUFFER_BIT: 0,
  makeXRCompatible: async () => {},
  bindFramebuffer() {},
  clearColor() {},
  clear() {},
};

const storage = new Map(Object.entries(scenario.local_storage || {}));
const localStorage = {
  getItem: (key) => (storage.has(key) ? storage.get(key) : null),
  setItem: (key, value) => storage.set(key, String(value)),
};

const sent = [];
class FakeWebSocket {
  constructor(url) {
    this.url = url;
    this.readyState = FakeWebSocket.OPEN;
    this.bufferedAmount = 0;
  }
  send(data) {
    sent.push(JSON.parse(data));
  }
  close() {}
}
FakeWebSocket.OPEN = 1;

const buttons = {
  left: {grip: 0, trigger: 0},
  right: {grip: 0, trigger: 0},
};
const pose = {
  position: {x: 0.1, y: 1.0, z: -0.3},
  orientation: {x: 0, y: 0, z: 0, w: 1},
};
let frameCallback = null;
const session = {
  inputSources: (scenario.hands || ["right"]).map((hand) => ({
    handedness: hand,
    gripSpace: {hand},
    get gamepad() {
      return {
        buttons: [{value: buttons[hand].trigger}, {value: buttons[hand].grip}],
        hapticActuators: [],
      };
    },
  })),
  renderState: {baseLayer: {framebuffer: null}},
  requestAnimationFrame(callback) {
    frameCallback = callback;
    return 1;
  },
  updateRenderState() {},
  requestReferenceSpace: async () => ({}),
  addEventListener() {},
  end() {},
};
const frame = {
  getViewerPose: () => ({transform: pose}),
  getPose: () => ({transform: pose, emulatedPosition: false}),
};

class QuestCameraView {
  canCycle() { return false; }
  cycle() {}
  configure() {}
  initializeWebGL() {}
  handleControllerButtons() {}
  render() {}
}

const sandbox = {
  document,
  localStorage,
  location: {
    href: `http://localhost:8443/${scenario.query || ""}`,
    protocol: "http:",
    hostname: "localhost",
    host: "localhost:8443",
  },
  URL,
  WebSocket: FakeWebSocket,
  navigator: {
    xr: {
      isSessionSupported: async () => true,
      requestSession: async () => session,
    },
  },
  fetch: async () => {
    throw new Error("camera service not running in the harness");
  },
  performance,
  setTimeout,
  clearTimeout,
  console,
  XRWebGLLayer: class {},
  Option: class {},
  window: {
    WidowXAICameraView: {
      CAMERA_ROLES: ["scene", "left_wrist", "right_wrist"],
      ROLE_LABELS: {scene: "Scene", left_wrist: "Left wrist", right_wrist: "Right wrist"},
      QuestCameraView,
    },
  },
};

async function main() {
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(clientPath, "utf8"), sandbox, {filename: "client.js"});

  if (scenario.arm_calibration) {
    for (const listener of document.getElementById("calibrate").listeners.click || []) listener();
  }
  for (const listener of document.getElementById("enter").listeners.click || []) await listener();
  if (!frameCallback) throw new Error("the page never requested an XR frame");

  let frameTime = 1000;
  const steps = [];
  for (const step of scenario.steps) {
    Object.assign(buttons.left, step.left || {});
    Object.assign(buttons.right, step.right || {});
    let silentFrames = 0;
    let silentRun = 0;
    let longestSilentMs = 0;
    for (let i = 0; i < step.frames; i += 1) {
      const before = sent.length;
      frameTime += FRAME_MS;
      frameCallback(frameTime, frame);
      const produced = sent.slice(before).some(
        (packet) => packet.type === "pose" || packet.type === "bimanual_pose",
      );
      if (produced) {
        silentRun = 0;
      } else {
        silentFrames += 1;
        silentRun += 1;
        longestSilentMs = Math.max(longestSilentMs, silentRun * FRAME_MS);
      }
    }
    steps.push({frames: step.frames, silent_frames: silentFrames, longest_silent_ms: longestSilentMs});
  }

  const poses = sent.filter((packet) => packet.type === "pose" || packet.type === "bimanual_pose");
  const last = poses[poses.length - 1] || null;
  process.stdout.write(JSON.stringify({
    steps,
    pose_packets: poses.length,
    packet_types: [...new Set(poses.map((packet) => packet.type))],
    last_packet: last,
    status: document.getElementById("status").textContent,
    calibration_text: document.getElementById("calibration").textContent,
  }));
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exit(1);
});
