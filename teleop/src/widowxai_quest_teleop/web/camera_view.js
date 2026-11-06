"use strict";

(function cameraViewModule() {
  const CAMERA_ROLES = ["scene", "left_wrist", "right_wrist"];
  const ROLE_LABELS = {
    scene: "Scene",
    left_wrist: "Left wrist",
    right_wrist: "Right wrist",
  };
  // Meta Quest Touch Plus uses xr-standard optional button slot 5 for B on the
  // right controller. Trigger/grip remain standardized at slots 0/1.
  const QUEST_B_BUTTON = 5;
  const GRIP_BUTTON = 1;
  const GRIP_RELEASED = 0.2;
  const CAMERA_PORT = 8444;
  const TRANSITION_DURATION_MS = 380;
  // Raise every overview, focus, and preview panel together so the live
  // passthrough work area remains visible below the camera UI.
  const PANEL_VERTICAL_OFFSET_M = 0.18;

  function buttonValue(gamepad, index) {
    const button = gamepad?.buttons?.[index];
    return button ? Number(button.value || (button.pressed ? 1 : 0)) : 0;
  }

  function cameraWebSocketUrl(role) {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    return `${scheme}://${location.hostname}:${CAMERA_PORT}/ws/${role}`;
  }

  function roundedRect(context, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    context.beginPath();
    context.moveTo(x + r, y);
    context.arcTo(x + width, y, x + width, y + height, r);
    context.arcTo(x + width, y + height, x, y + height, r);
    context.arcTo(x, y + height, x, y, r);
    context.arcTo(x, y, x + width, y, r);
    context.closePath();
  }

  class CameraFeed {
    constructor(role, previewCanvas, statusChanged) {
      this.role = role;
      this.label = ROLE_LABELS[role];
      this.previewCanvas = previewCanvas;
      this.statusChanged = statusChanged;
      this.canvas = document.createElement("canvas");
      this.canvas.width = 640;
      this.canvas.height = 480;
      this.context = this.canvas.getContext("2d", {alpha: false});
      this.socket = null;
      this.reconnectTimer = null;
      this.pendingBlob = null;
      this.decoding = false;
      this.connected = false;
      this.enabled = false;
      this.serial = null;
      this.lastFrameAt = 0;
      this.revision = 0;
      this.showingVideoLost = false;
      this.staleTimer = setInterval(() => {
        if (this.enabled && this.lastFrameAt && this.frameAgeMs() > 1000 && !this.showingVideoLost) {
          this.drawPlaceholder("VIDEO LOST");
        }
      }, 250);
      this.drawPlaceholder("NOT SELECTED", false);
    }

    connect() {
      if (!this.enabled || this.socket) return;
      clearTimeout(this.reconnectTimer);
      const socket = new WebSocket(cameraWebSocketUrl(this.role));
      socket.binaryType = "blob";
      this.socket = socket;
      socket.onopen = () => {
        this.connected = true;
        this.statusChanged();
      };
      socket.onmessage = (event) => {
        this.pendingBlob = event.data;
        this.decodeLatest();
      };
      socket.onclose = () => {
        if (this.socket === socket) this.socket = null;
        this.connected = false;
        this.statusChanged();
        if (!this.enabled) return;
        if (!this.lastFrameAt || performance.now() - this.lastFrameAt > 1000) {
          this.drawPlaceholder("VIDEO LOST");
        }
        this.reconnectTimer = setTimeout(() => this.connect(), 750);
      };
      socket.onerror = () => socket.close();
    }

    configure({enabled, label, serial}) {
      this.label = label || ROLE_LABELS[this.role];
      const nextEnabled = Boolean(enabled);
      const sourceChanged = (serial || null) !== this.serial;
      if (nextEnabled === this.enabled && !sourceChanged) {
        if (nextEnabled) this.connect();
        return;
      }
      this.disconnect();
      this.enabled = nextEnabled;
      this.serial = serial || null;
      this.lastFrameAt = 0;
      this.pendingBlob = null;
      if (this.enabled) {
        this.drawPlaceholder("CONNECTING");
        this.connect();
      } else {
        this.drawPlaceholder("NOT SELECTED");
      }
    }

    disconnect() {
      clearTimeout(this.reconnectTimer);
      this.connected = false;
      if (this.socket) {
        const socket = this.socket;
        this.socket = null;
        socket.onclose = null;
        socket.close();
      }
    }

    async decodeLatest() {
      if (this.decoding || !this.pendingBlob) return;
      this.decoding = true;
      const blob = this.pendingBlob;
      this.pendingBlob = null;
      try {
        const bitmap = await createImageBitmap(blob);
        this.drawFrame(bitmap);
        bitmap.close();
      } catch (_) {
        this.drawPlaceholder("DECODE ERROR");
      } finally {
        this.decoding = false;
        if (this.pendingBlob) this.decodeLatest();
      }
    }

    drawFrame(bitmap) {
      const width = this.canvas.width;
      const height = this.canvas.height;
      const sourceAspect = bitmap.width / bitmap.height;
      const targetAspect = width / height;
      let sourceX = 0;
      let sourceY = 0;
      let sourceWidth = bitmap.width;
      let sourceHeight = bitmap.height;
      if (sourceAspect > targetAspect) {
        sourceWidth = bitmap.height * targetAspect;
        sourceX = (bitmap.width - sourceWidth) / 2;
      } else if (sourceAspect < targetAspect) {
        sourceHeight = bitmap.width / targetAspect;
        sourceY = (bitmap.height - sourceHeight) / 2;
      }
      this.context.drawImage(
        bitmap,
        sourceX,
        sourceY,
        sourceWidth,
        sourceHeight,
        0,
        0,
        width,
        height,
      );
      this.lastFrameAt = performance.now();
      this.showingVideoLost = false;
      this.drawHeader();
      this.commitFrame();
    }

    drawHeader() {
      const context = this.context;
      const roleLabel = this.label.toUpperCase();
      const age = this.frameAgeMs();
      const stale = age > 1000;
      const gradient = context.createLinearGradient(0, 0, 0, 76);
      gradient.addColorStop(0, "rgba(4, 10, 17, .92)");
      gradient.addColorStop(1, "rgba(4, 10, 17, 0)");
      context.fillStyle = gradient;
      context.fillRect(0, 0, this.canvas.width, 76);
      context.font = "700 25px system-ui, sans-serif";
      context.fillStyle = "#f7fbff";
      context.fillText(roleLabel, 22, 37);
      context.beginPath();
      context.arc(this.canvas.width - 124, 29, 7, 0, Math.PI * 2);
      context.fillStyle = stale ? "#ffb85c" : "#4be28a";
      context.fill();
      context.font = "600 18px system-ui, sans-serif";
      context.fillStyle = stale ? "#ffd49a" : "#baf7d1";
      context.fillText(stale ? "STALE" : "LIVE", this.canvas.width - 108, 36);
    }

    drawPlaceholder(message, notify = true) {
      this.showingVideoLost = message === "VIDEO LOST";
      const context = this.context;
      context.fillStyle = "#0b1721";
      context.fillRect(0, 0, this.canvas.width, this.canvas.height);
      context.strokeStyle = "#38536a";
      context.lineWidth = 4;
      roundedRect(context, 15, 15, this.canvas.width - 30, this.canvas.height - 30, 22);
      context.stroke();
      context.textAlign = "center";
      context.fillStyle = "#90a9ba";
      context.font = "700 25px system-ui, sans-serif";
      context.fillText(`${this.label.toUpperCase()} CAMERA`, this.canvas.width / 2, 215);
      context.fillStyle = message === "VIDEO LOST" ? "#ffbd75" : "#c7d6df";
      context.font = "800 31px system-ui, sans-serif";
      context.fillText(message, this.canvas.width / 2, 268);
      context.textAlign = "start";
      this.commitFrame(notify);
    }

    commitFrame(notify = true) {
      this.revision += 1;
      if (this.previewCanvas) {
        const preview = this.previewCanvas.getContext("2d", {alpha: false});
        preview.drawImage(this.canvas, 0, 0, this.previewCanvas.width, this.previewCanvas.height);
      }
      if (notify) this.statusChanged();
    }

    frameAgeMs() {
      return this.lastFrameAt ? performance.now() - this.lastFrameAt : Infinity;
    }

    isFresh() {
      return this.enabled && this.connected && this.frameAgeMs() < 1000;
    }

    close() {
      this.enabled = false;
      this.disconnect();
      clearInterval(this.staleTimer);
    }
  }

  function compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const message = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`camera shader compile failed: ${message}`);
    }
    return shader;
  }

  function createProgram(gl) {
    const vertex = compileShader(gl, gl.VERTEX_SHADER, `
      attribute vec3 a_position;
      attribute vec2 a_texcoord;
      uniform mat4 u_mvp;
      varying vec2 v_texcoord;
      void main() {
        gl_Position = u_mvp * vec4(a_position, 1.0);
        v_texcoord = a_texcoord;
      }
    `);
    const fragment = compileShader(gl, gl.FRAGMENT_SHADER, `
      precision mediump float;
      varying vec2 v_texcoord;
      uniform sampler2D u_texture;
      void main() {
        gl_FragColor = texture2D(u_texture, v_texcoord);
      }
    `);
    const program = gl.createProgram();
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`camera shader link failed: ${gl.getProgramInfoLog(program)}`);
    }
    return program;
  }

  function multiplyMatrices(left, right) {
    const result = new Float32Array(16);
    for (let column = 0; column < 4; column += 1) {
      for (let row = 0; row < 4; row += 1) {
        let value = 0;
        for (let inner = 0; inner < 4; inner += 1) {
          value += left[inner * 4 + row] * right[column * 4 + inner];
        }
        result[column * 4 + row] = value;
      }
    }
    return result;
  }

  function copyLayout(layout) {
    return Object.fromEntries(
      Object.entries(layout).map(([role, rect]) => [role, [...rect]]),
    );
  }

  function interpolateRect(from, to, amount) {
    return from.map((value, index) => value + (to[index] - value) * amount);
  }

  function easeInOutCubic(amount) {
    return amount < 0.5
      ? 4 * amount * amount * amount
      : 1 - Math.pow(-2 * amount + 2, 3) / 2;
  }

  function panelRect(left, right, bottom, top, depth) {
    return [
      left,
      right,
      bottom + PANEL_VERTICAL_OFFSET_M,
      top + PANEL_VERTICAL_OFFSET_M,
      depth,
    ];
  }

  // Rectangles are [left, right, bottom, top, depth] in the fixed world-anchor
  // frame. Overview adapts to the number of enabled views; focus keeps every
  // secondary camera visible as a small preview.
  function panelLayout(enabledRoles, mode) {
    const layout = {};
    const count = enabledRoles.length;
    if (count === 0) return layout;
    if (mode === "overview" || !enabledRoles.includes(mode)) {
      if (count === 1) {
        layout[enabledRoles[0]] = panelRect(-0.78, 0.78, -0.28, 0.89, -1.40);
      } else if (count === 2) {
        layout[enabledRoles[0]] = panelRect(-0.88, -0.04, -0.08, 0.55, -1.38);
        layout[enabledRoles[1]] = panelRect(0.04, 0.88, -0.08, 0.55, -1.38);
      } else {
        layout[enabledRoles[0]] = panelRect(-1.10, -0.38, -0.06, 0.48, -1.52);
        layout[enabledRoles[1]] = panelRect(-0.36, 0.36, -0.06, 0.48, -1.52);
        layout[enabledRoles[2]] = panelRect(0.38, 1.10, -0.06, 0.48, -1.52);
      }
      return layout;
    }

    layout[mode] = panelRect(-0.75, 0.75, -0.55, 0.55, -1.35);
    const previews = enabledRoles.filter((role) => role !== mode);
    if (previews.length === 1) {
      layout[previews[0]] = panelRect(0.44, 0.76, 0.40, 0.66, -1.28);
    } else {
      layout[previews[0]] = panelRect(-0.76, -0.42, 0.40, 0.66, -1.28);
      layout[previews[1]] = panelRect(0.42, 0.76, 0.40, 0.66, -1.28);
    }
    return layout;
  }

  class QuestCameraView {
    constructor({previews, statusText, viewModeText, configurationChanged}) {
      this.statusText = statusText;
      this.viewModeText = viewModeText;
      this.configurationChanged = configurationChanged || (() => {});
      this.currentMode = "overview";
      this.roleConfiguration = Object.fromEntries(
        CAMERA_ROLES.map((role) => [role, {serial: null, enabled: false}]),
      );
      this.previousBPressed = false;
      this.gl = null;
      this.program = null;
      this.buffer = null;
      this.textures = {};
      this.uploadedRevisions = Object.fromEntries(CAMERA_ROLES.map((role) => [role, -1]));
      this.worldAnchorMatrix = null;
      this.transition = null;
      const statusChanged = () => this.refreshStatus();
      this.feeds = Object.fromEntries(
        CAMERA_ROLES.map((role) => [
          role,
          new CameraFeed(role, previews[role], statusChanged),
        ]),
      );
      this.refreshStatus();
    }

    connect() {
      for (const role of this.enabledRoles()) this.feeds[role].connect();
    }

    configure(roles, devices = []) {
      for (const role of CAMERA_ROLES) {
        const item = roles?.[role] || {};
        const serial = item.serial || null;
        const enabled = Boolean(item.enabled && serial);
        this.roleConfiguration[role] = {serial, enabled};
        this.feeds[role].configure({
          enabled,
          serial,
          label: ROLE_LABELS[role],
        });
      }
      this.currentMode = "overview";
      this.transition = null;
      this.configurationChanged(this.enabledRoles());
      this.refreshStatus();
    }

    enabledRoles() {
      return CAMERA_ROLES.filter((role) => this.roleConfiguration[role].enabled);
    }

    modeSequence() {
      const enabled = this.enabledRoles();
      return enabled.length > 1 ? ["overview", ...enabled] : ["overview"];
    }

    canCycle() {
      return this.modeSequence().length > 1;
    }

    mode() {
      return this.currentMode;
    }

    cycle() {
      const sequence = this.modeSequence();
      if (sequence.length <= 1) {
        this.refreshStatus();
        return this.mode();
      }
      const now = performance.now();
      const from = this.animatedLayout(now);
      const currentIndex = Math.max(0, sequence.indexOf(this.currentMode));
      this.currentMode = sequence[(currentIndex + 1) % sequence.length];
      this.transition = {
        from: copyLayout(from),
        to: copyLayout(panelLayout(this.enabledRoles(), this.mode())),
        startedAt: now,
      };
      this.refreshStatus();
      return this.mode();
    }

    animatedLayout(now = performance.now()) {
      if (!this.transition) return panelLayout(this.enabledRoles(), this.mode());
      const elapsed = Math.max(0, now - this.transition.startedAt);
      const amount = Math.min(1, elapsed / TRANSITION_DURATION_MS);
      if (amount >= 1) {
        this.transition = null;
        return panelLayout(this.enabledRoles(), this.mode());
      }
      const eased = easeInOutCubic(amount);
      return Object.fromEntries(
        this.enabledRoles().map((role) => [
          role,
          interpolateRect(this.transition.from[role], this.transition.to[role], eased),
        ]),
      );
    }

    refreshStatus() {
      const enabled = this.enabledRoles();
      const fresh = enabled.filter((role) => this.feeds[role].isFresh());
      if (this.statusText) {
        this.statusText.textContent = enabled.length === 0
          ? "No cameras selected for passthrough"
          : fresh.length === enabled.length
            ? `${fresh.length} camera stream${fresh.length === 1 ? "" : "s"} live`
            : `${fresh.length}/${enabled.length} camera streams live`;
        this.statusText.classList.toggle("ok", enabled.length > 0 && fresh.length === enabled.length);
      }
      if (this.viewModeText) {
        if (enabled.length === 0) {
          this.viewModeText.textContent = "No camera views selected";
        } else if (this.mode() === "overview") {
          this.viewModeText.textContent = `Passthrough overview · ${enabled.length} camera${enabled.length === 1 ? "" : "s"}`;
        } else {
          const previews = enabled.length - 1;
          this.viewModeText.textContent = `${ROLE_LABELS[this.mode()]} focus · ${previews} preview${previews === 1 ? "" : "s"}`;
        }
      }
    }

    handleControllerButtons(inputSources) {
      const right = inputSources.right;
      const pressed = buttonValue(right?.gamepad, QUEST_B_BUTTON) >= 0.7;
      const risingEdge = pressed && !this.previousBPressed;
      this.previousBPressed = pressed;
      if (!risingEdge) return;
      if (!this.canCycle()) return;

      const presentSources = Object.values(inputSources).filter(Boolean);
      const gripsReleased = presentSources.every(
        (source) => buttonValue(source.gamepad, GRIP_BUTTON) <= GRIP_RELEASED,
      );
      const actuator = right?.gamepad?.hapticActuators?.[0];
      if (!gripsReleased) {
        if (this.statusText) this.statusText.textContent = "Release both grips to change camera view";
        if (actuator) actuator.pulse(0.25, 70);
        return;
      }
      this.cycle();
      if (actuator) actuator.pulse(0.55, 45);
    }

    initializeWebGL(gl) {
      this.gl = gl;
      // A new immersive session gets a new world anchor on its first frame.
      // Mode changes outside WebXR should not begin halfway through a stale
      // animation when the next session starts.
      this.worldAnchorMatrix = null;
      this.transition = null;
      this.program = createProgram(gl);
      this.buffer = gl.createBuffer();
      for (const role of CAMERA_ROLES) {
        const texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        this.textures[role] = texture;
      }
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    }

    uploadLatestTextures() {
      const gl = this.gl;
      for (const role of this.enabledRoles()) {
        const feed = this.feeds[role];
        if (feed.revision === this.uploadedRevisions[role]) continue;
        gl.bindTexture(gl.TEXTURE_2D, this.textures[role]);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, feed.canvas);
        this.uploadedRevisions[role] = feed.revision;
      }
    }

    drawPanel(role, rect, mvp) {
      const gl = this.gl;
      const [left, right, bottom, top, depth] = rect;
      const vertices = new Float32Array([
        left, bottom, depth, 0, 0,
        right, bottom, depth, 1, 0,
        left, top, depth, 0, 1,
        left, top, depth, 0, 1,
        right, bottom, depth, 1, 0,
        right, top, depth, 1, 1,
      ]);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);
      const position = gl.getAttribLocation(this.program, "a_position");
      const texcoord = gl.getAttribLocation(this.program, "a_texcoord");
      gl.enableVertexAttribArray(position);
      gl.enableVertexAttribArray(texcoord);
      gl.vertexAttribPointer(position, 3, gl.FLOAT, false, 20, 0);
      gl.vertexAttribPointer(texcoord, 2, gl.FLOAT, false, 20, 12);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.textures[role]);
      gl.uniform1i(gl.getUniformLocation(this.program, "u_texture"), 0);
      gl.uniformMatrix4fv(gl.getUniformLocation(this.program, "u_mvp"), false, mvp);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    render(session, viewerPose) {
      const enabled = this.enabledRoles();
      if (!this.gl || !viewerPose || enabled.length === 0) return;
      const gl = this.gl;
      const layer = session.renderState.baseLayer;
      // Capture the entry pose once in local-floor space. Rendering every eye
      // from this fixed transform pins the panels in the room instead of
      // recomputing their placement from the moving headset pose.
      if (!this.worldAnchorMatrix) {
        this.worldAnchorMatrix = new Float32Array(viewerPose.transform.matrix);
      }
      const layout = this.animatedLayout();
      this.uploadLatestTextures();
      gl.useProgram(this.program);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      for (const view of viewerPose.views) {
        const viewport = layer.getViewport(view);
        gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
        const viewFromAnchor = multiplyMatrices(
          view.transform.inverse.matrix,
          this.worldAnchorMatrix,
        );
        const mvp = multiplyMatrices(view.projectionMatrix, viewFromAnchor);
        // Draw the focused panel first and its preview second so the preview
        // remains visible wherever the two rectangles overlap.
        const drawOrder = this.mode() === "overview"
          ? enabled
          : [this.mode(), ...enabled.filter((role) => role !== this.mode())];
        for (const role of drawOrder) this.drawPanel(role, layout[role], mvp);
      }
      gl.disable(gl.BLEND);
    }
  }

  window.WidowXAICameraView = {
    QuestCameraView,
    CAMERA_ROLES,
    ROLE_LABELS,
    QUEST_B_BUTTON,
  };
})();
