#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"

with_cameras=false
camera_args=()
while (($#)); do
  case "$1" in
    --with-cameras)
      with_cameras=true
      shift
      ;;
    --scene-serial|--wrist-serial|--left-wrist-serial|--right-wrist-serial)
      if (($# < 2)); then
        echo "ERROR: $1 requires a serial value" >&2
        exit 2
      fi
      with_cameras=true
      camera_args+=("$1" "$2")
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      echo "Usage: $0 [--with-cameras] [--scene-serial SERIAL] [--left-wrist-serial SERIAL] [--right-wrist-serial SERIAL]" >&2
      exit 2
      ;;
  esac
done

# Stop only this project's Quest relay. The bracketed expressions keep pgrep
# from matching its own command line.
mapfile -t relay_pids < <(
  pgrep -f '([w]idowxai-quest-relay|[p]ython(3(\.[0-9]+)?)? -m widowxai_quest_teleop\.relay)' || true
)
for pid in "${relay_pids[@]}"; do
  kill "$pid" 2>/dev/null || true
done

mapfile -t camera_pids < <(
  pgrep -f '([w]idowxai-camera-service|[p]ython(3(\.[0-9]+)?)? -m widowxai_quest_teleop\.camera_service)' || true
)
for pid in "${camera_pids[@]}"; do
  kill "$pid" 2>/dev/null || true
done

for _ in $(seq 1 30); do
  relay_alive=false
  for pid in "${relay_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      relay_alive=true
      break
    fi
  done
  if ! $relay_alive; then
    break
  fi
  sleep 0.1
done

for _ in $(seq 1 30); do
  camera_alive=false
  for pid in "${camera_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      camera_alive=true
      break
    fi
  done
  if ! $camera_alive; then
    break
  fi
  sleep 0.1
done

if ! command -v adb >/dev/null 2>&1; then
  echo "ERROR: adb is not installed or is not on PATH" >&2
  exit 1
fi
if ! adb get-state 2>/dev/null | grep -qx device; then
  echo "ERROR: Quest is not visible to adb. Connect it over USB and accept USB debugging." >&2
  exit 1
fi

adb reverse --remove tcp:8443 >/dev/null 2>&1 || true
adb reverse tcp:8443 tcp:8443 >/dev/null
if $with_cameras; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required to verify camera-service startup" >&2
    exit 1
  fi
  adb reverse --remove tcp:8444 >/dev/null 2>&1 || true
  adb reverse tcp:8444 tcp:8444 >/dev/null
else
  adb reverse --remove tcp:8444 >/dev/null 2>&1 || true
fi

# Keep development passthrough/WebXR alive while the headset is off the face.
# Horizon OS clears the virtual proximity override on reboot. Reset any stale
# automation state first, then force the virtual sensor to CLOSE (mounted).
# The stay-on mask covers USB or AC-powered hubs.
adb shell am broadcast \
  -a com.oculus.vrpowermanager.automation_disable >/dev/null
adb shell am broadcast \
  -a com.oculus.vrpowermanager.prox_close >/dev/null
adb shell svc power stayon true
adb shell input keyevent KEYCODE_WAKEUP

quest_power_state=$(adb shell dumpsys vrpowermanager 2>/dev/null | tr -d '\r')
if ! grep -Fqx 'Virtual proximity state: CLOSE' <<<"$quest_power_state" || \
   ! grep -Fqx 'State: HEADSET_MOUNTED' <<<"$quest_power_state"; then
  echo "ERROR: Quest did not accept the persistent mounted-state override." >&2
  echo "Reconnect ADB, restart the headset if necessary, and try again." >&2
  exit 1
fi

echo "Quest relay: http://localhost:8443"
echo "Quest virtual proximity: mounted until the next headset reboot."
echo "Enter passthrough once; it can then remain on the table across arm runs."
echo "Press Ctrl+C to stop it."

camera_pid=""
relay_pid=""
cleaned_up=false
cleanup() {
  if $cleaned_up; then
    return
  fi
  cleaned_up=true
  if [[ -n "$relay_pid" ]]; then
    kill "$relay_pid" 2>/dev/null || true
  fi
  if [[ -n "$camera_pid" ]]; then
    kill "$camera_pid" 2>/dev/null || true
  fi
  for _ in $(seq 1 30); do
    relay_stopped=true
    camera_stopped=true
    [[ -n "$relay_pid" ]] && kill -0 "$relay_pid" 2>/dev/null && relay_stopped=false
    [[ -n "$camera_pid" ]] && kill -0 "$camera_pid" 2>/dev/null && camera_stopped=false
    if $relay_stopped && $camera_stopped; then
      break
    fi
    sleep 0.1
  done
  [[ -n "$relay_pid" ]] && kill -KILL "$relay_pid" 2>/dev/null || true
  [[ -n "$camera_pid" ]] && kill -KILL "$camera_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if $with_cameras; then
  camera_log=/tmp/widowxai-camera-service.log
  : >"$camera_log"
  env -u PYTHONPATH .venv/bin/python -m widowxai_quest_teleop.camera_service \
    --host 0.0.0.0 --port 8444 "${camera_args[@]}" >"$camera_log" 2>&1 &
  camera_pid=$!
  camera_ready=false
  for _ in $(seq 1 80); do
    if ! kill -0 "$camera_pid" 2>/dev/null; then
      echo "ERROR: camera service stopped during startup" >&2
      sed -n '1,160p' "$camera_log" >&2
      exit 1
    fi
    if curl --silent --fail http://127.0.0.1:8444/health >/dev/null 2>&1; then
      camera_ready=true
      break
    fi
    sleep 0.1
  done
  if ! $camera_ready; then
    echo "ERROR: camera service did not become ready" >&2
    sed -n '1,160p' "$camera_log" >&2
    exit 1
  fi
  echo "Camera service: http://localhost:8444 (configurable scene + left/right wrist)"
fi

env -u PYTHONPATH .venv/bin/widowxai-quest-relay --host 0.0.0.0 --port 8443 &
relay_pid=$!
wait "$relay_pid"
