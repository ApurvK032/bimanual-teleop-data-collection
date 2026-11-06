#!/usr/bin/env bash
# Installer with windows instead of a terminal (started by Install.sh: right-click it -> "Run as a Program").
# Everything goes into this folder and your home folder; nothing system-wide, except the optional one-time
# System setup at the end, which Ubuntu's own password dialog guards.
set -uo pipefail
cd "$(dirname "$0")/.."
root=$(pwd)
mkdir -p state/logs
log="$root/state/logs/install.log"
echo "=== $(date '+%F %T') install" >>"$log"
export PATH="$HOME/.local/bin:$PATH"

gui=0
if command -v zenity >/dev/null 2>&1 && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then gui=1; fi
title="Robot data collection"

fail() {
  local message="$1"
  echo "FAILED: $message" >>"$log"
  if ((gui)); then
    zenity --error --title="$title" --width=520 --text="Installation stopped: $message\n\nDetails: $log\n\n$(tail -n 8 "$log" | sed 's/[<>&]//g')"
  else
    echo "Installation stopped: $message (details: $log)" >&2
  fi
  exit 1
}

steps() {
  echo "# Getting uv (the Python installer)…"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh 2>>"$log" | sh >>"$log" 2>&1 || { echo "FAIL uv"; return; }
  fi
  echo "# Getting the teleop code and the arm model…"
  if [[ -d .git ]] && command -v git >/dev/null 2>&1; then
    git submodule update --init --recursive >>"$log" 2>&1 || { echo "FAIL teleop"; return; }
  else
    # A ZIP download has no submodules: fetch each pinned commit listed in teleop.lock that is still missing
    # (e.g. Trossen's MuJoCo arm model, which the arm programs load).
    while read -r path commit url; do
      [[ -z "$path" || "$path" == \#* ]] && continue
      if [[ -z "$(ls -A "$path" 2>/dev/null)" ]]; then
        mkdir -p "$path"
        curl -LsSf "${url%.git}/archive/$commit.tar.gz" 2>>"$log" | tar -xz --strip-components=1 -C "$path" >>"$log" 2>&1 \
          || { echo "FAIL teleop"; return; }
      fi
    done < teleop.lock
  fi
  echo "# Building the main environment (a few minutes)…"
  bash scripts/install.sh --main-only >>"$log" 2>&1 || { echo "FAIL main"; return; }
  echo "# Building the arm driver environment…"
  bash scripts/install.sh --arm-only >>"$log" 2>&1 || { echo "FAIL arm"; return; }
  echo "# Building the dataset export environment (LeRobot; the longest step)…"
  bash scripts/install.sh --lerobot-only >>"$log" 2>&1 || { echo "FAIL lerobot"; return; }
  echo "# Fetching camera calibrations…"
  env -u PYTHONPATH .venv/bin/python -m bimanual_collect calibration >>"$log" 2>&1 || true
  echo "# Adding the desktop icon…"
  env -u PYTHONPATH .venv/bin/python -m bimanual_collect install-shortcut >>"$log" 2>&1 || { echo "FAIL shortcut"; return; }
  echo "OK"
}

result=$(mktemp)
if ((gui)); then
  steps | tee "$result" | grep --line-buffered '^#' | zenity --progress --pulsate --auto-close --no-cancel \
    --title="$title" --width=480 --text="Installing…"
else
  steps | tee "$result"
fi
status=$(grep -E '^(OK|FAIL)' "$result" | tail -n 1)
rm -f "$result"
case "$status" in
  OK) ;;
  "FAIL uv") fail "could not download uv (is the internet reachable?)" ;;
  "FAIL teleop") fail "could not get the teleop code" ;;
  "FAIL main") fail "the main Python environment did not build" ;;
  "FAIL arm") fail "the arm driver environment did not build" ;;
  "FAIL lerobot") fail "the LeRobot environment did not build" ;;
  "FAIL shortcut") fail "the desktop icon could not be added" ;;
  *) fail "unexpected end" ;;
esac

# One-time system parts (camera access rule, ffmpeg / v4l-utils / adb / power-profiles-daemon).
needs=$(env -u PYTHONPATH .venv/bin/python -c "from bimanual_collect import system_setup as s; print('; '.join(n['label'] for n in s.needs()))")
if [[ -n "$needs" ]]; then
  if ((gui)) && zenity --question --title="$title" --width=480 \
      --text="This PC still needs: $needs.\n\nInstall them now? Ubuntu will ask for your password." ; then
    pkexec "$root/scripts/system_setup_root.sh" >>"$log" 2>&1 \
      || zenity --warning --title="$title" --width=480 --text="System setup did not finish. You can run it later from the app (System tab)."
  elif ((!gui)); then
    echo "Still needed: $needs -> sudo $root/scripts/system_setup_root.sh"
  fi
fi

echo "=== done" >>"$log"
if ((gui)); then
  if zenity --question --title="$title" --width=420 --ok-label="Open the app" --cancel-label="Later" \
      --text="Installed. The app is in your app menu and on the desktop as 'Robot data collection'."; then
    env -u PYTHONPATH .venv/bin/python -m bimanual_collect app --open >>"$log" 2>&1 &
  fi
else
  echo "Installed. Start: .venv/bin/python -m bimanual_collect app --open   (or the 'Robot data collection' icon)"
fi
