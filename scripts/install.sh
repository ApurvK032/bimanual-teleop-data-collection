#!/usr/bin/env bash
# Set up the Python environments (needs uv; never uses sudo). Install.sh runs this behind a progress window.
#
#   scripts/install.sh                 everything: .venv, teleop/.venv-arm19, .venv-lerobot
#   scripts/install.sh --gpu-depth     also .venv-stereo (RAFT depth: torch + torchvision with CUDA)
#   scripts/install.sh --no-lerobot    skip the LeRobot environment
#   scripts/install.sh --main-only | --arm-only | --lerobot-only
#
# Environments:
#   .venv                  Python 3.12: the app, recording, checks, simulation (teleop package + MuJoCo)
#   teleop/.venv-arm19     Python 3.10 + trossen-arm 1.9.3: the arm driver (built by the teleop repo's own script)
#   .venv-lerobot          Python 3.10 + LeRobot 0.4.3 (CPU torch): dataset export
# System packages it expects: ffmpeg, v4l-utils, adb (the app's System setup installs them, or: sudo apt install ...)
set -euo pipefail
cd "$(dirname "$0")/.."

main=1 arm=1 lerobot=1 gpu=0
for arg in "$@"; do
  case "$arg" in
    --gpu-depth) gpu=1 ;;
    --no-lerobot) lerobot=0 ;;
    --arm-only) main=0 lerobot=0 ;;
    --lerobot-only) main=0 arm=0 ;;
    --main-only) arm=0 lerobot=0 ;;
    *) echo "unknown option $arg" >&2; exit 2 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is needed: curl -LsSf https://astral.sh/uv/install.sh | sh   (then open a new shell)" >&2
  exit 1
fi
missing=()
for tool in ffmpeg v4l2-ctl adb; do command -v "$tool" >/dev/null 2>&1 || missing+=("$tool"); done
if ((${#missing[@]})); then
  echo "NOTE: not found: ${missing[*]}  ->  the app's System setup, or: sudo apt install ffmpeg v4l-utils adb" >&2
fi

if [[ ! -f teleop/pyproject.toml || -z "$(ls -A teleop/third_party/trossen_arm_mujoco 2>/dev/null)" ]]; then
  git submodule update --init --recursive   # the teleop code and/or Trossen's arm model
fi

if ((main)); then
  echo "== .venv: app, recording, checks, simulation (Python 3.12)"
  uv venv --allow-existing -p 3.12 .venv
  uv pip install --python .venv/bin/python -e ./teleop -e ".[session,sim,dev]"
fi

if ((arm)); then
  echo "== teleop/.venv-arm19: arm driver (Python 3.10, trossen-arm 1.9.3)"
  uv python install 3.10
  py310=$(uv python find 3.10)
  if [[ -x teleop/.venv-arm19/bin/python ]] && env -u PYTHONPATH teleop/.venv-arm19/bin/python -c \
      "import importlib.metadata as m, sys; sys.exit(m.version('trossen-arm') != '1.9.3')" 2>/dev/null; then
    echo "already there"
  else
    PYTHON310="$py310" bash teleop/scripts/setup_legacy_hardware_env.sh
  fi
fi

if ((lerobot)); then
  echo "== .venv-lerobot: LeRobot export (Python 3.10, CPU torch)"
  uv venv --allow-existing -p 3.10 .venv-lerobot
  uv pip install --python .venv-lerobot/bin/python -e ".[lerobot]" pytest \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
fi

if ((gpu)); then
  echo "== .venv-stereo: RAFT depth on the GPU (Python 3.12, CUDA torch)"
  uv venv --allow-existing -p 3.12 .venv-stereo
  uv pip install --python .venv-stereo/bin/python -e ".[depth]"
fi

cat <<'EOF'

Done. Start the app: .venv/bin/python -m bimanual_collect app   (or install the desktop icon:
  .venv/bin/python -m bimanual_collect install-shortcut). Then follow the Bench tab.
EOF
