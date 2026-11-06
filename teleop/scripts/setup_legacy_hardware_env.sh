#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON310:-}" ]]; then
  python_bin="$PYTHON310"
elif command -v python3.10 >/dev/null 2>&1; then
  python_bin="$(command -v python3.10)"
elif [[ -x "$HOME/miniforge3/envs/xr/bin/python" ]]; then
  python_bin="$HOME/miniforge3/envs/xr/bin/python"
else
  echo "The hardware environment is pinned to CPython 3.10 (trossen-arm 1.9.3)." >&2
  echo "Set PYTHON310=/absolute/path/to/python3.10 and rerun." >&2
  exit 1
fi

# Both lab controllers run firmware 1.9.4 since 2026-10-20, so the driver must
# be 1.9.x. The older .venv-arm18 (driver 1.8.6) is kept only as history; the
# script name is historical too.
"$python_bin" -m venv .venv-arm19
env -u PYTHONPATH .venv-arm19/bin/python -m pip install --upgrade pip
env -u PYTHONPATH .venv-arm19/bin/python -m pip install -e '.[dev,hardware]'
env -u PYTHONPATH .venv-arm19/bin/python - <<'PY'
import importlib.metadata
import sys

driver = importlib.metadata.version("trossen-arm")
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"expected Python 3.10, found {sys.version.split()[0]}")
if driver != "1.9.3":
    raise SystemExit(f"expected trossen-arm 1.9.3, found {driver}")
print(f"hardware environment ready: Python {sys.version.split()[0]}, trossen-arm {driver}")
PY

echo "No arm connection or command was attempted."
