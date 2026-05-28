#!/usr/bin/env bash
# Sourced by every iter script. Pins paths so cron/launchd/agent contexts
# all find the same toolchain.
#
# NOTE: this script ALWAYS forces LOCAL_PY / REPO_ROOT / OPT_DIR to the
# values resolved from its own location. Earlier env vars are overridden,
# so a stale LOCAL_PY from a parent shell can't pin us to Apple's 3.9.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Resolve REPO_ROOT from this script's location. Works in bash (BASH_SOURCE)
# and zsh (${(%):-%x}); we fall through to the dirname of $0 as a last resort.
__env_script="${BASH_SOURCE[0]:-${(%):-%x}}"
__env_script="${__env_script:-$0}"
__env_dir="$(cd "$(dirname "$__env_script")" && pwd)"
export REPO_ROOT="$(cd "$__env_dir/.." && pwd)"

# Prefer the project venv (python 3.12 + torch/numpy/viser/...). Fall back to
# /usr/bin/python3 (Apple stock 3.9) only if the venv is missing.
if [[ -x "$REPO_ROOT/.venv/bin/python3" ]]; then
  export LOCAL_PY="$REPO_ROOT/.venv/bin/python3"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  export LOCAL_PY="$REPO_ROOT/.venv/bin/python"
else
  export LOCAL_PY="/usr/bin/python3"
fi
export OPT_DIR="$REPO_ROOT/opt"
export ITER_DIR_PARENT="$OPT_DIR/screenshots"
export REF_DIR="$REPO_ROOT/benchmarks/reference_v2"
# Default IRD box for metal port (override: REMOTE_HOST=yyzo-bh-07 ...)
export METAL_REMOTE_HOST="${METAL_REMOTE_HOST:-bh-30}"
export REMOTE_HOST="${REMOTE_HOST:-$METAL_REMOTE_HOST}"
