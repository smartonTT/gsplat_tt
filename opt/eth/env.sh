# shellcheck shell=bash
# t392: run environment for the ETH dispatch overlay made by opt/eth/make_overlay.sh.
#   source opt/eth/env.sh <overlay dir> [<jit cache dir>]          (exports into this shell)
#   bash opt/eth/env.sh <overlay dir> [<jit cache dir>] -- <cmd...> (runs cmd with them)
# Exports TT_METAL_RUNTIME_ROOT=<overlay> and a JIT cache of its own (default
# <overlay>-cache): TT_METAL_CACHE=<cache>/prod and TT_METAL_CACHE_RENDER=<cache>/render
# (render/run.py sets TT_METAL_CACHE from TT_METAL_CACHE_RENDER). ELFs linked with the
# overlay's 32 KB idle-ERISC bound must never mix with another cache, so this refuses any
# overlay or cache under /localdev/smarton/viewer (the live viewer's tt-metal, venv and
# cache) and the shared default caches. GSPLAT_TT_DISPATCH is left to the caller (both A/B
# arms run through the same overlay). Exit/return 2 on refusal, nothing exported.
_eth_env() {
  local here ov cache c
  here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  . "$here/_paths.sh"
  [ $# -ge 1 ] && [ -n "$1" ] || { echo "env.sh: usage: <overlay dir> [<jit cache dir>] [-- cmd...]" >&2; return 2; }
  if eth_forbidden "$1"; then echo "env.sh: refusing overlay $1: under the live viewer's tree $ETH_VIEWER_DIR" >&2; return 2; fi
  ov=$(eth_abspath "$1")
  [ -f "$ov/.gsplat-eth-overlay" ] || { echo "env.sh: $ov is not an overlay from opt/eth/make_overlay.sh" >&2; return 2; }
  cache=${2:-$ov-cache}
  if eth_forbidden "$cache"; then echo "env.sh: refusing JIT cache $cache: under the live viewer's tree $ETH_VIEWER_DIR" >&2; return 2; fi
  cache=$(eth_abspath "$cache")
  for c in /localdev/smarton/.cache/tt-metal-cache /localdev/smarton/.cache/tt-metal-cache-render \
           /localdev/smarton/.cache/tt-metal-cache-viewer; do
    eth_under "$cache" "$c" && { echo "env.sh: refusing shared JIT cache $cache" >&2; return 2; }
  done
  eth_under "$cache" "$ov" && { echo "env.sh: refusing JIT cache inside the overlay: $cache" >&2; return 2; }
  export TT_METAL_RUNTIME_ROOT=$ov TT_METAL_CACHE=$cache/prod TT_METAL_CACHE_RENDER=$cache/render
  echo "eth env: TT_METAL_RUNTIME_ROOT=$ov TT_METAL_CACHE=$cache/{prod,render} $(tr '\n' ' ' < "$ov/.gsplat-eth-overlay")" >&2
}
if [ "${BASH_SOURCE[0]}" != "$0" ]; then
  _eth_env "$@"
else
  set -u
  _eth_args=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do _eth_args+=("$1"); shift; done
  [ ${#_eth_args[@]} -ge 1 ] && [ "${1:-}" = -- ] && [ $# -ge 2 ] || { echo "env.sh: usage: $0 <overlay dir> [<jit cache dir>] -- <cmd...>" >&2; exit 2; }
  shift
  _eth_env "${_eth_args[@]}" || exit $?
  exec "$@"
fi
