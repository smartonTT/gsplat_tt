# shellcheck shell=bash
# Shared by opt/eth/make_overlay.sh and opt/eth/env.sh (sourced, no side effects).
# The live viewer's tree (/localdev/smarton/viewer: its tt-metal, venv and JIT cache) must
# never receive an overlay or a JIT cache. ETH_FORBID_EXTRA adds one more forbidden prefix
# (used by opt/eth/test_eth_overlay.sh; it cannot remove the viewer prefix).

ETH_VIEWER_DIR=/localdev/smarton/viewer

# Absolute path with symlinks resolved for the longest existing prefix; the rest is kept
# as written. Works for paths that do not exist yet.
eth_abspath() {
  local p=$1 rest="" d
  case $p in /*) ;; *) p=$PWD/$p ;; esac
  while [ ! -d "$p" ]; do
    rest=/$(basename "$p")$rest
    p=$(dirname "$p")
  done
  d=$(cd -P "$p" && pwd) || return 1
  [ "$d" = / ] && d=""
  # the part that does not exist has no symlinks: fold . and .. lexically
  local IFS=/ c
  for c in $rest; do
    case $c in ''|.) ;; ..) d=${d%/*} ;; *) d=$d/$c ;; esac
  done
  printf '%s\n' "${d:-/}"
}

# eth_under <path> <prefix>: true if path is prefix or inside it (both as given).
eth_under() {
  case "$1/" in "${2%/}/"*) return 0 ;; esac
  return 1
}

# eth_forbidden <path>: true if path (as written or resolved) is in a forbidden tree.
eth_forbidden() {
  local a pre
  a=$(eth_abspath "$1") || return 0
  for pre in "$ETH_VIEWER_DIR" ${ETH_FORBID_EXTRA:+"$ETH_FORBID_EXTRA"}; do
    eth_under "$1" "$pre" && return 0
    eth_under "$a" "$pre" && return 0
    [ -d "$pre" ] && eth_under "$a" "$(cd -P "$pre" && pwd)" && return 0
  done
  return 1
}
