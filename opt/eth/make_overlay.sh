#!/usr/bin/env bash
# t392 (overlay v2; v1 = docs/eth-dispatch-t387/make_overlay.sh on ttp/t387-*). Builds a
# TT_METAL_RUNTIME_ROOT overlay of a read-only tt-metal checkout so Blackhole ETH dispatch
# (GSPLAT_TT_DISPATCH=eth, 12x10 = 120 cores on the p150) can open without editing or
# rebuilding tt-metal. Everything in the overlay is a symlink into <tt-metal dir> except
# three real files (see docs/eth-dispatch-patch/README.md on ttp/t390-*, blockers (a), (b)):
#   tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml
#       every 14-core dispatch list [0,0]..[0,13] cut to [0,0]..[0,N-1] (N = live ETH
#       cores, 12 on a p150 with ETH harvesting mask 0x120); stock lists throw at open.
#   runtime/hw/toolchain/blackhole/kernel_ierisc.ld, kernel_subordinate_ierisc.ld
#       the one line LONG((24 * 1024) -> LONG((32 * 1024): the idle-ERISC kernel text link
#       bound. BH idle-ETH kernels run in place from the 25 KB config ring, and
#       program.cpp still TT_FATALs on a real overflow; cq_dispatch needs 0x3124 B > 0x2ab0.
# ETH_IERISC_KB=<32..48> (default 32; task #427) sets that bound for a profiler-only overlay:
# with TT_METAL_DEVICE_PROFILER=1 the idle-ERISC firmware grows and the instrumented
# cq_prefetch (0x4668 B) no longer fits the 32 KB link bound (0x4490 left, #397). Use a
# bigger bound only in its own overlay dir + JIT cache; the default overlay stays at 32.
# Fails (exit 1) unless each .ld changes exactly one line and every yaml list matches.
# Builds into a temp dir and swaps it in only on success; rerunning gives the same tree.
# Refuses (exit 2) an overlay under /localdev/smarton/viewer (the live viewer) or inside
# <tt-metal dir>, and an existing non-empty dir that make_overlay.sh did not make.
#   bash opt/eth/make_overlay.sh <tt-metal dir> <overlay dir> [N=12]
set -euo pipefail
. "$(dirname "$0")/_paths.sh"

usage() { echo "usage: $0 <tt-metal dir> <overlay dir> [live eth cores, default 12]" >&2; exit 2; }
die() { echo "make_overlay: $*" >&2; exit "${RC:-2}"; }
[ $# -ge 2 ] && [ $# -le 3 ] || usage
N=${3:-12}
case $N in ''|*[!0-9]*) die "N must be 1..13, got '$N'" ;; esac
[ "$N" -ge 1 ] && [ "$N" -le 13 ] || die "N must be 1..13, got $N"
KB=${ETH_IERISC_KB:-32}
case $KB in ''|*[!0-9]*) die "ETH_IERISC_KB must be 32..48, got '$KB'" ;; esac
[ "$KB" -ge 32 ] && [ "$KB" -le 48 ] || die "ETH_IERISC_KB must be 32..48, got $KB"

[ -d "$1" ] || die "no tt-metal dir: $1"
SRC=$(cd -P "$1" && pwd)
OV=$(eth_abspath "$2")
YAML=tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml
LDS="runtime/hw/toolchain/blackhole/kernel_ierisc.ld runtime/hw/toolchain/blackhole/kernel_subordinate_ierisc.ld"
MARK=.gsplat-eth-overlay

eth_forbidden "$2" && die "refusing overlay $OV: under the live viewer's tree $ETH_VIEWER_DIR"
eth_under "$OV" "$SRC" && die "refusing overlay $OV: inside the tt-metal dir $SRC"
eth_under "$SRC" "$OV" && die "refusing overlay $OV: contains the tt-metal dir $SRC"
for f in $YAML $LDS; do [ -f "$SRC/$f" ] || die "missing $SRC/$f"; done
if [ -e "$OV" ] || [ -L "$OV" ]; then
  [ -d "$OV" ] && [ ! -L "$OV" ] || die "refusing overlay $OV: exists and is not a directory"
  [ -f "$OV/$MARK" ] || [ -z "$(ls -A "$OV")" ] \
    || die "refusing overlay $OV: non-empty and not made by make_overlay.sh (no $MARK)"
fi

mkdir -p "$(dirname "$OV")"
NEW=$(mktemp -d "$OV.new.XXXXXX")
trap 'rm -rf "$NEW"' EXIT
NEWP=$(cd -P "$NEW" && pwd)

# Symlink every entry of $SRC/<rel> into $NEW/<rel> (a real dir).
link_children() {
  local e s=$SRC d=$NEW
  [ "$1" = . ] || { s=$SRC/$1; d=$NEW/$1; }
  for e in "$s"/* "$s"/.[!.]* "$s"/..?*; do
    [ -e "$e" ] || [ -L "$e" ] || continue
    ln -s "$e" "$d/"
  done
}
# Make $NEW/<rel> a real dir whose entries are symlinks, realizing its parents first so
# that $NEW/<rel> is never reached through a symlink into $SRC.
realize() {
  [ "$1" = . ] && return 0
  realize "$(dirname "$1")"
  [ -L "$NEW/$1" ] || return 0
  rm "$NEW/$1"
  mkdir "$NEW/$1"
  link_children "$1"
}
# Replace symlink <rel> by a real file: the source run through sed <args...>. Never write
# through a symlink: that would change the shared tt-metal checkout itself.
replace() {
  local d
  realize "$(dirname "$1")"
  d=$(cd -P "$NEW/$(dirname "$1")" && pwd)
  [ "$d" = "$NEWP/$(dirname "$1")" ] && [ -L "$NEW/$1" ] \
    || { RC=3; die "internal: $NEW/$1 is not a symlink in a real overlay dir ($d)"; }
  rm "$NEW/$1"
  sed "${@:2}" "$SRC/$1" > "$NEW/$1"
}
# Lines exactly equal to <text> after leading blanks.
count_lines() { sed 's/^ *//' "$1" | grep -cxF "$2" || true; }
# Number of lines diff marks as removed (<) or added (>).
changed() { { diff "$SRC/$1" "$NEW/$1" || true; } | grep -c "^$2" || true; }

link_children .

# yaml: every full 14-core list -> [0,0]..[0,N-1]
list="[0, 0]"; i=1
while [ $i -lt "$N" ]; do list="$list, [0, $i]"; i=$((i + 1)); done
full='\[\[0, 0\], \[0, 1\], \[0, 2\], \[0, 3\], \[0, 4\], \[0, 5\], \[0, 6\], \[0, 7\], \[0, 8\], \[0, 9\], \[0, 10\], \[0, 11\], \[0, 12\], \[0, 13\]\]'
m=$(grep -cE "^ *$full\$" "$SRC/$YAML" || true)
RC=1
[ "$m" -ge 1 ] || die "$YAML: no 14-core [0,0]..[0,13] dispatch list found"
had=$(count_lines "$SRC/$YAML" "[$list]")
replace $YAML -E "s/^( *)$full\$/\\1[$list]/"
[ "$(changed $YAML '<')" = "$m" ] && [ "$(changed $YAML '>')" = "$m" ] \
  && [ $(($(count_lines "$NEW/$YAML" "[$list]") - had)) = "$m" ] \
  || die "$YAML: expected $m lines cut to $N cores"
echo "== $YAML: $m dispatch lists cut to $N ETH cores"
diff "$SRC/$YAML" "$NEW/$YAML" || true

# .ld: exactly one LONG((24 * 1024) -> LONG((KB * 1024) per file
for f in $LDS; do
  c=$(grep -cF 'LONG((24 * 1024)' "$SRC/$f" || true)
  [ "$c" = 1 ] || die "$f: 'LONG((24 * 1024)' matches $c lines, need exactly 1"
  replace "$f" "s/LONG((24 \\* 1024)/LONG(($KB * 1024)/"
  [ "$(changed "$f" '<')" = 1 ] && [ "$(changed "$f" '>')" = 1 ] \
    || die "$f: sed did not change exactly one line"
  echo "== $f: idle-ERISC kernel text bound 24 KB -> $KB KB"
  diff "$SRC/$f" "$NEW/$f" || true
done
RC=2

printf 'src=%s\nn_eth=%s\n' "$SRC" "$N" > "$NEW/$MARK"
[ "$KB" = 32 ] || echo "ierisc_kb=$KB" >> "$NEW/$MARK"
[ -d "$OV" ] && rm -rf "$OV"
mv "$NEW" "$OV"
trap - EXIT
echo "overlay $OV ready (src $SRC, $N ETH dispatch cores, idle-ERISC bound $KB KB; 3 real files, rest symlinks)"
