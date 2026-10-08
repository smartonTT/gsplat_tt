#!/usr/bin/env bash
# Local tests for opt/eth/make_overlay.sh and opt/eth/env.sh on a fake, read-only tt-metal
# tree: no device, no ssh, nothing outside a temp dir. Checks the yaml cut, exactly one
# changed line per .ld, symlinks for everything else, the source left byte-identical,
# idempotent reruns, atomic failure on pattern mismatch, and refusal of viewer paths.
# Run: bash opt/eth/test_eth_overlay.sh   (exit 0 = all passed)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MK=$HERE/make_overlay.sh ENV=$HERE/env.sh
T=$(mktemp -d "${TMPDIR:-/tmp}/eth-overlay-test.XXXXXX")
T=$(cd -P "$T" && pwd)
trap 'chmod -R u+w "$T"; rm -rf "$T"' EXIT
fail=0
ok() { echo "ok   $*"; }
bad() { echo "FAIL $*"; fail=1; }
check() { local d=$1; shift; if "$@"; then ok "$d"; else bad "$d"; fi; }

FULL='[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12], [0, 13]]'
CUT12='[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11]]'
# fake <dir> [<lines of LONG((24 * 1024) per .ld>] [<14-core yaml lists>]
fake() {
  local s=$1 nld=${2:-1} ny=${3:-6} f i
  mkdir -p "$s/tt_metal/core_descriptors" "$s/tt_metal/llrt" "$s/runtime/hw/toolchain/blackhole" \
           "$s/runtime/hw/toolchain/wormhole" "$s/runtime/sfpi/include" "$s/build_Release/lib"
  echo top > "$s/.hidden"; echo lib > "$s/build_Release/lib/libtt_metal.so"
  echo llrt > "$s/tt_metal/llrt/x.cpp"; echo arch > "$s/tt_metal/core_descriptors/blackhole_140_arch.yaml"
  echo sfpi > "$s/runtime/sfpi/include/x.h"; echo wh > "$s/runtime/hw/toolchain/wormhole/kernel_ierisc.ld"
  : > "$s/tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml"
  for i in $(seq 1 6); do
    {
      echo "p150_$i:"; echo "  dispatch_cores:"
      if [ "$i" -le "$ny" ]; then echo "    $FULL"; else echo "    [[0, 0], [0, 1]]"; fi
      echo "  dispatch_core_type: eth"
    } >> "$s/tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml"
  done
  for f in kernel_ierisc.ld kernel_subordinate_ierisc.ld firmware_ierisc.ld; do
    {
      echo "  .segments 0 (INFO) :"; echo "  {"
      echo "    LONG(ADDR(.text)) LONG(ADDR(.text))"
      for i in $(seq 1 "$nld"); do echo "    LONG((24 * 1024)"; done
      echo "         - (__fw_export_text_end - 0x3320)"; echo "         )"
      echo "   LONG(ADDR(.data)) LONG(ADDR(.data)) LONG((8 * 1024) - 192"; echo "  }"
    } > "$s/runtime/hw/toolchain/blackhole/$f"
  done
  chmod -R a-w "$s"
}
snap() { (cd "$1" && find . -print | LC_ALL=C sort | while read -r p; do
  if [ -L "$p" ]; then echo "L $p -> $(readlink "$p")"; elif [ -d "$p" ]; then echo "D $p"
  else echo "F $p $(cksum < "$p")"; fi; done); }

S=$T/src; fake "$S"; snap "$S" > "$T/src.snap"
O=$T/out/ov
Y=tt_metal/core_descriptors/blackhole_140_arch_eth_dispatch.yaml
K=runtime/hw/toolchain/blackhole/kernel_ierisc.ld
KS=runtime/hw/toolchain/blackhole/kernel_subordinate_ierisc.ld

# 1. happy path, N=12
bash "$MK" "$S" "$O" > "$T/run1.log" 2>&1; rc=$?
check "run1 exit 0" [ $rc = 0 ]
check "source tree unchanged" cmp -s "$T/src.snap" <(snap "$S")
check "yaml is a real file" eval '[ -f "$O/$Y" ] && [ ! -L "$O/$Y" ]'
check "yaml: 6 lists cut to 12" [ "$(grep -cxF "    $CUT12" "$O/$Y")" = 6 ]
check "yaml: no [0, 12] left" [ "$(grep -c '\[0, 12\]' "$O/$Y")" = 0 ]
check "yaml: other lines kept" [ "$(diff "$S/$Y" "$O/$Y" | grep -c '^[<>]')" = 12 ]
for f in $K $KS; do
  check "$(basename $f): real file" eval '[ -f "$O/$f" ] && [ ! -L "$O/$f" ]'
  check "$(basename $f): one line changed" [ "$(diff "$S/$f" "$O/$f" | grep -c '^[<>]')" = 2 ]
  check "$(basename $f): 32 KB bound" grep -qxF '    LONG((32 * 1024)' "$O/$f"
done
check "exactly 3 real files + marker" [ "$(find "$O" -type f | wc -l | tr -d ' ')" = 4 ]
for l in .hidden build_Release tt_metal/llrt tt_metal/core_descriptors/blackhole_140_arch.yaml \
         runtime/sfpi runtime/hw/toolchain/wormhole runtime/hw/toolchain/blackhole/firmware_ierisc.ld; do
  check "symlink $l -> source" [ "$(readlink "$O/$l")" = "$S/$l" ]
done
check "overlay reads like source elsewhere" cmp -s "$O/build_Release/lib/libtt_metal.so" "$S/build_Release/lib/libtt_metal.so"
check "diffs printed" eval 'grep -q "^> *LONG((32 \* 1024)" "$T/run1.log" && grep -qF "> " "$T/run1.log"'
check "no temp dirs left" [ "$(ls "$T/out")" = ov ]

# 2. idempotent
snap "$O" > "$T/ov1.snap"
bash "$MK" "$S" "$O" > "$T/run2.log" 2>&1; rc=$?
check "rerun exit 0" [ $rc = 0 ]
check "rerun gives the same tree" cmp -s "$T/ov1.snap" <(snap "$O")
check "rerun prints the same output" cmp -s "$T/run1.log" "$T/run2.log"

# 3. N=4
bash "$MK" "$S" "$T/out/ov4" 4 > /dev/null 2>&1
check "N=4 cut" [ "$(grep -cxF '    [[0, 0], [0, 1], [0, 2], [0, 3]]' "$T/out/ov4/$Y")" = 6 ]

# 3b. ETH_IERISC_KB (task #427): profiler-only overlay with a bigger idle-ERISC bound
ETH_IERISC_KB=36 bash "$MK" "$S" "$T/out/ov36" > /dev/null 2>&1; rc=$?
check "KB=36 exit 0" [ $rc = 0 ]
for f in $K $KS; do
  check "KB=36 $(basename $f): 36 KB bound" grep -qxF '    LONG((36 * 1024)' "$T/out/ov36/$f"
done
check "KB=36 marker" grep -qxF 'ierisc_kb=36' "$T/out/ov36/.gsplat-eth-overlay"
check "default marker has no ierisc_kb" [ "$(grep -c ierisc_kb "$O/.gsplat-eth-overlay")" = 0 ]
for kb in 31 49 3x 036; do
  ETH_IERISC_KB=$kb bash "$MK" "$S" "$T/out/ovkb" > /dev/null 2>&1; rc=$?
  check "KB=$kb refused (exit 2)" eval '[ $rc = 2 ] && [ ! -e "$T/out/ovkb" ]'
done

# 4. pattern mismatches fail (exit 1) and leave the existing overlay alone
for c in "0 6 .ld-0-matches" "2 6 .ld-2-matches" "1 0 yaml-no-list"; do
  set -- $c; s=$T/src-$3; fake "$s" "$1" "$2"
  cp -R "$O" "$T/out/ovm"   # copy keeps the marker: a valid old overlay
  snap "$T/out/ovm" > "$T/ovm.snap"
  bash "$MK" "$s" "$T/out/ovm" > "$T/m.log" 2>&1; rc=$?
  check "$3: exit 1" [ $rc = 1 ]
  check "$3: old overlay untouched" cmp -s "$T/ovm.snap" <(snap "$T/out/ovm")
  check "$3: no temp dir left" [ -z "$(ls -d "$T"/out/ovm.new.* 2>/dev/null)" ]
  rm -rf "$T/out/ovm"
done

# 5. refusals (exit 2), nothing created
refuse() {  # refuse <desc> <args...>
  local d=$1; shift
  bash "$MK" "$@" > "$T/r.log" 2>&1; local rc=$?
  check "refuse $d (rc=$rc)" [ $rc = 2 ]
}
refuse "viewer dir" "$S" /localdev/smarton/viewer
refuse "under viewer" "$S" /localdev/smarton/viewer/ttm-eth12
refuse "under viewer, dotted" "$S" /localdev/smarton/x/../viewer/ttm
check "nothing created under /localdev/smarton/viewer" [ ! -e /localdev/smarton/viewer/ttm-eth12 ]
mkdir -p "$T/fakeviewer" "$T/out"; ln -s "$T/fakeviewer" "$T/out/vlink"
ETH_FORBID_EXTRA=$T/fakeviewer refuse "symlink into a forbidden tree" "$S" "$T/out/vlink/ov"
check "nothing created in the forbidden tree" [ -z "$(ls -A "$T/fakeviewer")" ]
refuse "inside the source" "$S" "$S/ov"
refuse "N=0" "$S" "$T/out/x" 0
refuse "N=14" "$S" "$T/out/x" 14
refuse "N=abc" "$S" "$T/out/x" abc
mkdir -p "$T/out/notours"; echo keep > "$T/out/notours/f"
refuse "non-empty dir without marker" "$S" "$T/out/notours"
check "non-overlay dir untouched" [ "$(cat "$T/out/notours/f")" = keep ]
refuse "missing tt-metal" "$T/nope" "$T/out/x"
check "source tree still unchanged" cmp -s "$T/src.snap" <(snap "$S")

# 6. env.sh
out=$(bash -c '. "$1" "$2" && echo "$TT_METAL_RUNTIME_ROOT|$TT_METAL_CACHE|$TT_METAL_CACHE_RENDER"' _ "$ENV" "$O" 2>/dev/null)
check "env.sh sourced exports" [ "$out" = "$O|$O-cache/prod|$O-cache/render" ]
out=$(bash "$ENV" "$O" "$T/c" -- sh -c 'echo "$TT_METAL_RUNTIME_ROOT|$TT_METAL_CACHE"' 2>/dev/null)
check "env.sh exec mode" [ "$out" = "$O|$T/c/prod" ]
envrefuse() {  # envrefuse <desc> <args...>
  local d=$1; shift
  bash -c '. "$@"; r=$?; [ $r = 2 ] && [ -z "${TT_METAL_RUNTIME_ROOT:-}" ]' _ "$ENV" "$@" 2>/dev/null
  check "env.sh refuses $d (sourced)" [ $? = 0 ]
  bash "$ENV" "$@" -- true 2>/dev/null
  check "env.sh refuses $d (exec, rc=2)" [ $? = 2 ]
}
envrefuse "overlay under viewer" /localdev/smarton/viewer/ttm-eth12
envrefuse "cache under viewer" "$O" /localdev/smarton/viewer/tt-metal-cache
envrefuse "viewer default cache" "$O" /localdev/smarton/.cache/tt-metal-cache-viewer
envrefuse "shared render cache" "$O" /localdev/smarton/.cache/tt-metal-cache-render
envrefuse "cache inside overlay" "$O" "$O/cache"
envrefuse "dir without marker" "$T/out/notours"
ETH_FORBID_EXTRA=$T/fakeviewer envrefuse "cache via symlink into forbidden tree" "$O" "$T/out/vlink/c"
bash "$ENV" "$O" 2>/dev/null; check "env.sh exec without -- is a usage error" [ $? = 2 ]

[ $fail = 0 ] && echo "ALL PASSED" || echo "SOME FAILED"
exit $fail
