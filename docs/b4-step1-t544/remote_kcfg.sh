#!/bin/bash
# t544 step 1: kcfg size of the fused mat+blend program, and per-RISC ELF sizes.
# One view (--view-range 0:1) per probe in a fresh JIT cache, so only this tree's kernels are in it:
#   def           defaults (must pass; fills the cache)
#   x26           GSPLAT_TT_KCFG_EXTRA_KB=26 (97,280 B ring)
#   ns<N>         GSPLAT_TT_PFWC_WRITER_SPLIT=0 (smaller pfwc) at KCFG_EXTRA_KB=N, N = 20 14 8 0
# A program over the ring stops the run with TT_FATAL "Program size (X) too large for kernel config
# buffer (Y)" (tt-metal program.cpp, the state.offset check), so each probe names the first program
# in launch order (pfwc, k2, sort_ol, fused mat+blend) that is larger than the ring.
# Then docs/b4-step1-t544/elfsize.py lists the per-RISC ELF load sizes in that cache.
#   remote_kcfg.sh a   (def, x26; starts from an empty cache)   remote_kcfg.sh b   (ns*, then elfsize.py)
# (run on the measurement box through devrun.sh, inside drive.sh's ttp lock p100)
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_OPACITY_PRUNE GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF \
  GSPLAT_TT_KCFG_EXTRA_KB GSPLAT_TT_PFWC_WRITER_SPLIT GSPLAT_TT_BLEND_ALIAS_CUT TT_METAL_DEVICE_PROFILER
T=${T:-/localdev/smarton/gstt2-t513}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t544; mkdir -p $S
C=/localdev/smarton/.cache/ttmc-t544-kcfg
phase=${1:?phase a|b}
[ "$phase" = a ] && rm -rf "$C"
probe() {  # tag env...
  local tag=$1; shift
  echo "=== kcfg $tag $* $(cut -c1-7 SHA) $(date +%T)"
  env "$@" TT_METAL_CACHE_RENDER=$C timeout ${RUN_TIMEOUT:-240} \
    python3 render/run.py --no-ref --view-range 0:1 --iter-dir t544-kcfg-$tag > $S/kcfg-$tag.log 2>&1
  local rc=$?
  echo "run rc=$rc"
  grep -oE "Program size \([0-9]+\) too large for kernel config buffer \([0-9]+\)" $S/kcfg-$tag.log | head -1 | sed 's/^/KCFG /'
  grep -E "^(SUMMARY|\[DEV\])" $S/kcfg-$tag.log | cut -c1-200 | head -3
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; exit 124; fi
}
if [ "$phase" = a ]; then
  probe def
  probe x26 GSPLAT_TT_KCFG_EXTRA_KB=26
  echo "=== kcfg a done $(date +%T)"; exit 0
fi
for n in 20 14 8 0; do probe ns$n GSPLAT_TT_PFWC_WRITER_SPLIT=0 GSPLAT_TT_KCFG_EXTRA_KB=$n; done
python3 docs/b4-step1-t544/elfsize.py "$C" matblend_ncrisc matblend_brisc matblend_compute \
  sort_bin_onelaunch sort_ol_town_compute | tee $S/elfsize.txt
echo "=== kcfg done $(date +%T)"
