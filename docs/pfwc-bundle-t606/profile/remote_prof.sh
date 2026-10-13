#!/bin/bash
# t608 (remote, bh-33, via devrun.sh inside drive_prof.sh's ttp lock p100): one device-profiler
# capture of views a:b for one arm, the t595 "tracy" mode of docs/profile-t227/bench.sh
# (ttp/t595-profile-227) with GSPLAT_TT_B4A_FOLD=0 (so the traced programs fit the profiler kcfg).
#   remote_prof.sh <arm> <a> <b> [stepcyc]
#   arm: base = defaults (best-iter-227), new = RECFAST=1 VIS_FAST=1, vis = VIS_FAST=1 only,
#        rec = RECFAST=1 only. stepcyc adds GSPLAT_TT_PFWC_STEPCYC=1 (all five RISCs, task #197).
#   One JIT cache per arm and mode (ttmc-t608<arm>-prof[-sc]), off /home.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P100} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_MAT_EARLY_CULL GSPLAT_TT_MATCULL_PROF TT_METAL_DEVICE_PROFILER \
  GSPLAT_TT_PROFILE GSPLAT_TT_PROFILE_READ_EVERY TT_METAL_PROFILER_DIR GSPLAT_TT_K2_PROF GSPLAT_TT_OL_EMIT_PROF \
  GSPLAT_TT_BLEND_PROF GSPLAT_VIEW_COUNTS GSPLAT_B2B_ALL_STAGES GSPLAT_TT_B4A_FOLD GSPLAT_TT_KCFG_EXTRA_KB \
  GSPLAT_TT_PFWC_RECFAST GSPLAT_TT_PFWC_VIS_FAST GSPLAT_TT_PFWC_STEPCYC GSPLAT_TT_PFWC_STEPRISC
T=${T:?T}; cd "$T" || exit 1; source .venv/bin/activate
S=$T/tmp/t608; mkdir -p $S
arm=${1:?arm}; a=${2:?a}; b=${3:?b}; sc=${4:-}
case $arm in base) e=0; v=0 ;; new) e=1; v=1 ;; vis) e=0; v=1 ;; rec) e=1; v=0 ;; *) echo "bad arm $arm"; exit 2 ;; esac
env_="GSPLAT_TT_B4A_FOLD=0 GSPLAT_TT_PFWC_RECFAST=$e GSPLAT_TT_PFWC_VIS_FAST=$v"
tag=$arm-$a; cache=/localdev/smarton/.cache/ttmc-t608$arm-prof
if [ "$sc" = stepcyc ]; then
  env_="$env_ GSPLAT_TT_PFWC_STEPCYC=1 GSPLAT_TT_PFWC_STEPRISC=9 ${KX:+GSPLAT_TT_KCFG_EXTRA_KB=$KX}"
  tag=$tag-sc; cache=$cache-sc
fi
hang() { echo "HANG in $1: tt-smi -r"; tt-smi -r > $S/reset-$1.log 2>&1; echo "reset rc=$?"; exit 124; }
SYS='/sys/class/tenstorrent/tenstorrent!0'
PR=$S/prof-$tag; rm -rf $PR; mkdir -p $PR
cat > $PR/inner.sh <<IN
#!/bin/bash
cd $T; source .venv/bin/activate
env $env_ TT_METAL_CACHE_RENDER=$cache GSPLAT_TT_PROFILE_READ_EVERY=11 \
  python3 render/run.py --no-ref --iter-dir t608-$tag --view-range $a:$b --back-to-back --b2b-passes 1 --b2b-warmup 0
IN
chmod +x $PR/inner.sh
echo "=== tracy $tag $(cut -c1-8 SHA) $a:$b [$env_] $(date +%T) card=$(cat "$SYS/tt_card_type" 2>/dev/null) load=$(cut -d' ' -f1-3 /proc/loadavg)"
# Device profiler only, as t595: the env tracy's -p --dump-device-data-mid-run sets, no capture server.
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_NO_CACHE_OP_INFO=1 \
  TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$PR \
  timeout ${TRACY_TIMEOUT:-300} $PR/inner.sh > $S/T-$tag.log 2>&1
rc=$?; echo "tracy $tag rc=$rc $(date +%T)"
{ [ $rc = 124 ] || [ $rc = 137 ]; } && hang $tag
grep -E "^(B2B |B2B_STAGES pass=0|SUMMARY)|Traceback|TT_FATAL|DRAM buffers were full" $S/T-$tag.log | cut -c1-300 | head -8
dl=$(find $PR -name profile_log_device.csv | head -1)
[ -n "$dl" ] || { echo "no device csv $tag"; rm -rf $PR tmp/t608-$tag; exit 2; }
gzip -c "$dl" > $S/dev-$tag.csv.gz
echo "device csv $tag rows $(wc -l < "$dl")"
rm -rf $PR tmp/t608-$tag
exit $rc
