#!/bin/bash
# t506 (remote, bh-30, viewer stopped by drive506.sh): b2b Tracy timeline of best-iter-223 defaults.
#   bench506.sh <out dir> <tree>
# 1) untraced b2b round (run.py --back-to-back, 20 passes): raw_md5 (want d9a60a3c), B2B_STAGES,
#    device hero_clean.png.
# 2) plain Tracy: 6 processes of 5 views (0:5 .. 25:30), each warm hero + check pass + 1 measured
#    b2b pass = 11 renders; the profiler is read once, after the 11th (GSPLAT_TT_PROFILE_READ_EVERY=11),
#    so nothing stalls the device between views and the xview overlap stays live.
# 3) GSPLAT_TT_MATCULL_PROF=1 Tracy (mj_wait/mj_job zones), 3 processes (0:5 10:15 20:25), same shape.
# 4) untraced b2b round again (drift check).
# aiclk: tt_aiclk sysfs sampled every 0.1 s for the whole bench (aiclk.txt: epoch_s mhz).
# With VSTART set it runs that viewer start script at exit and touches $O/vstarted.
set -u
O=$1; T=$2; rc=99
P=/localdev/smarton/p150bench; V=/localdev/smarton/viewer; CACHE=/localdev/smarton/p150bench/cache493
rm -rf $O; mkdir -p $O
SYS='/sys/class/tenstorrent/tenstorrent!0'
( while :; do echo "$(date +%s.%N | cut -c1-14) $(cat "$SYS/tt_aiclk" 2>/dev/null)"; sleep 0.1; done ) > $O/aiclk.txt 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; echo "=== bench end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/bench.rc
  if [ -n "${VSTART:-}" ]; then bash $VSTART > $O/vstart.log 2>&1; cat $O/vstart.log; touch $O/vstarted; fi' EXIT
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES \
  GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP GSPLAT_TT_PFWC_DEAL GSPLAT_TT_CHUNK_CULL GSPLAT_TT_OL_RING \
  GSPLAT_TT_OL_RING_DEPTH GSPLAT_TT_CHUNK_SCATTER GSPLAT_TT_PFWC_ACQ_FUSE GSPLAT_TT_PFWC_SKIP_RGB GSPLAT_TT_PFWC_VIS_FAST \
  GSPLAT_TT_MATCULL_PROF TT_METAL_DEVICE_PROFILER GSPLAT_TT_PROFILE GSPLAT_TT_PROFILE_READ_EVERY TT_METAL_PROFILER_DIR
export TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export TT_METAL_HOME=$V/tt-metal; export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
{ echo "=== host $(hostname) $(date -u +%FT%TZ) nproc=$(nproc) sha=$(cat $T/SHA)"
  echo "card=$(cat "$SYS/tt_card_type" 2>/dev/null) aiclk_idle=$(cat "$SYS/tt_aiclk") load=$(cat /proc/loadavg)"; } | tee $O/host.txt
mark() { echo "$(date +%s.%N | cut -c1-14) $*" >> $O/marks.txt; }
b2b() {  # tag
  local tag=$1 r
  echo "=== $tag b2b $(date -u +%T) load=$(cut -d' ' -f1 /proc/loadavg)"; mark "$tag start"
  ( cd $T || exit 9; source .venv/bin/activate; rm -rf tmp/t506-$tag
    TT_METAL_CACHE_RENDER=$CACHE/render timeout 300 python3 render/run.py --no-ref --back-to-back \
      --iter-dir t506-$tag > $O/run-$tag.log 2>&1 ); r=$?; mark "$tag end"
  echo "$tag rc=$r"; grep -E "^B2B scene|TT_FATAL|TT_THROW|Traceback" $O/run-$tag.log | cut -c1-400
  [ $r = 124 ] && { echo HANG; tt-smi -r > $O/reset-$tag.log 2>&1; rc=124; exit 124; }
  [ $r = 0 ] || { rc=20; return; }
  cp $T/tmp/t506-$tag/hero_clean.png $O/hero-$tag.png 2>/dev/null; rm -rf $T/tmp/t506-$tag
}
CSVX=$(ls $V/tt-metal/build*/tools/profiler/bin/csvexport-release 2>/dev/null | head -1)
tracy() {  # tag a b extra-env
  local tag=$1 a=$2 b=$3 extra=$4 PR=/localdev/smarton/p150bench/prof506/$1 tr trf dl
  rm -rf $PR; mkdir -p $PR
  cat > $PR/inner.sh <<IN
#!/bin/bash
cd $T; source .venv/bin/activate
env $extra GSPLAT_TT_PROFILE_READ_EVERY=11 python3 render/run.py --no-ref \
  --iter-dir t506-$tag --view-range $a:$b --back-to-back --b2b-passes 1 --b2b-warmup 0
IN
  chmod +x $PR/inner.sh
  echo "=== tracy $tag $a:$b [$extra] $(date -u +%T)"; mark "$tag start"
  # profiler cq_prefetch idle_erisc ELF overflows the default 32 KB ETH overlay: use the
  # profiler-only overlay ttm-eth12p36 (ETH_IERISC_KB=36, made by #427) with its own JIT cache.
  ( cd $T; source .venv/bin/activate
    unset GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE
    source opt/eth/env.sh $P/ttm-eth12p36 $P/cache506-p36${extra:+mcp} || exit 13
    TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$PR PYTHONPATH=$V/tt-metal/tools:${PYTHONPATH:-} \
      timeout 420 python3 -m tracy -r -p -v --dump-device-data-mid-run -o $PR $PR/inner.sh > $O/T-$tag.log 2>&1 ); tr=$?
  mark "$tag end"; echo "tracy $tag rc=$tr $(date -u +%T)"
  grep -E "^B2B scene|^B2B_STAGES|Traceback|TT_FATAL|DRAM buffers were full|buffer.*full" $O/T-$tag.log | cut -c1-300 | head -6
  [ $tr = 124 ] && { tt-smi -r > $O/reset-$tag.log 2>&1; rc=124; exit 124; }
  [ $tr = 0 ] || rc=21
  grep -qE '^\[DEV\] dispatch eth .*compute grid 12x10' $O/T-$tag.log || echo "T-$tag NOT_ETH12"
  grep -m1 -h "CHIP_FREQ\|ARCH" $(find $PR -name profile_log_device.csv | head -1) 2>/dev/null | cut -c1-200
  trf=$(find $PR -name '*.tracy' | head -1); dl=$(find $PR -name profile_log_device.csv | head -1)
  [ -n "$trf" ] && [ -n "$CSVX" ] && $CSVX -u "$trf" | gzip > $O/tracy-u-$tag.csv.gz
  [ -n "$dl" ] && gzip -c "$dl" > $O/dev-$tag.csv.gz && echo "device csv $tag rows $(wc -l < "$dl")"
  rm -rf $T/tmp/t506-$tag
}
rc=0
[ -n "$SKIP_U1" ] || b2b u1
for a in 0 5 10 15 20 25; do tracy P$a $a $((a + 5)) ""; done
for a in 0 10 20; do tracy M$a $a $((a + 5)) "GSPLAT_TT_MATCULL_PROF=1"; done
b2b u2
grep -m2 -h -E "firmware bundle version|KMD version" $O/run-u1.log 2>/dev/null
ls -la $O | cut -c25-
