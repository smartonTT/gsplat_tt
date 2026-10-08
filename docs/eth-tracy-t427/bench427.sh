#!/bin/bash
# t427: remote half on bh-30 (viewer stopped by drive427.sh; this script restarts it at exit via
# vstart427.sh and touches out427/vstarted). Tree p150bench/tree392 synced to the t427 head
# (best-iter-215 ff9bd5ce + zone_hash_check + ETH_IERISC_KB). Steps:
#  0) opt/profiler/zone_hash_check.py at this tree's path (#426): any collision -> no capture.
#  1) U: untraced 30 views, ETH 12x10 default through a fresh default overlay made by the new
#     make_overlay.sh (32 KB) -> md5 must be the 12x10 golden 39d84b28 (opt/md5_golden.py).
#  2) Profiler-only overlay ttm-eth12p36 (ETH_IERISC_KB=36) + own JIT cache; UP: untraced 30
#     views through it (md5 again: the 36 KB bound alone must not change bits).
#  3) Tracy, 3 chunks x 10 views (+1 warmup each), GSPLAT_TT_MATCULL_PROF=1,
#     GSPLAT_TT_PROFILE_READ_EVERY=11 (as #407/#413): arm E12 (eth 12x10), then arm E11
#     (eth, GSPLAT_TT_GRID_X=11 -> 11x10) on the same stack for a same-stack scaling check.
set -u
P=/localdev/smarton/p150bench; T=$P/tree392; V=/localdev/smarton/viewer; O=$P/out427
trap 'echo "=== bench end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/bench.rc; echo "=== viewer start (remote) $(date -u +%FT%TZ)"; bash $P/vstart427.sh; touch $O/vstarted' EXIT
rc=99
cd $T || exit 1; source .venv/bin/activate; rm -rf $O; mkdir -p $O
export TT_METAL_HOME=$V/tt-metal TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME TT_METAL_CACHE_RENDER=$P/cache427/render
export GSPLAT_TT_ETH_OVERLAY=$P/ttm-eth12-t427 GSPLAT_TT_ETH_CACHE=$P/cache427-eth
unset GSPLAT_TT_DISPATCH GSPLAT_TT_GRID_X GSPLAT_TT_GRID_Y GSPLAT_TT_MATCULL_PROF
echo "=== bench sha=$(cut -c1-8 SHA) host=$(hostname) $(date -u +%FT%TZ) card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type') aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null)"
BAD='overflows region|too large for kernel config buffer|No core coordinate|TT_FATAL|TT_THROW'
hang() { if [ $1 = 124 ] || [ $1 = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$2.log 2>&1; echo "reset rc=$?"; fi; }
hits_of() { grep -m1 "^STAGES " "$1" | sed -n 's/.* xview_hits=\([0-9]*\).*/\1/p'; }

# 0) zone hash pre-check at the path the kernels are compiled from
python3 opt/profiler/zone_hash_check.py --repo $T > $O/zone_hash_check.txt 2>&1; zh=$?
cat $O/zone_hash_check.txt
[ $zh = 0 ] || { echo "ZONE_HASH_FAIL rc=$zh: no capture"; rc=30; exit; }

untraced() {  # untraced <tag> [env...]: 30 views, md5 vs the golden of its grid
  local tag=$1 t=t427-$1 rr; shift; rm -rf tmp/$t tmp/$t-dump
  echo "=== $tag start $(date -u +%T)"
  env "$@" GSPLAT_PER_VIEW_STAGES=1 timeout ${RUN_TIMEOUT:-420} \
    python3 render/run.py --no-ref --iter-dir $t --dump-views $t-dump > $O/$tag.log 2>&1
  rr=$?; echo "run $tag rc=$rr $(date -u +%T)"
  grep -E "^\[eth\]|^\[DEV\] dispatch|^(SUMMARY|STAGES|SORT_STAGES)|$BAD|Traceback" $O/$tag.log | cut -c1-400 | head -8
  hang $rr $tag; [ $rr = 0 ] || return $rr
  (cd tmp/$t-dump && md5sum * | sort -k2) > $O/md5-$tag.txt; rm -rf tmp/$t-dump
  cp tmp/$t/hero_clean.png $O/hero-$tag.png
  python3 opt/md5_golden.py $O/md5-$tag.txt $O/$tag.log || return 20
}

# 1) default path, fresh 32 KB overlay from the new make_overlay.sh
rm -rf $GSPLAT_TT_ETH_OVERLAY $GSPLAT_TT_ETH_CACHE
untraced U || { rc=$?; exit; }
grep -qE '^\[DEV\] dispatch eth .*compute grid 12x10' $O/U.log || { echo "U_NOT_ETH12"; rc=11; exit; }
cp $GSPLAT_TT_ETH_OVERLAY/.gsplat-eth-overlay $O/overlay-default-marker.txt

# 2) profiler-only overlay (36 KB idle-ERISC bound), own JIT cache
ETH_IERISC_KB=36 bash opt/eth/make_overlay.sh $V/tt-metal $P/ttm-eth12p36 12 > $O/overlay-p36.log 2>&1 \
  || { echo "OVERLAY_P36_FAIL"; tail -3 $O/overlay-p36.log; rc=12; exit; }
tail -1 $O/overlay-p36.log
unset GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE
source opt/eth/env.sh $P/ttm-eth12p36 $P/cache427-p36 || { rc=13; exit; }
untraced UP || { rc=$?; exit; }
grep -qE '^\[DEV\] dispatch eth .*compute grid 12x10' $O/UP.log || { echo "UP_NOT_ETH12"; rc=14; exit; }

# 3) Tracy chunks
CSVX=$(ls $V/tt-metal/build*/tools/profiler/bin/csvexport-release 2>/dev/null | head -1)
rc=0
for arm in E12 E11; do
  gx=; [ $arm = E11 ] && gx=11
  for a in 0 10 20; do
    b=$((a + 10)); PR=$P/prof427/$arm-c$a; rm -rf $PR; mkdir -p $PR
    cat > $PR/inner.sh <<IN
#!/bin/bash
cd $T; source .venv/bin/activate
${gx:+GSPLAT_TT_GRID_X=$gx} GSPLAT_TT_MATCULL_PROF=1 GSPLAT_TT_PROFILE_READ_EVERY=11 python3 render/run.py --no-ref --iter-dir t427-T$arm-$a --view-range $a:$b
IN
    chmod +x $PR/inner.sh
    echo "=== tracy $arm c$a start $(date -u +%T)"
    TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$PR PYTHONPATH=$V/tt-metal/tools:${PYTHONPATH:-} \
      timeout ${TRACY_TIMEOUT:-480} python3 -m tracy -r -p -v --dump-device-data-mid-run -o $PR $PR/inner.sh > $O/T$arm-$a.log 2>&1
    tr=$?; echo "tracy $arm c$a rc=$tr $(date -u +%T)"
    grep -E "^\[DEV\] dispatch|^(SUMMARY|STAGES|SORT_STAGES)|$BAD|Traceback|DRAM buffers were full|hash" $O/T$arm-$a.log | cut -c1-400 | head -8
    echo "XVIEW_HITS $arm c$a $(hits_of $O/T$arm-$a.log)/10"
    trf=$(find $PR -name '*.tracy' | head -1); dl=$(find $PR -name profile_log_device.csv | head -1)
    [ -n "$trf" ] && [ -n "$CSVX" ] && $CSVX -u "$trf" | gzip > $O/tracy-u-$arm-c$a.csv.gz
    [ -n "$dl" ] && gzip -c "$dl" > $O/dev-$arm-c$a.csv.gz && echo "device csv $arm c$a rows $(wc -l < "$dl")"
    if [ $tr != 0 ]; then rc=$tr; hang $tr T$arm-$a; break 2; fi
  done
done
grep -m2 -h -E "firmware bundle version|KMD version" $O/U.log 2>/dev/null
ls -la $O | cut -c25-
