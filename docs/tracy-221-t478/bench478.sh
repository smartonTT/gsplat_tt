#!/bin/bash
# t478 (remote, bh-30; viewer stopped by drive478.sh, restarted at exit by vstart478.sh, then out478/vstarted).
# Tracy capture for kept iter 221 (best-iter-221, REC32 default) at its config: ETH 12x10 default dispatch,
# 30 bicycle views, same flags as opt/profiler/capture_tracy.sh (run.py --no-ref, per-render profiler read).
# Profiler-only overlay ttm-eth12p36 (ETH_IERISC_KB=36, #427: the profiler build of the idle-ERISC prefetch
# overflows the 32 KB bound) with a fresh JIT cache of its own. Steps:
#  0) opt/profiler/zone_hash_check.py at this tree's path (#416/#426): any collision -> no capture.
#  1) UP: untraced 30 views through the p36 overlay: ETH 12x10, "rec32 on", md5 = 12x10 golden 39d84b28.
#  2) T: one Tracy capture of all 30 views (+ warmup). If it fails, 3 chunks of 10 views (as #427).
set -u
P=/localdev/smarton/p150bench; T=$P/tree478; V=/localdev/smarton/viewer; O=$P/out478; PR=$P/prof478
rc=99
trap 'echo "=== bench end rc=$rc $(date -u +%FT%TZ)"; echo $rc > $O/bench.rc; echo "=== viewer start (remote) $(date -u +%FT%TZ)"; bash $P/vstart478.sh; touch $O/vstarted' EXIT
cd $T || exit 1; source .venv/bin/activate; rm -rf $O $PR; mkdir -p $O $PR
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE GSPLAT_PER_VIEW_STAGES GSPLAT_TT_GRID_X \
  GSPLAT_TT_GRID_Y GSPLAT_TT_MATCULL_PROF GSPLAT_TT_PROFILE_READ_EVERY GSPLAT_TT_HOST_PROFILE GSPLAT_TT_XVIEW_OVERLAP \
  GSPLAT_TT_PFWC_DEAL GSPLAT_TT_PERM_NOC GSPLAT_TT_SORT_PACKED GSPLAT_TT_PFWC_REC32
export TT_METAL_HOME=$V/tt-metal TT_METAL_ARCH_NAME=blackhole TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
export TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME
echo "=== bench sha=$(cut -c1-8 SHA) host=$(hostname) $(date -u +%FT%TZ) card=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_card_type') aiclk=$(cat '/sys/class/tenstorrent/tenstorrent!0/tt_aiclk' 2>/dev/null) load=$(cut -d' ' -f1-3 /proc/loadavg)"
BAD='overflows region|too large for kernel config buffer|No core coordinate|TT_FATAL|TT_THROW'
hang() { if [ $1 = 124 ] || [ $1 = 137 ]; then echo "HANG: tt-smi -r"; tt-smi -r > $O/reset-$2.log 2>&1; echo "reset rc=$?"; fi; }

# 0) zone hash pre-check at the path the kernels are compiled from
python3 opt/profiler/zone_hash_check.py --repo $T > $O/zone_hash_check.txt 2>&1; zh=$?
tail -3 $O/zone_hash_check.txt
[ $zh = 0 ] || { echo "ZONE_HASH_FAIL rc=$zh: no capture"; rc=30; exit; }

# 1) untraced through the profiler-only overlay (own fresh JIT cache)
[ -f $P/ttm-eth12p36/.gsplat-eth-overlay ] || { echo "NO_P36_OVERLAY"; rc=12; exit; }
rm -rf $P/cache478-p36
source opt/eth/env.sh $P/ttm-eth12p36 $P/cache478-p36 || { rc=13; exit; }
t=t478-UP; rm -rf tmp/$t tmp/$t-dump
echo "=== UP start $(date -u +%T)"
timeout 420 python3 render/run.py --no-ref --iter-dir $t --dump-views $t-dump > $O/UP.log 2>&1; rr=$?
echo "run UP rc=$rr $(date -u +%T)"
grep -E "^\[eth\]|^\[DEV\] dispatch|rec32|^(SUMMARY|STAGES|SORT_STAGES)|TTW_TIMING ms_view=|$BAD|Traceback" $O/UP.log | cut -c1-300 | head -10
hang $rr UP; [ $rr = 0 ] || { rc=$rr; exit; }
grep -qE '^\[DEV\] dispatch eth .*compute grid 12x10' $O/UP.log || { echo "UP_NOT_ETH12"; rc=14; exit; }
grep -q 'rec32 on' $O/UP.log || { echo "UP_NO_REC32"; rc=15; exit; }
(cd tmp/$t-dump && md5sum * | sort -k2) > $O/md5-UP.txt; rm -rf tmp/$t-dump
cp tmp/$t/hero_clean.png $O/hero-UP.png
echo "LIST_MD5 UP $(md5sum < $O/md5-UP.txt | cut -c1-8)"
python3 opt/md5_golden.py $O/md5-UP.txt $O/UP.log || { rc=20; exit; }

# 2) Tracy (canonical: all 30 views in one process, profiler read after every render)
CSVX=$(ls $V/tt-metal/build*/tools/profiler/bin/csvexport-release 2>/dev/null | head -1)
capture() {  # capture <tag> [START:END]
  local tag=$1 range=${2:-} d=$PR/$1 tr trf dl
  mkdir -p $d
  cat > $d/inner.sh <<IN
#!/bin/bash
cd $T; source .venv/bin/activate
python3 render/run.py --no-ref --iter-dir t478-$tag ${range:+--view-range $range}
IN
  chmod +x $d/inner.sh
  echo "=== tracy $tag start $(date -u +%T)"
  TT_METAL_DEVICE_PROFILER=1 GSPLAT_TT_PROFILE=1 TT_METAL_PROFILER_DIR=$d PYTHONPATH=$V/tt-metal/tools:${PYTHONPATH:-} \
    timeout ${TRACY_TIMEOUT:-480} python3 -m tracy -r -p -v --dump-device-data-mid-run -o $d $d/inner.sh > $O/T$tag.log 2>&1
  tr=$?; echo "tracy $tag rc=$tr $(date -u +%T)"
  grep -E "^\[DEV\] dispatch|rec32|^(SUMMARY|STAGES)|$BAD|Traceback|DRAM buffers were full|hash" $O/T$tag.log | cut -c1-300 | head -8
  trf=$(find $d -name '*.tracy' | head -1); dl=$(find $d -name profile_log_device.csv | head -1)
  [ -n "$trf" ] && [ -s "$trf" ] && cp "$trf" $O/render-$tag.tracy && echo "tracy file $tag $(stat -c %s "$trf") B"
  [ -n "$trf" ] && [ -n "$CSVX" ] && $CSVX -u "$trf" | gzip > $O/tracy-u-$tag.csv.gz
  [ -n "$dl" ] && gzip -c "$dl" > $O/dev-$tag.csv.gz && echo "device csv $tag rows $(wc -l < "$dl")"
  hang $tr T$tag
  [ $tr = 0 ] && [ -s $O/render-$tag.tracy ] && [ -n "$dl" ] && ! grep -qE "$BAD|DRAM buffers were full" $O/T$tag.log
}
if capture full; then
  rc=0
else
  echo "FULL_FAIL: chunked fallback"
  rc=0
  for a in 0 10 20; do capture c$a $a:$((a + 10)) || { rc=40; break; }; done
fi
grep -m2 -h -E "firmware bundle version|KMD version" $O/UP.log 2>/dev/null
ls -la $O | cut -c25-
