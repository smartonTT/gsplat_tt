#!/bin/bash
# t142 driver (Mac side): device A/B of lever C (GSPLAT_TT_PRECULL=1, task #140)
# on top of tip, in ONE `ttp lock p100` hold (t121 pattern: every step stops the
# chain on failure). Uses the t124 remote_job.sh / remote_tracy.sh and the t115
# remote_time.sh (dead-pair count) on a t142 tree.
#   ttp lock p100 -- bash docs/precull-t142/drive.sh <rev> [phase ...]
#   phases: smoke ab psnr t0 dc tracy size (default: all but psnr, t0)
set -o pipefail
REV=${1:?rev}; shift; PHASES=${*:-smoke ab dc tracy size}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t142
OUT=${OUT:-docs/precull-t142/out}; mkdir -p "$OUT"
J="T124_TREE=$D bash $D/docs/sort-onelaunch-v2-t124/remote_job.sh"
PC=pc:GSPLAT_TT_PRECULL=1
r() { ssh -o BatchMode=yes $H "$@"; }
step() {  # name cmd...: run one remote step, stop the chain on failure
  local n=$1; shift
  echo "##### $n $(date +%H:%M:%S)"
  r "$@"; local rc=$?; echo "##### $n rc=$rc"
  [ $rc = 0 ] || { echo "STOP at $n"; exit 10; }
}
echo "##### sync $(date +%H:%M:%S)"
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; exit 1; }
for PH in $PHASES; do case $PH in
smoke)  # first compile of PFWC_PRECULL; 5 views, md5 must match
  out=$(r "VIEWS=0:5 TMO=600 $J s $PC"); rc=$?; echo "$out"; echo "smoke rc=$rc"
  [ $rc = 0 ] || { echo SMOKE_FAIL; exit 2; }
  # 1-LSB diffs are expected (blend T early-out cadence, see README); psnr/t0 judge them
  echo "$out" | grep -q 'ALL_VIEWS_IDENTICAL' || echo SMOKE_NOT_IDENTICAL
  ;;
ab)
  step r1 "$J 1 base $PC"
  step r2 "$J 2 $PC base"
  step r3 "$J 3 base $PC"
  step r4 "$J 4 $PC base"
  ;;
dc)  # dead-record share (hero slab dump -> dead_pairs.py) without / with the pre-cull
  step dc "rm -rf $D/tmp/t115/dumpcull; T115_TREE=$D bash $D/docs/reprofile-t115/remote_time.sh 5 dc; \
rm -rf $D/tmp/t115/dumpcull; GSPLAT_TT_PRECULL=1 T115_TREE=$D bash $D/docs/reprofile-t115/remote_time.sh 6 dc"
  ;;
tracy)
  for t in t142-base t142-pc; do
    pc=0; [ $t = t142-pc ] && pc=1
    step tracy-$t "GSPLAT_TT_PRECULL=$pc T124_TREE=$D bash $D/docs/sort-onelaunch-v2-t124/remote_tracy.sh $t"
    for f in gaps zones deep; do
      scp -q -o BatchMode=yes $H:$D/opt/profiler/$t/$f.txt "$OUT/$t-$f.txt" || echo "no $t/$f.txt"
    done
  done
  ;;
psnr)  # pre-cull vs base pixel diff of each ab round (smoke showed max 1 LSB diffs)
  for k in 1 2 3 4; do
    r "cd $D && source .venv/bin/activate && python3 docs/precull-t142/imgdiff.py tmp/t124-dump-t124r$k-base tmp/t124-dump-t124r$k-pc" \
      | awk -v k=$k '{print "r" k " " $0}'
  done
  ;;
t0)  # T-saturation early-out off (BLEND_T_PERIOD=0): its readback cadence counts dead
      # records too, so pre-cull shifts it; with it off the two arms must be byte-identical
  step t0 "VIEWS=0:5 TMO=600 $J t0 b0:BLEND_T_PERIOD=0 p0:BLEND_T_PERIOD=0,GSPLAT_TT_PRECULL=1"
  r "cd $D && source .venv/bin/activate && python3 docs/precull-t142/imgdiff.py tmp/t124-dump-t124rt0-b0 tmp/t124-dump-t124rt0-p0" | sed 's/^/t0 /'
  ;;
size)  # PFWC_VIS was ~66 KB of the 70.6 KB kernel config buffer
  r "find /localdev/smarton/.cache/ttmc-gstt2-t142 -name '*.elf' -path '*pfwc*' -newer $D/SHA | xargs -r ls -l | sort -k5 -n | tail -12"
  ;;
*) echo "unknown phase $PH"; exit 9 ;;
esac; done
echo ALLDONE
