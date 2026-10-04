#!/bin/bash
# t161 driver (Mac side): device A/B of the pixel-centre pre-cull (GSPLAT_TT_PRECULL=2,
# task #157) vs lever C as built (1, the default since #156) vs off (0).
# Each phase is short; run each phase in its own lock hold:
#   ttp lock p100 -- bash docs/precull-pc-t161/drive.sh <rev> sync smoke size
#   ttp lock p100 -- bash docs/precull-pc-t161/drive.sh <rev> r1   (r2, r3, t0 psnr, dc, tracy0/1/2)
# Phases: sync smoke size r1 r2 r3 psnr t0 dc tracy0 tracy1 tracy2. The md5 golden
# is the PRECULL=1 output (md5-r82new.txt since #156 = docs/precull-default-t156/out/md5-t156r1-base.txt).
set -o pipefail
REV=${1:?rev}; shift; PHASES=${*:?phases}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t161
OUT=docs/precull-pc-t161/out; mkdir -p "$OUT"
J="T161_TREE=$D bash $D/docs/precull-pc-t161/remote_job.sh"
OFF=off:GSPLAT_TT_PRECULL=0; PC=pc:GSPLAT_TT_PRECULL=2
r() { ssh -o BatchMode=yes $H "$@"; }
step() {  # name cmd...: one remote step, stop the chain on failure
  local n=$1; shift
  echo "##### $n $(date +%H:%M:%S)"
  r "$@"; local rc=$?; echo "##### $n rc=$rc"
  [ $rc = 0 ] || { echo "STOP at $n"; exit 10; }
}
for PH in $PHASES; do case $PH in
sync) echo "##### sync $(date +%H:%M:%S)"
  opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; exit 1; }
  r "cat $D/SHA" | grep -q "$(git rev-parse $REV)" || { echo "SHA mismatch"; exit 1; } ;;
smoke)  # first compile of PFWC_PRECULL+PRECULL_PC: a kernel config buffer overflow throws here
  step smoke "VIEWS=0:3 TMO=600 $J 0 base $PC" ;;
size)  # fused pfwc ELFs (kernel config buffer is 70656 B)
  r "find /localdev/smarton/.cache/ttmc-gstt2-t161 -name '*.elf' -path '*pfwc*' | xargs -r ls -l | sort -k5 -n | tail -12" ;;
r1) step r1 "$J 1 $OFF base $PC" ;;
r2) step r2 "$J 2 base $PC $OFF" ;;
r3) step r3 "$J 3 $PC $OFF base" ;;
psnr)  # per-view diff vs the default (base = PRECULL=1)
  for k in 1 2 3; do for a in pc off; do
    r "cd $D && source .venv/bin/activate && python3 docs/precull-t142/imgdiff.py tmp/t161-dump-t161r$k-base tmp/t161-dump-t161r$k-$a" \
      | awk -v k=$k -v a=$a '{print "psnr " a " r" k " " $0}'
  done; done
  r "cd /localdev/smarton/t161_scripts && md5sum md5-t161r?-pc.txt md5-t161r?-base.txt" ;;
t0)  # T early-out off: mode 2 must be byte-identical to mode 1 and 0
  step t0 "VIEWS=0:5 TMO=600 $J t0 b0:BLEND_T_PERIOD=0 p0:BLEND_T_PERIOD=0,GSPLAT_TT_PRECULL=2 o0:BLEND_T_PERIOD=0,GSPLAT_TT_PRECULL=0"
  for a in p0 o0; do
    r "cd $D && source .venv/bin/activate && python3 docs/precull-t142/imgdiff.py tmp/t161-dump-t161rt0-b0 tmp/t161-dump-t161rt0-$a" | sed "s/^/t0 $a /"
  done ;;
dc)  # dead-record share (hero slab dump -> dead_pairs.py), modes 0 / 1 / 2
  for m in 0 1 2; do
    step dc$m "rm -rf $D/tmp/t115/dumpcull; GSPLAT_TT_PRECULL=$m T115_TREE=$D bash $D/docs/reprofile-t115/remote_time.sh dc$m dc"
  done ;;
tracy0|tracy1|tracy2) m=${PH#tracy}; t=t161-m$m
  step $PH "GSPLAT_TT_PRECULL=$m T124_TREE=$D bash $D/docs/sort-onelaunch-v2-t124/remote_tracy.sh $t"
  for f in gaps zones deep; do
    scp -q -o BatchMode=yes $H:$D/opt/profiler/$t/$f.txt "$OUT/$t-$f.txt" || echo "no $t/$f.txt"
  done ;;
*) echo "unknown phase $PH"; exit 9 ;;
esac; done
echo ALLDONE
