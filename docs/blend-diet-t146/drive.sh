#!/bin/bash
# t146 driver (Mac side): sync, md5 + interleaved A/B of the blend diet knobs,
# objdump of the new blend TRISC1 ELFs, and BLEND_PROF Tracy splits, all in ONE
# `ttp lock p100 -- ` hold. Stops on the first failed step.
#   drive.sh <rev> [phase ...]   phases: ab tip dis tracy (default: all)
set -o pipefail
REV=${1:?rev}; shift; PHASES=${*:-ab dis tracy}
H=yyzo-bh-07; D=/localdev/smarton/gstt2-t146
OUT=${OUT:-tmp/t146-out}; mkdir -p "$OUT"
J="T146_TREE=$D bash $D/docs/blend-diet-t146/remote_job.sh"
K=GSPLAT_TT_BLEND_CONST_HOIST; R=GSPLAT_TT_BLEND_RAW_STAGE; T=GSPLAT_TT_BLEND_FAST_TRED
OFF="off:$K=0,$R=0,$T=0"; HO="h:$K=1,$R=0,$T=0"; RA="r:$K=0,$R=1,$T=0"; TR="t:$K=0,$R=0,$T=1"
ALL="all:$K=1,$R=1,$T=1"; HT="ht:$K=1,$R=0,$T=1"; P256="p256:$K=1,$R=1,$T=1,BLEND_T_PERIOD=256u"
r() { ssh -o BatchMode=yes $H "$@"; }
step() {
  local n=$1; shift
  echo "##### $n $(date +%H:%M:%S)"
  r "$@"; local rc=$?; echo "##### $n rc=$rc"
  [ $rc = 0 ] || { echo "STOP at $n"; exit 10; }
}
echo "##### sync $(date +%H:%M:%S)"
opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; exit 1; }
for PH in $PHASES; do case $PH in
ab)
  step smoke "VIEWS=0:2 $J 0 $ALL"
  step r1 "$J 1 $OFF $HO $RA $TR $ALL $P256"
  step r2 "$J 2 $ALL $TR $RA $HO $OFF"
  step r3 "$J 3 $OFF $HO $RA $TR $ALL"
  ;;
tip)  # after the defaults flip: off vs ht vs all on the rebased tip
  step t1 "$J 11 $OFF $HT $ALL"
  step t2 "$J 12 $ALL $HT $OFF"
  step t3 "$J 13 $OFF $HT $ALL"
  ;;
dis)
  step dis "cd /localdev/smarton/.cache/ttmc-gstt2-t146 && for e in \$(find . -path '*alpha_blend_compute_mb*' -name trisc1.elf -newer $D/SHA); do d=\$(dirname \$(dirname \$e)); k=\$(grep -ho 'BLEND_\(CONST_HOIST\|RAW_STAGE\|FAST_TRED\|T_PERIOD\) [0-9a-z]*' -r \$d --include='*.h' | sort -u | tr ' \n' '=,'); echo \"ELF \$e \$k\"; /localdev/smarton/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump -d \$e > /tmp/t146-\$(echo \$k | md5sum | cut -c1-8).dis; echo \"  -> /tmp/t146-\$(echo \$k | md5sum | cut -c1-8).dis\"; done"
  scp -q -o BatchMode=yes "$H:/tmp/t146-*.dis" "$OUT/" || echo "no dis"
  ;;
tracy)
  for t in t146-off t146-all; do
    e="GSPLAT_TT_BLEND_PROF=1 $K=0 $R=0 $T=0"; [ $t = t146-all ] && e="GSPLAT_TT_BLEND_PROF=1 $K=1 $R=1 $T=1"
    step tracy-$t "$e T124_TREE=$D bash $D/docs/sort-onelaunch-v2-t124/remote_tracy.sh $t"
    for f in gaps zones deep; do
      scp -q -o BatchMode=yes $H:$D/opt/profiler/$t/$f.txt "$OUT/$t-$f.txt" || echo "no $t/$f.txt"
    done
  done
  ;;
*) echo "unknown phase $PH"; exit 9 ;;
esac; done
echo ALLDONE
