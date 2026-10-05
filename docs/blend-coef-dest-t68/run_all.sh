#!/bin/bash
# Task #68 device chain (run detached from the repo root): sync+build -> verify
# (k0 vs k1 md5) -> 3-round interleaved A/B -> Tracy k0,k1 -> objdump new TRISC1.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
S=/localdev/smarton/t68_scripts
L=${TTP_RUN_DIR:-/tmp}
H=yyzo-bh-07
# Each devrun is one chunk under the 400 s reservation limit (run 205: 900 s refused; run 216: no --host ran it on the Mac).
scp -q docs/blend-coef-dest-t68/remote_*.sh $H:$S/ || exit 1
ttp lock p100 -- bash -c "opt/sync_remote.sh $H /localdev/smarton/gstt2-t68 ${1:-HEAD} && $DEVRUN --host $H --no-verify --timeout 400 --tag t68-verify-k0 -- 'bash $S/remote_verify.sh 0'" 2>&1 | tee $L/t68-verify.log
rc=${PIPESTATUS[0]}; echo "VERIFY_K0_RC=$rc"
[ $rc -eq 75 ] && exit 75
[ $rc -eq 0 ] || { echo "STOP: k0 run failed"; exit 1; }
ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 400 --tag t68-verify-k1 -- "bash $S/remote_verify.sh 1" 2>&1 | tee -a $L/t68-verify.log
rc=${PIPESTATUS[0]}; echo "VERIFY_K1_RC=$rc"
[ $rc -eq 75 ] && exit 75
grep -q ALL_VIEWS_IDENTICAL $L/t68-verify.log || { echo "STOP: views not identical"; exit 1; }
: > $L/t68-ab.log
for r in 1 2 3; do
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 400 --tag t68-ab-$r -- "bash $S/remote_ab.sh $r" 2>&1 | tee -a $L/t68-ab.log
  echo "AB_${r}_RC=${PIPESTATUS[0]}"
done
for k in 0 1; do
  ttp lock p100 -- $DEVRUN --host $H --no-verify --timeout 400 --tag t68-tracy-k$k -- "bash $S/remote_tracy.sh $k" 2>&1 | tee $L/t68-tracy-k$k.log
  echo "TRACY_k${k}_RC=${PIPESTATUS[0]}"
done
ssh -o BatchMode=yes $H 'd=$(find /localdev/smarton/.cache/ttmc-gstt2-t68-k1 -path "*alpha_blend_compute_mb*" -name trisc1.elf | head -1); echo $d >&2; /localdev/smarton/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump -d $d' > $L/t68-trisc1-k1.dis
python3 docs/blend-coef-dest-t68/insn_mix.py $L/t68-trisc1-k1.dis | sort | uniq -c
echo CHAIN_DONE
