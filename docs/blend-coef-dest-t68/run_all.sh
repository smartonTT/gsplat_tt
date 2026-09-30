#!/bin/bash
# Task #68 device chain (run detached from the repo root): sync+build -> verify
# (k0 vs k1 md5) -> 3-round interleaved A/B -> Tracy k0,k1 -> objdump new TRISC1.
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
S=/localdev/smarton/t68_scripts
H=yyzo-bh-07
ttp lock p100 -- bash -c "opt/sync_remote.sh $H /localdev/smarton/gstt2-t68 ${1:-HEAD} && $DEVRUN --no-verify --timeout 900 --tag t68-verify -- 'bash $S/remote_verify.sh'" 2>&1 | tee /tmp/t68-verify.log
rc=${PIPESTATUS[0]}; echo "VERIFY_RC=$rc"
[ $rc -eq 75 ] && exit 75
grep -q ALL_VIEWS_IDENTICAL /tmp/t68-verify.log || { echo "STOP: views not identical"; exit 1; }
ttp lock p100 -- $DEVRUN --no-verify --timeout 1000 --tag t68-ab -- "bash $S/remote_ab.sh" 2>&1 | tee /tmp/t68-ab.log
echo "AB_RC=${PIPESTATUS[0]}"
for k in 0 1; do
  ttp lock p100 -- $DEVRUN --no-verify --timeout 400 --tag t68-tracy-k$k -- "bash $S/remote_tracy.sh $k" 2>&1 | tee /tmp/t68-tracy-k$k.log
  echo "TRACY_k${k}_RC=${PIPESTATUS[0]}"
done
ssh -o BatchMode=yes $H 'd=$(find /localdev/smarton/.cache/ttmc-gstt2-t68-k1 -path "*alpha_blend_compute_mb*" -name trisc1.elf | head -1); echo $d >&2; /localdev/smarton/tt-metal/runtime/sfpi/compiler/bin/riscv-tt-elf-objdump -d $d' > /tmp/t68-trisc1-k1.dis
python3 docs/blend-coef-dest-t68/insn_mix.py /tmp/t68-trisc1-k1.dis | sort | uniq -c
echo CHAIN_DONE
