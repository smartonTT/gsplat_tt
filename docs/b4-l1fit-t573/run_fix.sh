#!/bin/bash
# t573: verify the BRISC stack fix through t555's drive.sh (each call its own ttp lock p100, bh-33
# measurement reservation only, StrictHostKeyChecking=yes shims):
#   1) PLAN=check ARM=fuse: 30-view md5 of sort-in-blend + one b2b round base/fuse (stop on failure)
#   2) static stack margin of the new matblend BRISC ELFs (opt/brisc_stack_check.py on bh-33)
#   3) PLAN=check ARM=base: production md5 with the muldiv change
#   4) PLAN=ab ARM=fuse: device hero + diff + PSNR, warm-up, 3 alternating b2b rounds base/fuse
#   ttp detach t573fix -- docs/b4-l1fit-t573/run_fix.sh <rev>
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}; D=docs/b4-l1fit-t573; H=${H:-bh-33}; T=${T:-/localdev/smarton/gstt2-t513}
PLAN=check ARM=fuse O=$D/out-fix-check docs/b4-step3b-t555/drive.sh "$rev" || { echo "=== check fuse rc=$?"; exit 11; }
echo "=== stack check"
/usr/bin/ssh -o StrictHostKeyChecking=yes -o BatchMode=yes "$H" \
  "cd $T && for e in \$(find /localdev/smarton/.cache/ttmc-gstt2-t513 -path '*matblend_brisc*' -name brisc.elf -mmin -60); do python3 opt/brisc_stack_check.py \$e --reserve 0; done" \
  > $D/out-fix-check/stack.jsonl 2>&1; echo "stack rc=$?"; cut -c1-160 $D/out-fix-check/stack.jsonl
PLAN=check ARM=base O=$D/out-fix-base docs/b4-step3b-t555/drive.sh "$rev" || { echo "=== check base rc=$?"; exit 12; }
PLAN=ab ARM=fuse O=$D/out-fix-ab docs/b4-step3b-t555/drive.sh "$rev" || { echo "=== ab rc=$?"; exit 13; }
echo "=== t573 fix verify done"
