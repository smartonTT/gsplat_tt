#!/bin/bash
# t749 gate: on-device block-float split (fix arm: sh0/sh3 T0 oracle + synthetic K=8/K=1) vs the
# t697 RISC gather (311181db, synthetic K=8/K=1), 4 rounds (order alternates), all in ONE
# `ttp lock p100`, separate JIT cache per arm and round (wiped by drive.sh), ab_jit_check per round.
#   ttp detach t749 -- docs/t749/gate.sh <fix-rev> [gather-rev]
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never bh-30,
# sync only via opt/sync_remote.sh inside the lock (docs/t713/drive.sh), StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
fix=${1:?fix rev} base=${2:-311181db}
if [ "${3:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/docs/t749/gate.sh" "$fix" "$base" --locked; fi
PRE=${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight
"$PRE" @p100 || { p=$?; echo "=== ssh preflight @p100 failed rc=$p"; exit $p; }
export H=bh-49 T2_CORES=110 REPS=${REPS:-10}
O=docs/t749/out; rm -rf "$O"; mkdir -p "$O"
git show a48aa40a:opt/ab_jit_check.py > "$O/ab_jit_check.py"
arm() {  # arm round runs
  local a=$1 r=$2 runs=$3 rev=$fix; [ "$a" = gather ] && rev=$base
  OUT=$O/r$r/$a TTMC=ttmc-t749-$a RUNS="$runs" docs/t713/drive.sh "$rev" --locked; echo "r$r $a arm rc=$?"
}
arm fix 1 "sh0 sh3 syn8 syn1"
for n in sh0 sh3; do
  grep -q '"pass": true' "$O/r1/fix/check-$n.json" 2>/dev/null || { echo "=== GATE FAIL: $n oracle did not pass"; exit 5; }
done
arm gather 1 "syn8 syn1"
for r in 2 3 4; do
  if [ $((r % 2)) = 0 ]; then arm gather $r "syn8 syn1"; arm fix $r "syn8 syn1"
  else arm fix $r "syn8 syn1"; arm gather $r "syn8 syn1"; fi
done
jrc=0
for r in 1 2 3 4; do
  python3 "$O/ab_jit_check.py" --arm gather="$O/r$r/gather/remote/jit-list.txt" --arm fix="$O/r$r/fix/remote/jit-list.txt" \
    --kernel t2_compute; c=$?; echo "r$r jit check rc=$c"; [ $c = 0 ] || jrc=$c
done
for a in fix gather; do for n in syn8 syn1; do for r in 1 2 3 4; do
  echo "$a $n r$r: $(grep -h -m1 device_ms "$O/r$r/$a/remote/out-$n/run.log" 2>/dev/null)"; done; done; done
echo "=== done $(TZ=America/Los_Angeles date)"
exit $jrc
