#!/bin/bash
# t726 gate: fix arm (sh0, sh3 oracle + synthetic K=8/K=1) and the t697 RISC-gather arm (synthetic K=8/K=1)
# on bh-49 p150a, 110 cores, one `ttp lock p100` for both, separate JIT caches, then ab_jit_check.
#   ttp detach t726 -- docs/t726/gate.sh <fix-rev> <gather-rev>
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
fix=${1:?fix rev} base=${2:?gather rev}
if [ "${3:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/docs/t726/gate.sh" "$fix" "$base" --locked; fi
export H=bh-49 T2_CORES=110 REPS=${REPS:-10}
O=docs/t726/out; rm -rf "$O"; mkdir -p "$O"
OUT=$O/fix TTMC=ttmc-t726-fix RUNS="sh0 sh3 syn8 syn1" docs/t713/drive.sh "$fix" --locked; echo "fix arm rc=$?"
OUT=$O/gather TTMC=ttmc-t726-gather RUNS="syn8 syn1" docs/t713/drive.sh "$base" --locked; echo "gather arm rc=$?"
for a in fix gather; do for n in syn8 syn1; do
  echo "$a $n: $(grep -h -m1 device_ms "$O/$a/remote/out-$n/run.log" 2>/dev/null)"; done; done
git show a48aa40a:opt/ab_jit_check.py > "$O/ab_jit_check.py"
python3 "$O/ab_jit_check.py" --arm gather="$O/gather/remote/jit-list.txt" --arm fix="$O/fix/remote/jit-list.txt" --kernel t2_compute
echo "jit check rc=$?"
