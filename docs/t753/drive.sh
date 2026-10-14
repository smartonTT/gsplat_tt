#!/bin/bash
# t753 (copy of docs/t743/drive.sh): K-column strips of the fused L1 + D-SSIM loss (LOSS_MODE=ssim,
# LOSS_SSIM_K). Timing rounds also print the md5 of each arm's syn grad.f32 (bit-identity). ONE `ttp lock p100`:
#   ttp detach t753 -- docs/t753/drive.sh <rev>       (Mac, worktree root)
# ARMS="name:VAR=v,VAR=v name2:..." timing arms (own JIT caches), ROUNDS alternated rounds on the 1237x822
# syn case. CHECK=<arm name>: also run the 6 T0 oracle cases with that arm's env and check them on the Mac.
# t753 run 2 (fused V pass): ARMS="k1:LOSS_SSIM_K=1 k1vf:LOSS_SSIM_K=1,LOSS_SSIM_VF=1 k4:LOSS_SSIM_K=4 k4vf:LOSS_SSIM_K=4,LOSS_SSIM_VF=1" CHECK=k4vf.
# JITK: kernels opt/ab_jit_check.py checks on the first two arms. DISASM=1: objdump of each arm's ssim_fwd/bwd math (trisc1) ELF, SFPU opcode counts.
# Same steps as docs/t736/drive.sh: ssh preflight @p100, opt/sync_remote.sh (one-way, repo-scoped), ssh+tar
# (bh-49 has no rsync). Existing measurement reservation only (no ird reserve/extend/release), never bh-30,
# StrictHostKeyChecking=yes.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync} MESH=${MESH:-P150} ITERS=${ITERS:-50} ROUNDS=${ROUNDS:-4} RUN_TO=${RUN_TO:-120}
ARMS=${ARMS:-base:LOSS_SSIM_ABL=0} CHECK=${CHECK:-} DISASM=${DISASM:-0}
D=docs/t753; O=$D/out-$H
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" @p100 || { p=$?; echo "=== ssh preflight @p100 failed rc=$p"; exit $p; }
SO=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
SSH=(ssh "${SO[@]}" "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t753 $rev=$sha to $H:$T mesh=$MESH arms='$ARMS' check='$CHECK' $(TZ=America/Los_Angeles date)"
if [ -n "$CHECK" ]; then rm -rf tmp/t753/mac && python3 opt/t736/ssim_check.py prep tmp/t753/mac bike_top bike_mid hero || exit 6; fi
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
if [ $rc = 3 ]; then echo "sync rc=3 (stale-build heuristic), resync"; opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?; fi
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
[ "$rsha" = "$sha" ] || { echo "STAMP MISMATCH remote=$rsha local=$sha"; exit 4; }
C=/localdev/smarton/.cache/ttmc-t753
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=$MESH PYTHONDONTWRITEBYTECODE=1 && B=render/bench/build/loss_bench"
CASES="syn"; [ -n "$CHECK" ] && CASES="dump64 syncrop syn"
timeout 900 "${SSH[@]}" "$ENV && CXX=\$(grep '^CMAKE_CXX_COMPILER:' render/build-tt/CMakeCache.txt | cut -d= -f2) && \
  { cmake -G Ninja -S render/bench -B render/bench/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=\$CXX > tmp/t753cfg.log 2>&1 && \
    nice -n 19 cmake --build render/bench/build --target loss_bench -j 8 > tmp/t753build.log 2>&1 || { tail -n 40 tmp/t753cfg.log tmp/t753build.log; exit 5; }; } && \
  rm -rf tmp/t753 $C-* && source .venv/bin/activate && python3 opt/t736/ssim_check.py prep tmp/t753 $CASES"; rc=$?
echo "build+prep rc=$rc"; [ $rc = 0 ] || exit 5
armenv() { local a; for a in $ARMS; do [ "${a%%:*}" = "$1" ] && echo "${a#*:}" | tr ',' ' '; done; }
if [ -n "$CHECK" ]; then
  TARC=(env COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs)
  "${TARC[@]}" -C tmp/t753/mac -cf - . | timeout 300 "${SSH[@]}" "mkdir -p $T/tmp/t753 && tar -C $T/tmp/t753 -xf - 2>/dev/null"; echo "upload rc=$?"
  timeout 600 "${SSH[@]}" "$ENV && for c in dump64 syncrop bike_top bike_mid hero syn; do o=tmp/t753/\$c/out; mkdir -p \$o; \
      env TT_METAL_CACHE=$C-$CHECK $(armenv "$CHECK") LOSS_MODE=ssim timeout $RUN_TO \$B tmp/t753/\$c \$o 5 > \$o/run.log 2>&1; \
      echo \"check-run \$c rc=\$?\"; grep -E '^\[LOSS\] mode|TT_THROW|TT_FATAL|error' \$o/run.log | head -20; done"
  echo "check runs rc=$?"
fi
for r in $(seq 1 "$ROUNDS"); do
  for a in $ARMS; do n=${a%%:*}
    timeout 600 "${SSH[@]}" "$ENV && o=tmp/t753/syn/r$r-$n; mkdir -p \$o; env TT_METAL_CACHE=$C-$n $(armenv "$n") LOSS_MODE=ssim \
      timeout $RUN_TO \$B tmp/t753/syn \$o $ITERS > \$o/run.log 2>&1; echo \"round $r $n rc=\$?\"; \
      grep -E '^\[LOSS\] mode|TT_THROW|TT_FATAL' \$o/run.log | head -3; echo \"grad md5 \$(md5sum < \$o/grad.f32 | cut -c1-32)\"; rm -f \$o/grad.f32"
  done
done
echo "timing done"
timeout 300 "${SSH[@]}" "for a in $(for a in $ARMS; do printf '%s ' "${a%%:*}"; done); do find $C-\$a -maxdepth 8 -type d -path '*/kernels/*' > $T/tmp/t753/jit-\$a.txt; done"
if [ "$DISASM" = 1 ]; then
  timeout 300 "${SSH[@]}" "OD=\$(ls /localdev/smarton/tt-metal/runtime/sfpi/compiler/bin/*objdump | head -1); echo objdump=\$OD; \
    for a in $(for a in $ARMS; do printf '%s ' "${a%%:*}"; done); do for k in ssim_fwd_compute ssim_bwd_compute; do \
      e=\$(find $C-\$a -path \"*/\$k/*\" -name trisc1.elf | head -1); [ -n \"\$e\" ] || continue; \
      \$OD -d \$e > $T/tmp/t753/dis-\$a-\$k.txt; echo \"== \$a \$k \$(wc -l < $T/tmp/t753/dis-\$a-\$k.txt) lines\"; \
      awk '{print \$3}' $T/tmp/t753/dis-\$a-\$k.txt | grep -i '^sfp' | sort | uniq -c | sort -rn | head -14; done; done"
fi
rm -rf "$O/remote"; mkdir -p "$O/remote"
timeout 300 "${SSH[@]}" "cd $T/tmp/t753 && tar --exclude='*.f32' -cf - ." | tar -C "$O/remote" -xf -; echo "pull rc=$?"
if [ -n "$CHECK" ]; then
  for c in dump64 syncrop bike_top bike_mid hero syn; do
    [ -d "$O/remote/$c/out" ] || continue
    timeout 120 "${SSH[@]}" "cat $T/tmp/t753/$c/out/grad.f32" > "$O/remote/$c/out/grad.f32" || rm -f "$O/remote/$c/out/grad.f32"
    for f in hw.txt image.f32 target.f32; do
      if [ -f "tmp/t753/mac/$c/$f" ]; then cp "tmp/t753/mac/$c/$f" "$O/remote/$c/"
      else timeout 120 "${SSH[@]}" "cat $T/tmp/t753/$c/$f" > "$O/remote/$c/$f" || rm -f "$O/remote/$c/$f"; fi
    done
  done
fi
timeout 120 "${SSH[@]}" "rm -rf $T/tmp/t753 $C-*"; echo "remote cleanup rc=$?"
set -- $ARMS
if [ $# -ge 2 ]; then
  python3 opt/ab_jit_check.py --arm "${1%%:*}=$O/remote/jit-${1%%:*}.txt" --arm "${2%%:*}=$O/remote/jit-${2%%:*}.txt" \
    $(for k in ${JITK:-ssim_fwd_compute ssim_bwd_compute}; do printf -- '--kernel %s ' "$k"; done); echo "ab_jit_check rc=$? (kernels: ${JITK:-ssim_fwd_compute ssim_bwd_compute})"
fi
for a in $ARMS; do n=${a%%:*}; echo "-- $n"; grep -h "^\[LOSS\] mode" "$O"/remote/syn/r*-$n/run.log; done
if [ -n "$CHECK" ]; then python3 opt/t736/ssim_check.py check "$O/remote"; echo "check rc=$?"; fi
rm -f "$O"/remote/*/out/grad.f32 "$O"/remote/*/*.f32
echo "=== done $(TZ=America/Los_Angeles date)"
