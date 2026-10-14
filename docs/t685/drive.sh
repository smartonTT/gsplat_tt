#!/bin/bash
# t693 (T1): device blend-backward kernel vs the T0 gradient oracle, all in ONE `ttp lock p100`.
#   ttp detach t693 -- docs/t685/drive.sh <rev>      (Mac, worktree root; <rev> must be committed)
# Steps: ssh preflight, opt/sync_remote.sh --no-build (one-way, repo-scoped, this worktree),
# build render/train, for each oracle npz: `blend_backward_dev prep` on the box, run
# blend_bwd_bench, pull slots.f32 + timing.json back, then `blend_backward_dev check` here.
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never
# the viewer box bh-30, StrictHostKeyChecking=yes on every ssh. H defaults to bh-49 (board 0,
# p150a, the measurement box since #687; label 'p150'); H=bh-33 gives 'p100 (provisional)'.
# SHOT=1: after the runs, the device hero screenshot (opt/ttw/screenshot.sh, NO_SYNC=1, same lock)
# into opt/metal-screenshots/$SHOT_NAME (diff vs benchmarks/reference_v2/hero.png, PSNR).
# #707: NPZS entry `hero` = tmp/t707/hero.npz (docs/t685/hero.py, built on the Mac), pushed into
# the box's tmp/t707in by rsync from this worktree inside the same lock and checked here with
# `check --fast`. Prep dirs live in tmp/t707in/<n> (removed at the end); the bench writes its
# outputs (slots.f32, timing.json, stats.json, runs_ms.json) to tmp/t693/<n>, which is pulled.
# BWD_STATS=0 builds the kernels without the stage counters.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-49} T=${T:-/localdev/smarton/gstt2-sync}
D=docs/t685; O=$D/out
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t693 $rev=$sha to $H:$T $(TZ=America/Los_Angeles date)"
opt/sync_remote.sh --no-build "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
ENV="cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=/localdev/smarton/.cache/ttmc-t693 \
  TT_METAL_OPERATION_TIMEOUT_SECONDS=${OP_TIMEOUT:-60} BWD_STATS=${BWD_STATS:-1} && source .venv/bin/activate"
# WATCH=1: tt-metal watcher (waypoints per RISC) with its log pulled back under out/remote/logs.
[ "${WATCH:-0}" = 1 ] && ENV="$ENV && export TT_METAL_WATCHER=2 TT_METAL_LOGS_PATH=$T/tmp/t693/logs"
if [ "${RESET_FIRST:-0}" = 1 ]; then echo "=== tt-smi -r first"; timeout 300 "${SSH[@]}" "$ENV && tt-smi -r > /dev/null 2>&1; echo reset rc=\$?"; fi
echo "=== build $(date +%T)"
timeout 900 "${SSH[@]}" "$ENV && mkdir -p tmp/t693 && (cmake -G Ninja -S render/train -B render/train/build -DCMAKE_BUILD_TYPE=Release \
  && nice -n 19 cmake --build render/train/build -j 8) > tmp/t693/build.log 2>&1; rc=\$?; tail -25 tmp/t693/build.log; exit \$rc"
brc=$?; echo "build rc=$brc"
case " ${NPZS:-} " in *" hero "*)
  echo "=== push tmp/t707/hero.npz $(date +%T)"
  timeout 60 "${SSH[@]}" "mkdir -p $T/tmp/t707in" && \
    timeout 900 "${SSH[@]}" "cat > $T/tmp/t707in/hero.npz.tmp && mv $T/tmp/t707in/hero.npz.tmp $T/tmp/t707in/hero.npz" \
      < tmp/t707/hero.npz; echo "push rc=$?" ;;  # plain ssh: the p150 hosts have no rsync
esac
if [ $brc = 0 ]; then
  for n in ${NPZS:-sh0 sh3}; do
    echo "=== run $n $(date +%T)"
    # sh0/sh3: the T0 oracle dumps; name=H:W:N:seed: docs/t685/synth.py scaling dump made on the box
    # (pulled back with the outputs and checked against the reference on the same inputs).
    case "$n" in
      *=*) nm=${n%%=*}; a=(${n#*=}); a=(${a//:/ }); src=tmp/t693/$nm.npz
           mk="PYTHONPATH=. python3 docs/t685/synth.py $src ${a[*]} &&" ;;
      hero) nm=hero; src=tmp/t707in/hero.npz; mk= ;;
      *) nm=$n; src=benchmarks/train_oracle/t0_${n}_64x64_n50.npz; mk= ;;
    esac; n=$nm
    timeout 900 "${SSH[@]}" "$ENV && rm -rf tmp/t693/$n tmp/t707in/$n && mkdir -p tmp/t693/$n tmp/t707in && $mk python3 -m gsplat.blend_backward_dev prep $src tmp/t707in/$n \
      && timeout ${RUN_TIMEOUT:-240} render/train/build/blend_bwd_bench tmp/t707in/$n \${BWD_CORES:-0} ${REPS:-6} tmp/t693/$n > tmp/t693/$n/run.log 2>&1; rc=\$?; tail -20 tmp/t693/$n/run.log; \
      if [ \$rc != 0 ]; then echo \"run rc=\$rc: tt-smi -r\"; tt-smi -r > tmp/t693/$n/reset.log 2>&1; echo \"reset rc=\$?\"; fi; exit \$rc"
    echo "run $n rc=$?"
  done
fi
timeout 120 "${SSH[@]}" "rm -rf $T/tmp/t707in"; echo "rm t707in rc=$?"
if [ "${SHOT:-0}" = 1 ]; then
  echo "=== shot $(TZ=America/Los_Angeles date +%T)"
  mkdir -p tmp/t707-bin tmp/ttw-state
  for c in ssh scp; do
    printf '#!/bin/sh\nexec /usr/bin/%s -o StrictHostKeyChecking=yes -o ControlMaster=no -o ControlPath=none "$@"\n' $c > tmp/t707-bin/$c
    chmod +x tmp/t707-bin/$c
  done
  sed -e "s|^device_host = .*|device_host = \"$H\"|" -e "s|^remote_root = .*|remote_root = \"$T\"|" \
    "$HOME/dev/gstt2/ttw.toml" > tmp/ttw-state/ttw.toml
  [ -f tmp/ttw-state/buildid.cpp ] || cp "$HOME/dev/gstt2/.ttw/buildid.cpp" tmp/ttw-state/
  PATH=$PWD/tmp/t707-bin:$PATH TTW_CONFIG=$PWD/tmp/ttw-state/ttw.toml TTW_STATE=$PWD/tmp/ttw-state \
    H=$H T=$T NO_SYNC=1 NAME=${SHOT_NAME:-t707-hero} SHOT_TIMEOUT=500 opt/ttw/screenshot.sh 707 "$rev" --locked > "$O/shot.out" 2>&1
  echo "shot rc=$?"; grep -E "^(===|run rc|VIEWS|ALL_|SHOT|SHOT_FAIL)" "$O/shot.out"
fi
opt/sync_remote.sh --pull tmp/t693 "$O/remote" "$H" "$T"; echo "pull rc=$?"
for n in ${NPZS:-sh0 sh3}; do
  fast=
  case "$n" in *=*) n=${n%%=*}; src=$O/remote/$n.npz ;; hero) src=tmp/t707/hero.npz; fast=--fast ;;
    *) src=benchmarks/train_oracle/t0_${n}_64x64_n50.npz ;; esac
  f=$O/remote/$n/slots.f32
  # HERO_CHECK=0: skip the ~25 min hero check here (it holds the lock); run it after the lock.
  if [ "$n" = hero ] && [ "${HERO_CHECK:-1}" = 0 ]; then echo "CHECK hero: deferred"; continue; fi
  if [ -s "$f" ]; then python3 -m gsplat.blend_backward_dev check "$src" "$f" --bar 1e-2 $fast | tee "$O/check-$n.json"
  else echo "CHECK $n: no slots.f32"; fi
done
echo "=== done $(TZ=America/Los_Angeles date)"
