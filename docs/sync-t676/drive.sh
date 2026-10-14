#!/bin/bash
# t676: one real sync+build+run on the measurement box through opt/sync_remote.sh (one-way,
# repo-scoped rsync from this worktree; no devsync/mutagen involved), all in ONE `ttp lock p100`.
#   ttp detach t676 -- docs/sync-t676/drive.sh <rev>     (Mac, worktree root)
# Steps: ssh preflight, rsync --dry-run (what would be sent), sync+build <rev> into $T, check the
# remote SYNC_STAMP sha, b2b over the 30 bicycle views, pull the run log back with --pull, a
# second sync (only the stamps should change), remote dir size.
# Restrictions: existing measurement reservation only (no ird reserve/extend/release), never
# the viewer box bh-30, StrictHostKeyChecking=yes on every ssh.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-33} T=${T:-/localdev/smarton/gstt2-sync}
D=docs/sync-t676; O=$D/out
if [ "${2:-}" != --locked ]; then exec ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked; fi
case "$H" in bh-30|*bh-30.*) echo "=== refusing the viewer box $H"; exit 2 ;; esac
mkdir -p "$O"
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
"$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
SSH=(ssh -o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
sha=$(git rev-parse "$rev^{commit}")
echo "=== t676 $rev=$sha from $PWD to $H:$T $(TZ=America/Los_Angeles date)"
echo "=== dry run"; opt/sync_remote.sh --dry-run "$H" "$T" "$rev" > "$O/dryrun.log" 2>&1; echo "dry rc=$?"
grep -c '^would send' "$O/dryrun.log"; tail -1 "$O/dryrun.log"
echo "=== sync+build $(date +%T)"; s0=$(date +%s)
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1; rc=$?
echo "sync rc=$rc in $(( $(date +%s)-s0 )) s"; grep '^\[sync_remote\]' "$O/sync.log"
[ $rc = 0 ] || { tail -30 "$O/sync.log"; exit 3; }
rsha=$(timeout 60 "${SSH[@]}" "sed -n 's/^sha=//p' $T/SYNC_STAMP")
echo "STAMP remote=$rsha local=$sha $([ "$rsha" = "$sha" ] && echo MATCH || echo MISMATCH)"
echo "=== b2b $(date +%T)"
timeout 420 "${SSH[@]}" "cd $T && export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal \
  TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-t676 \
  && unset TT_METAL_CACHE GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE GSPLAT_PER_VIEW_STAGES GSPLAT_TT_HOST_PROFILE && source .venv/bin/activate && mkdir -p tmp/t676 && for r in 0 1 2; do timeout 300 python3 render/run.py --no-ref --iter-dir t676-r\$r --back-to-back > tmp/t676/run-r\$r.log 2>&1; echo run r\$r rc=\$?; done"
echo "b2b rc=$?"
opt/sync_remote.sh --pull tmp/t676 "$O/remote" "$H" "$T"; echo "pull rc=$?"
grep -E '^B2B |TTW_TIMING b2b_ms_frame_median=|Traceback|TT_THROW|TT_FATAL' "$O"/remote/run-r*.log | cut -c1-260
echo "=== resync (no change expected) $(date +%T)"
opt/sync_remote.sh "$H" "$T" "$rev" > "$O/resync.log" 2>&1; echo "resync rc=$?"; grep '^\[sync_remote\]' "$O/resync.log"
echo "=== remote size"; timeout 120 "${SSH[@]}" "du -sxh $T $T/render/build-tt $T/build-avx2-tt /localdev/smarton/.cache/ttmc-t676 2>/dev/null; df -h /localdev/smarton /proj_sw | tail -2"
echo "=== done $(TZ=America/Los_Angeles date)"
