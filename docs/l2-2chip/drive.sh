#!/bin/bash
# t582 (L2, 2-chip AFR on a 2-board p150 host): Mac-side driver.
#   ttp detach t582drive -- docs/l2-2chip/drive.sh <rev>    (worktree root; O=<out dir>)
# 1. waits (no lock) for the scene upload (local scene-up.rc) and the box setup
#    (/localdev/smarton/viewer/setup.rc on $H), then checks the scene md5 and moves it into place;
# 2. under one `ttp lock p100`: ssh preflight, extends IRD job $JOB if less than 80 min are left,
#    opt/sync_remote.sh <rev> into $T, starts docs/l2-2chip/remote_drive.sh on $H with
#    `ttp detach --remote` (it survives an ssh drop), waits for it and fetches $T/tmp/t582;
# 3. releases IRD job $JOB (also on failure) and checks that `ird list` no longer shows it;
# 4. hero diff x10 + PSNR vs benchmarks/reference_v2/hero.png for both chips' heroes.
# Restrictions: only IRD job $JOB (this task's own p150-2board reservation) is extended or
# released; never bh-30 or the measurement reservation; every ssh/scp is strict on host keys.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
rev=${1:?rev}
H=${H:-bh-50}; JOB=${JOB:-132945}; T=${T:-/localdev/smarton/gstt2}; R=/localdev/smarton
JOB_START=${JOB_START:-1791869358}   # 2026-10-12 22:29:18 PDT, from ird reserve's output
D=docs/l2-2chip; O=${O:-$D/out}; mkdir -p "$O" tmp/t582-bin
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o ControlMaster=no -o ControlPath=none -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=20 "$@"\n' $c > tmp/t582-bin/$c
  chmod +x tmp/t582-bin/$c
done
export PATH=$PWD/tmp/t582-bin:$PATH
case "$H" in bh-30|*bh-30.*|bh-33) echo "=== refusing $H"; exit 2 ;; esac
ts() { TZ=America/Los_Angeles date '+%F %T %Z'; }
jump() { timeout 120 ssh yyz-ird "$@"; }
sel_of() { jump "ird list" 2>/dev/null | awk -v j="$JOB" '$1 ~ /^[0-9]+$/ && $2 == "smarton" && $10 == j {print $1, $6}'; }

if [ "${2:-}" = --locked ]; then
  "$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
  read -r sel left < <(sel_of)
  [ -n "${sel:-}" ] || { echo "=== IRD job $JOB not in ird list"; exit 12; }
  lsec=$(awk -F'[-:]' '{s = 0; for (i = 1; i <= NF; i++) s = s * 60 + $i; print s}' <<<"$left")
  echo "=== $(ts) IRD job $JOB sel $sel left $left (${lsec}s)"
  if [ "$lsec" -lt 4800 ]; then
    tot=$(( $(date +%s) - JOB_START + 5400 ))
    nt=$(printf '%d:%02d:%02d' $((tot / 3600)) $((tot % 3600 / 60)) $((tot % 60)))
    echo "=== extending job $JOB to $nt total"; jump "ird change-timeout $sel $nt" 2>&1 | grep -vi password | tail -2
  fi
  opt/sync_remote.sh "$H" "$T" "$rev" > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 3; }
  tail -2 "$O/sync.log"
  ssh "$H" "cat > $T/tmp/remote_drive.sh.tmp && mv $T/tmp/remote_drive.sh.tmp $T/tmp/remote_drive.sh" < "$D/remote_drive.sh" || exit 3
  ssh "$H" "rm -rf $R/t582-detach" || exit 3
  echo "=== $(ts) start remote_drive"
  ttp detach --remote "$H" --dir "$R/t582-detach" t582 -- bash "$T/tmp/remote_drive.sh" || exit 3
  end=$(( $(date +%s) + 6600 ))
  while :; do
    ttp detach --check --host "$H" "$R/t582-detach/t582" >/dev/null 2>&1; c=$?
    [ $c = 0 ] && break
    [ "$(date +%s)" -lt "$end" ] || { echo "=== remote_drive still running after 110 min"; break; }
    sleep 30
  done
  mkdir -p "$O/remote"
  scp -q "$H:$T/tmp/t582/*" "$O/remote/" || echo "=== fetch failed"
  scp -q "$H:$R/t582-detach/t582*" "$O/remote/" 2>/dev/null
  rr=$(cat "$O/remote/DONE" 2>/dev/null || echo 99)
  echo "=== $(ts) remote_drive first_rc=$rr"
  exit "$rr"
fi

echo "=== $(ts) t582 drive $(git rev-parse --short "$rev") on $H (IRD job $JOB)"
# 1. wait for the scene upload and the box setup (no lock held)
end=$(( $(date +%s) + 9000 ))
while :; do
  up=$(cat "${TTP_RUN_DIR:-/nonexistent}/scene-up.rc" "${SCENE_RC:-/nonexistent}" 2>/dev/null | head -1)
  st=$(ssh "$H" "cat $R/viewer/setup.rc 2>/dev/null")
  [ -n "$up" ] && [ -n "$st" ] && break
  [ "$(date +%s)" -lt "$end" ] || { echo "=== setup/upload not done after 150 min (scene=$up setup=$st)"; break; }
  sleep 60
done
echo "=== $(ts) scene upload rc=${up:-none} setup rc=${st:-none}"
rc=0
if [ "${up:-x}" = 0 ] && [ "${st:-x}" = 0 ]; then
  m=$(ssh "$H" "cd $T/scenes && { test -s bicycle.ply || { md5sum bicycle.ply.part | cut -c1-8 | grep -qx 3745b7a6 && mv bicycle.ply.part bicycle.ply; }; } && md5sum bicycle.ply | cut -c1-8")
  echo "=== scene md5 ${m:-none}"
  if [ "$m" = 3745b7a6 ]; then
    ttp lock p100 -- "$PWD/$D/drive.sh" "$rev" --locked; rc=$?
  else rc=21; fi
else
  ssh "$H" "tail -15 $R/viewer/setup.log" > "$O/setup-tail.log" 2>&1; rc=20
fi
echo "=== $(ts) measurement rc=$rc"

# 3. release this task's reservation (also on failure)
read -r sel left < <(sel_of)
if [ -n "${sel:-}" ]; then
  echo "=== $(ts) releasing IRD job $JOB (sel $sel, $left left)"
  timeout 300 ssh yyz-ird "echo y | ird release $sel" 2>&1 | grep -vi password | tail -4 > "$O/release.txt"
  cat "$O/release.txt"
fi
if jump "ird list" 2>/dev/null | awk -v j="$JOB" '$10 == j {f = 1} END {exit !f}'; then
  echo "=== RELEASE_FAIL job $JOB still in ird list"; [ $rc = 0 ] && rc=30
else echo "=== RELEASED job $JOB gone from ird list at $(ts)" | tee -a "$O/release.txt"; fi

# 4. hero diff + PSNR (chip A and chip B heroes of the 2-chip run, and the 1-chip hero)
python3 - "$O/remote" <<'PY' | tee "$O/psnr.txt"
import sys
from pathlib import Path
import numpy as np
from PIL import Image
d = Path(sys.argv[1])
def rgb(p): return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)
def psnr(a, b):
    m = np.mean((a - b) ** 2)
    return float("inf") if m == 0 else round(float(10 * np.log10(255.0 ** 2 / m)), 2)
ref = rgb("benchmarks/reference_v2/hero.png")
for n in ("hero-afr-chipA", "hero-afr-chipB", "hero-1chip"):
    p = d / f"{n}.png"
    if not p.exists():
        print(f"PSNR {n} missing"); continue
    h = rgb(p)
    Image.fromarray(np.clip(np.abs(h - ref) * 10, 0, 255).astype(np.uint8)).save(d / f"{n}_diff10.png")
    print(f"PSNR {n} psnr_vs_ref={psnr(h, ref)} ref=benchmarks/reference_v2/hero.png size={h.shape[1]}x{h.shape[0]}")
PY
echo "=== $(ts) drive done rc=$rc"
exit $rc
