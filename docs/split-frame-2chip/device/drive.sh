#!/bin/bash
# t627: Mac-side driver of the split-frame 2-chip measurement on the p150-2board host.
#   ttp detach t627drive -- docs/split-frame-2chip/device/drive.sh      (worktree root)
# The tree is synced and built before (opt/sync_remote.sh bh-50 $T HEAD). Under one
# `ttp lock p150-2board`: ssh preflight, copy remote_drive.sh into place (tmp name + mv), start it
# with `ttp detach --remote` (survives an ssh drop), wait for it, fetch $T/tmp/t627, then
# hero diff x10 + PSNR vs benchmarks/reference_v2/hero.png. Does not reserve or release: the
# task does that itself (IRD job $JOB). Every ssh/scp is strict on host keys.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
H=${H:-bh-50}; R=${R:-/localdev/smarton/nosync/t582}; T=${T:-$R/gstt2}
D=docs/split-frame-2chip/device; O=${O:-$D/out}; mkdir -p "$O" tmp/t627-bin
case "$H" in bh-30|*bh-30.*|bh-33) echo "=== refusing $H"; exit 2 ;; esac
PRE=${SSH_PREFLIGHT:-${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight}
for c in ssh scp; do
  printf '#!/bin/sh\nexec /usr/bin/%s -o ControlMaster=no -o ControlPath=none -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=20 "$@"\n' $c > tmp/t627-bin/$c
  chmod +x tmp/t627-bin/$c
done
export PATH=$PWD/tmp/t627-bin:$PATH
ts() { TZ=America/Los_Angeles date '+%F %T %Z'; }

if [ "${1:-}" = --locked ]; then
  "$PRE" "$H" || { p=$?; echo "=== ssh preflight $H failed rc=$p"; exit $p; }
  ssh "$H" "mkdir -p $T/tmp && cat > $T/tmp/t627_remote_drive.sh.tmp && mv $T/tmp/t627_remote_drive.sh.tmp $T/tmp/t627_remote_drive.sh" < "$D/remote_drive.sh" || exit 3
  ssh "$H" "rm -rf $R/t627-detach" || exit 3
  echo "=== $(ts) start remote_drive ONLY=${ONLY:-all}"
  ttp detach --remote "$H" --dir "$R/t627-detach" t627 -- env ONLY="${ONLY:-all}" bash "$T/tmp/t627_remote_drive.sh" || exit 3
  end=$(( $(date +%s) + 7200 ))
  while :; do
    ttp detach --check --host "$H" "$R/t627-detach/t627" >/dev/null 2>&1; c=$?
    [ $c = 0 ] && break
    [ "$(date +%s)" -lt "$end" ] || { echo "=== remote_drive still running after 120 min"; break; }
    sleep 30
  done
  mkdir -p "$O/remote"
  scp -q "$H:$T/tmp/t627/*" "$O/remote/" || echo "=== fetch failed"
  scp -q "$H:$R/t627-detach/t627*" "$O/remote/" 2>/dev/null
  rr=$(cat "$O/remote/DONE" 2>/dev/null || echo 99)
  echo "=== $(ts) remote_drive first_rc=$rr"
  exit "$rr"
fi

echo "=== $(ts) t627 drive $(git rev-parse --short HEAD) on $H"
ttp lock p150-2board -- "$PWD/$D/drive.sh" --locked; rc=$?
echo "=== $(ts) measurement rc=$rc"
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
heroes = {}
for n in ("hero-1chip", "hero-1chip-b2b", "hero-chip1", "hero-split"):
    p = d / f"{n}.png"
    if not p.exists():
        print(f"PSNR {n} missing"); continue
    h = heroes[n] = rgb(p)
    Image.fromarray(np.clip(np.abs(h - ref) * 10, 0, 255).astype(np.uint8)).save(d / f"{n}_diff10.png")
    print(f"PSNR {n} psnr_vs_ref={psnr(h, ref)} ref=benchmarks/reference_v2/hero.png size={h.shape[1]}x{h.shape[0]}")
a = heroes.get("hero-1chip-b2b", heroes.get("hero-1chip"))
for n in ("hero-split", "hero-chip1"):
    if a is not None and n in heroes:
        diff = np.abs(heroes[n] - a).max(axis=2)
        cols = np.nonzero(diff.max(axis=0))[0]
        print(f"VS1CHIP {n} px_diff={int((diff > 0).sum())} max={int(diff.max())} "
              f"cols={cols.min() if cols.size else '-'}..{cols.max() if cols.size else '-'}")
        Image.fromarray(np.clip(diff * 20, 0, 255).astype(np.uint8)).save(d / f"{n}_vs1chip_x20.png")
PY
echo "=== $(ts) drive done rc=$rc"
exit $rc
