#!/bin/bash
# t687 chain (Mac, worktree root): wait for setup_box.sh on the new p150 box, finish its renew.sh
# measure run (sync + build of the base tree), then the re-baseline bench. Writes the first
# non-zero step code (or 0) to $MARK from an EXIT trap.
#   ttp detach t687-chain -- docs/p150-rebaseline-t687/chain.sh <rev>
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
rev=${1:?rev}; H=${H:-bh-49}
MARK=${MARK:-${TTP_RUN_DIR:-tmp}/t687-chain.done}
trap 'c=$?; echo "$c" > "$MARK"; echo "=== chain exit $c $(TZ=America/Los_Angeles date)"' EXIT
SSH=(timeout 60 ssh -o ControlMaster=no -o ControlPath=none -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=20 "$H")
echo "=== wait for setup.rc on $H $(TZ=America/Los_Angeles date)"
for i in $(seq 1 120); do   # up to 4 h; an ssh stall only costs one round
  src=$("${SSH[@]}" "cat /localdev/smarton/viewer/setup.rc 2>/dev/null" || true)
  [ -n "$src" ] && break
  sleep 120
done
echo "setup.rc=${src:-none}"
[ "$src" = 0 ] || { "${SSH[@]}" "tail -20 /localdev/smarton/viewer/setup.log" || true; exit 21; }
echo "=== renew measure (sync+build base tree) $(TZ=America/Los_Angeles date)"
set +e; opt/ird/renew.sh measure --keep-old; r=$?; set -e
echo "renew rc=$r"; [ $r = 0 ] || exit $((30 + r))
echo "=== bench $(TZ=America/Los_Angeles date)"
docs/p150-rebaseline-t687/drive.sh "$rev"
