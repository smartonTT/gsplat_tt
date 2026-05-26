#!/usr/bin/env bash
# ONLY path that touches bh-30. Promotes a KEEP commit to bh-30's stable_viewer.
# Per feedback-bh30-stable-only: do NOT use bh-30 for anything else.
#
# Usage: scripts/promote_to_stable.sh <commit-sha>
set -euo pipefail

SHA="${1:?usage: $0 <commit-sha>}"
BOX_USER="${BOX_USER:-smarton}"
BH14="yyzo-bh-14"
BH30="bh-30"  # devsync-managed ssh alias (see ~/.devsync/ssh-hosts)
REMOTE_BIN_BH14="/proj_sw/user_dev/smarton/gsplat_tt/backends/tt/tt-metal/build/programming_examples/metal_example_gaussian_splatting"
STABLE_DIR_BH30="/proj_sw/user_dev/smarton/gsplat_tt_stable"

# Verify the commit exists locally and matches HEAD or recent log
git rev-parse "$SHA" >/dev/null || { echo "unknown sha $SHA" >&2; exit 1; }

# Build is assumed already done on bh-14 at this sha; verify:
ssh "$BOX_USER@$BH14" "test -f $REMOTE_BIN_BH14" || { echo "no binary on $BH14" >&2; exit 2; }

# Copy from bh-14 to bh-30 via Mac (avoid box-to-box ssh perms)
TMP="/tmp/metal_example_gaussian_splatting_$SHA"
scp "$BOX_USER@$BH14:$REMOTE_BIN_BH14" "$TMP"
scp "$TMP" "$BOX_USER@$BH30:$STABLE_DIR_BH30/metal_example_gaussian_splatting_$SHA"

# SIGTERM → 10s → restart on bh-30
ssh "$BOX_USER@$BH30" "
  pkill -TERM -f metal_example_gaussian_splatting || true
  sleep 10
  ln -sf $STABLE_DIR_BH30/metal_example_gaussian_splatting_$SHA $STABLE_DIR_BH30/metal_example_gaussian_splatting
  bash $STABLE_DIR_BH30/start_stable_viewer.sh
"

echo "promoted $SHA to $BH30"
