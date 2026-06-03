#!/usr/bin/env bash
# Push gaussian_splatting kernel sources to remote TT box and rebuild host binary.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
REMOTE="${REMOTE_HOST}"
# Prefer gsplat_tt canonical tree (gstt2 symlink may be broken on Mac).
if [[ -d "$REPO_ROOT/../gsplat_tt/backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting" ]]; then
  GS="$REPO_ROOT/../gsplat_tt/backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting"
else
  GS="$REPO_ROOT/backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting"
fi
GS_REMOTE="/proj_sw/user_dev/smarton/gsplat_tt/backends/tt/tt-metal/tt_metal/programming_examples/gaussian_splatting"

rsync -av "$GS/" "$REMOTE:$GS_REMOTE/"
ssh "$REMOTE" "sudo ninja -C /proj_sw/user_dev/smarton/gsplat_tt/backends/tt/tt-metal/build metal_example_gaussian_splatting"
echo "sync ok: $REMOTE"
