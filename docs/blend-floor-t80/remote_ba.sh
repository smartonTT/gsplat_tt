#!/bin/bash
# One interleaved before/after round <r>: base tree, change tree, change tree with the old walk.
set -u
r=$1
run() {  # <label> <tree> [env...]
  local lab=$1 tree=$2; shift 2
  ( export T80_TREE=/localdev/smarton/$tree; source /localdev/smarton/t80_scripts/remote_env.sh
    echo "=== BA round=$r $lab $(date +%T)"
    env "$@" TT_METAL_CACHE_RENDER=$(cache 0 -$lab) timeout 150 python3 render/run.py --no-ref --iter-dir t80-ba-$r-$lab 2>&1 | grep -E "^(SUMMARY|STAGES)|Traceback|TT_THROW|TT_FATAL" )
}
run base gstt2-t80 X=1
run new gstt2-t80b X=1
run new-oldwalk gstt2-t80b GSPLAT_TT_BLEND_JUMP_WALK=0
echo "=== end $(date +%T)"
