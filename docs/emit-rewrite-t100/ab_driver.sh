#!/bin/bash
# t100 interleaved A/B: 3 rounds, step order rotated per round. Run from the worktree root.
ST=(base pb:GSPLAT_TT_EMIT_PB=8 ring:GSPLAT_TT_EMIT_RING=8 puboc:GSPLAT_TT_EMIT_PUBOC=1
    pbring:GSPLAT_TT_EMIT_PB=8,GSPLAT_TT_EMIT_RING=8
    all:GSPLAT_TT_EMIT_PB=8,GSPLAT_TT_EMIT_RING=8,GSPLAT_TT_EMIT_PUBOC=1
    all4:GSPLAT_TT_EMIT_PB=4,GSPLAT_TT_EMIT_RING=4,GSPLAT_TT_EMIT_PUBOC=1)
n=${#ST[@]}
for r in 1 2 3; do
  steps=(); for ((i=0;i<n;i++)); do steps+=("${ST[$(( (i + (r-1)*3) % n ))]}"); done
  ttp lock p100 -- ssh -o BatchMode=yes yyzo-bh-07 \
    "bash /localdev/smarton/gstt2-t100/docs/emit-rewrite-t100/remote_job.sh $r ${steps[*]}" || echo "round $r rc=$?"
done
