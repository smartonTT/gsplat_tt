#!/bin/bash
# t90: base (a5f2bd6 code, gstt2-t85) vs fix tree (gstt2-t86) with the fused
# mat+cull off / on, each also with GSPLAT_TT_SPLIT_BLEND=1 for the mat/cull/blend
# split. 30 bicycle views each, untraced, md5 vs md5-r82new.txt.
#   remote_job.sh <round> [steps]   steps: any of base nofuse nofuse_split fuse fuse_split
set -u
S=/localdev/smarton/t86_scripts
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
B=/localdev/smarton/gstt2-t85; F=/localdev/smarton/gstt2-t86
r=${1:-1}; shift
steps=${*:-base nofuse nofuse_split fuse fuse_split}
run() {  # tag tree [env...]; returns non-zero if the run failed
  local tag=$1 T=$2; shift 2
  echo "=== $tag $(cat $T/SHA | cut -c1-7) $*"
  local out; out=$(env "$@" T86_TREE=$T bash $S/remote_run.sh $tag $REF 2>&1)
  echo "$out" | grep -E "^(SUMMARY|STAGES|SORT_STAGES)|IDENTICAL|DIFFER|rc=|Traceback|TT_THROW|TT_FATAL|rror"
  echo "$out" | grep -q "run rc=0"
}
for s in $steps; do
  case $s in
    base)         run t90r$r-base $B || exit 2 ;;
    nofuse)       run t90r$r-nofuse $F GSPLAT_TT_FUSE_MATCULL=0 || exit 2 ;;
    nofuse_split) run t90r$r-nofuse-split $F GSPLAT_TT_FUSE_MATCULL=0 GSPLAT_TT_SPLIT_BLEND=1 || exit 2 ;;
    fuse)         run t90r$r-fuse $F GSPLAT_TT_FUSE_MATCULL=1 || { tail -40 $F/tmp/t86-run-t90r$r-fuse.log; exit 3; } ;;
    fuse_split)   run t90r$r-fuse-split $F GSPLAT_TT_FUSE_MATCULL=1 GSPLAT_TT_SPLIT_BLEND=1 || exit 3 ;;
  esac
done
