#!/bin/bash
# t144 driver (Mac side): materialize LPT cost calibration. Each phase is short
# enough for one `ttp lock p100` hold; run them one at a time:
#   ttp lock p100 -- bash docs/mat-lpt-t144/drive.sh <rev> sync
#   ttp lock p100 -- bash docs/mat-lpt-t144/drive.sh <rev> cap      # MPROF capture + LPT dump
#   ttp lock p100 -- bash docs/mat-lpt-t144/drive.sh <rev> "r1 A B"  # one A/B round, arms A then B
# Arms: base, or name:ENV=V[,ENV2=V2] (t124 remote_job.sh, 30 views, md5 vs ref).
set -o pipefail
REV=${1:?rev}; shift; PH=${1:?phase}
H=yyzo-bh-07; D=${D:-/localdev/smarton/gstt2-t142}  # landed t142 tree: incremental build
OUT=${OUT:-docs/mat-lpt-t144/out}; mkdir -p "$OUT"
J="T124_TREE=$D bash $D/docs/sort-onelaunch-v2-t124/remote_job.sh"
r() { ssh -o BatchMode=yes $H "$@"; }
case $PH in
sync)
  opt/sync_remote.sh $H $D $REV || { echo SYNC_FAIL; exit 1; } ;;
cap)  # 10-view Tracy chunk with fine mat zones, per-call worklist dump
  [ "$(r cat $D/SHA)" = "$(git rev-parse $REV)" ] || { echo "remote tree not at $REV"; exit 1; }
  tag=t144-${CAPTAG:-cap}
  r "rm -f $D/tmp/$tag-dump.txt; mkdir -p $D/tmp; GSPLAT_TT_MAT_DUMP=$D/tmp/$tag-dump.txt MPROF=1 T124_TREE=$D \
     bash $D/docs/sort-onelaunch-v2-t124/remote_tracy.sh $tag" | tail -40 || exit 2
  r "cd $D && gzip -c opt/profiler/$tag/chunks/0-10/profile_log_device.csv > tmp/$tag.csv.gz && gzip -f tmp/$tag-dump.txt"
  mkdir -p tmp
  scp -q -o BatchMode=yes $H:$D/tmp/$tag.csv.gz $H:$D/tmp/$tag-dump.txt.gz tmp/ || exit 3
  for f in gaps zones deep; do
    scp -q -o BatchMode=yes $H:$D/opt/profiler/$tag/$f.txt "$OUT/$tag-$f.txt" || echo "no $tag/$f.txt"
  done
  ls -la tmp/$tag.csv.gz tmp/$tag-dump.txt.gz ;;
r*)  # "rN armA armB ..."
  [ "$(r cat $D/SHA)" = "$(git rev-parse $REV)" ] || { echo "remote tree not at $REV"; exit 1; }
  set -- $PH; k=${1#r}; shift
  r "$J $k $*" ; rc=$?; echo "round rc=$rc"; exit $rc ;;
*) echo "unknown phase $PH"; exit 9 ;;
esac
