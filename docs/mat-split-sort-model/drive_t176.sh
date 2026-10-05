#!/bin/bash
# t176: one 30-view untraced bicycle run with GSPLAT_TT_MAT_DUMP (per-tile record counts
# per build_mat_worklist call) on yyzo-bh-07 p100a. Mac side; each device step under ttp lock p100.
#   drive_t176.sh <rev>
set -u
cd "$(git rev-parse --show-toplevel)"
DEVRUN=~/dev/tt-workflows/scripts/devrun.sh
H=yyzo-bh-07; T=/localdev/smarton/gstt2-t176; REV=${1:?rev}
REF=/localdev/smarton/t82_scripts/md5-r82new.txt
O=docs/mat-split-sort-model/out
lk() { ttp lock p100 -- "$@"; local rc=$?; [ $rc -eq 75 ] && { echo LOCK_BUSY; echo CHAIN_DONE; exit 75; }; return $rc; }
lk opt/sync_remote.sh $H $T "$REV"; rc=$?; echo "SYNC_RC=$rc"; [ $rc -eq 0 ] || { echo CHAIN_DONE; exit $rc; }
lk $DEVRUN --host $H --no-verify --timeout 560 --tag t176-dump -- "bash -c '
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100 TTW_DEVRUN=1
cd $T && source .venv/bin/activate && mkdir -p tmp/t176 && rm -f tmp/t176/dump.txt && rm -rf t176-dump
GSPLAT_TT_MAT_DUMP=$T/tmp/t176/dump.txt TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t176 timeout 400 \
  python3 render/run.py --no-ref --iter-dir t176-dump --dump-views t176-dump > tmp/t176/run.log 2>&1; echo run rc=\$?
grep -E \"^(STAGES|SUMMARY)|Traceback|TT_THROW|TT_FATAL\" tmp/t176/run.log | head -8
d=\$(find . -maxdepth 3 -type d -name t176-dump | head -1)
[ -n \"\$d\" ] && (cd \$d && md5sum * | sort -k2) > tmp/t176/md5.txt && rm -rf \$d
diff -q $REF tmp/t176/md5.txt >/dev/null && echo ALL_VIEWS_IDENTICAL || echo VIEWS_DIFFER
echo dump_lines=\$(wc -l < tmp/t176/dump.txt); gzip -kf tmp/t176/dump.txt'"
echo "RUN_RC=$?"
scp -q -o BatchMode=yes $H:$T/tmp/t176/dump.txt.gz $H:$T/tmp/t176/run.log $H:$T/tmp/t176/md5.txt $O/ ; echo "SCP_RC=$?"
echo CHAIN_DONE
