#!/bin/bash
# t493 driver (Mac): build tree493 at <rev> on bh-30 with the viewer up (nice -n 19, ionice -c3, -j nproc/2),
# then stop the viewer only for bench493.sh, which restarts it at exit (vstart493.sh = do_start of
# opt/viewer/viewer.sh); the Mac restarts it as a fallback. Existing viewer reservation only: no IRD
# reserve/extend/release. Every ssh keeps StrictHostKeyChecking=yes, off the ~/.ssh/config mux.
#   ttp detach t493 -- ttp lock p100 -- docs/chunk-cull-t493/drive493.sh <rev>   (task holds 'viewer')
set -eo pipefail
cd "$(git rev-parse --show-toplevel)"
D=docs/chunk-cull-t493; REV=$(git rev-parse "${1:?rev}")
H=bh-30; P=/localdev/smarton/p150bench; VDIR=/localdev/smarton/viewer; T=$P/tree493; O=$P/out493
TTMH=$VDIR/tt-metal
MARK=${TTP_RUN_DIR:-tmp}/drive493.done; rm -f "$MARK"
trap 'rc=$?; echo "drive493 exit $rc"; echo $rc > "$MARK"' EXIT
SSHO=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
rsh() { ssh "${SSHO[@]}" $H "$@"; }
put() { scp -q "${SSHO[@]}" "$1" "$H:$2.tmp493" && rsh "mv $2.tmp493 $2"; }
VSH=${TMPDIR:-/tmp}/viewer-t493.sh; git show origin/smarton/tt-project-opt:opt/viewer/viewer.sh > $VSH
"$TTP_PROJECT/harness/bin/ssh-preflight" $H
echo "=== sync tree493 ${REV:0:8} $(date -u +%FT%TZ)"
rsh "test -d $T || { mkdir -p $T && cd $P/tree392 && tar --exclude=./tmp --exclude=./render/build-tt \
  --exclude=./render/bench/build --exclude=./generated -cf - . | tar -xf - -C $T && mkdir -p $T/tmp && echo copied; }"
base=$(rsh "cat $T/SHA")
git diff --name-only --diff-filter=d "$base" "$REV" | grep -vE '^(opt/metal-screenshots|opt/profiler/ttw-)' > tmp/t493-files.txt
git archive --format=tar "$REV" $(cat tmp/t493-files.txt) | rsh "cd $T && tar -xmf - && echo $REV > SHA"
rsh "cd $T && export TT_METAL_HOME=$TTMH TT_METAL_ARCH_NAME=blackhole && J=\$(( \$(nproc) / 2 )) &&
  { test -f render/build-tt/build.ninja || nice -n 19 ionice -c3 cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release > tmp/cfg493.log 2>&1 || { tail -30 tmp/cfg493.log; exit 1; }; } &&
  { nice -n 19 ionice -c3 cmake --build render/build-tt -j \$J > tmp/build493.log 2>&1 || { tail -30 tmp/build493.log; exit 1; }; } &&
  tail -1 tmp/build493.log && ls -la render/render_clean*.so | cut -c25- && cat SHA"
echo "=== build ok $(date -u +%FT%TZ)"
# Remote failsafe start: the do_start body of viewer.sh with its variables bound.
{ echo "#!/bin/bash"; echo "VDIR=$VDIR; DIR=$VDIR/tree; PORT=8080; TTMH=$TTMH; ETHOV=/localdev/smarton/viewer-eth-overlay; DISPATCH="
  echo "GSPLAT_VIEWER_STALL_FILE=; GSPLAT_VIEWER_PYSPY="
  awk '/^do_start\(\)/{f=1} f&&/<<.R.$/{g=1;next} g&&/^R$/{exit} g' $VSH; } > tmp/vstart493.sh
grep -q "supervise.sh" tmp/vstart493.sh || { echo "vstart493 extraction failed"; exit 4; }
put $D/bench493.sh $P/bench493.sh; put tmp/vstart493.sh $P/vstart493.sh
rsh "rm -rf $O"
restart_fallback() {
  echo "=== viewer start (mac fallback) $(date -u +%FT%TZ)"; rsh "rm -f $VDIR/viewer.stop"
  for i in 1 2 3; do VIEWER_HOST=$H bash $VSH start && break; sleep 10; done
}
echo "=== viewer stop $(date -u +%FT%TZ)"
VIEWER_HOST=$H bash $VSH stop
trap 'rc=$?; restart_fallback; echo $rc > "$MARK"' EXIT
rsh "VSTART=$P/vstart493.sh setsid nohup bash $P/bench493.sh $O $T ${ROUNDS:-4} > $P/bench493.log 2>&1 < /dev/null &"
for _ in $(seq 360); do rsh "test -e $O/vstarted" && break; sleep 5; done
rsh "test -e $O/vstarted" || restart_fallback
trap 'rc=$?; echo $rc > "$MARK"' EXIT
for _ in $(seq 60); do rsh "grep -q READY $VDIR/viewer.log" && break; sleep 3; done
rsh "grep -E 'SELFTEST|READY' $VDIR/viewer.log | tail -2; echo SHA=\$(cat $VDIR/tree/SHA)"
echo "=== viewer ready $(date -u +%FT%TZ) localhost:8091 -> $(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)"
mkdir -p $D/out; scp -q "${SSHO[@]}" "$H:$O/*" $D/out/; scp -q "${SSHO[@]}" "$H:$P/bench493.log" $D/out/
cat $D/out/bench.rc; exit "$(cat $D/out/bench.rc)"
