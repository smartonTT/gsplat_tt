#!/bin/bash
# t587 driver (Mac): profile the live viewer on bh-30 with the B4a fold off/on.
#   ttp detach t587-<n> -- docs/viewer-t587/drive.sh <out name> [arms (default "0 1 0 1")]
# Copies the instrumented python files into the live viewer tree (python only, no rebuild),
# then per arm: viewer.sh restart (VIEWER_B4A_FOLD=<arm>, VIEWER_TIMELINE on), a 23 s sweep
# from the Mac through localhost:8091 and a 23 s sweep on the box (ws://127.0.0.1:8080, no
# link). Then the viewer is stopped for pipeline_probe (fold 0 and 1), and restarted with
# FINAL_FOLD (default 0). SWEEP_HZS (default 240): box sweep rates, one timeline burst each;
# MAC_PY: python with msgspec for the Mac sweep; SKIP_PROBE=1 skips pipeline_probe.
# Holds resource 'viewer' only: no reserve/extend/release, never the
# p100 lock or the measurement box. Strict host-key checking.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
N=${1:?out name}; ARMS=${2:-0 1 0 1}
D=docs/viewer-t587; V=/localdev/smarton/viewer; B=$V/t587/$N
SSHO=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20)
SSH=(ssh "${SSHO[@]}" bh-30)
L=$D/out/$N; mkdir -p "$L"
ts() { TZ=America/Los_Angeles date +%T%Z; }
"${TTP_PROJECT:-$HOME/dev/gsplat_tt/tt-project}/harness/bin/ssh-preflight" bh-30 || { echo "preflight failed: viewer untouched"; exit 3; }
"${SSH[@]}" "mkdir -p $B" || exit 3
for f in gsplat/nerfview_viewer.py opt/viewer/pipeline_probe.py opt/viewer/timeline_stats.py opt/viewer/ws_sweep_t532.py; do
  scp -q "${SSHO[@]}" $f bh-30:$V/tree/$f.t587tmp && "${SSH[@]}" "mv $V/tree/$f.t587tmp $V/tree/$f" || exit 3
done
scp -q "${SSHO[@]}" $D/probe587.sh bh-30:$B/.probe587.sh.tmp && "${SSH[@]}" "mv $B/.probe587.sh.tmp $B/probe587.sh" || exit 3
wait_ready() {
  for _ in $(seq 80); do "${SSH[@]}" "grep -q READY $V/viewer.log" && return 0; sleep 3; done; return 1
}
i=0
for arm in $ARMS; do
  i=$((i + 1)); tag=fold$arm-$i
  echo "=== $tag restart $(ts)"
  VIEWER_HOST=bh-30 VIEWER_B4A_FOLD=$arm VIEWER_TIMELINE=$B/tl-$tag.jsonl bash opt/viewer/viewer.sh restart
  wait_ready || { echo "viewer not READY"; "${SSH[@]}" "tail -20 $V/viewer.log"; }
  "${SSH[@]}" "grep -E 'SELFTEST|READY' $V/viewer.log | tail -2; tr '\0' '\n' < /proc/\$(cat $V/viewer.pid)/environ | grep -E 'B4A_FOLD|TIMELINE'"
  sleep 5
  "${MAC_PY:-python3}" opt/viewer/ws_sweep_t532.py --url ws://127.0.0.1:8091 --subprotocol viser-v1.0.27 --no-decode --secs 23 \
    > "$L/mac-$tag.out" 2>&1
  grep -o '"decoded_fps": [0-9.]*\|"link_mbs": [0-9.]*\|"interval_ms_p50": [0-9.]*' "$L/mac-$tag.out" | tr '\n' ' '; echo " (mac $tag)"
  sleep 3
  for hz in ${SWEEP_HZS:-240}; do
    "${SSH[@]}" "cd $V/tree && $V/venv/bin/python opt/viewer/ws_sweep_t532.py --url ws://127.0.0.1:8080 --subprotocol viser-v1.0.27 --no-decode --secs 23 --hz $hz" > "$L/box-$tag-$hz.out" 2>&1
    grep -o '"decoded_fps": [0-9.]*\|"link_mbs": [0-9.]*\|"interval_ms_p50": [0-9.]*' "$L/box-$tag-$hz.out" | tr '\n' ' '; echo " (box $tag ${hz}Hz)"
    sleep 2
  done
  "${SSH[@]}" "cd $V/tree && python3 opt/viewer/timeline_stats.py $B/tl-$tag.jsonl" | tee "$L/tl-$tag.txt"
done
if [ -z "${SKIP_PROBE:-}" ]; then
  echo "=== viewer stop for pipeline_probe $(ts)"; VIEWER_HOST=bh-30 bash opt/viewer/viewer.sh stop
  "${SSH[@]}" "setsid nohup bash $B/probe587.sh $B/probe > $B/probe587.log 2>&1 < /dev/null &"
  for _ in $(seq 300); do "${SSH[@]}" "test -e $B/probe/probe.rc" && break; sleep 5; done
  "${SSH[@]}" "cat $B/probe587.log; grep -h -E '^PIPE' $B/probe/probe-fold*.log" | tee "$L/probe.txt"
fi
echo "=== final viewer start fold=${FINAL_FOLD:-0} $(ts)"
VIEWER_HOST=bh-30 VIEWER_B4A_FOLD=${FINAL_FOLD:-0} bash opt/viewer/viewer.sh restart
wait_ready; "${SSH[@]}" "grep -E 'SELFTEST|READY' $V/viewer.log | tail -2"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 10 http://localhost:8091/)
echo "=== viewer ready $(ts) localhost:8091 -> $code"
scp -q -r "${SSHO[@]}" "bh-30:$B/probe" "$L/" 2>/dev/null
echo "=== drive end $(ts)"
