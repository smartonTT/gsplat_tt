#!/usr/bin/env bash
# Returns one of: OK | STALLED | DEVICE_HUNG | BUILD_STUCK
# Stdout is the verdict; stderr is human-readable details.
set -uo pipefail

BOX_USER="${BOX_USER:-smarton}"
BOX_HOST="${BOX_HOST:-yyzo-bh-14}"

# Check 1: tt-smi
if ! ssh -o ConnectTimeout=5 "$BOX_USER@$BOX_HOST" 'tt-smi -s' >/dev/null 2>&1; then
  echo "tt-smi failed on $BOX_HOST" >&2
  echo "DEVICE_HUNG"
  exit 0
fi

# Check 1.5: tt-triage (if available) — catches hangs/wedges that tt-smi misses.
# Gracefully skipped if the tool isn't installed. Look for any non-empty `errors`
# array in the JSON output as a wedge signal.
TRIAGE_JSON=$(ssh "$BOX_USER@$BOX_HOST" 'command -v tt-triage >/dev/null && tt-triage --json 2>/dev/null || echo ""' 2>/dev/null || echo "")
if [[ -n "$TRIAGE_JSON" ]]; then
  ERR_COUNT=$(echo "$TRIAGE_JSON" | jq -r '(.errors // []) | length' 2>/dev/null || echo "0")
  if [[ "$ERR_COUNT" != "0" ]]; then
    echo "tt-triage flagged $ERR_COUNT issue(s):" >&2
    echo "$TRIAGE_JSON" | jq -r '.errors[]?' >&2 || true
    echo "DEVICE_HUNG"
    exit 0
  fi
fi

# Check 2: viewer on 8080 responsive (tunnel must be up from Mac)
HTTP=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://localhost:8080 || echo "000")
if [[ "$HTTP" != "200" && "$HTTP" != "000" ]]; then
  # 000 = no tunnel; that's a transient mac-side issue not a box issue
  echo "viewer http $HTTP" >&2
fi

# Check 3: watcher waypoint freshness
WP_AGE=$(ssh "$BOX_USER@$BOX_HOST" '
  if [[ -f /tmp/watcher_waypoints.log ]]; then
    LAST=$(stat -c %Y /tmp/watcher_waypoints.log 2>/dev/null || stat -f %m /tmp/watcher_waypoints.log)
    NOW=$(date +%s)
    echo $((NOW - LAST))
  else
    echo 0
  fi' 2>/dev/null || echo 999)
if (( WP_AGE > 30 )); then
  echo "watcher waypoints stale (${WP_AGE}s)" >&2
  echo "STALLED"
  exit 0
fi

# Check 4: any active gsplat process advancing
ACTIVE=$(ssh "$BOX_USER@$BOX_HOST" 'pgrep -af metal_example_gaussian_splatting | head -1' 2>/dev/null || echo "")
if [[ -n "$ACTIVE" ]]; then
  # process running; consider it OK (run_iter.sh tracks its own progress)
  :
fi

echo "OK"
