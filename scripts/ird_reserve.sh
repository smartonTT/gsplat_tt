#!/usr/bin/env bash
# Reserve (or extend) an IRD box when health_check.sh returns RESERVATION_LAPSED.
# Source: tt-workflows/README.md (Reserve + sync section).
#
# Usage:
#   scripts/ird_reserve.sh                       # default: tt_yyz, blackhole, p300
#   scripts/ird_reserve.sh tt_aus p150           # Austin P150
#   scripts/ird_reserve.sh tt_yyz wormhole_b0    # WH (no --model needed)
set -uo pipefail

CLUSTER="${1:-tt_yyz}"
MODEL_OR_ARCH="${2:-p300}"
TIMEOUT="${IRD_TIMEOUT:-14:00:00}"

# Blackhole takes --model p300/p150; wormhole_b0 takes no model.
case "$MODEL_OR_ARCH" in
  p100|p150|p300) IRD_CMD="ird reserve --cluster $CLUSTER --timeout $TIMEOUT blackhole --model $MODEL_OR_ARCH" ;;
  wormhole_b0|wh) IRD_CMD="ird reserve --cluster $CLUSTER --timeout $TIMEOUT wormhole_b0" ;;
  *) echo "unknown model/arch: $MODEL_OR_ARCH (expected p100|p150|p300|wormhole_b0)" >&2; exit 2 ;;
esac

echo "[ird_reserve] $IRD_CMD" >&2
ssh yyz-ird "$IRD_CMD" 2>&1 | tail -20
RC="${PIPESTATUS[0]}"

if [[ "$RC" != "0" ]]; then
  echo "[ird_reserve] reservation failed (rc=$RC)" >&2
  exit "$RC"
fi

# devsync daemon ticks every ~5min and auto-picks up new reservations.
# Force a refresh now so the new ssh-hosts entry lands faster.
DEVSYNC="${DEVSYNC:-/Users/smarton/dev/tt-workflows/devsync}"
if [[ -x "$DEVSYNC" ]]; then
  echo "[ird_reserve] running devsync to refresh ssh-hosts + seed" >&2
  "$DEVSYNC" 2>&1 | tail -5
fi

echo "[ird_reserve] done — wait for devsync is-finished <host> before running iters" >&2
