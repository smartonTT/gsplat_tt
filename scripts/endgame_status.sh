#!/usr/bin/env bash
# Quick ledger for autonomous endgame execution (see opt/endgame-execution-plan.md).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "=== Endgame status $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "branch: $(git branch --show-current) @ $(git rev-parse --short HEAD)"
echo "HEAD: $(git log -1 --oneline)"
echo ""
echo "Phase A (host/driver):"
echo "  A1 JIT warmup: env default ON ideal path; verify_cmd has GSPLAT_TT_JIT_WARMUP=1"
echo "  A2 sort→blend in-sort: SORT_BLEND_PIPE (61f61ad)"
echo "  A3 device LPT layout: gated OFF (~900ms); verify IDENTICAL when enabled"
echo "  A4 Tracy zones: committed 0c72479"
echo ""
echo "Phase B: in-flight worker (chunk fusion)"
echo "Phase C: in-flight worker (DEST plan)"
echo ""
if [[ -f opt/ttw/iters.jsonl ]]; then
  echo "last ttw iter:"
  tail -1 opt/ttw/iters.jsonl | python3 -c "import json,sys; r=json.load(sys.stdin); print(' ',r.get('iter'),r.get('decision'),r.get('reason','')[:80])" 2>/dev/null || tail -1 opt/ttw/iters.jsonl
fi
test -f STOP && echo "STOP file present" || echo "no STOP"
pgrep -fl loopd.sh 2>/dev/null | head -2 || echo "loopd not running"
