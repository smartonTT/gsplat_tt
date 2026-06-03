#!/usr/bin/env bash
# Regenerate opt/REPORT.html and opt/ttw/REPORT.html (identical). Safe to call often.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$(command -v python3 || command -v python || true)"
[[ -n "$PY" ]] || { echo "report_refresh: python required" >&2; exit 1; }
exec "$PY" "$ROOT/opt/build_report.py" "$@"
