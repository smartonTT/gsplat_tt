#!/usr/bin/env bash
# Dispatch helper for the validator subagent (Sonnet 4.6, fresh context).
#
# Usage:
#   scripts/dispatch_validator.sh <iter_dir> --prepare-prompt
#       → writes <iter_dir>/.validator_prompt.md, prints the path
#   scripts/dispatch_validator.sh <iter_dir> --validate-response <response.json>
#       → schema-checks response.json and copies to <iter_dir>/validator.json on success
set -euo pipefail
source "$(dirname "$0")/_env.sh"

ITER_DIR="${1:?usage: $0 <iter_dir> [--prepare-prompt | --validate-response <path>]}"
PROMPT_TEMPLATE="$REPO_ROOT/prompts/validator.md"
REF_DIR="$REPO_ROOT/benchmarks/reference_v2"
OUT="$ITER_DIR/validator.json"

if [[ ! -f "$ITER_DIR/metrics.json" ]]; then
  echo "ERROR: $ITER_DIR/metrics.json missing" >&2
  exit 1
fi

build_prompt() {
  cat "$PROMPT_TEMPLATE"
  echo ""
  echo "## Concrete artifact paths for this iter"
  echo ""
  echo "- iter_dir: $ITER_DIR"
  echo "- ref_dir: $REF_DIR"
  echo "- renders: $(ls "$ITER_DIR"/*.png 2>/dev/null | grep -v _diff10 | tr '\n' ' ')"
  echo "- references: $(ls "$REF_DIR"/*.png 2>/dev/null | tr '\n' ' ')"
  echo "- diff10: $(ls "$ITER_DIR"/*_diff10.png 2>/dev/null | tr '\n' ' ')"
  echo "- metrics: $ITER_DIR/metrics.json"
  echo ""
  echo "Read the metrics JSON inline:"
  echo '```json'
  cat "$ITER_DIR/metrics.json"
  echo '```'
  echo ""
  echo "Now produce the required JSON output. Nothing else."
}

invoke_validator() {
  build_prompt > "$ITER_DIR/.validator_prompt.md"
  echo "VALIDATOR_PROMPT_READY: $ITER_DIR/.validator_prompt.md"
}

validate_json_schema() {
  local file="$1"
  jq -e '.verdict | IN("KEEP", "REJECT", "NEEDS_REVIEW")' "$file" >/dev/null 2>&1 && \
  jq -e '.visual_checks | length == 8' "$file" >/dev/null 2>&1 && \
  jq -e '.psnr_check.pass | type == "boolean"' "$file" >/dev/null 2>&1 && \
  jq -e '.reasoning | length > 5' "$file" >/dev/null 2>&1
}

case "${2:---prepare-prompt}" in
  --prepare-prompt)
    invoke_validator
    ;;
  --validate-response)
    RESPONSE="${3:?--validate-response requires response file path}"
    if validate_json_schema "$RESPONSE"; then
      cp "$RESPONSE" "$OUT"
      echo "VALIDATOR_OK: $OUT"
    else
      echo "VALIDATOR_MALFORMED: $RESPONSE" >&2
      exit 2
    fi
    ;;
  *)
    echo "ERROR: unknown mode $2" >&2; exit 1;;
esac
