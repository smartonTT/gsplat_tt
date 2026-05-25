#!/usr/bin/env bash
# Usage: scripts/dispatch_validator.sh <iter_dir> [--prepare-prompt | --validate-response <response.json>]
#
# Dispatches the validator subagent (Sonnet 4.6, fresh context) with only the
# artifacts in <iter_dir> and the rules in prompts/validator.md. The supervisor
# is the actual agent dispatcher; this script just prepares the prompt and
# validates the schema of the agent's JSON response.
set -euo pipefail

ITER_DIR="${1:?usage: $0 <iter_dir> [--prepare-prompt | --validate-response <response.json>]}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PROMPT="$REPO/prompts/validator.md"
REF_DIR="$REPO/benchmarks/reference"
OUT="$ITER_DIR/validator.json"

if [[ ! -f "$ITER_DIR/metrics.json" ]]; then
  echo "ERROR: $ITER_DIR/metrics.json missing" >&2
  exit 1
fi

# Build the validator subagent prompt: validator.md + concrete file paths.
build_prompt() {
  cat "$PROMPT"
  echo ""
  echo "## Concrete artifact paths for this iter"
  echo ""
  echo "- iter_dir: $ITER_DIR"
  echo "- ref_dir: $REF_DIR"
  echo "- renders: $ITER_DIR/hero.png $ITER_DIR/side.png $ITER_DIR/top.png"
  echo "- references: $REF_DIR/stitch_hero.png $REF_DIR/stitch_side.png $REF_DIR/stitch_top.png"
  echo "- diff10: $ITER_DIR/hero_diff10.png $ITER_DIR/side_diff10.png $ITER_DIR/top_diff10.png"
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
  jq -e '.reasoning | length > 10' "$file" >/dev/null 2>&1
}

# Two modes:
#   --prepare-prompt           → write prompt file, print path; supervisor dispatches the agent
#   --validate-response <file> → schema-check agent response; copy to validator.json or exit 2
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
