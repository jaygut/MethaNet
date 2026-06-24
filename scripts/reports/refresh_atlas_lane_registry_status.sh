#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
LANE_REGISTRY="${LANE_REGISTRY:-${REPO_ROOT}/configs/methanet_atlas_lanes.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/reports}"
STATUS_PREFIX="${STATUS_PREFIX:-atlas_lane_registry_status}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
SUMMARIZER="${SUMMARIZER:-${REPO_ROOT}/scripts/reports/summarize_atlas_lane_registry.py}"
VALIDATOR="${VALIDATOR:-${REPO_ROOT}/scripts/reports/validate_atlas_lane_registry.py}"
DRY_RUN="${DRY_RUN:-0}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -s "$LANE_REGISTRY" ]] || die "Lane registry missing or empty: $LANE_REGISTRY"
[[ -s "$SUMMARIZER" ]] || die "Summarizer missing: $SUMMARIZER"
[[ -s "$VALIDATOR" ]] || die "Validator missing: $VALIDATOR"
[[ -x "$PYTHON" ]] || die "Python executable missing or not executable: $PYTHON"

mkdir -p "$OUTPUT_DIR"

base="${OUTPUT_DIR}/${STATUS_PREFIX}_${STAMP}"
validation_json="${base}.validation.json"
validate_cmd=(
  "$PYTHON"
  "$VALIDATOR"
  --repo-root "$REPO_ROOT"
  --lane-registry "$LANE_REGISTRY"
  --output-json "$validation_json"
  --allow-missing-optional-paths
)
cmd=(
  "$PYTHON"
  "$SUMMARIZER"
  --repo-root "$REPO_ROOT"
  --lane-registry "$LANE_REGISTRY"
  --output-tsv "${base}.tsv"
  --output-json "${base}.json"
  --output-md "${base}.md"
)

printf 'Prepared lane-registry validation command:\n'
printf '%q ' "${validate_cmd[@]}"
printf '\n'
printf 'Prepared lane-registry refresh command:\n'
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: not writing status artifacts."
  exit 0
fi

"${validate_cmd[@]}"
"${cmd[@]}"
printf 'validation_json\t%s\n' "$validation_json"
printf 'status_tsv\t%s\n' "${base}.tsv"
printf 'status_json\t%s\n' "${base}.json"
printf 'status_md\t%s\n' "${base}.md"
