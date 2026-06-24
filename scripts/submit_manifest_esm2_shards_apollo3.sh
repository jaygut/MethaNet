#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
SHARD_MANIFEST="${SHARD_MANIFEST:-}"
OUTPUT_DIR_TEMPLATE="${OUTPUT_DIR_TEMPLATE:-}"
ESM2_WORKER="${ESM2_WORKER:-${REPO_ROOT}/scripts/slurm/run_manifest_esm2_embeddings_apolo3.sh}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-methanet_esm2}"
PARTITION="${PARTITION:-accel}"
CPUS="${CPUS:-6}"
MEM="${MEM:-48G}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00:00}"
DRY_RUN="${DRY_RUN:-1}"
DEPENDENCY="${DEPENDENCY:-}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  SHARD_MANIFEST=<shard_manifest.tsv> \
  OUTPUT_DIR_TEMPLATE='results/.../esm2_shard{shard}/artifacts' \
  ESM2_INCLUDE_COL=esm2_include \
  DRY_RUN=0 \
  scripts/submit_manifest_esm2_shards_apollo3.sh

The shard manifest must contain at least: shard, path, rows.
Template tokens available for OUTPUT_DIR_TEMPLATE:
  {shard}       zero-padded shard id from the manifest, e.g. 001
  {shard_int}   shard id without leading zeroes, e.g. 1
  {path_stem}   shard TSV filename stem

DRY_RUN defaults to 1 and prints sbatch commands without submitting.
EOF
}

[[ "${1:-}" != "-h" && "${1:-}" != "--help" ]] || {
  usage
  exit 0
}

[[ -n "$SHARD_MANIFEST" ]] || die "SHARD_MANIFEST is required"
[[ -n "$OUTPUT_DIR_TEMPLATE" ]] || die "OUTPUT_DIR_TEMPLATE is required"
[[ -s "$SHARD_MANIFEST" ]] || die "Shard manifest missing or empty: $SHARD_MANIFEST"
[[ -x "$ESM2_WORKER" ]] || die "ESM2 worker missing or not executable: $ESM2_WORKER"

render_template() {
  local template="$1"
  local shard="$2"
  local path_stem="$3"
  local shard_int
  shard_int="$((10#$shard))"
  template="${template//\{shard\}/$shard}"
  template="${template//\{shard_int\}/$shard_int}"
  template="${template//\{path_stem\}/$path_stem}"
  printf '%s' "$template"
}

submitted=0
skipped=0
line_no=1
declare -A seen_shards=()
declare -A seen_paths=()

IFS=$'\t' read -r header_shard header_path header_rows _header_rest < "$SHARD_MANIFEST" || die "Shard manifest unreadable: $SHARD_MANIFEST"
header_shard="${header_shard%$'\r'}"
header_path="${header_path%$'\r'}"
header_rows="${header_rows%$'\r'}"
[[ "$header_shard" == "shard" && "$header_path" == "path" && "$header_rows" == "rows" ]] || {
  die "Shard manifest header must start with: shard<TAB>path<TAB>rows"
}

while IFS=$'\t' read -r shard path rows _rest; do
  line_no=$((line_no + 1))
  shard="${shard%$'\r'}"
  path="${path%$'\r'}"
  rows="${rows%$'\r'}"
  [[ -n "$shard" && -n "$path" && -n "$rows" ]] || die "Malformed shard manifest row: shard=${shard:-} path=${path:-} rows=${rows:-}"
  [[ "$shard" =~ ^[0-9]+$ ]] || die "Shard id must be numeric at line ${line_no}: $shard"
  [[ "$rows" =~ ^[0-9]+$ ]] || die "Shard rows must be a non-negative integer at line ${line_no}: $rows"
  [[ -z "${seen_shards[$shard]:-}" ]] || die "Duplicate shard id in manifest at line ${line_no}: $shard"
  [[ -z "${seen_paths[$path]:-}" ]] || die "Duplicate shard path in manifest at line ${line_no}: $path"
  seen_shards[$shard]=1
  seen_paths[$path]=1
done < <(tail -n +2 "$SHARD_MANIFEST")

line_no=1
while IFS=$'\t' read -r shard path rows _rest; do
  line_no=$((line_no + 1))
  shard="${shard%$'\r'}"
  path="${path%$'\r'}"
  rows="${rows%$'\r'}"
  [[ -n "$shard" && -n "$path" && -n "$rows" ]] || die "Malformed shard manifest row: shard=${shard:-} path=${path:-} rows=${rows:-}"
  if [[ "$rows" -eq 0 ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  case "$path" in
    /*) shard_path="$path" ;;
    *) shard_path="${REPO_ROOT}/${path}" ;;
  esac
  [[ -s "$shard_path" ]] || die "Shard TSV missing or empty: $shard_path"

  path_stem="$(basename "$shard_path" .tsv)"
  output_dir="$(render_template "$OUTPUT_DIR_TEMPLATE" "$shard" "$path_stem")"
  case "$output_dir" in
    /*) output_abs="$output_dir" ;;
    *) output_abs="${REPO_ROOT}/${output_dir}" ;;
  esac
  job_name="${JOB_NAME_PREFIX}_s${shard}"

  export_value="ALL,REPO_ROOT=${REPO_ROOT},ESM2_MANIFEST=${shard_path},ESM2_OUTPUT_DIR=${output_abs}"
  for env_name in \
    CONDA_ENV SCRATCH_BASE ESM2_CACHE_BASE ESM2_ID_COL ESM2_FAA_COL ESM2_MAG_ID_COL \
    ESM2_SOURCE_COL ESM2_ECOSYSTEM_COL ESM2_DOMAIN_COL ESM2_SOURCE_GROUP_COL \
    ESM2_PROTEIN_COUNT_COL ESM2_INCLUDE_COL ESM2_MODEL_NAME ESM2_BATCH_SIZE \
    ESM2_MAX_LENGTH ESM2_MAX_PROTEINS_PER_PROTEOME ESM2_MIN_AA_LEN \
    ESM2_CHECKPOINT_EVERY ESM2_DEVICE ESM2_FP16; do
    if [[ -n "${!env_name:-}" ]]; then
      export_value+=",${env_name}=${!env_name}"
    fi
  done

  cmd=(
    sbatch
    --job-name="$job_name"
    --partition="$PARTITION"
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    --time="$TIME_LIMIT"
    --export="$export_value"
    "$ESM2_WORKER"
  )
  if [[ -n "$DEPENDENCY" ]]; then
    cmd=(sbatch --dependency="$DEPENDENCY" "${cmd[@]:1}")
  fi

  printf 'Prepared ESM2 shard %s (%s rows):\n' "$shard" "$rows"
  printf '%q ' "${cmd[@]}"
  printf '\n'

  submitted=$((submitted + 1))
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$output_abs"
    "${cmd[@]}"
  fi
done < <(tail -n +2 "$SHARD_MANIFEST")

printf 'shards_prepared\t%s\n' "$submitted"
printf 'empty_shards_skipped\t%s\n' "$skipped"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: not submitting."
fi
