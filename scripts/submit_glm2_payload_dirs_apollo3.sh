#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
GLM2_RESULTS_DIRS="${GLM2_RESULTS_DIRS:-}"
GLM2_RESULTS_DIR_LIST="${GLM2_RESULTS_DIR_LIST:-}"
GLM2_WORKER="${GLM2_WORKER:-${REPO_ROOT}/scripts/slurm/run_glm2_smoke_apolo3.sh}"
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-methanet_glm2}"
PARTITION="${PARTITION:-accel}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-8}"
MEM="${MEM:-80G}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
DRY_RUN="${DRY_RUN:-1}"
DEPENDENCY="${DEPENDENCY:-}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  GLM2_RESULTS_DIRS='results/contextual_genomics/lane_shard001;results/contextual_genomics/lane_shard002' \
  DRY_RUN=1 \
  scripts/submit_glm2_payload_dirs_apollo3.sh

Or:
  GLM2_RESULTS_DIR_LIST=<one-results-dir-per-line.txt> \
  scripts/submit_glm2_payload_dirs_apollo3.sh

Each results directory must already contain prepared_inputs/glm2_sequences.jsonl.
DRY_RUN defaults to 1 and prints sbatch commands without submitting.
EOF
}

[[ "${1:-}" != "-h" && "${1:-}" != "--help" ]] || {
  usage
  exit 0
}

[[ -x "$GLM2_WORKER" ]] || die "gLM2 worker missing or not executable: $GLM2_WORKER"
[[ -n "$GLM2_RESULTS_DIRS" || -n "$GLM2_RESULTS_DIR_LIST" ]] || die "Set GLM2_RESULTS_DIRS or GLM2_RESULTS_DIR_LIST"

results_dirs=()
if [[ -n "$GLM2_RESULTS_DIRS" ]]; then
  IFS=';' read -r -a results_dirs <<< "$GLM2_RESULTS_DIRS"
fi
if [[ -n "$GLM2_RESULTS_DIR_LIST" ]]; then
  [[ -s "$GLM2_RESULTS_DIR_LIST" ]] || die "GLM2_RESULTS_DIR_LIST missing or empty: $GLM2_RESULTS_DIR_LIST"
  while IFS= read -r line; do
    [[ -n "${line//[[:space:]]/}" ]] || continue
    [[ "${line:0:1}" != "#" ]] || continue
    results_dirs+=("$line")
  done < "$GLM2_RESULTS_DIR_LIST"
fi

normalized_results_dirs=()
declare -A seen_dirs=()
for raw_dir in "${results_dirs[@]}"; do
  raw_dir="${raw_dir#"${raw_dir%%[![:space:]]*}"}"
  raw_dir="${raw_dir%"${raw_dir##*[![:space:]]}"}"
  [[ -n "$raw_dir" ]] || continue
  case "$raw_dir" in
    /*) results_dir="$raw_dir" ;;
    *) results_dir="${REPO_ROOT}/${raw_dir}" ;;
  esac
  [[ -z "${seen_dirs[$results_dir]:-}" ]] || die "Duplicate gLM2 results directory in launch list: $results_dir"
  seen_dirs[$results_dir]=1
  normalized_results_dirs+=("$results_dir")
done
[[ "${#normalized_results_dirs[@]}" -gt 0 ]] || die "No gLM2 results directories selected"

prepared=0
skipped=0
for results_dir in "${normalized_results_dirs[@]}"; do
  if [[ ! -s "${results_dir}/prepared_inputs/glm2_sequences.jsonl" ]]; then
    printf 'Skipping unprepared gLM2 payload: %s\n' "$results_dir"
    skipped=$((skipped + 1))
    continue
  fi
  mkdir -p "${results_dir}/logs"
  job_suffix="$(printf '%s' "$(basename "$results_dir")" | tr -c '[:alnum:]_' '_')"
  job_name="${JOB_NAME_PREFIX}_${job_suffix}"
  export_value="ALL,REPO_ROOT=${REPO_ROOT},RESULTS_DIR=${results_dir}"
  for env_name in CONDA_ENV MODEL_NAME MODEL_REVISION HF_HOME TRANSFORMERS_CACHE; do
    if [[ -n "${!env_name:-}" ]]; then
      export_value+=",${env_name}=${!env_name}"
    fi
  done

  cmd=(
    sbatch
    --job-name="$job_name"
    --partition="$PARTITION"
    --gres="$GRES"
    --cpus-per-task="$CPUS"
    --mem="$MEM"
    --time="$TIME_LIMIT"
    --output="${results_dir}/logs/slurm-%j.out"
    --error="${results_dir}/logs/slurm-%j.err"
    --export="$export_value"
    "$GLM2_WORKER"
  )
  if [[ -n "$DEPENDENCY" ]]; then
    cmd=(sbatch --dependency="$DEPENDENCY" "${cmd[@]:1}")
  fi

  printf 'Prepared gLM2 payload %s:\n' "$results_dir"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  prepared=$((prepared + 1))
  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}"
  fi
done

printf 'payloads_prepared\t%s\n' "$prepared"
printf 'unprepared_payloads_skipped\t%s\n' "$skipped"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: not submitting."
fi
