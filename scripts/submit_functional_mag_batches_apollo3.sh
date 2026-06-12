#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
DB_ROOT="${DB_ROOT:-/home/rsg-jcorre38/scratch/methanet_db}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.proposed.tsv}"
COHORT_RUN_ID="${COHORT_RUN_ID:-fgx_662_apollo3_$(date -u +%Y%m%d)}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/functional_metagenomics/${COHORT_RUN_ID}}"
ARRAY_WORKER="${ARRAY_WORKER:-${REPO_ROOT}/scripts/slurm/run_functional_mag_array_apollo3.sh}"
THREADS="${THREADS:-16}"
MEM="${MEM:-128G}"
# The 24-MAG calibration finished in <1h per MAG, but it only covered
# wetland/MUCC genomes. The full cohort includes much larger rumen inputs, so
# default to 24h for production and override down only for calibration tranches.
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
PARTITION="${PARTITION:-longjobs}"
# 12-way concurrency fits two longjobs nodes at 16 CPUs/task without saturating
# all 64 cores on each node, leaving headroom for tool-level I/O and memory.
CONCURRENCY="${CONCURRENCY:-12}"
START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-}"
ARRAY_SPEC="${ARRAY_SPEC:-}"
DRY_RUN="${DRY_RUN:-1}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -s "$MANIFEST" ]] || die "Manifest missing or empty: $MANIFEST"
[[ -x "$ARRAY_WORKER" ]] || die "Array worker missing or not executable: $ARRAY_WORKER"

total="$(
  awk -F '\t' '
    NR == 1 {
      for (i = 1; i <= NF; i++) col[$i] = i
      if (!("functional_run_include" in col)) exit 2
      next
    }
    $(col["functional_run_include"]) == "True" || $(col["functional_run_include"]) == "true" || $(col["functional_run_include"]) == "1" { n += 1 }
    END { print n + 0 }
  ' "$MANIFEST"
)"
[[ "$total" -gt 0 ]] || die "No functional_run_include=True rows found in $MANIFEST"
if [[ -z "$END_INDEX" ]]; then
  END_INDEX="$total"
fi
[[ "$START_INDEX" -ge 1 ]] || die "START_INDEX must be >= 1"
[[ "$END_INDEX" -le "$total" ]] || die "END_INDEX must be <= included row count (${total})"
[[ "$START_INDEX" -le "$END_INDEX" ]] || die "START_INDEX must be <= END_INDEX"
if [[ -z "$ARRAY_SPEC" ]]; then
  ARRAY_SPEC="${START_INDEX}-${END_INDEX}%${CONCURRENCY}"
fi

mkdir -p "${RESULT_ROOT}/logs/array"

cmd=(
  sbatch
  --partition="$PARTITION"
  --cpus-per-task="$THREADS"
  --mem="$MEM"
  --time="$TIME_LIMIT"
  --array="$ARRAY_SPEC"
  --output="${RESULT_ROOT}/logs/array/%A_%a.out"
  --error="${RESULT_ROOT}/logs/array/%A_%a.err"
  --export="ALL,REPO_ROOT=${REPO_ROOT},DB_ROOT=${DB_ROOT},MANIFEST=${MANIFEST},COHORT_RUN_ID=${COHORT_RUN_ID},RESULT_BASE=${RESULT_ROOT}/per_mag,THREADS=${THREADS}"
  "$ARRAY_WORKER"
)

printf 'Prepared cohort batch command for %s included MAGs (array=%s, time=%s, mem=%s, cpus=%s):\n' \
  "$total" "$ARRAY_SPEC" "$TIME_LIMIT" "$MEM" "$THREADS"
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: not submitting."
  exit 0
fi

"${cmd[@]}"
