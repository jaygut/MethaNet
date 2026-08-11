#!/usr/bin/env bash
# SLURM array worker for scoped MethaNet functional-metagenomics manifests.
# Runs exactly one manifest row selected by SLURM_ARRAY_TASK_ID.
# This script does not submit jobs by itself.
#SBATCH --job-name=methanet_fgx_array
#SBATCH --partition=longjobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
DB_ROOT="${DB_ROOT:-/home/rsg-jcorre38/scratch/methanet_db}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.mag_bin_only.tsv}"
COHORT_RUN_ID="${COHORT_RUN_ID:-fgx_662_apollo3_$(date -u +%Y%m%d)}"
RESULT_BASE="${RESULT_BASE:-${REPO_ROOT}/results/functional_metagenomics/${COHORT_RUN_ID}/per_mag}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
RUNNER="${RUNNER:-${REPO_ROOT}/scripts/slurm/run_one_mag_functional_smoke_apollo3.sh}"
DBCAN_COMPAT_DIR="${DBCAN_COMPAT_DIR:-${DB_ROOT}/dbcan_compat_pressed}"
PRUNE_SUCCESS="${PRUNE_SUCCESS:-1}"
COMPRESS_LOGS="${COMPRESS_LOGS:-1}"
ARRAY_DRY_RUN="${ARRAY_DRY_RUN:-${DRY_RUN:-0}}"
ALLOW_ASSEMBLY_CONTEXT="${ALLOW_ASSEMBLY_CONTEXT:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CLEANUP_LINGERING_PROCESSES="${CLEANUP_LINGERING_PROCESSES:-0}"
LINGERING_PROCESS_MAX_TARGETS="${LINGERING_PROCESS_MAX_TARGETS:-32}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

collect_lingering_pids() {
  local root_pid="${1:-$$}"
  local self="$$"
  local parent="${PPID:-}"
  [[ "$root_pid" =~ ^[0-9]+$ ]] || return 0
  [[ "$root_pid" -gt 1 ]] || return 0

  ps -eo pid=,ppid= 2>/dev/null | awk -v root="$root_pid" -v self="$self" -v parent_pid="$parent" '
      {
        pid = $1
        ppid = $2
        parent[pid] = ppid
        seen[pid] = 1
      }
      END {
        desc[root] = 1
        changed = 1
        while (changed) {
          changed = 0
          for (pid in seen) {
            if (!desc[pid] && desc[parent[pid]]) {
              desc[pid] = 1
              changed = 1
            }
          }
        }
        for (pid in desc) {
          if (pid != root && pid != self && pid != parent_pid && pid > 1) print pid
        }
      }
    ' | awk '
    $1 ~ /^[0-9]+$/ && !seen[$1] {
      seen[$1] = 1
      print $1
    }
  '
}

describe_pids() {
  [[ "$#" -gt 0 ]] || return 0
  ps -o pid=,ppid=,pgid=,sid=,stat=,comm= -p "$(IFS=,; echo "$*")" 2>/dev/null || true
}

cleanup_lingering_processes() {
  local label="${1:-array closeout}"
  local root_pid="${2:-$$}"
  local pid
  local targets=()

  [[ "$CLEANUP_LINGERING_PROCESSES" == "1" ]] || return 0

  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    targets+=("$pid")
  done < <(collect_lingering_pids "$root_pid")

  [[ "${#targets[@]}" -gt 0 ]] || return 0
  if [[ "${#targets[@]}" -gt "$LINGERING_PROCESS_MAX_TARGETS" ]]; then
    printf 'Array worker refusing to clean %s lingering processes after %s; target count exceeds LINGERING_PROCESS_MAX_TARGETS=%s\n' \
      "${#targets[@]}" "$label" "$LINGERING_PROCESS_MAX_TARGETS" >&2
    describe_pids "${targets[@]}" >&2
    return 0
  fi
  printf 'Array worker cleaning lingering processes after %s: %s\n' "$label" "${targets[*]}" >&2
  describe_pids "${targets[@]}" >&2
  kill -TERM "${targets[@]}" 2>/dev/null || true
  sleep "${LINGERING_PROCESS_TERM_GRACE_SECONDS:-2}"

  local survivors=()
  for pid in "${targets[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      survivors+=("$pid")
    fi
  done
  if [[ "${#survivors[@]}" -gt 0 ]]; then
    printf 'Array worker force-killing lingering processes after %s: %s\n' "$label" "${survivors[*]}" >&2
    describe_pids "${survivors[@]}" >&2
    kill -KILL "${survivors[@]}" 2>/dev/null || true
  fi
}

[[ -n "$TASK_INDEX" ]] || die "TASK_INDEX or SLURM_ARRAY_TASK_ID is required"
[[ -s "$MANIFEST" ]] || die "Manifest missing or empty: $MANIFEST"
[[ -x "$RUNNER" ]] || die "Runner missing or not executable: $RUNNER"

case "$RESULT_BASE" in
  /*) ;;
  *) RESULT_BASE="${REPO_ROOT}/${RESULT_BASE#./}" ;;
esac

row="$(
  awk -F '\t' -v idx="$TASK_INDEX" '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        gsub(/\r$/, "", $i)
        col[$i] = i
      }
      required = "proteome_id mag_id mag_fasta proteome_faa functional_run_include analysis_unit_type mbag_mag_level_include claim_scope"
      split(required, r, " ")
      for (j in r) {
        if (!(r[j] in col)) {
          printf("missing_column:%s\n", r[j])
          exit 2
        }
      }
      next
    }
    {
      for (i = 1; i <= NF; i++) gsub(/\r$/, "", $i)
      include = $(col["functional_run_include"])
      unit = $(col["analysis_unit_type"])
      mbag = $(col["mbag_mag_level_include"])
      if ((include == "True" || include == "true" || include == "1") &&
          (unit == "mag_bin" || ENVIRON["ALLOW_ASSEMBLY_CONTEXT"] == "1") &&
          (mbag == "true" || mbag == "True" || mbag == "1" || ENVIRON["ALLOW_ASSEMBLY_CONTEXT"] == "1")) {
        included += 1
        if (included == idx) {
          printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", $(col["proteome_id"]), $(col["mag_id"]), $(col["mag_fasta"]), $(col["proteome_faa"]), include, unit, mbag, $(col["claim_scope"]))
          found = 1
          exit 0
        }
      }
    }
    END {
      if (!found) exit 3
    }
  ' "$MANIFEST"
)" || die "Could not read manifest row for task index ${TASK_INDEX}"

case "$row" in
  missing_column:*) die "$row in $MANIFEST" ;;
esac

IFS=$'\t' read -r PROTEOME_ID MAG_ID MAG_FASTA PROTEOME_FAA INCLUDE ANALYSIS_UNIT_TYPE MBAG_MAG_LEVEL_INCLUDE CLAIM_SCOPE <<< "$row"
[[ "$INCLUDE" == "True" || "$INCLUDE" == "true" || "$INCLUDE" == "1" ]] || die "Manifest row ${TASK_INDEX} is not marked functional_run_include=True"
if [[ "$ALLOW_ASSEMBLY_CONTEXT" != "1" ]]; then
  [[ "$ANALYSIS_UNIT_TYPE" == "mag_bin" ]] || die "Refusing non-MAG unit for MAG-level run: ${PROTEOME_ID} analysis_unit_type=${ANALYSIS_UNIT_TYPE}"
  [[ "$MBAG_MAG_LEVEL_INCLUDE" == "true" || "$MBAG_MAG_LEVEL_INCLUDE" == "True" || "$MBAG_MAG_LEVEL_INCLUDE" == "1" ]] || die "Refusing row not marked mbag_mag_level_include=true: ${PROTEOME_ID}"
fi

case "$MAG_FASTA" in
  /*) FASTA="$MAG_FASTA" ;;
  *) FASTA="${REPO_ROOT}/${MAG_FASTA}" ;;
esac
case "$PROTEOME_FAA" in
  /*) PROTEOME_FAA_ABS="$PROTEOME_FAA" ;;
  *) PROTEOME_FAA_ABS="${REPO_ROOT}/${PROTEOME_FAA}" ;;
esac

[[ -s "$FASTA" ]] || die "MAG FASTA missing or empty: $FASTA"
[[ -s "$PROTEOME_FAA_ABS" ]] || die "Proteome FAA missing or empty: $PROTEOME_FAA_ABS"

RUN_ID="${RUN_ID:-fgx_${TASK_INDEX}_${PROTEOME_ID}_$(date -u +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${RESULT_BASE}/${PROTEOME_ID}/${RUN_ID}}"
case "$RESULT_ROOT" in
  /*) ;;
  *) RESULT_ROOT="${REPO_ROOT}/${RESULT_ROOT#./}" ;;
esac

if [[ "$SKIP_COMPLETED" == "1" ]]; then
  existing_completed="$(
    { find "${RESULT_BASE}/${PROTEOME_ID}" -path '*/curated/run_record.json' -type f 2>/dev/null || true; } |
      while IFS= read -r run_record; do
        run_dir="$(dirname "$(dirname "$run_record")")"
        if [[ -f "${run_dir}/COMPLETE" ]]; then
          printf '%s\n' "$run_dir"
          break
        fi
      done
  )"
  if [[ -n "$existing_completed" ]]; then
    echo "Already complete: proteome_id=${PROTEOME_ID} existing_run=${existing_completed}"
    exit 0
  fi
fi

if [[ "$ARRAY_DRY_RUN" == "1" ]]; then
  printf 'task_index\t%s\n' "$TASK_INDEX"
  printf 'proteome_id\t%s\n' "$PROTEOME_ID"
  printf 'mag_id\t%s\n' "$MAG_ID"
  printf 'fasta\t%s\n' "$FASTA"
  printf 'proteome_faa\t%s\n' "$PROTEOME_FAA_ABS"
  printf 'result_root\t%s\n' "$RESULT_ROOT"
  printf 'cohort_run_id\t%s\n' "$COHORT_RUN_ID"
  printf 'analysis_unit_type\t%s\n' "$ANALYSIS_UNIT_TYPE"
  printf 'mbag_mag_level_include\t%s\n' "$MBAG_MAG_LEVEL_INCLUDE"
  printf 'claim_scope\t%s\n' "$CLAIM_SCOPE"
  printf 'dbcan_compat_dir\t%s\n' "$DBCAN_COMPAT_DIR"
  exit 0
fi

mkdir -p "${RESULT_ROOT}/logs"

export REPO_ROOT DB_ROOT FASTA PROTEOME_ID MAG_ID THREADS RUN_ID RESULT_ROOT COHORT_RUN_ID
export PROTEOME_FAA="$PROTEOME_FAA_ABS"
export ANALYSIS_UNIT_TYPE MBAG_MAG_LEVEL_INCLUDE CLAIM_SCOPE
export DBCAN_COMPAT_DIR PRUNE_SUCCESS COMPRESS_LOGS

set +e
"$RUNNER"
rc=$?
set -e

if [[ "$rc" -ne 0 ]]; then
  if [[ -f "${RESULT_ROOT}/COMPLETE" && -s "${RESULT_ROOT}/curated/run_record.json" ]]; then
    echo "Runner returned rc=${rc}, but curated completion exists; treating as successful closeout: ${RESULT_ROOT}"
    rc=0
  fi
fi

if [[ "$rc" -ne 0 ]]; then
  {
    printf 'timestamp\t%s\n' "$(date -Is)"
    printf 'task_index\t%s\n' "$TASK_INDEX"
    printf 'proteome_id\t%s\n' "$PROTEOME_ID"
    printf 'mag_id\t%s\n' "$MAG_ID"
    printf 'run_id\t%s\n' "$RUN_ID"
    printf 'result_root\t%s\n' "$RESULT_ROOT"
    printf 'runner\t%s\n' "$RUNNER"
    printf 'exit_code\t%s\n' "$rc"
  } > "${RESULT_ROOT}/failure.tsv"
  touch "${RESULT_ROOT}/FAILED"
fi

if [[ "$rc" -eq 0 ]]; then
  cleanup_lingering_processes "successful array closeout"
  echo "Completed proteome_id=${PROTEOME_ID} result_root=${RESULT_ROOT}"
fi

exit "$rc"
