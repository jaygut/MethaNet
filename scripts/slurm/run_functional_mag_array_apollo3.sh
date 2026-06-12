#!/usr/bin/env bash
# SLURM array worker for the 662-MAG functional-metagenomics cohort.
# Runs exactly one manifest row selected by SLURM_ARRAY_TASK_ID.
# This script does not submit jobs by itself.
#SBATCH --job-name=methanet_fgx_array
#SBATCH --partition=longjobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
DB_ROOT="${DB_ROOT:-/home/rsg-jcorre38/scratch/methanet_db}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.proposed.tsv}"
COHORT_RUN_ID="${COHORT_RUN_ID:-fgx_662_apollo3_$(date -u +%Y%m%d)}"
RESULT_BASE="${RESULT_BASE:-${REPO_ROOT}/results/functional_metagenomics/${COHORT_RUN_ID}/per_mag}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
RUNNER="${RUNNER:-${REPO_ROOT}/scripts/slurm/run_one_mag_functional_smoke_apollo3.sh}"
DBCAN_COMPAT_DIR="${DBCAN_COMPAT_DIR:-${DB_ROOT}/dbcan_compat_pressed}"
PRUNE_SUCCESS="${PRUNE_SUCCESS:-1}"
COMPRESS_LOGS="${COMPRESS_LOGS:-1}"
ARRAY_DRY_RUN="${ARRAY_DRY_RUN:-0}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -n "$TASK_INDEX" ]] || die "TASK_INDEX or SLURM_ARRAY_TASK_ID is required"
[[ -s "$MANIFEST" ]] || die "Manifest missing or empty: $MANIFEST"
[[ -x "$RUNNER" ]] || die "Runner missing or not executable: $RUNNER"

row="$(
  awk -F '\t' -v idx="$TASK_INDEX" '
    NR == 1 {
      for (i = 1; i <= NF; i++) col[$i] = i
      required = "proteome_id mag_id mag_fasta proteome_faa functional_run_include"
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
      include = $(col["functional_run_include"])
      if (include == "True" || include == "true" || include == "1") {
        included += 1
        if (included == idx) {
          printf("%s\t%s\t%s\t%s\t%s\n", $(col["proteome_id"]), $(col["mag_id"]), $(col["mag_fasta"]), $(col["proteome_faa"]), include)
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

IFS=$'\t' read -r PROTEOME_ID MAG_ID MAG_FASTA PROTEOME_FAA INCLUDE <<< "$row"
[[ "$INCLUDE" == "True" || "$INCLUDE" == "true" || "$INCLUDE" == "1" ]] || die "Manifest row ${TASK_INDEX} is not marked functional_run_include=True"

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
mkdir -p "${RESULT_ROOT}/logs"

export REPO_ROOT DB_ROOT FASTA PROTEOME_ID MAG_ID THREADS RUN_ID RESULT_ROOT COHORT_RUN_ID
export PROTEOME_FAA="$PROTEOME_FAA_ABS"
export DBCAN_COMPAT_DIR PRUNE_SUCCESS COMPRESS_LOGS

if [[ "$ARRAY_DRY_RUN" == "1" ]]; then
  printf 'task_index\t%s\n' "$TASK_INDEX"
  printf 'proteome_id\t%s\n' "$PROTEOME_ID"
  printf 'mag_id\t%s\n' "$MAG_ID"
  printf 'fasta\t%s\n' "$FASTA"
  printf 'proteome_faa\t%s\n' "$PROTEOME_FAA_ABS"
  printf 'result_root\t%s\n' "$RESULT_ROOT"
  printf 'cohort_run_id\t%s\n' "$COHORT_RUN_ID"
  printf 'dbcan_compat_dir\t%s\n' "$DBCAN_COMPAT_DIR"
  exit 0
fi

exec "$RUNNER"
