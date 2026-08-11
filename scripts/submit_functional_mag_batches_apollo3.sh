#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
DB_ROOT="${DB_ROOT:-/home/rsg-jcorre38/scratch/methanet_db}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.mag_bin_remaining.tsv}"
COHORT_RUN_ID="${COHORT_RUN_ID:-fgx_662_apollo3_$(date -u +%Y%m%d)}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/functional_metagenomics/${COHORT_RUN_ID}}"
ARRAY_WORKER="${ARRAY_WORKER:-${REPO_ROOT}/scripts/slurm/run_functional_mag_array_apollo3.sh}"
THREADS="${THREADS:-16}"
MEM="${MEM:-64G}"
# MAG/bin relaunch inputs are intentionally filtered away from assembly-scale
# no-bin rumen records. Keep a full-day walltime envelope for slower bacteria
# MAGs and for clean tool/Slurm closeout; reserve 48-72h/128G+ profiles for an
# explicit assembly-context tranche.
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
PARTITION="${PARTITION:-longjobs}"
# 12-way concurrency fits two longjobs nodes at 16 CPUs/task without saturating
# all 64 cores on each node, leaving headroom for tool-level I/O and memory.
CONCURRENCY="${CONCURRENCY:-12}"
START_INDEX="${START_INDEX:-1}"
END_INDEX="${END_INDEX:-}"
ARRAY_SPEC="${ARRAY_SPEC:-}"
DRY_RUN="${DRY_RUN:-1}"
ALLOW_ASSEMBLY_CONTEXT="${ALLOW_ASSEMBLY_CONTEXT:-0}"
DEPENDENCY="${DEPENDENCY:-}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -s "$MANIFEST" ]] || die "Manifest missing or empty: $MANIFEST"
[[ -x "$ARRAY_WORKER" ]] || die "Array worker missing or not executable: $ARRAY_WORKER"

case "$RESULT_ROOT" in
  /*) ;;
  *) RESULT_ROOT="${REPO_ROOT}/${RESULT_ROOT#./}" ;;
esac

if ! total="$(
  awk -F '\t' '
    function istrue(value) {
      return value == "True" || value == "true" || value == "1" || value == "yes" || value == "Y" || value == "y"
    }
    function fail(message) {
      print message > "/dev/stderr"
      bad = 1
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        gsub(/\r$/, "", $i)
        col[$i] = i
      }
      required = "proteome_id mag_fasta proteome_faa functional_run_include analysis_unit_type mbag_mag_level_include"
      split(required, r, " ")
      for (j in r) {
        if (!(r[j] in col)) {
          fail("missing required manifest column: " r[j])
        }
      }
      if (bad) exit 2
      next
    }
    {
      for (i = 1; i <= NF; i++) gsub(/\r$/, "", $i)
      include = $(col["functional_run_include"])
      unit = $(col["analysis_unit_type"])
      mbag = $(col["mbag_mag_level_include"])
      if (istrue(include) &&
          (unit == "mag_bin" || ENVIRON["ALLOW_ASSEMBLY_CONTEXT"] == "1") &&
          (istrue(mbag) || ENVIRON["ALLOW_ASSEMBLY_CONTEXT"] == "1")) {
        n += 1
        proteome_id = $(col["proteome_id"])
        if (proteome_id == "") {
          fail("included row " NR " has empty proteome_id")
        } else if (seen[proteome_id]++) {
          duplicates[proteome_id] = 1
        }
        if ($(col["mag_fasta"]) == "") {
          fail("included row " NR " proteome_id=" proteome_id " has empty mag_fasta")
        }
        if ($(col["proteome_faa"]) == "") {
          fail("included row " NR " proteome_id=" proteome_id " has empty proteome_faa")
        }
        if (col["match_status"] > 0 && $(col["match_status"]) == "missing_payload") {
          fail("included row " NR " proteome_id=" proteome_id " has match_status=missing_payload")
        }
      }
    }
    END {
      for (proteome_id in duplicates) {
        fail("duplicate included proteome_id: " proteome_id)
      }
      if (bad) exit 3
      print n + 0
    }
  ' "$MANIFEST"
)"; then
  die "Functional manifest preflight failed: $MANIFEST"
fi
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

cmd=(
  sbatch
  --partition="$PARTITION"
  --cpus-per-task="$THREADS"
  --mem="$MEM"
  --time="$TIME_LIMIT"
  --array="$ARRAY_SPEC"
  --output="${RESULT_ROOT}/logs/array/%A_%a.out"
  --error="${RESULT_ROOT}/logs/array/%A_%a.err"
  --export="ALL,REPO_ROOT=${REPO_ROOT},DB_ROOT=${DB_ROOT},MANIFEST=${MANIFEST},COHORT_RUN_ID=${COHORT_RUN_ID},RESULT_BASE=${RESULT_ROOT}/per_mag,RESULT_ROOT=,THREADS=${THREADS},ALLOW_ASSEMBLY_CONTEXT=${ALLOW_ASSEMBLY_CONTEXT}"
  "$ARRAY_WORKER"
)
if [[ -n "$DEPENDENCY" ]]; then
  cmd=(sbatch --dependency="$DEPENDENCY" "${cmd[@]:1}")
fi

printf 'Prepared cohort batch command for %s included MAGs (array=%s, time=%s, mem=%s, cpus=%s):\n' \
  "$total" "$ARRAY_SPEC" "$TIME_LIMIT" "$MEM" "$THREADS"
printf '%q ' "${cmd[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1: not submitting."
  exit 0
fi

mkdir -p "${RESULT_ROOT}/logs/array"
"${cmd[@]}"
