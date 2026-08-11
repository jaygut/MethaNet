#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
ACTION="${ACTION:-smoke}"
PARTITION="${PARTITION:-bigmem}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
SMOKE_TIME_LIMIT="${SMOKE_TIME_LIMIT:-04:00:00}"
FULL_TIME_LIMIT="${FULL_TIME_LIMIT:-48:00:00}"
THREADS="${THREADS:-16}"
MEM="${MEM:-64G}"
DRY_RUN="${DRY_RUN:-1}"
REQUIRE_HEALTHY="${REQUIRE_HEALTHY:-1}"
SMOKE_SHARD="${SMOKE_SHARD:-001}"
SMOKE_CONCURRENCY="${SMOKE_CONCURRENCY:-1}"
FULL_CONCURRENCY="${FULL_CONCURRENCY:-1}"

SUBMITTER="${REPO_ROOT}/scripts/submit_functional_mag_batches_apollo3.sh"
BASE="${REPO_ROOT}/results/functional_metagenomics"
PHASE_ROOT="${BASE}/futian_mangrove_2026_phase1"

declare -A SHARD_START_LABEL=(
  [001]="1_948"
  [002]="949_1896"
  [003]="1897_2844"
)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -s "$1" ]] || die "Required file is missing or empty: $1"
}

require_executable() {
  [[ -x "$1" ]] || die "Required executable is missing: $1"
}

manifest_for_shard() {
  local shard="$1"
  printf '%s/manifests/prioritized/futian_phase1_functional_bacteria_%s_rows_%s.tsv' \
    "$PHASE_ROOT" "$shard" "${SHARD_START_LABEL[$shard]}"
}

result_root_for_shard() {
  local shard="$1"
  printf '%s/futian_mangrove_2026_phase1_bacteria_%s' "$BASE" "$shard"
}

completed_count_for_shard() {
  local shard="$1"
  local manifest result_root completed
  manifest="$(manifest_for_shard "$shard")"
  result_root="$(result_root_for_shard "$shard")"
  require_file "$manifest"
  completed=0
  while IFS=$'\t' read -r idx proteome_id; do
    if has_completed_proteome "$result_root" "$proteome_id"; then
      completed=$((completed + 1))
    fi
  done < <(included_manifest_indices "$manifest")
  printf '%s\n' "$completed"
}

pending_count_for_shard() {
  local shard="$1"
  local manifest result_root pending
  manifest="$(manifest_for_shard "$shard")"
  result_root="$(result_root_for_shard "$shard")"
  require_file "$manifest"
  pending=0
  while IFS=$'\t' read -r idx proteome_id; do
    if ! has_completed_proteome "$result_root" "$proteome_id"; then
      pending=$((pending + 1))
    fi
  done < <(included_manifest_indices "$manifest")
  printf '%s\n' "$pending"
}

included_manifest_indices() {
  local manifest="$1"
  awk -F '\t' '
    function istrue(value) {
      return value == "True" || value == "true" || value == "1" || value == "yes" || value == "Y" || value == "y"
    }
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        gsub(/\r$/, "", $i)
        col[$i] = i
      }
      next
    }
    {
      for (i = 1; i <= NF; i++) gsub(/\r$/, "", $i)
      if (istrue($(col["functional_run_include"])) &&
          $(col["analysis_unit_type"]) == "mag_bin" &&
          istrue($(col["mbag_mag_level_include"]))) {
        included += 1
        printf "%s\t%s\n", included, $(col["proteome_id"])
      }
    }
  ' "$manifest"
}

has_completed_proteome() {
  local result_root="$1"
  local proteome_id="$2"
  local run_record run_dir
  while IFS= read -r run_record; do
    run_dir="$(dirname "$(dirname "$run_record")")"
    if [[ -f "${run_dir}/COMPLETE" ]]; then
      return 0
    fi
  done < <(find "${result_root}/per_mag/${proteome_id}" -path '*/curated/run_record.json' -type f 2>/dev/null)
  return 1
}

next_index_for_shard() {
  local shard="$1"
  local manifest result_root idx proteome_id
  manifest="$(manifest_for_shard "$shard")"
  result_root="$(result_root_for_shard "$shard")"
  require_file "$manifest"
  while IFS=$'\t' read -r idx proteome_id; do
    if ! has_completed_proteome "$result_root" "$proteome_id"; then
      printf '%s\n' "$idx"
      return 0
    fi
  done < <(included_manifest_indices "$manifest")
  printf '949\n'
}

check_partition_health() {
  [[ "$REQUIRE_HEALTHY" == "1" ]] || return 0
  command -v sinfo >/dev/null 2>&1 || return 0

  local states
  states="$(sinfo -h -p "$PARTITION" -o '%T' 2>/dev/null | tr '[:upper:]' '[:lower:]' | sort -u | tr '\n' ' ')"
  [[ -n "$states" ]] || die "Could not read Slurm partition state for ${PARTITION}"
  if ! grep -Eq '(idle|mix|alloc)' <<< "$states"; then
    die "Partition ${PARTITION} is not healthy for submission; states=${states}. Wait for admin/node cleanup or set REQUIRE_HEALTHY=0 to override."
  fi
}

submit_range() {
  local shard="$1"
  local start_index="$2"
  local end_index="$3"
  local concurrency="$4"
  local time_limit="${5:-$TIME_LIMIT}"
  local manifest result_root
  manifest="$(manifest_for_shard "$shard")"
  result_root="$(result_root_for_shard "$shard")"
  require_file "$manifest"
  mkdir -p "${result_root}/logs/array"

  echo "Preparing Futian bacteria shard ${shard}: START_INDEX=${start_index} END_INDEX=${end_index} CONCURRENCY=${concurrency} PARTITION=${PARTITION} TIME_LIMIT=${time_limit} DRY_RUN=${DRY_RUN}"
  DRY_RUN="$DRY_RUN" \
    PARTITION="$PARTITION" \
    TIME_LIMIT="$time_limit" \
    THREADS="$THREADS" \
    MEM="$MEM" \
    CONCURRENCY="$concurrency" \
    START_INDEX="$start_index" \
    END_INDEX="$end_index" \
    MANIFEST="$manifest" \
    COHORT_RUN_ID="futian_mangrove_2026_phase1_bacteria_${shard}" \
    RESULT_ROOT="$result_root" \
    "$SUBMITTER"
}

print_status() {
  local shard count pending next
  for shard in 001 002 003; do
    count="$(completed_count_for_shard "$shard")"
    pending="$(pending_count_for_shard "$shard")"
    next="$(next_index_for_shard "$shard")"
    printf 'shard_%s\tcompleted=%s\tpending=%s\tnext_index=%s\n' "$shard" "$count" "$pending" "$next"
  done
}

require_executable "$SUBMITTER"

case "$ACTION" in
  status)
    print_status
    ;;
  smoke)
    check_partition_health
    [[ -n "${SHARD_START_LABEL[$SMOKE_SHARD]:-}" ]] || die "Unknown SMOKE_SHARD=${SMOKE_SHARD}; expected 001, 002, or 003"
    smoke_index="$(next_index_for_shard "$SMOKE_SHARD")"
    [[ "$smoke_index" -le 948 ]] || die "Shard ${SMOKE_SHARD} is already complete"
    print_status
    submit_range "$SMOKE_SHARD" "$smoke_index" "$smoke_index" "$SMOKE_CONCURRENCY" "$SMOKE_TIME_LIMIT"
    ;;
  full)
    check_partition_health
    print_status
    for shard in 001 002 003; do
      start_index="$(next_index_for_shard "$shard")"
      if [[ "$start_index" -le 948 ]]; then
        submit_range "$shard" "$start_index" 948 "$FULL_CONCURRENCY" "$FULL_TIME_LIMIT"
      else
        echo "Shard ${shard} already complete; skipping."
      fi
    done
    ;;
  *)
    die "Unknown ACTION=${ACTION}; expected status, smoke, or full"
    ;;
esac
