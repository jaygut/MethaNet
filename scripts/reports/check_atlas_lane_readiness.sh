#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/reports}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
LANE_REGISTRY="${LANE_REGISTRY:-${REPO_ROOT}/configs/methanet_atlas_lanes.tsv}"
REFRESHER="${REFRESHER:-${REPO_ROOT}/scripts/reports/refresh_atlas_lane_registry_status.sh}"
CONSOLIDATION_GATE="${CONSOLIDATION_GATE:-${REPO_ROOT}/scripts/reports/check_atlas_lane_consolidation_gate.py}"
REPORT_GATE="${REPORT_GATE:-${REPO_ROOT}/scripts/reports/check_atlas_report_rebuild_gate.py}"
OVERLAP_AUDIT="${OVERLAP_AUDIT:-${REPO_ROOT}/scripts/reports/audit_atlas_lane_overlap.py}"
STATUS_COMPARE="${STATUS_COMPARE:-${REPO_ROOT}/scripts/reports/compare_atlas_lane_status.py}"
COMPLETION_CHECKLIST="${COMPLETION_CHECKLIST:-${REPO_ROOT}/scripts/reports/build_atlas_lane_completion_checklist.py}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
READINESS_PREFIX="${READINESS_PREFIX:-atlas_lane_readiness}"
READINESS_SUMMARY="${READINESS_SUMMARY:-${OUTPUT_DIR}/${READINESS_PREFIX}_${STAMP}.md}"
READINESS_JSON="${READINESS_JSON:-${OUTPUT_DIR}/${READINESS_PREFIX}_${STAMP}.json}"
OVERLAP_PREFIX="${OVERLAP_PREFIX:-atlas_lane_overlap}"
STATUS_DELTA_PREFIX="${STATUS_DELTA_PREFIX:-atlas_lane_registry_delta}"
CHECKLIST_PREFIX="${CHECKLIST_PREFIX:-atlas_lane_completion_checklist}"
PREVIOUS_STATUS_JSON="${PREVIOUS_STATUS_JSON:-}"
LANE_IDS="${LANE_IDS:-}"
RUN_OVERLAP_AUDIT="${RUN_OVERLAP_AUDIT:-auto}"
INCLUDE_SLURM="${INCLUDE_SLURM:-auto}"
SQUEUE="${SQUEUE:-squeue}"
SLURM_USER="${SLURM_USER:-${USER:-}}"
STRICT_GATES="${STRICT_GATES:-1}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

split_lanes() {
  local raw="${1//,/ }"
  raw="${raw//;/ }"
  for lane_id in $raw; do
    [[ -n "$lane_id" ]] && printf '%s\n' "$lane_id"
  done
}

run_gate() {
  shift
  local output
  local rc=0
  if output="$("$@" 2>&1)"; then
    rc=0
  else
    rc=$?
  fi
  printf '%s\n' "$output"
  return "$rc"
}

[[ -s "$LANE_REGISTRY" ]] || die "Lane registry missing or empty: $LANE_REGISTRY"
[[ -x "$REFRESHER" ]] || die "Registry refresher missing or not executable: $REFRESHER"
[[ -s "$CONSOLIDATION_GATE" ]] || die "Consolidation gate missing: $CONSOLIDATION_GATE"
[[ -s "$REPORT_GATE" ]] || die "Report gate missing: $REPORT_GATE"
[[ -s "$OVERLAP_AUDIT" ]] || die "Overlap audit missing: $OVERLAP_AUDIT"
[[ -s "$COMPLETION_CHECKLIST" ]] || die "Completion checklist tool missing: $COMPLETION_CHECKLIST"
[[ -x "$PYTHON" ]] || die "Python executable missing or not executable: $PYTHON"

mkdir -p "$OUTPUT_DIR"

if [[ "$PREVIOUS_STATUS_JSON" == "auto" ]]; then
  previous_status_auto="$(
    find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'atlas_lane_registry_status_*.json' ! -name '*.validation.json' -printf '%T@ %p\n' \
      | sort -nr \
      | awk 'NR == 1 {sub(/^[^ ]+ /, ""); print; exit}'
  )"
  if [[ -n "$previous_status_auto" ]]; then
    PREVIOUS_STATUS_JSON="$previous_status_auto"
    echo "Using latest previous status JSON: $PREVIOUS_STATUS_JSON"
  else
    PREVIOUS_STATUS_JSON=""
    echo "PREVIOUS_STATUS_JSON=auto requested, but no previous status JSON was found; status delta will be skipped."
  fi
fi

[[ -z "$PREVIOUS_STATUS_JSON" || -s "$STATUS_COMPARE" ]] || die "Status comparison tool missing: $STATUS_COMPARE"
[[ -z "$PREVIOUS_STATUS_JSON" || -s "$PREVIOUS_STATUS_JSON" ]] || die "Previous status JSON missing or empty: $PREVIOUS_STATUS_JSON"

refresh_log="$(mktemp "${TMPDIR:-/tmp}/methanet_lane_refresh.XXXXXX")"
trap 'rm -f "$refresh_log"' EXIT

echo "Refreshing atlas lane registry status..."
REPO_ROOT="$REPO_ROOT" \
LANE_REGISTRY="$LANE_REGISTRY" \
OUTPUT_DIR="$OUTPUT_DIR" \
STAMP="$STAMP" \
PYTHON="$PYTHON" \
"$REFRESHER" | tee "$refresh_log"

status_json="$(awk -F '\t' '$1 == "status_json" {print $2}' "$refresh_log" | tail -1)"
status_md="$(awk -F '\t' '$1 == "status_md" {print $2}' "$refresh_log" | tail -1)"
validation_json="$(awk -F '\t' '$1 == "validation_json" {print $2}' "$refresh_log" | tail -1)"
[[ -s "$status_json" ]] || die "Refresher did not produce a readable status JSON"

lane_args=()
selected_lanes=()
while IFS= read -r lane_id; do
  selected_lanes+=("$lane_id")
  lane_args+=(--lane-id "$lane_id")
done < <(split_lanes "$LANE_IDS")

if ((${#selected_lanes[@]})); then
  lane_selection="${selected_lanes[*]}"
else
  lane_selection="all eligible registered lanes"
fi

echo "Running consolidation readiness gate for ${lane_selection}..."
set +e
consolidation_output="$(
  run_gate \
    "consolidation" \
    "$PYTHON" "$CONSOLIDATION_GATE" \
    --status-json "$status_json" \
    "${lane_args[@]}" \
    --print-commands
)"
consolidation_rc=$?
set -e

echo "Running expanded-atlas report readiness gate for ${lane_selection}..."
set +e
report_output="$(
  run_gate \
    "report" \
    "$PYTHON" "$REPORT_GATE" \
    --status-json "$status_json" \
    --lane-registry "$LANE_REGISTRY" \
    "${lane_args[@]}" \
    --print-command
)"
report_rc=$?
set -e

overlap_rc=0
overlap_output="SKIPPED: overlap audit requires at least two selected lanes or RUN_OVERLAP_AUDIT=1."
overlap_summary_tsv=""
overlap_matches_tsv=""
overlap_json=""
overlap_md=""
run_overlap="0"
if [[ "$RUN_OVERLAP_AUDIT" == "1" ]]; then
  run_overlap="1"
elif [[ "$RUN_OVERLAP_AUDIT" == "auto" && ${#selected_lanes[@]} -ge 2 ]]; then
  run_overlap="1"
fi

if [[ "$run_overlap" == "1" ]]; then
  overlap_base="${OUTPUT_DIR}/${OVERLAP_PREFIX}_${STAMP}"
  overlap_summary_tsv="${overlap_base}.summary.tsv"
  overlap_matches_tsv="${overlap_base}.matches.tsv"
  overlap_json="${overlap_base}.json"
  overlap_md="${overlap_base}.md"
  echo "Running exact manifest-overlap audit for ${lane_selection}..."
  set +e
  overlap_output="$(
    run_gate \
      "overlap" \
      "$PYTHON" "$OVERLAP_AUDIT" \
      --repo-root "$REPO_ROOT" \
      --lane-registry "$LANE_REGISTRY" \
      "${lane_args[@]}" \
      --output-summary-tsv "$overlap_summary_tsv" \
      --output-matches-tsv "$overlap_matches_tsv" \
      --output-json "$overlap_json" \
      --output-md "$overlap_md"
  )"
  overlap_rc=$?
  set -e
  if [[ "$overlap_rc" == "0" && -z "$overlap_output" ]]; then
    overlap_output="PASS: exact manifest-overlap audit completed. See ${overlap_md}"
  fi
fi

slurm_output="SKIPPED: Slurm queue snapshot disabled or squeue unavailable."
run_slurm="0"
if [[ "$INCLUDE_SLURM" == "1" ]]; then
  run_slurm="1"
elif [[ "$INCLUDE_SLURM" == "auto" ]] && command -v "$SQUEUE" >/dev/null 2>&1; then
  run_slurm="1"
fi
if [[ "$run_slurm" == "1" ]]; then
  echo "Capturing Slurm queue snapshot..."
  set +e
  if [[ -n "$SLURM_USER" ]]; then
    slurm_output="$("$SQUEUE" -u "$SLURM_USER" --format='%.18i %.32j %.10T %.10M %.9l %.6D %R' 2>&1 | head -200)"
  else
    slurm_output="$("$SQUEUE" --format='%.18i %.32j %.10T %.10M %.9l %.6D %R' 2>&1 | head -200)"
  fi
  slurm_rc=$?
  set -e
  if [[ "$slurm_rc" != "0" ]]; then
    slurm_output="SKIPPED: squeue returned ${slurm_rc}: ${slurm_output}"
  fi
fi

status_delta_rc=0
status_delta_output="SKIPPED: set PREVIOUS_STATUS_JSON to compare this status snapshot to an earlier registry status JSON."
status_delta_tsv=""
status_delta_json=""
status_delta_md=""
run_status_delta="0"
if [[ -n "$PREVIOUS_STATUS_JSON" ]]; then
  run_status_delta="1"
  status_delta_base="${OUTPUT_DIR}/${STATUS_DELTA_PREFIX}_${STAMP}"
  status_delta_tsv="${status_delta_base}.tsv"
  status_delta_json="${status_delta_base}.json"
  status_delta_md="${status_delta_base}.md"
  echo "Comparing atlas lane status snapshots..."
  set +e
  status_delta_output="$(
    run_gate \
      "status-delta" \
      "$PYTHON" "$STATUS_COMPARE" \
      --previous-json "$PREVIOUS_STATUS_JSON" \
      --current-json "$status_json" \
      --output-tsv "$status_delta_tsv" \
      --output-json "$status_delta_json" \
      --output-md "$status_delta_md"
  )"
  status_delta_rc=$?
  set -e
  if [[ "$status_delta_rc" == "0" && -z "$status_delta_output" ]]; then
    status_delta_output="PASS: status delta generated. See ${status_delta_md}"
  fi
fi

checklist_rc=0
checklist_output="SKIPPED: completion checklist generation did not run."
checklist_json="${OUTPUT_DIR}/${CHECKLIST_PREFIX}_${STAMP}.json"
checklist_md="${OUTPUT_DIR}/${CHECKLIST_PREFIX}_${STAMP}.md"
echo "Building atlas lane completion checklist..."
set +e
checklist_output="$(
  run_gate \
    "completion-checklist" \
    "$PYTHON" "$COMPLETION_CHECKLIST" \
    --status-json "$status_json" \
    "${lane_args[@]}" \
    --output-json "$checklist_json" \
    --output-md "$checklist_md"
)"
checklist_rc=$?
set -e
if [[ "$checklist_rc" == "0" && -z "$checklist_output" ]]; then
  checklist_output="PASS: completion checklist generated. See ${checklist_md}"
fi

{
  printf '# MethaNet Atlas Lane Readiness Check\n\n'
  printf 'Generated UTC: `%s`\n\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Lane selection: `%s`\n\n' "$lane_selection"
  printf '| Artifact | Path |\n'
  printf '| --- | --- |\n'
  printf '| Status JSON | `%s` |\n' "$status_json"
  printf '| Status Markdown | `%s` |\n' "$status_md"
  printf '| Registry validation JSON | `%s` |\n' "$validation_json"
  printf '| Readiness JSON | `%s` |\n' "$READINESS_JSON"
  printf '| Completion checklist Markdown | `%s` |\n' "$checklist_md"
  printf '| Completion checklist JSON | `%s` |\n' "$checklist_json"
  if [[ -n "$status_delta_md" ]]; then
    printf '| Status delta Markdown | `%s` |\n' "$status_delta_md"
    printf '| Status delta JSON | `%s` |\n' "$status_delta_json"
    printf '| Status delta TSV | `%s` |\n' "$status_delta_tsv"
  fi
  if [[ -n "$overlap_md" ]]; then
    printf '| Overlap audit Markdown | `%s` |\n' "$overlap_md"
    printf '| Overlap audit JSON | `%s` |\n' "$overlap_json"
    printf '| Overlap audit summary TSV | `%s` |\n' "$overlap_summary_tsv"
    printf '| Overlap audit matches TSV | `%s` |\n' "$overlap_matches_tsv"
  fi
  printf '\n'
  printf '| Gate | Result |\n'
  printf '| --- | --- |\n'
  if [[ "$consolidation_rc" == "0" ]]; then
    printf '| Consolidation | PASS |\n'
  else
    printf '| Consolidation | BLOCKED |\n'
  fi
  if [[ "$report_rc" == "0" ]]; then
    printf '| Expanded atlas report | PASS |\n'
  else
    printf '| Expanded atlas report | BLOCKED |\n'
  fi
  if [[ "$run_overlap" == "1" && "$overlap_rc" == "0" ]]; then
    printf '| Exact manifest overlap audit | PASS |\n'
  elif [[ "$run_overlap" == "1" ]]; then
    printf '| Exact manifest overlap audit | BLOCKED |\n'
  else
    printf '| Exact manifest overlap audit | SKIPPED |\n'
  fi
  printf '\n'
  printf '## Consolidation Gate\n\n'
  printf '```text\n%s\n```\n\n' "$consolidation_output"
  printf '## Expanded Atlas Report Gate\n\n'
  printf '```text\n%s\n```\n\n' "$report_output"
  printf '## Exact Manifest Overlap Audit\n\n'
  printf '```text\n%s\n```\n\n' "$overlap_output"
  printf '## Slurm Queue Snapshot\n\n'
  printf '```text\n%s\n```\n\n' "$slurm_output"
  printf '## Status Delta\n\n'
  printf '```text\n%s\n```\n\n' "$status_delta_output"
  printf '## Completion Checklist\n\n'
  printf '```text\n%s\n```\n\n' "$checklist_output"
  printf '## Claim Boundary\n\n'
  printf 'This readiness check is operational evidence only. It does not assign final sample methane-risk tiers, measured methane flux, carbon-credit approval, or calibrated MRV scoring.\n'
} > "$READINESS_SUMMARY"

READINESS_JSON="$READINESS_JSON" \
GENERATED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
LANE_SELECTION="$lane_selection" \
STATUS_JSON="$status_json" \
STATUS_MD="$status_md" \
VALIDATION_JSON="$validation_json" \
OVERLAP_SUMMARY_TSV="$overlap_summary_tsv" \
OVERLAP_MATCHES_TSV="$overlap_matches_tsv" \
OVERLAP_JSON="$overlap_json" \
OVERLAP_MD="$overlap_md" \
STATUS_DELTA_TSV="$status_delta_tsv" \
STATUS_DELTA_JSON="$status_delta_json" \
STATUS_DELTA_MD="$status_delta_md" \
PREVIOUS_STATUS_JSON="$PREVIOUS_STATUS_JSON" \
CHECKLIST_JSON="$checklist_json" \
CHECKLIST_MD="$checklist_md" \
READINESS_SUMMARY="$READINESS_SUMMARY" \
CONSOLIDATION_RC="$consolidation_rc" \
REPORT_RC="$report_rc" \
OVERLAP_RC="$overlap_rc" \
STATUS_DELTA_RC="$status_delta_rc" \
CHECKLIST_RC="$checklist_rc" \
RUN_OVERLAP="$run_overlap" \
RUN_STATUS_DELTA="$run_status_delta" \
RUN_SLURM="$run_slurm" \
CONSOLIDATION_OUTPUT="$consolidation_output" \
REPORT_OUTPUT="$report_output" \
OVERLAP_OUTPUT="$overlap_output" \
STATUS_DELTA_OUTPUT="$status_delta_output" \
CHECKLIST_OUTPUT="$checklist_output" \
SLURM_OUTPUT="$slurm_output" \
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["READINESS_JSON"])
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "generated_utc": os.environ["GENERATED_UTC"],
    "lane_selection": os.environ["LANE_SELECTION"].split(),
    "artifacts": {
        "readiness_markdown": os.environ["READINESS_SUMMARY"],
        "readiness_json": os.environ["READINESS_JSON"],
        "status_json": os.environ["STATUS_JSON"],
        "status_markdown": os.environ["STATUS_MD"],
        "registry_validation_json": os.environ["VALIDATION_JSON"],
        "overlap_summary_tsv": os.environ["OVERLAP_SUMMARY_TSV"],
        "overlap_matches_tsv": os.environ["OVERLAP_MATCHES_TSV"],
        "overlap_json": os.environ["OVERLAP_JSON"],
        "overlap_markdown": os.environ["OVERLAP_MD"],
        "status_delta_tsv": os.environ["STATUS_DELTA_TSV"],
        "status_delta_json": os.environ["STATUS_DELTA_JSON"],
        "status_delta_markdown": os.environ["STATUS_DELTA_MD"],
        "previous_status_json": os.environ["PREVIOUS_STATUS_JSON"],
        "completion_checklist_json": os.environ["CHECKLIST_JSON"],
        "completion_checklist_markdown": os.environ["CHECKLIST_MD"],
    },
    "gates": {
        "consolidation": {
            "result": "pass" if os.environ["CONSOLIDATION_RC"] == "0" else "blocked",
            "return_code": int(os.environ["CONSOLIDATION_RC"]),
            "output": os.environ["CONSOLIDATION_OUTPUT"],
        },
        "expanded_atlas_report": {
            "result": "pass" if os.environ["REPORT_RC"] == "0" else "blocked",
            "return_code": int(os.environ["REPORT_RC"]),
            "output": os.environ["REPORT_OUTPUT"],
        },
        "exact_manifest_overlap": {
            "result": (
                "skipped"
                if os.environ["RUN_OVERLAP"] != "1"
                else "pass"
                if os.environ["OVERLAP_RC"] == "0"
                else "blocked"
            ),
            "return_code": int(os.environ["OVERLAP_RC"]),
            "output": os.environ["OVERLAP_OUTPUT"],
        },
    },
    "status_delta": {
        "result": (
            "skipped"
            if os.environ["RUN_STATUS_DELTA"] != "1"
            else "pass"
            if os.environ["STATUS_DELTA_RC"] == "0"
            else "blocked"
        ),
        "return_code": int(os.environ["STATUS_DELTA_RC"]),
        "output": os.environ["STATUS_DELTA_OUTPUT"],
    },
    "completion_checklist": {
        "result": "pass" if os.environ["CHECKLIST_RC"] == "0" else "blocked",
        "return_code": int(os.environ["CHECKLIST_RC"]),
        "output": os.environ["CHECKLIST_OUTPUT"],
    },
    "slurm": {
        "captured": os.environ["RUN_SLURM"] == "1",
        "output": os.environ["SLURM_OUTPUT"],
    },
    "claim_boundary": (
        "Operational evidence only; no final sample methane-risk tiers, measured methane flux, "
        "carbon-credit approval, or calibrated MRV scoring."
    ),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True))
PY

printf 'readiness_summary\t%s\n' "$READINESS_SUMMARY"
printf 'readiness_json\t%s\n' "$READINESS_JSON"
if [[ -n "$overlap_md" ]]; then
  printf 'overlap_summary_tsv\t%s\n' "$overlap_summary_tsv"
  printf 'overlap_matches_tsv\t%s\n' "$overlap_matches_tsv"
  printf 'overlap_json\t%s\n' "$overlap_json"
  printf 'overlap_md\t%s\n' "$overlap_md"
fi
if [[ -n "$status_delta_md" ]]; then
  printf 'status_delta_tsv\t%s\n' "$status_delta_tsv"
  printf 'status_delta_json\t%s\n' "$status_delta_json"
  printf 'status_delta_md\t%s\n' "$status_delta_md"
fi
printf 'completion_checklist_json\t%s\n' "$checklist_json"
printf 'completion_checklist_md\t%s\n' "$checklist_md"
printf 'consolidation_gate_rc\t%s\n' "$consolidation_rc"
printf 'report_gate_rc\t%s\n' "$report_rc"
printf 'overlap_audit_rc\t%s\n' "$overlap_rc"
printf 'status_delta_rc\t%s\n' "$status_delta_rc"
printf 'completion_checklist_rc\t%s\n' "$checklist_rc"

if [[ "$STRICT_GATES" == "1" && ( "$consolidation_rc" != "0" || "$report_rc" != "0" || "$overlap_rc" != "0" || "$status_delta_rc" != "0" || "$checklist_rc" != "0" ) ]]; then
  exit 1
fi
exit 0
