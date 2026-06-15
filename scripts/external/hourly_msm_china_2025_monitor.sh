#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/functional_metagenomics/msm_china_2025_20260615}"
HOURS="${HOURS:-8}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-3600}"
RUN_IMMEDIATE="${RUN_IMMEDIATE:-0}"
JOB_IDS="${JOB_IDS:-8804,8810,8813}"

MONITOR_ROOT="${RESULT_ROOT}/monitoring/hourly_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${MONITOR_ROOT}/snapshots"

LOCK_FILE="${RESULT_ROOT}/monitoring/hourly_msm_china_2025_monitor.lock"
mkdir -p "$(dirname "${LOCK_FILE}")"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another hourly MSM China 2025 monitor is already running: ${LOCK_FILE}" >&2
  exit 75
fi

cd "${REPO_ROOT}"

SUMMARY_TSV="${MONITOR_ROOT}/hourly_monitor_summary.tsv"
LATEST_MD="${MONITOR_ROOT}/hourly_monitor_latest.md"
RUN_LOG="${MONITOR_ROOT}/hourly_monitor.log"

printf 'check_index\tsnapshot_utc\tcomplete\tfailed\tpartial\tattempt_created\tnot_started\tcurated_manifests_present\tcompleted_runs\tobserved_local_quality_gate_pass_total\tcompleted_local_quality_gate_pass_total\tnonempty_stderr_files\tactive_msm_jobs\tpending_msm_arrays\n' > "${SUMMARY_TSV}"

echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${RUN_LOG}"
echo "repo_root=${REPO_ROOT}" | tee -a "${RUN_LOG}"
echo "result_root=${RESULT_ROOT}" | tee -a "${RUN_LOG}"
echo "hours=${HOURS}" | tee -a "${RUN_LOG}"
echo "interval_seconds=${INTERVAL_SECONDS}" | tee -a "${RUN_LOG}"
echo "run_immediate=${RUN_IMMEDIATE}" | tee -a "${RUN_LOG}"

run_check() {
  local idx="$1"
  local ts
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local stamp="${ts//[:]/}"
  stamp="${stamp//-/}"
  local snap="${MONITOR_ROOT}/snapshots/check_${idx}_${stamp}"
  mkdir -p "${snap}"

  {
    echo "--- check ${idx} ${ts} ---"
    echo "snapshot_dir=${snap}"
  } | tee -a "${RUN_LOG}"

  squeue -u "${USER}" -o "%.18i %.9P %.24j %.8T %.10M %.10l %.6D %.16R" > "${snap}/squeue.txt" || true
  sacct -j "${JOB_IDS}" --format=JobID,JobName%25,State,ExitCode,Elapsed,MaxRSS,AllocCPUS -P > "${snap}/sacct_${JOB_IDS//,/}.txt" || true
  find "${RESULT_ROOT}/per_mag" -maxdepth 4 \( -name COMPLETE -o -name FAILED \) -print | sort > "${snap}/terminal_sentinels.txt" || true
  find "${RESULT_ROOT}/logs/array" -maxdepth 1 -type f \( -name '8804_*.err' -o -name '8810_*.err' -o -name '8813_*.err' \) -size +0 -print | sort > "${snap}/nonempty_array_stderr.txt" || true

  {
    python scripts/external/summarize_msm_china_2025_functional_status.py
    python scripts/external/update_msm_china_2025_qc_from_functional_runs.py
  } > "${snap}/refresh_stdout.txt" 2>&1 || true

  cp "${RESULT_ROOT}/status/msm_china_2025_functional_status_summary.tsv" "${snap}/" 2>/dev/null || true
  cp "${RESULT_ROOT}/status/msm_china_2025_functional_status_detail.tsv" "${snap}/" 2>/dev/null || true
  cp "${RESULT_ROOT}/qc_reconciliation/msm_china_2025_qc_reconciliation_checkm2_summary.tsv" "${snap}/" 2>/dev/null || true
  cp "${RESULT_ROOT}/qc_reconciliation/msm_china_2025_qc_reconciliation_with_checkm2.tsv" "${snap}/" 2>/dev/null || true

  SNAP="${snap}" SUMMARY_TSV="${SUMMARY_TSV}" LATEST_MD="${LATEST_MD}" CHECK_INDEX="${idx}" SNAPSHOT_UTC="${ts}" python - <<'PY'
import csv
import os
from pathlib import Path

snap = Path(os.environ["SNAP"])
summary_tsv = Path(os.environ["SUMMARY_TSV"])
latest_md = Path(os.environ["LATEST_MD"])
idx = os.environ["CHECK_INDEX"]
ts = os.environ["SNAPSHOT_UTC"]

def read_metric_table(path):
    out = {}
    if not path.exists():
        return out
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            metric = row.get("metric")
            if metric and metric not in out:
                out[metric] = row.get("value", "")
    return out

status = read_metric_table(snap / "msm_china_2025_functional_status_summary.tsv")
qc = read_metric_table(snap / "msm_china_2025_qc_reconciliation_checkm2_summary.tsv")
squeue = (snap / "squeue.txt").read_text().splitlines() if (snap / "squeue.txt").exists() else []
stderr = [line for line in (snap / "nonempty_array_stderr.txt").read_text().splitlines() if line.strip()] if (snap / "nonempty_array_stderr.txt").exists() else []

active_msm = [
    line for line in squeue
    if any(job in line for job in ("8804", "8810", "8813")) and " RUNNING " in f" {line} "
]
pending_msm = [
    line for line in squeue
    if any(job in line for job in ("8804", "8810", "8813")) and " PENDING " in f" {line} "
]

row = [
    idx,
    ts,
    status.get("complete", ""),
    status.get("failed", ""),
    status.get("partial", ""),
    status.get("attempt_created", ""),
    status.get("not_started", ""),
    status.get("curated_manifests_present", ""),
    qc.get("completed_runs", ""),
    qc.get("observed_local_quality_gate_pass_total", ""),
    qc.get("completed_local_quality_gate_pass_total", ""),
    str(len(stderr)),
    str(len(active_msm)),
    str(len(pending_msm)),
]
with summary_tsv.open("a") as handle:
    handle.write("\t".join(row) + "\n")

latest_md.write_text(
    "\n".join([
        f"# MSM China 2025 Hourly Monitor Latest",
        "",
        f"- check_index: {idx}",
        f"- snapshot_utc: {ts}",
        f"- complete: {status.get('complete', '')}",
        f"- failed: {status.get('failed', '')}",
        f"- partial: {status.get('partial', '')}",
        f"- not_started: {status.get('not_started', '')}",
        f"- curated_manifests_present: {status.get('curated_manifests_present', '')}",
        f"- completed_local_quality_gate_pass_total: {qc.get('completed_local_quality_gate_pass_total', '')}",
        f"- observed_local_quality_gate_pass_total: {qc.get('observed_local_quality_gate_pass_total', '')}",
        f"- nonempty_stderr_files: {len(stderr)}",
        f"- active_msm_jobs: {len(active_msm)}",
        f"- pending_msm_arrays: {len(pending_msm)}",
        f"- snapshot_dir: {snap}",
        "",
        "Claim boundary: counts are operational MAG/proteome annotation status only; no sample-level methane-risk or MRV tier claim is implied.",
        "",
    ])
)
PY
}

for idx in $(seq 1 "${HOURS}"); do
  if [[ "${RUN_IMMEDIATE}" != "1" || "${idx}" != "1" ]]; then
    sleep "${INTERVAL_SECONDS}"
  fi
  run_check "${idx}"
done

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${RUN_LOG}"
