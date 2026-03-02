#!/usr/bin/env bash
set -euo pipefail

# Parallel rumen proteome prep (missing-only) + SLURM relaunch with same BC_RUN_ID.
#
# Usage:
#   RUN_ID=apolo_full_20260228_080644 \
#   OLD_JOB_ID=1497 \
#   WORKERS=24 \
#   /home/rsg-jcorre38/Jay_Proyects/MethaNet/scripts/apolo_parallel_resume_and_relaunch.sh
#
# Required env:
#   RUN_ID              Existing run id to reuse (same subset/artifacts path)
# Optional env:
#   MROOT               Project root (default: /home/rsg-jcorre38/Jay_Proyects/MethaNet)
#   WORKERS             Parallel workers for preprocessing (default: 24)
#   OLD_JOB_ID          If provided, this script will cancel it before relaunch
#   SKIP_PREP           Set to 1 to skip parallel preprocess stage

MROOT="${MROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
RUN_ID="${RUN_ID:-}"
WORKERS="${WORKERS:-24}"
OLD_JOB_ID="${OLD_JOB_ID:-}"
SKIP_PREP="${SKIP_PREP:-0}"

if [[ -z "$RUN_ID" ]]; then
  echo "ERROR: RUN_ID is required (e.g., apolo_full_20260228_080644)." >&2
  exit 1
fi

ART="$MROOT/results/blue_catalyst_poc/runs/$RUN_ID/artifacts"
RAW="$MROOT/data/blue_catalyst_poc/rumen/raw"
PROT="$MROOT/data/blue_catalyst_poc/proteomes"
SUBSET="$ART/prjeb31266_selected_subset.tsv"
JOB_SCRIPT="$MROOT/scripts/submit_blue_catalyst_poc_apolo3.sh"

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "ERROR: job script not found: $JOB_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$SUBSET" ]]; then
  echo "ERROR: subset file not found: $SUBSET" >&2
  exit 1
fi
if ! command -v prodigal >/dev/null 2>&1; then
  echo "ERROR: prodigal is not on PATH." >&2
  exit 1
fi
if ! command -v python >/dev/null 2>&1; then
  echo "ERROR: python is not on PATH." >&2
  exit 1
fi

mkdir -p "$PROT"

echo "[INFO] MROOT=$MROOT"
echo "[INFO] RUN_ID=$RUN_ID"
echo "[INFO] ART=$ART"
echo "[INFO] WORKERS=$WORKERS"

if [[ -n "$OLD_JOB_ID" ]]; then
  echo "[INFO] Cancelling old job: $OLD_JOB_ID"
  scancel "$OLD_JOB_ID" || true
fi

if [[ "$SKIP_PREP" != "1" ]]; then
  echo "[INFO] Starting parallel missing-only rumen proteome preparation..."

  MROOT="$MROOT" RUN_ID="$RUN_ID" WORKERS="$WORKERS" python - <<'PY'
import concurrent.futures as cf
import gzip
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

MROOT = Path(__import__("os").environ["MROOT"])
RUN_ID = __import__("os").environ["RUN_ID"]
WORKERS = int(__import__("os").environ["WORKERS"])

ART = MROOT / "results" / "blue_catalyst_poc" / "runs" / RUN_ID / "artifacts"
RAW = MROOT / "data" / "blue_catalyst_poc" / "rumen" / "raw"
PROT = MROOT / "data" / "blue_catalyst_poc" / "proteomes"
TMP = PROT / "_tmp_rumen"
TMP.mkdir(parents=True, exist_ok=True)

subset = pd.read_csv(ART / "prjeb31266_selected_subset.tsv", sep="\t")
filenames = [f for f in subset["filename"].astype(str).tolist() if f]


dna_chars = set("ACGTNacgtn")
aa_chars = set("ACDEFGHIKLMNPQRSTVWYBXZJUO*acdefghiklmnpqrstvwybxzjuo")


def sanitize(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x))


def is_probably_protein(seq: str) -> bool:
    if not seq:
        return False
    dna_fraction = sum(ch in dna_chars for ch in seq) / max(1, len(seq))
    aa_fraction = sum(ch in aa_chars for ch in seq) / max(1, len(seq))
    return aa_fraction > 0.95 and dna_fraction < 0.9


def preview(fp: Path, n_records: int = 20):
    checked = 0
    aa_like = 0
    seq = []
    with gzip.open(fp, "rt", errors="ignore") as h:
        for line in h:
            if line.startswith(">"):
                if seq:
                    s = "".join(seq).strip()
                    if is_probably_protein(s):
                        aa_like += 1
                    checked += 1
                    if checked >= n_records:
                        break
                    seq = []
            else:
                seq.append(line.strip())
        if checked < n_records and seq:
            s = "".join(seq).strip()
            if is_probably_protein(s):
                aa_like += 1
            checked += 1
    return checked, aa_like


def process(filename: str):
    fp = RAW / filename
    if not fp.exists():
        return filename, "missing_raw", "raw file not found"

    sample = sanitize(fp.stem.replace(".fa", ""))
    out_fp = PROT / f"rumen__{sample}.faa"

    # Critical: preserve previous work and skip reruns.
    if out_fp.exists() and out_fp.stat().st_size > 0:
        return filename, "reused", ""

    try:
        checked, aa_like = preview(fp)
    except Exception as e:
        return filename, "preview_error", str(e)

    if checked > 0 and aa_like >= max(1, int(0.7 * checked)):
        try:
            with gzip.open(fp, "rb") as src, out_fp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            return filename, "copied", ""
        except Exception as e:
            out_fp.unlink(missing_ok=True)
            return filename, "copy_error", str(e)

    nuc = TMP / fp.with_suffix("").name
    if not nuc.exists():
        try:
            with gzip.open(fp, "rb") as src, nuc.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        except Exception as e:
            nuc.unlink(missing_ok=True)
            return filename, "decompress_error", str(e)

    cmd = ["prodigal", "-i", str(nuc), "-a", str(out_fp), "-p", "meta", "-q"]
    rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    if rc != 0 or (not out_fp.exists()) or out_fp.stat().st_size == 0:
        out_fp.unlink(missing_ok=True)
        return filename, "prodigal_failed", f"returncode={rc}"
    return filename, "prodigal", ""


counts = Counter()
fail_rows = []

with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = [ex.submit(process, fn) for fn in filenames]
    for i, fut in enumerate(cf.as_completed(futures), start=1):
        filename, status, detail = fut.result()
        counts[status] += 1
        if detail and status in {
            "missing_raw",
            "preview_error",
            "copy_error",
            "decompress_error",
            "prodigal_failed",
        }:
            fail_rows.append({"filename": filename, "status": status, "detail": detail})
            print(f"[prep][warn] {status} :: {filename} :: {detail}")
        if i % 25 == 0 or i == len(filenames):
            print(f"[prep] {i}/{len(filenames)} :: {dict(counts)}")

if fail_rows:
    fail_df = pd.DataFrame(fail_rows)
    fail_path = ART / "parallel_prep_failures.tsv"
    fail_df.to_csv(fail_path, sep="\t", index=False)
    print(f"[prep] wrote failure report: {fail_path} (n={len(fail_rows)})")

print("[prep] FINAL:", dict(counts))
PY
else
  echo "[INFO] SKIP_PREP=1, skipping parallel preprocess."
fi

echo "[INFO] Relaunching notebook job with same RUN_ID to reuse existing artifacts/subset..."
NEW_JOB_ID=$(sbatch --parsable --export=ALL,BC_RUN_ID="$RUN_ID" "$JOB_SCRIPT")
echo "[INFO] Submitted new job: $NEW_JOB_ID"

echo
echo "Monitor:"
echo "watch -n 30 'squeue -j $NEW_JOB_ID; echo; sacct -j $NEW_JOB_ID --format=JobID,State,ExitCode,Elapsed,Reason%50'"
echo
echo "Validate on completion:"
echo "python $MROOT/scripts/validate_blue_catalyst_artifacts.py --artifacts-dir $ART"
