#!/usr/bin/env bash
#SBATCH --job-name=bc-fg-batch-worker
#SBATCH --output=%x.%A_%a.out
#SBATCH --error=%x.%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=accel

set -euo pipefail

MROOT="${MROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
FG_RUN_ID="${FG_RUN_ID:?FG_RUN_ID is required}"
FG_ART_DIR="${FG_ART_DIR:-$MROOT/results/blue_catalyst_poc/runs/$FG_RUN_ID/fg_artifacts}"
HMM_DIR="${HMM_DIR:-$MROOT/data/hmm}"
BATCH_THREADS="${BATCH_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
EVALUE_THRESHOLD="${EVALUE_THRESHOLD:-1e-10}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-50.0}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "ERROR: this script is intended for sbatch --array execution" >&2
  exit 1
fi

BATCH_NAME="$(printf 'batch_%04d' "${SLURM_ARRAY_TASK_ID}")"
BATCH_MANIFEST="$FG_ART_DIR/batches/${BATCH_NAME}.tsv"
BATCH_OUT_DIR="$FG_ART_DIR/batch_results/${BATCH_NAME}"
mkdir -p "$BATCH_OUT_DIR"

[[ -f "$BATCH_MANIFEST" ]] || {
  echo "ERROR: missing batch manifest $BATCH_MANIFEST" >&2
  exit 1
}

if [[ -n "${MODULESHOME:-}" ]] || command -v module >/dev/null 2>&1; then
  module load miniconda3/25.5.1 || true
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
export PYTHONPATH="$MROOT/src:${PYTHONPATH:-}"

if ! conda run -n methanet-fgintel python - <<'PY'
import importlib.util
import sys

required = ["numpy", "pandas"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    sys.stderr.write("ERROR: missing python deps in methanet-fgintel env: " + ", ".join(missing) + "\n")
    sys.exit(1)
PY
then
  echo "Install once and retry:" >&2
  echo "  conda activate methanet-fgintel" >&2
  echo "  conda install -y -c conda-forge numpy pandas" >&2
  exit 1
fi

conda run -n methanet-fgintel hmmsearch -h >/dev/null
[[ -d "$HMM_DIR" ]] || { echo "ERROR: missing HMM dir $HMM_DIR" >&2; exit 1; }

if ! conda run -n methanet-fgintel python - <<'PY'
import os
from pathlib import Path
import sys

hmm_dir = Path(os.environ["HMM_DIR"])
required = [
    "mcrA.hmm", "mcrB.hmm", "mcrG.hmm",
    "pmoA.hmm", "mmoX.hmm",
    "dsrA.hmm", "dsrB.hmm",
    "nifH.hmm", "cbbL.hmm",
    "mtaB.hmm", "mttB.hmm", "mtbA.hmm",
]

missing = [str(hmm_dir / name) for name in required if not (hmm_dir / name).exists()]
if missing:
    sys.stderr.write(
        "ERROR: missing required HMM marker files for worker node: "
        + ", ".join(missing[:6])
        + (" ..." if len(missing) > 6 else "")
        + "\n"
    )
    sys.exit(1)
PY
then
  echo "Rebuild HMM resources and retry:" >&2
  echo "  bash workflow/scripts/setup_hmm_resources.sh" >&2
  exit 1
fi

conda run -n methanet-fgintel python "$MROOT/scripts/blue_catalyst_fg_batch_pipeline.py" process-batch \
  --batch-manifest "$BATCH_MANIFEST" \
  --hmm-dir "$HMM_DIR" \
  --output-features "$BATCH_OUT_DIR/fg_features.tsv" \
  --output-failures "$BATCH_OUT_DIR/fg_failures.tsv" \
  --threads "$BATCH_THREADS" \
  --evalue-threshold "$EVALUE_THRESHOLD" \
  --score-threshold "$SCORE_THRESHOLD"

echo "[OK] Completed ${BATCH_NAME}"
