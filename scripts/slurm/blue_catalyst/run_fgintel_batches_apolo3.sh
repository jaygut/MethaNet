#!/usr/bin/env bash
#SBATCH --job-name=bc-fg-batches
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=accel

set -euo pipefail

MROOT="${MROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
RUNS_ROOT="${RUNS_ROOT:-$MROOT/results/blue_catalyst_poc/runs}"
SLURM_PARTITION="${SLURM_PARTITION:-accel}"
DB_ROOT="${DB_ROOT:-$HOME/scratch/methanet_db}"
HMM_DIR="${HMM_DIR:-$MROOT/data/hmm}"
FG_SOURCE_EMBED_RUN_ID="${FG_SOURCE_EMBED_RUN_ID:-}"
FG_RUN_ID="${FG_RUN_ID:-apolo_fgintel_$(date +%Y%m%d_%H%M%S)}"
FG_BATCH_SIZE="${FG_BATCH_SIZE:-25}"
FG_MIN_JOIN_COVERAGE="${FG_MIN_JOIN_COVERAGE:-0.95}"
FG_HASH_PROTEOMES="${FG_HASH_PROTEOMES:-0}"
BATCH_THREADS="${BATCH_THREADS:-4}"
WORKER_TIME_LIMIT="${WORKER_TIME_LIMIT:-04:00:00}"
WORKER_MEM="${WORKER_MEM:-24G}"
WORKER_CPUS="${WORKER_CPUS:-4}"

if [[ -z "$FG_SOURCE_EMBED_RUN_ID" ]]; then
  echo "ERROR: FG_SOURCE_EMBED_RUN_ID is required" >&2
  exit 1
fi

EMBED_ART_DIR="$RUNS_ROOT/$FG_SOURCE_EMBED_RUN_ID/artifacts"
EMBED_META="$EMBED_ART_DIR/embedding_metadata.tsv"
EMBED_NPZ="$EMBED_ART_DIR/genome_embeddings.npz"
[[ -f "$EMBED_META" ]] || { echo "ERROR: missing $EMBED_META" >&2; exit 1; }
[[ -f "$EMBED_NPZ" ]] || { echo "ERROR: missing $EMBED_NPZ" >&2; exit 1; }

FG_RUN_ROOT="$RUNS_ROOT/$FG_RUN_ID"
FG_ART_DIR="$FG_RUN_ROOT/fg_artifacts"
FG_PLAN_DIR="$FG_ART_DIR"
mkdir -p "$FG_ART_DIR" "$FG_ART_DIR/batch_results"

if [[ -n "${MODULESHOME:-}" ]] || command -v module >/dev/null 2>&1; then
  module load miniconda3/25.5.1 || true
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
export PYTHONPATH="$MROOT/src:${PYTHONPATH:-}"

# Cross-env readiness checks for production FG layer
conda run -n methanet-gunc gunc -h >/dev/null
[[ -f "${GUNC_DB:-$DB_ROOT/gunc/gunc_db_progenomes2.1.dmnd}" ]] || {
  echo "ERROR: GUNC DB not found" >&2
  exit 1
}

conda run -n methanet-annot emapper.py -h >/dev/null
conda run -n methanet-annot exec_annotation -h >/dev/null
[[ -d "${EGGNOG_DATA_DIR:-$DB_ROOT/eggnog}" ]] || { echo "ERROR: eggNOG DB missing" >&2; exit 1; }
[[ -d "$DB_ROOT/kofam/profiles" && -s "$DB_ROOT/kofam/ko_list" ]] || {
  echo "ERROR: KOfam DB missing" >&2
  exit 1
}

conda run -n methanet-fgintel DRAM.py --help >/dev/null
conda run -n methanet-fgintel hmmsearch -h >/dev/null
[[ -d "$HMM_DIR" ]] || { echo "ERROR: missing HMM dir $HMM_DIR" >&2; exit 1; }

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

HAS_JUPYTER=1
if ! conda run -n methanet-fgintel python -m jupyter --version >/dev/null 2>&1; then
  HAS_JUPYTER=0
  echo "[WARN] jupyter is not available in methanet-fgintel env; falling back to direct python stage execution" >&2
fi

conda run -n methanet-fgintel python "$MROOT/scripts/generate_blue_catalyst_fg_runbook.py"

RUNBOOK_NOTEBOOK="$MROOT/notebooks/blue_catalyst_fgintel_batch_runbook.ipynb"
[[ -f "$RUNBOOK_NOTEBOOK" ]] || { echo "ERROR: missing runbook notebook $RUNBOOK_NOTEBOOK" >&2; exit 1; }

export BC_FG_SOURCE_EMBED_RUN_ID="$FG_SOURCE_EMBED_RUN_ID"
export BC_FG_RUN_ID="$FG_RUN_ID"
export BC_FG_ARTIFACTS_DIR="$FG_ART_DIR"
export BC_FG_BATCH_SIZE="$FG_BATCH_SIZE"
export BC_FG_STAGE="plan"
export BC_FG_MIN_JOIN_COVERAGE="$FG_MIN_JOIN_COVERAGE"
export BC_FG_HASH_PROTEOMES="$FG_HASH_PROTEOMES"
export BC_FG_EMBED_METADATA="$EMBED_META"
export BC_FG_EMBED_NPZ="$EMBED_NPZ"
export BC_FG_HMM_DIR="$HMM_DIR"

if [[ "$HAS_JUPYTER" == "1" ]]; then
  conda run -n methanet-fgintel python -m jupyter nbconvert \
    --to notebook \
    --execute "$RUNBOOK_NOTEBOOK" \
    --output "blue_catalyst_fgintel_plan.executed.ipynb" \
    --output-dir "$FG_ART_DIR" \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3
else
  PLAN_CMD=(
    conda run -n methanet-fgintel python "$MROOT/scripts/blue_catalyst_fg_batch_pipeline.py" plan
    --embedding-metadata "$EMBED_META"
    --embedding-npz "$EMBED_NPZ"
    --embedding-run-id "$FG_SOURCE_EMBED_RUN_ID"
    --output-dir "$FG_ART_DIR"
    --batch-size "$FG_BATCH_SIZE"
  )
  if [[ "$FG_HASH_PROTEOMES" == "1" ]]; then
    PLAN_CMD+=(--hash-proteomes)
  fi
  "${PLAN_CMD[@]}"
fi

BATCH_PLAN="$FG_PLAN_DIR/fg_batch_plan.tsv"
[[ -f "$BATCH_PLAN" ]] || { echo "ERROR: missing batch plan $BATCH_PLAN" >&2; exit 1; }

N_BATCHES=$(($(wc -l < "$BATCH_PLAN") - 1))
[[ "$N_BATCHES" -gt 0 ]] || { echo "ERROR: no batches found in $BATCH_PLAN" >&2; exit 1; }

ARRAY_MAX=$((N_BATCHES - 1))
WORKER_SCRIPT="$MROOT/scripts/slurm/blue_catalyst/run_fgintel_batch_worker_apolo3.sh"
[[ -f "$WORKER_SCRIPT" ]] || { echo "ERROR: missing worker script $WORKER_SCRIPT" >&2; exit 1; }

WORKER_EXPORTS="ALL"
WORKER_EXPORTS+=",MROOT=$MROOT"
WORKER_EXPORTS+=",FG_RUN_ID=$FG_RUN_ID"
WORKER_EXPORTS+=",FG_ART_DIR=$FG_ART_DIR"
WORKER_EXPORTS+=",HMM_DIR=$HMM_DIR"
WORKER_EXPORTS+=",BATCH_THREADS=$BATCH_THREADS"

WORKER_JOB_ID="$(sbatch \
  --parsable \
  --partition="$SLURM_PARTITION" \
  --array="0-${ARRAY_MAX}" \
  --time="$WORKER_TIME_LIMIT" \
  --cpus-per-task="$WORKER_CPUS" \
  --mem="$WORKER_MEM" \
  --export="$WORKER_EXPORTS" \
  "$WORKER_SCRIPT")"

echo "[INFO] Submitted worker array job: $WORKER_JOB_ID (batches=$N_BATCHES)"

while squeue -j "$WORKER_JOB_ID" -h >/dev/null 2>&1 && [[ -n "$(squeue -j "$WORKER_JOB_ID" -h)" ]]; do
  echo "[INFO] waiting for worker array $WORKER_JOB_ID ... $(date '+%F %T')"
  sleep 30
done

echo "[INFO] Worker array sacct"
sacct -j "$WORKER_JOB_ID" --format=JobID,State,ExitCode,Elapsed,Reason%50

export BC_FG_STAGE="merge"
if [[ "$HAS_JUPYTER" == "1" ]]; then
  conda run -n methanet-fgintel python -m jupyter nbconvert \
    --to notebook \
    --execute "$RUNBOOK_NOTEBOOK" \
    --output "blue_catalyst_fgintel_merge.executed.ipynb" \
    --output-dir "$FG_ART_DIR" \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name=python3
else
  conda run -n methanet-fgintel python "$MROOT/scripts/blue_catalyst_fg_batch_pipeline.py" merge \
    --fg-plan-dir "$FG_ART_DIR" \
    --batch-results-dir "$FG_ART_DIR/batch_results" \
    --output-dir "$FG_ART_DIR" \
    --min-join-coverage "$FG_MIN_JOIN_COVERAGE"
fi

conda run -n methanet-fgintel python "$MROOT/scripts/validate_blue_catalyst_fg_artifacts.py" \
  --artifacts-dir "$FG_ART_DIR"

echo "[OK] FG batch pipeline completed. Artifacts: $FG_ART_DIR"
