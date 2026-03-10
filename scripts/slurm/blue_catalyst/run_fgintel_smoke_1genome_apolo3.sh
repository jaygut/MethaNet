#!/usr/bin/env bash
#SBATCH --job-name=bc-fg-smoke-1g
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=accel

set -euo pipefail

# Blue Catalyst functional-intel smoke test on a single previously used MAG lineage.
# This script:
# 1) checks tool/env readiness across split envs
# 2) builds a 1-genome subset from a previous run's selected subset
# 3) submits the notebook pipeline as a nested sbatch job
# 4) waits and validates required artifacts

MROOT="${MROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
RUNS_ROOT="${RUNS_ROOT:-$MROOT/results/blue_catalyst_poc/runs}"
DB_ROOT="${DB_ROOT:-$HOME/scratch/methanet_db}"
SLURM_PARTITION="${SLURM_PARTITION:-accel}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-}"
OUTER_TIME_LIMIT="${OUTER_TIME_LIMIT:-06:00:00}"
OUTER_CPUS_PER_TASK="${OUTER_CPUS_PER_TASK:-2}"
OUTER_MEM_PER_NODE="${OUTER_MEM_PER_NODE:-16G}"

if [[ -z "$SOURCE_RUN_ID" ]]; then
  SOURCE_RUN_ID="$(find "$RUNS_ROOT" -mindepth 2 -maxdepth 2 -type f -name prjeb31266_selected_subset.tsv \
    | sed "s|$RUNS_ROOT/||" | cut -d/ -f1 | tail -n1)"
fi
[[ -n "${SOURCE_RUN_ID:-}" ]] || { echo "ERROR: could not infer SOURCE_RUN_ID"; exit 1; }

SOURCE_ART="$RUNS_ROOT/$SOURCE_RUN_ID/artifacts"
SOURCE_SUBSET="$SOURCE_ART/prjeb31266_selected_subset.tsv"
[[ -f "$SOURCE_SUBSET" ]] || { echo "ERROR: missing $SOURCE_SUBSET"; exit 1; }

SMOKE_RUN_ID="${SMOKE_RUN_ID:-apolo_smoke_1genome_$(date +%Y%m%d_%H%M%S)}"
SMOKE_DIR="$RUNS_ROOT/$SMOKE_RUN_ID/artifacts"
mkdir -p "$SMOKE_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[INFO] SOURCE_RUN_ID=$SOURCE_RUN_ID"
echo "[INFO] SMOKE_RUN_ID=$SMOKE_RUN_ID"
echo "[INFO] OUTER resources: time=${OUTER_TIME_LIMIT} cpus=${OUTER_CPUS_PER_TASK} mem=${OUTER_MEM_PER_NODE}"
echo "[INFO] INNER resources: time=${INNER_TIME_LIMIT:-06:00:00} cpus=${INNER_CPUS_PER_TASK:-2} mem=${INNER_MEM_PER_NODE:-16G}"

# Build 1-genome subset (header + first data row)
head -n 2 "$SOURCE_SUBSET" > "$SMOKE_DIR/prjeb31266_selected_subset_1genome.tsv"
[[ $(wc -l < "$SMOKE_DIR/prjeb31266_selected_subset_1genome.tsv") -eq 2 ]] || {
  echo "ERROR: failed to build 1-genome subset"; exit 1;
}

# Cross-env sanity checks
conda activate methanet-gunc
gunc -h >/dev/null
[[ -f "${GUNC_DB:-$DB_ROOT/gunc/gunc_db_progenomes2.1.dmnd}" ]] || {
  echo "ERROR: GUNC DB not found"; exit 1;
}

conda activate methanet-annot
emapper.py -h >/dev/null
exec_annotation -h >/dev/null
[[ -d "${EGGNOG_DATA_DIR:-$DB_ROOT/eggnog}" ]] || { echo "ERROR: eggNOG DB missing"; exit 1; }
[[ -d "$DB_ROOT/kofam/profiles" && -s "$DB_ROOT/kofam/ko_list" ]] || {
  echo "ERROR: KOfam DB missing"; exit 1;
}

conda activate methanet-fgintel
DRAM.py --help >/dev/null
mmseqs version >/dev/null
hmmsearch -h >/dev/null
prodigal -h >/dev/null

# Submit underlying notebook pipeline job
JOB_SCRIPT="$MROOT/scripts/submit_blue_catalyst_poc_apolo3.sh"
[[ -f "$JOB_SCRIPT" ]] || { echo "ERROR: missing $JOB_SCRIPT"; exit 1; }

EXPORTS="ALL"
EXPORTS+=",BC_RUN_ID=$SMOKE_RUN_ID"
EXPORTS+=",BC_ARTIFACTS_DIR=$SMOKE_DIR"
EXPORTS+=",BC_SOURCE_SUBSET_PATH=$SMOKE_DIR/prjeb31266_selected_subset_1genome.tsv"
EXPORTS+=",BC_SUBSET_MODE=1"
EXPORTS+=",BC_SAFE_RUMEN_CAP=1"
EXPORTS+=",BC_PREFER_LOCAL_RUMEN_CACHE=1"
EXPORTS+=",BC_SUBSET_RUMEN=1"
EXPORTS+=",BC_SUBSET_MUCC=0"
EXPORTS+=",RUMEN_ALLOW_GENE_CALLING=0"
EXPORTS+=",BC_NETWORK_PREFLIGHT=0"
EXPORTS+=",BC_EMBED_COHORT_MODE=strict_run"
EXPORTS+=",BC_EXCLUDE_COASSEMBLY=1"
EXPORTS+=",BC_EMBED_CHECKPOINT_EVERY=1"
EXPORTS+=",BC_DEVICE=cpu"
EXPORTS+=",DB_ROOT=$DB_ROOT"
EXPORTS+=",EGGNOG_DATA_DIR=$DB_ROOT/eggnog"
EXPORTS+=",GUNC_DB=$DB_ROOT/gunc/gunc_db_progenomes2.1.dmnd"

SBATCH_CMD=(
  sbatch
  --parsable
  --partition="${SLURM_PARTITION}"
  --time="${INNER_TIME_LIMIT:-06:00:00}"
  --cpus-per-task="${INNER_CPUS_PER_TASK:-2}"
  --mem="${INNER_MEM_PER_NODE:-16G}"
  --export="$EXPORTS"
)
if [[ -n "${INNER_GPU_GRES:-}" ]]; then
  SBATCH_CMD+=(--gres="${INNER_GPU_GRES}")
fi
SBATCH_CMD+=("$JOB_SCRIPT")

JOB_ID="$("${SBATCH_CMD[@]}")"

echo "[INFO] Submitted inner smoke job: $JOB_ID"

while squeue -j "$JOB_ID" -h >/dev/null 2>&1 && [[ -n "$(squeue -j "$JOB_ID" -h)" ]]; do
  echo "[INFO] waiting for $JOB_ID ... $(date '+%F %T')"
  sleep 30
done

echo "[INFO] Final sacct for inner job"
if ! sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed,Reason%50; then
  echo "[WARN] sacct unavailable (slurmdbd down or inaccessible); continuing without accounting report" >&2
fi

conda activate methanet-fgintel
python "$MROOT/scripts/validate_blue_catalyst_artifacts.py" --artifacts-dir "$SMOKE_DIR"

test -s "$SMOKE_DIR/genome_embeddings.npz"
test -s "$SMOKE_DIR/embedding_metadata.tsv"
test -s "$SMOKE_DIR/embedding_projection_clusters.tsv"
test -s "$SMOKE_DIR/poc_metrics.json"
test -s "$SMOKE_DIR/advanced_analytics_summary.json"

# Stronger content checks so this validates real output generation, not only file presence.
SMOKE_DIR="$SMOKE_DIR" python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

smoke_dir = Path(os.environ["SMOKE_DIR"])

emb = np.load(smoke_dir / "genome_embeddings.npz")
arr = emb["embeddings"]
if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
    raise SystemExit("genome_embeddings.npz has invalid embedding shape")

meta = pd.read_csv(smoke_dir / "embedding_metadata.tsv", sep="\t")
if meta.shape[0] < 1:
    raise SystemExit("embedding_metadata.tsv has no embedded samples")

proj = pd.read_csv(smoke_dir / "embedding_projection_clusters.tsv", sep="\t")
if proj.shape[0] < 1:
    raise SystemExit("embedding_projection_clusters.tsv has no projection rows")

metrics = json.loads((smoke_dir / "poc_metrics.json").read_text(encoding="utf-8"))
summary = json.loads((smoke_dir / "advanced_analytics_summary.json").read_text(encoding="utf-8"))
if not isinstance(metrics, dict) or not metrics:
    raise SystemExit("poc_metrics.json is empty or invalid")
if not isinstance(summary, dict) or not summary:
    raise SystemExit("advanced_analytics_summary.json is empty or invalid")

print("[OK] Artifact content validation passed")
PY

echo "[OK] Smoke test passed. Artifacts: $SMOKE_DIR"
