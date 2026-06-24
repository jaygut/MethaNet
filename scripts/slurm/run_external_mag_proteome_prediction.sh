#!/usr/bin/env bash
# Generic Prodigal protein/gene prediction job for MethaNet external MAG lanes.
# This script does not submit itself; call it with sbatch and source-specific
# paths through environment variables.
#SBATCH --job-name=methanet_prodigal_lane
#SBATCH --partition=longjobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
MANIFEST="${MANIFEST:?MANIFEST is required}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:?OUTPUT_MANIFEST is required}"
PROTEOME_DIR="${PROTEOME_DIR:?PROTEOME_DIR is required}"
FFN_DIR="${FFN_DIR:?FFN_DIR is required}"
GFF_DIR="${GFF_DIR:?GFF_DIR is required}"
LOG_DIR="${LOG_DIR:?LOG_DIR is required}"
SCRATCH_DIR="${SCRATCH_DIR:-/home/rsg-jcorre38/scratch/methanet_prodigal}"
CONDA_ENV="${CONDA_ENV:-methanet-fgx}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
LIMIT="${LIMIT:-}"
INCLUDE_COL="${INCLUDE_COL:-}"
FNA_COL="${FNA_COL:-mag_fasta}"
FAA_COL="${FAA_COL:-proteome_faa}"
FORCE="${FORCE:-0}"

module load miniconda3/25.5.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p "$REPO_ROOT/logs" "$SCRATCH_DIR"
cd "$REPO_ROOT"

echo "[INFO] host=$(hostname)"
echo "[INFO] date=$(date -Is)"
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] MANIFEST=$MANIFEST"
echo "[INFO] OUTPUT_MANIFEST=$OUTPUT_MANIFEST"
echo "[INFO] WORKERS=$WORKERS"
echo "[INFO] LIMIT=${LIMIT:-none}"

cmd=(
  python "$REPO_ROOT/scripts/external/predict_external_mag_proteomes.py"
  --repo-root "$REPO_ROOT"
  --manifest "$MANIFEST"
  --output-manifest "$OUTPUT_MANIFEST"
  --proteome-dir "$PROTEOME_DIR"
  --ffn-dir "$FFN_DIR"
  --gff-dir "$GFF_DIR"
  --log-dir "$LOG_DIR"
  --scratch-dir "$SCRATCH_DIR"
  --fna-col "$FNA_COL"
  --faa-col "$FAA_COL"
  --workers "$WORKERS"
)

if [[ -n "$LIMIT" ]]; then
  cmd+=(--limit "$LIMIT")
fi
if [[ -n "$INCLUDE_COL" ]]; then
  cmd+=(--include-col "$INCLUDE_COL")
fi
if [[ "$FORCE" == "1" ]]; then
  cmd+=(--force)
fi

"${cmd[@]}"
echo "[INFO] finished=$(date -Is)"
