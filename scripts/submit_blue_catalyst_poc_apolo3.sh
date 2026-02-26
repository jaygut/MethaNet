#!/bin/bash
#SBATCH --job-name=bc_esm2_poc
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/blue_catalyst_poc_%j.out
#SBATCH --error=logs/blue_catalyst_poc_%j.err

set -euo pipefail

module load miniconda3/25.5.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MethaNet311

export METHANET_ROOT="${METHANET_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
export PYTHONPATH="$METHANET_ROOT/src:${PYTHONPATH:-}"

mkdir -p "$METHANET_ROOT/logs"
mkdir -p "$METHANET_ROOT/results/blue_catalyst_poc"

# Prefer scratch caches when writable.
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/$USER}"
if [[ -d "$SCRATCH_BASE" && -w "$SCRATCH_BASE" ]]; then
  CACHE_BASE="$SCRATCH_BASE/methanet_blue_catalyst_cache"
else
  CACHE_BASE="$HOME/.cache/methanet_blue_catalyst_cache"
fi
mkdir -p "$CACHE_BASE/hf" "$CACHE_BASE/tmp" "$CACHE_BASE/xdg"

export HF_HOME="$CACHE_BASE/hf"
export TRANSFORMERS_CACHE="$HF_HOME"
export XDG_CACHE_HOME="$CACHE_BASE/xdg"
export TMPDIR="$CACHE_BASE/tmp"

# Notebook knobs (override with sbatch --export=VAR=...)
export BC_SUBSET_MODE="${BC_SUBSET_MODE:-1}"
export BC_SUBSET_MUCC="${BC_SUBSET_MUCC:-20}"
export BC_SUBSET_RUMEN="${BC_SUBSET_RUMEN:-20}"
export BC_ESM2_MODEL="${BC_ESM2_MODEL:-facebook/esm2_t33_650M_UR50D}"
export BC_ESM2_BATCH="${BC_ESM2_BATCH:-4}"
export BC_ESM2_MAXLEN="${BC_ESM2_MAXLEN:-1022}"
export BC_DEVICE="${BC_DEVICE:-auto}"
export BC_MAX_PROTEINS="${BC_MAX_PROTEINS:-2000}"
export BC_MIN_AA_LEN="${BC_MIN_AA_LEN:-30}"
export RUMEN_ALLOW_GENE_CALLING="${RUMEN_ALLOW_GENE_CALLING:-1}"

# Optional override if the MUCC proteome bundle is not present in Zenodo record 14532347.
# Example:
# export MUCC_MANUAL_PROTEOME_URL="https://.../MUCC_v2.0.0_HQMQ_genes.faa.zip"

if ! python -m jupyter --version >/dev/null 2>&1; then
  echo "ERROR: jupyter is not available in current env. Install it before running this job."
  exit 1
fi

if [[ "$RUMEN_ALLOW_GENE_CALLING" == "1" ]] && ! command -v prodigal >/dev/null 2>&1; then
  echo "ERROR: RUMEN_ALLOW_GENE_CALLING=1 but 'prodigal' is not on PATH."
  echo "Load/install prodigal or set RUMEN_ALLOW_GENE_CALLING=0 if rumen files are protein FASTA."
  exit 1
fi

python -m jupyter nbconvert \
  --to notebook \
  --execute "$METHANET_ROOT/notebooks/blue_catalyst_esm2_poc.ipynb" \
  --output "blue_catalyst_esm2_poc.executed.ipynb" \
  --output-dir "$METHANET_ROOT/results/blue_catalyst_poc" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3

echo "Done. Outputs available under: $METHANET_ROOT/results/blue_catalyst_poc"
