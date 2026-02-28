#!/bin/bash
#SBATCH --job-name=bc_esm2_poc
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/blue_catalyst_poc_%j.out
#SBATCH --error=logs/blue_catalyst_poc_%j.err

set -euo pipefail

module load miniconda3/25.5.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MethaNet311

ENV_PREFIX="${ENV_PREFIX:-$HOME/.conda/envs/MethaNet311}"
ENV_PY="${ENV_PREFIX}/bin/python"
export PATH="${ENV_PREFIX}/bin:${PATH}"

echo "DEBUG CONDA_PREFIX=${CONDA_PREFIX:-unset}"
echo "DEBUG ENV_PY=${ENV_PY}"
echo "DEBUG which python=$(which python)"

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
export BC_RUN_ID="${BC_RUN_ID:-apolo_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export BC_ARTIFACTS_DIR="${BC_ARTIFACTS_DIR:-$METHANET_ROOT/results/blue_catalyst_poc/runs/$BC_RUN_ID/artifacts}"
export BC_OFFLINE_MODE="${BC_OFFLINE_MODE:-0}"
export BC_SUBSET_MODE="${BC_SUBSET_MODE:-1}"
export BC_SUBSET_MUCC="${BC_SUBSET_MUCC:-200}"
if [[ -z "${BC_SUBSET_RUMEN:-}" ]]; then
  LOCAL_RUMEN_RAW_DIR="$METHANET_ROOT/data/blue_catalyst_poc/rumen/raw"
  if [[ -d "$LOCAL_RUMEN_RAW_DIR" ]]; then
    local_rumen_count=$(find "$LOCAL_RUMEN_RAW_DIR" -maxdepth 1 -type f | wc -l)
  else
    local_rumen_count=0
  fi
  if [[ "$local_rumen_count" -gt 0 ]]; then
    export BC_SUBSET_RUMEN="$local_rumen_count"
  else
    export BC_SUBSET_RUMEN="200"
  fi
else
  export BC_SUBSET_RUMEN
fi
# Keep capped rumen intake even if BC_SUBSET_MODE is accidentally disabled.
export BC_SAFE_RUMEN_CAP="${BC_SAFE_RUMEN_CAP:-1}"
export BC_PREFER_LOCAL_RUMEN_CACHE="${BC_PREFER_LOCAL_RUMEN_CACHE:-1}"
export BC_RUMEN_MAX_PER_ANALYSIS="${BC_RUMEN_MAX_PER_ANALYSIS:-3}"
export BC_ESM2_MODEL="${BC_ESM2_MODEL:-facebook/esm2_t33_650M_UR50D}"
export BC_ESM2_BATCH="${BC_ESM2_BATCH:-4}"
export BC_ESM2_MAXLEN="${BC_ESM2_MAXLEN:-1022}"
export BC_DEVICE="${BC_DEVICE:-auto}"
export BC_MAX_PROTEINS="${BC_MAX_PROTEINS:-6000}"
export BC_MIN_AA_LEN="${BC_MIN_AA_LEN:-30}"
export RUMEN_ALLOW_GENE_CALLING="${RUMEN_ALLOW_GENE_CALLING:-1}"
export BC_NETWORK_PREFLIGHT="${BC_NETWORK_PREFLIGHT:-1}"
export BC_NETWORK_PREFLIGHT_N="${BC_NETWORK_PREFLIGHT_N:-3}"
export BC_NETWORK_PREFLIGHT_TIMEOUT="${BC_NETWORK_PREFLIGHT_TIMEOUT:-8}"

# Optional override if the MUCC proteome bundle is not present in Zenodo record 14532347.
# Example:
# export MUCC_MANUAL_PROTEOME_URL="https://.../MUCC_v2.0.0_HQMQ_genes.faa.zip"

if ! "$ENV_PY" -m jupyter --version >/dev/null 2>&1; then
  echo "ERROR: jupyter is not available in current env. Install it before running this job."
  exit 1
fi

if [[ "$RUMEN_ALLOW_GENE_CALLING" == "1" ]] && ! command -v prodigal >/dev/null 2>&1; then
  echo "ERROR: RUMEN_ALLOW_GENE_CALLING=1 but 'prodigal' is not on PATH."
  echo "Load/install prodigal or set RUMEN_ALLOW_GENE_CALLING=0 if rumen files are protein FASTA."
  exit 1
fi

echo "DEBUG BC_RUN_ID=${BC_RUN_ID}"
echo "DEBUG BC_ARTIFACTS_DIR=${BC_ARTIFACTS_DIR}"
echo "DEBUG BC_SUBSET_MODE=${BC_SUBSET_MODE}"
echo "DEBUG BC_SUBSET_MUCC=${BC_SUBSET_MUCC}"
echo "DEBUG BC_SUBSET_RUMEN=${BC_SUBSET_RUMEN}"
echo "DEBUG BC_SAFE_RUMEN_CAP=${BC_SAFE_RUMEN_CAP}"
echo "DEBUG BC_PREFER_LOCAL_RUMEN_CACHE=${BC_PREFER_LOCAL_RUMEN_CACHE}"
echo "DEBUG BC_RUMEN_MAX_PER_ANALYSIS=${BC_RUMEN_MAX_PER_ANALYSIS}"
echo "DEBUG BC_MAX_PROTEINS=${BC_MAX_PROTEINS}"
echo "DEBUG BC_NETWORK_PREFLIGHT=${BC_NETWORK_PREFLIGHT}"
echo "DEBUG BC_NETWORK_PREFLIGHT_N=${BC_NETWORK_PREFLIGHT_N}"
echo "DEBUG BC_NETWORK_PREFLIGHT_TIMEOUT=${BC_NETWORK_PREFLIGHT_TIMEOUT}"

echo "DEBUG: Using $ENV_PY for jupyter nbconvert"
"$ENV_PY" -m jupyter nbconvert \
  --to notebook \
  --execute "$METHANET_ROOT/notebooks/blue_catalyst_esm2_poc.ipynb" \
  --output "blue_catalyst_poc.executed.ipynb" \
  --output-dir "$BC_ARTIFACTS_DIR" \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3

echo "Done. Outputs available under: $BC_ARTIFACTS_DIR"
