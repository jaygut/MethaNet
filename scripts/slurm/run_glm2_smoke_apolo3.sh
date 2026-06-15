#!/usr/bin/env bash
#SBATCH --job-name=methanet_glm2_smoke
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=results/contextual_genomics/glm2_smoke_20260615_090023/logs/slurm-%j.out
#SBATCH --error=results/contextual_genomics/glm2_smoke_20260615_090023/logs/slurm-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/contextual_genomics/glm2_smoke_20260615_090023}"
MODEL_NAME="${MODEL_NAME:-tattabio/gLM2_650M}"
MODEL_REVISION="${MODEL_REVISION:-08754cba59a1f97d517f873fad6c672d2b1abdc7}"
CONDA_ENV="${CONDA_ENV:-methanet-glm2}"
HF_HOME="${HF_HOME:-/home/rsg-jcorre38/scratch/methanet_models/hf}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

mkdir -p "$RESULTS_DIR/logs" "$HF_HOME"
cd "$REPO_ROOT"

module load miniconda3/25.5.1
source /opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export HF_HOME
export TRANSFORMERS_CACHE
export TOKENIZERS_PARALLELISM=false

{
  date
  hostname
  sinfo -o "%P %a %D %c %m %G %l %N"
  nvidia-smi
  conda list
  python -m pip freeze
} > "$RESULTS_DIR/logs/runtime_lock_and_gpu_state.txt" 2>&1

python scripts/contextual_genomics/run_glm2_smoke_inference.py \
  --results-dir "$RESULTS_DIR" \
  --model-name "$MODEL_NAME" \
  --model-revision "$MODEL_REVISION" \
  --require-cuda
