#!/usr/bin/env bash
#SBATCH --job-name=methanet_esm2_manifest
#SBATCH --partition=accel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

module load miniconda3/25.5.1
source "$(conda info --base)/etc/profile.d/conda.sh"

REPO_ROOT="${REPO_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
CONDA_ENV="${CONDA_ENV:-MethaNet311}"
MANIFEST="${ESM2_MANIFEST:?ESM2_MANIFEST is required}"
OUTPUT_DIR="${ESM2_OUTPUT_DIR:?ESM2_OUTPUT_DIR is required}"

conda activate "$CONDA_ENV"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

SCRATCH_BASE="${SCRATCH_BASE:-/home/rsg-jcorre38/scratch}"
CACHE_BASE="${ESM2_CACHE_BASE:-$SCRATCH_BASE/methanet_models/esm2_hf}"
mkdir -p "$CACHE_BASE" "$REPO_ROOT/logs" "$OUTPUT_DIR/logs"
export HF_HOME="$CACHE_BASE"
export TRANSFORMERS_CACHE="$HF_HOME"

echo "[INFO] host=$(hostname)"
echo "[INFO] date=$(date -Is)"
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] CONDA_ENV=$CONDA_ENV"
echo "[INFO] MANIFEST=$MANIFEST"
echo "[INFO] OUTPUT_DIR=$OUTPUT_DIR"
echo "[INFO] HF_HOME=$HF_HOME"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

python "$REPO_ROOT/scripts/embedding/build_manifest_esm2_genome_embeddings.py" \
  --repo-root "$REPO_ROOT" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --id-col "${ESM2_ID_COL:-proteome_id}" \
  --faa-col "${ESM2_FAA_COL:-proteome_faa}" \
  --mag-id-col "${ESM2_MAG_ID_COL:-mag_id}" \
  --source-col "${ESM2_SOURCE_COL:-source}" \
  --ecosystem-col "${ESM2_ECOSYSTEM_COL:-ecosystem}" \
  --domain-col "${ESM2_DOMAIN_COL:-domain}" \
  --source-group-col "${ESM2_SOURCE_GROUP_COL:-source_group}" \
  --protein-count-col "${ESM2_PROTEIN_COUNT_COL:-protein_count}" \
  --model-name "${ESM2_MODEL_NAME:-facebook/esm2_t33_650M_UR50D}" \
  --batch-size "${ESM2_BATCH_SIZE:-4}" \
  --max-length "${ESM2_MAX_LENGTH:-1022}" \
  --max-proteins-per-proteome "${ESM2_MAX_PROTEINS_PER_PROTEOME:-6000}" \
  --min-aa-len "${ESM2_MIN_AA_LEN:-30}" \
  --checkpoint-every "${ESM2_CHECKPOINT_EVERY:-25}" \
  --device "${ESM2_DEVICE:-auto}" \
  --cache-dir "$HF_HOME" \
  ${ESM2_FP16:+--fp16}

echo "[INFO] finished=$(date -Is)"
nvidia-smi || true
