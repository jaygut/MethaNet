#!/usr/bin/env bash
# Submit the MethaNet functional-metagenomics database setup job to Apollo-3 SLURM.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
DB_ROOT="${DB_ROOT:-/home/rsg-jcorre38/scratch/methanet_db}"
RUN_ID="${RUN_ID:-fgx_db_setup_$(date -u +%Y%m%d_%H%M%S)}"
PARTITION="${PARTITION:-longjobs}"
CPUS="${CPUS:-16}"
MEM="${MEM:-128G}"
TIME="${TIME:-6-00:00:00}"
SETUP_STEPS="${SETUP_STEPS:-preflight,create_env,checkm2,gtdbtk_r232,gunc_progenomes3,kofam,eggnog_v2,mcycdb,scycdb,dbcan,dram,metabolic,mmseqs}"
EGGNOG_MODE="${EGGNOG_MODE:-gated}"
EGGNOG_BASE_URL="${EGGNOG_BASE_URL:-http://eggnog5.embl.de/download/emapperdb-5.0.2}"
DOWNLOAD_RESUME_ATTEMPTS="${DOWNLOAD_RESUME_ATTEMPTS:-24}"

mkdir -p "${DB_ROOT}/logs/${RUN_ID}" "${REPO_ROOT}/logs/functional_metagenomics/db_setup"

export DB_ROOT RUN_ID THREADS="$CPUS" REPO_ROOT SETUP_STEPS EGGNOG_MODE EGGNOG_BASE_URL DOWNLOAD_RESUME_ATTEMPTS

sbatch \
  --parsable \
  --job-name "methanet-fgx-db" \
  --partition "$PARTITION" \
  --cpus-per-task "$CPUS" \
  --mem "$MEM" \
  --time "$TIME" \
  --chdir "$REPO_ROOT" \
  --output "${DB_ROOT}/logs/${RUN_ID}/slurm-%j.out" \
  --error "${DB_ROOT}/logs/${RUN_ID}/slurm-%j.err" \
  --export ALL \
  "${REPO_ROOT}/scripts/setup_functional_metagenomics_dbs_apollo3.sh"
