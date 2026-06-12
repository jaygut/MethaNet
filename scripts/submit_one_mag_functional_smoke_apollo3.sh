#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
DB_ROOT="${DB_ROOT:-/home/rsg-jcorre38/scratch/methanet_db}"
FASTA="${FASTA:-${REPO_ROOT}/data/assemblies/OWC_0041.fasta}"
THREADS="${THREADS:-16}"
PROTEOME_ID="${PROTEOME_ID:-}"
MAG_ID="${MAG_ID:-$(basename "$FASTA")}"
MAG_ID="${MAG_ID%.gz}"
MAG_ID="${MAG_ID%.fasta}"
MAG_ID="${MAG_ID%.fna}"
MAG_ID="${MAG_ID%.fa}"
RUN_ID="${RUN_ID:-one_mag_fgx_${MAG_ID}_$(date -u +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/functional_metagenomics/one_mag_smoke/${RUN_ID}}"
SCRIPT="${REPO_ROOT}/scripts/slurm/run_one_mag_functional_smoke_apollo3.sh"

mkdir -p "${RESULT_ROOT}/logs"

sbatch \
  --cpus-per-task="$THREADS" \
  --output="${RESULT_ROOT}/logs/slurm-%j.out" \
  --error="${RESULT_ROOT}/logs/slurm-%j.err" \
  --export=ALL,REPO_ROOT="$REPO_ROOT",DB_ROOT="$DB_ROOT",FASTA="$FASTA",PROTEOME_ID="$PROTEOME_ID",MAG_ID="$MAG_ID",THREADS="$THREADS",RUN_ID="$RUN_ID",RESULT_ROOT="$RESULT_ROOT" \
  "$SCRIPT"

printf 'run_id\t%s\nresult_root\t%s\nproteome_id\t%s\nmag_id\t%s\nfasta\t%s\nthreads\t%s\n' \
  "$RUN_ID" "$RESULT_ROOT" "$PROTEOME_ID" "$MAG_ID" "$FASTA" "$THREADS" > "${RESULT_ROOT}/submission.tsv"
