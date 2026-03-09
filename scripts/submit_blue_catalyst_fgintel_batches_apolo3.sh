#!/usr/bin/env bash
set -euo pipefail

# Submit the production FG batch pipeline on Apolo3.
#
# Usage:
#   FG_SOURCE_EMBED_RUN_ID=apolo_full_... \
#   bash scripts/submit_blue_catalyst_fgintel_batches_apolo3.sh
#
# Optional env overrides:
#   MROOT                 project root
#   FG_RUN_ID             target FG run id
#   FG_BATCH_SIZE         MAGs per batch (default 25)
#   FG_MIN_JOIN_COVERAGE  minimum join coverage required (default 0.95)
#   SLURM_PARTITION       default accel
#   TIME_LIMIT            outer orchestrator walltime (default 08:00:00)
#   CPUS_PER_TASK         outer orchestrator CPUs (default 2)
#   MEM_PER_NODE          outer orchestrator memory (default 16G)
#   WORKER_TIME_LIMIT     worker walltime (default 04:00:00)
#   WORKER_CPUS           worker CPUs (default 4)
#   WORKER_MEM            worker memory (default 24G)
#   BATCH_THREADS         hmmsearch threads per worker (default 4)
#   HMM_DIR               marker HMM directory (default $MROOT/data/hmm)
#   DB_ROOT               shared functional DB root
#   FG_HASH_PROTEOMES     1 computes proteome SHA256 during planning (default 0)

MROOT="${MROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
FG_SOURCE_EMBED_RUN_ID="${FG_SOURCE_EMBED_RUN_ID:-}"
FG_RUN_ID="${FG_RUN_ID:-}"
FG_BATCH_SIZE="${FG_BATCH_SIZE:-25}"
FG_MIN_JOIN_COVERAGE="${FG_MIN_JOIN_COVERAGE:-0.95}"
SLURM_PARTITION="${SLURM_PARTITION:-accel}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-2}"
MEM_PER_NODE="${MEM_PER_NODE:-16G}"
WORKER_TIME_LIMIT="${WORKER_TIME_LIMIT:-04:00:00}"
WORKER_CPUS="${WORKER_CPUS:-4}"
WORKER_MEM="${WORKER_MEM:-24G}"
BATCH_THREADS="${BATCH_THREADS:-4}"
HMM_DIR="${HMM_DIR:-$MROOT/data/hmm}"
DB_ROOT="${DB_ROOT:-$HOME/scratch/methanet_db}"
FG_HASH_PROTEOMES="${FG_HASH_PROTEOMES:-0}"

if [[ -z "$FG_SOURCE_EMBED_RUN_ID" ]]; then
  echo "ERROR: FG_SOURCE_EMBED_RUN_ID is required" >&2
  exit 1
fi

if [[ -z "$FG_RUN_ID" ]]; then
  FG_RUN_ID="${FG_SOURCE_EMBED_RUN_ID}_fg_$(date +%Y%m%d_%H%M%S)"
fi

JOB_SCRIPT="$MROOT/scripts/slurm/blue_catalyst/run_fgintel_batches_apolo3.sh"
[[ -f "$JOB_SCRIPT" ]] || {
  echo "ERROR: missing job script $JOB_SCRIPT" >&2
  exit 1
}

export_list="ALL"
export_list+=",MROOT=$MROOT"
export_list+=",FG_SOURCE_EMBED_RUN_ID=$FG_SOURCE_EMBED_RUN_ID"
export_list+=",FG_RUN_ID=$FG_RUN_ID"
export_list+=",FG_BATCH_SIZE=$FG_BATCH_SIZE"
export_list+=",FG_MIN_JOIN_COVERAGE=$FG_MIN_JOIN_COVERAGE"
export_list+=",SLURM_PARTITION=$SLURM_PARTITION"
export_list+=",WORKER_TIME_LIMIT=$WORKER_TIME_LIMIT"
export_list+=",WORKER_CPUS=$WORKER_CPUS"
export_list+=",WORKER_MEM=$WORKER_MEM"
export_list+=",BATCH_THREADS=$BATCH_THREADS"
export_list+=",HMM_DIR=$HMM_DIR"
export_list+=",DB_ROOT=$DB_ROOT"
export_list+=",FG_HASH_PROTEOMES=$FG_HASH_PROTEOMES"

job_id="$(sbatch \
  --parsable \
  --partition="$SLURM_PARTITION" \
  --time="$TIME_LIMIT" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM_PER_NODE" \
  --export="$export_list" \
  "$JOB_SCRIPT")"

echo "[OK] Submitted FG orchestrator job: $job_id"
echo "[OK] FG_RUN_ID=$FG_RUN_ID"
echo "[OK] Follow with: squeue -j $job_id ; sacct -j $job_id --format=JobID,State,ExitCode,Elapsed,Reason%50"
