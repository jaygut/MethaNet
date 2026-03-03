#!/usr/bin/env bash
set -euo pipefail

# Submit a Blue Catalyst run that prioritizes embedding from existing proteomes.
#
# Usage:
#   SOURCE_RUN_ID=apolo_full_20260228_080644 \
#   bash scripts/submit_blue_catalyst_embedding_only_apolo3.sh
#
# Optional env:
#   MROOT                 project root (default set below)
#   RUN_ID                new run id (default: <SOURCE_RUN_ID>_embed_<timestamp>)
#   TIME_LIMIT            sbatch time limit override (default: 08:00:00)
#   BC_EMBED_COHORT_MODE  strict_run|extended_cache (default: strict_run)
#   BC_EXCLUDE_COASSEMBLY 1|0 (default: 1)

MROOT="${MROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-}"
RUN_ID="${RUN_ID:-}"
TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
BC_EMBED_COHORT_MODE="${BC_EMBED_COHORT_MODE:-strict_run}"
BC_EXCLUDE_COASSEMBLY="${BC_EXCLUDE_COASSEMBLY:-1}"

if [[ -z "$SOURCE_RUN_ID" ]]; then
  echo "ERROR: SOURCE_RUN_ID is required." >&2
  exit 1
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="${SOURCE_RUN_ID}_embed_$(date +%Y%m%d_%H%M%S)"
fi

SOURCE_ART="$MROOT/results/blue_catalyst_poc/runs/$SOURCE_RUN_ID/artifacts"
TARGET_ART="$MROOT/results/blue_catalyst_poc/runs/$RUN_ID/artifacts"
SOURCE_SUBSET="$SOURCE_ART/prjeb31266_selected_subset.tsv"
JOB_SCRIPT="$MROOT/scripts/submit_blue_catalyst_poc_apolo3.sh"

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "ERROR: job script not found: $JOB_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$SOURCE_SUBSET" ]]; then
  echo "ERROR: source subset not found: $SOURCE_SUBSET" >&2
  exit 1
fi

subset_rows=$(($(wc -l < "$SOURCE_SUBSET") - 1))
if [[ "$subset_rows" -lt 1 ]]; then
  echo "ERROR: source subset appears empty: $SOURCE_SUBSET" >&2
  exit 1
fi

mkdir -p "$TARGET_ART"
cp "$SOURCE_SUBSET" "$TARGET_ART/prjeb31266_source_subset_seed.tsv"

export_list="ALL"
export_list+=",BC_RUN_ID=$RUN_ID"
export_list+=",BC_ARTIFACTS_DIR=$TARGET_ART"
export_list+=",BC_SOURCE_SUBSET_PATH=$SOURCE_SUBSET"
export_list+=",BC_SUBSET_MODE=1"
export_list+=",BC_SAFE_RUMEN_CAP=1"
export_list+=",BC_PREFER_LOCAL_RUMEN_CACHE=1"
export_list+=",BC_SUBSET_RUMEN=$subset_rows"
export_list+=",RUMEN_ALLOW_GENE_CALLING=0"
export_list+=",BC_NETWORK_PREFLIGHT=0"
export_list+=",BC_EMBED_COHORT_MODE=$BC_EMBED_COHORT_MODE"
export_list+=",BC_EXCLUDE_COASSEMBLY=$BC_EXCLUDE_COASSEMBLY"

echo "[INFO] SOURCE_RUN_ID=$SOURCE_RUN_ID"
echo "[INFO] RUN_ID=$RUN_ID"
echo "[INFO] SOURCE_SUBSET=$SOURCE_SUBSET"
echo "[INFO] TARGET_ART=$TARGET_ART"
echo "[INFO] subset_rows=$subset_rows"
echo "[INFO] BC_EMBED_COHORT_MODE=$BC_EMBED_COHORT_MODE"
echo "[INFO] BC_EXCLUDE_COASSEMBLY=$BC_EXCLUDE_COASSEMBLY"
echo "[INFO] TIME_LIMIT=$TIME_LIMIT"

new_job_id=$(sbatch --parsable --time="$TIME_LIMIT" --export="$export_list" "$JOB_SCRIPT")

echo "[INFO] Submitted embedding-focused job: $new_job_id"
echo "Monitor:"
echo "watch -n 30 'squeue -j $new_job_id; echo; sacct -j $new_job_id --format=JobID,State,ExitCode,Elapsed,Reason%50'"

echo

echo "Validate on completion:"
echo "python $MROOT/scripts/validate_blue_catalyst_artifacts.py --artifacts-dir $TARGET_ART"
