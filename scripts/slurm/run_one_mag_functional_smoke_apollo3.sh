#!/usr/bin/env bash
#SBATCH --job-name=methanet_one_mag_fgx
#SBATCH --partition=longjobs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"
DB_ROOT="${DB_ROOT:-/home/rsg-jcorre38/scratch/methanet_db}"
FASTA="${FASTA:-${REPO_ROOT}/data/assemblies/OWC_0041.fasta}"
PROTEOME_ID="${PROTEOME_ID:-}"
PROTEOME_FAA="${PROTEOME_FAA:-}"
MAG_ID="${MAG_ID:-$(basename "$FASTA")}"
MAG_ID="${MAG_ID%.gz}"
MAG_ID="${MAG_ID%.fasta}"
MAG_ID="${MAG_ID%.fna}"
MAG_ID="${MAG_ID%.fa}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
RUN_ID="${RUN_ID:-one_mag_fgx_${MAG_ID}_$(date -u +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/results/functional_metagenomics/one_mag_smoke/${RUN_ID}}"
case "$RESULT_ROOT" in
  /*) ;;
  *) RESULT_ROOT="${REPO_ROOT}/${RESULT_ROOT#./}" ;;
esac
CONDA_SH="${CONDA_SH:-/opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh}"
TIME_BIN="${TIME_BIN:-}"
if [[ -z "$TIME_BIN" ]]; then
  for candidate in /usr/bin/time /bin/time; do
    if [[ -x "$candidate" ]]; then
      TIME_BIN="$candidate"
      break
    fi
  done
fi

OUT="${RESULT_ROOT}"
LOG_DIR="${OUT}/logs"
TIMINGS="${OUT}/timings.tsv"
SUMMARY="${OUT}/summary.tsv"
STATUS="${OUT}/status.tsv"
STAGED_FA="${OUT}/staged_fa"
STAGED_FASTA="${OUT}/staged_fasta"
INPUT_DIR="${OUT}/input"
DBCAN_COMPAT_DIR="${DBCAN_COMPAT_DIR:-${DB_ROOT}/dbcan_compat_pressed}"
PREPARE_DBCAN_CACHE="${PREPARE_DBCAN_CACHE:-1}"
CURATE_RUN="${CURATE_RUN:-1}"
PRUNE_SUCCESS="${PRUNE_SUCCESS:-0}"
COMPRESS_LOGS="${COMPRESS_LOGS:-0}"
WORK_FASTA="${INPUT_DIR}/${MAG_ID}.fasta"

mkdir -p "$OUT" "$LOG_DIR" "$STAGED_FA" "$STAGED_FASTA" "$INPUT_DIR"
mkdir -p "$OUT"/{genes,kofam,mcycdb,scycdb,dbcan,bakta,checkm2,gunc,gtdbtk,metabolic,tmp}
mkdir -p "${OUT}/tmp/gunc" "${OUT}/tmp/gtdbtk"

exec > >(tee -ai "$LOG_DIR/driver.out") 2> >(tee -ai "$LOG_DIR/driver.err" >&2)

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -s "$1" ]] || die "Required file is missing or empty: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "Required directory is missing: $1"
}

activate_env() {
  local env_name="$1"
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$env_name"
}

append_status() {
  local step="$1"
  local status="$2"
  local note="${3:-}"
  note="${note//$'\t'/ }"
  note="${note//$'\r'/ }"
  note="${note//$'\n'/ ; }"
  printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$step" "$status" "$note" >> "$STATUS"
}

run_step() {
  local step="$1"
  shift
  local step_dir="${OUT}/${step}"
  local stdout_log="${LOG_DIR}/${step}.out"
  local stderr_log="${LOG_DIR}/${step}.err"
  local time_log="${LOG_DIR}/${step}.time.txt"
  mkdir -p "$step_dir"

  local start_epoch end_epoch elapsed rc max_rss_kb
  start_epoch="$(date +%s)"
  append_status "$step" "started" "$*"
  set +e
  if [[ -n "$TIME_BIN" ]]; then
    "$TIME_BIN" -v -o "$time_log" "$@" >"$stdout_log" 2>"$stderr_log"
    rc=$?
  else
    "$@" >"$stdout_log" 2>"$stderr_log"
    rc=$?
    printf 'Maximum resident set size (kbytes): NA\n' > "$time_log"
  fi
  set -e
  end_epoch="$(date +%s)"
  elapsed=$((end_epoch - start_epoch))
  max_rss_kb="$(awk -F: '/Maximum resident set size/ {gsub(/^[ \t]+/, "", $2); print $2}' "$time_log" 2>/dev/null || true)"
  max_rss_kb="${max_rss_kb:-NA}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$step" "$start_epoch" "$end_epoch" "$elapsed" "$rc" "$max_rss_kb" "$time_log" >> "$TIMINGS"
  if [[ "$rc" -ne 0 ]]; then
    append_status "$step" "failed" "rc=${rc}; see ${stderr_log}"
    die "Step failed: ${step} (rc=${rc}); see ${stderr_log}"
  fi
  append_status "$step" "complete" "elapsed_seconds=${elapsed}"
}

run_step_shell() {
  local step="$1"
  local command="$2"
  run_step "$step" bash -c "$command"
}

count_noncomment_lines() {
  local file="$1"
  if [[ -s "$file" ]]; then
    awk 'NF && $1 !~ /^#/' "$file" | wc -l
  else
    printf '0\n'
  fi
}

record_metric() {
  printf '%s\t%s\n' "$1" "$2" >> "$SUMMARY"
}

write_assembly_stats() {
  local fasta="$1"
  local out_tsv="$2"
  awk '
    /^>/ {
      if (seen) {
        n += 1
        total += len
        lengths[n] = len
      }
      seen = 1
      len = 0
      next
    }
    {
      gsub(/[[:space:]]/, "")
      len += length($0)
    }
    END {
      if (seen) {
        n += 1
        total += len
        lengths[n] = len
      }
      for (i = 1; i <= n; i++) {
        for (j = i + 1; j <= n; j++) {
          if (lengths[j] > lengths[i]) {
            tmp = lengths[i]; lengths[i] = lengths[j]; lengths[j] = tmp
          }
        }
      }
      half = total / 2
      run = 0
      n50 = 0
      largest = 0
      for (i = 1; i <= n; i++) {
        if (lengths[i] > largest) largest = lengths[i]
        run += lengths[i]
        if (n50 == 0 && run >= half) n50 = lengths[i]
      }
      print "metric\tvalue"
      print "contigs\t" n
      print "total_bp\t" total
      print "largest_contig_bp\t" largest
      print "n50_bp\t" n50
    }
  ' "$fasta" > "$out_tsv"
}

echo "MethaNet one-MAG functional smoke run"
echo "repo=${REPO_ROOT}"
echo "db_root=${DB_ROOT}"
echo "dbcan_compat_dir=${DBCAN_COMPAT_DIR}"
echo "proteome_id=${PROTEOME_ID}"
echo "proteome_faa=${PROTEOME_FAA}"
echo "mag_id=${MAG_ID}"
echo "fasta=${FASTA}"
echo "threads=${THREADS}"
echo "out=${OUT}"
echo

require_file "$FASTA"
require_file "${DB_ROOT}/checkm2/CheckM2_database/uniref100.KO.1.dmnd"
require_dir "${DB_ROOT}/gtdbtk_r232/release232"
require_file "${DB_ROOT}/gunc/gunc_db_progenomes3.dmnd"
require_file "${DB_ROOT}/kofam/profiles/prokaryote.hal"
require_file "${DB_ROOT}/kofam/ko_list"
require_file "${DB_ROOT}/mcycdb/MCycDB_2021.dmnd"
require_file "${DB_ROOT}/scycdb/SCycDB_2020Mar.dmnd"
require_dir "${DB_ROOT}/dbcan"
require_file "${DB_ROOT}/dbcan/dbCAN.hmm"
require_file "${DB_ROOT}/dbcan/dbCAN-sub.hmm"
require_file "${REPO_ROOT}/scripts/prepare_dbcan_compat_cache_apollo3.sh"
require_file "${REPO_ROOT}/scripts/curate_functional_mag_run.py"
require_file "${DB_ROOT}/metabolic/METABOLIC/METABOLIC-G.pl"
require_file "${DB_ROOT}/bakta/db-light/bakta.db"
require_file "$CONDA_SH"

case "$FASTA" in
  *.gz)
    gzip -cd "$FASTA" > "$WORK_FASTA"
    ;;
  *)
    ln -sf "$(readlink -f "$FASTA")" "$WORK_FASTA"
    ;;
esac
require_file "$WORK_FASTA"

printf 'step\tstart_epoch\tend_epoch\telapsed_seconds\trc\tmax_rss_kb\ttime_log\n' > "$TIMINGS"
printf 'timestamp\tstep\tstatus\tnote\n' > "$STATUS"
printf 'metric\tvalue\n' > "$SUMMARY"

ln -sf "$(readlink -f "$WORK_FASTA")" "${STAGED_FA}/${MAG_ID}.fa"
cp "$WORK_FASTA" "${STAGED_FASTA}/${MAG_ID}.fasta"

INPUT_STATS_SCRIPT="${LOG_DIR}/write_assembly_stats.sh"
{
  declare -f write_assembly_stats
  printf 'write_assembly_stats "$1" "$2"\n'
} > "$INPUT_STATS_SCRIPT"
run_step input_stats bash "$INPUT_STATS_SCRIPT" "$WORK_FASTA" "${OUT}/input_stats.tsv"

activate_env methanet-fgx
run_step prodigal prodigal \
  -i "$WORK_FASTA" \
  -a "${OUT}/genes/${MAG_ID}.faa" \
  -d "${OUT}/genes/${MAG_ID}.ffn" \
  -o "${OUT}/genes/${MAG_ID}.gff" \
  -f gff \
  -p meta
require_file "${OUT}/genes/${MAG_ID}.faa"

run_step_shell kofam "cd '${OUT}/kofam' && exec_annotation \
  --cpu '$THREADS' \
  --profile '${DB_ROOT}/kofam/profiles/prokaryote.hal' \
  --ko-list '${DB_ROOT}/kofam/ko_list' \
  --format detail-tsv \
  -o '${OUT}/kofam/${MAG_ID}.kofam.detail.tsv' \
  '${OUT}/genes/${MAG_ID}.faa'"

run_step mcycdb diamond blastp \
  -q "${OUT}/genes/${MAG_ID}.faa" \
  -d "${DB_ROOT}/mcycdb/MCycDB_2021.dmnd" \
  -o "${OUT}/mcycdb/${MAG_ID}.diamond.tsv" \
  -f 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovhsp scovhsp \
  --evalue 1e-10 \
  --query-cover 70 \
  --subject-cover 50 \
  --threads "$THREADS"

run_step scycdb diamond blastp \
  -q "${OUT}/genes/${MAG_ID}.faa" \
  -d "${DB_ROOT}/scycdb/SCycDB_2020Mar.dmnd" \
  -o "${OUT}/scycdb/${MAG_ID}.diamond.tsv" \
  -f 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovhsp scovhsp \
  --evalue 1e-10 \
  --query-cover 70 \
  --subject-cover 50 \
  --threads "$THREADS"

activate_env methanet-dbcan
if [[ "$PREPARE_DBCAN_CACHE" == "1" ]]; then
  run_step dbcan_prepare_cache env \
    DB_ROOT="$DB_ROOT" \
    DBCAN_COMPAT_DIR="$DBCAN_COMPAT_DIR" \
    CONDA_SH="$CONDA_SH" \
    "${REPO_ROOT}/scripts/prepare_dbcan_compat_cache_apollo3.sh"
fi
require_file "${DBCAN_COMPAT_DIR}/dbCAN.hmm.h3p"
require_file "${DBCAN_COMPAT_DIR}/dbCAN_sub.hmm.h3p"
run_step dbcan run_dbcan "$WORK_FASTA" prok \
  --db_dir "$DBCAN_COMPAT_DIR" \
  --dbCANFile dbCAN.hmm \
  --out_dir "${OUT}/dbcan" \
  --dia_cpu "$THREADS" \
  --hmm_cpu "$THREADS" \
  --dbcan_thread "$THREADS" \
  --tools all
if grep -qi '^ERROR:' "${LOG_DIR}/dbcan.out" "${LOG_DIR}/dbcan.err"; then
  append_status dbcan failed "run_dbcan printed ERROR despite exit code 0"
  die "dbCAN printed an ERROR; see ${LOG_DIR}/dbcan.out and ${LOG_DIR}/dbcan.err"
fi
find "${OUT}/dbcan" -type f -print -quit | grep -q . || die "dbCAN completed without output files"

activate_env methanet-bakta
run_step bakta bakta \
  --db "${DB_ROOT}/bakta/db-light" \
  --meta \
  --skip-sorf \
  --threads "$THREADS" \
  --output "${OUT}/bakta" \
  --prefix "$MAG_ID" \
  --force \
  "$WORK_FASTA"

activate_env checkm2_py38
run_step checkm2 checkm2 predict \
  --input "$STAGED_FA" \
  --output-directory "${OUT}/checkm2" \
  --threads "$THREADS" \
  --extension fa \
  --database_path "${DB_ROOT}/checkm2/CheckM2_database/uniref100.KO.1.dmnd" \
  --force

activate_env methanet-gunc3
run_step gunc gunc run \
  -d "$STAGED_FA" \
  -o "${OUT}/gunc" \
  -t "$THREADS" \
  -e .fa \
  --db_file "${DB_ROOT}/gunc/gunc_db_progenomes3.dmnd" \
  --temp_dir "${OUT}/tmp/gunc"

activate_env methanet-gtdbtk272
export GTDBTK_DATA_PATH="${DB_ROOT}/gtdbtk_r232/release232"
run_step gtdbtk gtdbtk classify_wf \
  --genome_dir "$STAGED_FA" \
  --out_dir "${OUT}/gtdbtk" \
  --extension fa \
  --cpus "$THREADS" \
  --pplacer_cpus "$THREADS" \
  --tmpdir "${OUT}/tmp/gtdbtk" \
  --force

activate_env methanet-metabolic
run_step metabolic_prodigal_check prodigal -v
run_step metabolic_bioperl_check perl -MBio::SeqIO -e 1
run_step metabolic_r_check Rscript -e 'stopifnot(requireNamespace("openxlsx", quietly = TRUE))'
run_step metabolic perl "${DB_ROOT}/metabolic/METABOLIC/METABOLIC-G.pl" \
  -in-gn "$STAGED_FASTA" \
  -o "${OUT}/metabolic" \
  -t "$THREADS" \
  -p meta \
  -kofam-db full
if grep -Eiq '(^Error:|Execution halted|No such file|cannot access|readline\\(\\) on closed filehandle)' "${LOG_DIR}/metabolic.out" "${LOG_DIR}/metabolic.err" "${OUT}/metabolic/METABOLIC_log.log"; then
  append_status metabolic failed "METABOLIC logged internal errors despite exit code 0"
  die "METABOLIC logged internal errors; see ${LOG_DIR}/metabolic.err and ${OUT}/metabolic/METABOLIC_log.log"
fi
require_file "${OUT}/metabolic/METABOLIC_result.xlsx"

record_metric "run_id" "$RUN_ID"
record_metric "proteome_id" "$PROTEOME_ID"
record_metric "mag_id" "$MAG_ID"
record_metric "fasta" "$FASTA"
record_metric "work_fasta" "$WORK_FASTA"
record_metric "result_root" "$OUT"
record_metric "threads" "$THREADS"
record_metric "dbcan_compat_dir" "$DBCAN_COMPAT_DIR"
record_metric "input_contigs" "$(awk '$1=="contigs"{print $2}' "${OUT}/input_stats.tsv")"
record_metric "input_total_bp" "$(awk '$1=="total_bp"{print $2}' "${OUT}/input_stats.tsv")"
record_metric "input_n50_bp" "$(awk '$1=="n50_bp"{print $2}' "${OUT}/input_stats.tsv")"
record_metric "prodigal_proteins" "$(grep -c '^>' "${OUT}/genes/${MAG_ID}.faa")"
record_metric "kofam_rows" "$(count_noncomment_lines "${OUT}/kofam/${MAG_ID}.kofam.detail.tsv")"
record_metric "mcycdb_hits" "$(count_noncomment_lines "${OUT}/mcycdb/${MAG_ID}.diamond.tsv")"
record_metric "scycdb_hits" "$(count_noncomment_lines "${OUT}/scycdb/${MAG_ID}.diamond.tsv")"
record_metric "dbcan_overview_rows" "$(count_noncomment_lines "${OUT}/dbcan/overview.txt")"
record_metric "bakta_feature_rows" "$(count_noncomment_lines "${OUT}/bakta/${MAG_ID}.tsv")"
record_metric "checkm2_quality_rows" "$(count_noncomment_lines "${OUT}/checkm2/quality_report.tsv")"
record_metric "gunc_maxcss_rows" "$(find "${OUT}/gunc" -name '*maxCSS_level.tsv' -type f -print -quit | xargs -r awk 'NR > 1 && NF {n += 1} END {print n + 0}')"
record_metric "gtdbtk_summary_rows" "$(find "${OUT}/gtdbtk" -name 'gtdbtk.*.summary.tsv' -type f -print0 | xargs -0 -r awk 'FNR > 1 && NF {n += 1} END {print n + 0}')"
record_metric "metabolic_result_files" "$(find "${OUT}/metabolic" -type f | wc -l)"
record_metric "total_elapsed_seconds" "$(awk 'NR>1 {s += $4} END {print s + 0}' "$TIMINGS")"

append_status run complete "summary=${SUMMARY}"
touch "${OUT}/COMPLETE"
if [[ "$CURATE_RUN" == "1" ]]; then
  activate_env methanet-fgx
  curate_args=(
    --run-dir "$OUT"
    --repo-root "$REPO_ROOT"
    --proteome-id "$PROTEOME_ID"
    --mag-id "$MAG_ID"
    --cohort-run-id "${COHORT_RUN_ID:-one_mag_smoke}"
    --job-id "${SLURM_JOB_ID:-}"
    --cpus "$THREADS"
    --mag-fasta "$FASTA"
    --write-parquet
  )
  if [[ -n "$PROTEOME_FAA" ]]; then
    curate_args+=(--proteome-faa "$PROTEOME_FAA")
  fi
  if [[ "$COMPRESS_LOGS" == "1" ]]; then
    curate_args+=(--compress-logs)
  fi
  if [[ "$PRUNE_SUCCESS" == "1" ]]; then
    curate_args+=(--prune-success)
  fi
  "${REPO_ROOT}/scripts/curate_functional_mag_run.py" "${curate_args[@]}"
fi
echo "Complete: ${OUT}"
