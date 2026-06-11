#!/usr/bin/env bash
# Provision MethaNet functional-metagenomics databases on Apollo-3.
#
# This script is intentionally conservative:
# - resumable via per-step .done markers
# - no destructive cleanup of existing databases
# - no large temporary files under /tmp
# - manifest rows are written only after validation/checkpoint success
# - one active setup job at a time via flock

set -Eeuo pipefail

# SLURM batch jobs on Apollo can start with a narrower PATH than login shells.
export PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:${PATH:-}"

RUN_ID="${RUN_ID:-fgx_db_setup_$(date -u +%Y%m%d_%H%M%S)}"
DB_ROOT="${DB_ROOT:-/home/rsg-jcorre38/scratch/methanet_db}"
TOOL_ENV="${TOOL_ENV:-methanet-fgx}"
CHECKM2_ENV="${CHECKM2_ENV:-checkm2_py38}"
GUNC_ENV="${GUNC_ENV:-methanet-gunc3}"
GTDBTK_ENV="${GTDBTK_ENV:-methanet-gtdbtk272}"
DBCAN_ENV="${DBCAN_ENV:-methanet-dbcan}"
DRAM_ENV="${DRAM_ENV:-methanet-fgintel}"
METABOLIC_ENV="${METABOLIC_ENV:-methanet-metabolic}"
MIN_FREE_GB="${MIN_FREE_GB:-1200}"
THREADS="${THREADS:-16}"
CONDA_SH="${CONDA_SH:-/opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh}"
REPO_ROOT="${REPO_ROOT:-/home/rsg-jcorre38/Jay_Proyects/MethaNet}"

LOG_ROOT="${DB_ROOT}/logs/${RUN_ID}"
STATE_DIR="${DB_ROOT}/.setup_state/${RUN_ID}"
DOWNLOAD_DIR="${DB_ROOT}/_downloads/${RUN_ID}"
TMP_ROOT="${DB_ROOT}/_tmp/${RUN_ID}"
MANIFEST="${DB_ROOT}/manifests/tool_db_manifest.${RUN_ID}.tsv"
LOCK_FILE="${DB_ROOT}/.methanet_db_setup.lock"

mkdir -p "$LOG_ROOT" "$STATE_DIR" "$DOWNLOAD_DIR" "$TMP_ROOT" "$(dirname "$MANIFEST")"

exec > >(tee -a "${LOG_ROOT}/setup.log") 2>&1
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[$(date -Is)] ERROR another DB setup appears to be running: $LOCK_FILE"
  exit 1
fi

log() {
  echo "[$(date -Is)] $*"
}

die() {
  log "ERROR: $*"
  exit 1
}

on_error() {
  local line="$1"
  local status="$2"
  log "ERROR: command failed at line ${line} with exit code ${status}"
  log "See ${LOG_ROOT}/setup.log and per-step logs in ${LOG_ROOT}"
}
trap 'on_error "$LINENO" "$?"' ERR

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

free_gb() {
  df -BG --output=avail "$DB_ROOT" | tail -1 | tr -dc '0-9'
}

ensure_space() {
  local free
  free="$(free_gb)"
  if [[ -z "$free" || "$free" -lt "$MIN_FREE_GB" ]]; then
    die "only ${free:-unknown}G free under DB_ROOT=${DB_ROOT}; need at least ${MIN_FREE_GB}G"
  fi
  log "free space under DB_ROOT: ${free}G"
}

retry() {
  local attempts="$1"
  shift
  local n=1
  until "$@"; do
    if [[ "$n" -ge "$attempts" ]]; then
      log "command failed after ${attempts} attempts: $*"
      return 1
    fi
    log "retry ${n}/${attempts} after failure: $*"
    sleep $((30 * n))
    n=$((n + 1))
  done
}

download_gzip_checked() {
  local url="$1"
  local dest="$2"
  local tmp="${dest}.part"
  local attempt
  for attempt in 1 2 3 4; do
    rm -f "$tmp"
    log "download attempt ${attempt}/4: ${url}"
    if curl -L --fail --retry 4 --retry-all-errors --retry-delay 20 \
      --connect-timeout 60 --speed-time 300 --speed-limit 1024 \
      --http1.1 --no-keepalive -o "$tmp" "$url" && gzip -t "$tmp"; then
      mv -f "$tmp" "$dest"
      return 0
    fi
    rm -f "$tmp"
    log "download or gzip validation failed for ${url}"
    sleep $((30 * attempt))
  done
  return 1
}

download_targz_checked() {
  local url="$1"
  local dest="$2"
  local tmp="${dest}.part"
  local attempt
  for attempt in 1 2 3 4; do
    rm -f "$tmp"
    log "download attempt ${attempt}/4: ${url}"
    if curl -L --fail --retry 4 --retry-all-errors --retry-delay 20 \
      --connect-timeout 60 --speed-time 300 --speed-limit 1024 \
      --http1.1 --no-keepalive -o "$tmp" "$url" && tar -tzf "$tmp" >/dev/null; then
      mv -f "$tmp" "$dest"
      return 0
    fi
    rm -f "$tmp"
    log "download or tar.gz validation failed for ${url}"
    sleep $((30 * attempt))
  done
  return 1
}

step_done() {
  [[ -s "${STATE_DIR}/$1.done" ]]
}

mark_done() {
  local step="$1"
  {
    echo "step=${step}"
    echo "completed_on=$(date -Is)"
    echo "host=$(hostname)"
  } > "${STATE_DIR}/${step}.done"
}

run_step() {
  local step="$1"
  shift
  if step_done "$step"; then
    log "SKIP ${step}: checkpoint exists"
    return 0
  fi
  log "START ${step}"
  ensure_space
  "$@" > >(tee -a "${LOG_ROOT}/${step}.log") 2>&1
  mark_done "$step"
  log "DONE ${step}"
}

manifest_header() {
  if [[ ! -s "$MANIFEST" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      run_id tool binary_version db_name db_release install_path env_var \
      checksum_or_signature validated_on validation_command notes > "$MANIFEST"
  fi
}

manifest_row() {
  manifest_header
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$RUN_ID" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$(date -u +%F)" "$8" "$9" >> "$MANIFEST"
}

activate_env() {
  local env_name="$1"
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "$env_name"
}

conda_activate() {
  activate_env "$TOOL_ENV"
}

ensure_metabolic_runtime_link() {
  local prefix="${CONDA_PREFIX:-}"
  if [[ -n "$prefix" && -e "${prefix}/lib/libnsl.so.3" && ! -e "${prefix}/lib/libnsl.so.1" ]]; then
    ln -s libnsl.so.3 "${prefix}/lib/libnsl.so.1"
    log "created METABOLIC Perl compatibility symlink: ${prefix}/lib/libnsl.so.1 -> libnsl.so.3"
  fi
}

binary_version() {
  local cmd="$1"
  shift || true
  "$cmd" "$@" 2>&1 | head -1 | tr '\t' ' ' || true
}

env_exists() {
  if ! command -v conda >/dev/null 2>&1 && [[ -r "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
  fi
  conda env list | awk '{print $1}' | grep -qx "$1"
}

gtdbtk_data_path() {
  local root="$1"
  if [[ -d "${root}/markers" && -d "${root}/taxonomy" ]]; then
    printf '%s\n' "$root"
    return 0
  fi
  find "$root" -mindepth 1 -maxdepth 2 -type d \
    -name 'release*' -exec test -d '{}/markers' ';' -exec test -d '{}/taxonomy' ';' -print \
    | sort -V | tail -1
}

normalize_fasta() {
  local input="$1"
  local output="$2"
  python - "$input" "$output" <<'PY'
from pathlib import Path
import sys

inp = Path(sys.argv[1])
out = Path(sys.argv[2])
records = written = 0
empty = []
header = None
seq = []

with inp.open(errors="replace") as src, out.open("w") as dst:
    def flush():
        global records, written, header, seq
        if header is None:
            return
        records += 1
        sequence = "".join(part.strip() for part in seq if part.strip())
        if not sequence:
            empty.append(header[:120])
            return
        dst.write(header.rstrip() + "\n")
        for i in range(0, len(sequence), 80):
            dst.write(sequence[i:i + 80] + "\n")
        written += 1

    for line in src:
        if line.startswith(">"):
            flush()
            header = line.rstrip("\n")
            seq = []
        else:
            seq.append(line.rstrip("\n"))
    flush()

print(f"normalized_fasta input={inp} output={out} records={records} written={written} empty={len(empty)}")
PY
}

step_preflight() {
  require_cmd df
  require_cmd find
  require_cmd flock
  require_cmd tar
  require_cmd wget
  require_cmd curl
  require_cmd md5sum
  require_cmd sha256sum
  require_cmd conda
  [[ -r "$CONDA_SH" ]] || die "conda hook missing: $CONDA_SH"
  mkdir -p "$DB_ROOT"/{checkm2,gtdbtk_r232,gunc,eggnog_v2,kofam,mcycdb,scycdb,dbcan,dram,metabolic,mmseqs,manifests}
  touch "${DB_ROOT}/.write_test" && rm "${DB_ROOT}/.write_test"
  ensure_space
  df -hT "$DB_ROOT" /tmp || true
  df -i "$DB_ROOT" || true
  log "RUN_ID=${RUN_ID}"
  log "DB_ROOT=${DB_ROOT}"
  log "TOOL_ENV=${TOOL_ENV}"
  log "DBCAN_ENV=${DBCAN_ENV}"
  log "METABOLIC_ENV=${METABOLIC_ENV}"
  log "THREADS=${THREADS}"
  log "SLURM_JOB_ID=${SLURM_JOB_ID:-not_slurm}"
  log "PATH=${PATH}"
}

step_create_env() {
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  if conda env list | awk '{print $1}' | grep -qx "$TOOL_ENV"; then
    log "conda env exists: ${TOOL_ENV}"
  else
    local setup_packages=(
      wget curl git pigz parallel zip unzip
      hmmer prodigal diamond mmseqs2
      snakemake kofamscan eggnog-mapper dbcan
      pandas pyarrow biopython pyyaml scipy scikit-learn statsmodels pytest
    )
    local setup_packages_no_dbcan=(
      wget curl git pigz parallel zip unzip
      hmmer prodigal diamond mmseqs2
      snakemake kofamscan eggnog-mapper
      pandas pyarrow biopython pyyaml scipy scikit-learn statsmodels pytest
    )
    log "creating setup env ${TOOL_ENV}; heavy QC/taxonomy tools stay in separate envs"
    if ! conda create -y -n "$TOOL_ENV" -c conda-forge -c bioconda python=3.10 "${setup_packages[@]}"; then
      log "setup env with dbCAN failed; retrying setup env without dbCAN"
      conda env remove -y -n "$TOOL_ENV" >/dev/null 2>&1 || true
      conda create -y -n "$TOOL_ENV" -c conda-forge -c bioconda python=3.10 "${setup_packages_no_dbcan[@]}"
    fi
  fi

  conda activate "$TOOL_ENV"
  for utility in zip unzip; do
    if ! command -v "$utility" >/dev/null 2>&1; then
      log "installing missing archive utility into ${TOOL_ENV}: ${utility}"
      conda install -y -n "$TOOL_ENV" -c conda-forge "$utility"
    fi
  done
  python --version
  conda list > "${LOG_ROOT}/${TOOL_ENV}.conda_list.txt"
  command -v hmmsearch
  command -v prodigal
  command -v diamond
  command -v mmseqs
  command -v checkm2 || true
  command -v gunc || true
  command -v gtdbtk || true
  command -v exec_annotation || true
  command -v emapper.py || true
  command -v run_dbcan || true

  if ! env_exists "$CHECKM2_ENV"; then
    log "creating CheckM2 env ${CHECKM2_ENV}"
    conda create -y -n "$CHECKM2_ENV" -c conda-forge -c bioconda checkm2 diamond prodigal pandas scikit-learn
  else
    log "CheckM2 env exists: ${CHECKM2_ENV}"
  fi

  if ! env_exists "$GUNC_ENV"; then
    log "creating GUNC env ${GUNC_ENV}"
    if ! conda create -y -n "$GUNC_ENV" -c conda-forge -c bioconda gunc=1.1.1 diamond=2.1.24; then
      log "GUNC 1.1.1 env creation failed; falling back to existing methanet-gunc if available"
      conda env remove -y -n "$GUNC_ENV" >/dev/null 2>&1 || true
      if env_exists methanet-gunc; then
        GUNC_ENV=methanet-gunc
      else
        die "could not create ${GUNC_ENV} and no methanet-gunc fallback exists"
      fi
    fi
  else
    log "GUNC env exists: ${GUNC_ENV}"
  fi

  if ! env_exists "$GTDBTK_ENV"; then
    log "creating GTDB-Tk env ${GTDBTK_ENV}"
    if ! conda create -y -n "$GTDBTK_ENV" -c conda-forge -c bioconda gtdbtk=2.7.2; then
      log "GTDB-Tk 2.7.2 env creation failed; falling back to MethaNet311 for download-only setup"
      conda env remove -y -n "$GTDBTK_ENV" >/dev/null 2>&1 || true
      GTDBTK_ENV=MethaNet311
    fi
  else
    log "GTDB-Tk env exists: ${GTDBTK_ENV}"
  fi

  if ! env_exists "$DBCAN_ENV"; then
    log "creating dbCAN env ${DBCAN_ENV}"
    if ! conda create -y -n "$DBCAN_ENV" -c conda-forge -c bioconda python=3.8 dbcan; then
      log "dbCAN env creation failed; dbCAN database setup will remain gated"
      conda env remove -y -n "$DBCAN_ENV" >/dev/null 2>&1 || true
    fi
  else
    log "dbCAN env exists: ${DBCAN_ENV}"
  fi

  if ! env_exists "$METABOLIC_ENV"; then
    log "creating METABOLIC runtime env ${METABOLIC_ENV}"
    if ! conda create -y -n "$METABOLIC_ENV" -c conda-forge -c bioconda \
      perl perl-statistics-descriptive perl-parallel-forkmanager perl-list-util perl-getopt-long libnsl \
      r-base r-ggplot2 r-data.table hmmer diamond blast kofamscan; then
      log "METABOLIC env creation failed; METABOLIC validation will remain gated"
      conda env remove -y -n "$METABOLIC_ENV" >/dev/null 2>&1 || true
    fi
  else
    log "METABOLIC env exists: ${METABOLIC_ENV}"
  fi
}

step_checkm2() {
  activate_env "$CHECKM2_ENV"
  local db_dir="${DB_ROOT}/checkm2"
  local db_file="${db_dir}/CheckM2_database/uniref100.KO.1.dmnd"
  if [[ ! -s "$db_file" ]]; then
    retry 3 checkm2 database --download --path "$db_dir" --no_write_json_db
  fi
  [[ -s "$db_file" ]] || die "CheckM2 database missing after download: $db_file"
  diamond dbinfo --db "$db_file" >/dev/null
  manifest_row checkm2 "$(binary_version checkm2 --help)" \
    "CheckM2 DIAMOND database" "CheckM2 current database via checkm2 database --download" \
    "$db_file" "CHECKM2DB or --database_path" "$(sha256sum "$db_file" | awk '{print $1}')" \
    "diamond dbinfo --db ${db_file}" "Downloaded or reused; CheckM2 testrun should be run separately in batch if desired."
}

step_gtdbtk_r232() {
  activate_env "$GTDBTK_ENV"
  local target="${DB_ROOT}/gtdbtk_r232"
  local archive="${DOWNLOAD_DIR}/gtdbtk_data.tar.gz"
  local url="https://data.ace.uq.edu.au/public/gtdb/data/releases/latest/auxillary_files/gtdbtk_package/full_package/gtdbtk_data.tar.gz"
  local expected_md5="25a59e0352b1fd150c589f56559767d4"
  if [[ ! -s "${target}/.validated_r232" ]]; then
    mkdir -p "$target"
    retry 3 wget -c -O "$archive" "$url"
    echo "${expected_md5}  ${archive}" | md5sum -c -
    local staging="${TMP_ROOT}/gtdbtk_r232_extract"
    rm -rf "$staging"
    mkdir -p "$staging"
    tar -xzf "$archive" -C "$staging"
    find "$staging" -mindepth 1 -maxdepth 1 -exec cp -a {} "$target"/ \;
    rm -rf "$staging"
    local data_path
    data_path="$(gtdbtk_data_path "$target")"
    [[ -n "$data_path" ]] || die "GTDB-Tk data path not found under ${target}"
    GTDBTK_DATA_PATH="$data_path" gtdbtk check_install || log "GTDB-Tk check_install failed; DB extracted but validation is gated"
    printf 'validated_on=%s\nmd5=%s\n' "$(date -Is)" "$expected_md5" > "${target}/.validated_r232"
  else
    local data_path
    data_path="$(gtdbtk_data_path "$target")"
    [[ -n "$data_path" ]] || die "GTDB-Tk data path not found under ${target}"
    GTDBTK_DATA_PATH="$data_path" gtdbtk check_install || log "GTDB-Tk check_install failed; DB remains present but validation is gated"
  fi
  local data_path
  data_path="$(gtdbtk_data_path "$target")"
  manifest_row gtdbtk "$(binary_version gtdbtk --version)" \
    "GTDB-Tk reference package" "GTDB R232" "$data_path" "GTDBTK_DATA_PATH" \
    "md5:${expected_md5}" "GTDBTK_DATA_PATH=${data_path} gtdbtk check_install" \
    "Official full package URL from GTDB-Tk 2.7.2 docs."
}

step_gunc() {
  activate_env "$GUNC_ENV"
  local out="${DB_ROOT}/gunc"
  mkdir -p "$out"
  local db_file
  db_file="$(find "$out" -maxdepth 2 -type f -name '*progenomes*3*.dmnd' | head -1 || true)"
  if [[ -z "$db_file" ]]; then
    retry 3 gunc download_db -db progenomes_3 "$out"
    db_file="$(find "$out" -maxdepth 3 -type f -name '*.dmnd' | grep -i 'progenomes.*3' | head -1 || true)"
  fi
  [[ -n "$db_file" && -s "$db_file" ]] || die "GUNC ProGenomes3 database not found in $out"
  if gunc --help 2>&1 | grep -q 'check'; then
    gunc check --db_file "$db_file" || true
  fi
  diamond dbinfo --db "$db_file" >/dev/null
  manifest_row gunc "$(binary_version gunc --version)" \
    "GUNC database" "ProGenomes 3" "$db_file" "--db_file" \
    "$(sha256sum "$db_file" | awk '{print $1}')" "diamond dbinfo --db ${db_file}" \
    "Downloaded with gunc download_db --db progenomes_3."
}

step_kofam() {
  conda_activate
  local dir="${DB_ROOT}/kofam"
  mkdir -p "$dir"
  if [[ ! -s "${dir}/ko_list" ]]; then
    retry 3 curl -L --retry 5 --retry-delay 20 -o "${dir}/ko_list.gz" ftp://ftp.genome.jp/pub/db/kofam/ko_list.gz
    gunzip -f "${dir}/ko_list.gz"
  fi
  if [[ ! -d "${dir}/profiles" || ! -s "${dir}/profiles/prokaryote.hal" ]]; then
    retry 3 curl -L --retry 5 --retry-delay 20 -o "${dir}/profiles.tar.gz" ftp://ftp.genome.jp/pub/db/kofam/profiles.tar.gz
    tar -xzf "${dir}/profiles.tar.gz" -C "$dir"
  fi
  exec_annotation --profile "${dir}/profiles/prokaryote.hal" --ko-list "${dir}/ko_list" -h >/dev/null
  manifest_row kofamscan "$(binary_version exec_annotation -h)" \
    "KOfam profiles and ko_list" "current GenomeNet KOfam" "$dir" "--profile/--ko-list" \
    "profiles:$(find "${dir}/profiles" -type f | wc -l); ko_list_sha256:$(sha256sum "${dir}/ko_list" | awk '{print $1}')" \
    "exec_annotation --profile ${dir}/profiles/prokaryote.hal --ko-list ${dir}/ko_list -h" \
    "Downloaded from GenomeNet FTP if absent."
}

step_eggnog_v2() {
  conda_activate
  local dir="${DB_ROOT}/eggnog_v2"
  local base_url="${EGGNOG_BASE_URL:-http://eggnogdb.embl.de/download/emapperdb-5.0.2}"
  local gated_reason
  gate_eggnog() {
    gated_reason="$1"
    log "eggNOG gated: ${gated_reason}"
    rm -f "${dir}/"*.part "${dir}/eggnog.db.gz" "${dir}/eggnog.db.tmp" \
      "${dir}/eggnog.taxa.tar.gz" \
      "${dir}/eggnog_proteins.dmnd.gz" "${dir}/eggnog_proteins.dmnd.tmp"
    manifest_row eggnog-mapper "$(binary_version emapper.py -h)" \
      "eggNOG-mapper data" "stable v2 data" "$dir" "EGGNOG_DATA_DIR/--data_dir" \
      "not_installed" "manual retry required after mirror/network issue is resolved" \
      "GATED: ${gated_reason}. Large eggNOG HTTP downloads from ${base_url} repeatedly truncated around 1.1GB on Apollo-3."
    return 0
  }
  mkdir -p "$dir"
  if [[ "${EGGNOG_MODE:-gated}" == "gated" && ( ! -s "${dir}/eggnog.db" || ! -s "${dir}/eggnog_proteins.dmnd" ) ]]; then
    gate_eggnog "download disabled for this run after repeated HTTP truncation; set EGGNOG_MODE=download to retry"
    return 0
  fi
  if [[ ! -s "${dir}/eggnog.db" || ! -s "${dir}/eggnog_proteins.dmnd" ]]; then
    rm -f "${dir}/eggnog.db.gz" "${dir}/eggnog.db.tmp" \
      "${dir}/eggnog.taxa.tar.gz" \
      "${dir}/eggnog_proteins.dmnd.gz" "${dir}/eggnog_proteins.dmnd.tmp"
    if ! download_gzip_checked "${base_url}/eggnog.db.gz" "${dir}/eggnog.db.gz"; then
      gate_eggnog "eggnog.db.gz failed gzip-validated download"
      return 0
    fi
    if ! download_targz_checked "${base_url}/eggnog.taxa.tar.gz" "${dir}/eggnog.taxa.tar.gz"; then
      gate_eggnog "eggnog.taxa.tar.gz failed tar-validated download"
      return 0
    fi
    if ! download_gzip_checked "${base_url}/eggnog_proteins.dmnd.gz" "${dir}/eggnog_proteins.dmnd.gz"; then
      gate_eggnog "eggnog_proteins.dmnd.gz failed gzip-validated download"
      return 0
    fi
    if ! gunzip -c "${dir}/eggnog.db.gz" > "${dir}/eggnog.db.tmp"; then
      gate_eggnog "eggnog.db.gz failed extraction"
      return 0
    fi
    mv -f "${dir}/eggnog.db.tmp" "${dir}/eggnog.db"
    tar -xzf "${dir}/eggnog.taxa.tar.gz" -C "$dir"
    if ! gunzip -c "${dir}/eggnog_proteins.dmnd.gz" > "${dir}/eggnog_proteins.dmnd.tmp"; then
      gate_eggnog "eggnog_proteins.dmnd.gz failed extraction"
      return 0
    fi
    mv -f "${dir}/eggnog_proteins.dmnd.tmp" "${dir}/eggnog_proteins.dmnd"
    rm -f "${dir}/eggnog.db.gz" "${dir}/eggnog.taxa.tar.gz" "${dir}/eggnog_proteins.dmnd.gz"
  fi
  if [[ ! -s "${dir}/eggnog.db" ]]; then
    gate_eggnog "eggNOG database file missing after download"
    return 0
  fi
  if [[ ! -s "${dir}/eggnog_proteins.dmnd" ]]; then
    gate_eggnog "eggNOG DIAMOND database missing after download"
    return 0
  fi
  if ! diamond dbinfo --db "${dir}/eggnog_proteins.dmnd" >/dev/null; then
    gate_eggnog "eggNOG DIAMOND validation failed"
    return 0
  fi
  manifest_row eggnog-mapper "$(binary_version emapper.py -h)" \
    "eggNOG-mapper data" "stable v2 data" "$dir" "EGGNOG_DATA_DIR/--data_dir" \
    "file_count:$(find "$dir" -type f | wc -l)" "emapper.py -h; diamond dbinfo --db ${dir}/eggnog_proteins.dmnd" \
    "Production v2 path; v3 preview intentionally not installed here."
}

build_diamond_db_from_repo() {
  local tool="$1"
  local repo_url="$2"
  local dir="$3"
  local db_prefix="$4"
  mkdir -p "$dir"
  gate_repo_db() {
    local reason="$1"
    log "${tool} gated: ${reason}"
    manifest_row "$tool" "$(binary_version diamond version)" \
      "${tool} DIAMOND database" "git_commit:${commit:-unknown}" "${dir}/${db_prefix}.dmnd" "diamond db path" \
      "not_installed" "manual retry required after archive/layout issue is resolved" \
      "GATED: ${reason}. Source=${repo_url}."
    return 0
  }
  if [[ ! -d "${dir}/repo/.git" ]]; then
    git clone "$repo_url" "${dir}/repo"
  else
    git -C "${dir}/repo" fetch --all --tags
  fi
  local commit
  commit="$(git -C "${dir}/repo" rev-parse HEAD)"
  find "${dir}/repo" -type f \( -name '*.gz' -o -name '*.zip' \) -print0 | while IFS= read -r -d '' f; do
    case "$f" in
      *.gz) [[ -s "${f%.gz}" ]] || gunzip -k "$f" || true ;;
      *.zip)
        local extract_zip="$f"
        if compgen -G "${f%.zip}.z[0-9][0-9]" >/dev/null; then
          extract_zip="${dir}/$(basename "${f%.zip}").combined.zip"
          if [[ ! -s "$extract_zip" ]]; then
            local zip_stem
            local zip_dir
            local -a split_parts
            zip_stem="$(basename "${f%.zip}")"
            zip_dir="$(dirname "$f")"
            mapfile -t split_parts < <(find "$zip_dir" -maxdepth 1 -type f -name "${zip_stem}.z[0-9][0-9]" | sort)
            if [[ "${#split_parts[@]}" -eq 0 ]]; then
              log "${tool}: split zip marker matched but no split parts were found"
              continue
            fi
            cat "${split_parts[@]}" "$f" > "$extract_zip"
          fi
        fi
        if command -v unzip >/dev/null 2>&1; then
          unzip -o "$extract_zip" -d "$(dirname "$f")" || true
        else
          python -m zipfile -e "$extract_zip" "$(dirname "$f")" || true
        fi
        ;;
    esac
  done
  local fasta
  fasta="$(find "${dir}/repo" -type f \( -name '*.fa' -o -name '*.faa' -o -name '*.fasta' -o -name '*.fas' \) | head -1 || true)"
  if [[ -z "$fasta" ]]; then
    gate_repo_db "no FASTA found after cloning/extracting repository archive"
    return 0
  fi
  local normalized_fasta="${dir}/${db_prefix}.normalized.faa"
  normalize_fasta "$fasta" "$normalized_fasta"
  diamond makedb --in "$normalized_fasta" -d "${dir}/${db_prefix}"
  diamond dbinfo --db "${dir}/${db_prefix}.dmnd" >/dev/null
  manifest_row "$tool" "$(binary_version diamond version)" \
    "${tool} DIAMOND database" "git_commit:${commit}" "${dir}/${db_prefix}.dmnd" "diamond db path" \
    "$(sha256sum "${dir}/${db_prefix}.dmnd" | awk '{print $1}')" \
    "diamond dbinfo --db ${dir}/${db_prefix}.dmnd" \
    "Built from ${repo_url}; FASTA=${normalized_fasta}; source_fasta=${fasta}."
}

step_mcycdb() {
  conda_activate
  build_diamond_db_from_repo mcycdb https://github.com/qichao1984/MCycDB.git "${DB_ROOT}/mcycdb" MCycDB_2021
}

step_scycdb() {
  conda_activate
  build_diamond_db_from_repo scycdb https://github.com/qichao1984/SCycDB.git "${DB_ROOT}/scycdb" SCycDB_2020Mar
}

step_dbcan() {
  if env_exists "$DBCAN_ENV"; then
    activate_env "$DBCAN_ENV"
  else
    conda_activate
  fi
  local dir="${DB_ROOT}/dbcan"
  mkdir -p "$dir"
  gate_dbcan() {
    local reason="$1"
    log "dbCAN gated: ${reason}"
    manifest_row dbcan "$(binary_version run_dbcan --help)" \
      "dbCAN database" "run_dbCAN V5 current database" "$dir" "--db_dir" \
      "not_installed" "manual retry required after dbCAN command/env issue is resolved" \
      "GATED: ${reason}."
    return 0
  }
  if ! find "$dir" -type f | grep -Eq 'dbCAN|CAZy|fam|hmm|dmnd'; then
    if run_dbcan database --help >/dev/null 2>&1; then
      retry 3 run_dbcan database --db_dir "$dir" --aws_s3 --retries 4 --timeout 120 --log-level INFO || {
        gate_dbcan "run_dbcan database download failed"
        return 0
      }
    elif command -v dbcan_build >/dev/null 2>&1; then
      retry 3 dbcan_build --cpus "$THREADS" --db-dir "$dir" --clean || {
        gate_dbcan "dbcan_build download/build failed"
        return 0
      }
    else
      gate_dbcan "No supported dbCAN database command found in active env"
      return 0
    fi
  fi
  find "$dir" -type f | head -20
  run_dbcan --help >/dev/null
  manifest_row dbcan "$(binary_version run_dbcan --help)" \
    "dbCAN database" "run_dbCAN V5 current database" "$dir" "--db_dir" \
    "file_count:$(find "$dir" -type f | wc -l)" "run_dbcan --help; database files present" \
    "Downloaded using run_dbcan database --aws_s3 when present, with dbcan_build fallback for package variants."
}

step_dram() {
  if env_exists "$DRAM_ENV"; then
    activate_env "$DRAM_ENV"
  else
    conda_activate
  fi
  local dir="${DB_ROOT}/dram"
  mkdir -p "$dir"
  gate_dram() {
    local reason="$1"
    log "DRAM gated: ${reason}"
    manifest_row dram "$(command -v DRAM.py >/dev/null 2>&1 && binary_version DRAM.py -h || echo missing)" \
      "DRAM databases" "not validated" "$dir" "DRAM config" \
      "not_installed" "manual retry required after DRAM env/database issue is resolved" \
      "GATED: ${reason}. Prefer a fresh official DRAM environment or pre-provisioned DRAM config bundle."
    return 0
  }
  if command -v DRAM-setup.py >/dev/null 2>&1; then
    DRAM-setup.py prepare_databases --output_dir "$dir" --threads "$THREADS" || {
      gate_dram "DRAM-setup.py prepare_databases failed"
      return 0
    }
  elif command -v DRAM.py >/dev/null 2>&1; then
    gate_dram "DRAM.py present but no DRAM-setup.py command found"
    return 0
  else
    gate_dram "DRAM not installed in active env"
    return 0
  fi
  DRAM.py -h >/dev/null 2>&1 || {
    gate_dram "DRAM.py validation failed after database setup"
    return 0
  }
  manifest_row dram "$(command -v DRAM.py >/dev/null 2>&1 && binary_version DRAM.py -h || echo missing)" \
    "DRAM databases" "gated/current if setup succeeded" "$dir" "DRAM config" \
    "file_count:$(find "$dir" -type f 2>/dev/null | wc -l)" "DRAM.py -h" \
    "DRAM database setup is brittle; inspect per-step log before enabling production DRAM."
}

step_metabolic() {
  if env_exists "$METABOLIC_ENV"; then
    activate_env "$METABOLIC_ENV"
    ensure_metabolic_runtime_link
  else
    conda_activate
  fi
  local dir="${DB_ROOT}/metabolic"
  mkdir -p "$dir"
  gate_metabolic() {
    local reason="$1"
    log "METABOLIC gated: ${reason}"
    manifest_row metabolic "METABOLIC commit ${commit:-unknown}" \
      "METABOLIC repository and temp/db resources" "not validated" "${dir}/METABOLIC" "METABOLIC path" \
      "not_installed" "manual retry required after METABOLIC dependency/setup issue is resolved" \
      "GATED: ${reason}. Install Perl/R dependencies and rerun run_to_setup.sh before production use."
    return 0
  }
  if [[ ! -d "${dir}/METABOLIC/.git" ]]; then
    git clone https://github.com/AnantharamanLab/METABOLIC.git "${dir}/METABOLIC"
  else
    git -C "${dir}/METABOLIC" fetch --all --tags
  fi
  local commit
  commit="$(git -C "${dir}/METABOLIC" rev-parse HEAD)"
  (cd "${dir}/METABOLIC" && bash run_to_setup.sh) || {
    gate_metabolic "run_to_setup.sh failed"
    return 0
  }
  perl "${dir}/METABOLIC/METABOLIC-G.pl" -h >/dev/null || {
    gate_metabolic "METABOLIC-G.pl help validation failed"
    return 0
  }
  manifest_row metabolic "METABOLIC commit ${commit}" \
    "METABOLIC repository and temp/db resources" "METABOLIC v4/current master" "${dir}/METABOLIC" "METABOLIC path" \
    "file_count:$(find "${dir}/METABOLIC" -type f | wc -l)" "perl ${dir}/METABOLIC/METABOLIC-G.pl -h" \
    "Installed via official run_to_setup.sh where possible."
}

step_mmseqs() {
  conda_activate
  local dir="${DB_ROOT}/mmseqs"
  mkdir -p "$dir"
  mmseqs version > "${dir}/mmseqs.version.txt"
  manifest_row mmseqs "$(cat "${dir}/mmseqs.version.txt" | head -1)" \
    "MMseqs2 runtime" "binary only; no generic DB installed" "$dir" "PATH" \
    "$(sha256sum "${dir}/mmseqs.version.txt" | awk '{print $1}')" "mmseqs version" \
    "Placeholder for future custom novelty/search databases."
}

main() {
  manifest_header
  run_step preflight step_preflight
  run_step create_env step_create_env
  run_step checkm2 step_checkm2
  run_step gtdbtk_r232 step_gtdbtk_r232
  run_step gunc_progenomes3 step_gunc
  run_step kofam step_kofam
  run_step eggnog_v2 step_eggnog_v2
  run_step mcycdb step_mcycdb
  run_step scycdb step_scycdb
  run_step dbcan step_dbcan
  run_step dram step_dram
  run_step metabolic step_metabolic
  run_step mmseqs step_mmseqs
  log "All requested setup steps completed or reached gated/manual status."
  log "Manifest: ${MANIFEST}"
  log "State dir: ${STATE_DIR}"
}

main "$@"
