#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

PFAM_RELEASE=${PFAM_RELEASE:-"37.0"}
TIGRFAM_RELEASE=${TIGRFAM_RELEASE:-"15.0"}

PYTHON_BIN=${PYTHON_BIN:-python3}
MMSEQS_THREADS=${MMSEQS_THREADS:-8}

REF_DIR="${ROOT_DIR}/data/reference"
TIGR_DIR="${REF_DIR}/tigrfams/${TIGRFAM_RELEASE}"
PFAM_DIR="${REF_DIR}/pfam/${PFAM_RELEASE}"

DB_DIR="${ROOT_DIR}/db"
TMP_DIR="${ROOT_DIR}/.cache/marker_db_core"

OUT_FASTA_CORE="${DB_DIR}/marker_db_core.faa"
OUT_FASTA="${DB_DIR}/marker_db.faa"
OUT_MANIFEST="${DB_DIR}/marker_db_core.manifest.tsv"
OUT_MMSEQS_DB="${DB_DIR}/marker_db.mmsdb"

mkdir -p "${TIGR_DIR}" "${PFAM_DIR}" "${DB_DIR}" "${TMP_DIR}"

log() {
  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command '$1' on PATH"
}

download_url() {
  local url=$1
  local dest=$2

  if [[ -f "${dest}" ]]; then
    return 0
  fi

  log "Downloading: ${url}"
  if command -v wget >/dev/null 2>&1; then
    wget -c "${url}" -O "${dest}"
  elif command -v curl >/dev/null 2>&1; then
    curl -L -C - -o "${dest}" "${url}"
  else
    die "Need wget or curl to download reference files"
  fi
}

stockholm_to_fasta() {
  local in_sto=$1
  local out_fa=$2

  "${PYTHON_BIN}" - "$in_sto" "$out_fa" <<'PY'
import gzip
import sys
from pathlib import Path

in_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

# Minimal Stockholm parser: concatenates alignment lines by seqid, then ungaps.
seqs: dict[str, list[str]] = {}

def open_maybe_gz(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")

with open_maybe_gz(in_path) as handle:
    for raw in handle:
        line = raw.rstrip("\n")
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.strip() == "//":
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        seq_id, aln = parts[0], parts[1]
        seqs.setdefault(seq_id, []).append(aln)

if not seqs:
    raise SystemExit(f"No sequences parsed from Stockholm: {in_path}")

with out_path.open("w", encoding="utf-8") as out:
    for seq_id in sorted(seqs):
        aln = "".join(seqs[seq_id])
        ungapped = aln.replace("-", "").replace(".", "")
        if not ungapped:
            continue
        out.write(f">{seq_id}\n")
        for i in range(0, len(ungapped), 80):
            out.write(ungapped[i:i+80] + "\n")
PY
}

extract_pfam_family() {
  local pfam_seed_gz=$1
  local pfam_acc=$2
  local out_sto=$3

  "${PYTHON_BIN}" - "$pfam_seed_gz" "$pfam_acc" "$out_sto" <<'PY'
import gzip
import sys
from pathlib import Path

seed_path = Path(sys.argv[1])
acc = sys.argv[2]
out_path = Path(sys.argv[3])

in_family = False
buf: list[str] = []

with gzip.open(seed_path, "rt", encoding="utf-8", errors="replace") as handle:
    for line in handle:
        if line.startswith("#=GF AC"):
            # e.g. "#=GF AC   PF12176.7"
            value = line.split(None, 2)[2].strip() if len(line.split(None, 2)) >= 3 else ""
            if value.startswith(acc + ".") or value == acc:
                in_family = True
                buf.append(line)
                continue
            in_family = False

        if in_family:
            buf.append(line)
            if line.strip() == "//":
                break

if not buf:
    raise SystemExit(f"Could not locate Pfam family {acc} in {seed_path}")

out_path.write_text("".join(buf), encoding="utf-8")
PY
}

find_tigr_seed_file() {
  local seed_root=$1
  local tigr_acc=$2

  local direct
  for direct in "${seed_root}/${tigr_acc}.seed" "${seed_root}/${tigr_acc}.SEED" "${seed_root}/${tigr_acc}.stk" "${seed_root}/${tigr_acc}.sto"; do
    if [[ -f "${direct}" ]]; then
      echo "${direct}"
      return 0
    fi
  done

  local found
  found=$(grep -RIlE "^#=GF AC\s+${tigr_acc}(\.|\s|$)" "${seed_root}" 2>/dev/null | head -n 1 || true)
  if [[ -n "${found}" ]]; then
    echo "${found}"
    return 0
  fi

  found=$(find "${seed_root}" -type f -name "*${tigr_acc}*" 2>/dev/null | head -n 1 || true)
  if [[ -n "${found}" ]]; then
    echo "${found}"
    return 0
  fi

  return 1
}

build_core_fasta() {
  : >"${OUT_FASTA_CORE}"
  echo -e "marker\tsource\taccession\tseed_file\tn_seqs" >"${OUT_MANIFEST}"

  # TIGRFAM core marker families (release pinned)
  declare -A TIGR_MARKERS=(
    [mcrA]=TIGR03256
    [mcrB]=TIGR03258
    [mcrG]=TIGR03259
    [pmoA]=TIGR03080
    [mmoX]=TIGR01691
    [dsrA]=TIGR02064
    [dsrB]=TIGR02066
    [mttB]=TIGR02512
    [mtbA]=TIGR02506
    [nifH]=TIGR01287
    [cbbL]=TIGR01168
  )

  local tigr_seed_tar="${TIGR_DIR}/TIGRFAMs_${TIGRFAM_RELEASE}_SEED.tar.gz"
  local tigr_url="https://ftp.ncbi.nlm.nih.gov/hmm/TIGRFAMs/release_${TIGRFAM_RELEASE}/TIGRFAMs_${TIGRFAM_RELEASE}_SEED.tar.gz"
  download_url "$tigr_url" "$tigr_seed_tar"

  local tigr_extract_dir="${TMP_DIR}/tigrfams_seed_${TIGRFAM_RELEASE}"
  if [[ ! -d "${tigr_extract_dir}" ]]; then
    log "Extracting TIGRFAMs SEED to ${tigr_extract_dir}"
    mkdir -p "${tigr_extract_dir}"
    tar -xzf "${tigr_seed_tar}" -C "${tigr_extract_dir}"
  fi

  local marker acc seed_file sto_tmp fa_tmp n
  for marker in "${!TIGR_MARKERS[@]}"; do
    acc="${TIGR_MARKERS[$marker]}"
    seed_file=$(find_tigr_seed_file "${tigr_extract_dir}" "$acc" || true)
    if [[ -z "${seed_file}" ]]; then
      die "Could not find TIGRFAM seed file for ${marker} (${acc}) inside ${tigr_extract_dir}"
    fi

    sto_tmp="${TMP_DIR}/${marker}.${acc}.sto"
    fa_tmp="${TMP_DIR}/${marker}.${acc}.fa"

    # Some TIGRFAM seed files may contain multiple models; extract only the matching AC block when possible.
    if grep -qE "^#=GF AC\s+${acc}(\.|\s|$)" "${seed_file}" 2>/dev/null; then
      "${PYTHON_BIN}" - "$seed_file" "$acc" "$sto_tmp" <<'PY'
import sys
from pathlib import Path

in_path = Path(sys.argv[1])
acc = sys.argv[2]
out_path = Path(sys.argv[3])

in_family = False
buf = []

with in_path.open("r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
        if line.startswith("#=GF AC"):
            value = line.split(None, 2)[2].strip() if len(line.split(None, 2)) >= 3 else ""
            if value.startswith(acc + ".") or value == acc:
                in_family = True
                buf.append(line)
                continue
            in_family = False
        if in_family:
            buf.append(line)
            if line.strip() == "//":
                break

if not buf:
    # fallback: write full file
    out_path.write_text(in_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
else:
    out_path.write_text("".join(buf), encoding="utf-8")
PY
    else
      cp "${seed_file}" "${sto_tmp}"
    fi

    stockholm_to_fasta "${sto_tmp}" "${fa_tmp}"

    n=$(grep -c '^>' "${fa_tmp}" || true)
    if [[ "${n}" -eq 0 ]]; then
      die "No sequences extracted for ${marker} (${acc}) from ${seed_file}"
    fi

    awk -v prefix="${marker}|${acc}|TIGRFAM${TIGRFAM_RELEASE}|" '/^>/{sub(/^>/, ">" prefix); print; next} {print}' "${fa_tmp}" >>"${OUT_FASTA_CORE}"
    echo -e "${marker}\tTIGRFAM${TIGRFAM_RELEASE}\t${acc}\t${seed_file}\t${n}" >>"${OUT_MANIFEST}"
  done

  # Pfam-only marker family for mtaB
  local pfam_acc="PF12176"
  local pfam_seed="${PFAM_DIR}/Pfam-A.seed.gz"
  local pfam_url="https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam${PFAM_RELEASE}/Pfam-A.seed.gz"
  download_url "${pfam_url}" "${pfam_seed}"

  local pfam_sto="${TMP_DIR}/mtaB.${pfam_acc}.sto"
  local pfam_fa="${TMP_DIR}/mtaB.${pfam_acc}.fa"

  extract_pfam_family "${pfam_seed}" "${pfam_acc}" "${pfam_sto}"
  stockholm_to_fasta "${pfam_sto}" "${pfam_fa}"

  n=$(grep -c '^>' "${pfam_fa}" || true)
  if [[ "${n}" -eq 0 ]]; then
    die "No sequences extracted for mtaB (${pfam_acc}) from Pfam-A.seed.gz"
  fi

  awk -v prefix="mtaB|${pfam_acc}|PFAM${PFAM_RELEASE}|" '/^>/{sub(/^>/, ">" prefix); print; next} {print}' "${pfam_fa}" >>"${OUT_FASTA_CORE}"
  echo -e "mtaB\tPFAM${PFAM_RELEASE}\t${pfam_acc}\t${pfam_seed}\t${n}" >>"${OUT_MANIFEST}"
}

build_mmseqs_db() {
  need_cmd mmseqs

  if [[ ! -f "${OUT_FASTA_CORE}" ]]; then
    die "Missing core FASTA: ${OUT_FASTA_CORE}"
  fi

  log "Writing ${OUT_FASTA} (currently core-only; append your curated layer later if desired)"
  cp "${OUT_FASTA_CORE}" "${OUT_FASTA}"

  log "Building MMseqs2 DB: ${OUT_MMSEQS_DB}"
  mmseqs createdb "${OUT_FASTA}" "${OUT_MMSEQS_DB}"

  log "Indexing MMseqs2 DB"
  mkdir -p "${DB_DIR}/tmp"
  mmseqs createindex "${OUT_MMSEQS_DB}" "${DB_DIR}/tmp" --threads "${MMSEQS_THREADS}"
}

main() {
  need_cmd tar
  need_cmd gzip
  need_cmd "${PYTHON_BIN}"

  log "Building core marker FASTA: ${OUT_FASTA_CORE}"
  build_core_fasta

  log "Core FASTA sequences: $(grep -c '^>' "${OUT_FASTA_CORE}" || true)"
  log "Core manifest written: ${OUT_MANIFEST}"

  if [[ "${SKIP_MMSEQS_BUILD:-0}" == "1" ]]; then
    log "SKIP_MMSEQS_BUILD=1 set; skipping mmseqs DB build"
    return 0
  fi

  build_mmseqs_db

  log "Done. MMseqs DB ready: ${OUT_MMSEQS_DB}"
}

main "$@"
