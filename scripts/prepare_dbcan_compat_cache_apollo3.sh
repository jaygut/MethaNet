#!/usr/bin/env bash
set -Eeuo pipefail

DB_ROOT="${DB_ROOT:-/home/rsg-jcorre38/scratch/methanet_db}"
DBCAN_SOURCE_DIR="${DBCAN_SOURCE_DIR:-${DB_ROOT}/dbcan}"
DBCAN_COMPAT_DIR="${DBCAN_COMPAT_DIR:-${DB_ROOT}/dbcan_compat_pressed}"
CONDA_SH="${CONDA_SH:-/opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-methanet-dbcan}"
LOCK_DIR="${DBCAN_COMPAT_DIR}.lock"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-21600}"

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

cache_ready() {
  [[ -s "${DBCAN_COMPAT_DIR}/dbCAN.hmm.h3p" ]] &&
    [[ -s "${DBCAN_COMPAT_DIR}/dbCAN.hmm.h3m" ]] &&
    [[ -s "${DBCAN_COMPAT_DIR}/dbCAN.hmm.h3f" ]] &&
    [[ -s "${DBCAN_COMPAT_DIR}/dbCAN.hmm.h3i" ]] &&
    [[ -s "${DBCAN_COMPAT_DIR}/dbCAN_sub.hmm.h3p" ]] &&
    [[ -s "${DBCAN_COMPAT_DIR}/dbCAN_sub.hmm.h3m" ]] &&
    [[ -s "${DBCAN_COMPAT_DIR}/dbCAN_sub.hmm.h3f" ]] &&
    [[ -s "${DBCAN_COMPAT_DIR}/dbCAN_sub.hmm.h3i" ]]
}

require_dir "$DBCAN_SOURCE_DIR"
require_file "${DBCAN_SOURCE_DIR}/dbCAN.hmm"
require_file "${DBCAN_SOURCE_DIR}/dbCAN-sub.hmm"
require_file "$CONDA_SH"

if cache_ready; then
  echo "dbCAN compatibility cache already ready: ${DBCAN_COMPAT_DIR}"
  exit 0
fi

waited=0
while ! mkdir "$LOCK_DIR" 2>/dev/null; do
  if cache_ready; then
    echo "dbCAN compatibility cache became ready: ${DBCAN_COMPAT_DIR}"
    exit 0
  fi
  if [[ "$waited" -ge "$MAX_WAIT_SECONDS" ]]; then
    die "Timed out waiting for dbCAN compatibility cache lock: ${LOCK_DIR}"
  fi
  echo "Waiting for dbCAN cache builder lock (${waited}s): ${LOCK_DIR}"
  sleep "$WAIT_SECONDS"
  waited=$((waited + WAIT_SECONDS))
done

cleanup_lock() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup_lock EXIT

mkdir -p "$DBCAN_COMPAT_DIR"
for dbcan_file in "${DBCAN_SOURCE_DIR}/"*; do
  ln -sf "$dbcan_file" "${DBCAN_COMPAT_DIR}/$(basename "$dbcan_file")"
done
ln -sf "${DBCAN_SOURCE_DIR}/dbCAN-sub.hmm" "${DBCAN_COMPAT_DIR}/dbCAN_sub.hmm"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

HMMPRESS_BIN="$(command -v hmmpress)"
"$HMMPRESS_BIN" -f "${DBCAN_COMPAT_DIR}/dbCAN.hmm"
"$HMMPRESS_BIN" -f "${DBCAN_COMPAT_DIR}/dbCAN_sub.hmm"

cache_ready || die "dbCAN compatibility cache was not complete after hmmpress"
echo "dbCAN compatibility cache ready: ${DBCAN_COMPAT_DIR}"
