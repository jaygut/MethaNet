#!/usr/bin/env bash
set -euo pipefail

# Fetch and organize Blue Catalyst artifacts from Apolo into this repo.
#
# Usage:
#   ./scripts/fetch_apolo_blue_catalyst_artifacts.sh --stamp 20260226_194505
#
# Optional env vars:
#   APOLO_USER=rsg-jcorre38
#   APOLO_HOST=apolo-3.eafit.edu.co
#   APOLO_EXPORT_DIR=/home/rsg-jcorre38/Jay_Proyects/MethaNet/results/_export
#   APOLO_PASSWORD=...        # used only if sshpass is installed
#
# Notes:
# - Generated artifacts are stored under results/blue_catalyst_poc/runs/.
# - Script never commits anything.

usage() {
  cat <<'EOF'
Usage:
  fetch_apolo_blue_catalyst_artifacts.sh --stamp <YYYYmmdd_HHMMSS>

Options:
  --stamp   Required package timestamp suffix.
            Example: 20260226_194505 for blue_catalyst_poc_20260226_194505.tar.gz
  -h, --help  Show this help.
EOF
}

STAMP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stamp)
      STAMP="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$STAMP" ]]; then
  echo "ERROR: --stamp is required." >&2
  usage
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

APOLO_USER="${APOLO_USER:-rsg-jcorre38}"
APOLO_HOST="${APOLO_HOST:-apolo-3.eafit.edu.co}"
APOLO_EXPORT_DIR="${APOLO_EXPORT_DIR:-/home/rsg-jcorre38/Jay_Proyects/MethaNet/results/_export}"

PKG="blue_catalyst_poc_${STAMP}.tar.gz"
REMOTE_PKG="${APOLO_EXPORT_DIR}/${PKG}"
REMOTE_SHA="${REMOTE_PKG}.sha256"

RUNS_DIR="$REPO_ROOT/results/blue_catalyst_poc/runs"
RUN_DIR="$RUNS_DIR/apolo_${STAMP}"
mkdir -p "$RUN_DIR"

copy_file() {
  local remote_path="$1"
  local local_dir="$2"

  if command -v sshpass >/dev/null 2>&1 && [[ -n "${APOLO_PASSWORD:-}" ]]; then
    SSHPASS="$APOLO_PASSWORD" sshpass -e scp \
      -o StrictHostKeyChecking=accept-new \
      "${APOLO_USER}@${APOLO_HOST}:${remote_path}" "$local_dir/"
  else
    scp -o StrictHostKeyChecking=accept-new \
      "${APOLO_USER}@${APOLO_HOST}:${remote_path}" "$local_dir/"
  fi
}

echo "Fetching package: $REMOTE_PKG"
copy_file "$REMOTE_PKG" "$RUN_DIR"

echo "Fetching checksum: $REMOTE_SHA"
copy_file "$REMOTE_SHA" "$RUN_DIR"

(
  cd "$RUN_DIR"
  # Remote checksum files may include absolute Apolo paths.
  # Normalize to local package filename for portable verification.
  awk -v pkg="$PKG" 'NF { print $1 "  " pkg }' "${PKG}.sha256" > "${PKG}.sha256.local"
  shasum -a 256 -c "${PKG}.sha256.local"
)

mkdir -p "$RUN_DIR/artifacts"
tar -xzf "$RUN_DIR/$PKG" -C "$RUN_DIR/artifacts"

if [[ -d "$RUN_DIR/artifacts/blue_catalyst_poc" ]]; then
  # Flatten one level for convenience.
  mv "$RUN_DIR/artifacts/blue_catalyst_poc" "$RUN_DIR/artifacts/_tmp"
  rm -rf "$RUN_DIR/artifacts/blue_catalyst_poc"
  rsync -a "$RUN_DIR/artifacts/_tmp/" "$RUN_DIR/artifacts/"
  rm -rf "$RUN_DIR/artifacts/_tmp"
fi

cat > "$RUN_DIR/PROVENANCE.txt" <<EOF
Source host: ${APOLO_HOST}
Remote package: ${REMOTE_PKG}
Pulled at (UTC): $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Checksum file: ${PKG}.sha256
EOF

ln -sfn "$RUN_DIR" "$RUNS_DIR/latest"

echo "Done. Artifacts organized at: $RUN_DIR"
find "$RUN_DIR" -maxdepth 2 -type f | sort
