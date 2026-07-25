#!/usr/bin/env bash
# =====================================================================
# publish_site.sh assembles the public GitHub Pages site for
# EmergentBiome / MethaNet, and (optionally) deploy it to the gh-pages branch.
#
#   site root  = the scrollytelling landing (this folder)
#   /report/   = the full molecular-attestation report, self-contained for
#                public viewing without raw JSON, TSV, audit, or source-pointer files
#   legacy dated report paths on gh-pages are preserved (no broken links).
#
# Usage:
#   tools/publish_site.sh build                 # assemble ./_site only (safe)
#   tools/publish_site.sh deploy                # build + commit to a gh-pages worktree
#   tools/publish_site.sh deploy --push         # build + commit + push origin gh-pages
#   tools/publish_site.sh build /path/to/report_freeze   # use a specific freeze
# =====================================================================
set -euo pipefail

CMD="${1:-build}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"          # web/emergentbiome-methanet
REPO="$(cd "$HERE/../.." && pwd)"                  # repo root
DEFAULT_REPORT="$REPO/results/reports/mbag_nextgen_molecular_niche_atlas_20260724_scientific_reconciliation"
REPORT="${2:-$DEFAULT_REPORT}"
[[ "$REPORT" == --* ]] && REPORT="$DEFAULT_REPORT"   # allow `deploy --push`
OUT="$HERE/_site"

build() {
  echo "→ assembling site from:"
  echo "    landing : $HERE"
  echo "    report  : $REPORT"
  [[ -f "$REPORT/report.html" ]] || { echo "ERROR: report.html not found in $REPORT"; exit 1; }
  rm -rf "$OUT"; mkdir -p "$OUT"

  # --- landing → site root (only the runtime files; not dev docs/tools) ---
  for item in index.html styles.css main.js config.js lib scenes vendor data assets; do
    cp -R "$HERE/$item" "$OUT/"
  done
  touch "$OUT/.nojekyll"

  # --- custom domain: write CNAME from the source file if present (keeps GitHub Pages
  #     custom-domain binding across republishes; harmless no-op when absent) ---
  if [[ -f "$HERE/CNAME" ]]; then cp "$HERE/CNAME" "$OUT/CNAME"; echo "    CNAME   : $(cat "$HERE/CNAME")"; fi

  # --- report → /report/ (render-complete and self-contained) ---
  # The report embeds its interactive evidence payload. Detailed tables and audit
  # products remain in the internal warehouse and are deliberately excluded from
  # the public site.
  mkdir -p "$OUT/report/assets/js" "$OUT/report/assets/figures"
  cp "$REPORT/report.html"                  "$OUT/report/index.html"
  cp "$REPORT/assets/js/d3.v7.min.js"       "$OUT/report/assets/js/"
  cp "$REPORT"/assets/figures/*.png         "$OUT/report/assets/figures/"

  echo "✓ built $OUT  ($(du -sh "$OUT" | cut -f1) total; report/ = $(du -sh "$OUT/report" | cut -f1))"
}

deploy() {
  build
  git -C "$REPO" worktree prune
  # reuse an existing gh-pages worktree (the project's publish setup) if present
  local WT TEMP=0
  WT="$(git -C "$REPO" worktree list --porcelain | awk '/^worktree /{w=substr($0,10)} /^branch refs\/heads\/gh-pages$/{print w; exit}')"
  if [[ -z "$WT" ]]; then
    WT="/tmp/eb_ghpages_worktree"; TEMP=1
    rm -rf "$WT"; git -C "$REPO" worktree add --quiet "$WT" gh-pages
  fi
  # safety: never clobber uncommitted work in the worktree
  if [[ -n "$(git -C "$WT" status --porcelain)" ]]; then
    echo "ERROR: gh-pages worktree $WT has uncommitted changes; aborting (commit/stash first)."; exit 1
  fi
  git -C "$WT" fetch origin gh-pages --quiet || true
  git -C "$WT" reset --hard origin/gh-pages --quiet   # base on the live site
  # replace landing/report at root but PRESERVE dated report dirs, .git, and any
  # existing CNAME (so a custom domain set via the GitHub UI is never wiped)
  ( cd "$WT" && find . -maxdepth 1 -mindepth 1 \
      ! -name '.git' ! -name 'CNAME' ! -name 'mbag_nextgen_molecular_niche_atlas_*' -exec rm -rf {} + )
  cp -R "$OUT"/. "$WT"/
  git -C "$WT" add -A
  if git -C "$WT" diff --cached --quiet; then
    echo "✓ gh-pages already up to date (no changes)"
  else
    git -C "$WT" commit --quiet -m "Publish EmergentBiome/MethaNet landing + render-complete /report/ ($(basename "$REPORT"))"
    echo "✓ committed to gh-pages worktree at $WT"
    if [[ "${1:-}" == "--push" || "${2:-}" == "--push" || "${3:-}" == "--push" ]]; then
      git -C "$WT" push origin gh-pages && echo "✓ pushed origin gh-pages"
    else
      echo "  (not pushed. Re-run with --push, or:  git -C $WT push origin gh-pages )"
    fi
  fi
  [[ $TEMP -eq 1 ]] && git -C "$REPO" worktree remove --force "$WT" 2>/dev/null || true
}

case "$CMD" in
  build)  build ;;
  deploy) deploy "$@" ;;
  *) echo "usage: $0 {build|deploy [--push]} [report_freeze_dir]"; exit 1 ;;
esac
