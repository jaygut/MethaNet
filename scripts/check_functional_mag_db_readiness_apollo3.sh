#!/usr/bin/env bash
# Report Apollo-3 functional MAG database readiness as TSV.

set -Eeuo pipefail

DB_ROOT="${DB_ROOT:-/home/rsg-jcorre38/scratch/methanet_db}"
CONDA_SH="${CONDA_SH:-/opt/ohpc/pub/apps/miniconda3/etc/profile.d/conda.sh}"

status_for_file() {
  local path="$1"
  [[ -s "$path" ]] && printf 'ready' || printf 'missing'
}

status_for_dir() {
  local path="$1"
  [[ -d "$path" ]] && printf 'ready' || printf 'missing'
}

env_has_cmd() {
  local env_name="$1"
  local cmd="$2"
  if [[ -r "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda run -n "$env_name" "$cmd" --help >/dev/null 2>&1
  else
    return 1
  fi
}

print_row() {
  printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5"
}

printf 'layer\tstatus\tpath_or_env\tvalidation_hint\tnotes\n'
print_row checkm2 "$(status_for_file "$DB_ROOT/checkm2/CheckM2_database/uniref100.KO.1.dmnd")" \
  "$DB_ROOT/checkm2/CheckM2_database/uniref100.KO.1.dmnd" \
  "diamond dbinfo --db PATH" "MAG completeness/contamination"
print_row gtdbtk_r232 "$(status_for_dir "$DB_ROOT/gtdbtk_r232/release232")" \
  "$DB_ROOT/gtdbtk_r232/release232" \
  "GTDBTK_DATA_PATH=PATH gtdbtk check_install" "taxonomy; path must be release232"
print_row gunc "$(status_for_file "$DB_ROOT/gunc/gunc_db_progenomes3.dmnd")" \
  "$DB_ROOT/gunc/gunc_db_progenomes3.dmnd" \
  "diamond dbinfo --db PATH" "chimerism/artifact screen"
print_row kofam "$(status_for_file "$DB_ROOT/kofam/profiles/prokaryote.hal")" \
  "$DB_ROOT/kofam" \
  "exec_annotation --profile profiles/prokaryote.hal --ko-list ko_list -h" "KO/module layer"
print_row mcycdb "$(status_for_file "$DB_ROOT/mcycdb/MCycDB_2021.dmnd")" \
  "$DB_ROOT/mcycdb/MCycDB_2021.dmnd" \
  "diamond dbinfo --db PATH" "curated methane cycling markers"
print_row scycdb "$(status_for_file "$DB_ROOT/scycdb/SCycDB_2020Mar.dmnd")" \
  "$DB_ROOT/scycdb/SCycDB_2020Mar.dmnd" \
  "diamond dbinfo --db PATH" "curated sulfur cycling markers"
print_row dbcan "$(status_for_file "$DB_ROOT/dbcan/CAZy.dmnd")" \
  "$DB_ROOT/dbcan" \
  "run_dbcan --help; diamond dbinfo --db CAZy.dmnd" "CAZyme/CGC/substrate layer"
print_row metabolic "$(status_for_file "$DB_ROOT/metabolic/METABOLIC/METABOLIC-G.pl")" \
  "$DB_ROOT/metabolic/METABOLIC" \
  "perl METABOLIC-G.pl -h" "production biogeochemical distillation fallback"
print_row bakta_light "$(status_for_file "$DB_ROOT/bakta/db-light/bakta.db")" \
  "$DB_ROOT/bakta/db-light" \
  "bakta --db PATH --help" "optional standardized MAG annotation add-on"
print_row eggnog_v2 "$(status_for_file "$DB_ROOT/eggnog_v2/eggnog.db")" \
  "$DB_ROOT/eggnog_v2" \
  "emapper.py --data_dir PATH; diamond dbinfo --db eggnog_proteins.dmnd" \
  "gated until full files are staged from a complete-transfer network/mirror"
print_row dram "$(status_for_file "$DB_ROOT/dram/CONFIG")" \
  "$DB_ROOT/dram" \
  "DRAM.py -h; DRAM-setup.py print_config" "gated; use METABOLIC/Bakta unless fresh DRAM/DRAM2 is provisioned"

if env_has_cmd methanet-bakta bakta; then
  print_row bakta_env ready methanet-bakta "bakta --version" "Bakta 1.12.0 installed"
else
  print_row bakta_env missing methanet-bakta "conda create -n methanet-bakta ... bakta=1.12.0" "optional add-on env"
fi
