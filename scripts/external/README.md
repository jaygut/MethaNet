# External MAG Source-Lane Utilities

This directory contains reusable adapters for external MAG catalogues. Dataset
specific parsing, accession reconciliation, and provider quirks should stay in
the dataset provenance area under `data/external/<dataset>/source_docs/`.
Everything here should work for any normalized external MAG lane keyed by
`proteome_id`.

## Where Files Belong

Use this layout for every new external MAG case study:

| Artifact | Location | Git policy |
| --- | --- | --- |
| Source paper spreadsheets, provider JSON, checksums, and accession notes | `data/external/<dataset_id>/source_docs/` | Keep as provenance; large/bulk files stay ignored. |
| Dataset-specific parsing or download helper used during recovery | `data/external/<dataset_id>/source_docs/local_ingestion_scripts/` | Treat as provenance, not reusable pipeline code. |
| Normalized lane manifests, gap registers, and shard manifests | `data/external/<dataset_id>/manifests/` or the downstream run `manifests/` directory | Generated evidence; preserve locally and promote only small reviewed snapshots when needed. |
| Reusable adapters, validators, and launch wrappers | `scripts/external/`, `scripts/reports/`, `scripts/slurm/`, or top-level `scripts/submit_*.sh` | Version in git with tests. |
| Lane registry | `configs/methanet_atlas_lanes.tsv` | Version in git. This is the control plane for reports and status checks, including optional provenance pointers. |
| Functional, ESM2, gLM2, report, and warehouse outputs | `results/...` | Generated/regenerable; do not stage bulk outputs. |

If a new dataset requires a one-off transformation, first try to express it as
a normalized input manifest consumed by the reusable tools below. Keep the
source-specific transform beside the source documents unless the logic is
clearly reusable across future MAG catalogues.

## Standard Flow

1. Build or obtain a normalized source manifest with one row per MAG/proteome
   candidate and source-specific metadata. The minimum useful columns are:

```text
proteome_id
mag_id
source
ecosystem
domain
source_group
source_sample_ids
local_fna_path
mapped_ncbi_biosamples
mapped_ncbi_bioprojects
```

Rows that cannot yet be downloaded or protein-called should not disappear.
Record them in a gap register and preserve the reason as `gap_reason`.

2. Build or refresh the source-document checksum ledger with:

```bash
scripts/external/build_source_provenance_checksums.py \
  --source-dir data/external/<dataset_id>/source_docs \
  --output-tsv data/external/<dataset_id>/source_docs/source_file_checksums.tsv \
  --artifact-prefix <dataset_id> \
  --exclude local_ingestion_scripts
```

The ledger records relative paths, file sizes, MD5, SHA256, and optional
`source_url` values. The lane registry validator verifies these checksums when
`source_provenance_checksums` is registered.

3. Predict or reuse proteins with:

```bash
scripts/external/predict_external_mag_proteomes.py \
  --manifest <source-ready.tsv> \
  --output-manifest <proteome-manifest.tsv> \
  --proteome-dir <proteomes_faa/> \
  --ffn-dir <genes_ffn/> \
  --gff-dir <genes_gff/> \
  --log-dir <logs/prodigal/>
```

The prediction adapter validates the selected rows after any include filtering
or smoke-test limiting. Missing or duplicate `proteome_id` values in the
selected queue abort before Prodigal workers are started, preventing conflicting
FAA/GFF output paths.

4. Build source-lane and functional manifests with:

```bash
scripts/external/build_external_mag_source_lane.py \
  --dataset-id <dataset_id> \
  --input-manifest <source-ready.tsv> \
  --proteome-manifest <proteome-manifest.tsv> \
  --gap-register <gap-register.tsv> \
  --include-gaps \
  --output-source-lane <source-lane.tsv> \
  --output-functional-manifest <functional-manifest.tsv>
```

5. Shard large manifests for GPU or Slurm limits with:

```bash
scripts/external/split_manifest_shards.py \
  --input <source-lane.tsv> \
  --output-dir <shards/> \
  --prefix <dataset> \
  --include-col <esm2_include|glm2_include|functional_run_include> \
  --shards 4
```

When the configured ID column is present, the splitter validates the selected
rows after include filtering. Missing or duplicate IDs abort before any shard
manifest is written.

6. Launch ESM2 shards through the reusable submitter, which wraps the standard
   Slurm worker:

```bash
SHARD_MANIFEST=<shards>/<dataset>.shard_manifest.tsv \
OUTPUT_DIR_TEMPLATE='results/blue_catalyst_poc/runs/<dataset>_esm2_shard{shard}/artifacts' \
ESM2_INCLUDE_COL=esm2_include \
DRY_RUN=1 \
scripts/submit_manifest_esm2_shards_apollo3.sh
```

Set `DRY_RUN=0` only after inspecting the rendered `sbatch` commands. The
template tokens `{shard}`, `{shard_int}`, and `{path_stem}` can be used to keep
output directories deterministic and non-overlapping.
The submitter validates the shard manifest header, numeric shard/row fields,
and duplicate shard IDs or paths before printing or submitting commands.

7. Build gLM2 prepared payload directories with
   `scripts/contextual_genomics/build_glm2_smoke_inputs.py`, then submit those
   prepared directories with:

```bash
GLM2_RESULTS_DIRS='results/contextual_genomics/<dataset>_glm2_shard001;results/contextual_genomics/<dataset>_glm2_shard002' \
DRY_RUN=1 \
scripts/submit_glm2_payload_dirs_apollo3.sh
```

Set `DRY_RUN=0` only after inspecting the rendered `sbatch` commands. Each
payload directory must already contain `prepared_inputs/glm2_sequences.jsonl`.
The submitter normalizes the full payload list before rendering commands and
fails on duplicate result directories; unprepared unique payloads are reported
and skipped.

8. Launch functional shards with the existing MethaNet functional worker using
   the source-lane manifest column `functional_run_include`.

The functional submitter preflights the selected queue before rendering
`sbatch`: included rows must have non-empty, unique `proteome_id` values,
non-empty `mag_fasta` and `proteome_faa` fields, and must not be
`match_status=missing_payload`. CRLF line endings from spreadsheet-derived TSVs
are tolerated during this preflight.

9. Register every derived artifact directory in `configs/methanet_atlas_lanes.tsv`.
   For sharded ESM2 or gLM2 runs, separate artifact directories with semicolons.
   Registry/report readers must deduplicate by `proteome_id`; the manifest
   remains the denominator even when resume jobs or shard jobs overlap. Use
   `source_provenance_dir` and `source_provenance_checksums` to point back to
   source documents, accession screenshots, provider metadata, and checksum
   ledgers without moving bulky provenance files into version control.

10. Refresh the registry-driven status bundle with:

```bash
scripts/reports/refresh_atlas_lane_registry_status.sh
```

This writes timestamped TSV, JSON, and Markdown artifacts under
`results/reports/`, after validating the lane registry shape and registered
paths. It preserves pending, partial, failed, duplicate-attempt, and
missing-payload evidence as explicit status. The registry validator also checks
manifest-level invariants such as unique `proteome_id` values, valid boolean
include columns, and non-empty genome/proteome payload fields for rows marked
`functional_run_include=true`. It also rejects registry rows whose
`denominator_units` are smaller than the functional manifest row count or the
functional include count. For external lanes, where the source-lane manifest is
the catalogue denominator, it also rejects rows whose `denominator_units` are
smaller than the source-lane manifest row count and warns when the source-lane
manifest is missing normalized handoff fields such as `mag_id`, `source`,
`ecosystem`, `domain`, `mag_fasta`, `proteome_faa`, `match_status`,
`analysis_unit_type`, `claim_scope`, and `comparability_status`. For lanes
marked `complete`, it warns if registered warehouse, ESM2, or gLM2 artifact
directories lack the expected report-grade marker files; active lanes may keep
placeholder directories while Slurm jobs are still queued or running.

For day-to-day operations, use the readiness wrapper instead of running the
refresher alone:

```bash
PREVIOUS_STATUS_JSON=auto \
STRICT_GATES=0 \
LANE_IDS='<dataset_id>' \
scripts/reports/check_atlas_lane_readiness.sh
```

The wrapper produces a single timestamped handoff bundle: registry status,
gate results, exact-overlap audit when applicable, Slurm queue snapshot,
status delta, and `atlas_lane_completion_checklist_<timestamp>.{json,md}`.
The completion checklist converts the status snapshot into concrete remaining
actions such as functional rows still pending, ESM2/gLM2 rows still missing,
and whether consolidation or expanded-atlas rebuild is currently allowed.

11. Compare two registry status snapshots when monitoring long-running lanes:

```bash
scripts/reports/compare_atlas_lane_status.py \
  --previous-json results/reports/atlas_lane_registry_status_<old>.json \
  --current-json results/reports/atlas_lane_registry_status_<new>.json \
  --output-tsv results/reports/atlas_lane_registry_delta_<new>.tsv \
  --output-json results/reports/atlas_lane_registry_delta_<new>.json \
  --output-md results/reports/atlas_lane_registry_delta_<new>.md
```

The delta report is lane-agnostic. It tracks movement in functional complete,
ESM2, gLM2, tri-view, pending/partial, and failed counts without changing any
source-lane state or making sample/MRV claims.

When only a delta is needed, run the comparator directly. During routine
readiness checks, prefer letting the wrapper emit the delta bundle alongside
gate outputs:

```bash
PREVIOUS_STATUS_JSON=auto \
STRICT_GATES=0 \
LANE_IDS='<dataset_id>' \
scripts/reports/check_atlas_lane_readiness.sh
```

This writes `atlas_lane_registry_delta_<timestamp>.{tsv,json,md}` and records
those paths in the readiness JSON. `PREVIOUS_STATUS_JSON=auto` selects the
latest earlier `atlas_lane_registry_status_*.json` in `results/reports/` while
ignoring `.validation.json` sidecars. Set it to an explicit status JSON path
when comparing against a specific baseline.

For active lanes with retries, lane status is proteome-centric: a completed
attempt wins, otherwise the latest attempt determines whether the row is
`partial` or `failed`. Historical failed attempts remain visible in the attempt
audit, but a newer in-progress retry should not keep the selected proteome in
the failed-row blocker count.

## Futian Phase 1 Worked Example

Futian is the reference external-lane pattern for a large public MAG catalogue:

| Item | Futian path |
| --- | --- |
| Source provenance | `data/external/futian_mangrove_2026_qi/source_docs/` |
| Source checksum ledger | `data/external/futian_mangrove_2026_qi/source_docs/source_file_checksums.tsv` |
| Source-lane manifest | `data/external/futian_mangrove_2026_qi/manifests/futian_phase1_source_lane_manifest.tsv` |
| Explicit gap register | `data/external/futian_mangrove_2026_qi/manifests/futian_phase1_download_gap_register.tsv` |
| Functional manifest | `results/functional_metagenomics/futian_mangrove_2026_phase1/manifests/futian_phase1_functional_mag_manifest.tsv` |
| Registry row | `configs/methanet_atlas_lanes.tsv`, lane `futian_mangrove_2026_qi` |
| Status bundle | `results/reports/atlas_lane_registry_status_<timestamp>.*` |

The important reusable decision is not the Futian accession parsing. It is the
handoff boundary:

1. Source-specific recovery produced local FASTA/proteome payloads and a gap
   register.
2. `build_external_mag_source_lane.py` converted those into a manifest-defined
   lane with ready rows and explicit missing-payload rows.
3. ESM2, gLM2, functional annotation, registry refresh, and reports consumed the
   lane through generic scripts and `configs/methanet_atlas_lanes.tsv`.

Follow this same boundary for the next MAG catalogue. Do not add a new
dataset-specific report path, embedding loader, or functional status counter
unless the reusable lane contract itself is missing a field.

The source-lane builder fails fast when the input manifest, proteome manifest,
gap register, or combined source-lane output would contain missing or duplicate
`proteome_id` values. Fix those identity issues in the dataset-local provenance
step before launching embeddings or functional jobs.

## Consolidation Gates

Do not build a cohort warehouse for an active external lane just because some
per-MAG folders are complete. Consolidation is allowed only when the registry
status bundle shows:

```text
functional_failed_units = 0
functional_pending_units = 0
functional_complete_units = functional_include_rows
consolidation_ready = true
```

Check those conditions with the reusable gate before launching a warehouse
build:

```bash
scripts/reports/check_atlas_lane_consolidation_gate.py \
  --status-json results/reports/atlas_lane_registry_status_<timestamp>.json \
  --lane-id <dataset_id> \
  --print-commands
```

The gate exits non-zero and prints blockers when a lane still has failed,
pending, partial, or incomplete functional rows. When it passes, the printed
command is the registry-derived consolidation command.

For multi-view atlas reporting, the stricter tri-view gate is:

```text
unique ESM2 proteome_ids
AND unique gLM2 proteome_ids
AND unique selected functional proteome_ids
```

all joined against the manifest denominator. Resume jobs and shards may overlap;
registry and report readers must deduplicate by `proteome_id`.

Check final expanded-atlas rebuild readiness with:

```bash
scripts/reports/check_atlas_report_rebuild_gate.py \
  --status-json results/reports/atlas_lane_registry_status_<timestamp>.json \
  --lane-id <dataset_id> \
  --print-command
```

This gate requires the calibration lane warehouse to be current and every
selected external lane to have ESM2, gLM2, functional, and tri-view coverage
equal to its registered `functional_include_rows`.

For normal operator handoff, run the combined readiness wrapper instead of
manually stitching the refresh and gate commands together:

```bash
LANE_IDS='<dataset_id>' \
scripts/reports/check_atlas_lane_readiness.sh
```

`LANE_IDS` accepts comma-, semicolon-, or space-separated lane IDs. Leave it
empty to check all eligible registered lanes. The wrapper refreshes the
registry status, runs the consolidation gate, runs the expanded-atlas report
gate, runs the exact manifest-overlap audit when at least two lanes are
selected, captures a Slurm queue snapshot when `squeue` is available, and
writes Markdown and JSON handoff summaries under `results/reports/`. The JSON
companion records artifact paths, gate return codes, gate output text, overlap
status, the Slurm snapshot, and the claim boundary for automation or notebooks.
Default artifact names use UTC second-resolution timestamps to avoid overwriting
repeated monitoring runs; set `STAMP=<id>` only when an intentional deterministic
artifact name is needed.
By default it exits non-zero while gates are blocked; set `STRICT_GATES=0` for
a monitoring pass that should record blockers without failing the caller. Set
`RUN_OVERLAP_AUDIT=0` to skip overlap auditing, or `RUN_OVERLAP_AUDIT=1` to
force it. Set `INCLUDE_SLURM=0` to skip queue capture outside Apollo/Slurm
contexts.

## Operational Checklist For A New Lane

1. Put source files and source-specific helper scripts under
   `data/external/<dataset_id>/source_docs/`.
2. Build or refresh `source_file_checksums.tsv` with
   `build_source_provenance_checksums.py`.
3. Produce a normalized input manifest and, if needed, a gap register.
4. Run `predict_external_mag_proteomes.py` only for rows needing local protein
   calls.
5. Run `build_external_mag_source_lane.py` with `--include-gaps`.
6. Split source-lane or functional manifests only through
   `split_manifest_shards.py`.
7. Launch ESM2 and gLM2 with dry-run wrappers first; inspect the rendered
   `sbatch` commands before setting `DRY_RUN=0`.
8. Launch functional runs through the existing Apollo-3 functional worker and
   its dependency controls.
9. Add one registry row to `configs/methanet_atlas_lanes.tsv`.
10. Run `scripts/reports/check_atlas_lane_readiness.sh` to refresh registry
    status and execute the consolidation/report gates in one reproducible
    handoff step.
11. If you need a standalone exact manifest-overlap audit before final
    consolidation/reporting, run:

```bash
scripts/reports/audit_atlas_lane_overlap.py \
  --lane-id <dataset_id_a> \
  --lane-id <dataset_id_b> \
  --output-summary-tsv results/reports/<audit_id>.summary.tsv \
  --output-matches-tsv results/reports/<audit_id>.matches.tsv \
  --output-json results/reports/<audit_id>.json \
  --output-md results/reports/<audit_id>.md
```

This is an exact identifier check over registered source-lane manifests. It can
flag reused `proteome_id`, MAG IDs, BioSample/BioProject tokens, WGS accessions,
eLMSG IDs, source filenames, or genome checksums before a combined warehouse is
advertised. It is not an ANI/genome-similarity result; use it as a
deduplication-planning guard, not a novelty claim.

12. Rebuild the expanded atlas only after the status bundle proves the relevant
    lane has new complete evidence.
    `scripts/reports/build_mbag_expanded_multiview_atlas.py` requires
    `configs/methanet_atlas_lanes.tsv` by default, so final rebuilds stay tied
    to the registered lane control plane. The report gate rejects blank or
    duplicate lane IDs, unsupported lane roles, missing calibration or external
    lanes, invalid denominators, and missing registered report input paths
    before any HTML is written. Use `--allow-legacy-defaults` only for
    historical POC/MSM-only rebuilds.
13. Consolidate only after the readiness wrapper or consolidation gate prints a
    registry-derived consolidation command for the lane.

## Contract

- `proteome_id` is the canonical key.
- The manifest defines the denominator; successful output folders do not.
- Registered provenance pointers make source documents discoverable, but source
  documents and bulk downloads remain under `data/external/<dataset_id>/` and
  are not reusable pipeline code.
- Gap rows are preserved with `match_status=missing_payload` rather than
  dropped.
- Multiple ESM2/gLM2 artifact directories are allowed, but lane summaries and
  report builders must count unique `proteome_id` values rather than summing
  per-directory rows.
- Cross-lane overlap checks are exact manifest audits. They should be run
  before final multi-lane reporting, but they do not replace ANI or genome
  similarity analysis when biological nonredundancy is the claim.
- MAG/proteome-level evidence must not be promoted to sample-level or MRV risk
  claims without sample mapping, abundance/read coverage, environmental
  covariates, uncertainty propagation, and validation evidence.
