# Pipeline Reproducibility Contract

Date: 2026-06-13
Documentation refresh: 2026-07-24

Scope: MethaNet functional-metagenomics code, documentation, generated outputs,
and downstream analytics for the 662-proteome POC cohort plus the
mangrove/MSM and mangrove/Futian expansion lanes.

## Purpose

This contract defines what must remain true for the functional atlas to be
auditable, maintainable, and scientifically defensible while production jobs and
analysis scripts evolve.

## Non-Negotiable Invariants

1. `proteome_id` is the canonical cohort key.
2. The manifest for the active lane defines the denominator; downstream tools
   never define a cohort by successful output alone.
3. The 662-row cohort backbone defines the POC identity denominator. The
   625-row MAG-bin subset defines the current POC functional denominator. The
   37 assembly-context units remain explicit but quarantined from MAG-bin
   feature tables unless requested.
4. The mangrove/MSM local processing denominator is 1,428 candidates until a
   reconciled source-paper denominator is applied. The source paper's 966 final
   medium/high-quality MAG count must remain a separate denominator field, not a
   silent replacement.
5. The mangrove/Futian phase-1 source-lane denominator is 3,404 rMAGs, with a
   distinct 3,156-ready payload denominator and a 248-row gap register. These
   denominators must remain visible until missing payloads and depth-resolved
   sample mapping are reconciled.
6. Cohort integration uses left joins onto the backbone. Failures, missing
   annotations, and unresolved metadata become explicit status fields.
7. Per-MAG run folders are evidence bundles. The cohort warehouse is a derived,
   regenerable analytical layer.
8. Every cohort table must preserve `cohort_run_id`, `run_id`, `proteome_id`,
   `mag_id`, and `source_tool` unless the table is explicitly outside the
   functional atlas model.
9. Missing pathway evidence must never be interpreted without MAG QC and
   annotation coverage.
10. MAG-level functional potential must not be described as sample-level ecology
   unless sample/metagenome metadata and abundance or coverage weights are
   joined.
11. Current source design does not support source-independent methane MRV
   transfer claims without additional source-aware validation.

## Versioned Versus Generated Artifacts

| class | examples | git policy |
| --- | --- | --- |
| Source code | scripts, validators, parsers, plotting utilities | Version in git. |
| Contracts and runbooks | `ai_docs/functional_metagenomics_expansion/*.md` | Version in git. |
| Templates | schemas, manifest templates, example config skeletons | Version in git. |
| Generated run outputs | per-MAG folders, Parquet shards, DuckDB files, logs, figures | Keep under ignored `results/` unless intentionally promoted. |
| Raw public metadata caches | ENA, Zenodo, NCBI API/download responses | Keep under ignored `results/`; record source URLs and checksums in reviewed summaries when needed. |
| Reviewed audit snapshots | small validation reports, frozen counts, claim-boundary summaries | Promote to `ai_docs/` only when dated and explicitly labeled as a snapshot. |

## Required Run Evidence

Every production or calibration run should be reconstructable from:

- the cohort manifest or crosswalk used for launch,
- `cohort_run_id`,
- per-MAG `run_id`,
- locked tool/database paths and versions,
- per-MAG `curated/run_record.json`,
- per-MAG `curated/file_manifest.tsv`,
- per-MAG `curated/parquet_manifest.tsv`,
- cohort-level `cohort_table_manifest.tsv`,
- cohort-level `validation_gates.tsv`, and
- the final warehouse/report generation command.

Every mixed-lane MBAG report should additionally record:

- the `configs/methanet_atlas_lanes.tsv` revision used;
- the lane-registry status JSON/TSV generated immediately before reporting;
- the three-view freeze manifest and `freeze_decision.json` when a release or
  external handoff snapshot is being made;
- the explicit policy for partial, failed, duplicate, not-started, and gap rows.

If any of these are absent, the run can still be useful, but it is not fully
reproducible until the gap is documented.

Current reproducibility note, 2026-07-24: the scientific-reconciliation release
is backed by
`results/reports/methanet_3view_payload_freeze_20260724_scientific_reconciliation/`
and the matching MBAG report bundle. It registers 7,965 units and carries 7,484
data-complete tri-views across three explicit functional contracts. Every new
denominator requires a fresh lane-registry snapshot, freeze manifest, report
bundle, and validation-gate pass.

## Metadata Contract

Environmental context must carry a resolution tier. Use exact sample or
BioSample fields for environmental modeling where available. Treat MUCC source
bucket, site, project, publication, and sample-design context as provenance
unless a sample-level join is proven.

Minimum columns for a metadata crosswalk:

```text
proteome_id
mag_id_candidate
source
ecosystem
metadata_resolution
sample_or_biosample_accession
environment_context_primary
environment_material_primary
geo_context_primary
metadata_source_urls_or_accessions
metadata_caveat
```

Rows with source/site/project-only context should have an explicit caveat such
as:

```text
source_or_site_level_context_not_sample_level
```

## Documentation Freshness Rules

- Architecture docs describe stable design, contracts, and gates.
- Status reports describe dated observations and may become stale quickly.
- During active Slurm runs, do not edit architecture docs to contain live queue
  counts unless the counts are labeled as a dated snapshot.
- After a tranche completes, update the relevant status document or add a dated
  snapshot rather than overwriting historical deployment rationale.
- If a document contains historical launch language, add a freshness note rather
  than silently pretending it is live scheduler state.

## Pre-Commit Checklist

Before committing pipeline changes:

1. Confirm no generated artifacts, caches, logs, `__pycache__`, or large result
   files are staged.
2. Confirm new scripts have a README, docstring, or usage block that states
   inputs, outputs, and side effects.
3. Confirm documentation names the canonical key and output grain.
4. Confirm claim boundaries are explicit for MAG-level, sample-level, and
   transfer-learning interpretations.
5. Confirm lane and denominator labels are explicit for POC MAG-bin, POC
   assembly-context, mangrove/MSM local archive, mangrove/Futian source-lane
   and ready-payload counts, and any published-quality denominator.
6. Run only safe static checks while production jobs are active. Do not submit,
   cancel, requeue, prune, or consolidate active outputs unless that is the
   explicit task.

## Allowed Current Claims

Allowed now:

> MethaNet is building a source-audited functional atlas that links ESM2 latent
> bridge candidates to independent MAG quality, taxonomy, methane-cycle,
> sulfur-cycle, CAZyme, metabolic-trait, and genomic-context evidence across
> rumen, wetland/MUCC, mangrove/MSM, and mangrove/Futian molecular lanes.

Not allowed yet:

> MethaNet has proven source-independent methane MRV transfer from rumen to
> wetland ecosystems.

The stronger claim requires full cohort evidence, sample/metagenome context,
abundance or coverage rollups, source-aware controls, and external validation.
