# MUCC v1 authoritative sequence-to-ecology crosswalk: data-recovery handoff

## Why this is needed

The integrated MUCC v1 lane is a reproducible molecular-reference atlas, but
it cannot be promoted to ecological validation without an explicit,
row-level mapping from each sequencing sample to its field metadata and
environment/flux observations.

The [mSystems article](https://journals.asm.org/doi/10.1128/msystems.00680-25)
identifies its supplemental Tables S1-S13 workbook as the location of this
supporting metadata. The publisher lists the workbook as 95.67 KB; the
Europe PMC-provided payload retained in the source ledger has the same
97,962-byte size, but has no readable XLSX ZIP central directory. It is
therefore not a parseable Table S4 source. The existing Zenodo, KBase, NCBI,
JGI, ESS-DIVE, AmeriFlux, and NERR evidence layers do not provide an explicit
sequence-sample-to-field-observation bridge and must not be joined by label,
date, site, or depth inference. This includes the newly staged ESS-DIVE DOI
`10.15485/2500238` half-hourly gap-filled tower fluxes: their 2015-2016
temporal overlap is site/time context, not an inferred sample, plot, depth, or
flux-window mapping.

Source evidence and checksums are retained in
`results/functional_metagenomics/mucc_v1_owc_wetland_20260626/source_audit/mucc_v1_source_metadata_recovery_ledger.tsv`.

## Requested author or publisher deliverable

Please provide either a corrected `msystems.00680-25-s0002.xlsx` or a
tab-separated export with the same row-level semantics. A separate manifest is
acceptable when it directly joins the following records. Do not fill missing
values by interpretation of sample labels.

- Original sequencing-sample label and stable sample/accession identifier.
- Full collection timestamp, site/plot/core, measured depth, units, and depth
  reference.
- Sequencing assay/library provenance and an explicit reconciliation for any
  sample whose public SRA package is declared `WGS` rather than `RNA-Seq`.
- MAG abundance or read-coverage record identifier and units.
- Geochemistry/metabolite record identifier, source, full measurement
  timestamp, and units.
- Chamber or porewater methane observation identifier, source, measurement
  type and units, plus the full start/end time window used for the association.
- Replicate identity, missingness semantics, and uncertainty record/method.

The 133 source labels expected by the atlas are retained, without inferred
field attributes, in
`results/functional_metagenomics/mucc_v1_owc_wetland_20260626/environmental_metadata/mucc_v1_sample_columns_scaffold.tsv`.

## Canonical TSV schema

The supplied file must have the following tab-separated header. Leave a
declared unavailable relation as a row with
`source_evidence_status=authoritative_missing` rather than dropping it.

```text
mapping_id	source_sample_column	authoritative_sample_id	collection_datetime	site_id	core_or_plot_id	depth_cm	depth_reference	sequence_assay_type	assay_reconciliation_status	mag_abundance_or_read_coverage_record_id	mag_abundance_or_read_coverage_units	environment_source	environment_record_id	environment_measurement_datetime	environment_measurement_units	flux_source	flux_observation_id	flux_measurement_type	flux_units	flux_window_start_datetime	flux_window_end_datetime	replicate_id	uncertainty_record_id	uncertainty_method	source_evidence_status	missingness_status	source_url
```

For `source_evidence_status=authoritative_complete`, every field other than
`missingness_status` must be populated. Datetimes must be full ISO-8601 values
with a time component; the flux-window start must not be after its end; depth
must be finite and non-negative. When an ESS-DIVE identifier is supplied, the
stager verifies it against the already retained chamber/porewater observations
or, when `flux_source=ESS_DIVE_10.15485_2500238`, against the staged
gap-filled tower observation IDs. A tower identifier still requires the
authoritative sample/time/plot/depth correspondence and documented
tower-context rule described above.

## Controlled ingestion and promotion

Run the following command only on the supplied, unmodified author/publisher
file:

```bash
./.venv/bin/python scripts/external/stage_mucc_v1_authoritative_ecological_crosswalk.py \
  --input /absolute/path/to/authoritative_mucc_crosswalk.tsv
```

Then rebuild promotion, warehouse, audit, and dashboard using the commands in
the [MUCC v1 integrated-atlas playbook](mucc_v1_integrated_atlas_playbook.md).
Successful rows become eligible for *grouped ecological validation only*.
They do not establish a causal mechanism, MAG-level methane-flux effect, final
MRV score/A-E tier, carbon-crediting claim, or source-independent transfer.
