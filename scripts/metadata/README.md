# Metadata Recovery Utilities

This folder contains source-controlled utilities for recovering and auditing
sample, accession, and environmental context for MethaNet cohorts.

## Current Utility

`recover_environmental_metadata.py` reconstructs environmental context for the
662-proteome functional cohort by joining:

- the canonical 662-row proteome crosswalk,
- ENA `analysis` metadata for PRJEB31266 rumen records,
- MUCC Methanoregula Zenodo metadata,
- OWC metatranscriptome sample labels where available, and
- NCBI Assembly/BioSample summaries for GTDB/NCBI MAGs that resolve cleanly.

The script is intentionally conservative. It separates exact accession-level
metadata from source-bucket, site, project, or publication-level context. Do
not treat every non-empty environmental field as sample-level metadata.

## Output Policy

Default outputs are generated under:

```text
results/functional_metagenomics/environmental_metadata_recovery_20260612/
```

`results/` is ignored by git. Keep raw API responses, downloaded public metadata
archives, validation TSVs, and intermediate crosswalks there unless a small,
reviewed audit snapshot is intentionally promoted into `ai_docs/`.

## Metadata Resolution Semantics

Use `metadata_resolution` as the analytical confidence field:

| value | interpretation |
| --- | --- |
| `exact_analysis_accession` | Rumen MAG joined directly through ENA `analysis_accession`. |
| `exact_ncbi_assembly_biosample` | GTDB/NCBI MAG resolved through NCBI Assembly and BioSample. |
| `exact_owc_bin_plus_site_project` | OWC MAG matched by MAG/Bin ID; environmental context is still mostly site/project/sample-design level. |
| `exact_mucc_source_bucket` | MAG source bucket is exact, but compact public files do not expose per-MAG BioSample context. |
| `missing_*` | Metadata recovery did not resolve the expected source layer. |

For downstream modeling, use exact sample or BioSample fields where available.
Source-bucket and site/project rows are provenance, not environmental covariates
at sample grain.

## Safe Usage

Example command:

```bash
python scripts/metadata/recover_environmental_metadata.py \
  --repo-root . \
  --out-dir results/functional_metagenomics/environmental_metadata_recovery_20260612
```

This command reads local cohort files and public APIs, then writes generated
outputs under `results/`. It does not submit jobs or modify active per-MAG run
directories.

## Maintenance Checklist

- Keep `proteome_id` as the join key back to the 662-row cohort.
- Preserve `mag_id_candidate` and source-specific accessions for auditability.
- Cache raw public responses in the generated output folder.
- Promote only small reviewed summaries or contracts into `ai_docs/`.
- Document every non-sample-level row before using it in environmental models.
