# Snakemake Backbone for Apollo 3

This is a detailed operational scaffold for the MethaNet functional-metagenomics expansion. It is meant to be copied or promoted into `workflow/` once the parser/integration scripts are implemented.

## Why It Lives in `ai_docs`

The current request is to design the expansion plan and backbone, not to force a production workflow into the existing package before the MAG paths and database installations are locked. Keeping it here makes the scientific contract reviewable without breaking current CI.

## Main Files

- `Snakefile` - DAG shape for pilot and cohort runs.
- `config.apollo3.yaml` - Apollo 3 paths/resources to adapt.
- `cluster/config.yaml` - SLURM profile skeleton.
- `envs/*.yaml` - environment sketches for per-rule conda deployment.

## Pilot Mode

Use one MAG:

```bash
snakemake -n \
  -s ai_docs/functional_metagenomics_expansion/snakemake_backbone/Snakefile \
  --configfile ai_docs/functional_metagenomics_expansion/snakemake_backbone/config.apollo3.yaml \
  --config pilot_mag_id=<MAG_ID>
```

## Cohort Mode

Use all MAGs in `mag_manifest`:

```bash
snakemake \
  -s ai_docs/functional_metagenomics_expansion/snakemake_backbone/Snakefile \
  --configfile ai_docs/functional_metagenomics_expansion/snakemake_backbone/config.apollo3.yaml \
  --profile ai_docs/functional_metagenomics_expansion/snakemake_backbone/cluster \
  --use-conda \
  --rerun-incomplete
```

## Implementation Hooks Still Needed

The DAG calls parser/integration scripts under `workflow/scripts/functional_metagenomics/`. These should be implemented when the scaffold is promoted:

- `resolve_mag_manifest.py`
- `integrate_qc_taxonomy.py`
- `parse_mcycdb_hits.py`
- `parse_scycdb_hits.py`
- `build_methane_pathway_completeness.py`
- `aggregate_functional_matrices.py`
- `build_bridge_mechanism_cards.py`
- `build_latent_function_features.py`
- `build_platform_demo_artifacts.py`

Those scripts should be pure, deterministic, and covered by small fixture tests before the Apollo 3 production run.

