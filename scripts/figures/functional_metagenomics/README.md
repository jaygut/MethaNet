# Functional Metagenomics Figure Utilities

This folder contains source-controlled plotting utilities for MethaNet
functional-genomics results.

## Current Utility

`plot_functional_calibration_panel.py` builds a preliminary six-panel figure
from completed per-MAG curated run records and Parquet shards:

- runtime distribution,
- CheckM2/GUNC-aware QC tiers,
- resolved genus counts,
- functional yield across KOfam, MCycDB, SCycDB, and dbCAN,
- common CAZy family evidence, and
- a compact MAG-level signal heatmap.

The script is a reporting utility, not a pipeline dependency. It reads completed
curated outputs and writes figures plus a source CSV.

## Output Policy

Default outputs are generated under:

```text
results/functional_metagenomics/fgx_662_apollo3_20260612/figures/preliminary_functional_panel/
```

`results/` is ignored by git. Figure PNG/PDF files and source CSVs should stay
as generated artifacts unless a specific report or manuscript snapshot needs to
version a reviewed export.

## Claim Boundary

The panel is useful for early quality control and communication, but it is not
the canonical cohort feature store. For final analysis, regenerate figures from
the cohort warehouse after consolidation and validation gates have passed.

Do not use this panel alone to make source-independent transfer, flux, or
sample-level environmental claims.

## Safe Usage

Example command:

```bash
python scripts/figures/functional_metagenomics/plot_functional_calibration_panel.py \
  --root results/functional_metagenomics/fgx_662_apollo3_20260612 \
  --out-dir results/functional_metagenomics/fgx_662_apollo3_20260612/figures/preliminary_functional_panel
```

This command reads completed per-MAG curated artifacts and writes generated
figures. It does not submit jobs or modify active run state.
