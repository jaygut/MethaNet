# MethaNet Repository Guide

Documentation refresh: 2026-09-24

This guide is the shortest path from a fresh clone to a useful local
exploration. MethaNet is a code-and-contract repository; the large sequencing,
embedding, warehouse, and report outputs are intentionally kept outside Git.

## Start here

1. Read [`../README.md`](../README.md) for the product narrative and current
   controlled-diligence release counts.
2. Read [`methanet_positioning_and_claims.md`](methanet_positioning_and_claims.md)
   before writing or interpreting an external-facing claim.
3. Read [`current_artifact_inventory.md`](current_artifact_inventory.md) for
   dated local artifact paths and evidence-lane denominators.
4. Read [`methanet_triview_release_20260810.md`](methanet_triview_release_20260810.md)
   for the release freeze, original publication gate, and later scoped
   correction-deployment record.
5. Use [`atlas_data_foundation.md`](atlas_data_foundation.md) for dataset
   authority, table grains, and reproducible-release handoffs; use
   [`knowledge_graph_foundation.md`](knowledge_graph_foundation.md) for current
   graph scope and the next typed biogeochemical graph contract. The
   [release index](releases/README.md) explains the tracked pointer and receipts.

The functional-metagenomics contracts and MRV maturity ladder live under
[`../ai_docs/functional_metagenomics_expansion/`](../ai_docs/functional_metagenomics_expansion/).

## Repository map

| Path | Role |
| --- | --- |
| `src/methanet/` | Reusable Python package: models, features, embeddings, classification, and MBAG primitives |
| `scripts/` | Warehouse, source-staging, report, validation, and Apollo-3 execution entrypoints |
| `configs/` | Pipeline configuration, sample metadata, and the authoritative atlas-lane registry |
| `contracts/` | Versioned machine-readable atlas identity, grain, and status contracts |
| `tests/` | Unit tests for scientific contracts, report builders, validators, and release parity |
| `docs/` | Product positioning, data/graph foundations, methods, inventories, runbooks, and tracked release receipts |
| `web/emergentbiome-methanet/` | Public landing page, local report publisher, and browser verification tools |
| `workflow/` | Snakemake template/spec; it is not a bundled data run |
| `data/` and `results/` | Local inputs and generated evidence artifacts; excluded from normal commits |

## Local setup

The committed lockfile targets Python 3.11 or newer. Use that range for a
reproducible `uv sync`; the package metadata currently has a broader minimum.

```bash
git clone https://github.com/jaygut/MethaNet.git
cd MethaNet
uv sync --extra dev
```

The core package can be imported without the heavy model or HPC extras:

```bash
uv run python -c "import methanet; print(methanet.__version__)"
```

## Verification

Run the complete lightweight suite before opening a pull request or publishing
an artifact:

```bash
uv run pytest -q
uv run python scripts/reports/validate_atlas_foundation.py
uv run python -m compileall -q src scripts tests
git diff --check
```

The default foundation check uses tracked files only. On a workstation with
the registered ignored warehouses and August freeze, run
`uv run python scripts/reports/validate_atlas_foundation.py --require-local`
to check release hashes, all 7,965 freeze identities/status rows, and the four
warehouse manifests and gate reports. Its MUCC non-pass gates are surfaced in
the receipt and still require separate scientific resolution.

The landing page can be previewed without the large data warehouse:

```bash
cd web/emergentbiome-methanet
python -m http.server 8848
```

The full `tools/verify_page.py` and `tools/verify_page_firefox.py` checks also
exercise the generated `/report/` bundle, so run them after assembling a local
report with `tools/publish_site.sh build`. Firefox-based verification is
optional and requires a local Firefox/Selenium runtime. Report and warehouse
builders require the corresponding ignored local `results/` inputs; they are
not expected to run in a clean clone.

The live [EmergentBiome Molecular Atlas](https://emergentbiome.earth/) and
[technical report](https://emergentbiome.earth/report/) use the August 10 frozen
evidence, published through a September 2026 correction under `noindex`. A
fresh clone does not contain the large frozen warehouses or reconciled report
bundle. Use the dated release ledger, source manifests, checksums, and
deployment receipt described in the data foundation guide before rebuilding
or asserting parity. The deployed `gh-pages` tree and default source branch
can differ; check the receipt's source commit rather than assuming that a
fresh clone reproduces the live site.

## Operational conventions

- `proteome_id` is the canonical cohort key unless a source-specific contract
  explicitly requires another key.
- Preserve failed, partial, pending, and missing evidence as status rows.
- Treat Neo4j/Kuzu or other graph projections as serving layers; canonical
  release tables and provenance remain the source of truth.
- The existing queryable graph MVP covers the 662-record POC; the 7,965-row
  release union is not a materialized sample-resolved knowledge graph.
- A data-complete tri-view is not automatically mechanism-comparable.
- Current molecular outputs support candidate review and monitoring design;
  calibrated sample/project methane risk, final A-E tiers, measured flux, and
  carbon-credit approval require independent validation inputs.

## Generated files and clean working trees

Do not commit local sequencing or warehouse payloads. The repository ignores
`data/*`, `results/*`, `features/`, `v9out/`, `scratch_snap/`, scheduler logs,
and web build output. Keep temporary manifests and one-off snapshots in those
locations so a fresh clone stays small and reviewable.

Before handing a clone to another researcher, use:

```bash
git status --short --branch
git diff --check
```

A clean handoff has no uncommitted source, test, documentation, or configuration
changes and no untracked project files outside intentionally ignored data and
generated-output directories.
