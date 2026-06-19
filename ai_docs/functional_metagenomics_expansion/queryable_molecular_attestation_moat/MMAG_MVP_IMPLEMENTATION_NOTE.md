# MethaNet Molecular Attestation Graph MVP

Date: 2026-06-17

Scope: local, embedded MVP for a queryable MethaNet Molecular Attestation Graph
(MMAG). This is a developer-facing implementation note for the code in
`scripts/attestation/build_molecular_attestation_mvp.py`.

## What This Builds

The MVP turns current MethaNet evidence artifacts into a local query layer:

- `registry_artifact.parquet`: source artifact registry with file provenance;
- `evidence_atom.parquet`: typed evidence atoms with source artifacts;
- `graph_nodes.parquet` and `graph_edges.parquet`: portable graph export;
- `mmag.kuzu/`: embedded Kuzu property graph;
- `QUERY_LIBRARY.cypher`: canonical graph queries;
- `query_results/*.tsv`: executed query outputs;
- `validation_gates.tsv` and `validation_report.md`: automated gates and claim-boundary report.

The default output path is ignored/generated:

```bash
results/attestation/<snapshot_id>/
```

## Selected Stack

The MVP uses:

- **Parquet + PyArrow/Pandas** for durable local exports and source table reads.
- **Kuzu 0.11.3** for embedded Cypher-style property-graph querying.
- **DuckDB** as the already compatible SQL/lakehouse companion dependency.
- **NetworkX** remains available for future algorithmic checks and fallback graph analytics.

The stack is local-first and HPC-friendly: no Neo4j server, web service, daemon,
Docker service, cloud resource, or cluster job is required.

External tooling research favored Kuzu because its official documentation
describes an embedded in-process graph database with property graph modeling,
Cypher query support, columnar on-disk storage, Parquet interoperability, and no
external server requirement. DuckDB remains the SQL-side partner because its
official docs support direct Parquet scans and local analytical workflows.
NetworkX and RDFLib were reviewed as robust Python fallbacks, but NetworkX lacks
native persistent Cypher/SPARQL query semantics and RDFLib would force an RDF
model before MethaNet's property graph contract is stable.

Important caveat: the original KuzuDB GitHub repository was archived on
2025-10-10. Kuzu `0.11.3` installed and smoke-tested cleanly in the local
MethaNet `.venv`, so the MVP uses it while keeping Parquet node/edge exports as
the durable interchange layer.

## Data Sources Consumed

Defaults:

```text
ai_docs/functional_metagenomics_expansion/proteome_crosswalk/embedded_662_proteome_id_crosswalk.tsv
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.with_unit_scope.tsv
results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse_poc_magbin_union_20260616_075022/
results/functional_metagenomics/fgx_662_apollo3_20260612/reports/mbag_smoke_full_docx_scoped_20260614_0554_final/bridge_attestation_cards_smoke.tsv
results/functional_metagenomics/fgx_662_apollo3_20260612/reports/mbag_smoke_full_docx_scoped_20260614_0554_final/mbag_esm2_knn_edges.parquet
results/contextual_genomics/glm2_integration_20260616_poc_catchup_20260616_073441/feature_glm_mag_level.tsv
```

Warehouse tables used:

- `dim_mag`
- `fact_taxonomy_gtdbtk`
- `feature_methane_mechanism`
- `feature_sulfur_competition`
- `feature_mrv_mag_level`
- `feature_annotation_coverage`
- `fact_run_status`

## Graph Schema

Node tables:

- `MAG`
- `SourceDomain`
- `Taxon`
- `Feature`
- `EvidenceAtom`
- `Claim`
- `ValidationGap`
- `Artifact`

Relationship tables:

- `FROM_SOURCE`
- `HAS_TAXONOMY`
- `HAS_FEATURE`
- `HAS_EVIDENCE`
- `GENERATED_BY`
- `EVIDENCE_SUPPORTS_CLAIM`
- `MAG_SUPPORTS_CLAIM`
- `MAG_BLOCKED_FROM_CLAIM`
- `BLOCKED_BY`
- `NEAR_IN_ESM2_SPACE`

The graph uses `proteome_id` as the canonical MAG/proteome key. It preserves the
662-row denominator, marks 625 MAG-bin units, and quarantines the 37
assembly-context units from MAG-level attestation claims.

## Rebuild

Install local dependencies if needed:

```bash
./.venv/bin/python -m ensurepip --upgrade
./.venv/bin/python -m pip install 'duckdb>=1.0' 'kuzu==0.11.3'
```

Build the default snapshot:

```bash
./.venv/bin/python scripts/attestation/build_molecular_attestation_mvp.py \
  --repo-root /home/rsg-jcorre38/Jay_Proyects/MethaNet \
  --snapshot-id mmag_mvp_20260617
```

Run without Kuzu when only Parquet exports are needed:

```bash
./.venv/bin/python scripts/attestation/build_molecular_attestation_mvp.py \
  --repo-root /home/rsg-jcorre38/Jay_Proyects/MethaNet \
  --snapshot-id mmag_mvp_parquet_only \
  --skip-kuzu
```

## Tests

Run the integration test:

```bash
./.venv/bin/python -m pytest tests/integration/test_molecular_attestation_mvp.py
```

The test builds a temporary snapshot and validates:

- identity consistency by `proteome_id`;
- no dropped completed MAGs;
- no duplicate primary graph entities;
- every evidence atom has source/provenance;
- no final MRV risk tier is encoded as a fact;
- known POC counts: 662 cohort units, 625 MAG-bin units, 37 assembly-context units;
- canonical Kuzu query execution.

## Example Queries

The builder writes `QUERY_LIBRARY.cypher` and executes these queries:

- top complete multiview bridge candidates;
- wetland/MUCC methane plus sulfur-context candidates;
- strong bridge candidates blocked by missing or weak evidence;
- evidence paths for `mucc__GCA_002495465.1_ASM249546v1_genomic`;
- source-domain and taxonomy patterns that may indicate confounding;
- report-ready versus blocked counts;
- MAG molecular-attestation evidence that remains insufficient for final MRV risk.

Run the expert audit stress suite:

```bash
./.venv/bin/python scripts/attestation/audit_molecular_attestation_mvp.py \
  --snapshot-dir results/attestation/mmag_mvp_20260617
```

This writes:

- `expert_static_audit_gates.tsv`
- `expert_audit_query_results_summary.tsv`
- `expert_audit_query_results/*.tsv`
- `EXPERT_AUDIT_REPORT.md`

The stress suite includes both simple schema/count queries and complex multi-hop
queries that traverse candidate MAGs to features, evidence atoms, source
artifacts, claim blockers, source-domain/taxonomy context, and cross-domain
ESM2 neighbors.

## Claim Boundary

Allowed:

> MethaNet has a local queryable MAG/proteome-level molecular attestation graph
> linking ESM2 bridge evidence, functional annotations, gLM2 context, QC,
> taxonomy, run status, source artifacts, and explicit claim-boundary blockers.

Not allowed:

> This MVP assigns final sample/project methane-risk scores, measured flux,
> final A-E MRV tiers, source-independent rumen-to-wetland transfer proof, or
> carbon-credit approval.

Required upgrade evidence:

- sample/MAG mapping;
- abundance or read coverage;
- environmental covariates;
- uncertainty propagation;
- field/process methane validation;
- source-replicated cohorts and source-aware validation.

## Next Steps

1. Promote the graph schema into a stable LinkML or JSON Schema contract.
2. Add richer gene/marker/pathway nodes from `fact_mcycdb_hits`,
   `fact_scycdb_hits`, KOfam, and METABOLIC tables.
3. Add Kuzu query tests for candidate-card evidence paths once top rumen bridge
   candidates have complete functional evidence.
4. Add a Parquet-first API/MCP serving layer later, reading from the same
   snapshot outputs.
