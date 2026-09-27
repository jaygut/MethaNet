# Molecular Verification Ontology (MVO)

A working, standards-based foundation for source-audited molecular evidence in
blue-carbon verification. CH4, CO2 and N2O; mangroves, wetlands, peatlands and
extensible coastal habitats. Version **0.2.0**, local development release.

The ontology separates molecular potential, expression, observations,
hypotheses, rights, uncertainty and decisions. It does **not** turn the atlas
into measured emissions, calibrated risk tiers or approved carbon credits.

Start with [architecture](docs/ARCHITECTURE.md),
[MVO documentation index](docs/README.md),
[Notion requirement traceability](docs/NOTION_TRACEABILITY.md),
[competency questions](docs/COMPETENCY_QUESTIONS.md), and
[verification history](docs/VERIFICATION.md). The bounded
[SQL/Neo4j store comparison](docs/STORE_COMPARISON_20260926.md) records the
measured query parity and performance limits.

## Contents

The [2026-09-26 molecular extension](docs/MOLECULAR_HANDOFF_20260926.md)
is instantiated and queried. Its data-bearing whitepaper/demo bundle and raw
query receipts remain local-only pending independent scientific review and
external rights clearance; they are not distributed in this repository.
Its validated local snapshot contains 735,060 statements, including the original
233,190 unchanged atlas statements, selected sequence/annotation evidence and
explicit missing measurements. All 18 fixed molecular queries passed declared
source parity; three matched SQL controls returned identical complete answers.
Independent scientific approval and external rights clearance remain pending.

| Directory | Authority / purpose |
| --- | --- |
| `schema/` | Canonical modular OWL 2 RL ontology in RDF 1.1 Turtle |
| `vocabulary/` | SKOS controlled concepts |
| `shapes/` | SHACL admission contracts and scientific claim boundaries |
| `mappings/` | Executable warehouse field contract, standards and protocol mappings |
| `contracts/`, `config/` | Typed tool contracts, trusted local access policy, runtime pins |
| `src/mvo/` | Read-only ingestion, validation, immutable release builder, graph projection, tools |
| `queries/` | Fixed SPARQL and Cypher examples |
| `tests/` | Synthetic/adversarial semantic, policy, mapping and packaging tests |
| `docs/generated/` | Generated [term reference](docs/generated/schema-reference.md) and JSON-LD context |
| `build/`, `.cache/`, `.venv/` | Ignored local graph outputs, runtime/data cache, isolated dependencies |

## Install and validate

Run from this directory. No changes to the root Python environment are needed.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pytest -q
.venv/bin/python -m mvo local-neo4j install
.venv/bin/python -m mvo verify --robot .cache/robot-1.9.8.jar
.venv/bin/python -m mvo fixture --out build/synthetic-example
```

Installation downloads SHA-256-pinned ROBOT and Neo4j archives. Java 17 or 21
is required for the selected Neo4j LTS runtime. The Python lock records the
tested Linux/Python 3.13 versions; other platforms require dependency validation.
All fixture data are visibly synthetic, never mixed into the atlas graph.

## Materialize the actual registered atlas

```bash
.venv/bin/python -m mvo build \
  --repo .. \
  --metadata results/reports/methanet_atlas_metadata_readiness_20260810 \
  --out build/atlas-new-snapshot
```

The adapter follows `configs/atlas_current_release.json`, verifies its pinned
inputs, preserves the full registered denominator, hashes all cataloged files,
and checks warehouse row/byte counts. Optional analyst diagnostics can be
passed through `--diagnostic`; they remain separate from the frozen release and
are not required to build the governed evidence slice. Use `--recorded-at` from an
existing manifest to reproduce the same transaction timestamp; otherwise the
current UTC timestamp is recorded. Reproduction also requires identical code,
contracts, mappings and source bytes.

Outputs include canonical `graph.nt`, named `graph.trig`, source/code/version
manifest, SHACL receipt, RO-Crate metadata and an exact Neo4j projection. Existing
output directories are never overwritten. A failed validation leaves a
quarantined error receipt, not an approved graph.

The original 0.1.0 slice materializes release records, attempts, metadata identities,
screening claims and gaps. Large genes, raw hits, expression and process matrices
are checksum-addressed catalog assets, **not fully materialized fact nodes**.
No inferred molecular-to-flux edges or numerical re-harmonization are performed.
The additive 0.2.0 slice additionally materializes selected loci, proteins,
method-specific annotations, alternatives, processed RNA and typed context.
Full warehouse matrices remain outside the graph. See the molecular handoff
for the new snapshot and execution commands; the old snapshot remains valid.

## Local Neo4j

```bash
.venv/bin/python -m mvo local-neo4j start
.venv/bin/python -m mvo local-neo4j load \
  --projection build/atlas-20260926-validated/neo4j \
  --receipt build/atlas-20260926-validated/neo4j-receipt.json
.venv/bin/python -m mvo neo4j-audit \
  --snapshot-dir build/atlas-20260926-validated \
  --out build/atlas-20260926-validated/query-parity.json
.venv/bin/python -m mvo local-neo4j stop
```

Browser: `http://127.0.0.1:17474`; Bolt: `bolt://127.0.0.1:17687`.
Random local credentials are in `.cache/neo4j-local-auth.json` with mode 0600.
Do not commit/share this file or disable authentication. The graph is retained
when the service stops. A second load of the same snapshot is idempotent.
For environments that clean up detached processes, keep
`.cache/neo4j-community-5.26.31/bin/neo4j console` running in a separate terminal.

The runtime has a 768 MiB Java heap cap and 256 MiB page cache. This is a modest
development configuration, not a performance SLA or full-warehouse sizing claim.
No APOC plugin, cloud account or enterprise license is needed for this slice.

## Bounded evidence queries

```bash
.venv/bin/python -m mvo query \
  --graph build/atlas-20260926-validated/graph.nt \
  --request '{"query":"release_summary","purpose":"internal_review"}'
.venv/bin/python -m mvo query \
  --graph build/atlas-20260926-validated/graph.nt \
  --request '{"query":"validation_gaps","purpose":"internal_review","limit":5}'
```

Named tools enforce object/purpose policy, row/time/size limits and abstention.
Arbitrary Cypher/SPARQL, credit issuance, risk-tier assignment and review-state
mutations are absent. Human-operated query examples are separate from this
agent-facing interface. Local OS trust is not remote authentication.

See [research sources](docs/RESEARCH_AND_STANDARDS.md) and
[governance/extension gates](docs/GOVERNANCE_AND_EXTENSION.md) before expanding
scope or presenting external claims. Automated validation is not independent
scientific signoff, protocol certification, or production security approval.
