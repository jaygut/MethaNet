# Guideline-to-implementation traceability

Source: a user-supplied enterprise ontology guideline in a private workspace.
Its native page was not independently verified; treat this as design-input
traceability, not as a formal standard or a public citation. The source document
and its private workspace locator are not distributed with this repository.

| Guideline requirement | Materialized here | Boundary / remaining external dependency |
| --- | --- | --- |
| Decisions and competency questions before ontology expansion | Architecture charter; competency-question register and tests | Customer workflows need partner discovery |
| Distinct observation, hypothesis, claim and decision | OWL classes, disjointness, reified assertions, SHACL | No automatic evidence-to-credit rule |
| Stable scoped identity before integration | Injective composite IRIs, duplicate/FK checks, quarantine | 133 MUCC site references remain unresolved |
| Provenance and reconstruction | PROV-O, full file digests, source rows, code receipt, RO-Crate | Upstream historical raw sequence hashes are not invented |
| Explicit uncertainty and missingness | Uncertainty objects, missing/partial/failed/ambiguous SKOS states | No calibrated transfer intervals exist yet |
| Bounded validity and transaction history | Scope and time fields; immutable named graph snapshots | No retroactive backfill of unknown sample times |
| Meaning separate from storage | Canonical OWL/RDF and exact Neo4j projection | No proprietary graph dependency in the ontology |
| OWL and SHACL complementary | ROBOT profile, HermiT, pySHACL and adversarial fixtures | Formal consistency is not field validation |
| Independent ontology/shape/mapping/tool/graph releases | `release.json` versions; content-addressed snapshots | Human promotion/rollback process required |
| External vocabulary reuse with transparent mappings | PROV, DCAT, SOSA, QUDT, GeoSPARQL; SSSOM table | ENVO/MIxS source harmonization remains reviewed extension work |
| Source inventory and rights before access | Hashed warehouse/metadata catalog; object policies | No upstream reuse licenses assumed |
| Read-only typed tools and explicit budgets | JSON Schemas, named query registry, policy enforcement | Not a deployed MCP service or remote auth system |
| Model/run compatibility and scoped transfer | Configuration model, incompatible-similarity rejection, post-freeze diagnostic linkage | No ESM/gLM re-embedding or numerical harmonization performed |
| Semantic CI and negative fixtures | Dedicated test workflow, parse/profile/reasoning/SHACL/key/roundtrip tests | Human scientific and security review pending |
| Ecology scale and rights | Site/specimen/observation/project/tenure/consent model | No project ownership or FPIC inferred from public data |
| Molecular chain of custody | Sampling, custody, assay, gene/protein and annotation classes | Real custody records require authorized partners |
| Climate-finance bridges are explicit hypotheses | BridgeAxiom and blocked validation/decision states | Calibration and external protocol acceptance are future gates |
| Query routing by evidence need | Exact counts via source-derived records; graph for relationships; no vector truth claims | Warehouse query service and vector index are separate future adapters |
| Governance, semantic diff and deprecation | Extension/release runbook, semantic diff utility, version gates | No fictional appointed reviewers or enterprise SLAs |
| Operational metrics and maturity | Verification receipt, counts, gaps, runtime resource budget | Local foundation; not Level 4/5 organizational maturity |

The generic guideline's biomedical, industrial, MCP/A2A federation and
organization-wide rollout examples are not mandatory deployments for this
blue-carbon foundation. JSON Schema contracts and local tools are implemented;
OpenAPI service hosting, MCP remote auth, ODRL/DPV policy compilation, RML/R2RML
engines, ShEx, federated GraphRAG and production observability are deliberately
not claimed. The typed Python mapper is executable and versioned; its output is
standard RDF, but it is not presented as an RML engine.
