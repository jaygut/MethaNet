# Competency questions and prohibited questions

| ID | Question | Executable evidence / expected behavior |
| --- | --- | --- |
| CQ01 | How many release records exist in each lane? | `release_summary.rq/.cypher`, release adapter parity, tool summary |
| CQ02 | How many are data-complete, excluded and mechanism-comparable? | Freeze field mapping, ledger equality, summary tool |
| CQ03 | Why is each excluded record excluded, and what is next? | `exclusions.rq/.cypher`; mandatory gap reason/action |
| CQ04 | Where did a screening claim originate? | `evidence_chain.rq`; source file digest and row locator |
| CQ05 | Are failed/partial attempts and off-release attempts retained? | Adapter tests, CurationAttempt status, explicit quarantine |
| CQ06 | Which source tables contain expression, coverage and process records? | Artifact inventory, table row counts and source hashes |
| CQ07 | Which samples lack exact environment/process linkage? | `sample_readiness`, sample-flux gap nodes |
| CQ08 | Is a library accession a physical sample? | Disjoint classes and invalid crosswalk fixtures: no |
| CQ09 | May embeddings from incompatible runs support accepted retrieval? | Similarity shape and negative fixture: no |
| CQ10 | Does a marker or RNA signal imply gas flux or process direction? | No such OWL entailment; disjoint measurement types |
| CQ11 | Can an unassayed gene be reported as absent? | Covered-no-hit guard: no |
| CQ12 | Can CH4, CO2 and N2O evidence coexist without conflation? | Three-gas synthetic fixture; gas-scoped observations/bridges |
| CQ13 | Can a concentration be used as flux by relabeling it? | Quantity-kind and reviewed-unit guards: no |
| CQ14 | Are hypotheses preserved without being treated as facts? | Reified propositions, review states and non-entailment tests |
| CQ15 | Can unknown source rights permit external export? | Object policy and purpose tests: no |
| CQ16 | Can an agent query another tenant or submit raw Cypher? | Trusted principal separation and JSON Schema rejection: no |
| CQ17 | Can Neo4j reproduce the original RDF exactly? | Full live export/reconstruction hash and count receipt |
| CQ18 | Can a changed input silently reuse a graph release? | Pin mismatch, immutable snapshot and idempotence checks: no |
| CQ19 | Can a graph claim carbon-credit approval? | Current admission profile rejects it |
| CQ20 | Is a protocol requirement mapping evidence of compliance? | Explicit proposed-support status and blocking gap: no |

Prohibited outputs include final A–E methane-risk tiers, measured flux inferred
from MAGs, quantitative credit deductions based on embedding distance,
source-independent rumen-to-wetland transfer, accepted independent field
validation without a new reviewed profile, and automatic credit issuance.

The real slice answers availability, provenance, readiness and gap questions.
Biological mechanism ranking for every MAG and spatially paired field inference
are **not** answered from a catalog entry. Those require independently tested
warehouse fact adapters and sample/assay mapping. The gas-specific observation
fixtures demonstrate ontology behavior, not newly collected field evidence.
