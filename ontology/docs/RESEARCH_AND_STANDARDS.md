# Standards and protocol research register

Research cut: 2026-09-26. Primary sources only. This register distinguishes
implemented reuse from future alignment; selecting a standard is not evidence
that the warehouse already conforms to its complete application profile.

| Source / pinned baseline | Design consequence | Status |
| --- | --- | --- |
| [RDF 1.1 Concepts, 2014](https://www.w3.org/TR/2014/REC-rdf11-concepts-20140225/) | Stable IRIs, typed literals, named graph snapshots; explicit assertion resources | Implemented |
| [OWL 2 Profiles, 2012](https://www.w3.org/TR/owl2-profiles/) | RL-compatible class/property semantics, no automatic scientific bridge rules | Profile checked with OWLAPI |
| [SHACL, 2017 Recommendation](https://www.w3.org/TR/shacl/) | Closed-world admission and adversarial negative fixtures | Core/SPARQL implemented; no SHACL 1.2 dependency |
| [SKOS, 2009](https://www.w3.org/TR/skos-reference/) | Controlled concepts distinct from OWL classes | Implemented for gases, habitats, processes and evidence states |
| [PROV-O, 2013](https://www.w3.org/TR/prov-o/) | Sources, activities, attribution and derivation | Selective formal reuse; no full import closure |
| [DCAT 3](https://www.w3.org/TR/vocab-dcat-3/) | Dataset-release specialization and catalog discipline | Selective reuse; not a complete DCAT application profile |
| [SOSA/SSN, 2017 Recommendation](https://www.w3.org/TR/2017/REC-vocab-ssn-20171019/) | Physical samples and observations with explicit features/results | Class/property reuse; later SSN drafts not required |
| [GeoSPARQL 1.1](https://docs.ogc.org/is/22-047r1/22-047r1.html) | Site feature model; geometry/CRS extension point | Class reuse only; no geographic inference or auto-geocoding |
| [QUDT 3.5.1 molar areal flux unit](https://qudt.org/vocab/unit/MOL-PER-M2-SEC), [micromolar unit](https://qudt.org/vocab/unit/MicroMOL-PER-M2-SEC) | Explicit quantity units; molar amount/area/time dimensional whitelist | Two reviewed flux units admitted; no automatic mass/carbon/GWP conversion |
| [SSSOM 1.0 model](https://mapping-commons.github.io/sssom/1.0/spec-model/) | Mapping justification, source versions and explicit non-equivalence | SSSOM-shaped TSV, checked against declared axioms; no claim of external mapping certification |
| [MIxS release process](https://genomicsstandardsconsortium.github.io/mixs/releasing/), [v7.0.1 release changes](https://genomicsstandardsconsortium.github.io/mixs/version-changes/v7.0.0_to_v7.0.1/) | Source metadata needs versioned field-level reconciliation | Reviewed extension target; not claiming MIxS compliance |
| [ENVO](https://obofoundry.org/ontology/envo.html) | Reviewed environment term mappings should retain source labels and provenance | Extension target, not automatic source-label matching |
| [ODRL 2.2](https://www.w3.org/TR/odrl-model/) | Machine-enforced rights must not be confused with natural-language permissions | Local access-policy contract implemented; ODRL translation not yet implemented |
| [RO-Crate 1.2](https://www.researchobject.org/ro-crate/specification/1.2/structure) | Linked-data package metadata around graph/source/code receipts | Local crate structure generated and tested; no external conformance certification |
| [ROBOT profile validation](https://robot.obolibrary.org/validate-profile.html), [reasoning](https://robot.obolibrary.org/reason.html) | Formal profile and inconsistency/unsatisfiability checks | ROBOT 1.9.8 pinned by SHA-256, HermiT executed |
| [Neo4j system requirements](https://neo4j.com/docs/operations-manual/current/installation/requirements/), [release history](https://neo4j.com/release-notes/database/) | Use Java-17-compatible 5.26 LTS locally | Community 5.26.31, SHA-256-pinned archive; not an unsupported archived graph engine |

QUDT's displayed quantity-kind associations include photon flux. Here its molar
area/time **unit dimensions** are reused for explicitly gas-scoped observations;
no photon-process equivalence is asserted. Other valid gas-flux units require a
reviewed mapping addition and conversion tests before admission.

## Isometric interface: what is and is not supported

[Mangrove Restoration v1.1.0](https://registry.isometric.com/protocol/mangrove/1.1/1.1.0)
addresses net CO2e removal, including project GHG effects. Its section 4.1.2
screens restoration activities that could increase CH4 or N2O; potentially
decreasing salinity or increasing nitrogen inputs affects eligibility. This is
not permission to replace protocol measurements or eligibility assessment with
genomic predictions. Molecular signals can motivate investigation and monitoring
design; their quantitative verification role requires independent validation.

The [combined mangrove requirements](https://registry.isometric.com/requirements/mangrove)
include boundaries, baselines, data retention, uncertainty, ownership and
community considerations. The versioned local interface table maps a small
subset to evidence classes and lists unresolved prerequisites. It is a proposed
evidence interface, not a complete compliance checker, certified integration,
partnership, credit approval, or legal opinion. Protocol applicability must be
rechecked for each actual project and version.

The user-supplied image provides the commercial north star—verification
infrastructure—but is not used as technical or financial evidence. No funding
amounts, partnerships or market leadership are inferred from it.

## Choices that remain deliberately outside this sprint

No entire external ontology is vendored automatically. Source labels are not
assigned ENVO IDs by resemblance. No emission factors, GWP coefficients,
salinity thresholds, credit deductions or final risk tiers are invented.
CO2/N2O coverage is formal vocabulary and validation capability, not a claim
that the existing methane atlas has complete reviewed CO2/N2O mechanism labels.
MIxS, ENVO, ODRL/DPV, a remote API/MCP gateway and vector services are independent
extensions with their own source/version/rights and acceptance tests.
