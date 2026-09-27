# Biogeochemical knowledge graph foundation

Reviewed 2026-09-27. This document now describes both the current implementation and the next design horizon. The historical Kuzu/TSV/Parquet POC graph contains **662** MAG/bin and assembly-context molecular units. The August molecular release registers **7,965** lane-scoped records; that denominator is not a count of unique biological entities or validated ecological links.

Implementation update, 2026-09-27: [MVO 0.2.0](../ontology/README.md) now adds a local, immutable RDF/Neo4j molecular-evidence slice to the original 0.1.0 release-record graph. The current snapshot contains 735,060 statements and 37,071 resources, preserves all 233,190 original statements, and adds selected evidence for 145 MAGs, including loci, method-specific annotations, processed RNA, typed context measurements, alternatives, and explicit gaps. The full registered denominator remains 7,965 records; the selected slice does not claim complete molecular materialization of every record. No exact molecular-to-flux pairs or mechanism-comparable atlas units were admitted. See the [molecular handoff](../ontology/docs/MOLECULAR_HANDOFF_20260926.md), [verification history](../ontology/docs/VERIFICATION.md), and [SQL/Neo4j comparison](../ontology/docs/STORE_COMPARISON_20260926.md). The August release, warehouse source authority, and scientific claim boundaries remain unchanged.

## Current store roles

The ontology and explicit source mappings determine identity, evidence meaning, admission, provenance, and the semantics of fixed questions. The storage engine determines how admitted facts are stored and queried. In the three preregistered matched controls, DuckDB SQL and the graph returned identical complete rows, while SQL had the lower warm median in each local run. This is evidence for keeping SQL as the primary tabular/aggregation interface and Neo4j as a relationship/provenance traversal projection, not a universal performance ranking. See the [bounded comparison and limitations](../ontology/docs/STORE_COMPARISON_20260926.md).

## Scope and identity

The graph should answer provenance-aware questions such as “Which MAGs have tool-covered methane-pathway evidence in a mangrove or wetland source cohort?”, “Which sequencing samples have exact record identities and processed expression?”, and “Which site/process measurements are eligible for a future paired validation?” It must also return the denominator, gaps, competing evidence, assay/protocol class, and reason a stronger claim is blocked.

Use stable, namespaced IDs rather than display labels. A molecular node key is `(lane_id, proteome_id)` and a curation-attempt key adds `cohort_run_id` and `run_id`. Source accessions retain their authority namespace. Sampling event, physical sample, sequencing sample, instrument observation, and site are separate identities. An inferred crosswalk is an evidence assertion with its own ID and status; it does not collapse two entities until an exact identity is validated.

| Node class | Grain | Required provenance or guard |
| --- | --- | --- |
| Source artifact and dataset release | A file/accession and an immutable published or internal release | Source URI, access rights, checksum, snapshot, authority, version |
| Site, habitat, sampling event, physical sample | One named site, environment type, collection event, and collected material | Coordinates with precision, date/time, depth, method, original label, exact/ambiguous mapping status |
| Sequencing sample, run, MAG/proteome | One molecular specimen/run or one lane-scoped molecular unit | Accession/crosswalk, assembly and bin provenance, QC, selected/attempt status, `proteome_id` |
| Gene, annotation hit, pathway or feature | One method-specific event or derived molecular assertion | Gene ID, database and version, threshold, coverage, tool/run fingerprint, accepted-hit state |
| Environmental and process observation | One typed measurement with instrument, time window, unit, statistic and QC | Keep chamber flux, tower flux, porewater concentration, and covariates distinct |
| Candidate link, evidence assertion, claim, validation gap | One reviewable assertion or blocking item | Evidence source, method, confidence/uncertainty, status, allowed wording, reviewer/action |

Edges should be typed by what is actually established: `derived_from`, `observed_in`, `collected_at`, `has_annotation_hit`, `has_measurement`, `has_exact_crosswalk`, `has_candidate_crosswalk`, `nearest_core_neighbor`, `supported_by`, and `blocked_by`. Each edge needs source and target IDs, edge type, evidence record, creation method, source release, status, and uncertainty or rank when applicable. Preserve direction and distinguish exact identity from similarity. A nearest-neighbor or 2D proximity edge is **molecular similarity**, not gene transfer, ecological interaction, causal influence, or a flux association. Candidate assertions must be reversible and never erase contradictory or missing rows.

## Projection pipeline and minimum gates

1. Resolve the current release pointer, lane registry, source manifests, and warehouses. Freeze the input checksums; keep the source tables as the authority and the graph as a rebuildable projection.
2. Create source/release, lane-scoped MAG, attempt, gene/annotation, sample/event, typed measurement, assertion, and gap nodes. Include status rows even if no feature or graph edge can be made.
3. Generate only joins whose key, authority, and evidence grade are declared. Store ambiguous/alias matches as candidate edges with their source and reason. Do not mint an exact physical-sample link from a BioProject, site label, RNA-sample expression row, or coordinate alone.
4. Validate node-key uniqueness; source-manifest and release denominator parity; edge endpoint existence; reciprocal provenance; date/time/unit consistency; field value completeness; and no conversion of null to biological zero. Run a query-level assertion that every release-excluded molecular unit is retrievable with a reason and next action.
5. Publish a graph version with schema/ontology versions, source manifests, code/config hashes, QC and count receipts, unresolved-link counts, claim scope, and rollback pointer. Keep a previous projection until the new one passes.

The first implementable graph slice is source artifact → lane/source MAG → curation attempt → selected molecular evidence → annotation event or candidate link, with explicit release exclusions. A second slice can add **typed** sample and measurement records after the identity and tower-mapping audits. Ecological linkage and calibrated prediction remain later gates requiring exact pairing, abundance/read coverage, environmental covariates, uncertainty, and independent flux/process validation.

## Vocabulary alignment

Use established vocabularies as **candidate mappings**, then pin exact versions and test each local field before export. [W3C PROV-O](https://www.w3.org/TR/prov-o/) supplies a model for entities, activities, agents, and derivation; [GSC MIxS](https://genomicsstandardsconsortium.github.io/mixs/) covers contextual metadata for sequenced samples; [ENVO](https://obofoundry.org/ontology/envo.html) offers controlled environmental classes. Their existence does not make current source labels compliant automatically. Maintain a mapping table with local column/value, external term ID and version, transform, original value, confidence, and unmapped reason. Inference from these standards to this repository's proposed mapping is a design choice, not a statement of current implementation.

The graph's scientific boundary follows [the release ledger](releases/atlas_20260810_release_ledger.json) and [claim guidance](methanet_positioning_and_claims.md): molecular potential and review priority are supported; field activity, source-independent transfer, calibrated methane-risk tiers, and crediting decisions require additional evidence.
