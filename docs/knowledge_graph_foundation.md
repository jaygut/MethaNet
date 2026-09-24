# Biogeochemical knowledge graph foundation

Reviewed 2026-09-24. This is a design contract for a future marine and wetland biogeochemistry knowledge graph built from the [atlas data foundation](atlas_data_foundation.md). The existing Kuzu/TSV/Parquet POC graph contains **662** MAG/bin and assembly-context molecular units; it is a historical molecular evidence MVP. The August molecular release registers **7,965** units but is **not yet a materialized graph of 7,965 biological entities with validated ecological relations**.

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
