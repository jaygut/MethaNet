# Atlas data foundation

Reviewed 2026-09-24. This is the entry point for building on the EmergentBiome Molecular Atlas. The current promoted molecular release is the **2026-08-10 freeze**, selected by [`../configs/atlas_current_release.json`](../configs/atlas_current_release.json). Later manuscripts, status snapshots, repaired builders, and public report renderings do not silently replace that data release. Promote a successor only after a new freeze, parity checks, and an explicit pointer change.

## Authority and physical layout

| Layer | Authority | Unit and rule |
| --- | --- | --- |
| Acquired inputs | Source records and checksums under `data/external/`, plus the POC crosswalk | Preserve accession, license/access, checksum, source row, and source-specific identity. Raw inputs stay outside Git. |
| Lane membership | [`../configs/methanet_atlas_lanes.tsv`](../configs/methanet_atlas_lanes.tsv) and its registered source/functional manifests | Four lanes and their source denominators. `proteome_id` is canonical within a lane; `(lane_id, proteome_id)` is the atlas identity. |
| Curation attempts | `results/functional_metagenomics/*/per_mag/` and run records | An attempt is not a selected MAG. Preserve failed, partial, superseded, and integrity-failed attempts. |
| Functional warehouses | Each registered `functional_warehouse_dir` and its `cohort_table_manifest.tsv` and `validation_gates.tsv` | Three pipeline-normalized 24-table warehouses and one distinct 61-table MUCC source scaffold. Resolve paths through the lane registry. |
| Molecular release | Freeze manifest and release ledger named by [`../configs/atlas_current_release.json`](../configs/atlas_current_release.json) | Immutable dated denominator, payload flags, exclusions, claim boundaries, and checksums. A small [ledger snapshot](releases/atlas_20260810_release_ledger.json) is tracked for clean-clone review. |
| Reports and site | Versioned `results/reports/` bundles and [`../web/emergentbiome-methanet/`](../web/emergentbiome-methanet/) | Views of a declared release; visual projections and narrative do not change the release denominator or authorize a stronger claim. |

The warehouse manifests currently record `table`, physical `path`, `rows`, `columns`, and `bytes`. They do **not** by themselves establish primary keys, source hashes, selection rules, or scientific equivalence. The versioned [`../contracts/atlas_data_contract_v1.json`](../contracts/atlas_data_contract_v1.json) adds a minimum schema and semantic contract for critical release, molecular, expression, and environmental tables. It explicitly marks keys that still need a duplicate audit. All 24 pipeline and 61 MUCC physical tables remain discoverable through their registered manifests; the new contract is **not** a claim that every source-specific key is already validated.

## Frozen denominator and distinct evidence contracts

The tracked ledger and ignored freeze agree on **7,965 registered units**, **7,710 release-required, data-complete molecular tri-views**, and **255 explicit exclusions** (248 Futian source gaps and seven MUCC ESM-2 gaps). Of the 7,710, **5,209** are pipeline-normalized and **2,501** are MUCC source-scaffold tri-views. The release has **zero mechanism-comparable, release-authorized ecological sample-linked, field-validated, and calibrated units**. These are readiness statuses, not assertions that sample metadata or environmental observations do not exist.

| Lane | Source denominator | Registered warehouse `dim_mag` | Release-required molecular units | Warehouse contract |
| --- | ---: | ---: | ---: | --- |
| POC core | 625 | 625 | 625 | Pipeline-normalized |
| MSM mangrove | 1,428 | 1,428 | 1,428 | Pipeline-normalized; source-aware comparability pending |
| Futian mangrove | 3,404 | 3,156 | 3,156 | Pipeline-normalized; 248 source gaps retained |
| MUCC v1 wetland | 2,508 | 2,508 | 2,501 | Source scaffold; seven molecular-release exclusions retained |

The MUCC `dim_mag` count is the checksum-validated source roster. It must not be substituted for the 2,501 complete molecular payloads. MUCC's 61 tables contain source DRAM, RNA expression, sequencing-sample crosswalks, chamber observations, porewater concentrations, tower time series, and exploratory network outputs. These have different grains and protocols from the 24-table normalized pipeline. The current MUCC gate report includes partial, warning, and blocked gates, so a table's presence is not a scientific pass.

The July 18 operational `results/functional_metagenomics/atlas_lane_registry_summary.md` records an earlier **6,315** tri-view count. It is historical operational evidence, not the August release denominator. The historical 662-unit POC Kuzu graph is also not the 7,965-unit atlas graph.

## Identity, missingness, and claim rules

1. Use `(lane_id, proteome_id)` for an atlas molecular record. Use `(cohort_run_id, proteome_id)` for a selected warehouse MAG and add `run_id` for an attempt. A bare `mag_id`, sample label, site name, or coordinate is not a safe cross-source join key.
2. Keep source membership, selected molecular payload, QC, tool coverage, functional hits, expression, physical samples, field measurements, and claims as separate entities or tables. A MAG-level marker indicates functional potential under its assay; it does not establish in-situ activity, process direction, or methane flux.
3. `NULL` or an absent sparse hit can mean unavailable input, failed/partial run, uncovered tool, unresolved source mapping, or a covered no-hit. Interpret it through status and coverage columns. Only a completed, covered assay can support method-specific no-hit wording; never turn missing evidence into zero biology.
4. RNA expression detection in 133 processed MUCC sequencing samples is not DNA abundance. Staged chamber, porewater, and tower observations are different physical quantities; their presence does not make a validated MAG-to-field-sample or flux pair. The release's `sample_linked_units=0` means no **authorized ecological linkage for MRV scoring**.
5. The dated August generic process table mixes measurement types and has a known tower value/date mapping defect for 29,280 rows. Use typed source tables or a separately validated repair view for research, and rebuild a corrected generic table in a **new dated output** before promotion. Do not silently patch the August freeze or call its generic rows validated methane-flux measurements.
6. Candidate ranks, neighbors, and 2D positions support molecular exploration. Cross-lane mechanism equivalence, final A–E methane-risk tiers, measured-flux attribution, transfer, and carbon-credit approval remain unestablished by this release. See [positioning and claims](methanet_positioning_and_claims.md).

## Validation and change policy

For a fresh clone, run `python scripts/reports/validate_atlas_foundation.py` to check the tracked pointer, contract, ledger snapshot, arithmetic, registry structure, and SHA-256 pins without fetching terabytes of data. On a data-bearing workstation, add `--require-local` to check the pointed freeze checksum, unique `(lane_id, proteome_id)` rows, release flags, and each registered warehouse manifest/gate report. The validator reports MUCC's partial/blocked gates as warnings, not a scientific clearance. It deliberately does not scan Parquet payloads. The full registry/source-manifest gate remains `scripts/reports/validate_atlas_lane_registry.py`; targeted Parquet/schema and scientific checks remain necessary before release promotion.

Write new source acquisitions, warehouse rebuilds, and release candidates to versioned directories. Record source accession and rights, checksums, cohort/run IDs, code and configuration hashes, tool/database versions, row counts, grain/key/status rules, QC, transformation lineage, and a rollback pointer. Run source-to-manifest ID reconciliation, duplicate checks, manifest-to-Parquet row parity, gate review, and claim review before changing the active registry or release pointer. Preserve prior artifacts and failed candidate outputs for provenance. Do not move or commit the large ignored `data/` or `results/` trees to achieve a tidy Git status.

The next schema work is concrete: declare and audit four pipeline event/metric keys, audit the 61 MUCC table keys and foreign relations, pin source-access and method fingerprints, repair the typed-to-generic tower mapping in a new build, and establish exact sample/event joins. The [knowledge graph foundation](knowledge_graph_foundation.md) explains how these contracts can support a later biogeochemical graph without changing the evidence boundary.
