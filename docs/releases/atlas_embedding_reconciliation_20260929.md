# Atlas embedding reconciliation — 29 September 2026

## Scope and decision

The user authorized reconciliation of the embedding configuration, correction of the already-public landing/report, and preparation of the research deck. This is a scoped `noindex` correction. The August molecular release, functional warehouses, source gaps and public three-case records retain their existing scientific status.

## What was corrected

The original 662-proteome pilot used the mean of ESM-2 hidden layers 20–33. The subsequent atlas lanes used the final layer, 33. Mixed extraction protocols invalidated the prior cross-run interpretation of coordinates, nearest-core matches and neighbor counts.

The retained pilot FASTAs were recomputed with the 650M ESM-2 model, snapshot `08e4846e537177426273712802403f7ba8261b6c`, in float32 with TF32 disabled. Both historical and final-layer representations were generated from the same inputs. The run retains a 30-aa minimum, file-order cap of 6,000 proteins, truncation to 1,020 residues, attention-mask token means including special tokens, and proteome means. All 662 records passed input-hash, protein-count, finite-vector and historical-reproduction checks. The maximum `1 − cosine` reproduction difference was 4.33e-15, below the 1e−12 gate.

The new report recomputes projections, full-space neighbors, reference matches, candidate selection, sensitivity counts, tables and fallback figures. It does not reuse archived pilot ranks or realign old coordinates. A contract binds all ten NPZ artifacts and verification evidence to their SHA-256 values. The site export carries matching audit metadata; both manual publication and landing-only CI reject inconsistent report/landing pairs.

## Corrected public geometry

The 29 September geometry release maps **7,710 MAG/proteome records**. It draws **2,226 links**: 2,200 sampled cross-habitat neighbor links and 26 selected one-way nearest-core links. These are exploratory representation similarities.

Outside the 625-record core (518 rumen, 107 wetland), **1,403/2,501 wetland** and **2,823/4,584 mangrove** records have a rumen genome as their closest core match. The nearest-core median cosine is 0.9973; the random-pair median is 0.9942. The corrected dimension-standardized graph contains 486 mutual pairs involving rumen and another habitat. One-way core matches and reciprocal full-atlas neighbors answer different questions; neither validates ecological transfer.

Counts and candidate selection are recomputed from the corrected vectors. The public export includes its geometry audit and contract hash, and publication rejects stale quantities or a mismatched report. UMAP, diffusion, t-SNE and PCA show the same records; changing projection does not change high-dimensional link membership. The 255 registered gap rows remain explicit and unplotted.

## Evidence that remains incomplete

- Archived June lane model revisions are not fully recorded. Final-layer attribution is supported by code lineage, run statistics and numerical controls; this is not a claim of complete historical provenance.
- Source-aware and phylogeny-aware scientific controls remain necessary. Shared representation geometry is not a functional-transfer test.
- Exact ecological sample, abundance, environmental and process joins remain unresolved for calibrated sample risk. No A–E tiers, measured methane estimates, or carbon-credit approval are released.
- The selected same-DNA retrieval controls in the deck establish configuration consistency for those controls, not predictive performance on independent ecosystems.

## Validation and authority

- Pilot verification: `results/blue_catalyst_poc/reembed_single_configuration_20260928/verify_and_collate_summary.json`.
- Report gates: 20/20 passing.
- Browser checks: desktop, tablet, mobile, projections, keyboard interactions, three-case network and report fallbacks; `results/reports/atlas_embedding_release_20260929/browser/browser_audit.json`.
- Focused regression tests: `results/reports/atlas_embedding_release_20260929/focused_tests.xml`.
- Release parity and deployment receipts: `results/reports/atlas_embedding_release_20260929/`.
- Machine-readable receipt: [atlas_embedding_reconciliation_20260929.json](atlas_embedding_reconciliation_20260929.json).
- Active pointer: [atlas_current_release.json](../../configs/atlas_current_release.json). Its original `lane_registry`, freeze and ledger SHA pins still describe the immutable August payload. `embedding_lane_registry` and `embedding_contract` identify the geometry overlay.
- Previous pointer bytes are preserved in [atlas_20260810_pointer_before_geometry_reconciliation.json](atlas_20260810_pointer_before_geometry_reconciliation.json), so earlier ontology receipts remain traceable to their exact source. Those historical projections are not silently rewritten.

## Rebuild

After verified pilot outputs exist, run `scripts/reports/reconcile_atlas_embeddings.py`, then build the next-generation report using the dated embedding registry, contract and August freeze. Heavy geometry work runs through Slurm. Export with `web/emergentbiome-methanet/tools/export_atlas.py`, synchronize quantities using `tools/sync_geometry_release.py --write --report <bundle>`, and run release parity plus browser checks before `tools/publish_site.sh deploy --push`.

The source-count ledger remains the denominator authority. Geometry changes do not add field observations or stronger biological claims.
