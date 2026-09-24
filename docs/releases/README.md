# Tracked release receipts

The small JSON files here are reviewed snapshots of dated, ignored release artifacts. They make headline counts and claim boundaries inspectable from a fresh clone while preserving the full freeze and warehouses under `results/`.

The active release is selected only by [`../../configs/atlas_current_release.json`](../../configs/atlas_current_release.json). The tracked lane registry and ledger snapshot must match their declared SHA-256 values; the ignored generated release ledger, freeze manifest, and freeze decision must match theirs when local data is present. A newer report, manuscript, site edit, or operational status summary does not replace this pointer.

Do not put raw reads, embeddings, large tables, graph databases, or generated report bundles here. Add a successor snapshot only after release validation and an explicit promotion decision; retain prior dated snapshots for audit.
