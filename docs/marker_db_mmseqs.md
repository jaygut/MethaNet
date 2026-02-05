---
description: Build and enable the MethaNet MMseqs marker DB (core seed)
---

# Purpose

This runbook builds a reproducible **core** protein FASTA (`db/marker_db_core.faa`) and MMseqs2 database (`db/marker_db.mmsdb`) for the MethaNet `marker_annotator` stage.

The core DB is derived from **public, pinned** reference sources:

- TIGRFAMs **15.0** SEED (curated families)
- Pfam **37.0** seed **only where TIGRFAMs is missing a marker** (currently: `mtaB`, using `PF12176`)

This gives you a solid baseline you can extend over time with a curated MethaNet-specific layer.

# Outputs

After a successful run, you should have:

- `db/marker_db_core.faa`
- `db/marker_db_core.manifest.tsv`
- `db/marker_db.faa` (currently core-only copy)
- `db/marker_db.mmsdb` (+ MMseqs index files)

# Prerequisites (HPC)

- A working tools environment with:
  - `python3`
  - `mmseqs`
  - `tar`, `gzip`
  - `wget` or `curl`

Notes:

- The build script uses a small embedded Python Stockholm parser; no `esl-reformat` dependency.
- Downloads are resumable (`wget -c` or `curl -C -`).

# Build steps

From the repository root:

```bash
conda activate methanet-tools
bash workflow/scripts/build_marker_db_core.sh
```

Optional environment overrides:

```bash
# Pin releases explicitly (defaults shown)
export TIGRFAM_RELEASE=15.0
export PFAM_RELEASE=37.0

# Increase indexing parallelism
export MMSEQS_THREADS=16

# Skip MMseqs build (FASTA-only)
export SKIP_MMSEQS_BUILD=1

bash workflow/scripts/build_marker_db_core.sh
```

# Enable MMseqs in the pipeline

Set this in your pipeline config YAML:

```yaml
functional:
  mmseqs_enabled: true
```

Then re-run preflight:

```bash
python poc/mucc_rumen_poc/scripts/preflight_check.py \
  --manifest poc/mucc_rumen_poc/manifest_mucc_subset.tsv \
  --pipeline-config <your_pipeline_config.yaml>
```

# Extending the DB (MethaNet curated layer)

Recommended pattern:

- Keep `db/marker_db_core.faa` immutable.
- Create `db/marker_db_methanet.faa` with sequences you curate/validate.
- Concatenate to build a combined FASTA:

```bash
cat db/marker_db_core.faa db/marker_db_methanet.faa > db/marker_db.faa
mmseqs createdb db/marker_db.faa db/marker_db.mmsdb
mmseqs createindex db/marker_db.mmsdb db/tmp --threads 16
```

Update `db/marker_db_methanet.faa` via version-controlled curation criteria + provenance.
