# MUCC + Rumen Pilot POC

This folder holds the operational artifacts for a focused pilot POC that uses:
- MUCC v2.0.0 wetland MAGs (target domain)
- Rumen archaeome MAGs (source domain)

The POC exercises marker detection, ESM-2 embeddings, DNABERT-2 embeddings, and optional GenomeOcean embeddings on Apolo-3.

---

## 1) Recommended POC Size

Target counts (balanced for signal and feasibility):
- Rumen MAGs: 120 (source domain)
- MUCC MAGs: 240 (target domain), 40 per site across 6 wetland complexes

Total: 360 MAGs

This is large enough to validate cross-wetland behavior while remaining feasible for embeddings on H100.

---

## 2) File Layout

- `manifest.tsv`: generated sample manifest with placeholder IDs
- `pipeline.yaml`: Snakemake config with all sample IDs
- `scripts/`: helper scripts to fetch MUCC file lists, select MAGs, and split MAG FASTA

---

## 3) Generate the Pilot Files

```bash
python poc/mucc_rumen_poc/scripts/generate_pilot_assets.py \
  --rumen-count 120 \
  --mucc-per-site 40
```

This writes:
- `poc/mucc_rumen_poc/manifest.tsv`
- `poc/mucc_rumen_poc/pipeline.yaml`

---

## 4) MUCC File Listing and Download

List the Zenodo files for MUCC v2.0.0:

```bash
python poc/mucc_rumen_poc/scripts/fetch_mucc_zenodo_files.py --list
```

The current file list is also saved in `poc/mucc_rumen_poc/zenodo_files.tsv`.

Download a file by key:

```bash
python poc/mucc_rumen_poc/scripts/fetch_mucc_zenodo_files.py \
  --download-key YOUR_FILE_NAME.zip \
  --out-dir data/mucc
```

Note: The Zenodo record currently exposes a Methanoregula-focused subset (for example
`Methanoregula_MAGs_DB.zip`). Use the MUCC-linked BioProjects for full MAG coverage
if the full MAG archive is not present in the record.

The file `poc/mucc_rumen_poc/Methanoregula_MAGs_list.txt` is already downloaded as
an entry point for Methanoregula-focused pilot selection.

---

## 5) Select MUCC MAG IDs (from metadata)

After you download the MUCC MAG metadata TSV:

```bash
python poc/mucc_rumen_poc/scripts/select_mucc_mags.py \
  --metadata MUCC_v2.0.0_MAG_metadata.tsv \
  --out-ids poc/mucc_rumen_poc/mucc_mag_ids.txt \
  --per-site 40
```

Then split the MAG FASTA by ID:

```bash
python poc/mucc_rumen_poc/scripts/split_fasta_by_id.py \
  --fasta MUCC_MAGS.fna \
  --ids poc/mucc_rumen_poc/mucc_mag_ids.txt \
  --out-dir data/assemblies
```

---

## 6) Select Rumen MAG IDs (ENA PRJEB81441)

```bash
python poc/mucc_rumen_poc/scripts/select_rumen_mags.py \
  --assemblies ena_prjeb81441_assemblies.tsv \
  --out-ids poc/mucc_rumen_poc/rumen_mag_ids.txt \
  --count 120
```

Download those assemblies from ENA.

---

## 7) Reconcile Placeholder IDs (Recommended)

Replace placeholder sample IDs with your selected MAG IDs:

```bash
python poc/mucc_rumen_poc/scripts/reconcile_manifest_ids.py \
  --rumen-ids poc/mucc_rumen_poc/rumen_mag_ids.txt \
  --mucc-ids poc/mucc_rumen_poc/mucc_mag_ids.txt \
  --mucc-metadata MUCC_v2.0.0_MAG_metadata.tsv
```

This updates `manifest.tsv` and `pipeline.yaml` in place, writes `.bak` backups,
and produces `poc/mucc_rumen_poc/id_mapping.tsv`.

---

## 8) Preflight and Run the Pilot

Preflight check (includes HMM/DB/FASTA alignment and external tool checks):

```bash
python poc/mucc_rumen_poc/scripts/preflight_check.py \
  --manifest poc/mucc_rumen_poc/manifest.tsv \
  --pipeline-config poc/mucc_rumen_poc/pipeline.yaml
```

If you want to temporarily bypass external tool checks, add `--skip-tool-check`.

Copy the manifest to the pipeline location:

```bash
cp poc/mucc_rumen_poc/manifest.tsv configs/samples.tsv
```

Run Snakemake:

```bash
snakemake -s workflow/Snakefile --cores 8 --configfile poc/mucc_rumen_poc/pipeline.yaml --rerun-incomplete
```

---

## 9) GenomeOcean Embeddings (Manual)

Generate GenomeOcean embeddings for the same MAG list, then re-run fusion:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np
from methanet.embedding.genomeocean import GenomeOceanEmbedder, GenomeOceanConfig

model_path = Path("/path/to/GenomeOcean-4B.pt")
mag_ids = Path("poc/mucc_rumen_poc/mucc_mag_ids.txt").read_text().splitlines()
mag_ids += Path("poc/mucc_rumen_poc/rumen_mag_ids.txt").read_text().splitlines()

embedder = GenomeOceanEmbedder(GenomeOceanConfig(model_path=model_path, fp16=True))

out_dir = Path("features/embeddings")
out_dir.mkdir(parents=True, exist_ok=True)

for mag_id in mag_ids:
    fasta = Path("data/assemblies") / f"{mag_id}.fasta"
    vec = embedder.embed_genome(fasta)
    np.save(out_dir / f"{mag_id}_genomeocean.npy", vec)
PY
```

```bash
python workflow/scripts/fuse_features.py \
  --metadata configs/samples.tsv \
  --functional-dir features/functional \
  --esm2-dir features/embeddings \
  --genome-dir features/embeddings \
  --output-all features/all_features.parquet \
  --output-source features/rumen.parquet \
  --output-target features/coastal.parquet \
  --genome-backend genomeocean \
  --genome-dim 3072
```

---

## 9) Notes

- `domain=coastal` is used for MUCC samples to preserve the target split without code changes.
- If you want to use `domain=wetland`, we can extend `fuse_features.py` to accept a configurable target domain.
