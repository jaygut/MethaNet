# MethaNet: Transfer Learning for Methane Flux Prediction

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

**Bridging Rumen Microbiome Data to Climate Verification Through Cross-Ecosystem Genomic Analysis**

---

## Overview

MethaNet is a research initiative developing computational methods to predict net methane flux in coastal ecosystems using transfer learning from agricultural microbiome data. By leveraging the world's most comprehensive methanogen genomic resources, the ruminant gut archaeome, we aim to decode complex coastal wetland systems that are critical for carbon sequestration but remain data-sparse.

Methane has a global warming potential approximately 30× that of CO₂ over a 100-year horizon. Coastal wetlands can be net carbon sinks or sources depending on the balance between carbon uptake and methane emissions. Current measurement methods (chamber measurements, flux towers) are expensive, sparse, and unable to scale. This project addresses a critical gap in climate accounting: the inability to distinguish net climate benefits from net climate harms using molecular data.

### Core Hypothesis

The ratio of methanogen marker genes (`mcrA`) to methanotroph marker genes (`pmoA`) can predict **net methane flux** across diverse saline environments. This molecular signal persists due to conserved methanogenesis machinery, enabling cross-ecosystem transfer learning.

---

## Research Objectives

1. **Discover transferable feature sets**: Identify minimal molecular feature sets that maximize flux prediction while remaining transferable across ecosystems, ranked by sequence conservation.

2. **Characterize domain shift**: Quantify distribution shift between rumen and mangrove communities using clustering and embedding analysis to identify "bridge" training examples.

3. **Identify novel flux predictors**: Discover non-obvious gene associations beyond mcrA, including heterodisulfide reductase variants and electron-bifurcating complexes.

4. **Validate net flux prediction**: Test the mcrA/pmoA ratio against environmental covariates in mangrove samples with paired flux measurements.

---

## Datasets

### Primary Dataset: Ruminant Gut Archaeome Catalogue

The most comprehensive methanogen genomic resource with paired methane emission measurements.

| Resource | Description | Size | Source |
|----------|-------------|------|--------|
| Ruminant Gut Archaeome | Curated archaeal genomes from ruminant gut systems | 998 genomes | [Mi et al., 2024](https://doi.org/10.1038/s41467-024-54025-3) |
| RUG2 Catalog | Metagenome-assembled genomes from rumen | 4,941 MAGs | [Stewart et al., 2019](https://doi.org/10.1038/s41587-019-0202-3) |
| Hungate1000 | Cultivated rumen microbiome isolates | 410 genomes | [Seshadri et al., 2018](https://doi.org/10.1038/nbt.4110) |

**Why rumen data?** The rumen system provides:
- High-resolution genomic templates paired with flux measurements
- Standardized functional annotations and biochemical pathway data
- Detailed environmental metadata enabling predictive model training

### Target Dataset: Coastal Sediment Metagenomes

Target environmental datasets for model validation and transfer learning.

| Dataset | Description | Source |
|---------|-------------|--------|
| Global Mangrove Metagenomes | ~127 curated samples (from ~150 public samples) | NCBI SRA |
| Mangrove Methanogen Study | 13 MAGs with pathway analysis | [Zhang et al., 2020](https://doi.org/10.1186/s40168-020-00876-z) |

**Data disparity:** ~26,000 rumen microbiome sequencing runs exist in NCBI SRA compared to ~2,400 from mangrove sites, a >10× disparity that motivates our transfer learning approach.

**Validation strategy:** We are curating publicly available coastal metagenomes with co-located flux tower or chamber measurements for model validation, with 23 samples targeted for paired-flux evaluation.

### Blue Catalyst POC (Completed on Apolo-3)

We completed a cross-ecosystem proteome embedding POC between MUCC wetland genomes and rumen genomes (PRJEB31266) using ESM2-650M protein language model embeddings. The POC was developed for the [Hatch Blue — Blue Catalyst](https://www.hatch.blue/programs/blue-catalyst) accelerator program (Singapore, May 2026).

**662-genome cohort (current, `apolo_full_20260228_080644_embed_20260305_061952`)**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Cohort | 662 genomes (107 wetland MUCC + 555 rumen PRJEB31266) | 16.5× scale-up from baseline |
| Embedding | 662 × 1,280 (ESM2-650M, layer 33, mean-pooled) | Zero attrition, zero non-finite vectors |
| PERMANOVA R² | 0.202 (p=0.001) | Ecosystem explains 20.2% of embedding variance |
| Silhouette | 0.398 [95% CI: 0.364–0.439] | Bootstrap CI from 150 resamples |
| CV Classifier | AUC=1.000, balanced accuracy=0.999 | 5-fold CV, PCA-50, balanced class weights |
| Cohen's d | 3.63 | Very large effect size on trajectory axis |
| Bridge genomes | 14 with ≥1 opposite-ecosystem k-NN neighbor | Out of 662 total |
| Top bridge | bin.8 (Archaea), alpha-transfer score=3.47 | >6 SDs above cohort mean; all top 11 are rumen Archaea |

**Key scientific findings:**
- Rumen Archaea dominate bridge candidate rankings — all top 11 alpha-transfer scores are Archaea, consistent with conserved methanogenesis machinery (mcrA, HdrABC) across ecosystems
- Perfect classifier separation (AUC=1.0) under real-world class imbalance confirms the embedding manifold encodes a learnable ecosystem boundary
- **P0 caveat**: Source and ecosystem are perfectly confounded (all rumen = PRJEB31266, all wetland = MUCC). Deconfounding with additional sources is the highest-priority next step.

**40-genome baseline (initial validation, `apolo_20260226_194505`)**
- 40 samples (20 MUCC + 20 rumen), PERMANOVA R²=0.517, silhouette=0.433, trajectory t=13.97 (p=1.5e-16)
- bin.23 (Archaea) embeds 100% inside wetland cluster (mixing_coeff=1.0)
- The R² decrease from 0.517→0.202 at scale is expected: adding 515 rumen genomes increases intra-class diversity while the ecosystem boundary remains perfectly classifiable

**Deep-dive analytics report**
- 6 publication-grade figures + Word report: `results/blue_catalyst_poc/interim_snapshots/apolo_full_20260228_080644_embed_20260305_061952_notebook_interim_20260306_055012/deep_dive_report/`
- Fig 1: Embedding landscape (PCA/UMAP/t-SNE with KDE contours + bridge entropy histogram)
- Fig 2: Statistical validation (PCA scree, silhouette violin, trajectory + Cohen's d, boundary diagnostic)
- Fig 3: Top 20 alpha-transfer candidate ranking
- Fig 4: Confounding decomposition (domain composition, protein counts, confounding matrix)
- Fig 5: Cosine distance heatmap sorted by ecosystem
- Fig 6: Key metrics infographic dashboard
- Review memo: `ai_docs/Blue_Catalyst_Deep_Dive_Review_Memo.md`

**Execution paths**
- HPC notebook: `notebooks/blue_catalyst_esm2_poc.ipynb`
- Local analytics notebook: `notebooks/blue_catalyst_partial_report_local.ipynb`
- Deep-dive report builder: `scripts/build_blue_catalyst_deep_dive_report.py`
- Apolo-3 SLURM launcher: `scripts/submit_blue_catalyst_poc_apolo3.sh`
- Artifact fetch utility: `scripts/fetch_apolo_blue_catalyst_artifacts.sh`

**Artifact locations**
- 662-genome run: `results/blue_catalyst_poc/runs/apolo_full_20260228_080644_embed_20260305_061952/artifacts/`
- 40-genome baseline: `results/blue_catalyst_poc/runs/apolo_20260226_194505/artifacts/`
- Latest snapshot: `results/blue_catalyst_poc/interim_snapshots/apolo_full_20260228_080644_embed_20260305_061952_notebook_interim_20260306_055012/`

**POC hardening**
- Batch/runtime reliability on Apolo-3 by forcing explicit `MethaNet311` Python for notebook execution
- Per-file tolerance for corrupted gzip and `prodigal` failures to prevent full-run aborts
- Embedding checkpointing every 25 genomes with resume support
- Numerical stability (NaN/Inf guards at every aggregation step)
- Portable checksum verification in artifact pulls (normalization of remote-path SHA entries)

---

## Feature Engineering

Planned feature matrices for genomic language model analysis (designed for immediate analysis after data access and QC):

| Feature Type | Description | Tools |
|--------------|-------------|-------|
| Pathway completeness | MCR complex, HdrABC completeness scores | KEGG, MetaCyc |
| Protein embeddings | Embeddings for mcrA/pmoA marker genes | [ESM-2](https://github.com/facebookresearch/esm), [GenomeOcean](https://doi.org/10.1101/2025.01.30.635558) |
| Gene co-occurrence | Network-based features from marker associations | Custom pipeline |
| Environmental covariates | Salinity, temperature, sediment depth | Paired metadata |

---

## Methodology

### Transfer Learning Approach
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  SOURCE DOMAIN  │     │   THE BRIDGE     │     │  TARGET DOMAIN  │
│                 │     │                  │     │                 │
│  Rumen Archaeome│ ──▶ │ Genomic Language │ ──▶ │ Coastal         │
│  998 genomes    │     │ Model + Domain   │     │ Ecosystems      │
│  Paired CH₄     │     │ Adaptation       │     │ 127 metagenomes │
│  measurements   │     │                  │     │ 23 with flux    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Key Molecular Markers

MethaNet uses a strategic set of **12 HMM markers** (TIGRFAMs v15.0) to capture methanogenesis, oxidation, and competition dynamics.

| Marker | Gene | Role | Strategic Value |
|--------|------|------|-----------------|
| **mcrA** | Methyl-coenzyme M reductase α | Methanogenesis | Universal production proxy |
| **pmoA** | pMMO α | Aerobic Oxidation | Primary sink proxy (Copper-rich) |
| **mmoX** | sMMO α | Soluble Oxidation | **"Copper-Switch"** for stressed environments |
| **mtaB** | Methanol MT | Methylotrophic | **"Sulfate Bypass"** in saline/mangrove systems |
| **dsrA** | Dissimilatory sulfite reductase | Sulfate Reduction | Competitive exclusion signal |

**The mcrA/pmoA ratio** (augmented by mmoX) captures the balance between methane production and consumption. The inclusion of **mtaB** and **dsrA** allows the model to adjust for the unique thermodynamic constraints of coastal ecosystems.

### Methanogenesis Pathways

Three primary pathways are conserved across environments:
- **Hydrogenotrophic**: CO₂ + H₂ → CH₄
- **Methylotrophic**: Methylated compounds → CH₄  
- **Aceticlastic**: Acetate → CH₄ + CO₂

Key archaeal families (e.g., *Methanomethylophilaceae*) are globally distributed despite divergent community compositions, enabling cross-ecosystem feature transfer.

---

## Repository Structure
```
MethaNet/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── uv.lock                 # Lockfile (reproducible installs)
├── configs/                 # Pipeline configuration (template/spec)
├── workflow/                # Snakemake pipeline (template/spec)
├── .github/                 # CI workflows
├── src/methanet/           # Core package
│   ├── features.py         # Feature extraction
│   ├── models.py           # Transfer learning models
│   └── utils.py            # Utilities
├── notebooks/              # Analysis notebooks
├── tests/                  # Unit tests
└── data/                   # Data directory (not in git)
```

---

## Pipeline Orchestration (Template Spec)

The Snakemake workflow is intentionally a **template/spec**. It is designed to be adapted once datasets, storage, and cloud deployment details are finalized.

**Guiding principles**
- **Config-first**: `configs/pipeline.yaml` drives paths, stage toggles, and model settings.
- **Stage gating**: Enable only the stages you can satisfy with available data; keep others off until inputs are ready.
- **Portable by default**: The pipeline avoids hard-coded infrastructure; cloud/HPC specifics belong in Snakemake profiles.

**Structure**
- `workflow/Snakefile` wires the end-to-end DAG.
- `workflow/rules/*.smk` are modular stage definitions (curation, annotation, embeddings, adaptation, prediction).
- `workflow/scripts/*.py` are thin, reusable utilities so rule logic stays stable while implementations evolve.

**Config highlights (template)**
- `stages`: toggle `data_curator`, `marker_annotator`, `embedding_generator`, `domain_adapter`, `flux_predictor`.
- `paths`: update to match your storage layout once datasets land.
- `sra_accessions`, `ena_accessions`, `assembly_samples`: placeholders for discovery outputs.
- `source_features`, `target_features`, `flux_features`: point to finalized Parquet feature tables.

**CI/CD posture**
- CI currently **lints and dry-runs** the workflow only (no data required).
- Cloud deployment and runtime profiles will be introduced once datasets and infrastructure are locked.

### POC report (transferability + gating)

The pipeline can emit a single per-run POC report artifact that summarizes:
- Missingness / finiteness rates by feature group (functional / ESM-2 / genome / fused)
- Domain shift metrics by group (A-distance + MMD)
- Cross-domain ablations (train on source, evaluate on target)
- A conservative gating recommendation for whether to allow fused embeddings

**Snakemake**

1. Enable the report stage:
   - `stages.poc_report: true` in `configs/pipeline.yaml`

2. Run the workflow:

```bash
snakemake -s workflow/Snakefile --configfile configs/pipeline.yaml --cores 4
```

Outputs:
- `reports/poc/poc_report.json` (single consolidated report)
- `reports/poc/domain_shift_by_group.csv`
- `reports/poc/feature_finiteness_by_group.csv`
- `reports/poc/cross_domain_transfer_by_group.csv`

**Direct script**

If you already have a fused feature table (`paths.flux_features`), you can run:

```bash
uv run python workflow/scripts/report_transferability.py \
  --features features/all_features.parquet \
  --output-dir reports/poc \
  --dnabert2-metrics-dir features/embeddings
```

Interpreting `poc_report.json`:
- `gating.recommended_features` is the suggested feature set to use.
- `gating.allow_embeddings` is `true` only when the report finds consistent evidence that fused embeddings help on the target domain.
- `gating.delta_mae` (and CI) is the estimated MAE difference on target for `all` minus `functional` (negative favors fused).
- `dnabert2_truncation` summarizes per-sample DNABERT-2 truncation diagnostics (when available).

---

## Installation

We use [uv](https://docs.astral.sh/uv/) for fast, reproducible dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/jaygut/MethaNet.git
cd MethaNet

# Sync dependencies and create virtual environment
uv sync

# Run with ML dependencies
uv sync --extra ml
```

### Running

```bash
# Run scripts directly (uv manages the environment)
uv run python -c "import methanet; print(methanet.__version__)"

# Or activate the virtual environment
source .venv/bin/activate  # macOS/Linux
```

### Dependencies

Core: numpy, pandas, scikit-learn, biopython
ML (optional): torch, transformers

---

## Timeline

| Phase | Date | Milestone |
|-------|------|-----------|
| ✅ | Q4 2025 | Project initiation, data curation |
| 🔄 | Q1 2026 | Feature matrix construction, QC |
| ⏳ | Q2 2026 | Model development, domain adaptation |
| ⏳ | Q3 2026 | Validation, preprint, data release |
| ⏳ | Q4 2026 | Field validation planning |

---

## Citation

If you use MethaNet in your research, please cite:
```bibtex
@software{methanet2025,
  author       = {Philosof, Alon and Gutierrez, Jay},
  title        = {{MethaNet: Transfer Learning for Methane Flux Prediction}},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/jaygut/MethaNet}
}
```

### Key References

This work builds on the following foundational datasets and methods:

1. **Ruminant Gut Archaeome**
   Mi, J., et al. (2024). A metagenomic catalogue of the ruminant gut archaeome. *Nature Communications*, 15, 9426.
   DOI: [10.1038/s41467-024-54025-3](https://doi.org/10.1038/s41467-024-54025-3)

2. **RUG2 Genome Catalog**
   Stewart, R.D., et al. (2019). Compendium of 4,941 rumen metagenome-assembled genomes for rumen microbiome biology and enzyme discovery. *Nature Biotechnology*, 37, 953–961.
   DOI: [10.1038/s41587-019-0202-3](https://doi.org/10.1038/s41587-019-0202-3)

3. **Hungate1000 Collection**
   Seshadri, R., et al. (2018). Cultivation and sequencing of rumen microbiome members from the Hungate1000 Collection. *Nature Biotechnology*, 36, 359–367.
   DOI: [10.1038/nbt.4110](https://doi.org/10.1038/nbt.4110)

4. **Global Methane Marker Atlas**
   Nwokolo, N.L. & Enebe, M.C. (2025). Methane production and oxidation: A review on the pmoA and mcrA gene abundances. *Pedosphere*, 35(1), 161-181.
   DOI: [10.1016/j.pedsph.2024.05.006](https://doi.org/10.1016/j.pedsph.2024.05.006)

5. **Transfer Learning for Microbial Communities**
   Chong, H., et al. (2022). EXPERT: transfer learning-enabled context-aware microbial community classification. *Briefings in Bioinformatics*, 23(6), bbac396.
   DOI: [10.1093/bib/bbac396](https://doi.org/10.1093/bib/bbac396)

6. **Mangrove Methanogen Genomics**
   Zhang, C.J., et al. (2020). Genomic and transcriptomic insights into methanogenesis potential of novel methanogens from mangrove sediments. *Microbiome*, 8, 94.
   DOI: [10.1186/s40168-020-00876-z](https://doi.org/10.1186/s40168-020-00876-z)

7. **MCR Complex Identification**
   Hallam, S.J., et al. (2003). Identification of methyl coenzyme M reductase A (mcrA) genes associated with methane-oxidizing archaea. *Applied and Environmental Microbiology*, 69(9), 5483-5491.
   DOI: [10.1128/AEM.69.9.5483-5491.2003](https://doi.org/10.1128/AEM.69.9.5483-5491.2003)

8. **Blue Carbon Methodology**
   Verra (2023). VM0033 Methodology for Tidal Wetland and Seagrass Restoration, v2.1.
   URL: [verra.org/methodologies/vm0033](https://verra.org/methodologies/vm0033)

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International License** (CC BY 4.0).

You are free to:
- **Share**: copy and redistribute the material in any medium or format
- **Adapt**: remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made.

See [LICENSE](LICENSE) for full details.

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

---

## Contributing

We welcome contributions from the research community.

### Ways to Contribute

- Report bugs or suggest features via [Issues](https://github.com/jaygut/MethaNet/issues)
- Contribute validation datasets with paired flux measurements
- Collaborate on methodology development

---

## Contact

**Principal Investigators:**

- **Alon Philosof, PhD** - Microbial Ecology & Computational Biology  
  ORCID: [0000-0003-2684-8678](https://orcid.org/0000-0003-2684-8678)  
  Email: aphilosof@gmail.com
  LinkedIn: [alon-philosof](https://www.linkedin.com/in/aphilosof/)

- **Jay Gutierrez, PhD** - Systems Biology & Biodiversity Informatics  
  ORCID: [0000-0003-0214-4641](https://orcid.org/0000-0003-0214-4641)  
  Email: jg@graphoflife.com  
  LinkedIn: [jay-gutierrez](https://www.linkedin.com/in/jaygut)  
  Website: https://biome-translator.emergent.host/

---

## Acknowledgments

We thank the DOE Joint Genome Institute, NCBI, and the broader microbiome research community for making foundational datasets publicly available.

---

<p align="center">
  <i>Advancing molecular verification for climate science</i>
</p>
