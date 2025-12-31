# MethaNet: Transfer Learning for Methane Flux Prediction

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

**Bridging Rumen Microbiome Data to Climate Verification Through AI-Accelerated Knowledge Transfer**

---

## Overview

MethaNet is a research initiative developing computational methods to predict net methane flux in coastal ecosystems using transfer learning from agricultural microbiome data. By leveraging the world's most comprehensive methanogen genomic resources—the ruminant gut archaeome—we aim to decode complex coastal wetland systems that are critical for carbon sequestration but remain data-sparse.

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
| Global Mangrove Metagenomes | ~150 publicly available samples | NCBI SRA |
| Mangrove Methanogen Study | 13 MAGs with pathway analysis | [Zhang et al., 2020](https://doi.org/10.1186/s40168-020-00876-z) |

**Data disparity:** ~26,000 rumen microbiome sequencing runs exist in NCBI SRA compared to ~2,400 from mangrove sites—a >10× disparity that motivates our transfer learning approach.

**Validation strategy:** We are curating publicly available coastal metagenomes with co-located flux tower or chamber measurements for model validation.

---

## Feature Engineering

Planned feature matrices for genomic language model analysis:

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

| Marker | Gene | Function | Role |
|--------|------|----------|------|
| mcrA | Methyl-coenzyme M reductase α | Catalyzes final step of methanogenesis | Methanogen abundance proxy |
| pmoA | Particulate methane monooxygenase α | Catalyzes methane oxidation | Methanotroph abundance proxy |

**The mcrA/pmoA ratio** captures the balance between methane production and consumption—the key determinant of whether an ecosystem is a net methane source or sink.

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
├── src/methanet/           # Core package
│   ├── features.py         # Feature extraction
│   ├── models.py           # Transfer learning models
│   └── utils.py            # Utilities
├── notebooks/              # Analysis notebooks
├── tests/                  # Unit tests
└── data/                   # Data directory (not in git)
```

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
   Nwokolo, N.L. & Enebe, M.C. (2025). Methane production and oxidation—A review on the pmoA and mcrA gene abundances. *Pedosphere*, 35(1), 161-181.
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
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

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

- **Alon Philosof, PhD** — Microbial Ecology & Computational Biology  
  ORCID: [0000-0003-2684-8678](https://orcid.org/0000-0003-2684-8678)
  Email: aphilosof@gmail.com
  LinkedIn: [alon-philosof](https://www.linkedin.com/in/aphilosof/)

- **Jay Gutierrez, PhD** — Systems Biology & Biodiversity Informatics  
  ORCID: [0000-0003-0214-4641](https://orcid.org/0000-0003-0214-4641)
  Email: jg@graphoflife.com  
  LinkedIn: [jay-gutierrez](https://www.linkedin.com/in/jaygut)

---

## Acknowledgments

We thank the DOE Joint Genome Institute, NCBI, and the broader microbiome research community for making foundational datasets publicly available.

---

<p align="center">
  <i>Advancing molecular verification for climate science</i>
</p>