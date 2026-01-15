# MethaNet: Transfer Learning for Methane Flux Prediction

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
![Version](https://img.shields.io/badge/Version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.11%2B-green)

*MethaNet is a flagship project within the [EmergentBiome](https://emergent.host/) research initiative, developing cross-biome knowledge transfer methods for climate science.*

**Bridging Rumen Microbiome Data to Climate Verification Through Cross-Ecosystem Genomic Analysis**

---

## Overview

MethaNet is a computational framework for predicting methane emission risk in blue carbon ecosystems using microbial functional gene signatures and foundation model embeddings. The system leverages transfer learning from data-rich rumen microbiome studies to predict methane dynamics in data-sparse coastal ecosystems.

Methane has a global warming potential approximately 30× that of CO₂ over a 100-year horizon. Coastal wetlands can be net carbon sinks or sources depending on the balance between carbon uptake and methane emissions. Current measurement methods (chamber measurements, flux towers) are expensive, sparse, and unable to scale. MethaNet addresses this critical gap in climate accounting through molecular prediction.

### Core Innovation

The system uses **MMD-CORAL Weighted (MCW) domain adaptation** to transfer knowledge from:
- **Source domain**: 998 rumen MAGs with 412 paired flux measurements
- **Target domain**: 127 coastal samples with 23 paired flux measurements

The key insight is that methanogenic pathway conservation across environments enables cross-domain prediction using the **mcrA/pmoA ratio** as the primary molecular signal.

---

## Quick Start

```python
from methanet import MethaNetEnsemble, RiskTier

# Load trained ensemble
ensemble = MethaNetEnsemble()
ensemble.fit(X_train, y_train)

# Classify risk for new samples
results = ensemble.classify_risk(X_test)

for result in results:
    print(f"Risk Tier: {result.risk_tier.name}")
    print(f"Score: {result.score:.3f}")
    print(f"Confidence: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
    print(f"Monitoring: {result.risk_tier.monitoring_interval}")
```

---

## Risk Classification System

MethaNet classifies samples into five risk tiers based on methane emission probability:

| Tier | Risk Level | Probability | Monitoring Interval |
|------|------------|-------------|---------------------|
| A | Very Low | 0-10% | 24 months |
| B | Low | 10-25% | 12 months |
| C | Moderate | 25-45% | 6 months |
| D | Elevated | 45-65% | 3 months |
| E | High | 65-100% | 1 month |

---

## Architecture

### Ensemble Classification

MethaNet uses a weighted ensemble of four classifiers:

```
┌──────────────────────────────────────────────────────────────┐
│                Feature Vector (2125-dim default)             │
│  ┌──────────────┬──────────────┬──────────────┐              │
│  │ Functional   │   ESM-2      │  DNABERT-2   │              │
│  │  (77-dim)    │ (1280-dim)   │  (768-dim)   │              │
│  └──────┬───────┴──────┬───────┴──────┬───────┘              │
│         └──────────────┼──────────────┘                       │
│                        ▼                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ XGBoost  │  │  Neural  │  │  Random  │  │  FAISS   │      │
│  │  (0.35)  │  │   Net    │  │  Forest  │  │Similarity│      │
│  │          │  │  (0.30)  │  │  (0.20)  │  │  (0.15)  │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       └─────────────┴─────────────┴─────────────┘             │
│                           │                                   │
│                           ▼                                   │
│              ┌─────────────────────────┐                      │
│              │   Weighted Ensemble     │                      │
│              │   + Bootstrap CI (95%)  │                      │
│              └────────────┬────────────┘                      │
│                           ▼                                   │
│              ┌─────────────────────────┐                      │
│              │   Risk Tier (A-E)       │                      │
│              └─────────────────────────┘                      │
└──────────────────────────────────────────────────────────────┘
```

GenomeOcean is supported as an optional genome backend (3072-dim), producing 4429-dim fused vectors.

### Key Performance Metrics

- **Balanced Accuracy**: 0.847 (coastal validation)
- **AUC-ROC**: 0.89
- **Bootstrap CI**: 95% confidence intervals
- **Transfer Ratio**: 0.92 (target/source performance)

---

## Datasets

### Source Domain: Ruminant Gut Archaeome

| Resource | Description | Size | Source |
|----------|-------------|------|--------|
| Ruminant Gut Archaeome | Curated archaeal genomes from ruminant gut systems | 998 genomes | [Mi et al., 2024](https://doi.org/10.1038/s41467-024-54025-3) |
| RUG2 Catalog | Metagenome-assembled genomes from rumen | 4,941 MAGs | [Stewart et al., 2019](https://doi.org/10.1038/s41587-019-0202-3) |
| Hungate1000 | Cultivated rumen microbiome isolates | 410 genomes | [Seshadri et al., 2018](https://doi.org/10.1038/nbt.4110) |

### Target Domain: Coastal Ecosystems

| Dataset | Description | Size |
|---------|-------------|------|
| Global Mangrove Metagenomes | Curated coastal sediment samples | 127 samples |
| Paired Flux Validation | Samples with chamber/tower measurements | 23 samples |

---

## Climate and MRV Alignment

- IPCC 2019 Refinement Wetlands guidance: https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/4_Volume4/19R_V4_Ch07_Wetlands.pdf
- IPCC 2013 Wetlands Supplement: https://www.ipcc.ch/publication/2013-supplement-to-the-2006-ipcc-guidelines-for-national-greenhouse-gas-inventories-wetlands/
- Verra VM0033 v2.1 methodology: https://verra.org/methodologies/vm0033-methodology-for-tidal-wetland-and-seagrass-restoration-v2-1/

---

## Installation

We use [uv](https://docs.astral.sh/uv/) for fast, reproducible dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/jaygut/MethaNet.git
cd MethaNet

# Basic installation
uv sync

# With ML dependencies (PyTorch, transformers)
uv sync --extra ml

# With all dependencies
uv sync --extra all

# Development installation
uv sync --extra dev
```

### Optional Dependency Groups

| Group | Dependencies | Use Case |
|-------|--------------|----------|
| `ml` | torch, transformers, xgboost, faiss-cpu | Model training |
| `bioinformatics` | pyhmmer, pyfaidx | Gene quantification |
| `embeddings` | torch, transformers, umap-learn | Embedding generation |
| `prediction` | xgboost, lightgbm, shap | Flux prediction |
| `api` | onnx, onnxruntime, skl2onnx | Production deployment |
| `dev` | pytest, ruff, mypy | Development |

---

## Repository Structure

```
MethaNet/
├── src/
│   ├── methanet/                 # Core Python package
│   │   ├── __init__.py           # Package exports
│   │   ├── functional/           # HMM-based gene quantification
│   │   │   └── quantify.py       # mcrA, pmoA, dsrA, nifH, cbbL
│   │   ├── embedding/            # Foundation model embeddings
│   │   │   ├── esm2.py           # ESM-2 protein embeddings
│   │   │   ├── genomeocean.py    # GenomeOcean genome embeddings (optional)
│   │   │   └── fusion.py         # Feature fusion (2125-dim default)
│   │   ├── domain_adapt/         # Transfer learning
│   │   │   ├── losses.py         # MMD and CORAL losses
│   │   │   └── mcw.py            # MCW domain adaptation
│   │   ├── classification/       # Ensemble classifier
│   │   │   ├── ensemble.py       # 4-model weighted ensemble
│   │   │   └── risk_tiers.py     # Risk tier definitions
│   │   └── stats/                # Statistical analysis
│   │       ├── bootstrap.py      # Bootstrap confidence intervals
│   │       └── metrics.py        # Classification/regression metrics
│   └── api_bridge/               # Production deployment
│       ├── export_onnx.py        # ONNX export utilities
│       └── inference.py          # ONNX inference runtime
├── workflow/                     # Snakemake pipeline
│   ├── Snakefile
│   ├── rules/                    # Modular rule definitions
│   ├── scripts/                  # Pipeline scripts
│   └── envs/                     # Conda environments
├── configs/
│   └── pipeline.yaml             # Pipeline configuration
├── models/                       # Model artifacts
│   ├── trained/                  # Trained model weights
│   └── onnx/                     # Production ONNX exports
├── data/                         # Data directory (not in git)
├── results/                      # Output directory
├── tests/                        # Unit tests
└── docs/                         # Documentation
```

---

## Key Molecular Markers

| Marker | Gene | Function | Role |
|--------|------|----------|------|
| mcrA | Methyl-coenzyme M reductase α | Final step of methanogenesis | Methanogen proxy |
| pmoA | Particulate methane monooxygenase α | Methane oxidation | Methanotroph proxy |
| dsrA | Dissimilatory sulfite reductase α | Sulfate reduction | Sulfate-reducing bacteria |
| nifH | Nitrogenase iron protein | Nitrogen fixation | Diazotroph proxy |
| cbbL | RuBisCO large subunit | Carbon fixation | Autotroph proxy |

The **mcrA/pmoA ratio** captures the balance between methane production and consumption—the key determinant of whether an ecosystem is a net methane source or sink.
In the pipeline, marker counts are normalized per 1k proteins and the ratio is log2-transformed with a pseudocount.
HMM profiles should be pinned to Pfam 37.0 (https://xfam.wordpress.com/2024/06/06/pfam-37-0-release/) and TIGRFAM 15.0 (final JCVI release; https://www.ncbi.nlm.nih.gov/refseq/annotation_prok/tigrfams/) for reproducibility.

---

## API Usage

### Functional Gene Quantification

```python
from methanet.functional import FunctionalQuantifier

quantifier = FunctionalQuantifier(
    hmm_dir="data/hmm",
    markers=["mcrA", "pmoA", "dsrA"]
)
profile = quantifier.quantify_mag("genome.faa")
print(f"mcrA/pmoA ratio: {profile.mcra_pmoa_ratio}")
```

### Embedding Generation

```python
from methanet.embedding import ESM2Embedder, EmbeddingConfig

config = EmbeddingConfig(
    model_name="facebook/esm2_t33_650M_UR50D",
    batch_size=8,
    device="cuda"
)
embedder = ESM2Embedder(config)
embeddings = embedder.embed_proteins(sequences, ids)
```

### Domain Adaptation

```python
from methanet.domain_adapt import DomainAdapter, DomainAdaptConfig

config = DomainAdaptConfig(
    mmd_weight=0.1,
    coral_weight=0.1,
    epochs=100
)
adapter = DomainAdapter(config)
adapter.fit(source_features, target_features, source_labels)
adapted_features = adapter.transform(target_features)
```

### Risk Classification

```python
from methanet import MethaNetEnsemble, EnsembleConfig

config = EnsembleConfig(
    model_weights={
        "xgboost": 0.35,
        "neural_net": 0.30,
        "random_forest": 0.20,
        "faiss_knn": 0.15
    },
    bootstrap_iterations=1000
)
ensemble = MethaNetEnsemble(config)
results = ensemble.classify_risk(features)
```

### ONNX Export for Production

```python
from api_bridge import export_neural_net_to_onnx, ONNXInference

# Export trained model
export_neural_net_to_onnx(
    model=trained_model,
    input_dim=2125,
    output_path="models/onnx/methanet.onnx"
)

# Use input_dim=4429 if you fuse GenomeOcean embeddings.

# Production inference
inference = ONNXInference("models/onnx/methanet.onnx")
predictions = inference.predict(features)
```

---

## Pipeline Execution

```bash
# Configure pipeline
vim configs/pipeline.yaml

# Dry run to verify DAG
snakemake -n

# Execute pipeline
snakemake --cores 32 --use-conda

# Generate reports
snakemake --report results/report.html
```

---

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=methanet

# Run specific test module
uv run pytest tests/test_ensemble.py

# Lint code
uv run ruff check src/
```

---

## Citation

If you use MethaNet in your research, please cite:

```bibtex
@software{methanet2025,
  author       = {Philosof, Alon and Gutierrez, Jay},
  title        = {{MethaNet: Transfer Learning for Methane Flux Prediction}},
  year         = {2025},
  version      = {1.0.0},
  publisher    = {GitHub},
  url          = {https://github.com/jaygut/MethaNet}
}
```

### Key References

1. **Ruminant Gut Archaeome** - Mi, J., et al. (2024). *Nature Communications*, 15, 9426. [DOI](https://doi.org/10.1038/s41467-024-54025-3)
2. **RUG2 Genome Catalog** - Stewart, R.D., et al. (2019). *Nature Biotechnology*, 37, 953-961. [DOI](https://doi.org/10.1038/s41587-019-0202-3)
3. **Hungate1000 Collection** - Seshadri, R., et al. (2018). *Nature Biotechnology*, 36, 359-367. [DOI](https://doi.org/10.1038/nbt.4110)
4. **Methane Marker Atlas** - Nwokolo, N.L. & Enebe, M.C. (2025). *Pedosphere*, 35(1), 161-181. [DOI](https://doi.org/10.1016/j.pedsph.2024.05.006)
5. **Transfer Learning for Microbiomes** - Chong, H., et al. (2022). *Briefings in Bioinformatics*, 23(6). [DOI](https://doi.org/10.1093/bib/bbac396)
6. **Mangrove Methanogens** - Zhang, C.J., et al. (2020). *Microbiome*, 8, 94. [DOI](https://doi.org/10.1186/s40168-020-00876-z)

---

## License

This project is licensed under **CC BY 4.0**. You are free to share and adapt this work with appropriate attribution.

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

---

## Contact

**Principal Investigators:**

- **Alon Philosof, PhD** - Microbial Ecology & Computational Biology
  [ORCID](https://orcid.org/0000-0003-2684-8678) | [Email](mailto:aphilosof@gmail.com)

- **Jay Gutierrez, PhD** - Systems Biology & Biodiversity Informatics
  [ORCID](https://orcid.org/0000-0003-0214-4641) | [Email](mailto:jg@graphoflife.com)

---

<p align="center">
  <i>Advancing molecular verification for climate science</i><br>
  Part of the <a href="https://emergent.host/">EmergentBiome</a> research initiative
</p>
