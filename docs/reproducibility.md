# MethaNet Reproducibility Guide

This document provides comprehensive instructions for reproducing the MethaNet analysis results, from raw data acquisition through final model validation.

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup](#2-environment-setup)
3. [Data Acquisition](#3-data-acquisition)
4. [Pipeline Execution](#4-pipeline-execution)
5. [Model Training](#5-model-training)
6. [Validation](#6-validation)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8 cores | 32 cores |
| RAM | 32 GB | 128 GB |
| GPU | - | NVIDIA A100 40GB |
| Storage | 500 GB | 2 TB SSD |

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.9+ | Core runtime |
| uv | 0.4+ | Package management |
| Snakemake | 8.0+ | Workflow orchestration |
| Conda/Mamba | 24.0+ | Environment isolation |
| Git | 2.40+ | Version control |
| CUDA | 12.0+ | GPU acceleration (optional) |

---

## 2. Environment Setup

### 2.1 Clone Repository

```bash
git clone https://github.com/jaygut/MethaNet.git
cd MethaNet
```

### 2.2 Install Dependencies

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --extra all

# Verify installation
uv run python -c "import methanet; print(f'MethaNet v{methanet.__version__}')"
```

### 2.3 Configure Conda Environments

For Snakemake workflow execution:

```bash
# Install mamba for faster environment resolution
conda install -n base -c conda-forge mamba

# Pre-create environments (optional, speeds up first run)
mamba env create -f workflow/envs/qc.yaml
mamba env create -f workflow/envs/assembly.yaml
mamba env create -f workflow/envs/binning.yaml
mamba env create -f workflow/envs/ml.yaml
mamba env create -f workflow/envs/viz.yaml
```

### 2.4 Download HMM Profiles

```bash
# Create HMM directory
mkdir -p data/hmm

# Download marker gene HMMs from Pfam 37.0 and TIGRFAM 15.0
# (Final TIGRFAM release hosted by NCBI; Pfam 37.0 via Xfam)
wget -O data/hmm/mcrA.hmm https://example.com/mcrA.hmm
wget -O data/hmm/pmoA.hmm https://example.com/pmoA.hmm
wget -O data/hmm/dsrA.hmm https://example.com/dsrA.hmm
wget -O data/hmm/nifH.hmm https://example.com/nifH.hmm
wget -O data/hmm/cbbL.hmm https://example.com/cbbL.hmm
```

---

## 3. Data Acquisition

### 3.1 Source Domain: Rumen Archaeome

Download the Ruminant Gut Archaeome catalogue:

```bash
# Create data directory
mkdir -p data/raw/rumen

# Download genome catalogue (998 MAGs)
# Source: https://doi.org/10.1038/s41467-024-54025-3
wget -O data/raw/rumen/archaeome_catalogue.tar.gz \
  https://figshare.com/ndownloader/files/XXXXXXX

# Extract
tar -xzf data/raw/rumen/archaeome_catalogue.tar.gz -C data/raw/rumen/

# Download flux measurements
wget -O data/raw/rumen/flux_measurements.tsv \
  https://figshare.com/ndownloader/files/YYYYYYY
```

### 3.2 Target Domain: Coastal Metagenomes

Download coastal sediment metagenomes from NCBI SRA:

```bash
# Install SRA toolkit (if not installed)
conda install -c bioconda sra-tools

# Create accession list from configs/samples.tsv
cut -f1 configs/samples.tsv | tail -n+2 > accessions.txt

# Batch download
prefetch --option-file accessions.txt -O data/raw/coastal/

# Convert to FASTQ
fasterq-dump --split-files --outdir data/raw/coastal/ \
  data/raw/coastal/SRR*/
```

### 3.3 Prepare Sample Manifest

Create `configs/samples.tsv`:

```tsv
sample_id	domain	ecosystem	has_flux	flux_value	latitude	longitude
SRR12345678	rumen	cattle	TRUE	45.2	-	-
SRR12345679	rumen	sheep	TRUE	32.1	-	-
SRR23456789	coastal	mangrove	TRUE	12.4	25.7	-80.2
SRR23456790	coastal	mangrove	FALSE	-	25.8	-80.1
```

---

## 4. Pipeline Execution

### 4.1 Configure Pipeline

Edit `configs/pipeline.yaml`:

```yaml
# Enable stages as data becomes available
stages:
  data_curator: true
  marker_annotator: true
  embedding_generator: true
  domain_adapter: true
  flux_predictor: true

# Update paths to match your storage layout
paths:
  raw_sra: data/raw/coastal
  mags: data/mags
  embeddings: features/embeddings

# Genome embedding backend (default)
embedding:
  genome_backend: dnabert2
  genome_dim: 768

# Functional normalization
functional:
  normalization: per_1k_proteins
```

If you set `embedding.genome_backend: genomeocean`, provide per-sample
`features/embeddings/{sample}_genomeocean.npy` files or add a custom rule.

### 4.2 Validate Configuration

```bash
# Dry run to check DAG
snakemake -n --configfile configs/pipeline.yaml

# Print rule graph
snakemake --rulegraph | dot -Tpng > workflow_dag.png
```

### 4.3 Execute Pipeline

```bash
# Full pipeline execution
snakemake --cores 32 --use-conda --configfile configs/pipeline.yaml

# With cluster execution (SLURM)
snakemake --cores 32 --use-conda \
  --cluster "sbatch -p gpu -N 1 -c {threads} --mem={resources.mem_mb}M" \
  --jobs 10 --configfile configs/pipeline.yaml

# Resume after failure
snakemake --cores 32 --use-conda --configfile configs/pipeline.yaml --rerun-incomplete
```

### 4.4 Monitor Progress

```bash
# Watch log file
tail -f .snakemake/log/*.log

# Check completed rules
snakemake --summary --configfile configs/pipeline.yaml
```

---

## 5. Model Training

### 5.1 Feature Extraction

```python
import numpy as np
from methanet.functional import FunctionalQuantifier
from methanet.embedding import ESM2Embedder, EmbeddingConfig
from methanet.embedding import FeatureFusion, FusionConfig

# Quantify functional genes
quantifier = FunctionalQuantifier(hmm_dir="data/hmm")
functional_features = []
for mag in mags:
    profile = quantifier.quantify_mag(mag)
    functional_features.append(profile.to_array())

# Generate embeddings
config = EmbeddingConfig(
    model_name="facebook/esm2_t33_650M_UR50D",
    batch_size=8,
    device="cuda"
)
embedder = ESM2Embedder(config)
esm2_features = embedder.embed_all_mags(protein_fastas)

# Fuse features (DNABERT-2 default)
dnabert2_features = np.load("features/embeddings/dnabert2.npy")  # (n, 768)
fusion = FeatureFusion(FusionConfig(genome_dim=768, include_metadata=False))
X = []
for i, profile in enumerate(functional_features):
    X.append(
        fusion.fuse(
            sample_id=str(i),
            functional=profile,
            esm2_embedding=esm2_features[i],
            genomeocean_embedding=dnabert2_features[i],
        ).features
    )
X = np.vstack(X)
```

### 5.2 Domain Adaptation

```python
from methanet.domain_adapt import DomainAdapter, DomainAdaptConfig

# Split by domain
X_source = X[domain_labels == "rumen"]
X_target = X[domain_labels == "coastal"]
y_source = flux_labels[domain_labels == "rumen"]

# Configure adaptation
config = DomainAdaptConfig(
    mmd_weight=0.1,
    coral_weight=0.1,
    epochs=100,
    patience=10
)

# Fit adapter
adapter = DomainAdapter(config)
adapter.fit(X_source, X_target, y_source)

# Transform features
X_adapted = adapter.transform(X_target)
```

### 5.3 Train Ensemble

```python
from methanet import MethaNetEnsemble, EnsembleConfig
from sklearn.model_selection import StratifiedKFold

# Configure ensemble
config = EnsembleConfig(
    model_weights={
        "xgboost": 0.35,
        "neural_net": 0.30,
        "random_forest": 0.20,
        "faiss_knn": 0.15
    },
    bootstrap_iterations=1000,
    random_state=42
)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for train_idx, val_idx in cv.split(X_train, y_train):
    ensemble = MethaNetEnsemble(config)
    ensemble.fit(X_train[train_idx], y_train[train_idx])
    results = ensemble.classify_risk(X_train[val_idx])
    cv_results.append(compute_metrics(results, y_train[val_idx]))

# Final model
final_ensemble = MethaNetEnsemble(config)
final_ensemble.fit(X_train, y_train)
```

### 5.4 Save Models

```python
import joblib
from api_bridge import export_neural_net_to_onnx

# Save sklearn models
joblib.dump(final_ensemble.models, "models/trained/ensemble.joblib")

# Export neural net to ONNX
export_neural_net_to_onnx(
    final_ensemble.models["neural_net"],
    input_dim=2125,
    output_path="models/onnx/neural_net.onnx"
)

# Use input_dim=4429 if GenomeOcean embeddings are enabled.
```

---

## 6. Validation

### 6.1 Performance Metrics

```python
from methanet.stats import compute_classification_metrics, compute_transfer_metrics

# Classify test set
test_results = final_ensemble.classify_risk(X_test)
y_pred = [r.predicted_tier for r in test_results]
y_proba = [r.probabilities for r in test_results]

# Compute metrics
metrics = compute_classification_metrics(
    y_true=y_test,
    y_pred=y_pred,
    y_proba=np.array(y_proba),
    class_names=["A", "B", "C", "D", "E"]
)

print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
print(f"Macro F1: {metrics['macro_f1']:.3f}")
print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
```

### 6.2 Expected Results

| Metric | Source (Rumen) | Target (Coastal) |
|--------|----------------|------------------|
| Balanced Accuracy | 0.89 | 0.847 |
| Macro F1 | 0.87 | 0.82 |
| AUC-ROC | 0.94 | 0.89 |
| Transfer Ratio | - | 0.92 |

### 6.3 Bootstrap Confidence Intervals

```python
from methanet.stats import bootstrap_ci

# Compute CI for balanced accuracy
ci_lower, ci_upper = bootstrap_ci(
    metric_fn=lambda y_t, y_p: balanced_accuracy_score(y_t, y_p),
    y_true=y_test,
    y_pred=y_pred,
    n_iterations=1000,
    alpha=0.05
)

print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

---

## 7. Troubleshooting

### Common Issues

#### Out of Memory (OOM)

```bash
# Reduce batch size in configs/pipeline.yaml
embedding:
  batch_size: 4  # Reduce from 8

# Or use CPU instead of GPU
embedding:
  device: cpu
```

#### Snakemake Lock Error

```bash
# Unlock directory
snakemake --unlock --configfile configs/pipeline.yaml
```

#### Missing HMM Profiles

```bash
# Check HMM directory
ls -la data/hmm/

# Verify HMM format
hmmstat data/hmm/mcrA.hmm
```

#### CUDA Version Mismatch

```bash
# Check CUDA version
nvcc --version

# Install matching PyTorch
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Getting Help

- Open an issue: https://github.com/jaygut/MethaNet/issues
- Email: jg@graphoflife.com

---

## Reproducibility Checklist

- [ ] Environment setup complete (`uv sync --extra all`)
- [ ] Conda environments created (`workflow/envs/*.yaml`)
- [ ] HMM profiles downloaded (`data/hmm/*.hmm`)
- [ ] Sample manifest created (`configs/samples.tsv`)
- [ ] Pipeline configuration verified (`snakemake -n`)
- [ ] Raw data downloaded (source and target domains)
- [ ] Pipeline executed successfully
- [ ] Models trained and saved
- [ ] Validation metrics computed
- [ ] Results match expected values (within CI)

---

## Version Information

```
MethaNet: 1.0.0
Python: 3.11
PyTorch: 2.1.0
scikit-learn: 1.4.0
Snakemake: 8.0.0
```

Document last updated: January 2026
