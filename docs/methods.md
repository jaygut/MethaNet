# MethaNet Methods

This document provides detailed technical descriptions of the computational methods used in MethaNet for predicting methane emission risk in coastal ecosystems.

## Table of Contents

1. [Feature Engineering](#1-feature-engineering)
2. [Domain Adaptation](#2-domain-adaptation)
3. [Ensemble Classification](#3-ensemble-classification)
4. [Statistical Analysis](#4-statistical-analysis)
5. [Model Export](#5-model-export)

---

## 1. Feature Engineering

### 1.1 Functional Gene Quantification

MethaNet quantifies an expanded set of **12 key marker genes** to capture complex methane dynamics, specifically addressing the "Sulfate Bypass" in saline environments and "Copper-Switch" oxidation.

| Marker | Function | Rationale | HMM Source |
|--------|----------|-----------|------------|
| **mcrA** | Methyl-coenzyme M reductase α | Core methanogenesis | TIGR03256 |
| **mcrB** | MCR beta subunit | Complex validation | TIGR03258 |
| **mcrG** | MCR gamma subunit | Complex validation | TIGR03259 |
| **pmoA** | pMMO alpha subunit | Aerobic oxidation (Copper-rich) | TIGR03080 |
| **mmoX** | sMMO alpha subunit | **Copper-switch** oxidation (stress) | TIGR01691 |
| **dsrA** | Sulfite reductase α | **Competitor** (Sulfate reduction) | TIGR02064 |
| **dsrB** | Sulfite reductase β | Competitor validation | TIGR02066 |
| **mtaB** | Methanol methyltransferase | **Sulfate Bypass** (Methylotrophic) | TIGR02626 |
| **mttB** | Methylamine methyltransferase | **Sulfate Bypass** (Methylotrophic) | TIGR02512 |
| **mtbA** | Methylcobalamin:CoM MT | **Sulfate Bypass** (Methylotrophic) | TIGR02506 |
| **nifH** | Nitrogenase iron protein | Normalization/Control | TIGR01287 |
| **cbbL** | RuBisCO large subunit | Normalization/Control | TIGR01168 |

**HMM Source:** All profiles are extracted from **TIGRFAMs v15.0** (JCVI/NCBI) to ensure consistent score thresholds and full-length equivalog specificity.

**Quantification pipeline:**

1. Open reading frame (ORF) prediction using Prodigal (`-p meta`) or FragGeneScanRs.
2. HMM search against the 12 marker profiles using HMMER 3.
3. Normalization per 1k proteins (counts / (total proteins / 1000)).
4. Computation of derived features:
   - log2(mcrA/pmoA) ratio with pseudocount.
   - Pathway completeness (inferred from subunit presence).

**Feature vector:**
- Normalized abundances for all 12 markers.
- log2(mcrA/pmoA) ratio.

### 1.2 Foundation Model Embeddings

#### Blue Catalyst POC implementation note (Apolo-3 validated)

The methods in this section were exercised end-to-end in the Blue Catalyst notebook workflow using:
- `notebooks/blue_catalyst_esm2_poc.ipynb`
- `scripts/submit_blue_catalyst_poc_apolo3.sh`

Validated run context:
- Source notebook: `notebooks/blue_catalyst_esm2_poc.ipynb`
- Executed output: `notebooks/blue_catalyst_esm2_poc.executed.ipynb`
- Run directory: `results/blue_catalyst_poc/runs/apolo_20260226_194505/`
- Extracted artifacts: `results/blue_catalyst_poc/runs/apolo_20260226_194505/artifacts/` (28 files)

Observed run metrics from this implementation pass:
- `n_samples=40`, `samples_embedded=40` (no non-finite vectors)
- `silhouette_non_noise=0.433`, cluster purity ~0.99
- `n_clusters_excluding_noise=5`, `noise_fraction=0.1`
- **PERMANOVA** (999 permutations, Euclidean distance): F=40.63, p=0.001, R²=0.517 — ecosystem label explains 51.7% of total embedding variance
- **PCA**: PC1=62.9%, PC1+PC2=78.0% cumulative variance; 3 PCs capture ≥80% variance
- **Ecosystem trajectory t-test** (projection onto rumen→wetland centroid axis): t=13.97, p=1.5e-16
- **Key bridge genome**: `rumen__10674_0001_idba_bin.23` (mixing_coeff=1.0, all k-NN neighbors in wetland cluster)
- Machine-readable summary: `advanced_analytics_summary.json`

This provides a concrete, reproducible reference implementation for proposal materials while broader MethaNet training and adaptation stages continue to evolve.

#### ESM-2 Protein Embeddings (1280 dimensions)

We use ESM-2 (facebook/esm2_t33_650M_UR50D) to generate protein-level embeddings:

```
Input: Protein sequences from marker genes
Model: ESM-2 650M parameter model
Layer: 33 (final layer)
Pooling: Mean across sequence length
Output: 1280-dimensional embedding per protein
```

**Genome-level aggregation:**
1. Extract embeddings for all marker proteins in a MAG
2. Aggregate using mean pooling (default; max pooling optional)

#### Genome Embeddings (DNABERT-2 default, GenomeOcean optional)

**DNABERT-2 (default, 768 dimensions)** provides genome-level representations from nucleotide sequences:

```
Input: Concatenated contig sequences
Model: DNABERT-2 (zhihan1996/DNABERT-2-117M)
K-mer size: 6
Output: 768-dimensional genome embedding
```

**GenomeOcean (optional, 3072 dimensions)** provides genome-level representations from nucleotide sequences:

```
Input: Concatenated contig sequences (k-mer tokenization)
Model: GenomeOcean foundation model
K-mer size: 6
Output: 3072-dimensional genome embedding
```

### 1.3 Feature Fusion

Total feature dimensionality: **2125** (DNABERT-2 default) or **4429** (GenomeOcean)

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Functional | 77 | HMM-based marker quantification |
| ESM-2 | 1280 | Protein language model embeddings |
| DNABERT-2 | 768 | Genomic foundation model embeddings (default) |
| GenomeOcean | 3072 | Genomic foundation model embeddings (optional) |
| Environmental | 1000 | Optional environmental covariates |

Fusion is performed via concatenation with optional PCA dimensionality reduction for computational efficiency.
Environmental covariates are not included in the default pipeline outputs.

---

## 2. Domain Adaptation

### 2.1 Problem Formulation

MethaNet addresses the domain shift between:
- **Source domain (S)**: Rumen microbiomes (998 MAGs, 412 with flux labels)
- **Target domain (T)**: Coastal ecosystems (127 samples, 23 with flux labels)

The goal is to learn a feature representation that minimizes domain discrepancy while preserving task-relevant information.

### 2.2 MMD-CORAL Weighted (MCW) Adaptation

We combine two domain adaptation objectives:

#### Maximum Mean Discrepancy (MMD)

MMD measures the distance between feature distributions using kernel mean embeddings:

```
L_MMD = ||μ_S - μ_T||²_H

where:
  μ_S = (1/n_S) Σ φ(x_s)  (source mean embedding)
  μ_T = (1/n_T) Σ φ(x_t)  (target mean embedding)
  φ(·) = Gaussian kernel feature map
```

**Multi-kernel MMD:**
We use a mixture of Gaussian kernels with bandwidth parameters {0.1, 0.5, 1.0, 2.0, 5.0} for robustness.

#### Correlation Alignment (CORAL)

CORAL aligns the second-order statistics (covariances) between domains:

```
L_CORAL = (1/4d²) ||C_S - C_T||²_F

where:
  C_S = covariance matrix of source features
  C_T = covariance matrix of target features
  d = feature dimensionality
```

#### Combined Loss Function

```
L_total = L_task + λ₁·L_MMD + λ₂·L_CORAL

Default hyperparameters:
  λ₁ = 0.1 (MMD weight)
  λ₂ = 0.1 (CORAL weight)
```

### 2.3 Network Architecture

```
Input (D-dim; 2125 default with DNABERT-2)
    │
    ▼
Linear(D → 1024) + BatchNorm + ReLU + Dropout(0.3)
    │
    ▼
Linear(1024 → 512) + BatchNorm + ReLU + Dropout(0.3)
    │
    ▼
Linear(512 → 256) + BatchNorm + ReLU + Dropout(0.3)
    │
    ├──────────────────┐
    ▼                  ▼
Classifier         Domain Loss
(5 classes)       (MMD + CORAL)
```

### 2.4 Training Procedure

1. **Pre-training**: Train on source domain only (100 epochs)
2. **Adaptation**: Fine-tune with combined loss (100 epochs)
3. **Early stopping**: Patience of 10 epochs on validation loss
4. **Optimizer**: AdamW with learning rate 1e-4

---

## 3. Ensemble Classification

### 3.1 Base Classifiers

MethaNet employs four diverse classifiers with optimized weights:

| Model | Weight | Description |
|-------|--------|-------------|
| XGBoost | 0.35 | Gradient boosting with tree ensembles |
| Neural Network | 0.30 | 3-layer MLP with dropout |
| Random Forest | 0.20 | Bagging ensemble of decision trees |
| FAISS k-NN | 0.15 | Similarity-based using L2 distance |

#### XGBoost Configuration

```python
{
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0
}
```

#### Neural Network Architecture

```
Input → Linear(D, 512) → ReLU → Dropout(0.3)
     → Linear(512, 256) → ReLU → Dropout(0.3)
     → Linear(256, 128) → ReLU → Dropout(0.3)
     → Linear(128, 5) → Softmax
```

#### Random Forest Configuration

```python
{
    "n_estimators": 500,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt"
}
```

#### FAISS k-NN Configuration

```python
{
    "k": 15,
    "distance_metric": "L2",
    "index_type": "IndexFlatL2"  # Exact search
}
```

### 3.2 Ensemble Aggregation

Probability aggregation using weighted voting:

```
P_ensemble(y|x) = Σ w_i · P_i(y|x)

where:
  w_i = model weight (sum to 1.0)
  P_i = probability from model i
```

### 3.3 Risk Tier Assignment

| Tier | Risk Level | Probability Range | Monitoring |
|------|------------|-------------------|------------|
| A | Very Low | [0.00, 0.10) | 24 months |
| B | Low | [0.10, 0.25) | 12 months |
| C | Moderate | [0.25, 0.45) | 6 months |
| D | Elevated | [0.45, 0.65) | 3 months |
| E | High | [0.65, 1.00] | 1 month |

---

## 4. Statistical Analysis

### 4.1 Bootstrap Confidence Intervals

We compute 95% confidence intervals using the percentile bootstrap method:

```
Algorithm: Bootstrap CI
Input: Model M, data X, n_iterations=1000, alpha=0.05
Output: (CI_lower, CI_upper)

1. For i = 1 to n_iterations:
   a. Sample X* from X with replacement
   b. Compute prediction score θ* = M(X*)
   c. Store θ*
2. Sort all θ* values
3. CI_lower = percentile(θ*, alpha/2 * 100)
4. CI_upper = percentile(θ*, (1 - alpha/2) * 100)
```

### 4.2 Classification Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| Balanced Accuracy | (1/K) Σ Recall_k | Macro-averaged recall |
| Macro F1 | (1/K) Σ F1_k | Macro-averaged F1 |
| AUC-ROC | Area under ROC curve | Discrimination ability |
| Cohen's Kappa | (p_o - p_e) / (1 - p_e) | Agreement beyond chance |

### 4.3 Transfer Metrics

| Metric | Description |
|--------|-------------|
| Accuracy Drop | Source accuracy - Target accuracy |
| Transfer Ratio | Target accuracy / Source accuracy |
| Domain Discrepancy | MMD and A-distance before/after adaptation |

---

## 5. Model Export

### 5.1 ONNX Export Specification

Models are exported to ONNX format for production deployment:

```
ONNX Configuration:
  - Opset version: 17
  - Dynamic batch size: Yes
  - Constant folding: Enabled
  - Input: "features" (batch_size, D)
  - Output: "probabilities" (batch_size, 5)
```

D = 2125 by default (DNABERT-2); D = 4429 when using GenomeOcean embeddings.

### 5.2 Inference Pipeline

```
┌─────────────────┐
│  Raw Features   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  (Normalization)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ONNX Runtime   │
│    Inference    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Risk Tier      │
│  Assignment     │
└─────────────────┘
```

---

## References

1. Lin, T.Y., et al. (2017). Focal loss for dense object detection. ICCV.
2. Sun, B., & Saenko, K. (2016). Deep CORAL: Correlation alignment for deep domain adaptation. ECCV Workshops.
3. Long, M., et al. (2015). Learning transferable features with deep adaptation networks. ICML.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
5. Rives, A., et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. PNAS.
6. Pfam 37.0 release (2024). Xfam Blog. https://xfam.wordpress.com/2024/06/06/pfam-37-0-release/
7. TIGRFAMs release 15.0 (2014). NCBI. https://www.ncbi.nlm.nih.gov/refseq/annotation_prok/tigrfams/
8. GenomeOcean (2025). bioRxiv. https://www.biorxiv.org/content/10.1101/2025.01.30.635558v1
