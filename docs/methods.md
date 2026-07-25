# MethaNet Methods

This document records implemented molecular-atlas methods alongside earlier
prediction-model specifications. The current validated product layer is MBAG
molecular attestation at MAG/proteome grain. It supports molecular
neighborhood navigation, functional evidence review, QC and provenance
guardrails, candidate cards, and monitoring-readiness design.

Calibrated sample or project methane-risk prediction requires exact sample
linkage, abundance or read coverage, environmental covariates, uncertainty
propagation, and paired field or process validation. Sections that describe
domain adaptation, ensemble classification, A to E tiers, or model export are
design specifications until those gates pass.

Use the following sources for current interpretation:

- [`methanet_positioning_and_claims.md`](methanet_positioning_and_claims.md)
  for product language and claim boundaries;
- [`current_artifact_inventory.md`](current_artifact_inventory.md) for
  release counts and artifact paths;
- [the public MBAG report](https://emergentbiome.earth/report/) for the current
  tri-view evidence contract and scientific reconciliation.

## Current MBAG Method Stack

The July 24, 2026 release links 7,965 registered MAG/proteome units through:

1. ESM-2 proteome embeddings for molecular-neighborhood navigation;
2. gLM2 native and shuffled genomic context under protocol-stratified
   comparison;
3. functional and expression payloads under explicit evidence contracts;
4. CheckM2, GUNC, GTDB-Tk, annotation coverage, source provenance, and
   missingness guardrails;
5. evidence cards that carry authorized claim wording, blocking gaps, and the
   next validation action.

The release contains 7,484 data-complete tri-views. The 625-unit POC core is
mechanism-comparable. The 4,358 mangrove rows await common accepted/present
feature aggregation. The 2,501 MUCC v1 rows retain their source-scaffold
functional contract.

## Table of Contents

1. [Feature Engineering](#1-feature-engineering)
2. [Domain Adaptation](#2-domain-adaptation)
3. [Ensemble Classification](#3-ensemble-classification)
4. [Statistical Analysis](#4-statistical-analysis)
5. [Model Export](#5-model-export)

---

## 1. Feature Engineering

### 1.1 Functional Gene Quantification

The marker-screening workflow quantifies 12 selected genes as
mechanism-relevant molecular evidence. Their presence records functional
potential. Activity and sample-level pathway balance require expression,
abundance, environmental, and process evidence.

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
- Current canonical run directory:
  `results/blue_catalyst_poc/runs/apolo_full_20260228_080644_embed_20260305_061952/`
- Current canonical artifacts:
  `results/blue_catalyst_poc/runs/apolo_full_20260228_080644_embed_20260305_061952/artifacts/`
- Current analytics summary:
  `results/blue_catalyst_poc/interim_snapshots/apolo_full_20260228_080644_embed_20260305_061952_notebook_interim_20260306_055012/analytics/analytics_summary.json`

Observed 662-genome metrics:
- Final embedded cohort: **662 genomes = 555 rumen + 107 wetland**.
- Input/source denominator: `sample_source_counts.tsv` reports 555 rumen +
  108 MUCC wetland before the final embedded cohort; one wetland coassembly/input
  record is excluded from the primary embedding denominator.
- Embedding matrix: `662 x 1280`, ESM2-650M layer 33 mean-pooled, zero attrition
  and zero non-finite vectors.
- **Silhouette**: 0.398.
- **PERMANOVA** (999 permutations, Euclidean distance): F=167.05, p=0.001,
  R2=0.202.
- **PCA**: PC1=44.4%, PC2=21.8%, PC3=11.0%.
- **Classifier separability**: wetland-vs-rumen AUC=1.0 and balanced
  accuracy=0.999 in cross-validation.
- **Bridge candidates**: top candidates are emitted in `bridging_genomes_top.tsv`
  with k=15 neighbor mixing metrics.

This is the current reproducible baseline for proposal and engineering work.
The earlier 40-genome run remains a historical smoke validation, not the
canonical denominator. The 662-genome interpretation must retain the caveat
that source and ecosystem are perfectly confounded: all rumen genomes come from
PRJEB31266 and all wetland genomes come from MUCC.

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
1. Extract embeddings for all proteins or marker proteins in a MAG.
2. Aggregate using final-layer mean pooling for POC/workflow parity.

Package-level multi-layer pooling remains configurable, but it is not the
default used for the 662-genome POC.

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

Total feature dimensionality: **2061** (DNABERT-2 default) or **4365** (GenomeOcean)

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Functional | 13 | 12 marker abundances plus `mcrA_pmoA_ratio` |
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
Input (D-dim; 2061 default with DNABERT-2)
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

D = 2061 by default (DNABERT-2); D = 4365 when using GenomeOcean embeddings.

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
