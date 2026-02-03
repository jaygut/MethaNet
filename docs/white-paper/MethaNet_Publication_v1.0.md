# MethaNet: A Metagenomic Framework for Carbon Permanence Risk Classification in Blue Carbon Ecosystems

**Authors:**
Alon Philosof^1,2,*^, Jay Gutierrez^1,3,*^

^1^EmergentBiome, San Francisco, CA, USA
^2^Department of Microbial Ecology, [Institution]
^3^Department of Systems Biology, [Institution]
*Corresponding authors: [email addresses]

---

## Abstract

The voluntary carbon market has grown to exceed $2 billion annually, yet blue carbon ecosystems (among the most efficient carbon sinks on Earth) represent less than 1% of total transactions due to verification challenges. A critical barrier is the inability to reliably quantify methane emissions, which can offset 20-98% of sequestered carbon depending on ecosystem conditions. Current Monitoring, Reporting, and Verification (MRV) methodologies rely on expensive flux tower installations and historical proxy-based permanence assumptions that lack biological grounding. Here we present MethaNet, a transfer learning framework that leverages functional gene conservation across methanogen communities to provide scalable permanence risk classification. By training on 998 high-quality metagenome-assembled genomes from rumen ecosystems with paired methane flux measurements, we transfer predictive models to data-sparse coastal ecosystems. We validate our approach on 127 coastal wetland samples spanning mangrove, seagrass, salt marsh, and peatland ecosystems across Indo-Pacific, Caribbean, West African, and Brazilian sites. Our ensemble classification approach, combining gradient boosting, neural networks, and foundation model-based similarity search, achieves cross-ecosystem transfer with risk tier assignments (A-E) accompanied by bootstrap-derived confidence intervals. The mcrA/pmoA functional gene ratio emerges as a robust leading indicator of net methane flux potential (R² = 0.72, p < 0.001), providing a mechanistic, scalable alternative to retrospective flux measurement. MethaNet outputs directly address ICVCM Assessment Framework Criteria 9.1-9.3 for permanence, facilitating integration with existing carbon registry verification workflows. This work proposes functional ecology-based carbon verification as an approach for nature-based solution assessment that merits further validation.

**Keywords:** blue carbon, methane, metagenomics, transfer learning, carbon verification, permanence, mcrA, pmoA, foundation models, ICVCM

---

## 1. Introduction

### 1.1 The Permanence Verification Crisis in Carbon Markets

The voluntary carbon market has experienced rapid growth, reaching $2 billion in annual transactions with projections suggesting continued expansion as corporate net-zero commitments accelerate^1^. Within this market, blue carbon ecosystems (mangroves, seagrasses, salt marshes, and other coastal wetlands) command premium pricing of $25-30 per tonne CO₂e, approximately 4-5 times higher than terrestrial forest credits^2^. This premium reflects both the exceptional carbon sequestration rates of coastal ecosystems (up to 50 times faster per unit area than terrestrial forests) and the co-benefits they provide including coastal protection, biodiversity habitat, and fisheries support^3,4^.

Despite their recognized value, blue carbon projects represent less than 1% of voluntary carbon market transactions, with only 10-11 projects actively issuing credits globally^5^. This underrepresentation does not reflect insufficient demand but rather a verification bottleneck that constrains supply. At the core of this bottleneck lies the permanence problem: the inability to credibly demonstrate that sequestered carbon will remain stored over the crediting period, typically 100 years.

The permanence challenge is particularly acute for blue carbon because coastal wetlands exist in dynamic biogeochemical environments where carbon can be released through multiple pathways. While carbon dioxide emissions from decomposition are relatively well-characterized, methane emissions present a fundamentally different challenge. Methane's global warming potential of 28-30 times that of CO₂ over a 100-year horizon (and approximately 80 times over 20 years) means that even modest methane emissions can substantially offset carbon sequestration benefits^6^.

Recent research has quantified this offset at concerning magnitudes. Rosentreter et al. demonstrated that methane emissions offset approximately 20% of mangrove carbon burial from sediments alone^7^. Subsequent work has revealed that tree-stem methane emissions, previously unaccounted for in most carbon budgets, contribute additional offsets^8^. Under certain hydrological and biogeochemical conditions, methane can offset 94-98% of buried carbon, potentially converting blue carbon projects from net sinks to net sources of warming^9^.

### 1.2 Limitations of Current MRV Approaches

Current MRV methodologies for blue carbon projects, exemplified by Verra's VM0033 Methodology for Tidal Wetland and Seagrass Restoration, address methane risk primarily through exclusion criteria and conservative default assumptions^10^. VM0033 specifies salinity thresholds below which projects are either excluded or required to apply substantial deductions to account for potential methane emissions. While scientifically conservative, this approach has significant limitations.

First, the relationship between salinity and methane flux is mediated by complex microbial community dynamics that salinity alone cannot capture. Methane production is catalyzed by methanogenic archaea, whose activity depends not only on salinity-driven sulfate concentrations but also on organic matter availability, temperature, redox conditions, and the presence of competing microbial guilds^11,12^. A site with low salinity but robust methanotrophic communities may emit less methane than a higher-salinity site with impaired methane oxidation capacity. Current methodologies cannot distinguish these scenarios.

Second, flux tower and chamber-based methane measurements, the standard approach for quantifying emissions, are prohibitively expensive for most project developers. Installation and maintenance costs of $50,000-200,000 per site, combined with the need for continuous monitoring over multi-year periods, create barriers that effectively exclude smaller projects and developing-country sites where blue carbon potential is greatest^13^. Even where flux measurements are feasible, they provide retrospective data on past emissions rather than predictive indicators of future risk.

Third, the binary nature of current verification (creditable vs. non-creditable) fails to capture the continuous gradient of permanence risk that characterizes real ecosystems. A project site with 15% methane offset is treated identically to one with 85% offset under current frameworks, despite their vastly different climate value propositions. This coarse granularity impedes efficient capital allocation and prevents risk-based pricing that could unlock marginal projects with appropriate discounting.

### 1.3 Microbial Functional Genes as Leading Indicators

MethaNet exploits the observation that microbial functional gene abundances provide leading indicators of ecosystem carbon dynamics that can complement, and in some contexts substitute for, direct flux measurement. All biological methane production depends on the methyl-coenzyme M reductase (MCR) complex, encoded by the mcrABG operon and present in all known methanogens^14^. The abundance of mcrA genes in environmental samples quantitatively correlates with methanogenic capacity across diverse ecosystems^15,16^.

Conversely, aerobic methane oxidation is catalyzed by particulate methane monooxygenase (pMMO), encoded by the pmoA gene found in methanotrophic bacteria^17^. The balance between methanogenic and methanotrophic potential, measured as the mcrA/pmoA abundance ratio, thus provides a mechanistic indicator of net methane flux that integrates community-level metabolism in a single interpretable metric.

Prior work has demonstrated correlations between functional gene ratios and methane emissions. Lee et al. reported strong relationships between mcrA/pmoA ratios and net methane flux in wetland ecosystems^18^. He et al. showed that functional gene repertoires predict methane emission patterns more accurately than taxonomic composition alone^19^. Zhang et al. demonstrated that methanogen and methanotroph abundances respond predictably to environmental drivers including water content, carbon availability, and nitrogen status^20^.

These observations suggest that metagenomic profiling of functional genes could provide a scalable, cost-effective approach to methane risk assessment. However, translating this insight into operational verification tools faces a critical challenge: the scarcity of paired metagenomic-flux datasets in blue carbon ecosystems.

### 1.4 Transfer Learning Across Microbial Ecosystems

The data asymmetry problem in environmental genomics is well-documented. Agricultural microbiome research, driven by the economic importance of livestock methane emissions, has generated extensive datasets linking methanogen community composition to methane production rates. The rumen microbiome, in particular, has been characterized through large-scale initiatives including the Hungate1000 project^21^ and numerous national herd characterization studies^22^, yielding thousands of samples with paired genomic and phenotypic data.

Environmental ecosystems, including the blue carbon habitats of primary interest for carbon markets, remain severely data-sparse by comparison, with roughly an order of magnitude fewer characterized samples. Generating ground-truth flux measurements for each ecosystem type would require years of fieldwork and substantial investment, creating an apparent barrier to developing verification tools in the near term.

Transfer learning offers a potential solution. The core methanogenesis machinery is evolutionarily conserved across ecosystems. The three primary methanogenic pathways (hydrogenotrophic, methylotrophic, and acetoclastic) utilize homologous enzyme systems, and key archaeal families including Methanobacteriales, Methanomicrobiales, and Methanosarcinales are globally distributed despite divergent community compositions under different selective pressures^23,24^. This conservation of biological function creates an opportunity: molecular features that predict methane flux in data-rich rumen ecosystems may transfer to data-sparse environmental systems if the features encode functional properties rather than ecosystem-specific taxonomic identities.

Recent advances in foundation models for biological sequences enhance this transfer learning potential. Protein language models such as ESM-2^25^ learn general representations of protein function from hundreds of millions of sequences, capturing evolutionarily conserved properties that predict catalytic efficiency and substrate specificity regardless of taxonomic origin. Genomic foundation models like GenomeOcean^26^ extend this approach to entire genomes, encoding operon structure, gene neighborhood patterns, and regulatory architecture that influence metabolic phenotypes. By representing methanogen communities in these foundation model embedding spaces, we can identify functional similarities invisible to traditional sequence alignment approaches.

### 1.5 Study Objectives

This study addresses the following research objectives:

1. **Quantify functional gene conservation** across rumen and coastal wetland methanogen communities to assess transfer learning feasibility.

2. **Develop and validate an ensemble classification framework** that combines gradient boosting, neural networks, and foundation model-based similarity search to predict permanence risk from metagenomic data.

3. **Identify minimal transferable feature sets** that maximize cross-ecosystem predictive accuracy while maintaining interpretability.

4. **Establish alignment with carbon market verification requirements** by mapping classification outputs to ICVCM Assessment Framework criteria for permanence.

5. **Provide open resources** including validated feature sets, trained models, and reference datasets to accelerate adoption of functional ecology-based verification.

We test the central hypothesis that functional gene features trained on rumen methanogens transfer reliably to coastal ecosystems, enabling scalable permanence risk classification without requiring extensive ground-truth flux data collection for each target environment.

---

## 2. Methods

### 2.1 Dataset Assembly and Curation

#### 2.1.1 Rumen Microbiome Training Dataset

The primary training dataset leverages the Ruminant Gut Archaeome Catalogue^27^, the most comprehensive publicly available methanogen genome collection at time of analysis. This dataset provides sufficient sample size for robust model training while ensuring taxonomic and functional diversity across methanogenic lineages.

From the original catalogue, we selected 998 high-quality metagenome-assembled genomes (MAGs) meeting the following criteria: completeness >90% and contamination <5% as assessed by CheckM v1.0.2^28^; minimum N50 >10 kb; presence of at least one complete methanogenesis pathway module. Genome quality was independently verified using CheckM2 v1.0.1^29^ to confirm estimates from the original curation.

Functional annotation was performed using a hybrid pipeline combining traditional database searches with foundation model-based inference. KEGG ortholog assignments were obtained via KofamScan v1.3.0^30^ with default parameters. Pfam domain annotations used HMMER v3.4 against Pfam-A release 36.0^31^. For novel genes lacking database matches, we applied ESM-2-based functional inference as described below.

Paired methane flux measurements were available for 412 samples representing diverse host species, geographic origins, and dietary regimes. Flux values were normalized to dry matter intake where available to enable cross-study comparison.

#### 2.1.2 Coastal Wetland Validation Dataset

The validation dataset was assembled from publicly available metagenomes in the NCBI Sequence Read Archive, supplemented by collaborative data sharing from ongoing blue carbon monitoring programs. We prioritized samples meeting the following criteria: sequencing depth >10 million paired reads; host contamination <10% (assessed via Kraken2 against RefSeq); metadata completeness including geographic coordinates, salinity measurements, and ecosystem classification.

The final validation dataset comprises 127 coastal wetland samples distributed across four ecosystem types: mangrove sediments (n=58), seagrass beds (n=31), salt marshes (n=24), and peatlands (n=14). Geographic coverage spans Indo-Pacific (Indonesia, Philippines, Australia), Caribbean (Mexico, Belize, Colombia), West African (Senegal, Nigeria), and Brazilian sites, representing the major blue carbon regions globally.

Critically, 23 samples include paired in situ methane flux measurements obtained through static chamber or eddy covariance methods, providing the ground-truth validation subset. These samples span the full salinity gradient (0.5-38 ppt) and include both net source and net sink sites. We acknowledge that this validation sample size limits the strength of claims that can be made about cross-ecosystem generalization.

#### 2.1.3 Metagenomic Processing Pipeline

Raw sequencing reads were processed using a standardized pipeline to ensure consistency across datasets. Quality filtering used fastp v0.23.4^32^ with default parameters, removing adapters, low-quality bases (Q<20), and reads shorter than 50 bp. Host contamination removal (for rumen samples) used Bowtie2 v2.5.1^33^ against host reference genomes.

Metagenomic assembly employed MEGAHIT v1.2.9^34^ with the meta-large preset optimized for complex environmental samples. Assembly quality was assessed using metaQUAST v5.2.0^35^. Binning combined MetaBAT2 v2.15^36^ and MaxBin2 v2.2.7^37^ outputs refined through DAS Tool v1.1.6^38^ to maximize MAG recovery while controlling contamination.

Taxonomic assignment of recovered MAGs used GTDB-Tk v2.3.0^39^ against the Genome Taxonomy Database release 214. For phylogenetic analysis, marker gene sequences were aligned using MAFFT v7.520^40^ and trees constructed using IQ-TREE v2.2.5^41^ with automatic model selection.

### 2.2 Functional Gene Quantification

#### 2.2.1 Marker Gene Detection

We targeted five functional marker genes representing key biogeochemical processes relevant to carbon permanence:

**mcrA (methyl-coenzyme M reductase alpha subunit):** The terminal enzyme of methanogenesis, present in all known methanogens. Detection used custom HMM profiles trained on the FunGene mcrA reference alignment^42^ (threshold: E-value <1e-20, coverage >70%).

**pmoA (particulate methane monooxygenase alpha subunit):** Catalyzes aerobic methane oxidation in methanotrophic bacteria. Detection used HMM profiles for both Type I and Type II methanotrophs^43^ (threshold: E-value <1e-15, coverage >60%).

**dsrA (dissimilatory sulfite reductase alpha subunit):** Marker for sulfate-reducing bacteria that compete with methanogens for electron donors. Detection used profiles from established dsrAB reference databases^44^ (threshold: E-value <1e-20, coverage >70%).

**nifH (nitrogenase iron protein):** Marker for nitrogen fixation capacity, relevant to ecosystem productivity and carbon inputs. Detection used HMM profiles from the nifH reference database^45^ (threshold: E-value <1e-15, coverage >50%).

**cbbL (RuBisCO large subunit):** Marker for autotrophic carbon fixation, indicating primary production potential. Detection distinguished Form I and Form II variants using separate HMM profiles (threshold: E-value <1e-30, coverage >70%).

#### 2.2.2 Abundance Quantification

Gene abundances were quantified using a two-step approach. First, metagenomic reads were mapped to detected marker gene sequences using BWA-MEM2 v2.2.1^46^ with default parameters. Second, read counts were converted to transcripts per million (TPM) normalized values to account for differences in gene length and sequencing depth:

$$TPM_i = \frac{counts_i / length_i}{\sum_j (counts_j / length_j)} \times 10^6$$

For the key mcrA/pmoA ratio, we computed the log2-transformed ratio of TPM values, with a pseudocount of 1 added to avoid division by zero:

$$ratio = \log_2\left(\frac{TPM_{mcrA} + 1}{TPM_{pmoA} + 1}\right)$$

#### 2.2.3 Pathway Completeness Scoring

Beyond individual gene markers, we computed pathway-level completeness scores for seven methanogenesis modules defined in KEGG (M00567, M00357, M00356, M00563, M00358, M00359, M00360). Module completeness was calculated as the fraction of constituent genes detected with at least 50% of expected copy numbers. Pathway scores were aggregated into three summary features: hydrogenotrophic completeness, methylotrophic completeness, and acetoclastic completeness.

### 2.3 Foundation Model Embedding Generation

#### 2.3.1 Genomic-Level Embeddings

We employed GenomeOcean^26^, a 4-billion parameter genome foundation model trained on >600 Gbp of metagenomic assemblies, to generate genomic-level embeddings capturing operon structure and regulatory context. For each methanogen MAG, we extracted the complete methanogenesis operon region (mcrABG plus 5 kb flanking sequence) and obtained 1,024-dimensional embeddings through forward pass inference. These embeddings encode gene neighborhood patterns, intergenic distances, and sequence compositional features that influence metabolic regulation.

Additionally, we incorporated representations from MGM (Microbial General Model)^47^, a recently published foundation model specifically designed for microbiome analyses. MGM embeddings capture cross-sample compositional relationships that complement the within-genome focus of GenomeOcean.

#### 2.3.2 Protein-Level Embeddings

For protein-level functional representations, we used ESM-2 (650M parameter variant)^25^ to embed translated sequences of marker genes. ESM-2 embeddings capture evolutionarily conserved structural and functional properties that determine enzyme catalytic efficiency independent of taxonomic identity.

For each detected marker gene, we obtained the mean-pooled 1,280-dimensional embedding across all sequence positions. Multi-gene representations were constructed by concatenating embeddings from mcrA, pmoA, and hdrABC (where detected), yielding combined 3,840-dimensional vectors.

#### 2.3.3 Dual-Scale Integration

Genomic and protein embeddings capture complementary biological information: the former encodes regulatory and contextual features; the latter encodes catalytic functional properties. We evaluated three integration strategies:

1. **Early fusion:** Concatenation of genomic and protein embeddings prior to classification.
2. **Late fusion:** Separate classifiers on each embedding type with prediction averaging.
3. **Attention-weighted fusion:** Learned attention weights over embedding sources.

The attention-weighted fusion achieved optimal performance and was adopted for final models (see Results).

### 2.4 Transfer Learning Framework

#### 2.4.1 Domain Characterization

Before training transfer models, we quantified the distribution shift between source (rumen) and target (coastal) domains using Maximum Mean Discrepancy (MMD)^48^. MMD measures the distance between domain distributions in embedding space; lower values indicate greater transferability.

We computed MMD using radial basis function kernels with bandwidth selected via median heuristic. Domain overlap was further characterized by identifying "bridge taxa" (methanogen lineages present in both rumen and coastal samples) that serve as anchor points for transfer.

#### 2.4.2 Model Architecture

Our ensemble classification framework combines four complementary approaches:

**Gradient Boosted Trees (XGBoost):** An interpretable baseline using traditional functional gene features (abundances, ratios, pathway scores). XGBoost v2.0.3^49^ was trained with learning rate 0.1, maximum depth 6, and 500 estimators with early stopping on validation loss.

**Multi-Layer Perceptron:** A deep learning model operating on foundation model embeddings. Architecture: 3 hidden layers (1024, 512, 256 units) with ReLU activations, batch normalization, and dropout (0.3). Training used Adam optimizer with learning rate 1e-4 for 100 epochs.

**Random Forest:** An ensemble of decision trees providing complementary variance reduction. Scikit-learn implementation with 500 estimators, maximum depth 20, and minimum samples per leaf of 5.

**Similarity-Based Predictor:** A non-parametric approach using FAISS-indexed reference embeddings. For query samples, we retrieve the k=10 nearest neighbors in embedding space and compute weighted-average risk scores using cosine similarity weights.

#### 2.4.3 Ensemble Integration

Individual model predictions were integrated using learned weights optimized on the validation set:

$$\hat{y}_{ensemble} = \sum_{m \in models} w_m \cdot \hat{y}_m$$

Weights were constrained to sum to 1 and optimized to minimize log loss on held-out validation data. Final weights were: gradient boosting 0.35, neural network 0.30, random forest 0.20, similarity-based 0.15.

#### 2.4.4 Cross-Validation Strategy

Given the limited size of the validation dataset with paired flux measurements (n=23), we employed leave-one-out cross-validation (LOOCV) to maximize training data utilization while providing unbiased performance estimates. For the larger rumen dataset, we used stratified 5-fold cross-validation with ecosystem-aware splitting to prevent data leakage.

Transfer efficiency was quantified as:

$$TE = \frac{Performance_{target}}{Performance_{source}}$$

where performance is measured as AUC-ROC on held-out data.

### 2.5 Risk Classification Framework

#### 2.5.1 Risk Tier Definitions

Classification outputs are mapped to five risk tiers aligned with carbon market decision-making:

| Tier | Risk Score Range | Interpretation | Recommended Action |
|------|-----------------|----------------|-------------------|
| A | 0-10% | Very Low Risk | Standard monitoring |
| B | 10-25% | Low Risk | Enhanced baseline |
| C | 25-45% | Moderate Risk | Active mitigation planning |
| D | 45-65% | Elevated Risk | Intensive monitoring + mitigation |
| E | 65-100% | High Risk | Project restructuring or exclusion |

Tier thresholds were calibrated against the distribution of known methane offsets in the literature, with Tier C corresponding approximately to the 27% mean offset reported for typical mangrove sites.

#### 2.5.2 Uncertainty Quantification

We provide confidence intervals for all classifications using bootstrap resampling. For each prediction, we generate 1,000 bootstrap samples of the ensemble model predictions and compute the 2.5th and 97.5th percentiles to obtain 95% confidence intervals.

Additionally, we flag samples as "out-of-distribution" when their embedding-space distance to the training data exceeds the 95th percentile of within-training distances, signaling elevated prediction uncertainty.

#### 2.5.3 ICVCM Criteria Mapping

Classification outputs directly address ICVCM Assessment Framework criteria for permanence^50^:

- **Criterion 9.1 (Risk Identification):** Risk tier and score identify permanence threats.
- **Criterion 9.2 (Mitigation Measures):** Recommendations endpoint provides mitigation guidance based on functional profile analysis.
- **Criterion 9.3 (Monitoring Requirements):** Monitoring schedule endpoint specifies re-assessment intervals based on risk tier.

### 2.6 Statistical Analysis

All statistical analyses were performed in Python 3.11 using NumPy 1.26, SciPy 1.12, and scikit-learn 1.4. Correlation analyses used Pearson's r with two-tailed significance tests. Multiple testing correction applied Benjamini-Hochberg false discovery rate control at α=0.05. Effect sizes are reported as Cohen's d for continuous comparisons. Confidence intervals are 95% unless otherwise specified.

Feature importance was assessed using both permutation importance (100 permutations per feature) and SHAP (SHapley Additive exPlanations) values^51^ to ensure robustness to model-specific biases.

---

## 3. Results

### 3.1 Functional Gene Conservation Across Ecosystems

To assess the feasibility of cross-ecosystem transfer learning, we first characterized the phylogenetic and functional relationships between rumen and coastal wetland methanogen communities.

#### 3.1.1 Taxonomic Overlap

Of the 12 methanogen families detected across all samples, 6 were present in both rumen and coastal ecosystems (50% family-level overlap). Shared families included Methanobacteriaceae, Methanomicrobiaceae, Methanosarcinaceae, Methanocorpusculaceae, Methanomethylophilaceae, and Methanomassiliicoccaceae. These shared lineages represented 73% and 61% of methanogen abundance in rumen and coastal samples, respectively, indicating that the majority of methanogenic capacity in both environments derives from phylogenetically related organisms.

At the genus level, overlap was lower (38%) but still substantial, with key genera including *Methanobrevibacter*, *Methanosphaera*, and *Methanomethylophilus* detected in both environments. Methanogenic pathway distribution differed between ecosystems: rumen communities were predominantly hydrogenotrophic (74% of MAGs), while coastal communities showed depth-stratified distribution with methylotrophic methanogens more abundant in surface sediments (60%) and hydrogenotrophic methanogens dominating deeper layers (55%).

#### 3.1.2 Functional Gene Sequence Conservation

Phylogenetic analysis of mcrA sequences revealed strong conservation across ecosystems (Fig. 1A). Pairwise amino acid identity between rumen and coastal mcrA sequences averaged 78.4% (±8.2% SD), with the catalytic core region showing >90% identity across all samples. Key residues involved in methyl-coenzyme M binding and catalysis were invariant, consistent with strong purifying selection on methanogenesis function.

The pmoA gene showed similar conservation patterns among methanotrophic bacteria, with 72.1% (±10.3% SD) average identity and highly conserved active site residues. The mcrA/pmoA sequences from coastal samples clustered phylogenetically with their rumen counterparts in embedding space despite the environmental differences (Fig. 1B), supporting the hypothesis that functional representations would transfer across ecosystems.

#### 3.1.3 Foundation Model Embedding Analysis

UMAP visualization of GenomeOcean embeddings revealed distinct but overlapping clusters for rumen and coastal methanogens (Fig. 2A). The overlap region, comprising approximately 23% of samples, was enriched for hydrogenotrophic methanogens and samples from lower-salinity coastal sites. These "bridge samples" exhibited intermediate embedding characteristics and provided anchor points for domain adaptation.

ESM-2 protein embeddings of mcrA showed tighter clustering by metabolic pathway than by ecosystem origin (Fig. 2B). Hydrogenotrophic, methylotrophic, and acetoclastic mcrA variants formed distinct clusters regardless of whether they originated from rumen or coastal environments, indicating that the foundation model successfully captured functional distinctions relevant to methane production phenotypes.

Maximum Mean Discrepancy between domains was 0.31 (95% CI: 0.27-0.35) for raw sequence features but decreased to 0.14 (95% CI: 0.11-0.17) in foundation model embedding space, representing a 55% reduction in domain shift. This reduction in domain distance suggests that foundation model representations capture transferable functional properties that abstract away ecosystem-specific sequence variation.

### 3.2 Transfer Learning Performance

#### 3.2.1 Within-Domain Performance (Rumen)

On the rumen training data with paired flux measurements (n=412), our ensemble model achieved strong performance on the three-class classification task (Low/Moderate/High methane production):

- **Accuracy:** 84.7% (95% CI: 81.2-88.1%)
- **Macro F1-Score:** 0.83 (95% CI: 0.79-0.87)
- **AUC-ROC (one-vs-rest):** 0.91 (95% CI: 0.88-0.94)

Individual model performance varied, with the neural network on foundation embeddings performing best (AUC 0.89) followed by gradient boosting on traditional features (AUC 0.86), random forest (AUC 0.84), and similarity-based prediction (AUC 0.81). The ensemble exceeded all individual models, demonstrating the value of combining complementary approaches.

Feature importance analysis revealed that the mcrA/pmoA ratio was the most predictive single feature (SHAP value: 0.42), followed by hydrogenotrophic pathway completeness (0.31), hdrABC complex abundance (0.28), and ESM-2 embedding principal components 1-3 (collectively 0.35). Taxonomic features showed lower importance, confirming that functional representations outperform taxonomic markers for flux prediction.

#### 3.2.2 Cross-Domain Transfer (Rumen → Coastal)

The critical test of our approach is performance on the coastal validation samples with paired flux measurements (n=23). Under leave-one-out cross-validation:

- **Accuracy:** 73.9% (95% CI: 54.5-87.3%)
- **Macro F1-Score:** 0.71 (95% CI: 0.52-0.85)
- **AUC-ROC:** 0.79 (95% CI: 0.64-0.91)

Transfer efficiency (coastal AUC / rumen AUC) was 0.87, indicating that 87% of within-domain predictive power transferred across ecosystems. This transfer efficiency exceeds baseline models using taxonomic features alone (transfer efficiency: 0.52) or raw sequence k-mer features (transfer efficiency: 0.61).

Domain adaptation using MMD regularization during training improved transfer efficiency to 0.91 (AUC 0.83), demonstrating that explicit domain alignment enhances cross-ecosystem generalization.

We note that the wide confidence intervals on coastal performance reflect the small validation sample size (n=23), and these results require validation on larger independent datasets.

#### 3.2.3 Risk Tier Classification Performance

Mapping continuous risk scores to five-tier classifications, we observed:

| Tier | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| A (Very Low) | 0.86 | 0.75 | 0.80 | 4 |
| B (Low) | 0.71 | 0.83 | 0.77 | 6 |
| C (Moderate) | 0.67 | 0.67 | 0.67 | 6 |
| D (Elevated) | 0.80 | 0.80 | 0.80 | 5 |
| E (High) | 1.00 | 0.50 | 0.67 | 2 |

Adjacent-tier confusion (e.g., B classified as C) accounted for 82% of misclassifications, indicating that errors tend to be near-boundary rather than gross misclassifications. No Tier A sample was classified as Tier D or E, and no Tier E sample was classified as Tier A or B.

### 3.3 mcrA/pmoA Ratio as Net Flux Predictor

Given the importance of the mcrA/pmoA ratio in our feature analysis, we conducted focused validation of this metric against measured net methane flux.

Across the 23 coastal samples with paired measurements, log2(mcrA/pmoA) correlated with log10-transformed net methane flux (R² = 0.72, p < 0.001; Fig. 3A). The relationship was consistent across ecosystem types, with similar slopes observed for mangrove (R² = 0.68), salt marsh (R² = 0.74), and mixed wetland samples (R² = 0.71). Insufficient sample size in seagrass (n=3) and peatland (n=2) subsets precluded ecosystem-specific analysis for these types.

Threshold analysis identified mcrA/pmoA ratio breakpoints corresponding to risk tier boundaries:

| Ratio Threshold | Approximate Risk Tier | Sensitivity | Specificity |
|-----------------|----------------------|-------------|-------------|
| log2 < 0.5 | A/B boundary | 0.82 | 0.89 |
| log2 < 1.5 | B/C boundary | 0.78 | 0.84 |
| log2 < 2.5 | C/D boundary | 0.85 | 0.79 |
| log2 < 3.5 | D/E boundary | 0.91 | 0.76 |

These thresholds provide interpretable decision boundaries for practitioners without access to full ensemble model predictions.

### 3.4 Salinity Gradient Analysis

A potential application of MethaNet is enabling risk stratification for sites currently excluded from carbon crediting due to low salinity. We analyzed model performance across salinity bins spanning the VM0033 applicability threshold (Fig. 4).

For samples with salinity <18 ppt (below typical VM0033 threshold), model accuracy was 71.4% (n=7), compared to 75.0% for samples ≥18 ppt (n=16). The difference was not statistically significant (Fisher's exact p = 0.83), indicating that MethaNet maintains predictive power across the full salinity gradient.

Among the limited low-salinity samples (n=7), three were classified as Tier A or B (low risk) based on their functional gene profiles. These sites exhibited elevated pmoA abundances consistent with robust methanotrophic activity, providing a potential mechanistic explanation for their low methane emissions despite favorable salinity conditions for methanogenesis. However, given the small sample size, these observations should be considered preliminary and require validation with larger datasets.

### 3.5 Uncertainty Quantification and Calibration

Bootstrap confidence intervals for risk scores averaged ±8.3 percentage points (range: 4.2-15.7). Wider intervals were associated with samples in the embedding space periphery (correlation r = 0.61 between embedding-space distance and CI width), confirming that uncertainty estimates appropriately reflect model confidence.

Probability calibration, assessed via reliability diagrams (Fig. 5), showed that predicted tier probabilities were well-calibrated after isotonic regression (expected calibration error: 0.062). This calibration enables stakeholders to interpret classification probabilities as genuine risk likelihoods rather than arbitrary scores.

Three samples (13%) were flagged as out-of-distribution based on embedding distance exceeding the 95th percentile threshold. Manual inspection revealed these samples had unusual taxonomic compositions (dominated by uncultured archaeal lineages) or atypical environmental conditions (hypersaline lagoon, anoxic hot spring). For such samples, we recommend supplementary validation before relying on model predictions.

### 3.6 Feature Transferability Ranking

Recursive feature elimination with cross-validation identified a minimal feature set of 12 features that preserved 95% of full-model performance (Fig. 6). These transferable features include:

1. log2(mcrA/pmoA) ratio
2. Hydrogenotrophic pathway completeness
3. mcrA ESM-2 embedding PC1
4. mcrA ESM-2 embedding PC2
5. hdrABC complex abundance
6. GenomeOcean embedding PC1
7. Acetoclastic pathway completeness
8. pmoA abundance (TPM)
9. mcrA ESM-2 embedding PC3
10. dsrA abundance (TPM)
11. GenomeOcean embedding PC2
12. Methylotrophic pathway completeness

This minimal set enables efficient operational deployment while maintaining robust cross-ecosystem transfer. A practitioner with only the mcrA/pmoA ratio (Feature 1) can achieve 78% of full-model accuracy, demonstrating the utility of this simple metric for first-pass screening.

---

## 4. Discussion

### 4.1 Functional Conservation Enables Cross-Ecosystem Transfer

Our results suggest that transfer learning from data-rich agricultural microbiomes to data-sparse environmental ecosystems is feasible when models are trained on evolutionarily conserved functional representations. The key insight is that methanogenesis machinery is under strong purifying selection across all environments, creating molecular features that predict phenotype regardless of community taxonomic composition.

The 87% transfer efficiency we observe, with 91% achievable through domain adaptation, exceeds what would be predicted by taxonomy-based models. Traditional approaches that rely on presence/absence of specific methanogen genera achieve transfer efficiencies of only 40-60% in our experiments, reflecting the poor generalization of taxonomic features across environments with different community compositions. Foundation model embeddings capture the functional properties that matter for methane production while abstracting away ecosystem-specific sequence variation.

The "bridge taxa" we identify (methanogen lineages present in both rumen and coastal environments) may serve as calibration points for future transfer learning efforts. These organisms span the domain gap in embedding space and could be targeted for laboratory characterization to generate high-quality ground-truth data bridging agricultural and environmental contexts.

### 4.2 Foundation Models as Feature Extractors

The success of GenomeOcean and ESM-2 embeddings in our framework points to broader applications of foundation models in environmental metagenomics. These models, trained on massive sequence databases, learn generalizable representations that transfer to downstream tasks with limited training data, precisely the scenario facing most environmental prediction challenges.

Our dual-scale embedding approach, combining genomic context (GenomeOcean) with protein function (ESM-2), outperforms either modality alone. This suggests that methane phenotypes depend on both the regulatory architecture controlling gene expression and the catalytic properties of expressed enzymes. Future work could extend this approach to include additional modalities such as 3D protein structure predictions or metatranscriptomic expression profiles.

The computational efficiency of foundation model inference enables operational deployment at scale. Embedding generation for a complete metagenomic sample requires approximately 30 seconds on consumer GPU hardware (NVIDIA RTX 3080), making real-time risk assessment feasible for project monitoring applications.

### 4.3 Implications for Blue Carbon Verification

#### 4.3.1 Addressing the VM0033 Methane Gap

Current Verra methodology applies conservative default deductions for methane emissions in lower-salinity sites, effectively excluding many potential project areas. MethaNet may enable risk stratification within these excluded zones, identifying sites where functional gene profiles indicate low methane risk despite salinity below threshold.

Our preliminary observation that three of seven low-salinity validation samples exhibit Tier A/B risk profiles suggests potential unrealized blue carbon capacity in sites currently assumed to be methane-impaired. While these results require validation with substantially larger sample sizes, they indicate that functional ecology assessment could provide additional information for project evaluation.

We propose that VM0033 and similar methodologies could consider MethaNet classifications as one input for evaluating lower-salinity sites. Sites achieving Tier A or B classification through molecular assessment might qualify for crediting with monitoring requirements specified by tier assignment, pending methodological review and approval.

#### 4.3.2 Leading Indicators vs. Lagging Measurements

An advantage of functional gene assessment over flux measurement is its predictive rather than retrospective nature. Flux towers measure emissions that have already occurred; functional gene abundances indicate the metabolic capacity for future emissions. This distinction has potential implications for permanence risk management.

Changes in mcrA/pmoA ratios could provide early warning of shifting methane risk before measurable flux changes manifest. Restoration projects could use MethaNet to track trajectory toward low-methane equilibrium states. Verification bodies could use functional profiles to identify sites requiring enhanced monitoring before permanence events occur.

#### 4.3.3 Cost-Benefit Analysis

A single metagenomic sequencing run costs approximately $500-1,000 per sample including DNA extraction, library preparation, and sequencing. Bioinformatic analysis via cloud computing adds approximately $50-100 per sample. Total per-sample cost of $550-1,100 compares favorably to flux tower installation ($50,000-200,000) or continuous chamber monitoring ($10,000-50,000 per site per year).

More importantly, metagenomic sampling scales efficiently across project portfolios. A single field campaign can collect samples from dozens of sites, and sequencing throughput continues to improve following sequencing-cost reduction trends. For project developers managing multiple sites, MethaNet could enable portfolio-wide risk assessment at a fraction of traditional monitoring costs.

### 4.4 Alignment with ICVCM Core Carbon Principles

The ICVCM Assessment Framework establishes criteria that high-quality carbon credits must satisfy^50^. MethaNet outputs directly address permanence-related criteria:

**Criterion 9.1 (Risk Identification):** Our tiered classification explicitly identifies and quantifies permanence risk. Risk scores with confidence intervals enable transparent communication of uncertainty to credit buyers and registries.

**Criterion 9.2 (Mitigation Measures):** Recommendations generated from functional profile analysis provide specific guidance for methane risk mitigation, including water table management strategies, methanotrophy enhancement approaches, and site selection considerations.

**Criterion 9.3 (Monitoring Requirements):** Monitoring schedules derived from risk tier assignments specify re-assessment intervals proportionate to identified risk, enabling risk-based monitoring intensity allocation.

This alignment positions MethaNet as a verification tool that could be considered for integration into registry workflows pending methodological approval. We are engaged in discussions with Verra and Gold Standard regarding potential incorporation into future methodology revisions.

### 4.5 Limitations and Future Directions

#### 4.5.1 Sample Size and Geographic Coverage

The primary limitation of this study is the small sample size for coastal validation (n=23 with paired flux measurements). While leave-one-out cross-validation maximizes data utilization, larger validation datasets are needed to establish performance across the full range of blue carbon conditions.

We are collaborating with the Australian Institute of Marine Science, Smithsonian Tropical Research Institute, and KAUST to expand the validation dataset to >100 samples with paired flux measurements over the next 18 months. These partnerships will also extend geographic coverage to currently underrepresented regions including the Red Sea, Pacific Islands, and South Asia.

#### 4.5.2 Temporal Dynamics

Our current analysis treats samples as static snapshots, but methanogen community composition and activity vary seasonally and in response to environmental disturbance. Time-series sampling at reference sites is needed to characterize temporal dynamics and inform monitoring schedule recommendations.

Preliminary data from a mangrove site in Baja California sampled monthly over 12 months suggests that mcrA/pmoA ratios are relatively stable within sites (coefficient of variation: 0.18) compared to between-site variation (CV: 0.67), supporting the use of point-in-time sampling for site classification. However, sites experiencing disturbance (e.g., storm damage, salinity intrusion) showed rapid ratio shifts detectable within 1-2 months, indicating that event-driven re-assessment may be warranted.

#### 4.5.3 Capacity vs. Rate

Functional gene abundances measure metabolic capacity (the potential for methane production/consumption) rather than instantaneous rates. The relationship between capacity and realized flux depends on environmental conditions including temperature, substrate availability, and redox status. While capacity and rate are correlated in our validation data (R² = 0.72), the residual variance likely reflects environmental modulation of gene expression.

Integration of metatranscriptomic data capturing actual gene expression, rather than gene abundance alone, could improve rate prediction. We are developing protocols for combined DNA/RNA extraction from sediment samples that would enable this extension.

#### 4.5.4 Model Updating

As additional validation data become available, model performance will improve through retraining on expanded datasets. We have implemented a continuous integration pipeline that automates model retraining and validation when new ground-truth samples are added, ensuring that MethaNet predictions reflect the latest available evidence.

Additionally, we are developing monitoring for model drift (systematic changes in prediction accuracy as new ecosystems or conditions are encountered). Early detection of drift will trigger targeted data collection to maintain prediction quality.

---

## 5. Conclusions

### 5.1 Summary of Contributions

This study makes the following contributions to the intersection of microbial ecology and carbon market verification:

1. **Early demonstration of cross-ecosystem transfer learning for methane prediction:** We show that functional gene features trained on rumen methanogens transfer to coastal wetlands with 87-91% efficiency, reducing data requirements for environmental model development.

2. **Validation of foundation models for environmental metagenomics:** GenomeOcean and ESM-2 embeddings capture transferable functional properties that outperform traditional sequence features for cross-domain prediction.

3. **Identification of mcrA/pmoA ratio as a risk indicator:** This simple, interpretable metric explains 72% of variance in net methane flux across our coastal validation samples and provides practitioner-friendly risk assessment.

4. **ICVCM-aligned classification framework:** MethaNet outputs directly address Assessment Framework criteria for permanence, facilitating potential integration with existing verification infrastructure.

5. **Open resources for community adoption:** We release validated feature sets, trained models, and reference datasets to accelerate development of functional ecology-based verification approaches.

### 5.2 Operational Implications

MethaNet provides a framework for permanence risk assessment for blue carbon projects. Key operational advantages include:

- **Reduced data collection burden:** Transfer learning reduces the need for extensive ground-truth flux data collection in each target ecosystem.
- **Lower monitoring costs:** Metagenomic sampling costs 1-2% of flux tower installation.
- **Predictive rather than retrospective:** Functional gene profiles indicate future emission potential, enabling proactive risk management.
- **Tiered classification:** Risk tiers map directly to verification requirements and pricing decisions.

We envision MethaNet deployment through an API service enabling project developers and verification bodies to submit metagenomic data and receive classification reports within hours. Beta API access is available to qualified partners during the pre-publication period.

### 5.3 Broader Impact

Beyond blue carbon, the transfer learning framework we develop has potential applications to other environmental prediction challenges. Denitrification, sulfur cycling, and organic matter decomposition all depend on conserved microbial pathways that may exhibit similar cross-ecosystem transferability. The paradigm of training on data-rich model systems and transferring to data-sparse target environments could accelerate development of molecular verification tools across the nature-based solutions landscape.

More broadly, MethaNet demonstrates that rigorous molecular science and carbon market infrastructure can be productively integrated. As nature-based solutions scale to meet climate commitments, verification integrity becomes paramount. Functional ecology-based assessment offers a path toward verification systems grounded in biological mechanism rather than historical assumption, a direction that merits continued development and validation.

---

## 6. Data Availability

Raw metagenomic sequences are deposited in NCBI Sequence Read Archive under BioProject PRJNA[XXXXXX]. Processed MAG catalogs and functional gene annotations are available through Zenodo (DOI: 10.5281/zenodo.[XXXXXX]). The validation dataset including paired flux measurements will be released upon publication acceptance.

All data analysis code is available at https://github.com/emergentbiome/methanet under MIT license. Interactive Jupyter notebooks reproducing main figures are included in the repository.

Foundation model embeddings for all samples are provided in HDF5 format for researchers wishing to develop alternative classification approaches.

---

## 7. Code Availability

The complete MethaNet software suite is available at https://github.com/emergentbiome/methanet. This includes:

- Metagenomic processing pipeline (Snakemake workflow)
- Functional gene detection and quantification scripts
- Foundation model embedding generation code
- Ensemble classification training and inference
- API service implementation
- Interactive visualization notebooks

Documentation is available at https://methanet.emergentbiome.earth/docs. A containerized version (Docker) enables turnkey deployment without local dependency management.

---

## 8. References

1. Ecosystem Marketplace. State of the Voluntary Carbon Markets 2024. Forest Trends, Washington, DC (2024).

2. Friess, D. A., Rogers, K., Lovelock, C. E., et al. The state of the world's mangrove forests: past, present, and future. Annu. Rev. Environ. Resour. 44, 89-115 (2019).

3. McLeod, E., Chmura, G. L., Bouillon, S., et al. A blueprint for blue carbon: toward an improved understanding of the role of vegetated coastal habitats in sequestering CO2. Front. Ecol. Environ. 9, 552-560 (2011).

4. Pendleton, L., Donato, D. C., Murray, B. C., et al. Estimating global "blue carbon" emissions from conversion and degradation of vegetated coastal ecosystems. PLoS ONE 7, e43542 (2012).

5. Verra. VCS Project Database: Blue Carbon Projects. https://registry.verra.org (accessed January 2026).

6. IPCC. Climate Change 2023: Synthesis Report. Contribution of Working Groups I, II and III to the Sixth Assessment Report. IPCC, Geneva (2023).

7. Rosentreter, J. A., Maher, D. T., Erler, D. V., Murray, R. H. & Eyre, B. D. Methane emissions partially offset "blue carbon" burial in mangroves. Sci. Adv. 4, eaao4985 (2018).

8. Qin, Z., et al. Mangrove sediment carbon burial offset by methane emissions from mangrove tree stems. Nat. Geosci. (2025).

9. Al-Haj, A. N. & Fulweiler, R. W. A synthesis of methane emissions from shallow vegetated coastal ecosystems. Glob. Change Biol. 26, 2988-3005 (2020).

10. Verra. VM0033 Methodology for Tidal Wetland and Seagrass Restoration, v2.1. Verra, Washington, DC (2023).

11. Conrad, R. The global methane cycle: recent advances in understanding the microbial processes involved. Environ. Microbiol. Rep. 1, 285-292 (2009).

12. Bridgham, S. D., Cadillo-Quiroz, H., Keller, J. K. & Zhuang, Q. Methane emissions from wetlands: biogeochemical, microbial, and modeling perspectives from local to global scales. Glob. Change Biol. 19, 1325-1346 (2013).

13. Baldocchi, D. Measuring fluxes of trace gases and energy between ecosystems and the atmosphere—the state and future of the eddy covariance method. Glob. Change Biol. 20, 3600-3609 (2014).

14. Thauer, R. K. Biochemistry of methanogenesis: a tribute to Marjory Stephenson. Microbiology 144, 2377-2406 (1998).

15. Steinberg, L. M. & Regan, J. M. Phylogenetic comparison of the methanogenic communities from an acidic, oligotrophic fen and an anaerobic digester treating municipal wastewater sludge. Appl. Environ. Microbiol. 74, 6663-6671 (2008).

16. Luton, P. E., Wayne, J. M., Sharp, R. J. & Riley, P. W. The mcrA gene as an alternative to 16S rRNA in the phylogenetic analysis of methanogen populations in landfill. Microbiology 148, 3521-3530 (2002).

17. Knief, C. Diversity and habitat preferences of cultivated and uncultivated aerobic methanotrophic bacteria evaluated based on pmoA as molecular marker. Front. Microbiol. 6, 1346 (2015).

18. Lee, H. J., Kim, S. Y., Kim, P. J., Madsen, E. L. & Jeon, C. O. Methane emission and dynamics of methanotrophic and methanogenic communities in a flooded rice field ecosystem. FEMS Microbiol. Ecol. 88, 195-212 (2014).

19. He, S., Malfatti, S. A., McFarland, J. W., Anderson, F. E. & Pati, A. Patterns in wetland microbial community composition and functional gene repertoire associated with methane emissions. mBio 6, e00066-15 (2015).

20. Zhang, W., Kang, X., Kang, E., et al. Soil water content, carbon, and nitrogen determine the abundances of methanogens, methanotrophs, and methane emission in the Zoige alpine wetland. J. Soils Sediments 22, 470-481 (2022).

21. Seshadri, R., Leahy, S. C., Attwood, G. T., et al. Cultivation and sequencing of rumen microbiome members from the Hungate1000 Collection. Nat. Biotechnol. 36, 359-367 (2018).

22. Henderson, G., Cox, F., Ganesh, S., et al. Rumen microbial community composition varies with diet and host, but a core microbiome is found across a wide geographical range. Sci. Rep. 5, 14567 (2015).

23. Lyu, Z., Shao, N., Akinyemi, T. & Whitman, W. B. Methanogenesis. Curr. Biol. 28, R727-R732 (2018).

24. Kirschke, S., Bousquet, P., Ciais, P., et al. Three decades of global methane sources and sinks. Nat. Geosci. 6, 813-823 (2013).

25. Lin, Z., Akin, H., Rao, R., et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 379, 1123-1130 (2023).

26. Zhou, Z., Riley, R., Kautsar, S., et al. GenomeOcean: An Efficient Genome Foundation Model Trained on Large-Scale Metagenomic Assemblies. bioRxiv doi:10.1101/2025.01.30.635558 (2025).

27. Mi, J., Jing, X., Ma, C., et al. A metagenomic catalogue of the ruminant gut archaeome. Nat. Commun. 15, 9609 (2024).

28. Parks, D. H., Imelfort, M., Skennerton, C. T., Hugenholtz, P. & Tyson, G. W. CheckM: assessing the quality of microbial genomes recovered from isolates, single cells, and metagenomes. Genome Res. 25, 1043-1055 (2015).

29. Chklovski, A., Parks, D. H., Woodcroft, B. J. & Tyson, G. W. CheckM2: a rapid, scalable and accurate tool for assessing microbial genome quality using machine learning. Nat. Methods 20, 1203-1212 (2023).

30. Aramaki, T., Blanc-Mathieu, R., Endo, H., et al. KofamKOALA: KEGG ortholog assignment based on profile HMM and adaptive score threshold. Bioinformatics 36, 2251-2252 (2020).

31. Mistry, J., Chuguransky, S., Williams, L., et al. Pfam: The protein families database in 2021. Nucleic Acids Res. 49, D412-D419 (2021).

32. Chen, S., Zhou, Y., Chen, Y. & Gu, J. fastp: an ultra-fast all-in-one FASTQ preprocessor. Bioinformatics 34, i884-i890 (2018).

33. Langmead, B. & Salzberg, S. L. Fast gapped-read alignment with Bowtie 2. Nat. Methods 9, 357-359 (2012).

34. Li, D., Liu, C. M., Luo, R., Sadakane, K. & Lam, T. W. MEGAHIT: an ultra-fast single-node solution for large and complex metagenomics assembly via succinct de Bruijn graph. Bioinformatics 31, 1674-1676 (2015).

35. Mikheenko, A., Saveliev, V. & Gurevich, A. MetaQUAST: evaluation of metagenome assemblies. Bioinformatics 32, 1088-1090 (2016).

36. Kang, D. D., Li, F., Kirton, E., et al. MetaBAT 2: an adaptive binning algorithm for robust and efficient genome reconstruction from metagenome assemblies. PeerJ 7, e7359 (2019).

37. Wu, Y. W., Simmons, B. A. & Singer, S. W. MaxBin 2.0: an automated binning algorithm to recover genomes from multiple metagenomic datasets. Bioinformatics 32, 605-607 (2016).

38. Sieber, C. M., Probst, A. J., Sharrar, A., et al. Recovery of genomes from metagenomes via a dereplication, aggregation and scoring strategy. Nat. Microbiol. 3, 836-843 (2018).

39. Chaumeil, P. A., Mussig, A. J., Hugenholtz, P. & Parks, D. H. GTDB-Tk v2: memory friendly classification with the Genome Taxonomy Database. Bioinformatics 38, 5315-5316 (2022).

40. Katoh, K. & Standley, D. M. MAFFT multiple sequence alignment software version 7: improvements in performance and usability. Mol. Biol. Evol. 30, 772-780 (2013).

41. Minh, B. Q., Schmidt, H. A., Chernomor, O., et al. IQ-TREE 2: New models and efficient methods for phylogenetic inference in the genomic era. Mol. Biol. Evol. 37, 1530-1534 (2020).

42. Fish, J. A., Chai, B., Wang, Q., et al. FunGene: the functional gene pipeline and repository. Front. Microbiol. 4, 291 (2013).

43. Dumont, M. G. Primers: Functional Marker Genes for Methylotrophs and Methanotrophs. In Hydrocarbon and Lipid Microbiology Protocols (eds. McGenity, T. J., et al.) Springer, Berlin (2014).

44. Müller, A. L., Kjeldsen, K. U., Rattei, T., Pester, M. & Loy, A. Phylogenetic and environmental diversity of DsrAB-type dissimilatory (bi)sulfite reductases. ISME J. 9, 1152-1165 (2015).

45. Frank, I. E., Turk-Kubo, K. A. & Zehr, J. P. Rapid annotation of nifH gene sequences using classification and regression trees facilitates environmental functional gene analysis. Environ. Microbiol. Rep. 8, 905-916 (2016).

46. Vasimuddin, M., Misra, S., Li, H. & Aluru, S. Efficient architecture-aware acceleration of BWA-MEM for multicore systems. IEEE Int. Parallel Distrib. Process. Symp. 314-324 (2019).

47. Chen, Y., Zheng, L., Yang, K., et al. MGM: A large-scale pretrained foundation model for microbiome analyses in diverse contexts. bioRxiv doi:10.1101/2024.12.30.630825 (2024).

48. Gretton, A., Borgwardt, K. M., Rasch, M. J., et al. A kernel two-sample test. J. Mach. Learn. Res. 13, 723-773 (2012).

49. Chen, T. & Guestrin, C. XGBoost: A scalable tree boosting system. Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. 785-794 (2016).

50. ICVCM. Core Carbon Principles Assessment Framework. Integrity Council for the Voluntary Carbon Market (2023).

51. Lundberg, S. M. & Lee, S. I. A unified approach to interpreting model predictions. Adv. Neural Inf. Process. Syst. 30, 4765-4774 (2017).

52. Lovelock, C. E., Fourqurean, J. W. & Morris, J. T. Modeled CO2 emissions from coastal wetland transitions to other land uses: tidal marshes, mangrove forests, and seagrass beds. Front. Mar. Sci. 4, 143 (2017).

53. Rosentreter, J. A., Laruelle, G. G., Bange, H. W., et al. Coastal vegetation and estuaries collectively are a greenhouse gas sink. Nat. Clim. Change 13, 579-587 (2023).

54. West, T. A. P., Börner, J., Sills, E. O. & Kontoleon, A. Action needed to make carbon offsets from tropical forest conservation work for climate change mitigation. Science 381, 873-877 (2023).

55. Hoarfrost, A., Aptekmann, A., Farfañuk, G. & Bromberg, Y. Deep learning of a bacterial and archaeal universal language of life enables transfer learning and illuminates microbial dark matter. Nat. Commun. 13, 2606 (2022).

56. Hug, L. A., Baker, B. J., Anantharaman, K., et al. A new view of the tree of life. Nat. Microbiol. 1, 16048 (2016).

57. Parks, D. H., Rinke, C., Chuvochina, M., et al. Recovery of nearly 8,000 metagenome-assembled genomes substantially expands the tree of life. Nat. Microbiol. 2, 1533-1542 (2017).

58. Wallace, R. J., Rooke, J. A., McKain, N., et al. A heritable subset of the core rumen microbiome dictates dairy cow productivity and emissions. Sci. Adv. 5, eaav8391 (2019).

---

## Acknowledgements

We thank the bio.xyz community for valuable feedback on early versions of this work. We acknowledge the Australian Institute of Marine Science, Smithsonian Tropical Research Institute, and King Abdullah University of Science and Technology for discussions regarding validation data access. Computational resources were provided by [Cloud Provider]. This work was supported by the MethaNet IP-Token through bio.xyz decentralized science funding infrastructure.

---

## Author Contributions

A.P. and J.G. conceived the study. A.P. led metagenomic data curation and bioinformatic analysis. J.G. developed the transfer learning framework and classification models. Both authors contributed to writing and approved the final manuscript.

---

## Competing Interests

A.P. and J.G. are co-founders of EmergentBiome, which is developing commercial applications of the MethaNet framework. The authors have filed provisional patent applications related to the mcrA/pmoA ratio methodology for carbon verification.

---

## Supplementary Information

### Extended Data Figure 1: Geographic Distribution of Samples
[Map showing rumen (998) and coastal (127) sample locations with ecosystem type color coding]

### Extended Data Figure 2: Quality Control Metrics
[Panels showing: A) Completeness/contamination distributions; B) Genome size by ecosystem; C) Sequencing depth distributions]

### Extended Data Figure 3: Full Methanogenesis Pathway Analysis
[Complete KEGG pathway coverage heatmap for all 7 methanogenesis modules across samples]

### Extended Data Table 1: Complete Sample Metadata
[Detailed metadata for all 1,125 samples including coordinates, salinity, flux measurements where available]

### Extended Data Table 2: Model Hyperparameters
[Full specification of XGBoost, neural network, random forest, and similarity-based model configurations]

### Extended Data Table 3: Feature Importance Rankings
[Complete SHAP values for all 47 features across all model types]

### Supplementary Note 1: Computational Resource Requirements
[Detailed compute specifications: GPU hours for embedding generation, CPU hours for assembly, storage requirements]

### Supplementary Note 2: Sensitivity Analysis
[Robustness checks across: hyperparameter ranges, training/validation splits, embedding model versions, classification thresholds]

### Supplementary Note 3: Alternative Model Architectures
[Results from alternative approaches tested: convolutional networks, attention mechanisms, graph neural networks on gene networks]

---

## Figure Legends

**Figure 1. Functional gene conservation across rumen and coastal methanogen communities.**
(A) Maximum-likelihood phylogenetic tree of mcrA amino acid sequences colored by ecosystem origin (blue: rumen; green: coastal). Scale bar indicates 0.1 substitutions per site. (B) Sequence identity distribution between rumen and coastal mcrA sequences, showing 78.4% mean identity with highly conserved catalytic core.

**Figure 2. Foundation model embedding space analysis.**
(A) UMAP projection of GenomeOcean genomic embeddings with samples colored by ecosystem (blue: rumen; green: coastal) and sized by mcrA abundance. Overlap region indicated by dashed ellipse. (B) ESM-2 protein embeddings of mcrA sequences colored by methanogenic pathway type (red: hydrogenotrophic; yellow: methylotrophic; purple: acetoclastic), demonstrating that functional clustering supersedes ecosystem clustering.

**Figure 3. mcrA/pmoA ratio validation against net methane flux.**
(A) Scatter plot of log2(mcrA/pmoA) vs. log10(net CH₄ flux) for 23 coastal samples with paired measurements. Line shows linear regression (R² = 0.72, p < 0.001). Points colored by ecosystem type. (B) Receiver operating characteristic curves for binary classification of high vs. low methane risk at various ratio thresholds.

**Figure 4. Classification performance across salinity gradient.**
Classification accuracy (bars) and sample distribution (line) across salinity bins. Vertical dashed line indicates VM0033 threshold (18 ppt). Performance is maintained below threshold, with three of seven low-salinity samples classified as low risk.

**Figure 5. Model calibration assessment.**
(A) Reliability diagram showing predicted vs. observed tier probabilities before (gray) and after (blue) isotonic calibration. Diagonal indicates perfect calibration. (B) Distribution of confidence interval widths colored by out-of-distribution flag status.

**Figure 6. Minimal transferable feature set identification.**
(A) Performance (AUC-ROC) vs. number of features retained during recursive feature elimination. Horizontal dashed line indicates 95% of full-model performance. (B) Top 12 transferable features ranked by importance, with bars colored by feature category (blue: gene ratios; green: pathway scores; orange: embeddings).

---

*Manuscript word count: ~9,200 (main text)*
*References: 58*
*Figures: 6*
*Extended Data Figures: 3*
*Extended Data Tables: 3*
*Supplementary Notes: 3*
