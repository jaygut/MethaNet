# MethaNet: Molecular Attestation For Blue-Carbon Methane Diligence

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

**A molecular evidence graph for candidate review, monitoring design, and
validation-ready methane-risk intelligence**

[Landing page](https://emergentbiome.earth/) ·
[Interactive MBAG report](https://emergentbiome.earth/report/) ·
[Positioning and claim contract](docs/methanet_positioning_and_claims.md) ·
[Fresh-clone repository guide](docs/repository_guide.md)

---

## Overview

MethaNet is a molecular-attestation system for blue-carbon methane diligence.
It links protein-language geometry, genomic context, functional machinery,
expression evidence, genome quality, taxonomy, provenance, and validation
readiness through the MethaNet Bridge Attestation Graph, or MBAG. The current
system helps reviewers inspect molecular evidence, prioritize candidates, and
decide which measurement will most improve a monitoring or validation plan.

Methane can materially reduce the climate value of coastal carbon storage.
Chambers, flux towers, and process assays provide essential field evidence,
although their cost and spatial coverage limit portfolio-scale use. MethaNet
adds a traceable molecular layer that can guide diligence and measurement
design today. Calibrated sample and project risk follows after molecular
evidence is paired with abundance, environmental context, uncertainty, and
field or process validation.

### Core Hypothesis

MAG/proteome-level molecular fingerprints can help identify biologically
plausible methane pathways and prioritize field validation. These fingerprints
include methanogenesis and methane-oxidation markers, sulfur competition,
substrate-processing capacity, genomic context, QC, taxonomy, and source
provenance. Sample methane-risk estimation requires exact sample linkage,
abundance or read coverage, environmental covariates, uncertainty propagation,
and measured flux or process validation.

## What Works Today

The August 10, 2026 end-to-end controlled-diligence release is a governed molecular
warehouse and evidence graph:

| Evidence layer | Current release | Decision use |
| --- | ---: | --- |
| Registered MAG/proteome units | 7,965 | Auditable warehouse denominator |
| ESM-2-bearing units | 7,710 | Proteome-neighborhood navigation |
| gLM2 payloads | 7,717 | Protocol-aware genomic context |
| Data-complete tri-views | 7,710 | Three-view evidence availability |
| Schema-normalized tri-views | 7,710 | Common long-form table and event semantics |
| Pipeline-normalized tri-views | 5,209 | POC, MSM, and Futian; comparability audit pending |
| Mechanism-comparable tri-views | 0 | No cross-lane mechanism comparison is authorized yet |
| MUCC v1 source-scaffold tri-views | 2,501 | Wetland reference and expression-detection review |

A data-complete tri-view contains ESM-2, gLM2, and a functional payload. Its
evidence state records whether those payloads share a common quantitative
contract. POC, MSM, and Futian now share accepted/best/present event semantics,
but code, configuration, and database fingerprints plus source-aware statistical
gates remain pending. MUCC v1 stays useful under its distinct source-scaffold
contract.

Current outputs include candidate evidence cards, molecular diligence,
monitoring prioritization, validation-gap routing, and study design. Final
sample/project methane-risk scores, calibrated A to E tiers, measured methane
flux, source-independent transfer conclusions, and carbon-credit decisions
remain validation outcomes.

---

## Research Objectives

1. **Build reviewable molecular attestation.** Link each MAG or proteome to
   ESM-2, gLM2, functional evidence, QC, taxonomy, provenance, and claim
   eligibility.

2. **Characterize domain shift.** Quantify source, ecosystem, taxonomy, and
   protocol effects before interpreting molecular neighborhoods as transferable
   biology.

3. **Prioritize mechanism hypotheses.** Combine methane, sulfur, substrate, and
   genomic-context evidence into candidate cards with explicit comparability and
   missingness states.

4. **Earn calibrated methane-risk outputs.** Link molecular evidence to physical
   samples, abundance, environmental conditions, uncertainty, and paired field
   measurements under site and season holdouts.

---

## Evidence Sources

### Rumen Reference Resources

Rumen genomes provide a data-rich reference for methane-system molecular
biology. Current MethaNet transfer claims remain source-aware because ecosystem
and source are confounded in the original POC.

| Resource | Description | Size | Source |
|----------|-------------|------|--------|
| Ruminant Gut Archaeome | Curated archaeal genomes from ruminant gut systems | 998 genomes | [Mi et al., 2024](https://doi.org/10.1038/s41467-024-54025-3) |
| RUG2 Catalog | Metagenome-assembled genomes from rumen | 4,941 MAGs | [Stewart et al., 2019](https://doi.org/10.1038/s41587-019-0202-3) |
| Hungate1000 | Cultivated rumen microbiome isolates | 410 genomes | [Seshadri et al., 2018](https://doi.org/10.1038/nbt.4110) |

### Blue-Carbon And Wetland Evidence Lanes

| Lane | Registered or source denominator | Current role |
| --- | ---: | --- |
| Wetland/MUCC POC | 107 MAG/proteome units | Mechanism-comparable target-domain POC |
| MSM China mangrove | 1,428 local candidates | Annotation-complete target expansion with one release exclusion |
| Futian mangrove | 3,404 rMAGs, including 3,156 ready payload rows | Time, depth, and habitat expansion |
| MUCC v1 Old Woman Creek | 2,508 archive MAGs | Wetland source-scaffold, expression detection, and field-validation lane |

Each source retains its own denominator, metadata resolution, functional
contract, and provenance. Sample-level ecological interpretation begins after
MAG-to-sample, abundance, environmental, and outcome joins are authoritative.

### Blue Catalyst POC (Completed on Apolo-3)

We completed a cross-ecosystem proteome embedding POC between MUCC wetland
genomes and rumen genomes (PRJEB31266) using ESM2-650M protein language model
embeddings. The POC was developed for the
[Hatch Blue Blue Catalyst](https://www.hatch.blue/programs/blue-catalyst)
accelerator program in Singapore in May 2026.

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

**Key scientific findings**

- Rumen Archaea dominate the historical bridge-candidate ranking. All top 11
  alpha-transfer scores are Archaea, which is consistent with conserved
  methanogenesis machinery and remains a hypothesis for functional review.
- Classifier performance of AUC 1.0 documents strong source and ecosystem label
  separability within this POC. Methane transfer and activity remain separate
  validation questions.
- **P0 caveat.** Source and ecosystem are perfectly confounded. All rumen
  genomes come from PRJEB31266, and all wetland genomes come from MUCC.
  Deconfounding with additional sources is the highest-priority next step.

**40-genome baseline (initial validation, `apolo_20260226_194505`)**
- 40 samples (20 MUCC + 20 rumen), PERMANOVA R²=0.517, silhouette=0.433, trajectory t=13.97 (p=1.5e-16)
- bin.23 (Archaea) embeds 100% inside wetland cluster (mixing_coeff=1.0)
- The R² decrease from 0.517 to 0.202 at scale is consistent with greater
  within-rumen diversity after adding 515 rumen genomes.

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

### MBAG Molecular Attestation Warehouse And MRV Roadmap

The original 662-genome ESM-2 POC has developed into a source-audited
multi-lane molecular atlas and MBAG evidence graph. MAG-level evidence supports
candidate review and monitoring follow-up. Sample and project MRV requires
sample mapping, abundance or read coverage, environmental covariates,
uncertainty propagation, and flux or process validation.

Current implemented artifact arc:

| Layer | Current local artifact | Status |
|-------|------------------------|--------|
| POC ESM2/crosswalk backbone | `ai_docs/functional_metagenomics_expansion/proteome_crosswalk/embedded_662_proteome_id_crosswalk.tsv` | 662 proteome IDs: 555 rumen + 107 wetland/MUCC |
| POC unit scope | `results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.with_unit_scope.tsv` | 625 MAG/bin-comparable units + 37 assembly-context units |
| POC functional atlas warehouse | `results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse_semantic_rebuild_20260810/` | 625 selected MAG/bin runs under accepted KOfam and best-ranked MCycDB/SCycDB event semantics; 712 validation gates passing |
| POC gLM2 context | `results/contextual_genomics/glm2_integration_20260616_poc_catchup_20260616_073441/` | 625/625 MAG-bin units complete after rumen catch-up |
| Mangrove/MSM ESM2 expansion | `results/blue_catalyst_poc/runs/msm_china_2025_esm2_20260616_082112/artifacts/` | 1,428/1,428 local mangrove/MSM proteomes embedded |
| Mangrove/MSM gLM2 expansion | `results/contextual_genomics/glm2_msm_magbin_full_20260615_092737/` | 1,428/1,428 contextual genome units complete |
| Mangrove/MSM functional expansion | `results/functional_metagenomics/msm_china_2025_20260615/` | 1,428/1,428 functional MAGs complete in the rebuilt Parquet/DuckDB warehouse; failed, partial, and superseded attempts remain explicit |
| Mangrove/Futian ESM2 + gLM2 expansion | `results/blue_catalyst_poc/runs/futian_mangrove_2026_esm2_phase1_shard*_20260621/` and `results/contextual_genomics/glm2_futian_phase1_shard*_20260621/` | 3,156/3,156 ready Futian MAG/proteome units complete in both ESM2 and gLM2 |
| Mangrove/Futian functional expansion | `results/functional_metagenomics/futian_mangrove_2026_phase1_archaea/` and `results/functional_metagenomics/futian_mangrove_2026_phase1_bacteria_00*/` | 3,156/3,156 ready-payload rows carry validated functional output in the rebuilt Parquet/DuckDB warehouse; 248 source gaps remain explicit |
| MUCC v1 Old Woman Creek wetland lane | `results/functional_metagenomics/mucc_v1_owc_wetland_20260626/` | 2,508 registered wetland MAGs; 2,501 ESM-2-bearing and data-complete source-scaffold tri-views; processed expression detection and staged field evidence retain explicit linkage gaps |
| Metadata provenance | `results/functional_metagenomics/environmental_metadata_recovery_20260612/`, `data/external/msm_china_2025/metadata/`, and `data/external/futian_mangrove_2026_qi/metadata/` | source/environmental metadata with resolution tiers across rumen, wetland/MUCC, MSM, and Futian lanes |
| Molecular attestation graph | `results/attestation/mmag_mvp_20260617/` plus the release-level MBAG projection | POC graph MVP plus current warehouse-wide evidence-contract, candidate-card, and validation-readiness views |
| Current controlled-diligence atlas | `results/reports/mbag_nextgen_molecular_niche_atlas_20260810_end_to_end/report.html` | 7,965 registered units, 7,710 ESM-2 embeddings, 7,717 gLM2 payloads, and 7,710 data-complete tri-views; public deployment remains gated |

These artifacts support MAG/proteome-level molecular attestation,
bridge-candidate prioritization, evidence-card review, and monitoring-readiness
design. The [positioning and claim contract](docs/methanet_positioning_and_claims.md)
records the shared language used by the repository, landing page, and report.

The current system should be read as five evidence lanes:

- rumen POC: source reference lane for methane-system molecular neighborhoods;
- wetland/MUCC POC: target-domain wetland lane with complete MAG-bin molecular evidence;
- mangrove/MSM expansion: broader blue-carbon target lane with ESM2, gLM2, and functional payloads complete at 1,428/1,428, and source metadata that links local MAG candidates to grouped sediment-sample metadata rather than final abundance-weighted sample scores.
- mangrove/Futian expansion: larger time, depth, and habitat target lane with
  ESM-2, gLM2, and functional payloads complete for all 3,156 ready rows;
- MUCC v1 Old Woman Creek: wetland reference lane with 2,501 data-complete
  source-scaffold tri-views, processed expression detection, and staged
  field-validation evidence whose exact ecological joins remain unresolved.

For the freshest dated payload inventory, see `docs/current_artifact_inventory.md`.
For live multi-lane payload state, regenerate the registry summary with
`scripts/reports/refresh_atlas_lane_registry_status.sh`; for report freezes,
use `scripts/reports/build_methanet_3view_payload_freeze.py` and preserve
blocked/pending rows rather than silently dropping them.

Key roadmap and contract documents:

- `ai_docs/functional_metagenomics_expansion/README.md` - functional-metagenomics expansion index.
- `ai_docs/functional_metagenomics_expansion/final_mrv_risk_scoring_roadmap.md` - maturity ladder from MBAG molecular screening to final MRV risk scoring.
- `ai_docs/functional_metagenomics_expansion/embedding_functional_transfer_framework/methanet_embedding_functional_transfer_framework.md` - MBAG framework and next-generation MethaNet Intelligence Report blueprint.
- `docs/current_artifact_inventory.md` - current datasets, databases, warehouses, metadata outputs, and attestation artifacts created and used locally.

---

## Evidence Engineering

The current system separates evidence availability from quantitative
comparability:

| Feature Type | Description | Tools |
|--------------|-------------|-------|
| Functional mechanism | Accepted/present methane, sulfur, substrate, and pathway evidence | KOfam, MCycDB, SCycDB, METABOLIC, dbCAN |
| Proteome representation | Genome-level aggregation of protein embeddings | [ESM-2](https://github.com/facebookresearch/esm) |
| Genomic context | Native and shuffled context under protocol-stratified comparison | gLM2 |
| Reliability | QC, taxonomy, provenance, annotation coverage, and missingness | CheckM2, GUNC, GTDB-Tk, source manifests |
| Environmental readiness | Sample identity, abundance, covariates, uncertainty, and field validation | Current integration roadmap |

---

## Methodology

### Evidence-To-Validation Approach
```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  REFERENCE      │     │  MBAG EVIDENCE   │     │  TARGET DOMAIN  │
│                 │     │                  │     │                 │
│  Rumen + POC    │ ──▶ │ Tri-view graph + │ ──▶ │ Coastal wetland │
│  molecular      │     │ QC + provenance  │     │ monitoring and  │
│  context        │     │ + claim gates    │     │ validation      │
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

The marker panel provides mechanism-relevant molecular potential. Marker ratios
and pathway balances become sample-level predictors only after abundance,
environmental context, uncertainty, and field validation are joined.

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
├── docs/                    # Positioning, methods, inventory, and runbooks
├── ai_docs/functional_metagenomics_expansion/
│                             # Scientific contracts and MRV roadmap
├── configs/                 # Pipeline and atlas-lane configuration
├── scripts/                 # Curation, warehouse, attestation, and report builders
├── workflow/                # Snakemake template and execution rules
├── src/methanet/            # Core package
├── notebooks/               # Analysis notebooks
├── tests/                   # Unit and integration tests
├── web/emergentbiome-methanet/
│                             # Landing page and public report publisher
└── data/                    # Local data directory, excluded from git
```

For a fresh-clone walkthrough, verification commands, generated-output
boundaries, and the scientific claim guardrails, see
[`docs/repository_guide.md`](docs/repository_guide.md).

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

## Product Maturity

| Level | Current state | Decision unlocked |
| --- | --- | --- |
| 0. Molecular attestation | Available now | Candidate review, evidence cards, and monitoring prioritization |
| 1. Sample identity and metadata | In progress | Physical sample and site rollups |
| 2. Abundance and community capacity | Planned | Abundance-weighted molecular capacity |
| 3. Environmental permissiveness | Planned | Context-aware methane-pathway interpretation |
| 4. Flux and process validation | Planned | Target-domain outcome calibration |
| 5. Probabilistic methane risk | Target | Calibrated sample and site risk distributions |
| 6. MRV and audit integration | Horizon | Registry-aligned evidence packages after independent review |

---

## Citation

If you use MethaNet in your research, please cite:
```bibtex
@software{methanet2026,
  author       = {Philosof, Alon and Gutierrez, Jay},
  title        = {{MethaNet: Molecular Attestation for Blue-Carbon Methane Diligence}},
  year         = {2026},
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
