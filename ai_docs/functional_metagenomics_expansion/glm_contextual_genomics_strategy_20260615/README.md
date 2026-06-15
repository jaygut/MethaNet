# MethaNet gLM Contextual Genomics Strategy

Date: 2026-06-15

Scope: strategic deployment plan for using gLM/gLM2 as a contextual genomic feature layer in MethaNet's multi-view methane-risk atlas. This document is a planning/runbook artifact. It does not submit jobs, regenerate outputs, or change production state.

## Executive Verdict

Yes, gLM-style contextual genomics is low-hanging fruit for MethaNet, but the right framing is precise:

> gLM should complement, not replace, the current ESM2 proteome embedding and functional annotation layers.

The immediate product value is that gLM adds gene-neighborhood and co-regulation context around methane, sulfur, substrate, and unknown genes. This directly addresses one weakness of current proteome-level ESM2 mean pooling: ESM2 tells us that a protein or proteome is functionally similar in sequence/structure space, but it does not know whether that protein lives inside a methanogenesis operon, a methane-oxidation context, a sulfur-competition context, a mobile element, or a substrate-utilization island.

The strongest near-term MethaNet claim enabled by gLM is:

> MethaNet can identify context-supported methane-risk architectures at MAG/proteome level by combining protein mechanism, gene neighborhood, operon/co-regulation signal, functional annotation, QC, and source-aware controls.

This is still MAG/proteome-level functional potential. It is not final sample methane flux, final A-E methane-risk tiering, or carbon-credit approval.

## Why The gLM Paper Matters For MethaNet

The Nature Communications gLM paper is unusually aligned with MethaNet because it was designed for microbial/metagenomic scaffolds and explicitly tests methane biology.

Important findings from the paper:

- gLM was trained on millions of metagenomic contig fragments containing 15 to 30 genes from MGnify. Each gene is represented by a 1280-dimensional ESM2 protein embedding plus strand/orientation context, and the transformer learns masked gene prediction over genomic neighborhoods.
- The model generates contextualized protein embeddings that encode both protein sequence and genomic neighborhood.
- The paper's McrA example is directly relevant to MethaNet. Context-free ESM2 McrA embeddings separate methane-production and methane-oxidation directionality poorly, while gLM contextualized embeddings separate ANME-1, ANME-2, and methanogen-associated McrA contexts much better.
- gLM attention patterns capture operon-like co-regulation signal, validated against E. coli K-12 operon data.
- gLM improves enzyme-function prediction over context-free ESM2 for many EC classes, demonstrating that context adds information orthogonal to protein sequence.
- gLM contextual variance highlights mobile genes such as phage genes and transposases, making it potentially useful for horizontal transfer and mobile-context flags.
- gLM helps position unknown proteins closer to known functional neighborhoods, which is valuable for metagenomic dark matter around methane/sulfur/substrate modules.

MethaNet implication:

> This is exactly the missing layer between "the protein looks methanogenesis-like" and "the genome architecture around this protein supports a methane-risk mechanism."

Sources:

- gLM Nature Communications article: https://www.nature.com/articles/s41467-024-46947-9
- original gLM repository: https://github.com/y-hwang/gLM
- gLM2 repository: https://github.com/TattaBio/gLM2
- gLM2 model card: https://huggingface.co/tattabio/gLM2_650M
- OMG/gLM2 ICLR 2025 paper page: https://openreview.net/forum?id=jlzNb1iWs3

## Model Choice

### Original gLM

Original gLM is the paper-anchored, conceptually lowest-risk pilot because it matches the Nature Communications methods and directly outputs contextualized protein embeddings from ESM2 protein vectors plus contig gene order.

Strengths:

- Closest to the published McrA/methane evidence.
- Works with protein FASTA plus contig-to-protein orientation mapping.
- Reuses ESM2-style embeddings, so it fits the current MethaNet ESM2 lineage.
- Produces gene-level contextual embeddings and optional attention matrices.
- Maximum sequence length is 30 genes, which matches marker-neighborhood use cases.

Constraints:

- The repository license is academic/non-commercial. This is fine for internal research and partner-scientific evaluation, but not automatically safe for commercial product embedding without legal review or author permission.
- The published environment pins old PyTorch/CUDA-era packages. The README suggests `torch==1.12.1+cu116`, which is not an ideal match for Apolo's H100 GPUs.
- The repo's ESM2 embedding helper uses FairScale/FSDP and old `fair-esm`; MethaNet should not rely on that exact helper on H100 unless smoke-tested.
- The provided batching script looks up each protein ID by list search, which is acceptable for a smoke test but should be replaced with an indexed map for thousands to millions of proteins.

Best use:

- First scientific pilot for marker-neighborhood contextual features.
- Use original gLM to validate whether contextual embeddings improve McrA/mcr/hdr/mtr/dsr/apr/sat/methylotrophy candidate interpretation over ESM2 alone.

### gLM2

gLM2 is the more production-attractive follow-on because it is Apache-2.0 licensed, Hugging Face-native, H100-friendly with `bfloat16`, and includes both coding sequence and intergenic DNA context.

Strengths:

- Apache-2.0 license in the repo/model card.
- Available as `tattabio/gLM2_150M` and `tattabio/gLM2_650M`.
- Encodes genomic scaffolds as mixed-modality sequences: amino-acid tokens for CDS and nucleotide tokens for intergenic sequence.
- 4096-token context length in the updated model card.
- Trained on OMG, an open metagenomic corpus combining MGnify and IMG at very large scale.
- Better aligned with future MethaNet features involving regulatory/intergenic context, promoter-like sequence, operon boundaries, and CDS-to-intergenic interactions.

Constraints:

- The public repo is much lighter than original gLM. It provides model code and examples but not a full MAG-scale batching pipeline.
- Inputs must be carefully constructed from FNA/GFF/FAA so that amino acids are uppercase, intergenic DNA is lowercase, and strand tokens are inserted correctly.
- Output is token-level; MethaNet needs a pooling layer to convert token embeddings into CDS, intergenic, window, contig, and MAG features.
- It has less direct published methane-specific validation than the original gLM paper's McrA analysis, though it is the newer generation from the same model family.

Best use:

- Product-oriented second pilot and likely production candidate after original gLM validates the feature value.
- Use gLM2 for high-quality MAG subsets where FNA/GFF/FAA are aligned and intergenic context can be reconstructed accurately.

### Recommendation

Use a two-track pilot:

1. **Track A, immediate:** original gLM on protein/CDS neighborhoods for the 625 MAG-bin POC plus a small MSM mangrove high-quality subset. This gives the fastest scientific answer.
2. **Track B, product-facing:** gLM2 on FNA/GFF/FAA-derived mixed-modality contig windows. This gives the cleaner license path and richer genomic context.

If we must choose one for long-term MethaNet product integration, choose gLM2. If we must choose one for fastest proof that genomic context helps MethaNet methane biology, choose original gLM.

## What Proteome Embeddings Provide That gLM Does Not Replace

ESM2 proteome embeddings remain strategically important:

- They encode protein sequence/structure/function similarity directly.
- They are robust to fragmented assemblies because proteins can be embedded independently.
- They support broad proteome-level geometry and bridge candidate discovery.
- They abstract away synonymous nucleotide variation and local assembly-order issues.
- They are easier to compute, cache, compare, and explain as a stable baseline.

gLM adds:

- gene neighborhood;
- orientation;
- co-regulation/operon signal;
- contextual meaning of the same marker in different genomic architectures;
- mobile/HGT context;
- unknown-gene context transfer;
- optional intergenic/regulatory context in gLM2.

The multi-view architecture should therefore be:

```text
ESM2 proteome view
  + functional annotation view
  + MAG QC/taxonomy view
  + gLM/gLM2 genomic-context view
  + later abundance/environment/flux view
```

## Best MethaNet Data Payloads To Use

### Priority 0: External Reference Anchors

Purpose: calibrate and sanity-check the gLM feature layer before touching MethaNet claims.

Recommended inputs:

- published McrA/ANME/methanogen reference sequences and contexts from the gLM paper/source-data trail where available;
- curated methanogenesis and anaerobic methane oxidation marker neighborhoods;
- marker-negative but taxonomically adjacent archaeal controls;
- shuffled-gene-order controls and random-window controls.

Expected outputs:

- `mcrA_context_directionality_probe`;
- `operon_attention_sanity_probe`;
- ESM2-only versus gLM contextual benchmark.

### Priority 1: 625 MAG-Bin POC Backbone

Source:

```text
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.mag_bin_only.tsv
```

Why this is the best first MethaNet payload:

- It is already tied to the 662-proteome ESM2 POC, but quarantines the 37 assembly-context rumen records.
- It has 625 MAG-bin-comparable units: 518 rumen and 107 wetland/MUCC.
- It preserves `proteome_id` and source/ecosystem labels needed for bridge-candidate validation.
- It lets us ask the exact scientific question: do gLM context features explain, support, or downgrade ESM2 bridge candidates?

Known quality facts from the manifest:

- 625 MAG-bin units.
- `n_proteins_used` median 1,941; p95 about 2,859; max 4,761.
- Completed MUCC-side functional predicted proteins, where available, are in a plausible MAG range: median 2,094, max 3,413.
- Current source/ecosystem confounding still applies: rumen and wetland are source-confounded.

Use cases:

- bridge candidate reranking;
- McrA/methanogenesis marker context;
- sulfur-competition neighborhood features;
- source leakage tests;
- ESM2-only versus ESM2+gLM ablation.

Do not use:

- the 37 no-bin rumen assembly-context records as MAG-level evidence.

### Priority 2: MSM China 2025 Mangrove Sediment Payload

Source:

```text
results/functional_metagenomics/msm_china_2025_20260615/manifests/msm_china_2025_functional_mag_manifest.tsv
```

Why this matters:

- It is the first large mangrove sediment target-domain expansion.
- It contains 1,428 MAG/proteome units: 1,348 Bacteria and 80 Archaea.
- All rows currently map to NCBI BioSample in the manifest.
- It is the best immediate dataset for expanding MethaNet beyond the original MUCC wetland target and into a true mangrove methane-relevant environment.

Quality caveats:

- The archive denominator is 1,428, while the paper-reported medium/high-quality MAG denominator is 966; reconciliation is still pending.
- Assembly fragmentation is substantial: median N50 about 5.3 kb, p95 about 28.6 kb.
- Median contig count is about 491; p95 about 1,276.
- Protein counts are plausible for most rows, but 134 rows have fewer than 1,000 proteins and 53 have fewer than 500 proteins.
- Only 42 functional runs were complete in the latest local status snapshot read for this document; 0 failed, 4 partial, 1,382 not started.

Best use:

- do not start with whole-MAG global gLM2 summarization;
- start with marker-centered windows and high-quality, higher-N50 subsets;
- prioritize the 80 archaeal MAGs and any marker-bearing bacterial/sulfur/substrate MAGs;
- include all 1,428 later as a QC-weighted target-domain discovery layer.

Recommended MSM subset order:

1. Completed functional runs with CheckM2/GUNC observed and local quality-gate pass.
2. All archaeal MAGs with plausible protein count and strongest assembly stats.
3. MAGs with methane/sulfur marker hits once MCycDB/SCycDB/KOfam/METABOLIC evidence is available.
4. High-N50, protein-plausible bacterial MAGs for sulfur/substrate and syntrophy context.
5. Full 1,428 MAG payload as exploratory, QC-weighted context discovery.

### Priority 3: Assembly-Context Rumen Records

Source:

```text
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.assembly_context.tsv
```

Use as:

- rumen assembly/metagenome reservoir context;
- non-MAG evidence lane;
- stress test for long contig/window generation;
- not as direct MAG-level bridge evidence.

## Feature Design

### New Feature Tables

Add gLM as a separate derived evidence namespace. Do not overwrite ESM2 or functional annotation tables.

Recommended tables:

```text
dim_glm_window
fact_glm_gene_embedding
fact_glm_attention_edge
feature_glm_marker_neighborhood
feature_glm_contig_context
feature_glm_mag_level
feature_glm_unknown_gene_context
feature_glm_bridge_context_support
```

Required identity columns:

```text
cohort_run_id
glm_run_id
model_family              # gLM or gLM2
model_name
model_version_or_checkpoint
model_license
proteome_id
mag_id
contig_id
gene_id
window_id
source_tool
```

Recommended `dim_glm_window` columns:

```text
cohort_run_id
glm_run_id
proteome_id
mag_id
contig_id
window_id
window_type               # nonoverlap_30, sliding_30, marker_centered, gLM2_token_chunk
window_gene_count
window_token_count
start_gene_ordinal
end_gene_ordinal
center_gene_id
center_marker_family
strand_pattern
context_completeness_tier
source_fna
source_faa
source_gff
```

Recommended `feature_glm_marker_neighborhood` columns:

```text
proteome_id
mag_id
contig_id
gene_id
marker_family             # mcrA, mcrB, mcrG, hdrA, hdrB, hdrC, mtr*, dsrA, dsrB, aprA, sat, sox, pmoA, mmoX, etc.
marker_source             # MCycDB, KOfam, METABOLIC, Bakta, manual panel
window_id
glm_embedding_mean
glm_embedding_delta_from_esm2
neighbor_function_entropy
operon_attention_score
same_strand_neighbor_fraction
methane_context_score
sulfur_competition_context_score
substrate_context_score
hgt_mobile_context_flag
context_qc_tier
```

Recommended `feature_glm_bridge_context_support` columns:

```text
proteome_id
mag_id
bridge_rank
alpha_transfer_score
esm2_bridge_priority
glm_context_support_score
glm_context_contradiction_score
mcr_context_directionality
operon_context_confidence
sulfur_context_modifier
unknown_neighbor_support
source_leakage_flag
qc_penalty
final_contextual_bridge_label
allowed_claim_wording
blocking_gaps
```

### High-Value Hidden Features

These are the features most likely to strengthen MethaNet's value proposition:

1. **Mcr context directionality:** distinguish methanogenesis-like versus anaerobic methane-oxidation-like McrA neighborhoods.
2. **Methanogenesis operon integrity:** whether mcr/hdr/mtr/eha/ehb and electron-transfer genes sit in coherent gene neighborhoods.
3. **Sulfur competition context:** whether methanogenesis markers co-occur with sulfate/sulfide/sulfur-energy genes that may suppress or reshape methane risk.
4. **Syntrophy/substrate island context:** whether acetate, methylamine, methanol, methyl-sulfur, hydrogen/formate, CAZy, and transporter neighborhoods point to substrate availability.
5. **Unknown-gene context transfer:** identify hypothetical proteins repeatedly embedded inside methane/sulfur neighborhoods as new candidate markers.
6. **Mobile methane module flags:** use high contextual variance, transposase/phage proximity, and attention/mobile-neighborhood signals to flag horizontally mobile pathway modules.
7. **Context-supported bridge reranking:** upgrade ESM2 bridge candidates only when their methane/sulfur/substrate context agrees with the latent bridge signal.
8. **Context contradiction flags:** downgrade candidates where ESM2 similarity is high but gene-neighborhood evidence points to a different mechanism or poor context support.

## Installation And Deployment Path On Apolo

### Confirmed Apolo GPU State

Observed on 2026-06-15:

```text
partition: accel
node: a3-accel-0
cpus: 64
memory: about 251 GiB
gres: gpu:2
gpu model: NVIDIA H100 NVL
gpu memory: 95,830 MiB each
driver: 575.57.08
compute capability: 9.0
max partition time: 3 days
```

CPU partitions:

```text
longjobs: 2 nodes, 64 CPUs each, about 377 GiB/node, max 6 days
bigmem: 2 nodes, 64 CPUs each, about 503 GiB/node, max 4 days
```

Implication:

- CPU preprocessing should run on `longjobs` or `bigmem`.
- GPU inference should run on `accel`.
- H100 bf16 is ideal for gLM2.
- Original gLM's old CUDA 11.6 README environment should not be assumed valid on H100; smoke-test with a modern CUDA/PyTorch environment.

### Track A Environment: Original gLM

Recommended approach:

```bash
module load miniconda3/25.5.1
conda create -n methanet-glm python=3.10 -y
conda activate methanet-glm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.22.2 fair-esm fairscale numpy scipy scikit-learn tqdm h5py pyyaml
git clone https://github.com/y-hwang/gLM.git "$HOME/scratch/tools/gLM"
mkdir -p "$HOME/scratch/methanet_models/gLM"
wget -O "$HOME/scratch/methanet_models/gLM/glm.bin" https://zenodo.org/record/7855545/files/glm.bin
```

Important adjustment:

- Use MethaNet's existing ESM2 embedding code or a modern ESM2 embedding helper instead of relying blindly on the original repo's `data/plm_embed.py`.
- Patch or wrap `batch_data.py` to use a dictionary from protein ID to embedding index instead of `all_prot_ids.index(pid)` for scale.
- Run a 10-MAG smoke test before full POC inference.

Original gLM job shape:

```bash
sbatch \
  --partition=accel \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=64G \
  --time=08:00:00 \
  scripts/slurm/run_glm_context_inference_apolo3.sh
```

Recommended batching:

- one GPU per job;
- one shard per source slice or 25 to 50 MAGs;
- reduce `glm_embed.py -b` if GPU memory errors occur;
- output both `*.glm.embs.pkl` and, for marker windows only, attention matrices.

### Track B Environment: gLM2

Recommended approach:

```bash
module load miniconda3/25.5.1
conda create -n methanet-glm2 python=3.11 -y
conda activate methanet-glm2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate safetensors biopython numpy scipy scikit-learn pyarrow tqdm
export HF_HOME="$HOME/scratch/hf"
export TRANSFORMERS_CACHE="$HF_HOME"
```

Smoke-test model load:

```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    "tattabio/gLM2_650M",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained("tattabio/gLM2_650M", trust_remote_code=True)
```

gLM2 job shape:

```bash
sbatch \
  --partition=accel \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --mem=80G \
  --time=08:00:00 \
  scripts/slurm/run_glm2_context_inference_apolo3.sh
```

Use `gpu:2` only for a later throughput run after single-GPU memory and I/O are stable.

## Data Preparation

### Original gLM Input Contract

Original gLM needs:

1. protein FASTA with amino-acid sequences;
2. `contig_to_prots.tsv`, where each row maps a contig/window to ordered protein IDs with `+` or `-` strand markers;
3. ESM2 protein embeddings in the repo's pickle format;
4. max 30 proteins per sequence/window.

MethaNet should derive this from:

- FNA;
- FAA;
- GFF or Bakta/Prodigal feature table;
- canonical `proteome_id`;
- canonical `mag_id`;
- stable gene IDs.

Window policy:

- sort CDS by contig coordinate;
- create non-overlapping 30-gene windows for broad context;
- create marker-centered windows for methane/sulfur/substrate genes;
- label windows with fewer than 15 genes as `low_context_length`;
- keep contig fragments separate rather than pretending fragmented MAGs are continuous genomes;
- never join separate contigs into a fake gLM sequence.

### gLM2 Input Contract

gLM2 needs a mixed sequence:

```text
<+>AMINOACIDSEQUENCE<+>intergenicdna<->AMINOACIDSEQUENCE
```

Rules:

- CDS amino acids uppercase.
- intergenic DNA lowercase.
- each genomic element prepended by `<+>` or `<->`.
- preserve contig order and strand.
- split long contigs/windows at safe token limits below 4096.
- map output token spans back to CDS/intergenic/gene/window records.

MethaNet-specific gLM2 preprocessing should build:

```text
glm2_sequence_id
proteome_id
mag_id
contig_id
start_bp
end_bp
token_count
cds_count
intergenic_count
sequence_string
span_map_json_or_parquet
```

## Integration With MethaNet Workflow

Recommended output layout:

```text
results/contextual_genomics/
  glm_original_YYYYMMDD/
    manifests/
    prepared_inputs/
    embeddings/
    features/
    validation/
    logs/
  glm2_YYYYMMDD/
    manifests/
    prepared_inputs/
    embeddings/
    features/
    validation/
    logs/
```

Recommended reviewed docs:

```text
ai_docs/functional_metagenomics_expansion/glm_contextual_genomics_strategy_20260615/
  README.md
  future: feature_contract.md
  future: apolo_runbook.md
  future: validation_report_YYYYMMDD.md
```

Integration sequence:

1. Build `dim_glm_window` from selected MAG manifests.
2. Generate ESM2 gene embeddings or reuse compatible existing protein embeddings.
3. Run original gLM on marker-centered and non-overlapping windows.
4. Aggregate gene embeddings into marker/window/MAG features.
5. Join gLM features onto the MRV feature layer by `proteome_id`, `mag_id`, and gene IDs.
6. Run ESM2-only versus ESM2+functional versus ESM2+functional+gLM ablations.
7. Promote only validated features into MBAG/candidate-card outputs.
8. Run gLM2 pilot on the same windows where FNA/GFF/FAA reconstruction is reliable.

## Validation Plan

### Required Scientific Controls

Run these before partner-facing claims:

1. **ESM2-only baseline:** confirms what proteome embeddings already explain.
2. **gLM context gain:** tests whether contextual features improve methane/sulfur/substrate mechanism prediction.
3. **Shuffled gene order control:** destroys operon/neighborhood syntax while preserving protein content.
4. **Random window control:** tests whether marker-centered signal is real.
5. **Source label leakage test:** checks whether gLM features encode source/ecosystem artifacts more strongly than biology.
6. **QC-stratified sensitivity:** repeats all results by CheckM2/GUNC/protein-count/N50 tier.
7. **Fragmentation sensitivity:** compares high-N50 versus low-N50 MAGs, especially in MSM.
8. **Known-marker holdout:** masks known genes and tests whether context recovers mechanism class.

### Required Business/Product Tests

The gLM layer is useful only if it improves product artifacts:

- bridge candidate cards become more mechanistically specific;
- contradictory ESM2 bridge candidates get downgraded;
- unknown genes near methane/sulfur modules become ranked validation targets;
- MSM mangrove MAGs get richer methane-risk feature profiles;
- partner-facing claim boundaries become stronger, not blurrier.

Recommended scoring output:

```text
contextual_feature_readiness
  ready_for_internal_mbag
  promising_needs_qc
  promising_needs_validation
  blocked_by_fragmentation
  blocked_by_source_leakage
  blocked_by_license_or_reproducibility
```

## Implementation Roadmap

### Phase 1: 10-MAG Smoke Test

Inputs:

- 3 wetland/MUCC marker-rich MAGs;
- 3 rumen MAG-bin bridge candidates;
- 2 MSM archaeal MAGs;
- 2 negative/random MAGs.

Goals:

- verify environment on Apolo H100;
- verify protein/GFF/FNA ID mapping;
- verify original gLM embedding output;
- verify gLM2 sequence construction;
- produce the first `dim_glm_window` and `feature_glm_marker_neighborhood`.

Exit criteria:

- no gene-ID loss in window construction;
- embeddings non-empty and finite;
- output joins back to `proteome_id` and `mag_id`;
- shuffled-order control runs;
- runtime/memory profile recorded.

### Phase 2: POC 625 MAG-Bin Context Layer

Inputs:

- all 625 MAG-bin-comparable POC rows.

Goals:

- rerank ESM2 bridge candidates with context evidence;
- produce context-supported and context-contradicted bridge cards;
- compare original ESM2, functional annotations, and gLM context.

Exit criteria:

- `feature_glm_bridge_context_support` generated;
- bridge cards include gLM context fields;
- source leakage and shuffled-order controls complete;
- no final sample/MRV claims made.

### Phase 3: MSM High-Quality Mangrove Pilot

Inputs:

- MSM completed/quality-gated MAGs first;
- then 80 archaeal MAGs;
- then marker-bearing MAGs;
- then full 1,428 with QC weights.

Goals:

- discover mangrove-specific methane/sulfur/substrate context architectures;
- identify high-value MAGs for MAG-resolved methane environment expansion;
- surface unknown marker-neighbor proteins as validation targets.

Exit criteria:

- MSM feature table includes QC/context completeness tiers;
- low-N50 and low-protein MAGs are retained with explicit caution flags;
- denominator reconciliation against the paper-reported 966 medium/high-quality MAGs remains visible.

### Phase 4: Productionize gLM2

Inputs:

- high-confidence FNA/GFF/FAA-aligned payloads.

Goals:

- replace or complement original gLM with Apache-2.0, mixed-modality gLM2 for product-safe deployment;
- add intergenic-context features;
- establish stable H100 inference scripts and cached model assets.

Exit criteria:

- gLM2 features reproduce or improve original gLM pilot findings;
- model/license provenance is recorded;
- output contracts are stable enough to join into the cohort warehouse.

## Claim Boundaries

Allowed now:

> gLM/gLM2 can be piloted as a contextual genomic feature layer to test whether MethaNet bridge candidates and mangrove MAGs carry gene-neighborhood support for methane, sulfur, substrate, and mobile-context mechanisms.

Allowed after successful POC validation:

> MethaNet's multi-view atlas identifies MAG-level methane-relevant candidates whose protein embeddings, functional annotations, QC, and genomic context agree.

Not allowed:

> gLM proves measured methane flux.

Not allowed:

> gLM makes final sample/project methane-risk tiers scoreable by itself.

Not allowed:

> gLM removes source/ecosystem confounding in the original rumen-vs-wetland POC.

Not allowed:

> gLM outputs are carbon-credit approval evidence without abundance, environmental covariates, uncertainty propagation, flux/process validation, and methodology alignment.

## Immediate Next Actions

1. Create a small, versioned gLM smoke manifest with 10 MAGs and explicit windows.
2. Build a reusable `prepare_glm_windows` script that emits both original gLM and gLM2 input contracts from FNA/GFF/FAA.
3. Set up `methanet-glm2` on Apolo first because H100 + bf16 + Apache-2.0 is clean.
4. Set up original gLM second, using modern CUDA/PyTorch and existing MethaNet ESM2 embeddings where possible.
5. Run shuffled-order and random-window controls from day one.
6. Add `feature_glm_bridge_context_support` only after smoke outputs pass finite-value, join, QC, and source-control gates.

## Bottom Line

This is one of the best available next layers for MethaNet. The low-hanging fruit is not "switch from ESM2 to a genomic model." The low-hanging fruit is to add contextual genomic evidence around the exact methane/sulfur/substrate genes that drive MethaNet's value proposition. Original gLM gives the fastest methane-specific scientific pilot; gLM2 gives the stronger product path. Together they can turn MethaNet from a proteome-geometry atlas into a context-aware molecular attestation system for methane-risk architectures in mangroves, wetlands, and other methane-producing environments.
