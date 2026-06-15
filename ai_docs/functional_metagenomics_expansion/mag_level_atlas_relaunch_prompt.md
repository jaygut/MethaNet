# MethaNet MAG-Level Functional Atlas Relaunch Prompt

Date: 2026-06-14

Purpose: copy-ready operating prompt for resuming the MethaNet functional-genomics run after discovering a unit-of-analysis mismatch between geometry-aware ESM2 proteome embeddings and functional annotations for early rumen `10676_*_idba` assembly-scale records.

This prompt is intended for a future MethaNet agent/operator working inside:

```text
/home/rsg-jcorre38/Jay_Proyects/MethaNet
```

It should be used before canceling, relaunching, consolidating, reporting, or interpreting the functional atlas after the June 2026 Apollo-3 production run.

---

## Copy-Ready Prompt

You are operating inside the MethaNet repository on Apolo-3 as a multidisciplinary team: computational biologist, microbial ecologist, bioinformatician, graph ML scientist, statistician, data architect, MRV strategist, and scientific-communication lead.

Your task is to recover, harden, and relaunch the MethaNet functional-genomics atlas in the most scientifically defensible way possible after discovering that some rumen functional annotation inputs are assembly-scale rather than MAG/bin-scale. The final objective is a clean, auditable, MAG-level functional atlas that can be integrated with geometry-aware ESM2/proteome embeddings for MBAG bridge-candidate interpretation, while preserving assembly-scale outputs as separate contextual evidence rather than letting them contaminate MAG-level downstream analyses.

Do not overclaim. Do not collapse MAG-level, assembly-level, sample-level, and MRV-level evidence into one bucket. MethaNet's immediate product primitive is molecular screening and bridge-candidate prioritization, not final carbon-credit approval, not measured methane flux, and not calibrated A-E methane-risk scoring.

### Required Local Grounding

Start by reading these files before making decisions:

```text
AGENTS.md
ai_docs/functional_metagenomics_expansion/final_mrv_risk_scoring_roadmap.md
ai_docs/functional_metagenomics_expansion/data_aggregation_strategy.md
ai_docs/functional_metagenomics_expansion/cohort_data_architecture_hardening.md
ai_docs/functional_metagenomics_expansion/output_contracts_and_gates.md
ai_docs/functional_metagenomics_expansion/pipeline_reproducibility_contract.md
ai_docs/functional_metagenomics_expansion/embedding_functional_transfer_framework/methanet_embedding_functional_transfer_framework.md
ai_docs/functional_metagenomics_expansion/proteome_crosswalk/embedded_662_proteome_id_crosswalk.tsv
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.proposed.tsv
```

Use these as the source of truth unless direct live checks prove that operational state has changed.

### Non-Negotiable Scientific Framing

The central issue is a unit-of-analysis mismatch:

- ESM2 geometry uses `proteome_id` records from the 662-proteome POC.
- Wetland/MUCC functional inputs are MAG/bin-like and comparable to their ESM2 proteomes.
- Most rumen `idba_bin.*` records are MAG/bin-like and should be comparable once run.
- Early rumen no-bin `10676_*_idba` records are assembly-scale inputs. Several produce tens to hundreds of thousands of predicted proteins, while the ESM2 POC used a capped/subsampled 6,000-protein representation for those same `proteome_id` values.

Therefore:

```text
Do not use no-bin rumen 10676_*_idba functional outputs as MAG-level evidence in MBAG.
```

They may still be useful as:

- source-level rumen assembly context;
- metagenome/assembly reservoir evidence;
- future sample/community-context evidence;
- operational stress tests for the pipeline.

They must not be used as direct MAG-level bridge-candidate functional support.

### Immediate Live-State Check

Before making any operational change, refresh live status:

```bash
cd /home/rsg-jcorre38/Jay_Proyects/MethaNet

squeue -u "$USER" -o "%.18i %.10P %.24j %.2t %.10M %.10l %.5D %.5C %.10m %.20R"

find results/functional_metagenomics/fgx_662_apollo3_20260612/per_mag \
  -name COMPLETE | wc -l

find results/functional_metagenomics/fgx_662_apollo3_20260612/per_mag \
  -name FAILED | wc -l
```

Then identify active partials:

```bash
find results/functional_metagenomics/fgx_662_apollo3_20260612/per_mag \
  -mindepth 2 -maxdepth 2 -type d \
  | while read -r d; do
      test -f "$d/COMPLETE" && continue
      test -f "$d/FAILED" && continue
      test -f "$d/status.tsv" && echo "$d"
    done
```

If Slurm array `8504` is still running the problematic no-bin rumen tranche, and the user explicitly authorizes cancellation, cancel it:

```bash
scancel 8504
```

Do not delete outputs. Do not prune partial folders. Do not rewrite production run directories. Treat all per-MAG folders as evidence bundles.

### Preserve And Classify Current Evidence

The current production run directory is expected to be:

```text
results/functional_metagenomics/fgx_662_apollo3_20260612/
```

Preserve all completed/partial outputs in place.

Create or specify a reviewed classification layer that assigns every `proteome_id` to an analytical unit:

```text
analysis_unit_type
  mag_bin
  assembly_context
  embedding_subset
  unresolved

mbag_mag_level_include
  true
  false

assembly_context_include
  true
  false

claim_scope
  MAG functional potential
  assembly/metagenome context
  embedding-only hypothesis
  not comparable
```

Minimum required classification fields:

```text
cohort_run_id
proteome_id
mag_id
source
ecosystem
mag_fasta
proteome_faa
mag_fasta_basename
proteome_faa_basename
source_analysis_accession
analysis_alias
source_filename
n_proteins_used
functional_predicted_proteins
scope_ratio
input_total_bp
input_contigs
input_n50_bp
input_fasta_compressed_bytes
input_fasta_uncompressed_bytes
input_fasta_kind
analysis_unit_type
mbag_mag_level_include
assembly_context_include
claim_scope
comparability_status
comparability_reason
recommended_action
```

Recommended classification rules:

| Condition | Classification | MBAG MAG-level include? |
| --- | --- | --- |
| FASTA basename contains `bin.` or source is wetland/MUCC MAG/bin and `scope_ratio` is near 1 | `mag_bin` | yes |
| no-bin rumen `10676_*_idba`, input FASTA >10-20 Mbp, or `scope_ratio` >2 | `assembly_context` | no |
| functional annotation is run only on the exact proteins used by ESM2 | `embedding_subset` | no by default; use as comparability QA |
| incomplete paths, ambiguous IDs, or contradictory evidence | `unresolved` | no |

Use `proteome_id` as the canonical key, but never assume that a `proteome_id` is biologically equivalent to one MAG until the classification gate supports it.

### What To Do With Existing Outputs

Treat existing outputs as follows:

| Evidence group | Expected treatment |
| --- | --- |
| 107 completed wetland/MUCC outputs | Preserve as valid MAG/bin-level evidence if QC and table gates pass. |
| completed rumen no-bin `10676_*_idba` outputs | Preserve as `assembly_context`; exclude from MAG-level MBAG and MAG mechanism cards. |
| partial no-bin rumen outputs from canceled jobs | Preserve as partial `assembly_context` attempts; include in run-status tables, not in analytical feature matrices unless explicitly reviewed. |
| pending no-bin rumen `10676_*_idba` jobs | Do not relaunch as MAG-level annotation. Consider a separate assembly-context tranche only after the MAG-level atlas is stable. |
| pending rumen `idba_bin.*` jobs | Relaunch/continue as the primary rumen MAG-level production target after classification and manifest filtering. |

No current evidence should be thrown away. The fix is analytical quarantine plus a clean relaunch, not destructive cleanup.

### Clean Relaunch Strategy

The relaunch must use a filtered manifest, not raw task-number assumptions.

Create a new manifest from the 662-row backbone with explicit unit classification. The manifest should select only `analysis_unit_type == mag_bin` and `mbag_mag_level_include == true` for the primary MAG-level production run.

Recommended new manifest names:

```text
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.with_unit_scope.tsv
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.mag_bin_only.tsv
results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.assembly_context.tsv
```

Recommended new run IDs:

```text
fgx_magbin_apollo3_YYYYMMDD
fgx_assembly_context_apollo3_YYYYMMDD
fgx_embedding_subset_apollo3_YYYYMMDD
```

Do not reuse `fgx_662_apollo3_20260612` for a conceptually different run unless the goal is only to preserve historical status.

### Resource Strategy

Use separate resource profiles by analytical unit.

Primary MAG/bin tranche:

```text
analysis_unit_type: mag_bin
cpus: 8-16
memory: 32G-64G initially, increase only if evidence requires
walltime: 4h-8h initially
parallelism: tune to cluster availability and I/O pressure
```

Assembly-context tranche:

```text
analysis_unit_type: assembly_context
cpus: 16-32
memory: 128G or more if required
walltime: 48h-72h for the largest assemblies
parallelism: low, intentionally throttled
```

Embedding-subset QA tranche:

```text
analysis_unit_type: embedding_subset
cpus: 4-8
memory: 16G-32G
walltime: 2h-4h
purpose: direct function-on-embedded-protein-set comparability check
```

The MAG/bin run should be the main priority. Assembly-scale annotation is scientifically useful but must not block the MBAG MAG-level atlas.

### Pipeline And Script Requirements Before Relaunch

Before submitting new jobs, ensure the pipeline can enforce:

1. Unit-scope columns are present in the manifest.
2. The runner refuses to run `assembly_context` inputs unless explicitly invoked in assembly-context mode.
3. The cohort consolidator preserves all attempts in `fact_run_status`.
4. The cohort consolidator excludes non-MAG units from `dim_mag`, `feature_methane_mechanism`, `feature_sulfur_competition`, `feature_mrv_mag_level`, and MBAG candidate cards by default.
5. Assembly-context outputs write to distinct tables or carry `analysis_unit_type = assembly_context`.
6. All cohort-level tables retain:

```text
cohort_run_id
run_id
proteome_id
mag_id
source_tool
analysis_unit_type
claim_scope
```

Where an existing table model cannot yet carry `analysis_unit_type`, add it before relaunch or block downstream consolidation.

### Required Validation Gates

Add or enforce the following gates before any downstream analytics:

#### Gate 1: Identity And Denominator

- The 662-row embedded POC backbone remains intact.
- Every `proteome_id` has exactly one unit-scope classification.
- The excluded MUCC coassembly remains excluded unless explicitly marked as a control.
- No downstream table defines the cohort by successful annotation alone.

#### Gate 2: MAG-Level Comparability

- `dim_mag` contains only `analysis_unit_type = mag_bin`.
- `mbag_mag_level_include = true` is required for MAG-level MBAG.
- No no-bin rumen `10676_*_idba` record enters MAG-level bridge cards.
- `scope_ratio` is reported and reviewed for every completed run.

#### Gate 3: Functional Completeness

- Every selected completed MAG has:

```text
curated/run_record.json
curated/file_manifest.tsv
curated/parquet_manifest.tsv
fact_tool_timing
fact_qc_checkm2
fact_qc_gunc
fact_kofam_hits
fact_mcycdb_hits
fact_scycdb_hits
fact_dbcan_hits
fact_bakta_features
normalized METABOLIC tables
feature_annotation_coverage
```

#### Gate 4: Annotation Interpretation

- KOfam accepted hits must be distinguishable from all KOfam hits.
- MCycDB and SCycDB best-hit ranking must be present.
- Annotation coverage must be computed per MAG x tool.
- Missing methane/sulfur/substrate pathways must be caveated by completeness, contamination, GUNC, and annotation coverage.

#### Gate 5: METABOLIC Long-Form Integrity

- No tool-native wide MAG columns in cohort analytical tables.
- METABOLIC outputs are normalized into:

```text
fact_metabolic_hmm_hits
fact_metabolic_function_presence
fact_metabolic_module_presence
fact_metabolic_module_step_presence
fact_cazy_hits
fact_merops_hits
```

#### Gate 6: MBAG Claim Boundary

- Embedding-only bridge candidates remain hypotheses.
- Functional evidence can support MAG-level mechanism potential only for comparable MAG/bin units.
- Assembly-context evidence can support source/community context only.
- No final MRV risk tiers.
- No carbon-credit approval or measured methane flux claims.

### Downstream Analytics Policy

Downstream analytics must operate on three explicit evidence lanes:

#### Lane A: MAG-Level MBAG Lane

Use only:

```text
analysis_unit_type = mag_bin
mbag_mag_level_include = true
```

Allowed outputs:

- bridge candidate cards;
- MAG-level methane/sulfur/substrate mechanism features;
- QC-aware MAG-level functional potential;
- embedding-function concordance;
- uncertainty and missing-evidence labels;
- partner-facing candidate prioritization with caveats.

Not allowed:

- sample-level methane-risk scoring;
- final A-E tiers;
- registry or carbon-credit claims;
- source-independent transfer proof.

#### Lane B: Assembly-Context Lane

Use:

```text
analysis_unit_type = assembly_context
assembly_context_include = true
```

Allowed outputs:

- source-level rumen functional reservoir context;
- operational stress-test metrics;
- evidence that a source assembly contains methane/sulfur/substrate potential;
- future sample/metagenome rollup planning.

Not allowed:

- MAG-level pathway completeness;
- MAG-level MBAG bridge support;
- direct comparison to wetland MAGs as if all units are genomes.

#### Lane C: Embedding-Subset QA Lane

Use only if a protein-subset annotation workflow is intentionally built.

Allowed outputs:

- direct check of whether proteins used in ESM2 carry detectable functional labels;
- sanity-check bridge-feature enrichment;
- diagnostic comparison between embedded subset and full MAG/bin annotation.

Not allowed:

- absence claims;
- full pathway completeness;
- replacement of MAG/bin annotation.

### Recommended Analytical Outputs After Relaunch

Once a clean MAG/bin cohort is complete, produce:

```text
cohort_identity_with_unit_scope.tsv
fact_run_status.parquet
fact_tool_timing.parquet
dim_mag.parquet
dim_gene.parquet
feature_annotation_coverage.parquet
feature_methane_mechanism.parquet
feature_sulfur_competition.parquet
feature_mrv_mag_level.parquet
bridge_mechanism_cards.json
candidate_recommendation_table.tsv
claim_boundary_matrix.tsv
validation_gap_register.tsv
DATA_ARCHITECTURE_VALIDATION.md
```

For every partner-facing or investor-facing result, include:

```text
allowed_claim
evidence_status
blocking_gap
next_validation_action
claim_scope
confidence_tier
```

### MBAG Integration Requirements

The MBAG model should use only comparable MAG/bin units for MAG-level scoring.

Inputs:

- ESM2/proteome latent features keyed by `proteome_id`;
- functional features from comparable MAG/bin annotations;
- QC and taxonomy;
- annotation coverage;
- source/ecosystem labels;
- run status and missingness;
- optional assembly-context features as source-level covariates, not MAG-level facts.

MBAG must report:

```text
latent_priority
functional_support
qc_penalty
coverage_penalty
source_leakage_flag
analysis_unit_type
evidence_tier
claim_scope
recommended_action
```

Recommended candidate tiers:

| Tier | Meaning |
| --- | --- |
| `high_evidence_bridge_candidate` | latent bridge plus comparable MAG-level functional support, good QC, adequate coverage |
| `moderate_evidence_bridge_candidate` | latent bridge plus partial functional support or moderate uncertainty |
| `hypothesis_only_bridge_candidate` | latent bridge, functional data pending or insufficient |
| `blocked_noncomparable_unit` | assembly-context or unresolved unit; not valid for MAG-level MBAG |
| `blocked_qc_or_coverage` | comparable unit but QC/coverage prevents interpretation |

### Statistical And Validation Plan

After clean MAG/bin consolidation:

1. Compare embeddings-only, functions-only, and hybrid MBAG rankings.
2. Run source-aware nulls and within-source permutations.
3. Downsample rumen to wetland-scale denominators.
4. Bootstrap bridge ranks by feature group and by MAG.
5. Report confidence intervals or stability tiers, not just point ranks.
6. Check whether bridge evidence survives removal of low-completeness/high-contamination MAGs.
7. Check whether bridge evidence survives annotation-coverage filtering.
8. Report top candidates with direct evidence, missing evidence, and blockers separated.

No claim of source-independent transfer is allowed until source-aware validation is adequate.

### Operational Deliverables For The Next Agent

Produce, in order:

1. A live run-state snapshot:

```text
slurm_status
complete_count
failed_count
partial_count
running_tasks
pending_tasks
```

2. A unit-scope audit:

```text
counts by analysis_unit_type
counts by source x analysis_unit_type
list of no-bin rumen assembly-context records
list of MAG/bin-comparable records
list of unresolved records
```

3. A recommended action:

```text
cancel_current_array: yes/no/already_done
relaunch_mag_bin_only: yes/no
quarantine_assembly_context: yes/no
```

4. A clean manifest or clear specification for it.

5. A launch command in dry-run form first.

6. Only after user approval, submit the clean MAG/bin run.

7. After completion, consolidate and validate.

8. Generate MBAG/MethaNet intelligence outputs only from comparable MAG/bin units.

### Strict Prohibitions

Do not:

- delete production outputs without explicit user approval;
- stage generated result folders into git;
- silently overwrite historical run directories;
- use assembly-context outputs in MAG-level MBAG;
- describe MAG-level evidence as sample/metagenome-level ecology;
- describe molecular screening as final MRV risk scoring;
- assign final A-E methane-risk tiers;
- claim carbon-credit approval, measured methane flux, or registry readiness;
- use successful outputs alone as the cohort denominator;
- hide failed, partial, skipped, or non-comparable units.

### Desired End State

The desired end state is:

1. A clean MAG/bin-level functional atlas with explicit unit-scope classification.
2. Assembly-scale rumen outputs preserved as contextual evidence, not discarded and not mixed into MAG-level evidence.
3. A Parquet/DuckDB cohort warehouse with validation gates passing.
4. MBAG bridge-candidate cards that integrate ESM2 geometry with comparable MAG-level functional evidence.
5. A partner-facing MethaNet Intelligence Report that is visually compelling, scientifically cautious, and claim-boundary-safe.
6. A clear next-step path from MAG-level screening toward sample/metagenome-level MRV readiness through abundance, environmental metadata, source-aware validation, and field/process validation.

### Final Communication Contract

When reporting back, clearly separate:

```text
What was checked
What was changed
What was preserved
What is comparable
What is quarantined
What can be claimed now
What remains blocked
What exact next action is recommended
```

Use precise language:

- Say "MAG-level functional potential" for comparable bins.
- Say "assembly-context functional reservoir" for no-bin large rumen assemblies.
- Say "hypothesis-generating ESM2 bridge geometry" for latent evidence.
- Say "not final MRV risk scoring" unless the missing sample-level evidence and validation have been added.

Finish only when the operational plan and scientific claim boundaries are aligned.

---

## Rationale For This Prompt

This relaunch prompt exists because the June 2026 production run surfaced a high-impact mismatch:

- Wetland/MUCC functional runs behaved like MAG/bin annotations.
- Most completed rumen no-bin `10676_*_idba` runs behaved like large assembly annotations.
- The ESM2 POC used capped/subsampled proteome evidence for some of those large rumen records.
- Directly joining assembly-scale functional breadth to MAG-level bridge geometry would inflate signal and weaken scientific credibility.

The solution is not to discard data. The solution is to make evidence grain explicit and to enforce separate analytical lanes.

The high-quality path for MethaNet is:

```text
preserve everything
classify analytical unit
exclude non-comparable units from MAG-level MBAG
relaunch comparable MAG/bin production
validate schemas and claim boundaries
then generate intelligence outputs
```

