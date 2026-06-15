# MethaNet Agent Instructions

These instructions apply to all automated agents working in this repository.

## Start Here For MRV And Functional Atlas Work

For any task involving MethaNet MRV, MBAG, bridge candidates, methane-risk scoring, blue carbon sample interpretation, functional-genomics analytics, carbon-crediting language, or partner-facing intelligence reports, read this roadmap first:

- `ai_docs/functional_metagenomics_expansion/final_mrv_risk_scoring_roadmap.md`

Then read the immediately relevant supporting contracts:

- `ai_docs/functional_metagenomics_expansion/mag_level_atlas_relaunch_prompt.md`
- `ai_docs/functional_metagenomics_expansion/mag_level_atlas_relaunch_recovery_20260614.md`
- `ai_docs/functional_metagenomics_expansion/data_aggregation_strategy.md`
- `ai_docs/functional_metagenomics_expansion/cohort_data_architecture_hardening.md`
- `ai_docs/functional_metagenomics_expansion/output_contracts_and_gates.md`
- `ai_docs/functional_metagenomics_expansion/pipeline_reproducibility_contract.md`
- `ai_docs/functional_metagenomics_expansion/embedding_functional_transfer_framework/methanet_embedding_functional_transfer_framework.md`

## Non-Negotiable Claim Boundaries

- Use `proteome_id` as the canonical cohort key unless a source explicitly requires another key.
- Distinguish MAG/proteome-level functional potential from sample/metagenome-level ecological interpretation.
- Do not describe current MBAG outputs as final MRV risk scores.
- Do not assign final A-E methane-risk tiers until sample mapping, abundance/read coverage, environmental covariates, uncertainty propagation, and flux/process validation exist.
- Do not claim carbon-credit approval, measured methane flux, or source-independent rumen-to-wetland transfer from the current molecular atlas alone.
- Preserve failed, pending, partial, and missing evidence as explicit status rows rather than dropping them from analyses.

## Preferred Output Pattern

For MethaNet intelligence work, produce decision-useful artifacts:

- candidate cards;
- MRV feature tables;
- sample risk readiness tables;
- validation gap registers;
- claim-boundary matrices;
- dashboard/report-ready summaries with explicit caveats.

Every external-facing claim should include allowed wording, evidence status, blocking gaps, and the next validation action.
