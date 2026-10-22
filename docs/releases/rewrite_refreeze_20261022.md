# History rewrite re-freeze record, 22 October 2026

On 22 October 2026 the repository history was rewritten with git-filter-repo before it was published again. This record shows what changed and confirms that the 10 August 2026 molecular atlas release keeps its meaning.

## What changed

- Every commit's author and committer now read Jay Gutierrez <jay.guti40@gmail.com>. The GitHub Pages deploy commits by github-actions[bot] on `gh-pages` keep that identity. Commit dates, parents and Co-Authored-By trailers are unchanged.
- In every file of every commit, machine-specific details were replaced by tokens. The checkout root became `$METHANET_ROOT`, the scratch root `$METHANET_SCRATCH`, the account home `$HOME`, the cluster account name `$HPC_USER` and the cluster login host `$HPC_HOST`. The cluster's name in prose became "HPC", and retired contact addresses became jay.guti40@gmail.com. Commit messages received the same replacements.
- Every commit SHA changed. Commit SHAs cited in tracked files at the branch tips were updated from the commit map, as listed below.
- Scripts now read `METHANET_ROOT`, `METHANET_SCRATCH`, `HPC_HOST` and `HPC_USER` from the environment or the git-ignored `.methanet.env` (see `.methanet.env.example`).

The history before the rewrite is preserved in a sealed private archive and is not published.

## Release integrity for `molecular-atlas-20260810`

Every file named by [`configs/atlas_current_release.json`](../../configs/atlas_current_release.json), hashed before and after the rewrite. Ignored artifacts live under `results/` and were not touched.

| Pointer field | File | Kind | Declared SHA-256 | Before | After | Status |
|---|---|---|---|---|---|---|
| `lane_registry` | `configs/methanet_atlas_lanes.tsv` | tracked | `7e02e18033b1e33b` | `7e02e18033b1e33b` | `7e02e18033b1e33b` | unchanged, matches declared |
| `freeze_manifest` | `results/reports/methanet_3view_payload_freeze_20260810_end_to_end/freeze_manifest.tsv` | ignored, local copy | `7dd870ac3cdf0142` | `7dd870ac3cdf0142` | `7dd870ac3cdf0142` | unchanged, matches declared |
| `freeze_decision` | `results/reports/methanet_3view_payload_freeze_20260810_end_to_end/freeze_decision.json` | ignored, local copy | `c59abd192f46124d` | `c59abd192f46124d` | `c59abd192f46124d` | unchanged, matches declared |
| `release_ledger` | `results/reports/methanet_3view_payload_freeze_20260810_end_to_end/release_ledger.json` | ignored, local copy | `e59ee4df0e244e73` | `e59ee4df0e244e73` | `e59ee4df0e244e73` | unchanged, matches declared |
| `tracked_ledger_snapshot` | `docs/releases/atlas_20260810_release_ledger.json` | tracked | `none` | `e59ee4df0e244e73` | `e59ee4df0e244e73` | unchanged |
| `embedding_lane_registry` | `configs/methanet_atlas_lanes_20260929.tsv` | tracked | `53b8f8778bf8deeb` | `53b8f8778bf8deeb` | `53b8f8778bf8deeb` | unchanged, matches declared |
| `embedding_contract` | `configs/atlas_embedding_contract_20260929.json` | tracked | `fae6d2c31c0cba7c` | `fae6d2c31c0cba7c` | `fae6d2c31c0cba7c` | unchanged, matches declared |
| `geometry_receipt` | `docs/releases/atlas_embedding_reconciliation_20260929.json` | tracked | `none` | `084c1579f96d0c1f` | `084c1579f96d0c1f` | unchanged |

Hashes are shown to 16 characters; the full values are in the pointer and can be recomputed with `sha256sum`.

## Branch tips

| Branch | Before | After rewrite |
|---|---|---|
| `codex/embedding-reconciliation-grant-deck` | `c0662a2378c538640c0035c2d034c5f11b8e4a6c` | `779ab8b60d34f022dd68c38cd554515bfeb74b69` |
| `fix/strategic-review-20261021` | `0dcba95e2837ea5073e09d0026aeed9405eea24f` | `c786a8566d1379ead8837c02ca51d88ad81eaf65` |
| `gh-pages` | `234d9e108fbb43d378419dd96b14a0695dd14cce` | `fa7d6519eab8cd855bac2722c294a330b603e053` |
| `main` | `41ac4bf79b430c758100401fd3cb3bd13edd964a` | `556da34d79492d74ceeec432cb09178c163e4603` |
| `site/core-comparison-withdrawal-20260928` | `0a868b7849b16c766493468c58746e55f3285a64` | `3403f3241665935ad547e6083f21cc517d84ba69` |

Forward-fix commits sit on top of these rewritten tips.

## Commit SHAs cited in tracked files at the `main` tip

| File | Line | Before | After |
|---|---|---|---|
| `docs/current_artifact_inventory.md` | 485 | `458954b545ed21c2adbf099e9797d1f07e459b37` | `1f22abe44877b1ba9371fbb9791a7fe1579aa5cb` |
| `docs/current_artifact_inventory.md` | 498 | `30d8715` | `84e47ac` |
| `docs/project_state_20260927.md` | 128 | `458954b` | `1f22abe` |
| `docs/project_state_20260927.md` | 128 | `458954b545ed21c2adbf099e9797d1f07e459b37` | `1f22abe44877b1ba9371fbb9791a7fe1579aa5cb` |
| `docs/releases/emergentbiome_report_review_20261013.json` | 6 | `234d9e108fbb43d378419dd96b14a0695dd14cce` | `fa7d6519eab8cd855bac2722c294a330b603e053` |
| `results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/README.md` | 9 | `190bc69` | `0466712` |
| `web/emergentbiome-methanet/README.md` | 18 | `30d8715` | `84e47ac` |
| `web/emergentbiome-methanet/design-qa.md` | 18 | `d9c6665` | `f5c9e7a` |
| `web/emergentbiome-methanet/design-qa.md` | 123 | `30d8715` | `84e47ac` |
| `web/emergentbiome-methanet/design-qa.md` | 123 | `458954b` | `1f22abe` |
| `web/emergentbiome-methanet/design-qa.md` | 151 | `8c58cd0` | `486aaf0` |

## Tracked files whose bytes changed at the `main` tip

SHA-256 of each file before the rewrite and at the rewritten `main` tip, before forward fixes. No release pointer or receipt records the hash of any of these files.

| File | Before | After |
|---|---|---|
| `CITATION.cff` | `edf5832d39c9eec4` | `5b54e11fbee32482` |
| `README.md` | `0e5371e72ae0e81e` | `c87b4ec1f9c2f410` |
| `ai_docs/functional_metagenomics_expansion/README.md` | `a5f64b597a478557` | `0e76db81e4838b2c` |
| `ai_docs/functional_metagenomics_expansion/mag_level_atlas_relaunch_prompt.md` | `2d81b0d53d2c929a` | `ff2e0185872d270a` |
| `ai_docs/functional_metagenomics_expansion/pipeline_plan.md` | `6ee4102e65532d95` | `2e67a38a001f694f` |
| `ai_docs/functional_metagenomics_expansion/proteome_crosswalk/embedded_662_proteome_id_crosswalk.tsv` | `a4633558978280df` | `c94514ff868ef073` |
| `ai_docs/functional_metagenomics_expansion/proteome_crosswalk/input_663_proteome_manifest_with_embedding_status.tsv` | `85fb1b717c91560b` | `6768503ec2e986b2` |
| `ai_docs/functional_metagenomics_expansion/snakemake_backbone/cluster/config.yaml` | `95a8103839eed54e` | `23c61b5db9e3fe63` |
| `ai_docs/functional_metagenomics_expansion/tool_database_registry.md` | `c2314b66cbd2a67c` | `dc9658e2cf0f9663` |
| `docs/apollo3_functional_mag_runbook.md` | `59215764d2f9479c` | `66b453bd3a978541` |
| `docs/apollo3_mag_functional_analytics_ops.md` | `dd9ccff871a76c0c` | `bc856f1412ac83df` |
| `docs/current_artifact_inventory.md` | `8576194af577fbe6` | `c87feaa1a739851c` |
| `docs/methods.md` | `be9a6fad24b61846` | `bba91eca0c0977bb` |
| `docs/reproducibility.md` | `cf041131e854aa81` | `97a84e62bead769e` |
| `notebooks/blue_catalyst_esm2_poc.executed.archive_local.ipynb` | `ced84c7520e0340f` | `99e3d9072b126c67` |
| `notebooks/blue_catalyst_esm2_poc.ipynb` | `8bbe5e09a10db24f` | `434d682980c7241d` |
| `poc/mucc_rumen_poc/README.md` | `97bf42f42467091e` | `90adc12e8ab7db58` |
| `pyproject.toml` | `1fca9aaf57688a8b` | `d40580a348fb7a6a` |
| `results/functional_metagenomics/calibration_submit_20260612.log` | `268bff7cc82e0fca` | `50d6ec4061728dc7` |
| `results/functional_metagenomics/calibration_submit_20260612_retry1.log` | `3e9ea8aa83f15ef8` | `8f4dc8ca8986b0f1` |
| `results/functional_metagenomics/fgx_662_apollo3_20260612/cohort_warehouse/cohort_table_manifest.tsv` | `e0395be268c0aeb1` | `444592ffec69335b` |
| `results/functional_metagenomics/production_gate_preflight_20260612.json` | `0663a70de7fcd1ac` | `eab8b69265129aa2` |
| `results/functional_metagenomics/production_gate_preflight_20260612_calibration.json` | `0663a70de7fcd1ac` | `eab8b69265129aa2` |
| `results/functional_metagenomics/production_gate_preflight_20260612_precalibration.json` | `0663a70de7fcd1ac` | `eab8b69265129aa2` |
| `results/functional_metagenomics/production_gate_preflight_20260612_resource_review.json` | `0663a70de7fcd1ac` | `eab8b69265129aa2` |
| `results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.assembly_context.tsv` | `0280a107b56eeb55` | `1b840ef4a99ba3dd` |
| `results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.mag_bin_only.tsv` | `ddd2fe01e7b3c612` | `e3f60d818201af43` |
| `results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.mag_bin_remaining.tsv` | `7b4d6f8976f7b1af` | `4b528618319b736b` |
| `results/functional_metagenomics/proteome_crosswalk_audit_20260612_0255/poc_662_functional_mag_manifest.with_unit_scope.tsv` | `736010e48119aa7c` | `e4aa6a439988f1f9` |
| `scripts/build_blue_catalyst_deep_dive_report.py` | `072e6ff8b372745e` | `084965b36dbaa659` |
| `scripts/fetch_apolo_blue_catalyst_artifacts.sh` | `ec76f406aa4ec002` | `82f3026345566a1e` |
