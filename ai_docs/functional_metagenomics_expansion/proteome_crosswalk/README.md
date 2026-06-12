# Proteome ID Crosswalk for the 662-Genome POC

Source artifacts: `results/blue_catalyst_poc/runs/apolo_full_20260228_080644_embed_20260305_061952/artifacts`

Files generated here:

- `embedded_662_proteome_id_crosswalk.tsv`: final embedded proteomes used in the geometry-aware ESM2 analysis.
- `embedded_662_proteome_ids.txt`: one proteome ID per line for quick matching.
- `input_663_proteome_manifest_with_embedding_status.tsv`: pre-final input proteome manifest with the one excluded coassembly marked.
- `proteome_crosswalk_summary.tsv`: count/QC summary.

Key result: final embedded set has 662 unique proteome IDs: 555 rumen + 107 wetland/MUCC. The input manifest has 663 proteomes because `mucc__PPR_1022_P7D_M_E_concat_coassembly_mesocosms_megahit_bin.197` was excluded before final embedding.

Cross-reference guidance:

- Use `proteome_id` or `sample` as the canonical POC proteome identifier.
- Use `mag_id_candidate` to match against MAG FASTA/bin names after removing the `mucc__` or `rumen__` source prefix.
- For rumen genomes, also use `source_analysis_accession` (`ERZ...`) plus `analysis_alias`/`filename` from PRJEB31266.
- For wetland/MUCC genomes, `source_analysis_accession` is absent in the current metadata, so matching should use `mag_id_candidate`, `proteome_faa_stem`, or the original MAG FASTA basename.
