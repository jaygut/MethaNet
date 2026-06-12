# Proteome Crosswalk Local FASTA Audit

Generated: 2026-06-12 02:58:20Z
Repository: `/home/rsg-jcorre38/Jay_Proyects/MethaNet`
Crosswalk: `ai_docs/functional_metagenomics_expansion/proteome_crosswalk/embedded_662_proteome_id_crosswalk.tsv`

## Git State

`origin/main` contains `190bc69 docs: add proteome crosswalk for POC cohort`. Local `main` is diverged/ahead with Apollo-specific commits, so the crosswalk folder was restored from `origin/main` into the working tree for this audit without rebasing or overwriting local commits.

## Crosswalk Sanity

- Final embedded proteomes: 662
- Rumen: 555
- Wetland/MUCC: 107
- Input manifest rows: 663
- Excluded input proteome: `mucc__PPR_1022_P7D_M_E_concat_coassembly_mesocosms_megahit_bin.197`

## Local File Inventory

- Candidate MAG nucleotide FASTA files indexed: 1452
- Candidate proteome FAA files indexed: 669
- Main MAG FASTA locations found:
  - `data/assemblies` (108)
  - `data/blue_catalyst_poc/mucc/extracted/Methanoregula_MAGs_DB` (108)
  - `data/blue_catalyst_poc/proteomes/_tmp_rumen` (435)
  - `data/blue_catalyst_poc/proteomes/_tmp_rumen_nuc` (127)
  - `data/blue_catalyst_poc/rumen/raw` (566)
  - `data/mucc/Methanoregula_MAGs_DB/Methanoregula_MAGs_DB` (108)
- Main proteome FAA locations found:
  - `data/blue_catalyst_poc/proteomes` (669)

## Match Results

- Embedded proteomes with local MAG FASTA match: 662/662
  - MUCC/wetland: 107/107
  - Rumen: 555/555
- Embedded proteomes with local proteome FAA match: 662/662
  - MUCC/wetland: 107/107
  - Rumen: 555/555

### Best MAG FASTA Keys Used

- `filename`: 555
- `mag_id_candidate`: 107

### Independent Key Coverage

| file_kind | key | total | mucc | rumen |
| --- | --- | ---: | ---: | ---: |
| mag_fasta | `proteome_id` | 0 | 0 | 0 |
| mag_fasta | `mag_id_candidate` | 662 | 107 | 555 |
| mag_fasta | `proteome_faa_stem` | 0 | 0 | 0 |
| mag_fasta | `source_analysis_accession` | 0 | 0 | 0 |
| mag_fasta | `analysis_alias` | 0 | 0 | 0 |
| mag_fasta | `filename` | 555 | 0 | 555 |
| protein_faa | `proteome_id` | 662 | 107 | 555 |
| protein_faa | `mag_id_candidate` | 0 | 0 | 0 |
| protein_faa | `proteome_faa_stem` | 662 | 107 | 555 |
| protein_faa | `source_analysis_accession` | 0 | 0 | 0 |
| protein_faa | `analysis_alias` | 0 | 0 | 0 |
| protein_faa | `filename` | 0 | 0 | 0 |

## Answers

1. MAG FASTA files are present in `data/assemblies`, `data/blue_catalyst_poc/mucc/extracted/Methanoregula_MAGs_DB`, `data/mucc/Methanoregula_MAGs_DB/Methanoregula_MAGs_DB`, `data/blue_catalyst_poc/rumen/raw`, and `data/blue_catalyst_poc/proteomes/_tmp_rumen_nuc`. The preferred manifest paths should be `data/assemblies` for MUCC/wetland and `data/blue_catalyst_poc/rumen/raw` for rumen because those are clean source nucleotide FASTA pools.
2. `662` of 662 embedded proteome IDs can be matched to local MAG FASTA files.
3. For MAG FASTA matching, `filename` is best for rumen raw files and `mag_id_candidate` is best as a source-independent basename key. For proteome FAA matching, `proteome_faa_stem`/`proteome_id` are exact against `data/blue_catalyst_poc/proteomes`. `source_analysis_accession` is a provenance key into PRJEB31266 metadata, not a local filename key.
4. Unmatched MAG FASTA records: 0. See `unmatched_mag_fasta.tsv`.
5. The 107 wetland/MUCC proteomes are matchable by MAG/bin basename: 107/107 matched.
6. The 555 rumen proteomes are matchable locally by `filename`/MAG basename: 555/555 matched. `source_analysis_accession` is present for all 555 and should be retained for provenance; `analysis_alias` contains the MAG alias but is less direct than `filename` or `mag_id_candidate`.
7. Recommended final MAG manifest schema: `proteome_id`, `sample`, `source`, `ecosystem`, `domain`, `mag_id`, `mag_fasta`, `mag_fasta_basename`, `proteome_faa`, `proteome_faa_basename`, `source_analysis_accession`, `analysis_alias`, `source_filename`, `match_key`, `match_status`, `n_proteins_used`, `embedded_final_662`, `functional_run_include`, `notes`.

## Excluded Coassembly Check

The excluded coassembly is present locally as nucleotide FASTA but correctly absent from the final 662 crosswalk:
- `data/assemblies/PPR_1022_P7D_M_E_concat_coassembly_mesocosms_megahit_bin.197.fasta`
- `data/blue_catalyst_poc/mucc/extracted/Methanoregula_MAGs_DB/PPR_1022_P7D_M_E_concat_coassembly_mesocosms_megahit_bin.197.fna`
- `data/mucc/Methanoregula_MAGs_DB/Methanoregula_MAGs_DB/PPR_1022_P7D_M_E_concat_coassembly_mesocosms_megahit_bin.197.fna`

## Companion Files

- `match_details.tsv`: one row per final embedded proteome with best MAG/proteome matches.
- `unmatched_mag_fasta.tsv`: unmatched final embedded proteomes, if any.
- `directory_inventory.tsv`: indexed local FASTA/FAA directories and counts.
- `key_coverage.tsv`: independent coverage by candidate key.
- `poc_662_functional_mag_manifest.proposed.tsv`: proposed 662-row manifest for the functional-metagenomics pipeline.
