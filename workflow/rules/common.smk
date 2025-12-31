from pathlib import Path
from snakemake.io import directory, is_flagged

SIMULATE = config.get("simulate", False)
STAGES = config.get("stages", {})

def _path(key, default):
    return config.get("paths", {}).get(key, default)

RAW_SRA = _path("raw_sra", "data/raw/sra")
RAW_ENA = _path("raw_ena", "data/raw/ena")
QC_READS = _path("qc_reads", "data/qc_reads")
QC_REPORTS = _path("qc_reports", "data/qc_reports")
QC_ASSEMBLIES = _path("qc_assemblies", "data/qc_assemblies")
QC_MAGS = _path("qc_mags", "data/qc_mags")
ASSEMBLIES = _path("assemblies", "data/assemblies")
MAGS = _path("mags", "data/mags")
HMM_DIR = _path("hmm_dir", "data/hmm")
ORFS = _path("orfs", "features/orfs")
MARKER_HITS = _path("marker_hits", "features/marker_hits")
MARKER_SEQS = _path("marker_sequences", "features/marker_sequences")
DNA_SEQS = _path("dna_sequences", "features/dna_sequences")
MARKER_DB = _path("marker_db", "db/marker_db.mmsdb")
EMBEDDINGS = _path("embeddings", "features/embeddings")
ADAPTED_FEATURES = _path("adapted_features", "features/adapted")
MODELS = _path("models", "models")
REPORTS = _path("reports", "reports")
FIGURES = _path("figures", "figures")
SOURCE_FEATURES = _path("source_features", "features/rumen.parquet")
TARGET_FEATURES = _path("target_features", "features/coastal.parquet")
FLUX_FEATURES = _path("flux_features", "features/all_features.parquet")

SRA_ACCESSIONS = config.get("sra_accessions", [])
ENA_ACCESSIONS = config.get("ena_accessions", [])
ASSEMBLY_SAMPLES = config.get("assembly_samples", [])
MAG_SETS = config.get("mag_sets", [])
MARKER_SAMPLES = config.get("marker_samples", []) or ASSEMBLY_SAMPLES
EMBED_SAMPLES = config.get("embedding_samples", []) or MARKER_SAMPLES
DNA_EMBED_SAMPLES = config.get("dna_embedding_samples", [])
FRAGGENE_INPUTS = config.get("fraggenescan_inputs", {})

THREADS = config.get("threads", {})
MODELS_CFG = config.get("models", {})
EMBEDDING_CFG = config.get("embedding", {})
TRAINING = config.get("training", {})


def ensure_outputs(outputs):
    for out in outputs:
        if is_flagged(out, "directory"):
            Path(out).mkdir(parents=True, exist_ok=True)
            continue
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")


def stage_enabled(name: str) -> bool:
    return STAGES.get(name, False)


ALL_TARGETS = []

if stage_enabled("data_curator"):
    if SRA_ACCESSIONS:
        ALL_TARGETS += expand(f"{QC_READS}/{{acc}}_R1.fq.gz", acc=SRA_ACCESSIONS)
        ALL_TARGETS += expand(f"{QC_READS}/{{acc}}_R2.fq.gz", acc=SRA_ACCESSIONS)
        ALL_TARGETS.append(f"{QC_REPORTS}/multiqc_report.html")
    if ENA_ACCESSIONS:
        ALL_TARGETS += [directory(f"{RAW_ENA}/{acc}") for acc in ENA_ACCESSIONS]
    if ASSEMBLY_SAMPLES:
        ALL_TARGETS += [directory(f"{QC_ASSEMBLIES}/{sample}") for sample in ASSEMBLY_SAMPLES]
    if MAG_SETS:
        ALL_TARGETS += [directory(f"{QC_MAGS}/{mag_set}") for mag_set in MAG_SETS]

if stage_enabled("marker_annotator"):
    if MARKER_SAMPLES:
        ALL_TARGETS += expand(f"{ORFS}/{{sample}}.faa", sample=MARKER_SAMPLES)
        ALL_TARGETS += expand(f"{MARKER_HITS}/{{sample}}/mcrA.tbl", sample=MARKER_SAMPLES)
        ALL_TARGETS += expand(f"{MARKER_HITS}/{{sample}}/pmoA.tbl", sample=MARKER_SAMPLES)
        ALL_TARGETS += expand(f"{MARKER_HITS}/{{sample}}/mmseqs.tsv", sample=MARKER_SAMPLES)
    if FRAGGENE_INPUTS:
        ALL_TARGETS += expand(
            f"{ORFS}/fraggenescan/{{sample}}.faa",
            sample=list(FRAGGENE_INPUTS.keys()),
        )

if stage_enabled("embedding_generator"):
    if EMBED_SAMPLES:
        ALL_TARGETS += expand(f"{EMBEDDINGS}/{{sample}}_esm2.npy", sample=EMBED_SAMPLES)
    if DNA_EMBED_SAMPLES:
        ALL_TARGETS += expand(f"{EMBEDDINGS}/{{sample}}_dnabert2.npy", sample=DNA_EMBED_SAMPLES)

if stage_enabled("domain_adapter"):
    ALL_TARGETS.append(f"{REPORTS}/domain_shift/shift_metrics.json")
    ALL_TARGETS.append(f"{MODELS}/adapted/coral_transform.npy")
    ALL_TARGETS.append(f"{MODELS}/adapted/dann.pt")
    ALL_TARGETS.append(f"{REPORTS}/domain_shift/transferable_features.csv")
    ALL_TARGETS.append(f"{ADAPTED_FEATURES}/source_coral.parquet")
    ALL_TARGETS.append(f"{ADAPTED_FEATURES}/target_coral.parquet")

if stage_enabled("flux_predictor"):
    ALL_TARGETS.append(f"{MODELS}/flux_predictor/model.pkl")
    ALL_TARGETS.append(f"{REPORTS}/flux_predictor/train_metrics.json")
    ALL_TARGETS.append(f"{REPORTS}/flux_predictor/validation_metrics.json")
    ALL_TARGETS.append(f"{REPORTS}/flux_predictor/predictions.csv")
    ALL_TARGETS.append(f"{REPORTS}/flux_predictor/bootstrap_intervals.csv")
    ALL_TARGETS.append(f"{FIGURES}/flux_predictor/shap_summary.png")
