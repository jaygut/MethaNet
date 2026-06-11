rule fuse_features:
    input:
        metadata="configs/samples.tsv",
        functional=expand(f"{FUNCTIONAL_FEATURES}/{{sample}}.tsv", sample=MARKER_SAMPLES),
        esm2=expand(f"{EMBEDDINGS}/{{sample}}_esm2.npy", sample=EMBED_SAMPLES),
        genome=expand(f"{EMBEDDINGS}/{{sample}}_{GENOME_BACKEND}.npy", sample=DNA_EMBED_SAMPLES),
    output:
        all=FLUX_FEATURES,
        source=SOURCE_FEATURES,
        target=TARGET_FEATURES,
    log:
        f"{REPORTS}/logs/fuse_features.log",
    params:
        genome_backend=GENOME_BACKEND,
        genome_dim=EMBEDDING_CFG.get("genome_dim", 768),
    threads: THREADS.get("embedding", 1)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/fuse_features.py "
                "--metadata {input.metadata} "
                "--functional-dir {FUNCTIONAL_FEATURES} "
                "--esm2-dir {EMBEDDINGS} "
                "--genome-dir {EMBEDDINGS} "
                "--output-all {output.all} "
                "--output-source {output.source} "
                "--output-target {output.target} "
                "--genome-backend {params.genome_backend} "
                "--genome-dim {params.genome_dim}"
            )
