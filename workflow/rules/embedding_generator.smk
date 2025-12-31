rule embed_esm2:
    input:
        fasta=f"{MARKER_SEQS}/{{sample}}.fasta",
    output:
        npy=f"{EMBEDDINGS}/{{sample}}_esm2.npy",
    threads: THREADS.get("embedding", 1)
    params:
        model=MODELS_CFG.get("esm2", "facebook/esm2_t33_650M_UR50D"),
        batch_size=EMBEDDING_CFG.get("batch_size", 4),
        device=EMBEDDING_CFG.get("device", "auto"),
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/embed_esm2.py "
                "--input {input.fasta} --output {output.npy} "
                "--model {params.model} --batch-size {params.batch_size} "
                "--device {params.device}"
            )


rule embed_dnabert2:
    input:
        fasta=f"{DNA_SEQS}/{{sample}}.fasta",
    output:
        npy=f"{EMBEDDINGS}/{{sample}}_dnabert2.npy",
    threads: THREADS.get("embedding", 1)
    params:
        model=MODELS_CFG.get("dnabert2", "zhihan1996/DNABERT-2-117M"),
        batch_size=EMBEDDING_CFG.get("batch_size", 4),
        device=EMBEDDING_CFG.get("device", "auto"),
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/embed_dnabert2.py "
                "--input {input.fasta} --output {output.npy} "
                "--model {params.model} --batch-size {params.batch_size} "
                "--device {params.device}"
            )
