rule orf_prodigal:
    input:
        fasta=f"{ASSEMBLIES}/{{sample}}.fasta",
    output:
        proteins=f"{ORFS}/{{sample}}.faa",
    threads: THREADS.get("qc", 8)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell("prodigal -i {input.fasta} -a {output.proteins} -p meta")


rule orf_fraggenescan:
    input:
        fasta=lambda wc: FRAGGENE_INPUTS[wc.sample],
    output:
        proteins=f"{ORFS}/fraggenescan/{{sample}}.faa",
    params:
        out_prefix=lambda wc: f"{ORFS}/fraggenescan/{wc.sample}",
    threads: THREADS.get("qc", 8)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell("FragGeneScanRs -t complete -s {input.fasta} -o {params.out_prefix} -w 1")


rule hmmsearch_mcrA:
    input:
        proteins=f"{ORFS}/{{sample}}.faa",
        hmm=f"{HMM_DIR}/mcrA.hmm",
    output:
        hits=f"{MARKER_HITS}/{{sample}}/mcrA.tbl",
    threads: THREADS.get("hmmer", 4)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "mkdir -p {MARKER_HITS}/{wildcards.sample} "
                "&& hmmsearch --cpu {threads} --tblout {output.hits} {input.hmm} {input.proteins}"
            )


rule hmmsearch_pmoA:
    input:
        proteins=f"{ORFS}/{{sample}}.faa",
        hmm=f"{HMM_DIR}/pmoA.hmm",
    output:
        hits=f"{MARKER_HITS}/{{sample}}/pmoA.tbl",
    threads: THREADS.get("hmmer", 4)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "mkdir -p {MARKER_HITS}/{wildcards.sample} "
                "&& hmmsearch --cpu {threads} --tblout {output.hits} {input.hmm} {input.proteins}"
            )


rule hmmsearch_dsrA:
    input:
        proteins=f"{ORFS}/{{sample}}.faa",
        hmm=f"{HMM_DIR}/dsrA.hmm",
    output:
        hits=f"{MARKER_HITS}/{{sample}}/dsrA.tbl",
    threads: THREADS.get("hmmer", 4)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "mkdir -p {MARKER_HITS}/{wildcards.sample} "
                "&& hmmsearch --cpu {threads} --tblout {output.hits} {input.hmm} {input.proteins}"
            )


rule hmmsearch_nifH:
    input:
        proteins=f"{ORFS}/{{sample}}.faa",
        hmm=f"{HMM_DIR}/nifH.hmm",
    output:
        hits=f"{MARKER_HITS}/{{sample}}/nifH.tbl",
    threads: THREADS.get("hmmer", 4)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "mkdir -p {MARKER_HITS}/{wildcards.sample} "
                "&& hmmsearch --cpu {threads} --tblout {output.hits} {input.hmm} {input.proteins}"
            )


rule hmmsearch_cbbL:
    input:
        proteins=f"{ORFS}/{{sample}}.faa",
        hmm=f"{HMM_DIR}/cbbL.hmm",
    output:
        hits=f"{MARKER_HITS}/{{sample}}/cbbL.tbl",
    threads: THREADS.get("hmmer", 4)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "mkdir -p {MARKER_HITS}/{wildcards.sample} "
                "&& hmmsearch --cpu {threads} --tblout {output.hits} {input.hmm} {input.proteins}"
            )


rule mmseqs_search:
    input:
        proteins=f"{ORFS}/{{sample}}.faa",
    output:
        hits=f"{MARKER_HITS}/{{sample}}/mmseqs.tsv",
    params:
        tmp_dir=lambda wc: f"{MARKER_HITS}/{wc.sample}/tmp",
        db=MARKER_DB,
    threads: THREADS.get("mmseqs", 8)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "mkdir -p {MARKER_HITS}/{wildcards.sample} {params.tmp_dir} "
                "&& mmseqs easy-search {input.proteins} {params.db} {output.hits} {params.tmp_dir} --threads {threads}"
            )


rule extract_marker_sequences:
    input:
        proteins=f"{ORFS}/{{sample}}.faa",
        mcrA=f"{MARKER_HITS}/{{sample}}/mcrA.tbl",
        pmoA=f"{MARKER_HITS}/{{sample}}/pmoA.tbl",
        dsrA=f"{MARKER_HITS}/{{sample}}/dsrA.tbl",
        nifH=f"{MARKER_HITS}/{{sample}}/nifH.tbl",
        cbbL=f"{MARKER_HITS}/{{sample}}/cbbL.tbl",
    output:
        fasta=f"{MARKER_SEQS}/{{sample}}.fasta",
    params:
        evalue_threshold=FUNCTIONAL_CFG.get("evalue_threshold", 1e-10),
    threads: THREADS.get("hmmer", 4)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/extract_marker_seqs.py "
                "--proteins {input.proteins} "
                "--hits {input.mcrA} --hits {input.pmoA} --hits {input.dsrA} "
                "--hits {input.nifH} --hits {input.cbbL} "
                "--evalue-threshold {params.evalue_threshold} "
                "--output {output.fasta}"
            )


rule build_functional_features:
    input:
        proteins=f"{ORFS}/{{sample}}.faa",
        mcrA=f"{MARKER_HITS}/{{sample}}/mcrA.tbl",
        pmoA=f"{MARKER_HITS}/{{sample}}/pmoA.tbl",
        dsrA=f"{MARKER_HITS}/{{sample}}/dsrA.tbl",
        nifH=f"{MARKER_HITS}/{{sample}}/nifH.tbl",
        cbbL=f"{MARKER_HITS}/{{sample}}/cbbL.tbl",
    output:
        features=f"{FUNCTIONAL_FEATURES}/{{sample}}.tsv",
    params:
        evalue_threshold=FUNCTIONAL_CFG.get("evalue_threshold", 1e-10),
    threads: THREADS.get("hmmer", 4)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/build_functional_features.py "
                "--sample-id {wildcards.sample} "
                "--proteins {input.proteins} "
                "--mcrA {input.mcrA} --pmoA {input.pmoA} --dsrA {input.dsrA} "
                "--nifH {input.nifH} --cbbL {input.cbbL} "
                "--evalue-threshold {params.evalue_threshold} "
                "--output {output.features}"
            )
