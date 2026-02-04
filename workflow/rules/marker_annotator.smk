# Get list of markers from config
MARKERS = [m["name"] for m in config["functional"]["markers"]]

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


# Dynamic rule generation for all configured markers
rule hmmsearch_marker:
    input:
        proteins=f"{ORFS}/{{sample}}.faa",
        hmm=f"{HMM_DIR}/{{marker}}.hmm",
    output:
        hits=f"{MARKER_HITS}/{{sample}}/{{marker}}.tbl",
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
        # Dynamically request all marker hits
        hits=expand(f"{MARKER_HITS}/{{sample}}/{{marker}}.tbl", marker=MARKERS)
    output:
        fasta=f"{MARKER_SEQS}/{{sample}}.fasta",
    params:
        evalue_threshold=FUNCTIONAL_CFG.get("evalue_threshold", 1e-10),
    threads: THREADS.get("hmmer", 4)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            # Construct --hits arguments dynamically
            hits_args = " ".join([f"--hits {h}" for h in input.hits])
            shell(
                "python workflow/scripts/extract_marker_seqs.py "
                "--proteins {input.proteins} "
                "{hits_args} "
                "--evalue-threshold {params.evalue_threshold} "
                "--output {output.fasta}"
            )


rule build_functional_features:
    input:
        proteins=f"{ORFS}/{{sample}}.faa",
        # Dynamically request all marker hits
        hits=expand(f"{MARKER_HITS}/{{sample}}/{{marker}}.tbl", marker=MARKERS)
    output:
        features=f"{FUNCTIONAL_FEATURES}/{{sample}}.tsv",
    params:
        evalue_threshold=FUNCTIONAL_CFG.get("evalue_threshold", 1e-10),
        markers=MARKERS
    threads: THREADS.get("hmmer", 4)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            # Construct arguments like --mcrA path/to/mcrA.tbl
            marker_args = []
            for marker_name, hit_path in zip(params.markers, input.hits):
                marker_args.append(f"--{marker_name} {hit_path}")
            
            cmd_args = " ".join(marker_args)
            
            shell(
                "python workflow/scripts/build_functional_features.py "
                "--sample-id {wildcards.sample} "
                "--proteins {input.proteins} "
                "{cmd_args} "
                "--evalue-threshold {params.evalue_threshold} "
                "--output {output.features}"
            )
