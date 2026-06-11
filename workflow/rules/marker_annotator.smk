# Get list of markers from config
MARKERS = [
    marker["name"]
    for marker in config.get("functional", {}).get("markers", [])
    if "name" in marker
]

rule orf_prodigal:
    input:
        fasta=f"{ASSEMBLIES}/{{sample}}.fasta",
    output:
        proteins=f"{ORFS}/{{sample}}.faa",
    log:
        f"{REPORTS}/logs/orf_prodigal/{{sample}}.log",
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
    log:
        f"{REPORTS}/logs/orf_fraggenescan/{{sample}}.log",
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
    log:
        f"{REPORTS}/logs/hmmsearch_marker/{{sample}}_{{marker}}.log",
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
        db=MARKER_DB,
    output:
        hits=f"{MARKER_HITS}/{{sample}}/mmseqs.tsv",
    log:
        f"{REPORTS}/logs/mmseqs_search/{{sample}}.log",
    params:
        tmp_dir=lambda wc: f"{MARKER_HITS}/{wc.sample}/tmp",
    threads: THREADS.get("mmseqs", 8)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "mkdir -p {MARKER_HITS}/{wildcards.sample} {params.tmp_dir} "
                "&& mmseqs easy-search {input.proteins} {input.db} {output.hits} {params.tmp_dir} --threads {threads}"
            )


rule extract_marker_sequences:
    input:
        proteins=f"{ORFS}/{{sample}}.faa",
        # Dynamically request all marker hits
        hits=lambda wc: expand(
            f"{MARKER_HITS}/{wc.sample}/{{marker}}.tbl",
            marker=MARKERS,
        )
    output:
        fasta=f"{MARKER_SEQS}/{{sample}}.fasta",
    log:
        f"{REPORTS}/logs/extract_marker_sequences/{{sample}}.log",
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
        hits=lambda wc: expand(
            f"{MARKER_HITS}/{wc.sample}/{{marker}}.tbl",
            marker=MARKERS,
        )
    output:
        features=f"{FUNCTIONAL_FEATURES}/{{sample}}.tsv",
    log:
        f"{REPORTS}/logs/build_functional_features/{{sample}}.log",
    params:
        evalue_threshold=FUNCTIONAL_CFG.get("evalue_threshold", 1e-10),
        marker_args=lambda wc, input: " ".join(
            f"--{marker_name} {hit_path}"
            for marker_name, hit_path in zip(MARKERS, input.hits)
        ),
    threads: THREADS.get("hmmer", 4)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/build_functional_features.py "
                "--sample-id {wildcards.sample} "
                "--proteins {input.proteins} "
                "{params.marker_args} "
                "--evalue-threshold {params.evalue_threshold} "
                "--output {output.features}"
            )
