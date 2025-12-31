rule sra_download:
    output:
        r1=f"{RAW_SRA}/{{acc}}/{{acc}}_1.fastq",
        r2=f"{RAW_SRA}/{{acc}}/{{acc}}_2.fastq",
    threads: THREADS.get("download", 8)
    params:
        out_dir=lambda wc: f"{RAW_SRA}/{wc.acc}",
        raw_root=RAW_SRA,
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "mkdir -p {params.out_dir} "
                "&& prefetch {wildcards.acc} --output-directory {params.raw_root} "
                "&& fasterq-dump {wildcards.acc} --split-files --threads {threads} "
                "--outdir {params.out_dir}"
            )


rule ena_download:
    output:
        out_dir=directory(f"{RAW_ENA}/{{acc}}"),
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell("enaDataGet -f fastq -a {wildcards.acc} -d {output.out_dir}")


rule fastp_qc:
    input:
        r1=f"{RAW_SRA}/{{acc}}/{{acc}}_1.fastq",
        r2=f"{RAW_SRA}/{{acc}}/{{acc}}_2.fastq",
    output:
        r1=f"{QC_READS}/{{acc}}_R1.fq.gz",
        r2=f"{QC_READS}/{{acc}}_R2.fq.gz",
        html=f"{QC_READS}/{{acc}}_fastp.html",
        json=f"{QC_READS}/{{acc}}_fastp.json",
    threads: THREADS.get("qc", 8)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "fastp -i {input.r1} -I {input.r2} "
                "-o {output.r1} -O {output.r2} "
                "--html {output.html} --json {output.json} "
                "--thread {threads}"
            )


rule fastqc:
    input:
        r1=f"{QC_READS}/{{acc}}_R1.fq.gz",
        r2=f"{QC_READS}/{{acc}}_R2.fq.gz",
    output:
        qc_dir=directory(f"{QC_READS}/fastqc/{{acc}}"),
    threads: THREADS.get("qc", 8)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "mkdir -p {output.qc_dir} "
                "&& fastqc {input.r1} {input.r2} --outdir {output.qc_dir} --threads {threads}"
            )


rule multiqc:
    input:
        fastqc_dirs=expand(f"{QC_READS}/fastqc/{{acc}}", acc=SRA_ACCESSIONS),
        fastp_jsons=expand(f"{QC_READS}/{{acc}}_fastp.json", acc=SRA_ACCESSIONS),
    output:
        report=f"{QC_REPORTS}/multiqc_report.html",
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell("multiqc {QC_READS} -o {QC_REPORTS}")


rule assembly_qc:
    input:
        fasta=f"{ASSEMBLIES}/{{sample}}.fasta",
    output:
        out_dir=directory(f"{QC_ASSEMBLIES}/{{sample}}"),
    threads: THREADS.get("qc", 8)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell("quast {input.fasta} -o {output.out_dir} -t {threads}")


rule mag_qc:
    input:
        mags=lambda wc: f"{MAGS}/{wc.mag_set}",
    output:
        out_dir=directory(f"{QC_MAGS}/{{mag_set}}"),
    threads: THREADS.get("qc", 8)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell("checkm2 predict --input {input.mags} --output-directory {output.out_dir} --threads {threads}")
