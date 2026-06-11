rule measure_shift:
    input:
        source=SOURCE_FEATURES,
        target=TARGET_FEATURES,
    output:
        metrics=f"{REPORTS}/domain_shift/shift_metrics.json",
    log:
        f"{REPORTS}/logs/measure_shift.log",
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/measure_shift.py "
                "--source {input.source} --target {input.target} --output {output.metrics}"
            )


rule train_coral:
    input:
        source=SOURCE_FEATURES,
        target=TARGET_FEATURES,
    output:
        transform=f"{MODELS}/adapted/coral_transform.npy",
        source_aligned=f"{ADAPTED_FEATURES}/source_coral.parquet",
        target_aligned=f"{ADAPTED_FEATURES}/target_coral.parquet",
    log:
        f"{REPORTS}/logs/train_coral.log",
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/train_coral.py "
                "--source {input.source} --target {input.target} "
                "--output-transform {output.transform} "
                "--output-source {output.source_aligned} --output-target {output.target_aligned}"
            )


rule measure_shift_adapted:
    input:
        source=f"{ADAPTED_FEATURES}/source_coral.parquet",
        target=f"{ADAPTED_FEATURES}/target_coral.parquet",
    output:
        metrics=f"{REPORTS}/domain_shift/shift_metrics_adapted.json",
    log:
        f"{REPORTS}/logs/measure_shift_adapted.log",
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/measure_shift.py "
                "--source {input.source} --target {input.target} --output {output.metrics}"
            )


rule train_dann:
    input:
        source=SOURCE_FEATURES,
        target=TARGET_FEATURES,
    output:
        model=f"{MODELS}/adapted/dann.pt",
        metrics=f"{REPORTS}/domain_shift/dann_metrics.json",
    log:
        f"{REPORTS}/logs/train_dann.log",
    params:
        label_column=TRAINING.get("label_column", "flux_value"),
        epochs=TRAINING.get("dann", {}).get("epochs", 20),
        batch_size=TRAINING.get("dann", {}).get("batch_size", 64),
        lr=TRAINING.get("dann", {}).get("lr", 0.001),
        hidden_dim=TRAINING.get("dann", {}).get("hidden_dim", 256),
        alpha=TRAINING.get("dann", {}).get("alpha", 1.0),
    threads: THREADS.get("training", 1)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/train_dann.py "
                "--source {input.source} --target {input.target} "
                "--label-column {params.label_column} "
                "--epochs {params.epochs} --batch-size {params.batch_size} "
                "--lr {params.lr} --hidden-dim {params.hidden_dim} --alpha {params.alpha} "
                "--output-model {output.model} --metrics-out {output.metrics}"
            )


rule select_transferable:
    input:
        source=SOURCE_FEATURES,
        target=TARGET_FEATURES,
    output:
        csv=f"{REPORTS}/domain_shift/transferable_features.csv",
    log:
        f"{REPORTS}/logs/select_transferable.log",
    params:
        top_k=TRAINING.get("dann", {}).get("top_k", 50),
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/select_transferable.py "
                "--source {input.source} --target {input.target} "
                "--output {output.csv} --top-k {params.top_k}"
            )


rule report_transferability:
    input:
        features=FLUX_FEATURES,
    output:
        report=f"{REPORTS}/poc/poc_report.json",
        domain_shift=f"{REPORTS}/poc/domain_shift_by_group.csv",
        finiteness=f"{REPORTS}/poc/feature_finiteness_by_group.csv",
        transfer=f"{REPORTS}/poc/cross_domain_transfer_by_group.csv",
        summary=f"{REPORTS}/poc/transferability_summary.json",
    log:
        f"{REPORTS}/logs/report_transferability.log",
    params:
        source_domain="rumen",
        target_domain="coastal",
        target_col=TRAINING.get("label_column", "flux_value"),
        functional_dim=config.get("feature_dims", {}).get("functional", 13),
        esm2_dim=config.get("feature_dims", {}).get("esm2", 1280),
        genome_dim=config.get("feature_dims", {}).get(
            "genome",
            EMBEDDING_CFG.get("genome_dim", 768),
        ),
        bootstrap_samples=TRAINING.get("flux", {}).get("bootstrap_samples", 500),
    threads: THREADS.get("training", 1)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            metrics_arg = ""
            if GENOME_BACKEND == "dnabert2":
                metrics_arg = f" --dnabert2-metrics-dir {EMBEDDINGS}"
            shell(
                "python workflow/scripts/report_transferability.py "
                "--features {input.features} --output-dir {REPORTS}/poc "
                "--source-domain {params.source_domain} --target-domain {params.target_domain} "
                "--target-col {params.target_col} "
                "--functional-dim {params.functional_dim} --esm2-dim {params.esm2_dim} "
                "--genome-dim {params.genome_dim} "
                "--bootstrap-samples {params.bootstrap_samples} "
                "--report-out {output.report}"
                + metrics_arg
            )
