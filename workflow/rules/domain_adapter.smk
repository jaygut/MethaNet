rule measure_shift:
    input:
        source=SOURCE_FEATURES,
        target=TARGET_FEATURES,
    output:
        metrics=f"{REPORTS}/domain_shift/shift_metrics.json",
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


rule train_dann:
    input:
        source=SOURCE_FEATURES,
        target=TARGET_FEATURES,
    output:
        model=f"{MODELS}/adapted/dann.pt",
        metrics=f"{REPORTS}/domain_shift/dann_metrics.json",
    params:
        label_column=TRAINING.get("label_column", "measured_flux"),
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
