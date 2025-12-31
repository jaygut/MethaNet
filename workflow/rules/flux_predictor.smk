rule train_flux_model:
    input:
        features=FLUX_FEATURES,
    output:
        model=f"{MODELS}/flux_predictor/model.pkl",
        metrics=f"{REPORTS}/flux_predictor/train_metrics.json",
    params:
        target=TRAINING.get("label_column", "measured_flux"),
        model_type=TRAINING.get("flux", {}).get("model", "linear"),
        random_state=TRAINING.get("flux", {}).get("random_state", 42),
    threads: THREADS.get("training", 1)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/train_flux_model.py "
                "--features {input.features} --target {params.target} "
                "--model {params.model_type} --random-state {params.random_state} "
                "--output-model {output.model} --metrics-out {output.metrics}"
            )


rule validate_flux_model:
    input:
        features=FLUX_FEATURES,
        model=f"{MODELS}/flux_predictor/model.pkl",
    output:
        metrics=f"{REPORTS}/flux_predictor/validation_metrics.json",
        predictions=f"{REPORTS}/flux_predictor/predictions.csv",
    params:
        target=TRAINING.get("label_column", "measured_flux"),
        method="loocv",
    threads: THREADS.get("training", 1)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/validate_flux_model.py "
                "--features {input.features} --target {params.target} "
                "--model {input.model} --method {params.method} "
                "--metrics-out {output.metrics} --predictions-out {output.predictions}"
            )


rule bootstrap_ci:
    input:
        features=FLUX_FEATURES,
        model=f"{MODELS}/flux_predictor/model.pkl",
    output:
        intervals=f"{REPORTS}/flux_predictor/bootstrap_intervals.csv",
    params:
        target=TRAINING.get("label_column", "measured_flux"),
        n_boot=TRAINING.get("flux", {}).get("bootstrap_samples", 1000),
        ci=TRAINING.get("flux", {}).get("ci", 0.95),
    threads: THREADS.get("training", 1)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/bootstrap_ci.py "
                "--features {input.features} --target {params.target} "
                "--model {input.model} --n {params.n_boot} --ci {params.ci} "
                "--output {output.intervals}"
            )


rule explain_model:
    input:
        features=FLUX_FEATURES,
        model=f"{MODELS}/flux_predictor/model.pkl",
    output:
        plot=f"{FIGURES}/flux_predictor/shap_summary.png",
    params:
        target=TRAINING.get("label_column", "measured_flux"),
    threads: THREADS.get("training", 1)
    run:
        if SIMULATE:
            ensure_outputs(output)
        else:
            shell(
                "python workflow/scripts/explain_model.py "
                "--features {input.features} --target {params.target} "
                "--model {input.model} --output {output.plot}"
            )
