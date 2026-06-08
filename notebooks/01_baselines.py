"""marimo-App: Baseline-Anomaliedetection auf TEP (Slice 1).

Zeigt die durchgehende Pipeline: Daten laden → skalieren → unsupervise Detektoren mit
Defaults → Evaluation → Plots. Start interaktiv mit ``uv run marimo edit notebooks/01_baselines.py``
oder headless mit ``uv run python notebooks/01_baselines.py``.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from automl_ad.data import load_split
    from automl_ad.detectors import make_detector
    from automl_ad.eval import summarize
    from automl_ad.eval.plots import comparison_bars, per_fault_recall_heatmap, score_timeseries

    # Bewusst nur die klassischen Detektoren (der torch-Autoencoder kommt in Notebook 04).
    classical = ["ecod", "iforest", "ocsvm", "pca"]

    return (
        classical,
        comparison_bars,
        load_split,
        make_detector,
        mo,
        per_fault_recall_heatmap,
        score_timeseries,
        summarize,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Baselines: Unsupervised Anomaliedetection auf TEP

        Vier PyOD-Detektoren mit **Default-Hyperparametern**, trainiert nur auf Gutdaten.
        Dies ist der Vergleichsanker — die AutoML-Strategien (HPO, Ensembling, …) bauen darauf auf.
        """
    )
    return


@app.cell
def _(load_split):
    # Run-Level-Split (Scaler nur auf Gutdaten); Demo-Subsample für Tempo.
    split = load_split(n_train_good_runs=15, n_test_good_runs=15, n_test_fault_runs=15)
    split
    return (split,)


@app.cell
def _(classical, make_detector, split, summarize):
    results = {}
    fitted = {}
    for name in classical:
        det = make_detector(name).fit(split.X_train_good)
        scores = det.decision_function(split.X_test)
        fitted[name] = (det, scores)
        results[name] = summarize(
            split.y_test, scores, meta=split.meta_test, threshold=det.threshold_
        )
    results
    return fitted, results


@app.cell
def _(comparison_bars, results):
    comparison_bars(results, metric="roc_auc", save_as="01_baselines_roc_auc.png")
    return


@app.cell
def _(fitted, score_timeseries, split):
    # Beispiel-Lauf: leichter Fehler (IDV 1) mit Isolation Forest.
    det_if, scores_if = fitted["iforest"]
    score_timeseries(
        split.meta_test, scores_if, det_if.threshold_, fault=1,
        save_as="01_baselines_timeseries_fault1.png",
    )
    return


@app.cell
def _(fitted, per_fault_recall_heatmap, split):
    det_if2, scores_if2 = fitted["iforest"]
    per_fault_recall_heatmap(
        split.meta_test, scores_if2, det_if2.threshold_,
        save_as="01_baselines_recall_heatmap.png",
    )
    return


if __name__ == "__main__":
    app.run()
