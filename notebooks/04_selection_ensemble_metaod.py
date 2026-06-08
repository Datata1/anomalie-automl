"""marimo-App: AutoML-Strategien 3 & 4 — Ensembling + Meta-Learning (Slice 5).

- **Ensembling** (Strategie 4): normalisierte Score-Aggregation über mehrere Detektoren
  (inkl. torch-Autoencoder), label-frei. Robust ohne perfektes Tuning.
- **Meta-Learning-Modellselektion** (Strategie 3): MetaOD-Empfehlung, mit Fallback auf eine
  Konsens-Heuristik (derselbe label-freie Mechanismus), falls MetaOD nicht installierbar ist.

Beide adressieren das Kernproblem: Modellwahl **ohne Labels**.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from automl_ad.automl.ensemble import ensemble_scores
    from automl_ad.automl.metaod import metaod_available, recommend
    from automl_ad.data import load_split
    from automl_ad.detectors import make_detector
    from automl_ad.eval import summarize
    from automl_ad.eval.plots import comparison_bars

    # Kandidaten inkl. torch-Autoencoder (kleines Budget für Tempo).
    candidates = [
        ("ecod", {}),
        ("iforest", {}),
        ("ocsvm", {}),
        ("pca", {}),
        ("autoencoder", {"epoch_num": 20, "hidden_neuron_list": [32, 16], "batch_size": 256, "verbose": 0}),
    ]
    return (
        candidates,
        comparison_bars,
        ensemble_scores,
        load_split,
        make_detector,
        metaod_available,
        mo,
        recommend,
        summarize,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Strategien 3 & 4: Ensembling + Meta-Learning

        Alle Detektoren teilen dasselbe PyOD-Interface — der torch-Autoencoder ist nur ein
        weiterer Kandidat.
        """
    )
    return


@app.cell
def _(load_split):
    split = load_split(n_train_good_runs=10, n_test_good_runs=8, n_test_fault_runs=6)
    split
    return (split,)


@app.cell
def _(candidates, make_detector, split, summarize):
    # Einzeldetektoren (inkl. AE).
    individual = {}
    for name, hp in candidates:
        det = make_detector(name, **hp).fit(split.X_train_good)
        individual[name] = summarize(split.y_test, det.decision_function(split.X_test))["roc_auc"]
    individual
    return (individual,)


@app.cell
def _(candidates, ensemble_scores, individual, split, summarize):
    # Ensembles (label-frei).
    results = {k: {"roc_auc": v} for k, v in individual.items()}
    for method in ["average", "maximization"]:
        es = ensemble_scores(candidates, split.X_train_good, split.X_test, method=method)
        results[f"ensemble[{method}]"] = {"roc_auc": summarize(split.y_test, es)["roc_auc"]}
    results
    return (results,)


@app.cell
def _(comparison_bars, results):
    comparison_bars(results, metric="roc_auc", save_as="04_ensemble_vs_individual.png")
    return


@app.cell
def _(candidates, metaod_available, recommend, split):
    # Meta-Learning-Modellselektion (label-frei). MetaOD oder Konsens-Fallback.
    rec = recommend(split.X_train_good, split.X_test, candidates=candidates, n_selection=1)
    {"metaod_available": metaod_available(), "source": rec["source"], "empfehlung": rec["choice"]}
    return


if __name__ == "__main__":
    app.run()
