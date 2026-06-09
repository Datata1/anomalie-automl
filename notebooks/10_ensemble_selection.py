"""marimo-App: Ensemble Selection & Feature Bagging (TICKET-07).

Vergleicht vier Aggregationen: bester Einzeldetektor, einfaches Mittelwert-Ensemble,
**Feature Bagging** (label-frei) und **greedy Ensemble Selection** (label-basiert/Oracle).
Headless: ``uv run python notebooks/10_ensemble_selection.py``.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from automl_ad.automl.ensemble_selection import (
        build_score_matrices,
        feature_bagging_scores,
        greedy_ensemble_selection,
    )
    from automl_ad.automl.selection import DEFAULT_CANDIDATES
    from automl_ad.data import load_split, load_validation
    from automl_ad.detectors import make_detector
    from automl_ad.eval import summarize
    from automl_ad.eval.plots import comparison_bars

    return (
        DEFAULT_CANDIDATES,
        build_score_matrices,
        comparison_bars,
        feature_bagging_scores,
        greedy_ensemble_selection,
        load_split,
        load_validation,
        make_detector,
        mo,
        np,
        summarize,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Ensemble Selection & Feature Bagging

        Greedy Selection nutzt ein gelabeltes Val-Set (Oracle); Feature Bagging und Mittelwert
        sind label-frei. Vergleich gegen den besten Einzeldetektor.
        """
    )
    return


@app.cell
def _(load_split, load_validation, np):
    split = load_split(n_train_good_runs=12, n_test_good_runs=12, n_test_fault_runs=12)
    Xv, yv, _ = load_validation(split, n_good_runs=6, n_fault_runs=6, source="testing")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xv), size=min(4000, len(Xv)), replace=False)
    X_val, y_val = Xv[idx], yv[idx]
    return X_val, split, y_val


@app.cell
def _(DEFAULT_CANDIDATES, build_score_matrices, X_val, split):
    val_norm, test_norm = build_score_matrices(
        DEFAULT_CANDIDATES, split.X_train_good, X_val, split.X_test
    )
    val_norm.shape, test_norm.shape
    return test_norm, val_norm


@app.cell
def _(
    DEFAULT_CANDIDATES,
    feature_bagging_scores,
    greedy_ensemble_selection,
    make_detector,
    np,
    split,
    summarize,
    test_norm,
    val_norm,
    y_val,
):
    def _auc(scores):
        return float(summarize(split.y_test, scores)["roc_auc"])

    # Bester Einzeldetektor (auf Testdaten, threshold-frei).
    best_individual = max(_auc(test_norm[:, i]) for i in range(test_norm.shape[1]))

    # Mittelwert-Ensemble (label-frei).
    avg_auc = _auc(test_norm.mean(axis=1))

    # Feature Bagging auf Isolation Forest (label-frei).
    fb = feature_bagging_scores(
        lambda: make_detector("iforest"), split.X_train_good, split.X_test,
        n_estimators=10, max_features=0.6,
    )
    fb_auc = _auc(fb)

    # Greedy Ensemble Selection (label-basiert, Oracle): Gewichte auf Val, anwenden auf Test.
    weights = greedy_ensemble_selection(val_norm, y_val, n_rounds=20)
    greedy_auc = _auc(test_norm @ weights)

    results = {
        "bester Einzeldetektor": {"roc_auc": best_individual},
        "Mittelwert (label-frei)": {"roc_auc": avg_auc},
        "Feature Bagging (label-frei)": {"roc_auc": fb_auc},
        "Greedy Selection (Oracle)": {"roc_auc": greedy_auc},
    }
    results
    return (results,)


@app.cell
def _(comparison_bars, results):
    comparison_bars(results, metric="roc_auc", save_as="10_ensemble_selection.png")
    return


if __name__ == "__main__":
    app.run()
