"""marimo-App: AutoML-Strategie 1 — HPO & Modellselektion (Slice 2).

Zwei Botschaften:
1. **HPO** (Optuna/TPE) tunt Detektoren gegen ein gelabeltes, test-disjunktes Validierungsset.
   Der AUC-Gewinn vs. Default ist auf TEP moderat (Bibliotheks-Defaults sind stark) — der
   eigentliche Wert zeigt sich in der **Trial-Verteilung**: viele Konfigurationen sind
   schlecht, HPO findet zuverlässig gute.
2. **Modellselektion** Oracle (label-basiert) vs. label-frei (Konsens) — der Kern des
   AutoML-für-AD-Problems.

Hinweis: bewusst kleine Subsamples (OCSVM ist O(n²)), damit die Demo schnell durchläuft.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import marimo as mo

    from automl_ad.automl.hpo import run_optuna
    from automl_ad.automl.selection import DEFAULT_CANDIDATES, select_internal, select_oracle
    from automl_ad.data import load_split, load_validation
    from automl_ad.detectors import make_detector
    from automl_ad.eval.plots import comparison_bars, grouped_bars, hpo_trial_distribution
    from sklearn.metrics import roc_auc_score

    return (
        DEFAULT_CANDIDATES,
        comparison_bars,
        grouped_bars,
        hpo_trial_distribution,
        load_split,
        load_validation,
        make_detector,
        mo,
        np,
        roc_auc_score,
        run_optuna,
        select_internal,
        select_oracle,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # AutoML-Strategie 1: HPO & Modellselektion

        Validierung aus der **Test-Verteilung mit disjunkten Läufen** (kein Leakage),
        zufällig auf ~4000 Punkte gekürzt für Tempo.
        """
    )
    return


@app.cell
def _(load_split, load_validation, np):
    split = load_split(n_train_good_runs=8, n_test_good_runs=8, n_test_fault_runs=8)

    X_val_full, y_val_full, _mv = load_validation(split, n_good_runs=5, n_fault_runs=5, source="testing")
    _rng = np.random.default_rng(0)
    _idx = _rng.choice(len(X_val_full), size=min(4000, len(X_val_full)), replace=False)
    X_val, y_val = X_val_full[_idx], y_val_full[_idx]

    X_hpo_train = split.X_train_good[:3000]  # OCSVM-Trainingsbasis cappen
    split, X_val.shape, float(y_val.mean())
    return X_hpo_train, X_val, split, y_val


@app.cell
def _(X_hpo_train, X_val, make_detector, roc_auc_score, run_optuna, split, y_val):
    # Default vs. getunt für drei Detektoren (gleiche Trainingsbasis, fair).
    hpo_results = {}
    studies = {}
    for name in ["iforest", "ocsvm", "pca"]:
        det_def = make_detector(name).fit(X_hpo_train)
        auc_def = roc_auc_score(split.y_test, det_def.decision_function(split.X_test))

        best, study = run_optuna(name, X_hpo_train, X_val, y_val, n_trials=15)
        det_tuned = make_detector(name, **best).fit(X_hpo_train)
        auc_tuned = roc_auc_score(split.y_test, det_tuned.decision_function(split.X_test))

        hpo_results[name] = {"default": auc_def, "tuned": auc_tuned}
        studies[name] = study
    hpo_results
    return hpo_results, studies


@app.cell
def _(grouped_bars, hpo_results):
    grouped_bars(
        hpo_results, ylabel="roc_auc", title="HPO: Default vs. getunt (Test-ROC-AUC)",
        save_as="02_hpo_default_vs_tuned.png",
    )
    return


@app.cell
def _(hpo_trial_distribution, studies):
    # Spread der OCSVM-Trials: viele Configs sind schlecht → HPO-Wert = Robustheit.
    ocsvm_vals = [t.value for t in studies["ocsvm"].trials if t.value is not None]
    hpo_trial_distribution(ocsvm_vals, save_as="02_hpo_ocsvm_trial_distribution.png")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Modellselektion: Oracle vs. label-frei

        Im echten unüberwachten Betrieb gibt es **keine Labels** zur Auswahl. Wir vergleichen
        die label-basierte Obergrenze (Oracle) mit einer label-freien Konsens-Heuristik.
        """
    )
    return


@app.cell
def _(
    DEFAULT_CANDIDATES,
    X_hpo_train,
    X_val,
    make_detector,
    roc_auc_score,
    select_internal,
    select_oracle,
    split,
    y_val,
):
    oracle_best, oracle_aucs = select_oracle(DEFAULT_CANDIDATES, X_hpo_train, X_val, y_val)
    internal_best, internal_scores = select_internal(DEFAULT_CANDIDATES, X_hpo_train, split.X_test)

    def _test_auc(name):
        d = make_detector(name).fit(X_hpo_train)
        return float(roc_auc_score(split.y_test, d.decision_function(split.X_test)))

    selection_summary = {
        "oracle": {"choice": oracle_best, "test_roc_auc": _test_auc(oracle_best)},
        "internal": {"choice": internal_best, "test_roc_auc": _test_auc(internal_best)},
    }
    selection_summary
    return (selection_summary,)


@app.cell
def _(comparison_bars, selection_summary):
    comparison_bars(
        {k: {"roc_auc": v["test_roc_auc"]} for k, v in selection_summary.items()},
        metric="roc_auc", save_as="02_selection_oracle_vs_internal.png",
    )
    return


if __name__ == "__main__":
    app.run()
