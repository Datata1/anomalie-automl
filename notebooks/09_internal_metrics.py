"""marimo-App: Interne Validierungsmetriken (TICKET-06).

Zeigt, wie gut die label-freie interne Metrik **SIREOS** mit der echten ROC-AUC korreliert,
und vergleicht die intern gewählte Konfiguration mit Konsens (selection.select_internal) und
Oracle. Headless: ``uv run python notebooks/09_internal_metrics.py``.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from automl_ad.automl.internal_metrics import select_by_internal, sireos
    from automl_ad.automl.selection import DEFAULT_CANDIDATES, select_internal, select_oracle
    from automl_ad.data import load_split, load_validation
    from automl_ad.detectors import make_detector
    from automl_ad.eval import summarize
    from automl_ad.eval.plots import comparison_bars

    return (
        DEFAULT_CANDIDATES,
        comparison_bars,
        load_split,
        load_validation,
        make_detector,
        mo,
        np,
        select_by_internal,
        select_internal,
        select_oracle,
        sireos,
        summarize,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Interne Metrik (SIREOS) vs. echte Performance

        SIREOS ist **label-frei** (niedriger = besser). Wir prüfen die Korrelation zur echten
        ROC-AUC und vergleichen die Selektion intern vs. Konsens vs. Oracle.
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
def _(DEFAULT_CANDIDATES, make_detector, sireos, split, summarize):
    # Pro Detektor: echte Test-ROC-AUC vs. label-freier SIREOS.
    per_detector = {}
    for name, hp in DEFAULT_CANDIDATES:
        det = make_detector(name, **hp).fit(split.X_train_good)
        s_test = det.decision_function(split.X_test)
        per_detector[name] = {
            "roc_auc": summarize(split.y_test, s_test)["roc_auc"],
            "sireos": sireos(split.X_test, s_test),
        }
    per_detector
    return (per_detector,)


@app.cell
def _(np, per_detector):
    # Korrelation: niedriger SIREOS sollte mit höherer ROC-AUC einhergehen (neg. Korrelation).
    aucs = np.array([v["roc_auc"] for v in per_detector.values()])
    sir = np.array([v["sireos"] for v in per_detector.values()])
    corr = float(np.corrcoef(aucs, sir)[0, 1])
    {"pearson_corr(roc_auc, sireos)": round(corr, 3),
     "interpretation": "negativ = interne Metrik nützlich"}
    return


@app.cell
def _(
    DEFAULT_CANDIDATES,
    X_val,
    comparison_bars,
    make_detector,
    select_by_internal,
    select_internal,
    select_oracle,
    split,
    summarize,
    y_val,
):
    internal_best, _ = select_by_internal(DEFAULT_CANDIDATES, split.X_train_good, split.X_test)
    consensus_best, _ = select_internal(DEFAULT_CANDIDATES, split.X_train_good, split.X_test)
    oracle_best, _ = select_oracle(DEFAULT_CANDIDATES, split.X_train_good, X_val, y_val)

    def _auc(name):
        d = make_detector(name).fit(split.X_train_good)
        return float(summarize(split.y_test, d.decision_function(split.X_test))["roc_auc"])

    comparison_bars(
        {
            f"SIREOS ({internal_best})": {"roc_auc": _auc(internal_best)},
            f"Konsens ({consensus_best})": {"roc_auc": _auc(consensus_best)},
            f"Oracle ({oracle_best})": {"roc_auc": _auc(oracle_best)},
        },
        metric="roc_auc", save_as="09_internal_vs_consensus_vs_oracle.png",
    )
    return


if __name__ == "__main__":
    app.run()
