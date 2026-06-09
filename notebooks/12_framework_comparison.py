"""marimo-App: AutoML-Framework-Vergleich — RF vs FLAML vs AutoGluon (Review-Ergänzung).

Vergleicht drei supervise Ansätze auf der TEP-Fehlerklassifikation: handgesetzter Random
Forest, FLAML (CASH) und AutoGluon (tiefes Stacking). Metriken: Macro-F1 (Mehrklassen) und
binäre AD-ROC-AUC (1 − P(normal)).

**AutoGluon** ist unter Python 3.13 nicht installierbar (Konflikt mit numpy 2.x / pandas 3.x);
der Benchmark läuft daher in einem separaten py3.12-venv und wird hier über
``reports/autogluon_result.json`` eingelesen (siehe scripts/_autogluon_bench.py). Fehlt die
Datei, zeigt das Notebook nur RF vs FLAML.

Headless: ``uv run python notebooks/12_framework_comparison.py``.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, roc_auc_score

    from automl_ad.automl.frameworks import run_flaml_classification
    from automl_ad.config import REPORTS_DIR
    from automl_ad.data import load_supervised
    from automl_ad.eval.plots import grouped_bars

    return (
        Path,
        RandomForestClassifier,
        REPORTS_DIR,
        f1_score,
        grouped_bars,
        json,
        load_supervised,
        mo,
        np,
        roc_auc_score,
        run_flaml_classification,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Framework-Vergleich: RF vs FLAML vs AutoGluon

        Macro-F1 (Mehrklassen) und binäre AD-ROC-AUC. AutoGluon aus isoliertem py3.12-venv.
        """
    )
    return


@app.cell
def _(load_supervised):
    data = load_supervised(n_good_runs=10, n_fault_runs=10)  # identisch zum AutoGluon-Dump
    {"X_train": data.X_train.shape, "X_test": data.X_test.shape}
    return (data,)


@app.cell
def _(f1_score, np, roc_auc_score):
    def clf_metrics(clf, X_test, y_test):
        """(macro_f1, binäre AD-ROC-AUC) für einen Klassifikator mit predict/predict_proba."""
        y_pred = clf.predict(X_test)
        macro = float(f1_score(y_test, y_pred, average="macro"))
        proba = clf.predict_proba(X_test)
        classes = list(clf.classes_)
        p_normal = proba[:, classes.index(0)] if 0 in classes else np.zeros(len(X_test))
        ad_auc = float(roc_auc_score((y_test != 0).astype(int), 1.0 - p_normal))
        return macro, ad_auc

    return (clf_metrics,)


@app.cell
def _(RandomForestClassifier, clf_metrics, data):
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=0
    ).fit(data.X_train, data.y_train)
    rf_f1, rf_auc = clf_metrics(rf, data.X_test, data.y_test)
    {"RandomForest": {"macro_f1": rf_f1, "ad_roc_auc": rf_auc}}
    return rf_auc, rf_f1


@app.cell
def _(clf_metrics, data, run_flaml_classification):
    automl = run_flaml_classification(data.X_train, data.y_train, time_budget=60)
    flaml_f1, flaml_auc = clf_metrics(automl, data.X_test, data.y_test)
    {"FLAML": {"macro_f1": flaml_f1, "ad_roc_auc": flaml_auc}}
    return flaml_auc, flaml_f1


@app.cell
def _(Path, REPORTS_DIR, json):
    # AutoGluon-Ergebnis aus dem isolierten Benchmark (falls vorhanden).
    ag_path = Path(REPORTS_DIR) / "autogluon_result.json"
    ag = json.loads(ag_path.read_text()) if ag_path.exists() else None
    ag
    return (ag,)


@app.cell
def _(ag, flaml_auc, flaml_f1, grouped_bars, rf_auc, rf_f1):
    results = {
        "RandomForest": {"macro_f1": rf_f1, "ad_roc_auc": rf_auc},
        "FLAML": {"macro_f1": flaml_f1, "ad_roc_auc": flaml_auc},
    }
    if ag is not None:
        results["AutoGluon"] = {"macro_f1": ag["macro_f1"], "ad_roc_auc": ag["ad_roc_auc"]}

    grouped_bars(
        results, ylabel="Score", title="Framework-Vergleich (Macro-F1 & AD-ROC-AUC)",
        save_as="12_framework_comparison.png",
    )
    return


if __name__ == "__main__":
    app.run()
