"""marimo-App: AutoML-Strategie 2 — fertige Frameworks & „Klassifikation vs. AD" (Slice 4).

Supervised Fehlerklassifikation mit **FLAML** (CASH, statt auto-sklearn) gegen eine
Random-Forest-Baseline. Anschließend der Kernvergleich aus der Vorlesungs-Übung:
überwachte Klassifikation vs. unüberwachte Anomaliedetection (binär, ROC-AUC).
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import marimo as mo

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, roc_auc_score

    from automl_ad.automl.frameworks import run_flaml_classification
    from automl_ad.data import load_supervised
    from automl_ad.detectors import make_detector
    from automl_ad.eval.plots import comparison_bars

    return (
        RandomForestClassifier,
        comparison_bars,
        f1_score,
        load_supervised,
        make_detector,
        mo,
        np,
        roc_auc_score,
        run_flaml_classification,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # AutoML-Strategie 2: Frameworks (FLAML) & Klassifikation vs. AD

        Mehrklassen-Fehlerklassifikation (0 = normal, 1..20 = Fehlertyp). FLAML übernimmt
        Modellwahl + HPO automatisch.
        """
    )
    return


@app.cell
def _(load_supervised):
    data = load_supervised(n_good_runs=12, n_fault_runs=12)
    {"X_train": data.X_train.shape, "X_test": data.X_test.shape,
     "klassen": sorted(set(data.y_train.tolist()))}
    return (data,)


@app.cell
def _(RandomForestClassifier, data, f1_score):
    # Baseline: Random Forest (handgesetzt).
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=0
    ).fit(data.X_train, data.y_train)
    rf_macro_f1 = f1_score(data.y_test, rf.predict(data.X_test), average="macro")
    rf_macro_f1
    return rf, rf_macro_f1


@app.cell
def _(data, f1_score, run_flaml_classification):
    # AutoML-Framework: FLAML mit Zeitbudget (CASH).
    automl = run_flaml_classification(data.X_train, data.y_train, time_budget=60)
    flaml_macro_f1 = f1_score(data.y_test, automl.predict(data.X_test), average="macro")
    {"best_estimator": automl.best_estimator, "macro_f1": flaml_macro_f1}
    return automl, flaml_macro_f1


@app.cell
def _(comparison_bars, flaml_macro_f1, rf_macro_f1):
    comparison_bars(
        {"RandomForest": {"macro_f1": rf_macro_f1}, "FLAML": {"macro_f1": flaml_macro_f1}},
        metric="macro_f1", save_as="03_classification_macro_f1.png",
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Klassifikation vs. Anomaliedetection (binär)

        Klassifikator-Anomaliescore = 1 − P(normal). Vergleich gegen einen unüberwachten
        Isolation Forest (nur auf Gutdaten trainiert) per ROC-AUC.

        **Kernaussage:** Mit vielen Labels schlägt die Klassifikation die AD klar — aber sie
        erkennt nur *bekannte* Fehler. Die AD braucht keine Fehlerlabels und generalisiert auf
        neue Fehler.
        """
    )
    return


@app.cell
def _(automl, data, make_detector, np, roc_auc_score):
    y_test_bin = (data.y_test != 0).astype(int)

    # Klassifikator als Anomaliedetektor: P(nicht normal).
    proba = automl.predict_proba(data.X_test)
    classes = list(automl.classes_)
    p_normal = proba[:, classes.index(0)] if 0 in classes else 0.0
    clf_score = 1.0 - p_normal
    auc_clf = roc_auc_score(y_test_bin, clf_score)

    # Unüberwachte AD: Isolation Forest, nur auf Gutdaten (Klasse 0) trainiert.
    X_good = data.X_train[data.y_train == 0]
    iforest = make_detector("iforest").fit(X_good)
    auc_ad = roc_auc_score(y_test_bin, iforest.decision_function(data.X_test))

    {"klassifikation_roc_auc": round(float(auc_clf), 3), "ad_roc_auc": round(float(auc_ad), 3)}
    return auc_ad, auc_clf


@app.cell
def _(auc_ad, auc_clf, comparison_bars):
    comparison_bars(
        {"Klassifikation (FLAML)": {"roc_auc": auc_clf}, "AD (Isolation Forest)": {"roc_auc": auc_ad}},
        metric="roc_auc", save_as="03_classification_vs_ad.png",
    )
    return


if __name__ == "__main__":
    app.run()
