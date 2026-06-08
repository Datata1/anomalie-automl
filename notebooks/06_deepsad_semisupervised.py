"""marimo-App: DeepSAD / Deep SVDD — semi-supervised AD (TICKET-03).

Bringt das **semi-supervised Setting** live: Deep SVDD lernt rein unüberwacht aus Gutdaten;
DeepSAD nutzt zusätzlich ein **kleines Label-Budget** bekannter Anomalien (aus
``faulty_training``). Erwartung: wenige Labels helfen v. a. bei schweren Fehlern (IDV 3/9/15).
Headless: ``uv run python notebooks/06_deepsad_semisupervised.py``.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import marimo as mo

    from automl_ad.data import load_semisupervised
    from automl_ad.detectors import available_detectors, make_detector
    from automl_ad.eval import summarize
    from automl_ad.eval.plots import comparison_bars

    return (
        available_detectors,
        comparison_bars,
        load_semisupervised,
        make_detector,
        mo,
        np,
        summarize,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # Semi-supervised: Deep SVDD vs. DeepSAD

        Deep SVDD = unüberwacht (0 Labels). DeepSAD = Gutdaten + wenige gelabelte Anomalien.
        Fehlerauswahl (Kühlwasser-/Feed-Fehler), bei der wenige Labels stark helfen.
        """
    )
    return


@app.cell
def _(load_semisupervised):
    ssd = load_semisupervised(
        faults=[4, 5, 11, 12], n_train_good_runs=20, n_labeled_fault_runs=8,
        n_test_good_runs=10, n_test_fault_runs=10,
    )
    {"data": repr(ssd), "labeled_anomalies": int(len(ssd.X_labeled))}
    return (ssd,)


@app.cell
def _(make_detector, ssd, summarize):
    # Deep SVDD — rein unüberwacht (nur Gutdaten).
    svdd = make_detector("deep_svdd", epochs=50).fit(ssd.X_train_good)
    auc_svdd = summarize(ssd.y_test, svdd.decision_function(ssd.X_test))["roc_auc"]
    auc_svdd
    return (auc_svdd,)


@app.cell
def _(available_detectors, make_detector, np, ssd, summarize):
    # DeepSAD — Gutdaten (0) + bekannte Anomalien (1). Fallback: Deep SVDD, falls deepod fehlt.
    if "deep_sad" in available_detectors():
        X_comb = np.vstack([ssd.X_train_good, ssd.X_labeled])
        y_comb = np.concatenate([np.zeros(len(ssd.X_train_good), int), ssd.y_labeled])
        sad = make_detector("deep_sad", epochs=50, device="cpu").fit(X_comb, y_comb)
        auc_sad = summarize(ssd.y_test, sad.decision_function(ssd.X_test))["roc_auc"]
        sad_available = True
    else:
        auc_sad, sad_available = float("nan"), False
    {"deep_sad_verfuegbar": sad_available, "auc_sad": auc_sad}
    return (auc_sad,)


@app.cell
def _(auc_sad, auc_svdd, comparison_bars):
    comparison_bars(
        {"Deep SVDD (0 Labels)": {"roc_auc": auc_svdd},
         "DeepSAD (wenige Labels)": {"roc_auc": auc_sad}},
        metric="roc_auc", save_as="06_deepsad_vs_deepsvdd.png",
    )
    return


if __name__ == "__main__":
    app.run()
