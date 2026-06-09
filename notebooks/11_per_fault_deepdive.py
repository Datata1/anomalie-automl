"""marimo-App: Per-Fault-Deep-Dive (TICKET-13).

„Welche Methode fängt welchen Fehler?" — Recall je (Detektor × Fehlertyp) über alle
tabellarisch-unüberwachten Detektoren der Registry. Macht sichtbar, dass schwere Fehler
(IDV 3/9/15) für die meisten Methoden hart sind. Headless:
``uv run python notebooks/11_per_fault_deepdive.py``.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from automl_ad import config
    from automl_ad.data import load_split
    from automl_ad.detectors import available_detectors, make_detector
    from automl_ad.eval.plots import method_fault_heatmap

    # Nur tabellarisch-unüberwachte Detektoren (2D-fit). lstm_ae/deep_sad ausgeschlossen.
    tabular = ["ecod", "iforest", "ocsvm", "pca", "autoencoder", "som", "deep_svdd"]
    return available_detectors, config, load_split, make_detector, method_fault_heatmap, mo, tabular


@app.cell
def _(mo):
    mo.md(
        """
        # Per-Fault-Deep-Dive

        Recall = Anteil korrekt alarmierter Anomalie-Punkte (nach Onset) je Fehlertyp.
        """
    )
    return


@app.cell
def _(config, load_split):
    split = load_split(
        faults=config.DEMO_FAULTS, n_train_good_runs=10, n_test_good_runs=6, n_test_fault_runs=6
    )
    split
    return (split,)


@app.cell
def _(available_detectors, config, make_detector, split, tabular):
    meta = split.meta_test
    fault_ids = sorted(set(meta["faultNumber"].unique()) - {0})

    per_method_recall = {}
    for name in [n for n in tabular if n in available_detectors()]:
        det = make_detector(name).fit(split.X_train_good)
        pred = det.decision_function(split.X_test) > det.threshold_
        recall = {}
        for f in fault_ids:
            mask = ((meta["faultNumber"] == f) & (meta["sample"] > config.ONSET_TESTING)).to_numpy()
            if mask.sum() > 0:
                recall[int(f)] = float(pred[mask].mean())
        per_method_recall[name] = recall
    per_method_recall
    return (per_method_recall,)


@app.cell
def _(method_fault_heatmap, per_method_recall):
    method_fault_heatmap(per_method_recall, save_as="11_per_fault_heatmap.png")
    return


if __name__ == "__main__":
    app.run()
