"""marimo-App: LSTM-Autoencoder — zeitliche Anomaliedetection (TICKET-02).

Erfasst die **Dynamik** des Prozesses über Sequenzfenster (statt punktweise). Besonders stark
bei driftenden/zeitlichen Fehlern (z. B. IDV 13 = Slow Drift). Headless:
``uv run python notebooks/05_lstm_autoencoder.py``.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from automl_ad.data import load_windowed
    from automl_ad.detectors import make_detector
    from automl_ad.eval import summarize
    from automl_ad.eval.plots import per_fault_recall_heatmap, score_timeseries

    return load_windowed, make_detector, mo, per_fault_recall_heatmap, score_timeseries, summarize


@app.cell
def _(mo):
    mo.md(
        """
        # LSTM-Autoencoder (zeitlich)

        Training nur auf Gutdaten-**Sequenzen**; Score = Rekonstruktionsfehler eines Fensters.
        Fenster respektieren Lauf-Grenzen (kein lauf-übergreifendes Lecken). Fehlerauswahl
        enthält den Drift-Fehler **IDV 13**, den punktweise Detektoren schwer fangen.
        """
    )
    return


@app.cell
def _(load_windowed):
    wd = load_windowed(
        faults=[1, 4, 13, 3], n_train_good_runs=12,
        n_test_good_runs=10, n_test_fault_runs=10, window=30, stride=5,
    )
    wd
    return (wd,)


@app.cell
def _(make_detector, summarize, wd):
    lstm_ae = make_detector("lstm_ae", epoch_num=25, hidden=64, latent_dim=16, batch_size=256).fit(
        wd.X_train_seq
    )
    scores = lstm_ae.decision_function(wd.X_test_seq)
    metrics = summarize(wd.y_test_seq, scores, meta=wd.meta_test_seq, threshold=lstm_ae.threshold_)
    metrics
    return lstm_ae, scores


@app.cell
def _(lstm_ae, score_timeseries, scores, wd):
    # Drift-Fehler IDV 13: Score steigt nach dem Onset (Sample 160).
    score_timeseries(
        wd.meta_test_seq, scores, lstm_ae.threshold_, fault=13,
        save_as="05_lstm_ae_timeseries_fault13.png",
    )
    return


@app.cell
def _(lstm_ae, per_fault_recall_heatmap, scores, wd):
    per_fault_recall_heatmap(
        wd.meta_test_seq, scores, lstm_ae.threshold_, save_as="05_lstm_ae_recall_heatmap.png"
    )
    return


if __name__ == "__main__":
    app.run()
