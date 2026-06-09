"""marimo-App: Explorative Datenanalyse (TICKET-12, Port von eda_dataset.ipynb).

Reaktive EDA über die ``automl_ad``-Bausteine: Korrelations-Heatmap, Sensor-Zeitreihe mit
Fehler-Onset und PCA-2D der Fehlertypen. Das Jupyter-Notebook ``eda_dataset.ipynb`` bleibt
erhalten. Headless: ``uv run python notebooks/00_eda.py``.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from automl_ad import config
    from automl_ad.data import _read_runs

    return PCA, StandardScaler, _read_runs, config, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(
        """
        # TEP — Explorative Datenanalyse

        52 Features (41 `xmeas` + 11 `xmv`), 20 Fehlertypen. In `faulty_testing` setzt der
        Fehler erst ab Sample 161 ein. Hier auf wenigen Läufen exploriert (kein Vollscan).
        """
    )
    return


@app.cell
def _(_read_runs, config):
    # Wenige Läufe laden (Gutdaten + ausgewählte Fehler) für Tempo.
    df_good = _read_runs(config.FAULT_FREE_TESTING, runs=[1, 2, 3])
    df_fault = _read_runs(config.FAULTY_TESTING, runs=[1, 2, 3], faults=[1, 4, 11, 13])
    {"good_rows": len(df_good), "fault_rows": len(df_fault)}
    return df_fault, df_good


@app.cell
def _(config, df_good, plt):
    # Korrelations-Heatmap der Features (auf Gutdaten).
    corr = df_good[config.FEATURE_COLS].corr().to_numpy()
    _fig, _ax = plt.subplots(figsize=(7, 6))
    _im = _ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    _ax.set_title("Feature-Korrelationen (Gutdaten)")
    _ax.set_xticks([]); _ax.set_yticks([])
    _fig.colorbar(_im, ax=_ax, fraction=0.046)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _fig.savefig(config.REPORTS_DIR / "00_eda_correlation.png", dpi=120, bbox_inches="tight")
    _fig
    return


@app.cell
def _(config, df_fault, plt):
    # Zeitreihe Reaktortemperatur (xmeas_9) eines Fehlerlaufs mit Onset-Markierung.
    run = df_fault[(df_fault["faultNumber"] == 13) & (df_fault["simulationRun"] == 1)].sort_values("sample")
    _fig, _ax = plt.subplots(figsize=(9, 3.2))
    _ax.plot(run["sample"], run["xmeas_9"], lw=1.0)
    _ax.axvline(config.ONSET_TESTING, color="tab:green", ls=":", label=f"Onset (>{config.ONSET_TESTING})")
    _ax.set(xlabel="sample", ylabel="xmeas_9 (Reaktortemp.)", title="Fehler 13 — Sensorverlauf")
    _ax.legend(fontsize=8)
    _fig.savefig(config.REPORTS_DIR / "00_eda_timeseries.png", dpi=120, bbox_inches="tight")
    _fig
    return


@app.cell
def _(PCA, StandardScaler, config, df_fault, df_good, np, pd, plt):
    # PCA-2D: Gutdaten + Fehlertypen (nur Post-Onset-Anomalien der Fehler).
    post = df_fault[df_fault["sample"] > config.ONSET_TESTING]
    data = pd.concat([df_good, post], ignore_index=True)
    X = StandardScaler().fit_transform(data[config.FEATURE_COLS].to_numpy())
    pcs = PCA(n_components=2, random_state=0).fit_transform(X)

    _fig, _ax = plt.subplots(figsize=(6, 5))
    for f in sorted(data["faultNumber"].unique()):
        mask = (data["faultNumber"] == f).to_numpy()
        _ax.scatter(pcs[mask, 0], pcs[mask, 1], s=4, alpha=0.4, label=f"Fehler {f}" if f else "normal")
    _ax.set(xlabel="PC1", ylabel="PC2", title="PCA-2D der Fehlertypen")
    _ax.legend(fontsize=7, markerscale=2)
    _fig.savefig(config.REPORTS_DIR / "00_eda_pca.png", dpi=120, bbox_inches="tight")
    _fig
    return


if __name__ == "__main__":
    app.run()
