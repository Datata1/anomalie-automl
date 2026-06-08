"""marimo-App: Self-Organizing Map (TICKET-04).

Unüberwachte AD mit einer SOM (topologie-erhaltend). Score = Distanz zu den k nächsten
Neuronen. Zusätzlich eine 2D-Distanzkarte (U-Matrix) des Normalzustands — anschaulich für die
Präsentation. Headless: ``uv run python notebooks/07_som.py``.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    from automl_ad import config
    from automl_ad.data import load_split
    from automl_ad.detectors import make_detector
    from automl_ad.eval import summarize
    from automl_ad.eval.plots import per_fault_recall_heatmap

    return config, load_split, make_detector, mo, per_fault_recall_heatmap, plt, summarize


@app.cell
def _(mo):
    mo.md(
        """
        # Self-Organizing Map (SOM)

        Training auf Gutdaten; Score = mittlere Distanz zu den k nächsten Neuronen. Die
        U-Matrix visualisiert die gelernte Topologie des Normalzustands.
        """
    )
    return


@app.cell
def _(load_split):
    split = load_split(n_train_good_runs=12, n_test_good_runs=12, n_test_fault_runs=12)
    split
    return (split,)


@app.cell
def _(make_detector, split, summarize):
    som = make_detector("som", map_size=15, num_iteration=8000).fit(split.X_train_good)
    scores = som.decision_function(split.X_test)
    metrics = summarize(split.y_test, scores, meta=split.meta_test, threshold=som.threshold_)
    metrics
    return scores, som


@app.cell
def _(plt, som):
    # U-Matrix: mittlere Distanz benachbarter Neuronen (helle Bereiche = Cluster-Grenzen).
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(som.som_.distance_map().T, cmap="bone_r")
    ax.set_title("SOM U-Matrix (Normalzustand)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    from automl_ad import config as _cfg
    _cfg.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_cfg.REPORTS_DIR / "07_som_umatrix.png", dpi=120, bbox_inches="tight")
    fig
    return


@app.cell
def _(per_fault_recall_heatmap, scores, som, split):
    per_fault_recall_heatmap(split.meta_test, scores, som.threshold_, save_as="07_som_recall_heatmap.png")
    return


if __name__ == "__main__":
    app.run()
