"""Plots für die Präsentation (Matplotlib). Speichern nach ``reports/``.

Alle Funktionen geben die Matplotlib-Figure zurück (für marimo-Anzeige) und speichern
optional zusätzlich auf Platte.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config


def _save(fig, save_as: str | None):
    if save_as:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.REPORTS_DIR / save_as if not Path(save_as).is_absolute() else Path(save_as)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    return fig


def score_timeseries(
    meta: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    fault: int,
    run: int | None = None,
    onset: int = config.ONSET_TESTING,
    save_as: str | None = None,
):
    """Anomaly-Score über die Zeit für einen Beispiel-Lauf, mit Onset + Threshold."""
    df = meta.copy()
    df["score"] = scores
    sub = df[df["faultNumber"] == fault]
    if run is None:
        run = int(sub["simulationRun"].iloc[0])
    sub = sub[sub["simulationRun"] == run].sort_values("sample")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(sub["sample"], sub["score"], lw=1.0, label="Anomaly-Score")
    ax.axhline(threshold, color="tab:red", ls="--", lw=1, label="Threshold")
    ax.axvline(onset, color="tab:green", ls=":", lw=1.5, label=f"Fehler-Onset (>{onset})")
    ax.set(xlabel="sample", ylabel="Score", title=f"Fehler {fault}, Lauf {run}")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, save_as)


def per_fault_recall_heatmap(
    meta: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    onset: int = config.ONSET_TESTING,
    save_as: str | None = None,
):
    """Recall je Fehlertyp (Anteil korrekt alarmierter Anomalie-Punkte nach Onset)."""
    df = meta.copy()
    df["score"] = scores
    post = df[(df["faultNumber"] != 0) & (df["sample"] > onset)].copy()
    post["alarm"] = (post["score"] > threshold).astype(int)
    recall = post.groupby("faultNumber")["alarm"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(7, 1.6))
    ax.imshow(recall.to_numpy()[None, :], cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(recall)))
    ax.set_xticklabels(recall.index, fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Fehlertyp (IDV)")
    ax.set_title("Recall je Fehlertyp")
    for i, v in enumerate(recall.to_numpy()):
        ax.text(i, 0, f"{v:.2f}", ha="center", va="center", fontsize=7)
    return _save(fig, save_as)


def comparison_bars(
    results: dict[str, dict],
    metric: str = "roc_auc",
    save_as: str | None = None,
):
    """Balkenvergleich einer Metrik über mehrere (Methode/Strategie)-Ergebnisse."""
    names = list(results)
    values = [results[n].get(metric, float("nan")) for n in names]

    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(names)), 3.5))
    bars = ax.bar(names, values, color="tab:blue")
    ax.set(ylabel=metric, title=f"Vergleich: {metric}")
    ax.set_ylim(0, 1 if metric in {"roc_auc", "pr_auc", "f1"} else None)
    ax.tick_params(axis="x", rotation=30)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    return _save(fig, save_as)


def grouped_bars(
    results_by_group: dict[str, dict[str, float]],
    ylabel: str = "roc_auc",
    title: str = "",
    save_as: str | None = None,
):
    """Gruppierte Balken: {Gruppe: {Serie: Wert}} — z. B. Detektor × {default, tuned}."""
    groups = list(results_by_group)
    series = list(next(iter(results_by_group.values())))
    n_series = len(series)
    width = 0.8 / n_series

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(groups)), 3.8))
    for i, s in enumerate(series):
        vals = [results_by_group[g].get(s, float("nan")) for g in groups]
        xs = [j + i * width for j in range(len(groups))]
        bars = ax.bar(xs, vals, width=width, label=s)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks([j + width * (n_series - 1) / 2 for j in range(len(groups))])
    ax.set_xticklabels(groups)
    ax.set(ylabel=ylabel, title=title)
    ax.set_ylim(0, 1 if ylabel in {"roc_auc", "pr_auc", "f1"} else None)
    ax.legend(fontsize=8)
    return _save(fig, save_as)


def hpo_trial_distribution(values: list[float], ylabel: str = "Val-ROC-AUC", save_as: str | None = None):
    """Verteilung der HPO-Trial-Scores + Laufender Bestwert (zeigt: viele Configs sind schlecht)."""
    import numpy as np

    vals = np.asarray(values, dtype=float)
    running_best = np.maximum.accumulate(vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    ax1.hist(vals, bins=15, color="tab:gray")
    ax1.axvline(vals.max(), color="tab:green", ls="--", label=f"best={vals.max():.3f}")
    ax1.axvline(vals.min(), color="tab:red", ls="--", label=f"worst={vals.min():.3f}")
    ax1.set(xlabel=ylabel, ylabel="Anzahl Trials", title="Verteilung der Trials")
    ax1.legend(fontsize=8)
    ax2.plot(range(1, len(vals) + 1), running_best, marker=".", color="tab:blue")
    ax2.set(xlabel="Trial", ylabel=f"bester {ylabel}", title="Optimierungsverlauf")
    fig.tight_layout()
    return _save(fig, save_as)
