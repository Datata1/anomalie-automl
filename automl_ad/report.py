"""Konsolidierter Abschluss-Report & Asset-Generator (TICKET-11).

Ein Aufruf erzeugt das komplette PowerPoint-Material nach ``reports/``:
- ``results.csv`` — maschinenlesbar (Methode × Strategie × Metriken),
- ``summary_table.md`` — Markdown-Tabelle,
- ``summary_defaults_roc_auc.png`` — Default-Detektoren im Vergleich,
- ``summary_label_free_vs_oracle.png`` — die zentrale Projektaussage.

Die Default-Detektoren werden **dynamisch aus der Registry** gezogen (neue tabellarische
Detektoren erscheinen automatisch). Reproduzierbar über ``config.RANDOM_SEED``.

Ausführen: ``uv run python -m automl_ad.report``
"""

from __future__ import annotations

import csv

import numpy as np

from . import config
from .automl.ensemble import ensemble_scores
from .automl.hpo import run_optuna
from .automl.selection import DEFAULT_CANDIDATES, select_internal, select_oracle
from .data import load_split, load_validation
from .detectors import available_detectors, make_detector
from .eval import summarize
from .eval.plots import comparison_bars

# Tabellarische, unüberwachte Detektoren (2D-fit(X)-Kontrakt). lstm_ae (3D) und deep_sad
# (Label-Budget) sind hier bewusst ausgeschlossen — siehe plans/phase-c/00_README.md.
TABULAR_UNSUPERVISED = ["ecod", "iforest", "ocsvm", "pca", "autoencoder", "som", "deep_svdd"]

_METRIC_COLS = ["roc_auc", "pr_auc", "f1", "detection_delay", "false_alarm_rate"]


def _val_subsample(X_val, y_val, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_val), size=min(n, len(X_val)), replace=False)
    return X_val[idx], y_val[idx]


def build(n_train_good_runs=10, n_test_good_runs=10, n_test_fault_runs=10, hpo_detector="iforest"):
    """Berechnet alle Strategien auf einem gemeinsamen Split; gibt ``(rows, choices)`` zurück."""
    split = load_split(
        n_train_good_runs=n_train_good_runs,
        n_test_good_runs=n_test_good_runs,
        n_test_fault_runs=n_test_fault_runs,
    )
    X_val_full, y_val_full, _ = load_validation(split, n_good_runs=5, n_fault_runs=5)
    X_val, y_val = _val_subsample(X_val_full, y_val_full)

    rows: dict[str, dict] = {}

    # 1) Default-Detektoren — dynamisch aus der Registry (tabellarisch, unüberwacht).
    available = set(available_detectors())
    detectors = [n for n in TABULAR_UNSUPERVISED if n in available]
    best_default_name, best_default = None, -1.0
    for name in detectors:
        det = make_detector(name).fit(split.X_train_good)
        m = summarize(split.y_test, det.decision_function(split.X_test),
                      meta=split.meta_test, threshold=det.threshold_)
        rows[f"default:{name}"] = m
        if m["roc_auc"] > best_default:
            best_default, best_default_name = m["roc_auc"], name

    # 2) HPO (Strategie 1) auf einem Detektor.
    best_hp, _ = run_optuna(hpo_detector, split.X_train_good[:3000], X_val, y_val, n_trials=15)
    det = make_detector(hpo_detector, **best_hp).fit(split.X_train_good)
    rows[f"hpo:{hpo_detector}"] = summarize(
        split.y_test, det.decision_function(split.X_test),
        meta=split.meta_test, threshold=det.threshold_,
    )

    # 3) Ensemble (Strategie 4, label-frei) — contamination-basierter Threshold.
    es, es_thr = ensemble_scores(
        DEFAULT_CANDIDATES, split.X_train_good, split.X_test, method="average", return_threshold=True
    )
    rows["ensemble:average"] = summarize(split.y_test, es, meta=split.meta_test, threshold=es_thr)

    # 4) Modellselektion: label-frei (Konsens) vs. Oracle.
    internal_best, _ = select_internal(DEFAULT_CANDIDATES, split.X_train_good, split.X_test)
    oracle_best, _ = select_oracle(DEFAULT_CANDIDATES, split.X_train_good, X_val, y_val)
    for tag, choice in [("select_internal", internal_best), ("select_oracle", oracle_best)]:
        det = make_detector(choice).fit(split.X_train_good)
        rows[f"{tag}:{choice}"] = summarize(
            split.y_test, det.decision_function(split.X_test),
            meta=split.meta_test, threshold=det.threshold_,
        )

    choices = {
        "best_default": best_default_name,
        "internal": internal_best,
        "oracle": oracle_best,
        "default_detectors": detectors,
    }
    return rows, choices


def _write_markdown_table(rows: dict[str, dict], path):
    lines = ["| Strategie | " + " | ".join(_METRIC_COLS) + " |",
             "|" + "---|" * (len(_METRIC_COLS) + 1)]
    for name, m in rows.items():
        vals = [f"{m[c]:.3f}" if (c in m and m[c] == m[c]) else "—" for c in _METRIC_COLS]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_results_csv(rows: dict[str, dict], path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "method", *_METRIC_COLS])
        for key, m in rows.items():
            strategy, _, method = key.partition(":")
            writer.writerow([strategy, method, *[m.get(c, "") for c in _METRIC_COLS]])


def main():
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, choices = build()

    _write_markdown_table(rows, config.REPORTS_DIR / "summary_table.md")
    _write_results_csv(rows, config.REPORTS_DIR / "results.csv")

    # Default-Detektoren im Vergleich.
    defaults = {n: rows[f"default:{n}"] for n in choices["default_detectors"]}
    comparison_bars(defaults, metric="roc_auc", save_as="summary_defaults_roc_auc.png")

    # Zentrale Grafik: label-frei (Konsens, Ensemble) vs. Oracle.
    highlight = {
        "Ensemble (label-frei)": rows["ensemble:average"],
        f"Konsens-Select ({choices['internal']})": rows[f"select_internal:{choices['internal']}"],
        f"Oracle-Select ({choices['oracle']})": rows[f"select_oracle:{choices['oracle']}"],
    }
    comparison_bars(highlight, metric="roc_auc", save_as="summary_label_free_vs_oracle.png")

    print("Auswahl:", {k: v for k, v in choices.items() if k != "default_detectors"})
    print("Artefakte -> reports/: results.csv, summary_table.md, summary_*.png")
    for name, m in rows.items():
        print(f"  {name:28s} ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} F1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
