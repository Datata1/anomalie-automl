"""Konsolidierter Abschluss-Report: AutoML-Strategien im Vergleich.

Erzeugt eine Gesamt-Vergleichstabelle und die zentrale „label-frei vs. Oracle"-Grafik nach
``reports/``. Bündelt die Botschaft des Projekts in einem reproduzierbaren Artefakt.

Ausführen: ``uv run python -m automl_ad.report``
"""

from __future__ import annotations

import numpy as np

from . import config
from .automl.ensemble import ensemble_scores
from .automl.hpo import run_optuna
from .automl.selection import DEFAULT_CANDIDATES, select_internal, select_oracle
from .data import load_split, load_validation
from .detectors import make_detector
from .eval import summarize
from .eval.plots import comparison_bars


def _val_subsample(X_val, y_val, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_val), size=min(n, len(X_val)), replace=False)
    return X_val[idx], y_val[idx]


def build(n_train_good_runs=10, n_test_good_runs=10, n_test_fault_runs=10, hpo_detector="iforest"):
    """Berechnet alle Strategien auf einem gemeinsamen Split und gibt ein Ergebnis-Dict zurück."""
    split = load_split(
        n_train_good_runs=n_train_good_runs,
        n_test_good_runs=n_test_good_runs,
        n_test_fault_runs=n_test_fault_runs,
    )
    X_val_full, y_val_full, _ = load_validation(split, n_good_runs=5, n_fault_runs=5)
    X_val, y_val = _val_subsample(X_val_full, y_val_full)

    rows: dict[str, dict] = {}

    # 1) Bester Einzeldetektor mit Defaults (label-frei nicht wählbar — nur als Referenz).
    best_default_name, best_default = None, -1.0
    for name, hp in DEFAULT_CANDIDATES:
        det = make_detector(name, **hp).fit(split.X_train_good)
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

    # 3) Ensemble (Strategie 4, label-frei) — mit contamination-basiertem Threshold.
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

    return rows, {"best_default": best_default_name, "internal": internal_best, "oracle": oracle_best}


def _write_markdown_table(rows: dict[str, dict], path):
    cols = ["roc_auc", "pr_auc", "f1", "detection_delay", "false_alarm_rate"]
    lines = ["| Strategie | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for name, m in rows.items():
        vals = []
        for c in cols:
            v = m.get(c, float("nan"))
            vals.append(f"{v:.3f}" if v == v else "—")  # NaN-Check
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, choices = build()

    _write_markdown_table(rows, config.REPORTS_DIR / "summary_table.md")

    # Zentrale Grafik: label-frei (Konsens, Ensemble) vs. Oracle.
    highlight = {
        "Ensemble (label-frei)": rows["ensemble:average"],
        f"Konsens-Select ({choices['internal']})": rows[f"select_internal:{choices['internal']}"],
        f"Oracle-Select ({choices['oracle']})": rows[f"select_oracle:{choices['oracle']}"],
    }
    comparison_bars(highlight, metric="roc_auc", save_as="summary_label_free_vs_oracle.png")

    print("Auswahl:", choices)
    print("Tabelle ->", config.REPORTS_DIR / "summary_table.md")
    for name, m in rows.items():
        print(f"  {name:28s} ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f} F1={m['f1']:.3f}")


if __name__ == "__main__":
    main()
