"""Fertiges AutoML-Framework für die supervise Fehlerklassifikation (Strategie 2).

Nutzt **FLAML** (statt auto-sklearn — robust unter Python 3.13) für CASH: automatische
Modellauswahl + HPO. Dient als supervised Vergleichspol zur unüberwachten AD
(„Klassifikation vs. AD", siehe docs/methoden/ad-methoden/random_forest_klassifikation.md).
"""

from __future__ import annotations

from flaml.automl import AutoML

from .. import config


def run_flaml_classification(
    X_train,
    y_train,
    time_budget: int = 60,
    metric: str = "macro_f1",
    seed: int = config.RANDOM_SEED,
) -> AutoML:
    """Trainiert ein FLAML-AutoML-Modell (CASH) für Mehrklassen-Fehlerklassifikation.

    Gibt das gefittete ``AutoML``-Objekt zurück (``.predict`` / ``.predict_proba`` /
    ``.best_estimator`` / ``.best_config``).
    """
    automl = AutoML()
    automl.fit(
        X_train,
        y_train,
        task="classification",
        metric=metric,
        time_budget=time_budget,
        seed=seed,
        verbose=0,
    )
    return automl
