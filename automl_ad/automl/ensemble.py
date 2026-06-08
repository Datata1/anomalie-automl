"""Score-Ensembling (AutoML-Strategie 4).

Kombiniert mehrere Detektoren label-frei durch Score-Normalisierung + Aggregation
(PyOD ``combination``). Umgeht das Modellselektionsproblem teilweise: statt *das eine* beste
Modell label-frei zu finden, aggregiert man robust über viele.
Siehe docs/methoden/automl-strategien/ensembling.md.
"""

from __future__ import annotations

import numpy as np
from pyod.utils.utility import standardizer

from .. import config
from ..detectors import make_detector
from .selection import DEFAULT_CANDIDATES

# Score-Aggregation über die Detektor-Achse (axis=1). Trivial, daher ohne Extra-Paket `combo`.
_COMBINERS = {
    "average": lambda scores: scores.mean(axis=1),
    "maximization": lambda scores: scores.max(axis=1),
}


def ensemble_scores(
    candidates=DEFAULT_CANDIDATES,
    X_train=None,
    X_test=None,
    method: str = "average",
    return_threshold: bool = False,
    contamination: float = config.DEFAULT_CONTAMINATION,
):
    """Fittet alle Kandidaten auf Gutdaten, normalisiert ihre Scores und aggregiert sie.

    ``method`` ∈ {"average", "maximization"}. Gibt einen kombinierten Anomaly-Score je
    Testpunkt zurück (höher = anomaler). Mit ``return_threshold=True`` zusätzlich einen
    contamination-basierten Threshold (Quantil der aggregierten Trainings-Scores), damit die
    Schwellwert-Metriken vergleichbar zu den Einzeldetektoren sind.
    """
    if method not in _COMBINERS:
        raise KeyError(f"method muss in {list(_COMBINERS)} sein.")

    train_mat, test_mat = [], []
    for name, hp in candidates:
        det = make_detector(name, **hp).fit(X_train)
        train_mat.append(det.decision_scores_)
        test_mat.append(det.decision_function(X_test))

    train_mat = np.asarray(train_mat).T  # (n_train, n_detektoren)
    test_mat = np.asarray(test_mat).T    # (n_test, n_detektoren)

    # Gemeinsame Skala (Z-Score), gefittet auf Trainings-Scores.
    train_norm, test_norm = standardizer(train_mat, test_mat)
    agg_test = _COMBINERS[method](test_norm)
    if return_threshold:
        thr = float(np.quantile(_COMBINERS[method](train_norm), 1 - contamination))
        return agg_test, thr
    return agg_test
