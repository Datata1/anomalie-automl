"""Interne (label-freie) Validierungsmetriken (TICKET-06).

Implementiert **SIREOS** (Similarity-based Internal, Relative Evaluation of Outlier Solutions):
gewichtet jeden Punkt mit seinem normalisierten Anomalie-Score und misst dessen mittlere
**Ähnlichkeit** zum Rest. Gute Detektoren vergeben hohe Scores an *isolierte* Punkte → niedrige
mittlere Ähnlichkeit → **niedriger SIREOS = besser**. Damit lässt sich label-frei selektieren.

Literatur warnt: interne Metriken korrelieren teils nur schwach mit der echten Performance —
das Notebook 09 zeigt diese Korrelation transparent. Siehe
docs/methoden/automl-strategien/00_modellselektion_ohne_labels.md.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from ..detectors import make_detector
from .selection import DEFAULT_CANDIDATES


def _minmax(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng > 0 else np.zeros_like(s)


def sireos(
    X: np.ndarray,
    scores: np.ndarray,
    t_quantile: float = 0.01,
    subsample: int = 2000,
    seed: int = 0,
) -> float:
    """SIREOS einer Detektor-Lösung (niedriger = besser). Auf Subsample für Tractabilität."""
    rng = np.random.default_rng(seed)
    if len(X) > subsample:
        idx = rng.choice(len(X), subsample, replace=False)
        X, scores = X[idx], np.asarray(scores)[idx]

    w = _minmax(scores)
    if w.sum() == 0:
        return 1.0
    p = w / w.sum()

    D = cdist(X, X)
    pos = D[D > 0]
    t = float(np.quantile(pos, t_quantile)) if pos.size else 1.0
    t = t if t > 0 else 1.0
    similarity = np.exp(-(D**2) / (2 * t**2)).mean(axis=1)  # mittlere Ähnlichkeit je Punkt
    return float(np.sum(p * similarity))


def select_by_internal(
    candidates=DEFAULT_CANDIDATES,
    X_train=None,
    X_eval=None,
    seed: int = 0,
) -> tuple[str, dict[str, float]]:
    """Label-freie Auswahl per SIREOS: fittet Kandidaten auf Gutdaten, wählt **min** SIREOS."""
    metrics: dict[str, float] = {}
    for name, hp in candidates:
        det = make_detector(name, **hp).fit(X_train)
        metrics[name] = sireos(X_eval, det.decision_function(X_eval), seed=seed)
    best = min(metrics, key=metrics.get)  # niedriger SIREOS = besser
    return best, metrics
