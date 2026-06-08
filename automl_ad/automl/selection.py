"""Modellselektion: Oracle (label-basiert) vs. label-frei (Konsens).

Kernstück des Projekts (siehe docs/methoden/automl-strategien/00_modellselektion_ohne_labels.md):
Im unüberwachten Betrieb fehlen Labels zur Auswahl des besten Detektors. Hier stehen die
label-basierte Obergrenze (``select_oracle``) und eine label-freie Konsens-Heuristik
(``select_internal``) nebeneinander, um den **Gap** sichtbar zu machen.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from ..detectors import make_detector

# Kandidat = (Detektorname, Hyperparameter-Dict)
Candidate = tuple[str, dict]

DEFAULT_CANDIDATES: list[Candidate] = [
    ("ecod", {}),
    ("iforest", {}),
    ("ocsvm", {}),
    ("pca", {}),
]


def _fit_scores(candidates, X_train, X_eval) -> dict[str, np.ndarray]:
    """Fittet alle Kandidaten auf Gutdaten und gibt ihre Eval-Scores zurück."""
    out = {}
    for name, hp in candidates:
        det = make_detector(name, **hp).fit(X_train)
        out[name] = det.decision_function(X_eval)
    return out


def select_oracle(candidates, X_train, X_val, y_val) -> tuple[str, dict[str, float]]:
    """Label-basierte Auswahl (Obergrenze): bester ROC-AUC auf gelabeltem Val-Set."""
    scores = _fit_scores(candidates, X_train, X_val)
    aucs = {name: float(roc_auc_score(y_val, s)) for name, s in scores.items()}
    best = max(aucs, key=aucs.get)
    return best, aucs


def select_internal(candidates, X_train, X_eval) -> tuple[str, dict[str, float]]:
    """Label-freie Auswahl per **Konsens/Model-Centrality**.

    Idee: Kein einzelnes Modell ist die Wahrheit, aber der Rang-Konsens vieler Modelle ist
    robust. Bevorzuge den Detektor, dessen Score-Ranking am stärksten mit dem Konsens
    korreliert (zentralstes Modell). Benötigt keine Labels.
    """
    scores = _fit_scores(candidates, X_train, X_eval)
    names = list(scores)
    ranks = {n: rankdata(scores[n]) for n in names}
    consensus = np.mean([ranks[n] for n in names], axis=0)
    centrality = {n: float(np.corrcoef(ranks[n], consensus)[0, 1]) for n in names}
    best = max(centrality, key=centrality.get)
    return best, centrality
