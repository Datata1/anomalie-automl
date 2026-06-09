"""Fortgeschrittenes Ensembling (TICKET-07): greedy Ensemble Selection + Feature Bagging.

- ``greedy_ensemble_selection`` — Caruana-Greedy (label-basiert, Oracle): wählt iterativ mit
  Zurücklegen die Detektoren, die die ROC-AUC auf einem Validierungsset maximieren.
- ``feature_bagging_scores`` — label-frei: trainiert denselben Detektor auf zufälligen
  Feature-Teilmengen, normalisiert und mittelt (Diversität ohne Labels).
- ``build_score_matrices`` — Hilfsfunktion: standardisierte Score-Matrizen (Punkte × Detektoren)
  für beliebige Eval-Sets (Normalisierung via PyOD ``standardizer``, gefittet auf Train-Scores).

Vergleich zum einfachen Mittelwert-Ensemble in ``automl_ad/automl/ensemble.py``.
"""

from __future__ import annotations

import numpy as np
from pyod.utils.utility import standardizer
from sklearn.metrics import roc_auc_score

from ..detectors import make_detector


def build_score_matrices(candidates, X_train, *eval_sets) -> list[np.ndarray]:
    """Fittet Kandidaten auf Gutdaten; liefert je Eval-Set eine standardisierte Score-Matrix.

    Rückgabe: Liste von Arrays der Form ``(n_points, n_detektoren)``, Z-normalisiert anhand der
    Trainings-Scores (gemeinsame Skala für die Aggregation).
    """
    train_cols: list[np.ndarray] = []
    eval_cols: list[list[np.ndarray]] = [[] for _ in eval_sets]
    for name, hp in candidates:
        det = make_detector(name, **hp).fit(X_train)
        train_cols.append(det.decision_scores_)
        for i, X_eval in enumerate(eval_sets):
            eval_cols[i].append(det.decision_function(X_eval))

    train_mat = np.asarray(train_cols).T
    out: list[np.ndarray] = []
    for cols in eval_cols:
        mat = np.asarray(cols).T
        _, norm = standardizer(train_mat, mat)
        out.append(norm)
    return out


def greedy_ensemble_selection(val_scores: np.ndarray, y_val: np.ndarray, n_rounds: int = 20) -> np.ndarray:
    """Greedy Ensemble Selection (mit Zurücklegen). Gibt einen Gewichtsvektor über Detektoren.

    ``val_scores``: standardisierte Score-Matrix ``(n_points, n_detektoren)`` auf dem Val-Set.
    """
    n_det = val_scores.shape[1]
    weights = np.zeros(n_det)
    current_sum = np.zeros(val_scores.shape[0])
    for r in range(n_rounds):
        best_auc, best_i = -1.0, 0
        for i in range(n_det):
            cand = (current_sum + val_scores[:, i]) / (r + 1)
            auc = roc_auc_score(y_val, cand)
            if auc > best_auc:
                best_auc, best_i = auc, i
        current_sum += val_scores[:, best_i]
        weights[best_i] += 1
    return weights / weights.sum()


def feature_bagging_scores(
    make_fn,
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 10,
    max_features: float = 0.6,
    seed: int = 0,
) -> np.ndarray:
    """Feature Bagging (label-frei): Detektor auf zufälligen Feature-Teilmengen, Scores gemittelt."""
    rng = np.random.default_rng(seed)
    n_features = X_train.shape[1]
    k = max(1, int(max_features * n_features))

    train_cols, test_cols = [], []
    for _ in range(n_estimators):
        cols = rng.choice(n_features, size=k, replace=False)
        det = make_fn().fit(X_train[:, cols])
        train_cols.append(det.decision_scores_)
        test_cols.append(det.decision_function(X_test[:, cols]))

    train_mat = np.asarray(train_cols).T
    test_mat = np.asarray(test_cols).T
    _, test_norm = standardizer(train_mat, test_mat)
    return test_norm.mean(axis=1)
