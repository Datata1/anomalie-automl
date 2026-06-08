"""Self-Organizing-Map-Detektor (TICKET-04, optionale ``som``-Dependency ``minisom``).

PyOD-artiges Interface (``fit`` / ``decision_function`` / ``decision_scores_`` /
``threshold_``). Score = mittlere Distanz eines Punktes zu seinen k nächsten Neuronen
(höher = anomaler). Registrierung via Auto-Discovery (``detectors/base.py``) als ``"som"``;
nur verfügbar, wenn ``minisom`` installiert ist. Der SOM-Detektor liefert zusätzlich eine 2D-
Karte (``self.som_``) zur Visualisierung.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .. import config


class _SOMDetector:
    def __init__(
        self,
        map_size: int = 15,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        num_iteration: int = 10000,
        k: int = 3,
        prune: bool = True,
        contamination: float = config.DEFAULT_CONTAMINATION,
        random_state: int = config.RANDOM_SEED,
    ):
        self.map_size = map_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.k = k
        self.prune = prune
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> "_SOMDetector":
        from minisom import MiniSom

        self.som_ = MiniSom(
            self.map_size, self.map_size, X.shape[1],
            sigma=self.sigma, learning_rate=self.learning_rate, random_seed=self.random_state,
        )
        self.som_.random_weights_init(X)
        self.som_.train_random(X, self.num_iteration)

        weights = self.som_.get_weights().reshape(-1, X.shape[1])
        if self.prune:
            # Rauschreduktion: nur Neuronen behalten, die für >=1 Trainingspunkt BMU sind.
            bmu = np.argmin(cdist(X, weights), axis=1)
            keep = np.unique(bmu)
            weights = weights[keep]
        self.neurons_ = weights

        self.decision_scores_ = self._score(X)
        self.threshold_ = float(np.quantile(self.decision_scores_, 1 - self.contamination))
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        d = cdist(X, self.neurons_)
        k = min(self.k, d.shape[1])
        return np.sort(d, axis=1)[:, :k].mean(axis=1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._score(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > self.threshold_).astype(int)


def make_som(**hp) -> _SOMDetector:
    return _SOMDetector(**hp)


FACTORIES = {"som": make_som}
