"""Deep One-Class Detektoren (TICKET-03): Deep SVDD (unüberwacht) & DeepSAD (semi-supervised).

- ``deep_svdd`` — immer verfügbar (pyod), Lazy-Wrapper, da pyod ``DeepSVDD`` ``n_features`` bei
  Konstruktion benötigt.
- ``deep_sad`` — nur, wenn ``deepod`` importierbar ist (semi-supervised; nutzt ein kleines
  Label-Budget bekannter Anomalien). Sonst greift im Notebook der Deep-SVDD-Fallback.

Beide erfüllen das PyOD-artige Interface (``fit`` / ``decision_function`` /
``decision_scores_`` / ``threshold_``); Score höher = anomaler. Registrierung via
Auto-Discovery (``detectors/base.py``).
"""

from __future__ import annotations

import numpy as np

from .. import config


class _DeepSVDD:
    """Lazy-Wrapper um ``pyod.models.deep_svdd.DeepSVDD`` (unüberwacht)."""

    def __init__(self, **hp):
        self.hp = hp

    def fit(self, X: np.ndarray, y=None) -> "_DeepSVDD":
        from pyod.models.deep_svdd import DeepSVDD

        params = {
            "epochs": 20,
            "contamination": config.DEFAULT_CONTAMINATION,
            "random_state": config.RANDOM_SEED,
            "verbose": 0,
        }
        params.update(self.hp)
        self.model_ = DeepSVDD(n_features=X.shape[1], **params)
        self.model_.fit(X)
        self.decision_scores_ = self.model_.decision_scores_
        self.threshold_ = self.model_.threshold_
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.model_.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > self.threshold_).astype(int)


def make_deep_svdd(**hp) -> _DeepSVDD:
    return _DeepSVDD(**hp)


FACTORIES = {"deep_svdd": make_deep_svdd}


# --- Optional: echtes DeepSAD (semi-supervised) via deepod -----------------------------
try:  # pragma: no cover - abhängig von optionaler Installation
    from deepod.models import DeepSAD as _DeepSADImpl

    class _DeepSAD:
        """Wrapper um deepod ``DeepSAD``. ``fit(X, y)`` mit y: 0=unlabeled, 1=bekannte Anomalie."""

        def __init__(self, **hp):
            self.hp = {"epochs": 20, "random_state": config.RANDOM_SEED, "verbose": 0}
            self.hp.update(hp)
            self.contamination = self.hp.pop("contamination", config.DEFAULT_CONTAMINATION)

        def fit(self, X: np.ndarray, y=None) -> "_DeepSAD":
            self.model_ = _DeepSADImpl(**self.hp)
            self.model_.fit(X, y)
            self.decision_scores_ = self.model_.decision_function(X)
            self.threshold_ = float(np.quantile(self.decision_scores_, 1 - self.contamination))
            return self

        def decision_function(self, X: np.ndarray) -> np.ndarray:
            return self.model_.decision_function(X)

        def predict(self, X: np.ndarray) -> np.ndarray:
            return (self.decision_function(X) > self.threshold_).astype(int)

    def make_deep_sad(**hp) -> "_DeepSAD":
        return _DeepSAD(**hp)

    FACTORIES["deep_sad"] = make_deep_sad
except Exception:  # noqa: BLE001 - deepod nicht (kompatibel) installiert → nur Deep SVDD
    pass
