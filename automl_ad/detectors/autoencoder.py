"""Torch-Autoencoder als PyOD-Detektor (optionale ``dl``-Abhängigkeit).

Nutzt ``pyod.models.auto_encoder.AutoEncoder`` (torch-basiert), damit der AE dasselbe
``BaseDetector``-Interface wie die klassischen Detektoren erfüllt und überall austauschbar ist.
Wird nur registriert, wenn torch/pyod-AE importierbar sind (siehe ``base.py``).
"""

from __future__ import annotations

from pyod.models.auto_encoder import AutoEncoder

from .. import config


class _ChainableAutoEncoder(AutoEncoder):
    """PyOD-AutoEncoder, dessen ``fit`` ``self`` zurückgibt (wie alle anderen Detektoren).

    Der PyOD-AutoEncoder gibt aus ``fit`` ``None`` zurück; das bricht das übliche
    ``make_detector(...).fit(X)``-Pattern. Diese Subklasse stellt das einheitliche Verhalten
    wieder her.
    """

    def fit(self, X, y=None):
        super().fit(X, y)
        return self


def make_autoencoder(**hp):
    """Factory für den PyOD-AutoEncoder mit projektweiten Defaults."""
    params = {
        "contamination": config.DEFAULT_CONTAMINATION,
        "random_state": config.RANDOM_SEED,
    }
    params.update(hp)
    return _ChainableAutoEncoder(**params)


AE_FACTORIES = {"autoencoder": make_autoencoder}

# Von der Auto-Discovery (detectors/base.py) eingesammelt.
FACTORIES = AE_FACTORIES
