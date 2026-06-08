"""Klassische PyOD-Detektoren als schlanke Factories.

Alle nutzen das PyOD-``BaseDetector``-Interface (``fit`` / ``decision_function`` mit
„höher = anomaler" / ``decision_scores_`` / ``threshold_``). Dadurch sind sie überall
austauschbar — siehe ``automl_ad/detectors/base.py``.
"""

from __future__ import annotations

from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

from .. import config


def make_iforest(**hp):
    params = {"random_state": config.RANDOM_SEED, "contamination": config.DEFAULT_CONTAMINATION}
    params.update(hp)
    return IForest(**params)


def make_ocsvm(**hp):
    params = {"contamination": config.DEFAULT_CONTAMINATION}
    params.update(hp)
    return OCSVM(**params)


def make_pca(**hp):
    params = {"random_state": config.RANDOM_SEED, "contamination": config.DEFAULT_CONTAMINATION}
    params.update(hp)
    return PCA(**params)


def make_ecod(**hp):
    # ECOD ist parameterfrei (außer contamination) — exzellenter Default-Anker.
    params = {"contamination": config.DEFAULT_CONTAMINATION}
    params.update(hp)
    return ECOD(**params)


CLASSICAL_FACTORIES = {
    "iforest": make_iforest,
    "ocsvm": make_ocsvm,
    "pca": make_pca,
    "ecod": make_ecod,
}

# Von der Auto-Discovery (detectors/base.py) eingesammelt.
FACTORIES = CLASSICAL_FACTORIES
