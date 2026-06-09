"""Detektor-Kontrakt (TICKET-09): tabellarische, unüberwachte Detektoren. Marker ``data``.

``lstm_ae`` (3D-Sequenzen) und ``deep_sad`` (Label-Budget) werden hier bewusst **ausgeschlossen**
— sie erfüllen nicht den 2D-``fit(X)``-Standardkontrakt (siehe plans/phase-c/00_README.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from automl_ad.data import load_split
from automl_ad.detectors import available_detectors, make_detector

TABULAR_UNSUPERVISED = {"ecod", "iforest", "ocsvm", "pca", "autoencoder", "som", "deep_svdd"}


@pytest.mark.data
@pytest.mark.parametrize("name", sorted(TABULAR_UNSUPERVISED))
def test_detector_contract(name):
    if name not in available_detectors():
        pytest.skip(f"Detektor '{name}' nicht verfügbar (optionale Dependency fehlt)")
    s = load_split(faults=[1], n_train_good_runs=3, n_test_good_runs=2, n_test_fault_runs=2, seed=0)
    # Kleines Trainings-Subsample für Tempo (teure Detektoren wie OCSVM/AE).
    det = make_detector(name).fit(s.X_train_good[:1200])
    scores = det.decision_function(s.X_test)
    assert scores.shape[0] == s.X_test.shape[0]
    assert np.all(np.isfinite(scores))
    assert hasattr(det, "threshold_") and np.isfinite(det.threshold_)
    assert hasattr(det, "decision_scores_")


@pytest.mark.data
def test_registry_contains_core_detectors():
    av = available_detectors()
    for name in ["ecod", "iforest", "ocsvm", "pca"]:
        assert name in av
