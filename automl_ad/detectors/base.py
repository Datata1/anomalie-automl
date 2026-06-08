"""Einheitliches Detektor-Interface + Auto-Discovery-Registry.

Wir standardisieren auf die PyOD-``BaseDetector``-Signatur. Jede AutoML-Strategie spricht nur
gegen dieses Protokoll und kann so mit jedem registrierten Detektor arbeiten.

**Auto-Discovery (T01):** Die Registry wird automatisch aus allen Modulen in diesem Package
aufgebaut, die ein Dict ``FACTORIES`` exportieren. Ein neuer Detektor ist damit ein reines
Drop-in (neue Datei mit ``FACTORIES = {"name": make_fn}``) — kein Edit an dieser Datei nötig.
Module, deren Import scheitert (z. B. fehlende optionale Dependency wie torch/minisom), werden
übersprungen statt zu propagieren.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AnomalyDetector(Protocol):
    """Minimal-Kontrakt (von allen PyOD-Detektoren erfüllt)."""

    decision_scores_: np.ndarray
    threshold_: float

    def fit(self, X: np.ndarray) -> "AnomalyDetector": ...
    def decision_function(self, X: np.ndarray) -> np.ndarray: ...  # höher = anomaler


def _discover_factories() -> dict[str, Callable]:
    """Sammelt ``FACTORIES`` aus allen Geschwister-Modulen ein (guarded)."""
    registry: dict[str, Callable] = {}
    pkg_path = [os.path.dirname(__file__)]
    for mod_info in pkgutil.iter_modules(pkg_path):
        if mod_info.name == "base":
            continue
        try:
            module = importlib.import_module(f".{mod_info.name}", package=__package__)
        except Exception:  # noqa: BLE001 - optionale Dependency fehlt → Modul überspringen
            continue
        factories = getattr(module, "FACTORIES", None)
        if isinstance(factories, dict):
            registry.update(factories)
    return dict(sorted(registry.items()))


# Registry: name -> factory(**hyperparams) -> Detector
REGISTRY: dict[str, Callable] = _discover_factories()


def make_detector(name: str, **hp) -> AnomalyDetector:
    """Erzeugt einen Detektor über die Registry."""
    if name not in REGISTRY:
        raise KeyError(f"Unbekannter Detektor '{name}'. Verfügbar: {sorted(REGISTRY)}")
    return REGISTRY[name](**hp)


def available_detectors() -> list[str]:
    return sorted(REGISTRY)
