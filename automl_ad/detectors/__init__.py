"""Detektor-Subpackage: einheitliches Interface + Registry."""

from .base import REGISTRY, AnomalyDetector, available_detectors, make_detector

__all__ = ["REGISTRY", "AnomalyDetector", "available_detectors", "make_detector"]
