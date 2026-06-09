"""Exportiert den supervised Train/Test-Split als .npy (für den isolierten AutoGluon-Benchmark).

AutoGluon ist unter Python 3.13 nicht installierbar → der Benchmark läuft in einem separaten
py3.12-venv. Dieses Skript (Haupt-venv) dumpt die Daten, die das isolierte Skript dann lädt.

Ausführen:  uv run python scripts/export_supervised_for_ag.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from automl_ad.data import load_supervised

OUT = Path("/tmp/ag_data")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_supervised(n_good_runs=10, n_fault_runs=10)
    np.save(OUT / "X_train.npy", d.X_train)
    np.save(OUT / "y_train.npy", d.y_train)
    np.save(OUT / "X_test.npy", d.X_test)
    np.save(OUT / "y_test.npy", d.y_test)
    print(f"dumped train={d.X_train.shape} test={d.X_test.shape} -> {OUT}")


if __name__ == "__main__":
    main()
