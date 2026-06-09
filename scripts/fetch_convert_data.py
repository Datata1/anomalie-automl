"""Daten-Beschaffung & -Validierung für den TEP-Datensatz (TICKET-10).

Lädt **keine** Daten automatisch herunter (Kaggle/Harvard Dataverse erfordern Auth), sondern
prüft Vorhandensein und **Schema** der vier Parquet-Dateien und gibt sonst eine klare Anleitung.

Ausführen:  ``uv run python scripts/fetch_convert_data.py --check``
"""

from __future__ import annotations

import argparse
import sys

import pyarrow.parquet as pq

from automl_ad import config

_EXPECTED_COLS = set(config.META_COLS + config.FEATURE_COLS)  # 55 Spalten

_FILES = {
    "FaultFree_Training": config.FAULT_FREE_TRAINING,
    "FaultFree_Testing": config.FAULT_FREE_TESTING,
    "Faulty_Training": config.FAULTY_TRAINING,
    "Faulty_Testing": config.FAULTY_TESTING,
}

_ANLEITUNG = """
So beschaffst du die Daten:
  1. Lade die vier .RData-Dateien von Kaggle/Harvard Dataverse nach data/:
     https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset
     (TEP_FaultFree_Training/Testing.RData, TEP_Faulty_Training/Testing.RData)
  2. Konvertiere sie nach Parquet:  Rscript export_rdata.R
  3. Prüfe erneut:  uv run python scripts/fetch_convert_data.py --check
Details: docs/methoden/01_datensatz_tep.md §6
""".strip()


def check() -> bool:
    ok = True
    for name, path in _FILES.items():
        if not path.exists():
            print(f"[FEHLT]  {path}")
            ok = False
            continue
        cols = set(pq.read_schema(path).names)
        if cols == _EXPECTED_COLS:
            print(f"[OK]     {name}: {len(cols)} Spalten (Schema korrekt)")
        else:
            missing = _EXPECTED_COLS - cols
            extra = cols - _EXPECTED_COLS
            print(f"[SCHEMA] {name}: fehlend={sorted(missing)} unerwartet={sorted(extra)}")
            ok = False
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="TEP-Daten prüfen/konvertieren.")
    parser.add_argument("--check", action="store_true", help="Vorhandensein + Schema prüfen")
    args = parser.parse_args()

    if not args.check:
        parser.print_help()
        return

    if check():
        print("\nAlle vier Parquet-Dateien vorhanden und Schema korrekt (52 Features + 3 Meta).")
        sys.exit(0)
    print("\n" + _ANLEITUNG)
    sys.exit(1)


if __name__ == "__main__":
    main()
