"""Zentrale Konstanten & Pfade.

Alles, was an mehreren Stellen gebraucht wird (Spaltennamen, Onset-Logik, Default-Seeds,
Subsample-Größen, Verzeichnisse), liegt hier — keine Magic Numbers im restlichen Code.
"""

from __future__ import annotations

from pathlib import Path

# --- Verzeichnisse -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

# Parquet-Dateien (siehe docs/methoden/01_datensatz_tep.md)
FAULT_FREE_TRAINING = DATA_DIR / "TEP_FaultFree_Training.parquet"
FAULT_FREE_TESTING = DATA_DIR / "TEP_FaultFree_Testing.parquet"
FAULTY_TRAINING = DATA_DIR / "TEP_Faulty_Training.parquet"
FAULTY_TESTING = DATA_DIR / "TEP_Faulty_Testing.parquet"

# --- Spalten -----------------------------------------------------------------------
META_COLS = ["faultNumber", "simulationRun", "sample"]
XMEAS_COLS = [f"xmeas_{i}" for i in range(1, 42)]  # 41 Messvariablen
XMV_COLS = [f"xmv_{i}" for i in range(1, 12)]      # 11 Stellgrößen (bestätigt: 11, nicht 12)
FEATURE_COLS = XMEAS_COLS + XMV_COLS               # 52 Features

# --- Fehler-Onset (siehe docs/methoden/01_datensatz_tep.md §4) ----------------------
# In faulty_* ist der Fehler erst ab diesen Sample-Indizes aktiv; davor ist der Lauf normal.
ONSET_TESTING = 160   # faulty_testing: Fehler ab sample > 160
ONSET_TRAINING = 20   # faulty_training: Fehler ab sample > 20
SAMPLES_PER_RUN_TESTING = 960
SAMPLES_PER_RUN_TRAINING = 500

# --- Defaults ----------------------------------------------------------------------
RANDOM_SEED = 0
N_RUNS_TOTAL = 500                 # simulationRun reicht von 1..500
ALL_FAULTS = list(range(1, 21))    # 20 Fehlertypen
# Repräsentative Auswahl für schnelle Demos: leicht (1,2,6,7), mittel/schwer (4,11,13), schwer (3,9,15)
DEMO_FAULTS = [1, 2, 4, 6, 11, 13, 3, 9, 15]
DEFAULT_CONTAMINATION = 0.1        # Annahme für Threshold-Quantil (PyOD-Default)
