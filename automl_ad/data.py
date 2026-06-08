"""Daten-Loading, Run-Level-Split, Skalierung und Onset-korrektes Labeling.

Konventionen (siehe docs/methoden/02_evaluationsprotokoll.md):
- Split **nach simulationRun** (nie zeilenweise), Train-/Test-Läufe disjunkt.
- Unsupervised AD: Training nur auf Gutdaten; StandardScaler nur auf Gutdaten gefittet.
- Punktweises Label berücksichtigt den Fehler-Onset (faultNumber!=0 ≠ jede Zeile anomal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config


# --------------------------------------------------------------------------------------
# Roh-Loading (mit Predicate-Pushdown auf simulationRun / faultNumber)
# --------------------------------------------------------------------------------------
def _read_runs(
    path,
    runs: list[int],
    faults: list[int] | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Liest nur die gewünschten Läufe (und optional Fehlertypen) aus einer Parquet-Datei."""
    filters = [("simulationRun", "in", list(runs))]
    if faults is not None:
        filters.append(("faultNumber", "in", list(faults)))
    cols = columns if columns is not None else config.META_COLS + config.FEATURE_COLS
    return pd.read_parquet(path, columns=cols, filters=filters)


def _pick_runs(n: int, seed: int, exclude: set[int] | None = None) -> list[int]:
    """Wählt reproduzierbar n Lauf-IDs aus 1..N_RUNS_TOTAL (optional unter Ausschluss)."""
    rng = np.random.default_rng(seed)
    pool = [r for r in range(1, config.N_RUNS_TOTAL + 1) if not exclude or r not in exclude]
    if n > len(pool):
        raise ValueError(f"{n} Läufe angefragt, aber nur {len(pool)} verfügbar.")
    return sorted(rng.choice(pool, size=n, replace=False).tolist())


# --------------------------------------------------------------------------------------
# Labeling
# --------------------------------------------------------------------------------------
def make_labels(df: pd.DataFrame, onset: int) -> np.ndarray:
    """Punktweises Anomalie-Label (1=anomal) unter Berücksichtigung des Fehler-Onsets."""
    return ((df["faultNumber"] != 0) & (df["sample"] > onset)).to_numpy().astype(int)


def make_multiclass_labels(df: pd.DataFrame, onset: int) -> np.ndarray:
    """Mehrklassen-Label: faultNumber, falls Fehler aktiv (sample>onset), sonst 0 (normal)."""
    active = df["sample"] > onset
    return np.where(active, df["faultNumber"], 0).astype(int)


# --------------------------------------------------------------------------------------
# Split-Container
# --------------------------------------------------------------------------------------
@dataclass
class Split:
    """Ein fertig vorbereiteter, skalierter Train/Test-Split für unsupervised AD."""

    X_train_good: np.ndarray          # nur Gutdaten, skaliert
    X_test: np.ndarray                # gemischt (normal + Fehler), skaliert
    y_test: np.ndarray                # punktweises Label (1=anomal, Onset-korrekt)
    meta_test: pd.DataFrame           # faultNumber/simulationRun/sample (für Detection-Delay etc.)
    scaler: StandardScaler
    faults: list[int]                 # im Test enthaltene Fehlertypen
    train_runs: list[int]             # Gutdaten-Läufe im Training (für disjunkte Validierung)
    test_good_runs: list[int]         # Gutdaten-Läufe im Test (disjunkt zu Validierung)
    test_fault_runs: list[int]        # Fehler-Läufe im Test (disjunkt zu Validierung)

    def __repr__(self) -> str:  # kompakte Übersicht
        return (
            f"Split(train_good={self.X_train_good.shape}, test={self.X_test.shape}, "
            f"anomaly_rate={self.y_test.mean():.3f})"
        )


def load_split(
    faults: list[int] | None = None,
    n_train_good_runs: int = 25,
    n_test_good_runs: int = 20,
    n_test_fault_runs: int = 20,
    seed: int = config.RANDOM_SEED,
) -> Split:
    """Baut einen unsupervised-AD-Split aus den TEP-Parquet-Dateien.

    - Training: Gutdaten aus ``fault_free_training`` (``n_train_good_runs`` Läufe).
    - Test: Gutdaten aus ``fault_free_testing`` (``n_test_good_runs`` Läufe) + Fehlerdaten aus
      ``faulty_testing`` (``n_test_fault_runs`` Läufe je Fehlertyp in ``faults``).
    - Scaler wird nur auf den Trainings-Gutdaten gefittet.

    Train- und Test-Gutdaten stammen aus unterschiedlichen Dateien (verschiedene Simulationen)
    → kein Leakage trotz evtl. gleicher Lauf-Nummern.
    """
    faults = faults if faults is not None else config.DEMO_FAULTS

    train_runs = _pick_runs(n_train_good_runs, seed=seed)
    test_good_runs = _pick_runs(n_test_good_runs, seed=seed + 1)
    test_fault_runs = _pick_runs(n_test_fault_runs, seed=seed + 2)

    # Trainings-Gutdaten
    df_train = _read_runs(config.FAULT_FREE_TRAINING, train_runs)

    # Test: normal + Fehler
    df_test_good = _read_runs(config.FAULT_FREE_TESTING, test_good_runs)
    df_test_fault = _read_runs(config.FAULTY_TESTING, test_fault_runs, faults=faults)
    df_test = pd.concat([df_test_good, df_test_fault], ignore_index=True)

    # Skalierung (nur auf Gutdaten gefittet)
    scaler = StandardScaler().fit(df_train[config.FEATURE_COLS].to_numpy())
    X_train_good = scaler.transform(df_train[config.FEATURE_COLS].to_numpy())
    X_test = scaler.transform(df_test[config.FEATURE_COLS].to_numpy())

    y_test = make_labels(df_test, onset=config.ONSET_TESTING)
    meta_test = df_test[config.META_COLS].reset_index(drop=True)

    return Split(
        X_train_good, X_test, y_test, meta_test, scaler,
        faults, train_runs, test_good_runs, test_fault_runs,
    )


def load_validation(
    split: Split,
    n_good_runs: int = 10,
    n_fault_runs: int = 10,
    seed: int = 99,
    source: str = "testing",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Gelabeltes Validierungsset für (Oracle-)HPO/Selektion — **disjunkt vom Test**.

    - ``source="testing"`` (Default): aus den Testdateien (gleiche Verteilung wie der Test),
      aber mit Läufen, die **disjunkt** zu den Test-Läufen sind → standardgerechtes,
      leakage-freies HPO/Selektions-Set, Onset 160.
    - ``source="training"``: aus den **Trainings**-Dateien (``faulty_training``, Onset 20 +
      held-out ``fault_free_training``). Alternative, wenn die Testdateien völlig unberührt
      bleiben sollen.

    Skaliert mit dem Scaler aus ``split``.
    """
    if source == "testing":
        good_runs = _pick_runs(
            n_good_runs, seed=seed, exclude=set(split.test_good_runs)
        )
        fault_runs = _pick_runs(
            n_fault_runs, seed=seed + 1, exclude=set(split.test_fault_runs)
        )
        df_good = _read_runs(config.FAULT_FREE_TESTING, good_runs)
        df_fault = _read_runs(config.FAULTY_TESTING, fault_runs, faults=split.faults)
        onset = config.ONSET_TESTING
    elif source == "training":
        good_runs = _pick_runs(n_good_runs, seed=seed, exclude=set(split.train_runs))
        fault_runs = _pick_runs(n_fault_runs, seed=seed + 1)
        df_good = _read_runs(config.FAULT_FREE_TRAINING, good_runs)
        df_fault = _read_runs(config.FAULTY_TRAINING, fault_runs, faults=split.faults)
        onset = config.ONSET_TRAINING
    else:
        raise ValueError("source muss 'testing' oder 'training' sein.")

    df_val = pd.concat([df_good, df_fault], ignore_index=True)
    X_val = split.scaler.transform(df_val[config.FEATURE_COLS].to_numpy())
    y_val = make_labels(df_val, onset=onset)
    meta_val = df_val[config.META_COLS].reset_index(drop=True)
    return X_val, y_val, meta_val


@dataclass
class SupervisedData:
    """Gelabelter Datensatz für die supervise Fehlerklassifikation (Mehrklassen)."""

    X_train: np.ndarray
    y_train: np.ndarray       # 0=normal, 1..20=Fehlertyp
    X_test: np.ndarray
    y_test: np.ndarray
    meta_test: pd.DataFrame
    scaler: StandardScaler


def load_supervised(
    faults: list[int] | None = None,
    n_good_runs: int = 20,
    n_fault_runs: int = 20,
    seed: int = config.RANDOM_SEED,
) -> SupervisedData:
    """Baut Train/Test für die Mehrklassen-Klassifikation aus den TEP-Dateien.

    Train aus ``*_training`` (Onset 20), Test aus ``*_testing`` (Onset 160). Pre-Onset-Zeilen
    der Fehlerläufe sind normal → Label 0. Scaler wird nur auf den Trainingsdaten gefittet.
    """
    faults = faults if faults is not None else config.DEMO_FAULTS

    train_good_runs = _pick_runs(n_good_runs, seed=seed)
    train_fault_runs = _pick_runs(n_fault_runs, seed=seed + 1)
    test_good_runs = _pick_runs(n_good_runs, seed=seed + 2)
    test_fault_runs = _pick_runs(n_fault_runs, seed=seed + 3)

    df_tr = pd.concat(
        [
            _read_runs(config.FAULT_FREE_TRAINING, train_good_runs),
            _read_runs(config.FAULTY_TRAINING, train_fault_runs, faults=faults),
        ],
        ignore_index=True,
    )
    df_te = pd.concat(
        [
            _read_runs(config.FAULT_FREE_TESTING, test_good_runs),
            _read_runs(config.FAULTY_TESTING, test_fault_runs, faults=faults),
        ],
        ignore_index=True,
    )

    scaler = StandardScaler().fit(df_tr[config.FEATURE_COLS].to_numpy())
    X_train = scaler.transform(df_tr[config.FEATURE_COLS].to_numpy())
    X_test = scaler.transform(df_te[config.FEATURE_COLS].to_numpy())
    y_train = make_multiclass_labels(df_tr, onset=config.ONSET_TRAINING)
    y_test = make_multiclass_labels(df_te, onset=config.ONSET_TESTING)
    meta_test = df_te[config.META_COLS].reset_index(drop=True)

    return SupervisedData(X_train, y_train, X_test, y_test, meta_test, scaler)


# ======================================================================================
# Zeitreihen-Fenster (TICKET-02, additiv) — für sequenzbasierte Detektoren (LSTM-AE).
# ======================================================================================
def make_windows(
    X: np.ndarray,
    meta: pd.DataFrame,
    window: int = 30,
    stride: int = 1,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Erzeugt überlappende Sequenzfenster **innerhalb** eines Laufs.

    Gruppiert nach ``(faultNumber, simulationRun)`` (nie über Lauf-Grenzen), sortiert nach
    ``sample`` und schneidet Fenster der Länge ``window`` mit ``stride`` heraus. Pro Fenster
    wird die Metazeile des **letzten** Samples zurückgegeben (für Onset-korrektes Mapping des
    Scores auf Punktebene).

    Rückgabe: ``X_seq`` mit Form ``(n_windows, window, n_features)`` und ``meta_seq``
    (gleiche Spalten wie ``meta``, je Zeile das End-Sample des Fensters).
    """
    meta = meta.reset_index(drop=True)
    seqs: list[np.ndarray] = []
    end_idx: list[int] = []
    for _, g in meta.groupby(["faultNumber", "simulationRun"], sort=False):
        idx = g.sort_values("sample").index.to_numpy()
        for start in range(0, len(idx) - window + 1, stride):
            win = idx[start : start + window]
            seqs.append(X[win])
            end_idx.append(int(win[-1]))
    if not seqs:
        return np.empty((0, window, X.shape[1])), meta.iloc[0:0].copy()
    X_seq = np.stack(seqs)
    meta_seq = meta.iloc[end_idx].reset_index(drop=True)
    return X_seq, meta_seq


@dataclass
class WindowedData:
    """Gefensterter (sequenzbasierter) Split für unsupervise zeitliche AD."""

    X_train_seq: np.ndarray       # (n_train_windows, window, n_features), nur Gutdaten
    X_test_seq: np.ndarray        # (n_test_windows, window, n_features), gemischt
    y_test_seq: np.ndarray        # punktweises Label des End-Samples je Fenster (Onset-korrekt)
    meta_test_seq: pd.DataFrame   # End-Sample-Meta je Fenster
    scaler: StandardScaler

    def __repr__(self) -> str:
        return (
            f"WindowedData(train={self.X_train_seq.shape}, test={self.X_test_seq.shape}, "
            f"anomaly_rate={self.y_test_seq.mean():.3f})"
        )


def load_windowed(
    faults: list[int] | None = None,
    n_train_good_runs: int = 15,
    n_test_good_runs: int = 15,
    n_test_fault_runs: int = 15,
    window: int = 30,
    stride: int = 5,
    seed: int = config.RANDOM_SEED,
) -> WindowedData:
    """Wie ``load_split``, aber liefert **Sequenzfenster** (für den LSTM-Autoencoder).

    Scaler nur auf Trainings-Gutdaten gefittet; Fenster respektieren Lauf-Grenzen
    (``make_windows``). Onset-korrektes Label über das End-Sample je Fenster.
    """
    faults = faults if faults is not None else config.DEMO_FAULTS

    train_runs = _pick_runs(n_train_good_runs, seed=seed)
    test_good_runs = _pick_runs(n_test_good_runs, seed=seed + 1)
    test_fault_runs = _pick_runs(n_test_fault_runs, seed=seed + 2)

    df_train = _read_runs(config.FAULT_FREE_TRAINING, train_runs)
    df_test = pd.concat(
        [
            _read_runs(config.FAULT_FREE_TESTING, test_good_runs),
            _read_runs(config.FAULTY_TESTING, test_fault_runs, faults=faults),
        ],
        ignore_index=True,
    )

    scaler = StandardScaler().fit(df_train[config.FEATURE_COLS].to_numpy())
    X_train = scaler.transform(df_train[config.FEATURE_COLS].to_numpy())
    X_test = scaler.transform(df_test[config.FEATURE_COLS].to_numpy())

    X_train_seq, _ = make_windows(X_train, df_train[config.META_COLS], window, stride)
    X_test_seq, meta_test_seq = make_windows(X_test, df_test[config.META_COLS], window, stride)
    y_test_seq = make_labels(meta_test_seq, onset=config.ONSET_TESTING)

    return WindowedData(X_train_seq, X_test_seq, y_test_seq, meta_test_seq, scaler)


# ======================================================================================
# Semi-supervised Setting (TICKET-03, additiv) — Gutdaten + kleines Label-Budget.
# ======================================================================================
@dataclass
class SemiSupervisedData:
    """Gutdaten (unlabeled) + wenige gelabelte Anomalien + Test (für DeepSAD/Deep SVDD)."""

    X_train_good: np.ndarray      # skaliert, nur Gutdaten (unlabeled)
    X_labeled: np.ndarray         # skaliert, bekannte Anomalie-Punkte (Label-Budget)
    y_labeled: np.ndarray         # 1 = Anomalie (Konvention für dieses Setting)
    X_test: np.ndarray
    y_test: np.ndarray            # punktweises Anomalie-Label (Onset-korrekt)
    meta_test: pd.DataFrame
    scaler: StandardScaler

    def __repr__(self) -> str:
        return (
            f"SemiSupervisedData(good={self.X_train_good.shape}, "
            f"labeled={self.X_labeled.shape}, test={self.X_test.shape})"
        )


def load_semisupervised(
    faults: list[int] | None = None,
    n_train_good_runs: int = 20,
    n_labeled_fault_runs: int = 3,
    n_test_good_runs: int = 15,
    n_test_fault_runs: int = 15,
    seed: int = config.RANDOM_SEED,
) -> SemiSupervisedData:
    """Baut ein semi-supervised Set: viele Gutdaten + **kleines** gelabeltes Anomalie-Budget.

    Gelabelte Anomalien stammen aus ``faulty_training`` (nur Samples nach Onset 20, also echt
    anomal). Test wie ``load_split`` (aus den Testdateien). Scaler nur auf Gutdaten gefittet.
    """
    faults = faults if faults is not None else config.DEMO_FAULTS

    train_runs = _pick_runs(n_train_good_runs, seed=seed)
    labeled_runs = _pick_runs(n_labeled_fault_runs, seed=seed + 1)
    test_good_runs = _pick_runs(n_test_good_runs, seed=seed + 2)
    test_fault_runs = _pick_runs(n_test_fault_runs, seed=seed + 3)

    df_good = _read_runs(config.FAULT_FREE_TRAINING, train_runs)
    df_lab = _read_runs(config.FAULTY_TRAINING, labeled_runs, faults=faults)
    df_lab = df_lab[df_lab["sample"] > config.ONSET_TRAINING]  # nur echte Anomalien
    df_test = pd.concat(
        [
            _read_runs(config.FAULT_FREE_TESTING, test_good_runs),
            _read_runs(config.FAULTY_TESTING, test_fault_runs, faults=faults),
        ],
        ignore_index=True,
    )

    scaler = StandardScaler().fit(df_good[config.FEATURE_COLS].to_numpy())
    X_train_good = scaler.transform(df_good[config.FEATURE_COLS].to_numpy())
    X_labeled = scaler.transform(df_lab[config.FEATURE_COLS].to_numpy())
    y_labeled = np.ones(len(X_labeled), dtype=int)
    X_test = scaler.transform(df_test[config.FEATURE_COLS].to_numpy())
    y_test = make_labels(df_test, onset=config.ONSET_TESTING)
    meta_test = df_test[config.META_COLS].reset_index(drop=True)

    return SemiSupervisedData(X_train_good, X_labeled, y_labeled, X_test, y_test, meta_test, scaler)
