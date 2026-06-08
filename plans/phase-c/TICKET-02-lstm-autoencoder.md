# TICKET-02: LSTM-Autoencoder (zeitlicher Detektor)

**Stream:** A (Neue Detektoren) · **Priorität:** mittel · **Aufwand:** L · **Abhängigkeiten:** T01 · **Status:** offen

## Kontext
AutoML-für-Anomaliedetection auf dem TEP (`uv`, Python 3.13). Package `automl_ad/` auf
PyOD-`BaseDetector` standardisiert (`fit`/`decision_function`, höher=anomaler). Daten: 4
Parquet-Dateien unter `data/` (git-ignored), 52 Features (41 `xmeas` + 11 `xmv`). **TEP ist
eine Zeitreihe je `simulationRun`** (Abtastung alle 3 min); in `faulty_testing` ist der Fehler
erst ab `sample > 160` aktiv. Bisher sind alle Detektoren punktweise — dieser bringt die
**zeitliche** Dynamik ein. torch 2.x ist installiert.

## Ziel
Einen LSTM-Autoencoder als regulären Registry-Detektor bereitstellen: trainiert auf
Gutdaten-**Sequenzen**, Score = Rekonstruktionsfehler eines Fensters. Erkennt driftende/zeitliche
Fehler (z. B. IDV 13) besser als punktweise Methoden.

## Zu erstellende/ändernde Dateien
- **NEU** `automl_ad/detectors/lstm_autoencoder.py` — torch-LSTM-AE im PyOD-Stil
  (`fit`/`decision_function`/`decision_scores_`/`threshold_`), `FACTORIES={"lstm_ae": make_fn}`.
- **additiv** `automl_ad/data.py` — `make_windows(X, meta, window=30, stride=1)` erzeugt
  Sequenzen **innerhalb** eines `simulationRun` (nie über Lauf-Grenzen) und gibt Fenster +
  zugehörige End-Sample-Meta zurück (für punktweise Rückabbildung des Scores).
- **NEU** `notebooks/05_lstm_autoencoder.py` (marimo).

## Schnittstellen-Kontrakt
- `make_windows(X: np.ndarray, meta: pd.DataFrame, window: int, stride: int) -> tuple[np.ndarray, pd.DataFrame]`
  — `X_seq` Form `(n_windows, window, n_features)`, `meta_seq` mit der Zeile des **letzten**
  Sample je Fenster (für Onset-korrektes Mapping auf `y`).
- Detektor `make_fn(window=30, latent_dim=16, hidden=64, epoch_num=30, lr=1e-3, batch_size=256, contamination=0.1, **hp)`
  → Objekt mit PyOD-Interface; `decision_function` liefert je **Fenster** einen Score
  (Aggregation auf Punktebene über das End-Sample).
- `FACTORIES = {"lstm_ae": make_fn}` (Auto-Discovery via T01).

## Umsetzungshinweise
- Fenster-Erzeugung gruppiert nach `simulationRun` (siehe `data._read_runs`/`Split.meta_test`).
- Encoder/Decoder als `nn.LSTM`; Rekonstruktion der Eingabesequenz; MSE als Score; `fit` gibt
  `self` zurück (vgl. `_ChainableAutoEncoder` in `detectors/autoencoder.py`).
- Threshold = Quantil `1-contamination` der Trainings-Fensterfehler → `threshold_`.
- Für die Evaluation Scores auf das letzte Sample je Fenster mappen, dann mit
  `eval.metrics.summarize` (Onset `>160`) vergleichen. Subsampling (wenige Runs) für Tempo.

## Akzeptanzkriterien
- `make_detector("lstm_ae")` ist verfügbar; `fit` auf Gutdaten-Sequenzen, `decision_function`
  liefert endliche Scores.
- Auf einem Demo-Split (inkl. driftendem Fehler IDV 13) erreicht der LSTM-AE eine plausible
  ROC-AUC (> 0.6) und der Score steigt nach dem Onset sichtbar.
- `make_windows` erzeugt **keine** lauf-übergreifenden Fenster (Test mit 2 Läufen prüfen).

## Verifikation
```bash
uv run python notebooks/05_lstm_autoencoder.py     # Exit 0; schreibt reports/05_lstm_ae_*.png
```

## Konfliktvermeidung
`data.py` nur **additiv** (Funktion `make_windows` anhängen; T03 hängt parallel
`load_semisupervised` an — beide nur anhängen). Detektor self-register via `FACTORIES` (kein
base.py-Edit, dank T01). Eigenes Notebook `05`.

## Referenzen
[docs/methoden/ad-methoden/autoencoder.md](../../docs/methoden/ad-methoden/autoencoder.md),
[01_datensatz_tep.md](../../docs/methoden/01_datensatz_tep.md) (Zeitstruktur/Onset),
[02_evaluationsprotokoll.md](../../docs/methoden/02_evaluationsprotokoll.md).
