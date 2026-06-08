# TICKET-04: SOM-Detektor (Self-Organizing Map)

**Stream:** A (Neue Detektoren) · **Priorität:** niedrig · **Aufwand:** M · **Abhängigkeiten:** T01 · **Status:** offen

## Kontext
AutoML-für-Anomaliedetection auf dem TEP (`uv`, Python 3.13, PyOD-`BaseDetector`-Interface).
Daten unter `data/` (52 Features, Onset `>160` im Test). Die SOM ist in der Vorlesung enthalten
und liefert zusätzlich eine **2D-Karten-Visualisierung** des Normalzustands — gut für die
PowerPoint. Sie ist als Stretch/Anschauungsmodell gedacht.

## Ziel
Eine Self-Organizing Map als Registry-Detektor (PyOD-konform) bereitstellen; Score = mittlere
Distanz zu den k nächsten Neuronen. Plus eine Karten-Visualisierung im Notebook.

## Zu erstellende/ändernde Dateien
- `uv add --optional som minisom` (eigener optionaler Extra; additiv in `pyproject.toml`).
- **NEU** `automl_ad/detectors/som.py` — Wrapper um `MiniSom` mit PyOD-Interface,
  `FACTORIES={"som": make_fn}`. Guarded Import (T01 toleriert fehlendes minisom).
- **NEU** `notebooks/07_som.py` (marimo) inkl. BMU-/Distanz-Karte.

## Schnittstellen-Kontrakt
- `make_fn(map_size=15, sigma=1.0, learning_rate=0.5, num_iteration=10000, k=3, contamination=0.1, **hp)`
  → PyOD-konformes Objekt.
- `fit(X)` trainiert die SOM (optional Pruning seltener Neuronen), setzt `decision_scores_`
  (Trainings-Scores) und `threshold_` (Quantil `1-contamination`); gibt `self` zurück.
- `decision_function(X)` → mittlere Distanz zu k-NN-Neuronen (höher=anomaler).

## Umsetzungshinweise
- Skalierung erfolgt upstream (Scaler in `data.load_split`); SOM erwartet bereits skalierte X.
- Distanz-Scoring: Gewichtsmatrix `som.get_weights().reshape(-1, n_features)`, Norm zu
  k nächsten (vgl. Code-Skizze in der Doku).
- `fit` muss `self` zurückgeben (Konsistenz mit übrigen Detektoren).

## Akzeptanzkriterien
- `make_detector("som")` verfügbar (wenn minisom installiert); liefert plausible Scores
  (ROC-AUC > 0.6 auf Demo-Split mit leichten Fehlern).
- Notebook erzeugt eine SOM-Karten-Grafik (z. B. Treffer-/Distanz-Heatmap).

## Verifikation
```bash
uv run python notebooks/07_som.py        # Exit 0; reports/07_som_*.png
```

## Konfliktvermeidung
Self-register via `FACTORIES` (kein base.py-Edit, dank T01). `pyproject.toml` nur additiv
(eigener optionaler Extra `som`). Eigenes Notebook `07`.

## Referenzen
[docs/methoden/ad-methoden/som.md](../../docs/methoden/ad-methoden/som.md).
