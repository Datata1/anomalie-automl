# TICKET-09: pytest-Testsuite

**Stream:** C (Infrastruktur) · **Priorität:** mittel · **Aufwand:** M · **Abhängigkeiten:** – · **Status:** offen

## Kontext
AutoML-für-Anomaliedetection auf dem TEP (`uv`, Python 3.13). Package `automl_ad/` mit
`data.py` (Run-Level-Split, Onset-Labels), `eval/metrics.py` (ROC/PR/Delay/FAR),
`detectors/` (Registry), `automl/selection.py`. Es gibt **keine Tests**. Wichtige Invarianten
(kein Leakage, korrektes Onset-Labeling, Detektor-Kontrakt) sind ungeprüft.

## Ziel
Eine pytest-Suite, die die Kern-Invarianten absichert — schnell, mit kleinsten Daten-Subsamples,
und sauber übersprungen, wenn die Parquet-Dateien fehlen (CI-tauglich).

## Zu erstellende/ändernde Dateien
- **NEU** `tests/conftest.py` — Marker `data` registrieren; Fixture, die die Existenz der
  Parquet-Dateien prüft und sonst `pytest.skip`.
- **NEU** `tests/test_data.py` — Run-Level-Disjunktheit (Train/Val/Test), Onset-Labels
  (pre-Onset faulty = 0), Scaler nur auf Gutdaten, Shapes von `load_split`/`load_validation`/
  `load_supervised`.
- **NEU** `tests/test_metrics.py` — `summarize` auf **synthetischen** Scores (kein Datenzugriff,
  daher immer lauffähig): perfekte Trennung → ROC-AUC≈1; Detection-Delay/FAR-Grenzfälle.
- **NEU** `tests/test_detectors.py` — für jeden Registry-Detektor `fit`/`decision_function`-
  Kontrakt + endliche Scores auf Mini-Subsample (Marker `data`).
- **NEU** `tests/test_selection.py` — `select_oracle`/`select_internal` geben gültige
  Kandidaten zurück (Marker `data` für den datenbasierten Teil).

## Schnittstellen-Kontrakt
- Tests ohne Daten (`test_metrics.py`) laufen **immer**; datenabhängige sind mit
  `@pytest.mark.data` markiert und werden via `pytest -m "not data"` (CI) übersprungen.
- `pytest` als dev-dep (`uv add --dev pytest`).

## Umsetzungshinweise
- Kleinste Subsamples (`n_train_good_runs=3`, `faults=[1,3]`, wenige Test-Runs) für Tempo.
- Synthetische Metrik-Tests: z. B. `y=[0,0,1,1]`, `scores=[0.1,0.2,0.8,0.9]`.
- Detektor-Loop über `automl_ad.detectors.available_detectors()`, **aber** heterogene
  Detektoren filtern: `lstm_ae` braucht 3D-Sequenzen (`data.load_windowed`), `deep_sad` ein
  Label-Budget (`fit(X, y)`). Für den 2D-`fit(X)`-Kontrakt-Test nur die tabellarischen,
  unüberwachten Detektoren nehmen: `ecod, iforest, ocsvm, pca, autoencoder, som, deep_svdd`
  (oder `lstm_ae`/`deep_sad` separat mit passenden Eingaben testen).

## Akzeptanzkriterien
- `uv run pytest -m "not data"` läuft ohne Parquet grün.
- `uv run pytest` (mit Daten) läuft vollständig grün.
- Mind. je ein Test für Leakage-Freiheit, Onset-Labeling, Metrik-Korrektheit, Detektor-Kontrakt.

## Verifikation
```bash
uv run pytest -m "not data" -q
uv run pytest -q                 # mit lokalen Parquet-Daten
```

## Konfliktvermeidung
Komplett **neues** Verzeichnis `tests/`. Berührt keinen Produktivcode. Marker `data` wird von
T08 (CI) genutzt — Name abstimmen (hier definiert).

## Referenzen
[docs/methoden/02_evaluationsprotokoll.md](../../docs/methoden/02_evaluationsprotokoll.md),
[01_datensatz_tep.md](../../docs/methoden/01_datensatz_tep.md).
