# Phase C — Tickets (Übersicht)

Erweiterung & Härtung des AutoML-für-Anomaliedetection-Projekts (TEP). Jedes Ticket ist
**self-contained** und kann von **einem** Agent unabhängig bearbeitet werden. Bitte vor Start
die Konventionen unten und das jeweilige Ticket vollständig lesen.

## Status-Tabelle

| Ticket | Titel | Stream | Aufwand | Deps | Status |
|---|---|---|---|---|---|
| [T01](TICKET-01-detector-auto-discovery.md) | Detector-Auto-Discovery | Enabler | S | – | **erledigt** |
| [T02](TICKET-02-lstm-autoencoder.md) | LSTM-Autoencoder (zeitlich) | A | L | T01 | **erledigt** |
| [T03](TICKET-03-deepsad-semisupervised.md) | DeepSAD / Deep SVDD (semi-sup.) | A | L | T01 | **erledigt** |
| [T04](TICKET-04-som-detector.md) | SOM-Detektor | A | M | T01 | **erledigt** |
| [T05](TICKET-05-multifidelity-hpo.md) | Multi-Fidelity-HPO (Hyperband/BOHB) | B | M | – | **erledigt** |
| [T06](TICKET-06-internal-metrics.md) | Interne Validierungsmetriken | B | M | – | **erledigt** |
| [T07](TICKET-07-ensemble-selection.md) | Ensemble Selection (greedy) + LSCP | B | M | – | **erledigt** |
| [T08](TICKET-08-ruff-ci.md) | Lint (ruff) + CI | C | S | – | **erledigt** |
| [T09](TICKET-09-pytest-suite.md) | pytest-Testsuite | C | M | – | **erledigt** |
| [T10](TICKET-10-data-fetch-script.md) | Daten-Fetch/Convert-Skript | C | S | – | **erledigt** |
| [T11](TICKET-11-asset-generator.md) | Asset-/Ergebnis-Generator | D | M | weich A/B | **erledigt** |
| [T12](TICKET-12-eda-marimo-port.md) | EDA → marimo-Port | D | S | – | **erledigt** |
| [T13](TICKET-13-per-fault-deepdive.md) | Per-Fault-Deep-Dive | D | M | weich A | **erledigt** |

## Abhängigkeitsgraph

```
T01 ──► T02, T03, T04          (neue Detektoren = Drop-in nach dem Discovery-Refactor)
T05 … T13                       unabhängig, sofort parallel startbar
weich: T11 / T13 lesen die REGISTRY dynamisch → neue Detektoren erscheinen automatisch,
       sobald T02–T04 gemergt sind (keine harte Blockade).
```

**Empfohlene Reihenfolge:** zuerst **T01** mergen (entkoppelt Stream A). Alles andere parallel.

## Konventionen (verbindlich)

- **Sprache:** Deutsch in Doku/Kommentaren, Englisch für Code/Variablen.
- **Additiv-Regel:** neue Dateien bevorzugen; *shared files* nur **anhängen**, nie Bestehendes
  umschreiben.
- **PyOD-Interface:** jeder Detektor erfüllt `fit(X)` / `decision_function(X)→score`
  (höher = anomaler) / `decision_scores_` / `threshold_`.
- **Demos:** je Ticket ein **neues** marimo-Notebook (reservierte Nummer, s. u.); headless
  ausführbar via `uv run python notebooks/NN_*.py` (Exit 0), Artefakte nach `reports/`.
- **Keine Regression:** bestehende Notebooks (`01`–`04`) und `report.py` müssen weiterhin
  durchlaufen.

## Shared-File-Ownership

| Datei | Eigentümer | Regel für andere |
|---|---|---|
| `automl_ad/detectors/base.py` | T01 | danach niemand — Detektoren self-register via `FACTORIES` |
| `automl_ad/data.py` | additiv: T02 (`make_windows`), T03 (`load_semisupervised`) | nur anhängen |
| `automl_ad/report.py` | T11 | andere editieren es nicht |
| `pyproject.toml` | additiv: T04 (minisom), T08 (ruff) | je via `uv add` / eigener `[tool.*]`-Block |
| `automl_ad/eval/plots.py` | additiv (neue Funktionen) | sonst Plot im eigenen Notebook |
| `notebooks/*` | je Ticket eigene neue Nummer | bestehende nie editieren |

## Reservierte Notebook-Nummern

`00` EDA-Port (T12) · `05` LSTM-AE (T02) · `06` DeepSAD (T03) · `07` SOM (T04) ·
`08` Multi-Fidelity (T05) · `09` Interne Metriken (T06) · `10` Ensemble-Selection (T07) ·
`11` Per-Fault-Deep-Dive (T13).

## Umsetzungsnotizen (für nachfolgende Tickets relevant)

- **T01 erledigt:** `detectors/base.py` baut die Registry per Auto-Discovery (Module mit
  `FACTORIES`-Dict). Neue Detektoren = Drop-in. Registry aktuell:
  `autoencoder, deep_sad, deep_svdd, ecod, iforest, lstm_ae, ocsvm, pca, som`.
- **Achtung Detektor-Heterogenität:** `lstm_ae` erwartet **3D-Sequenzen**
  (`data.make_windows`/`load_windowed`); `deep_sad` erwartet ein **Label-Budget** (`fit(X, y)`).
  Tickets, die blind über `available_detectors()` iterieren und 2D-`fit(X)` aufrufen
  (insb. **T09** Tests, **T11** Asset-Generator, **T13** Deep-Dive), müssen diese filtern —
  z. B. nur tabellarische, unüberwachte Detektoren
  (`ecod, iforest, ocsvm, pca, autoencoder, som, deep_svdd`).
- **Neue Daten-Loader (additiv in `data.py`):** `make_windows`, `load_windowed`,
  `load_semisupervised` (+ Dataclasses `WindowedData`, `SemiSupervisedData`).
- **Neue Dependencies:** `deepod` (DeepSAD), `minisom` (SOM) in den Haupt-Dependencies (damit
  Notebooks out-of-the-box laufen).

## Definition of Done (jedes Ticket)

1. Alle Akzeptanzkriterien erfüllt.
2. Verifikationsbefehl läuft headless mit Exit 0 und erzeugt das genannte Artefakt.
3. Keine Regression der bestehenden Notebooks/Reports.
4. Status in dieser Tabelle auf „erledigt" gesetzt; ggf. README/Doku-Verweis ergänzt.
