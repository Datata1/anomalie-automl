# AutoML für Anomaliedetection (Tennessee Eastman Process)

Recherche- und Beispielprojekt: **Welche AutoML-Methoden gibt es für die Anomaliedetection?**,
demonstriert am TEP-Datensatz. Drei Lern-Settings (unsupervised / semi-supervised / supervised)
× vier AutoML-Strategien (HPO, fertige Frameworks, Meta-Learning, Ensembling).

- **Methoden-Doku:** [`docs/methoden/`](docs/methoden/) — startet mit
  [`00_uebersicht.md`](docs/methoden/00_uebersicht.md).
- **Code:** [`automl_ad/`](automl_ad/) — schlanke Pipeline auf dem PyOD-`BaseDetector`-Interface.
- **Demos:** [`notebooks/`](notebooks/) — marimo-Apps (reine `.py`).

## Setup

```bash
uv sync                     # installiert Abhängigkeiten + Projekt (editierbar)
```

Die vier TEP-Parquet-Dateien müssen unter `data/` liegen (siehe
[`docs/methoden/01_datensatz_tep.md`](docs/methoden/01_datensatz_tep.md) §6). Konvertierung
aus den `.RData`-Quellen via [`export_rdata.R`](export_rdata.R).

## Demos ausführen

Interaktiv (reaktive marimo-Oberfläche):

```bash
uv run marimo edit notebooks/01_baselines.py
```

Headless (läuft als Skript durch, schreibt Plots nach `reports/`):

```bash
uv run python notebooks/01_baselines.py             # Baseline-Detektoren
uv run python notebooks/02_hpo.py                   # HPO + Modellselektion (Oracle vs. label-frei)
uv run python notebooks/03_frameworks_supervised.py # FLAML + Klassifikation vs. AD
uv run python notebooks/04_selection_ensemble_metaod.py  # Ensembling + Meta-Learning
```

## Externe Dokumentation

- [Google Docs (Bericht & PowerPoint)](https://docs.google.com/document/d/1xCzNuHb4aJFeL6dh9fqxzVupTzS7ex81C6WqlYu80aE/edit?usp=sharing)
- [Google Drive (Daten)](https://drive.google.com/drive/folders/1p-McaULO5VP9qN01V3a_34H8y4cIHShN?usp=sharing)
