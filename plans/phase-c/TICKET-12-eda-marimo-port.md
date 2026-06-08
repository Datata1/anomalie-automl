# TICKET-12: EDA → marimo-Port

**Stream:** D (Präsentation) · **Priorität:** niedrig · **Aufwand:** S · **Abhängigkeiten:** – · **Status:** offen

## Kontext
AutoML-für-Anomaliedetection auf dem TEP (`uv`, Python 3.13). Die explorative Datenanalyse
liegt aktuell als Jupyter-Notebook [eda_dataset.ipynb](../../eda_dataset.ipynb) vor (Korrelations-
Heatmap, Zeitreihen-Plots, PCA-2D). Das Projekt nutzt ansonsten durchgehend **marimo** statt
Jupyter. Eine marimo-Version fehlt.

## Ziel
Die EDA als reaktive marimo-App nachbauen, die `automl_ad`-Bausteine wiederverwendet — als
konsistenter, git-freundlicher Einstieg. Das bestehende Jupyter-Notebook bleibt erhalten.

## Zu erstellende/ändernde Dateien
- **NEU** `notebooks/00_eda.py` (marimo) — lädt ein Daten-Subsample über
  `automl_ad.data._read_runs` / `automl_ad.config`, zeigt: Korrelations-Heatmap der Features,
  Zeitreihe ausgewählter Sensoren (mit Onset-Markierung), PCA-2D der Fehlertypen,
  Datensatz-Steckbrief (Spalten/Onset/Fehlertypen).

## Schnittstellen-Kontrakt
- Headless ausführbar (`uv run python notebooks/00_eda.py`, Exit 0); speichert mind. eine
  EDA-Figur nach `reports/` (z. B. `reports/00_eda_correlation.png`).
- Nur **lesende** Nutzung bestehender Module; keine Änderungen an `automl_ad`.

## Umsetzungshinweise
- Reuse `config.FEATURE_COLS`, `config.ONSET_TESTING`, `data._read_runs` für gezieltes Laden
  weniger Läufe (kein Vollscan der 9,6-Mio-Zeilen-Datei).
- Visualisierung mit matplotlib/seaborn/plotly (bereits Dependencies); Stil wie `eval/plots.py`.
- Inhaltlich an [01_datensatz_tep.md](../../docs/methoden/01_datensatz_tep.md) orientieren.

## Akzeptanzkriterien
- `notebooks/00_eda.py` läuft headless (Exit 0) und erzeugt mindestens eine EDA-Figur.
- Korrelations-Heatmap, eine Zeitreihe mit Onset-Markierung und ein PCA-2D-Plot sind enthalten.

## Verifikation
```bash
uv run python notebooks/00_eda.py        # Exit 0; reports/00_eda_*.png
```

## Konfliktvermeidung
Reines **neues** Notebook `00`. Keine Änderung an `automl_ad` oder am Jupyter-Notebook.

## Referenzen
[eda_dataset.ipynb](../../eda_dataset.ipynb),
[docs/methoden/01_datensatz_tep.md](../../docs/methoden/01_datensatz_tep.md).
