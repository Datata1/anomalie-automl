# TICKET-11: Asset-/Ergebnis-Generator (PowerPoint-Material)

**Stream:** D (Präsentation) · **Priorität:** mittel · **Aufwand:** M · **Abhängigkeiten:** weich A/B (nicht blockierend) · **Status:** offen

## Kontext
AutoML-für-Anomaliedetection auf dem TEP (`uv`, Python 3.13). Es gibt bereits
[automl_ad/report.py](../../automl_ad/report.py) (`build`, `main`), das eine
Vergleichstabelle (`reports/summary_table.md`) und die zentrale „label-frei vs. Oracle"-Grafik
erzeugt. Für die **PowerPoint** soll daraus ein vollständiger, reproduzierbarer Asset-Generator
werden, der **alle** finalen Figuren + eine maschinenlesbare Ergebnis-CSV ausgibt.

## Ziel
`report.py` zum One-Stop-Generator ausbauen: iteriert die **REGISTRY dynamisch** (schließt neue
Detektoren aus Stream A automatisch ein), erzeugt `reports/results.csv`
(Methode × Strategie × Metriken) und alle Präsentationsfiguren in einheitlichem Stil.

## Zu erstellende/ändernde Dateien
- **ändern (Eigentümer)** `automl_ad/report.py` — `build()` so erweitern, dass es über
  `automl_ad.detectors.available_detectors()` iteriert; zusätzlich `write_results_csv(rows, path)`;
  `main()` erzeugt CSV + alle Figuren. Optional dünner CLI-Wrapper.
- ggf. **additiv** `automl_ad/eval/plots.py` — wiederverwendbare Figur-Helfer (nur anhängen).

## Schnittstellen-Kontrakt
- `uv run python -m automl_ad.report` erzeugt: `reports/results.csv`, `reports/summary_table.md`,
  `reports/summary_label_free_vs_oracle.png` und weitere benannte Figuren — reproduzierbar
  (fixe Seeds aus `config.RANDOM_SEED`).
- `results.csv`-Spalten mindestens: `method, strategy, roc_auc, pr_auc, f1, detection_delay, false_alarm_rate`.

## Umsetzungshinweise
- Reuse `eval.summarize`, `eval.plots.comparison_bars/grouped_bars`. Detektoren über die
  Registry holen (nicht hartkodieren) → Stream-A-Detektoren erscheinen automatisch.
- Laufzeit deckeln: moderate Subsamples; teure Detektoren (OCSVM/AE) optional via Flag.
- CSV mit `csv`/`pandas` schreiben; keine neue Schwergewichts-Dependency.

## Akzeptanzkriterien
- Ein einziger Befehl erzeugt CSV + alle Figuren ohne Fehler.
- `results.csv` enthält je verfügbarer (Methode × Strategie)-Kombination eine Zeile; neue
  Detektoren (falls bereits gemergt) tauchen automatisch auf.
- Bestehende Notebooks bleiben unberührt/lauffähig.

## Verifikation
```bash
uv run python -m automl_ad.report
ls reports/results.csv reports/summary_*.png
```

## Konfliktvermeidung
**Alleiniger Eigentümer von `report.py`.** Andere Tickets editieren `report.py` nicht. Plot-
Helfer nur **additiv** in `plots.py`.

## Referenzen
[docs/methoden/02_evaluationsprotokoll.md](../../docs/methoden/02_evaluationsprotokoll.md)
(Reporting-Schema), [00_modellselektion_ohne_labels.md](../../docs/methoden/automl-strategien/00_modellselektion_ohne_labels.md).
