# TICKET-06: Interne Validierungsmetriken (label-freie Selektion)

**Stream:** B (Tiefere AutoML) · **Priorität:** mittel · **Aufwand:** M · **Abhängigkeiten:** – · **Status:** offen

## Kontext
AutoML-für-Anomaliedetection auf dem TEP (`uv`, Python 3.13, PyOD-Interface). Das **Kernproblem**
des Projekts: im unüberwachten Fall fehlen Labels zur Modellselektion. Bisher gibt es nur die
Konsens-Heuristik [automl_ad/automl/selection.py](../../automl_ad/automl/selection.py)
(`select_internal`, Model-Centrality) und den label-basierten `select_oracle`. Die Doku nennt
zusätzlich **interne Metriken** (Excess-Mass/Mass-Volume, IREOS/SIREOS).

## Ziel
Echte interne (label-freie) Validierungsmetriken implementieren und als Selektionskriterium
nutzbar machen; empirisch zeigen, **wie gut** sie mit der echten ROC-AUC korrelieren (laut
Literatur oft schwach — ehrlich darstellen) und wie sie sich zu Konsens und Oracle verhalten.

## Zu erstellende/ändernde Dateien
- **NEU** `automl_ad/automl/internal_metrics.py` — mindestens **eine** robuste interne Metrik
  (empfohlen: **SIREOS** — schnell; oder Excess-Mass/Mass-Volume), plus
  `select_by_internal(candidates, X_train, X_eval) -> tuple[str, dict[str, float]]`.
- **NEU** `notebooks/09_internal_metrics.py` (marimo) — Scatter/Korrelation interne Metrik ↔
  echte ROC-AUC über mehrere Detektoren; Balken Konsens vs. intern vs. Oracle.

## Schnittstellen-Kontrakt
- `internal_score(X_eval, scores) -> float` (höher = besser; Konvention im Modul dokumentieren).
- `select_by_internal(candidates, X_train, X_eval) -> (best_name, {name: internal_score})`,
  Signatur analog zu `selection.select_internal`.
- Kandidatenformat wie `selection.DEFAULT_CANDIDATES` (`list[tuple[str, dict]]`).

## Umsetzungshinweise
- Reuse `selection._fit_scores` (oder analog) zum Erzeugen der Detektor-Scores.
- SIREOS/EM-MV sind rechenintensiv → auf Subsample der Testpunkte auswerten.
- Vergleichsanker: echte ROC-AUC via `eval.metrics.summarize` (nur zur **Auswertung** der
  Metrikgüte, nicht zur Selektion!).

## Akzeptanzkriterien
- `select_by_internal` wählt label-frei einen Detektor aus den Kandidaten.
- Notebook zeigt die Korrelation (Spearman/Pearson) der internen Metrik mit der echten ROC-AUC
  und vergleicht die Test-ROC-AUC der intern/konsens/oracle gewählten Detektoren.

## Verifikation
```bash
uv run python notebooks/09_internal_metrics.py      # Exit 0; reports/09_internal_metrics_*.png
```

## Konfliktvermeidung
Reine **neue** Dateien (`automl/internal_metrics.py`, Notebook `09`). `selection.py` nur
**importieren**, nicht ändern.

## Referenzen
[docs/methoden/automl-strategien/00_modellselektion_ohne_labels.md](../../docs/methoden/automl-strategien/00_modellselektion_ohne_labels.md),
[meta_learning_modellselektion.md](../../docs/methoden/automl-strategien/meta_learning_modellselektion.md).
