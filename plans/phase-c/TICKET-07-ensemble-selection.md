# TICKET-07: Ensemble Selection (greedy) + LSCP/FeatureBagging

**Stream:** B (Tiefere AutoML) · **Priorität:** mittel · **Aufwand:** M · **Abhängigkeiten:** – · **Status:** offen

## Kontext
AutoML-für-Anomaliedetection auf dem TEP (`uv`, Python 3.13, PyOD-Interface). Bisheriges
Ensembling: [automl_ad/automl/ensemble.py](../../automl_ad/automl/ensemble.py)
(`ensemble_scores` mit `average`/`maximization`, label-frei). Die Doku nennt zusätzlich
**Ensemble Selection** (greedy, Caruana — label-basiert, wie in auto-sklearn) und lokale
Kombination (LSCP) bzw. **Feature Bagging** (label-frei, Diversität).

## Ziel
Fortgeschrittenes Ensembling ergänzen: (1) greedy **Ensemble Selection** als Oracle-Variante
(label-basiert) und (2) **Feature Bagging** als label-freie Diversitäts-Variante; beide gegen
das bestehende `average`-Ensemble und den besten Einzeldetektor vergleichen.

## Zu erstellende/ändernde Dateien
- **NEU** `automl_ad/automl/ensemble_selection.py` —
  `greedy_ensemble_selection(scores_matrix, y_val, n_rounds=20)` (gewichtete Auswahl mit
  Zurücklegen, optimiert ROC-AUC) **und** `feature_bagging_scores(make_fn, X_train, X_test, n_estimators, max_features, seed)`
  (label-frei). Optional `pyod.models.lscp.LSCP` einbinden.
- **NEU** `notebooks/10_ensemble_selection.py` (marimo).

## Schnittstellen-Kontrakt
- `greedy_ensemble_selection(scores_matrix: np.ndarray, y_val: np.ndarray, n_rounds: int) -> np.ndarray`
  (Gewichtsvektor über Detektoren) — zur Anwendung auf Test-Score-Matrix.
- `feature_bagging_scores(...) -> np.ndarray` (Test-Scores, höher=anomaler).
- Score-Matrizen wie in `ensemble.ensemble_scores` (Detektoren × Punkte, via `standardizer`).

## Umsetzungshinweise
- Reuse `ensemble._COMBINERS`/`standardizer`-Muster für Normalisierung.
- Greedy Selection braucht ein gelabeltes Val-Set → `data.load_validation(split, source="testing")`.
- Feature Bagging: je Estimator zufällige Feature-Teilmenge; Scores normalisieren + mitteln.

## Akzeptanzkriterien
- `greedy_ensemble_selection` liefert ein Ensemble, dessen Test-ROC-AUC ≥ bestes `average`-
  Ensemble auf dem Demo-Split (oder Gleichstand, ehrlich berichten).
- `feature_bagging_scores` läuft label-frei und liefert plausible Scores.
- Notebook vergleicht: bester Einzel · average · feature-bagging · greedy-selection.

## Verifikation
```bash
uv run python notebooks/10_ensemble_selection.py    # Exit 0; reports/10_ensemble_selection_*.png
```

## Konfliktvermeidung
Reine **neue** Dateien (`automl/ensemble_selection.py`, Notebook `10`). `ensemble.py` nur
**importieren**, nicht ändern.

## Referenzen
[docs/methoden/automl-strategien/ensembling.md](../../docs/methoden/automl-strategien/ensembling.md).
