# TICKET-05: Multi-Fidelity-HPO (Hyperband/BOHB)

**Stream:** B (Tiefere AutoML) · **Priorität:** mittel · **Aufwand:** M · **Abhängigkeiten:** – · **Status:** offen

## Kontext
AutoML-für-Anomaliedetection auf dem TEP (`uv`, Python 3.13). Bestehendes HPO:
[automl_ad/automl/hpo.py](../../automl_ad/automl/hpo.py) nutzt Optuna/TPE mit
`run_optuna(name, X_train_good, X_val, y_val, n_trials, metric)` und `suggest_params(trial, name)`.
Validierung kommt aus `data.load_validation(split, source="testing")` (test-disjunkt). Bisher
fehlt die **Multi-Fidelity**-Dimension (Budget = Epochen/Daten-Subsample), die in der Doku
zentral ist (Successive Halving, Hyperband, BOHB).

## Ziel
Multi-Fidelity-Optimierung demonstrieren: Pruning wenig vielversprechender Konfigurationen über
ein **Budget** (AE-`epoch_num` bzw. Trainings-Subsample-Größe). Vergleich Random vs. BO/TPE vs.
Hyperband nach Bestwert, Anzahl ausgewerteter Trials und Wandzeit.

## Zu erstellende/ändernde Dateien
- **NEU** `automl_ad/automl/multifidelity.py` — Optuna-Study mit `HyperbandPruner`
  (bzw. `SuccessiveHalvingPruner`); Objective mit `trial.report(value, step)` +
  `trial.should_prune()` über Budgetstufen. Funktion
  `run_multifidelity(name, X_train_good, X_val, y_val, max_budget, n_trials, reduction_factor=3)`.
- **NEU** `notebooks/08_multifidelity_hpo.py` (marimo) — Vergleichsplot Random/BO/Hyperband.

## Schnittstellen-Kontrakt
- `run_multifidelity(...) -> tuple[dict, optuna.Study]` (best_params, study).
- Budget-Semantik im Modul dokumentieren: für `autoencoder`/`lstm_ae` = Epochen
  (inkrementelles Training + `report`); für klassische Detektoren = Trainings-Subsample-Größe.
- Reuse `hpo.suggest_params` für den Konfigurationsraum.

## Umsetzungshinweise
- Für AE: zwischen Budgetstufen weitertrainieren und Zwischen-ROC-AUC auf `X_val` reporten;
  bei klassischen Detektoren je Stufe auf wachsendem Subsample neu fitten.
- Wandzeit/Trial-Anzahl je Methode messen und plotten (zeigt Multi-Fidelity-Effizienz).
- OCSVM ist teuer → kleine Subsamples wählen (vgl. Notebook 02: Val auf ~4000 Punkte cappen).

## Akzeptanzkriterien
- `run_multifidelity` läuft auf mindestens einem Detektor und prunt nachweislich Trials
  (Study enthält `PRUNED`-Trials).
- Notebook zeigt einen Vergleich Random vs. BO vs. Hyperband (Bestwert über Zeit/Budget).

## Verifikation
```bash
uv run python notebooks/08_multifidelity_hpo.py     # Exit 0; reports/08_multifidelity_*.png
```

## Konfliktvermeidung
Reine **neue** Dateien (`automl/multifidelity.py`, Notebook `08`). `hpo.py` wird nur
**importiert**, nicht geändert.

## Referenzen
[docs/methoden/automl-strategien/hpo_bayesian_optimization.md](../../docs/methoden/automl-strategien/hpo_bayesian_optimization.md).
