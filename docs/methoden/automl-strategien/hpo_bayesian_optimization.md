# AutoML-Strategie 1: HPO-Engines & Bayesian Optimization

## 1. Idee & Problembezug

Löst **Hyperparameter-Optimierung (HPO)** und optional **CASH** (Modell + HP gemeinsam):
Finde im Suchraum Λ die Konfiguration, die eine Zielmetrik optimiert. Statt jedes Modell
von Hand zu tunen, übernimmt ein **Optimizer** die intelligente Suche. Auf AD angewandt:
tune `nu/gamma` der OC-SVM, `max_samples` des Isolation Forest, die AE-Architektur etc.

## 2. Funktionsweise / Algorithmus

**Suchraum:** gemischt — float (`learning_rate`), int (`n_estimators`), kategorisch
(`kernel`), **hierarchisch/bedingt** (AE-Tiefe nur relevant, wenn DL gewählt). Kein
gradientenbasiertes Verfahren möglich.

**Naive Optimierer:**
- *Grid Search* — exponentiell teuer, da pro Punkt ein Modelltraining; meist unpraktikabel.
- *Random Search* — überraschend stark, einfache Baseline; Prior über Parameter möglich.

**Sequential Model-Based Optimization (SMBO / Bayesian Optimization):**
1. **Surrogate-Modell** `f̂: λ → cost` lernt den Zusammenhang Konfiguration → Kosten aus den
   bisherigen Auswertungen (billiger als echtes Training).
   - *Gaussian Process (GPR):* klein-dimensionale, numerische Räume; kann nicht gut mit
     kategorischen Größen.
   - *Random Forest:* hierarchische, hochdimensionale Räume (Engine **SMAC3**).
   - *TPE* (Tree-structured Parzen Estimator): probabilistisch, Standard in **Optuna/Hyperopt**.
2. **Acquisition Function** wählt die nächste Konfiguration, meist **Expected Improvement**:
   `EI_n(x) = E[ max(f(x) − f*_n, 0) ]`, balanciert Exploration vs. Exploitation. Praktisch:
   viele zufällige Kandidaten ziehen, EI auswerten, beste wählen.

**Multi-Fidelity-Optimierung** (billige Teilauswertungen):
- *Budget* = Daten-Subsample **oder** Trainings-Epochen.
- **Successive Halving:** viele Konfigs auf kleinem Budget, schlechtere Hälfte je Runde
  verwerfen, Rest auf größerem Budget.
- **Hyperband:** mehrere SH-Brackets (Exploration ↔ Exploitation balanciert).
- **BOHB:** Hyperband + Bayesian Optimization (nächste Konfig per Acquisition statt zufällig).

## 3. Anbindung an AD

- **Zielmetrik im überwachten/semi-supervised Fall:** ROC-AUC/PR-AUC auf gelabeltem
  Validierungsset (Standard-Optuna-Objective).
- **Sonderfall fehlende Labels (unüberwacht):** Es gibt **kein** direktes Gütesignal! Optionen:
  1. **interne Metrik** (EM/MV, IREOS/SIREOS, Konsens) als Objective,
  2. **TEP-Oracle** mit Labels (nur als Obergrenze kennzeichnen),
  3. HPO ganz vermeiden und auf robuste Defaults/Ensembles setzen.
  Siehe [00_modellselektion_ohne_labels.md](00_modellselektion_ohne_labels.md). **Diese
  Einschränkung explizit in der Präsentation benennen.**
- **Natürliche Budgets für TEP:** Anzahl `simulationRun`s (Daten) oder AE-Epochen — beide
  perfekt für Hyperband/BOHB.

## 4. Tools / Libraries

| Tool | Optimizer | Multi-Fidelity | Hinweis |
|---|---|---|---|
| **Optuna** | TPE, CMA-ES, GP | Hyperband, SH (Pruner) | empfohlener Default; einfache API; `uv add optuna` |
| **SMAC3** | RF-SMBO | BOHB/Hyperband | für komplexe hierarchische Räume; Py-Version prüfen |
| **Hyperopt** | TPE | – | älter, weit verbreitet |
| **Ray Tune** | viele | ASHA, BOHB | für Parallelisierung/Cluster |

Minimal-Beispiel (Optuna auf Isolation Forest, Objective = ROC-AUC auf Val-Labels):

```python
import optuna
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

def objective(trial):
    m = IsolationForest(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_samples=trial.suggest_int("max_samples", 128, 1024),
        max_features=trial.suggest_float("max_features", 0.5, 1.0),
        random_state=0,
    ).fit(X_good_scaled)
    score = -m.score_samples(X_val_scaled)
    return roc_auc_score(y_val, score)            # label-frei: interne Metrik einsetzen

study = optuna.create_study(direction="maximize",
                            pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=50)
print(study.best_params)
```

## 5. Stärken / Grenzen für TEP

- **Stärken:** größter Hebel bei sensitiven Methoden (OC-SVM, AE); Multi-Fidelity macht die
  9,6-Mio-Zeilen-Daten beherrschbar; klar messbarer Gewinn vs. Defaults.
- **Grenzen:** unüberwachte Zielmetrik ist das Kernrisiko; jede Trial = ein Modelltraining →
  Budget bewusst über Subsampling steuern.

## 6. Demonstrierte Capability & Referenzen

**Capability:** Kernstück „klassisches AutoML" — Suchraum, Surrogate, Acquisition,
Multi-Fidelity, anschaulich auf AD. Vergleich Default vs. Random vs. BO vs. BOHB als
Schlüsselfolie.

**Referenzen:** Hutter/Kotthoff/Vanschoren (AutoML-Buch); Li et al. (2017, Hyperband);
Falkner et al. (2018, BOHB); Akiba et al. (2019, Optuna); SMAC3 (Lindauer et al. 2022). Siehe
[../referenzen.md](../referenzen.md).
