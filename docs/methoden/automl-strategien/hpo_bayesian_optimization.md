# AutoML-Strategie 1: HPO-Engines & Bayesian Optimization

> **Intuition zuerst** — dieser Block erklärt die Strategie in Alltagssprache. Die formale
> Behandlung folgt ab Abschnitt 1.

### In einem Satz

Statt die **Stellschrauben** eines Modells von Hand zu drehen, übernimmt ein Optimierer die
*intelligente* Suche — und lernt aus jedem Versuch, wo es sich lohnt weiterzusuchen.

### Das Bild im Kopf

Du justierst eine **Maschine mit vielen Reglern, in die du nicht hineinsehen kannst**, und jeder
Testlauf ist teuer.

- **Grid/Random Search:** Reglerstellungen blind durchprobieren — erstaunlich brauchbar als
  Baseline, aber verschwenderisch.
- **Bayesian Optimization:** Du baust dir ein **mentales Modell** „in dieser Reglerregion lief es
  zuletzt gut — probier nebenan weiter" und steckst die nächsten Tests gezielt dorthin. Das
  *Surrogat* ist dieses Gedächtnis, die *Acquisition Function* die Entscheidung „wo als Nächstes".
- **Multi-Fidelity (Hyperband/BOHB):** wie ein **Casting in Runden** — alle Kandidaten bekommen
  zuerst eine *kurze* Probe (wenig Daten/Epochen), nur die Vielversprechenden kommen in die
  Vollprobe. So verschwendest du teure Rechenzeit nicht an offensichtliche Nieten.

### Wann sinnvoll – und wann nicht

| Stark, wenn … | Heikel/schwach, wenn … |
|---|---|
| das Modell **sensibel** auf HP reagiert (OC-SVM, AE) | im **unüberwachten** Fall — es fehlt die Zielmetrik! (siehe unten) |
| jeder Trainingslauf teuer ist (→ Multi-Fidelity) | Defaults ohnehin schon stark sind (Gewinn klein) |
| du den Gewinn vs. Defaults *messbar* zeigen willst | der Suchraum trivial klein ist |

### So liest und erklärst du das Ergebnis

- **Die Schlüsselfolie ist „Default vs. Random vs. BO vs. BOHB":** Wie viel ROC-AUC kaufte die
  *kluge* Suche gegenüber den Defaults?
- **Ehrliches TEP-Ergebnis:** Die HPO-Gewinne waren **moderat**, weil die Defaults der Detektoren
  schon stark sind. Das ist *kein Misserfolg*, sondern ein häufiger und berichtenswerter Befund —
  „starke Defaults" ist selbst eine Erkenntnis.
- **Der entscheidende Haken in der AD:** Im **unüberwachten** Fall gibt es **kein direktes
  Gütesignal**, an dem der Optimierer „besser/schlechter" festmachen könnte. Dann braucht man eine
  interne Metrik, das Oracle (nur als Obergrenze!) — oder man verzichtet auf HPO und setzt auf
  robuste Defaults/Ensembles. **Diese Einschränkung in der Präsentation klar benennen.**

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
