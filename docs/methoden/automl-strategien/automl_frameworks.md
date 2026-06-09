# AutoML-Strategie 2: Fertige AutoML-Frameworks

> **Intuition zuerst** — dieser Block erklärt die Strategie in Alltagssprache. Die formale
> Behandlung folgt ab Abschnitt 1.

### In einem Satz

End-to-End-Frameworks lösen die ganze Kette — **Vorverarbeitung, Modellwahl, Tuning,
Ensembling** — auf Knopfdruck, aber praktisch nur für die **überwachte** Aufgabe.

### Das Bild im Kopf

Ein **Brotbackautomat**: Zutaten (gelabelte Daten) rein, Programm wählen, fertig kommt ein
gebackenes Modell heraus — die Maschine kümmert sich selbst um Kneten, Gehen und Backen (CASH +
Meta-Learning + Ensembling). Bequem und stark. Der Haken: Diese Automaten backen nur
**„überwachtes" Brot** — für **unüberwachte AD** sind sie nicht gebaut. Dafür braucht es Umwege
(PyCaret als PyOD-Wrapper) oder die anderen Strategien.

### Wann sinnvoll – und wann nicht

| Stark, wenn … | Heikel/schwach, wenn … |
|---|---|
| **gelabelte** Fehlerklassifikation gefragt ist | du echtes **label-freies** AutoML-für-AD brauchst (→ Strategie 3/4) |
| du mit minimalem Code eine starke Baseline willst | du **Erklärtiefe** brauchst (Black-Box) |
| Meta-Learning & Ensembling „live" gezeigt werden sollen | Installation heikel ist (auto-sklearn unter Py 3.13!) |

### So liest und erklärst du das Ergebnis

- **Wenig Code, starkes Ergebnis** — der bequeme überwachte Vergleichspol zur AD.
- **Ehrliches TEP-Ergebnis:** **RF ≈ FLAML ≈ AutoGluon** (~0.66–0.68 Macro-F1, ~0.83 AD-AUC) —
  *das aufwändige Framework schlug den handgetunten Random Forest nicht*. Auch das ist ein
  berichtenswerter Befund: „mehr Maschinerie ≠ automatisch besser".
- **Faustregel zum Erklären:** „Aus der Box bekommt man schnell ein solides überwachtes Modell —
  aber den Sprung gegenüber einer sauberen Baseline muss man sich nicht erkaufen."
- **Praxis-Hinweis:** Im Projekt ersetzte **FLAML** das schwer installierbare auto-sklearn;
  AutoGluon lief nur in einem isolierten Python-3.12-Env. Solche Installationsrealitäten ruhig
  miterzählen — sie gehören zur ehrlichen AutoML-Erfahrung.

## 1. Idee & Problembezug

Statt einzelne Bausteine (Suchraum, Optimizer, Ensembling) selbst zu verdrahten, lösen
**End-to-End-Frameworks** das gesamte **CASH**-Problem (Vorverarbeitung + Modellauswahl +
HPO + Ensembling + Meta-Learning) out-of-the-box. Für die **supervise
Fehlerklassifikation** ([../ad-methoden/random_forest_klassifikation.md](../ad-methoden/random_forest_klassifikation.md))
ist das der direkte, produktive Weg — und der sauberste Kontrast zur unüberwachten AD.

## 2. Funktionsweise / Algorithmus

Typische Pipeline eines Frameworks (am Beispiel auto-sklearn):
1. **Meta-Learning Warm-Start:** Meta-Features des Datensatzes berechnen, ähnliche
   Datensätze finden, deren beste Konfigurationen als Startpunkte (KNN-Ansatz / Greedy
   Portfolio).
2. **Bayesian Optimization (SMAC):** durchsucht den Pipeline-Raum (Preprocessing + Modell +
   HP).
3. **Ensembling:** baut aus den vielen trainierten Modellen ein **Ensemble Selection**
   (gewichtete Kombination) statt nur das Einzelbeste zu nehmen.

→ Genau die Konzepte aus den Strategien 1, 3 und 4, hier integriert verpackt.

## 3. Anbindung an AD

- **Supervised (Kernfall):** Fehlerklassifikation direkt — `fit(X, y)` / `predict`. Liefert
  oft mehr als ein handgetunter RF und demonstriert Meta-Learning + Ensembling „live".
- **Unsupervised AD:** Die großen Frameworks sind **überwacht** ausgelegt und decken AD
  **nicht** nativ ab. Wege:
  - **PyCaret `anomaly`-Modul** — komfortabler Wrapper um PyOD-Detektoren (`setup` →
    `create_model("iforest")` → `assign_model`); kein echtes label-freies CASH, aber gute
    einheitliche API für Strategie 4.
  - **PyOD** selbst als „AD-Toolkit" mit einheitlichem Interface (Basis für Strategie 3 & 4).
  - Echtes label-freies AutoML-für-AD → Strategie 3
    ([meta_learning_modellselektion.md](meta_learning_modellselektion.md)).

## 4. Tools / Libraries

| Framework | Aufgabe | AD-Eignung | Installierbarkeit (Py 3.13) |
|---|---|---|---|
| **auto-sklearn** | supervised CASH | nur supervised | **kritisch** — Linux-only, historisch ≤Py3.10/3.11; **vor Nutzung Kompatibilität prüfen**, ggf. separates Env |
| **AutoGluon** (Tabular) | supervised CASH | nur supervised | i. d. R. aktuelle Python-Versionen; größere Abhängigkeiten |
| **PyCaret** | supervised + `anomaly`-Modul | AD via PyOD-Wrapper | meist breit kompatibel; Versions-Pins beachten |
| **FLAML** | leichtgewichtiges CASH | nur supervised | gut installierbar, schnell |

> **Phase-B-Hinweis:** Unter Python 3.13 ist **auto-sklearn** der wahrscheinlichste
> Stolperstein. Optionen: (a) separates Conda-Env mit älterem Python nur für auto-sklearn,
> (b) Ersatz durch **FLAML/AutoGluon**, (c) auto-sklearn als reines „Konzept-Demo" mit
> kleinem Datensatz. Erst in Phase B final festlegen.

Minimal-Beispiel (FLAML, supervised Fehlerklassifikation — robust installierbar):

```python
from flaml import AutoML
automl = AutoML()
automl.fit(X_train, y_train, task="classification",
           metric="macro_f1", time_budget=300)     # 5 min Budget
print(automl.best_estimator, automl.best_config)
y_pred = automl.predict(X_test)
```

PyCaret-AD-Skizze (einheitlicher Wrapper für Strategie 4):

```python
from pycaret.anomaly import setup, create_model, assign_model
setup(data=df_good, normalize=True)
iforest = create_model("iforest")
results = assign_model(iforest)        # Spalten: Anomaly, Anomaly_Score
```

## 5. Stärken / Grenzen für TEP

- **Stärken:** minimaler Code für ein starkes supervised Ergebnis; demonstriert Meta-Learning
  & Ensembling ohne Eigenbau; gute „Baseline aus der Box".
- **Grenzen:** echtes **label-freies AutoML-für-AD** liefern die Frameworks nicht; Installation
  (v. a. auto-sklearn) kann unter Py 3.13 hakeln; Black-Box-Charakter (weniger
  Erklärtiefe als Strategie 1).

## 6. Demonstrierte Capability & Referenzen

**Capability:** zeigt CASH + Meta-Learning + Ensembling als integriertes Produkt und liefert
den **supervised Vergleichspol** zur AD — die Kontrastfolie „mit vielen Labels".

**Referenzen:** Feurer et al. (2015/2020, auto-sklearn); Erickson et al. (2020, AutoGluon);
Wang et al. (2021, FLAML); PyCaret-Doku; PyOD (Zhao et al. 2019). Siehe
[../referenzen.md](../referenzen.md).
