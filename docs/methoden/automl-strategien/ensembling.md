# AutoML-Strategie 4: Ensembling

## 1. Idee & Problembezug

Beim AutoML werden viele Modelle trainiert; einige sind fast so gut wie das Beste. Statt sie
zu verwerfen, **kombiniert** man sie. In der AD hat Ensembling einen besonderen Reiz: Es
**umgeht das Modellselektionsproblem** (
[00_modellselektion_ohne_labels.md](00_modellselektion_ohne_labels.md)) teilweise — man muss
nicht *das eine* beste label-frei finden, sondern aggregiert robust über viele.

## 2. Funktionsweise / Algorithmus

**Score-Normalisierung (Pflicht-Vorstufe):** Detektoren liefern Scores auf
unterschiedlichen Skalen → vor der Aggregation normalisieren (z. B. Z-Score- oder
Min-Max-Normalisierung, oder rangbasiert).

**Aggregationsstrategien (label-frei):**
- **Average / Median** der normalisierten Scores — robust, einfacher Default.
- **Maximum** — „ein Detektor reicht": sensitiv, aber anfällig für Fehlalarme.
- **AOM / MOA** (Average-of-Maximum / Maximum-of-Average) — Kompromisse über Detektor-Gruppen.

**Diversität erzeugen:** verschiedene Detektor-Typen, verschiedene HP, verschiedene
Daten-Subsamples/Feature-Subsets (Feature Bagging).

**Label-basiertes Ensembling (überwacht / Oracle):**
- **Ensemble Selection** (Greedy, Caruana et al.): Modelle iterativ ins Ensemble aufnehmen,
  solange die Validierungsmetrik steigt — genau das Verfahren in auto-sklearn.
- **Stacking:** Meta-Modell (z. B. logistische Regression) auf den Detektor-Scores;
  Overfitting-Gefahr auf Validierungsdaten beachten.

## 3. Anbindung an AD

- **Unsupervised:** normalisierte Scores von Isolation Forest + OC-SVM + PCA + AE per
  Average/Median kombinieren — typischerweise robuster als jeder Einzeldetektor und **ohne
  Labels** umsetzbar.
- **Sonderfall fehlende Labels:** Average/Median/Max brauchen **keine** Labels → Ensembling
  ist der **pragmatischste** label-freie AutoML-Hebel für TEP. Ensemble Selection/Stacking nur
  im überwachten/Oracle-Vergleich.
- **Konsens als Selektor:** der Ensemble-Konsens kann zugleich als interne Metrik dienen
  (Modelle nahe am Konsens bevorzugen) — Brücke zur Kernproblem-Doku.

## 4. Tools / Libraries

- **PyOD** — `pyod.models.combination` mit `average`, `maximization`, `aom`, `moa`, plus
  `standardizer` zur Score-Normalisierung; `pyod.models.feature_bagging.FeatureBagging`,
  `pyod.models.lscp.LSCP` (lokal selektive Kombination). `uv add pyod`.
- **combo** (`yzhao062/combo`) — allgemeine Modellkombination.
- Ensemble Selection: in auto-sklearn integriert (Strategie 2).

Skizze (PyOD, label-frei):

```python
import numpy as np
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.combination import average, maximization
from pyod.utils.utility import standardizer

detectors = [IForest(), OCSVM(), PCA()]
train_scores, test_scores = [], []
for clf in detectors:
    clf.fit(X_good_scaled)
    train_scores.append(clf.decision_scores_)
    test_scores.append(clf.decision_function(X_test_scaled))

train_scores = np.array(train_scores).T
test_scores  = np.array(test_scores).T
_, test_norm = standardizer(train_scores, test_scores)   # gemeinsame Skala
ens_score = average(test_norm)        # oder maximization(test_norm)
```

## 5. Stärken / Grenzen für TEP

- **Stärken:** robust gegen die Schwäche einzelner Detektoren bei bestimmten Fehlertypen;
  **label-frei** einsetzbar; geringer Zusatzaufwand, wenn die Einzeldetektoren ohnehin
  existieren; oft bester label-freier Score insgesamt.
- **Grenzen:** Rechenkosten = Summe der Mitglieder; `max` neigt zu Fehlalarmen; korrelierte
  Mitglieder bringen wenig Mehrwert (Diversität nötig); Stacking braucht Labels.

## 6. Demonstrierte Capability & Referenzen

**Capability:** zeigt, dass man das Selektionsproblem teils **umgehen** kann und dass robuste
Aggregation oft den besten **label-freien** Detektor schlägt — eine starke, praxisnahe
PowerPoint-Aussage.

**Referenzen:** Aggarwal & Sathe (2017, Outlier Ensembles); Caruana et al. (2004, Ensemble
Selection); PyOD/combo (Zhao et al. 2019). Siehe [../referenzen.md](../referenzen.md).
