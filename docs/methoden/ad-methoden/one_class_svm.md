# One-Class SVM (OC-SVM)

> **Intuition zuerst** — dieser Block erklärt die Methode in Alltagssprache und hilft dir,
> sie anderen zu erklären und Ergebnisse einzuordnen. Die formale Referenz folgt ab Abschnitt 1.

### In einem Satz

Zieh eine möglichst **enge, flexibel gekrümmte Hülle** um alle Gutdaten — was draußen liegt,
ist anomal.

### Das Bild im Kopf

Stell dir vor, du **vakuumierst die Punktwolke der Gutdaten in Schrumpffolie**. Eine simple
Kugel wäre zu grob; der **RBF-Kernel** erlaubt der Folie, sich in Buchten und Dellen
hineinzuschmiegen, sodass sie sich eng an die echte Form anlegt. Alles außerhalb der Folie ist
verdächtig. Der Parameter **`nu`** ist dein Toleranzregler: „Wie viele Gutpunkte darf die Folie
ruhig draußen lassen?" Klein = enge, strenge Folie; größer = lockerer. **`gamma`** steuert, wie
*wellig* die Folie sein darf.

### Wann sinnvoll – und wann nicht

| Stark, wenn … | Heikel/schwach, wenn … |
|---|---|
| **subtile, nichtlineare** multivariate Fehler gesucht | sehr **viele Zeilen** (Training ~O(n²–n³)) |
| Datenmenge klein bis mittel (oder Subsample) | Defaults reichen sollen — `nu`/`gamma` sind **sehr** sensibel |
| du den Mehrwert von **HPO** plakativ zeigen willst | wenig Zeit fürs Tuning ist |
| eine theoretisch fundierte Methode gewünscht | du Interpretierbarkeit pro Feature brauchst |

### So liest und erklärst du das Ergebnis

- **Der Score** kommt aus `decision_function`: *positiv = innerhalb der Folie (normal)*,
  *negativ = außerhalb (anomal)*; Vorzeichen umdrehen für „höher = anomaler".
- **Faustregel zum Erklären:** „Dieser Zustand passt nicht mehr unter die Hülle des
  Normalbetriebs."
- **Die typische OC-SVM-Geschichte:** Mit Defaults oft enttäuschend, mit getuntem
  `(nu, gamma)` plötzlich stark. **Genau dieser Sprung** ist die Lehre — die OC-SVM ist das
  „Poster-Child" dafür, *warum* sich Bayesian Optimization lohnt.
- **Praxis-Warnung:** Auf TEP (9,6 Mio. Zeilen) ist Volltraining unmöglich → immer auf einem
  **Subsample** trainieren. Berichtest du Zahlen, nenne die Subsample-Größe dazu.

---

## 1. Titel & Einordnung

- **Setting:** unsupervised (One-Class)
- **AD-Typ:** multivariat (punktweise)
- **Reife:** klassische, theoretisch fundierte Methode; stark bei nichtlinearen Grenzen,
  aber rechenintensiv.

## 2. Grundidee

Modelliere den Raum der Gutdaten und finde eine **Grenze (Hyperebene/Hypersphäre)**, die
möglichst viele Gutpunkte einschließt und maximalen Abstand zum Ursprung hat. Alles
außerhalb der Grenze ist eine Anomalie. Über **Kernels** (meist RBF) werden die Daten implizit
in einen höherdimensionalen Raum abgebildet, in dem sie besser trennbar sind.

## 3. Funktionsweise & Mathematik

Schölkopf-Formulierung: Trenne die Daten mit maximalem Margin ρ/‖w‖ vom Ursprung:

```
min_{w,ρ,ξ}  ½‖w‖² + (1/(νn)) Σ ξ_i − ρ
s.t.         w·Φ(x_i) ≥ ρ − ξ_i ,  ξ_i ≥ 0
Entscheidung: f(x) = sign( w·Φ(x) − ρ )      (negativ → Anomalie)
```

Der Parameter **ν ∈ (0,1]** ist eine obere Schranke für den Anteil Ausreißer und eine untere
Schranke für den Anteil Support-Vektoren — er steuert direkt, „wie viel" außerhalb der Grenze
liegen darf.

## 4. Annahmen & Datenanforderungen

- **Skalierung zwingend** (RBF-Kernel ist distanzbasiert).
- Sensitiv gegenüber `gamma`/`nu` — Defaults reichen selten → **idealer HPO-Kandidat**.
- **Skaliert schlecht**: Training ist ~O(n²)–O(n³). Auf TEP **nur auf einer Teilmenge**
  (z. B. 5k–50k Gutdaten-Punkte) trainieren oder `SGDOneClassSVM`/Nyström-Approximation
  verwenden.

## 5. Hyperparameter

| Name | Bedeutung | Typischer Bereich | Von AutoML getunt? |
|---|---|---|---|
| `nu` | obere Schranke Ausreißeranteil | 0.01 – 0.5 (log) | ja (zentral) |
| `gamma` | RBF-Kernelbreite | „scale" / 1e-4 – 1e1 (log) | ja (zentral) |
| `kernel` | rbf / poly / sigmoid | meist rbf | optional |

## 6. Anomaly-Score & Threshold

- sklearn `OneClassSVM.decision_function`: >0 normal, <0 Anomalie; Vorzeichen umdrehen für
  „höher = anomaler".
- Threshold ergibt sich implizit aus `nu` bzw. via Quantil-Regel.

## 7. AutoML-Anbindung

- **Fokus 1 (HPO):** `(nu, gamma)`-Gitter/BO ist das Lehrbuchbeispiel — kleiner, sensibler,
  kontinuierlicher Suchraum, perfekt für Bayesian Optimization.
- **Fokus 3/4:** Standardmitglied in MetaOD-Bibliothek und Ensembles.
- **Multi-Fidelity:** Trainings-Subsample-Größe als Budget (wegen O(n²)) — sehr natürlicher
  Hebel für Successive Halving/Hyperband.

## 8. Referenz-Implementierung

```python
from sklearn.svm import OneClassSVM
import numpy as np

ocsvm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(X_good_sub)  # Teilmenge!
score = -ocsvm.decision_function(X_test_scaled)   # höher = anomaler
thr = np.quantile(-ocsvm.decision_function(X_good_sub), 0.99)
y_pred = (score > thr).astype(int)
```

Für große n: `from sklearn.linear_model import SGDOneClassSVM` (+ `Nystroem`-Kernel-Map) oder
`from pyod.models.ocsvm import OCSVM`.

## 9. Eignung für TEP

- **Stärken:** flexible, nichtlineare Grenze — kann subtile multivariate Fehler erfassen, die
  PCA/iForest entgehen.
- **Schwächen:** Rechenkosten auf 9,6 Mio. Zeilen prohibitiv → Subsampling Pflicht; sehr
  HP-sensitiv (zeigt eindrücklich, warum AutoML nötig ist).

## 10. Demonstrierte Capability

Das „Poster-Child" für **HPO**: Mit Defaults oft schwach, mit getuntem `(nu, gamma)` stark.
Macht den Mehrwert von Bayesian Optimization plakativ sichtbar.

## 11. Referenzen

Schölkopf et al. (2001), „Estimating the Support of a High-Dimensional Distribution"; PyOD
(Zhao et al. 2019). Siehe [../referenzen.md](../referenzen.md).
