# Isolation Forest

## 1. Titel & Einordnung

- **Setting:** unsupervised
- **AD-Typ:** multivariat (punktweise)
- **Reife:** Standard-Detektor, exzellentes Preis-Leistungs-Verhältnis, oft starke Baseline.

## 2. Grundidee

Anomalien sind „few and different" und lassen sich daher **leichter isolieren** als normale
Punkte. Ein Wald aus zufälligen *Isolation Trees* trennt Datenpunkte durch zufällige
Feature-/Schwellwert-Splits. Anomalien werden im Schnitt mit **weniger Splits** abgetrennt
(kürzere Pfadlänge) → hoher Anomaly-Score.

## 3. Funktionsweise & Mathematik

Jeder iTree wählt rekursiv ein zufälliges Feature und einen zufälligen Split, bis Punkte
isoliert sind. Sei `h(x)` die Pfadlänge von x, `E[h(x)]` der Mittelwert über alle Bäume und
`c(n)` die erwartete Pfadlänge in einem zufälligen Binärbaum mit n Punkten. Der Score:

```
s(x, n) = 2^( − E[h(x)] / c(n) )
```

- `s → 1`: sehr kurze Pfade → **Anomalie**.
- `s → 0.5`: normal.

## 4. Annahmen & Datenanforderungen

- Keine Verteilungsannahme; skaleninvariant gegenüber monotonen Transformationen, aber
  Standardisierung schadet nicht und vereinheitlicht die Pipeline.
- Punktweise (keine Zeitdynamik). Für TEP optional Zeit-Features (Lag/Rolling-Mean) ergänzen.
- Robust gegen irrelevante Features, aber sehr hochdimensionale, dünn besetzte Anomalien
  können untergehen → Extended/erweiterte Varianten.

## 5. Hyperparameter

| Name | Bedeutung | Typischer Bereich | Von AutoML getunt? |
|---|---|---|---|
| `n_estimators` | Anzahl Bäume | 100 – 500 | ja |
| `max_samples` | Subsample-Größe je Baum | 128 – 1024 (oder „auto"=256) | ja |
| `max_features` | Feature-Anteil je Baum | 0.5 – 1.0 | ja |
| `contamination` | erwarteter Anomalieanteil (→ Threshold) | 0.01 – 0.1 / „auto" | ja |

> `max_samples` ist der wichtigste Performance-Hebel und ein gutes Beispiel, wo AutoML
> echten Mehrwert bringt.

## 6. Anomaly-Score & Threshold

- sklearn: `score_samples` (höher = normaler) bzw. `decision_function` (<0 = Anomalie);
  Vorzeichen für die einheitliche Konvention (höher = anomaler) umdrehen.
- PyOD: `decision_function` (höher = anomaler), `threshold_` aus `contamination`.

## 7. AutoML-Anbindung

- **Fokus 1 (HPO):** Suchraum = {`n_estimators`, `max_samples`, `max_features`,
  `contamination`}. Multi-Fidelity über `max_samples`/Daten-Subsample sehr natürlich
  (billig auswertbar) → ideal für Hyperband/BOHB.
- **Fokus 3 (Meta-Learning):** iForest mit verschiedenen Konfigurationen ist ein
  Standardbaustein der MetaOD-Modellbibliothek.
- **Fokus 4 (Ensembling):** mehrere iForests/Seeds aggregieren.
- **Selektionsproblem:** ohne Labels ist v. a. `contamination`/Threshold schwer zu setzen →
  Quantil-Regel oder interne Metrik nutzen
  ([../automl-strategien/00_modellselektion_ohne_labels.md](../automl-strategien/00_modellselektion_ohne_labels.md)).

## 8. Referenz-Implementierung

```python
from sklearn.ensemble import IsolationForest
import numpy as np

iforest = IsolationForest(n_estimators=200, max_samples=256,
                          contamination="auto", random_state=0).fit(X_good_scaled)
score = -iforest.score_samples(X_test_scaled)   # höher = anomaler
thr = np.quantile(-iforest.score_samples(X_good_scaled), 0.99)
y_pred = (score > thr).astype(int)
```

PyOD-Variante (einheitliches Interface mit anderen Detektoren):
`from pyod.models.iforest import IForest`.

## 9. Eignung für TEP

- **Stärken:** skaliert gut auf 9,6 Mio. Zeilen (mit Subsampling), wenige HP, robuste
  Defaults — perfekter „erster echter Detektor". Fängt klare Step-Fehler (IDV 1,2,6,7) gut.
- **Schwächen:** rein punktweise → schwache, langsam driftende oder rein zeitliche Fehler
  (IDV 13, teils 3/9/15) werden schwerer erkannt.

## 10. Demonstrierte Capability

Zeigt, wie viel ein **gut getunter** Tree-Detektor gegenüber Defaults bringt — der
Paradefall für HPO/Multi-Fidelity. Gleichzeitig Standardmitglied in Ensembles und
MetaOD-Bibliothek.

## 11. Referenzen

Liu, Ting & Zhou (2008), „Isolation Forest"; PyOD (Zhao et al. 2019). Siehe
[../referenzen.md](../referenzen.md).
