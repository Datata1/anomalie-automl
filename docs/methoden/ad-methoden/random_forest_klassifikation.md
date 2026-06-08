# Random-Forest-Fehlerklassifikation (supervised Baseline)

## 1. Titel & Einordnung

- **Setting:** supervised (Mehrklassen-Fehlerklassifikation)
- **AD-Typ:** punktweise Klassifikation (nicht im engeren Sinne AD)
- **Reife:** robuster Standard; dient als **Vergleichspol** zu den AD-Methoden.

## 2. Grundidee

Wenn **viele gelabelte** Daten aller Fehlertypen vorliegen, kann man statt „Anomalie ja/nein"
direkt **klassifizieren**: Kein Fehler / Fehler 1 / … / Fehler 20. Ein Random Forest aus vielen
Entscheidungsbäumen (Bagging + Feature-Subsampling) liefert dafür eine starke, robuste
Baseline. Die Vorlesungs-Übung verlangt explizit den **Vergleich Klassifikation vs. AD**.

## 3. Funktionsweise & Mathematik

Viele Entscheidungsbäume werden auf Bootstrap-Stichproben mit zufälliger Feature-Auswahl je
Split trainiert; die Vorhersage ist die **Mehrheit** (bzw. gemittelte Klassenwahrscheinlichkeit)
über alle Bäume. Reduziert Varianz gegenüber Einzelbäumen, kaum Overfitting-Tendenz.

- **Zwei Aufgabenvarianten:**
  1. **Binär** (Fehler ja/nein) → direkt mit AD vergleichbar (gleiche Metriken).
  2. **Mehrklassig** (0 + 20 Fehler) → zusätzliche Diagnose-Information (welcher Fehler).

## 4. Annahmen & Datenanforderungen

- Braucht **gelabelte Fehlerdaten** in ausreichender Menge (`faulty_training` liefert das).
- Skalierung nicht nötig (baumbasiert).
- **Labeling-Fallstrick:** In `faulty_training` ist der Fehler erst ab Sample 21 aktiv —
  binäres Label = `(faultNumber!=0) & (sample>20)`; analog Test ab Sample 161 (siehe
  [../01_datensatz_tep.md](../01_datensatz_tep.md) §4).
- **Split nach `simulationRun`** (kein Leakage).

## 5. Hyperparameter

| Name | Bedeutung | Typischer Bereich | Von AutoML getunt? |
|---|---|---|---|
| `n_estimators` | Anzahl Bäume | 100 – 1000 | ja |
| `max_depth` | Baumtiefe | 5 – None | ja |
| `max_features` | Features je Split | sqrt / 0.3 – 1.0 | ja |
| `min_samples_leaf` | Blattgröße (Reg.) | 1 – 50 | ja |
| `class_weight` | Klassen-Gewichtung | None / balanced | ja |

## 6. Anomaly-Score & Threshold

- Für den AD-Vergleich: `predict_proba` der Fehlerklasse(n) als Score; Threshold via
  PR-Kurve. So sind ROC-AUC/PR-AUC direkt mit den AD-Methoden vergleichbar.

## 7. AutoML-Anbindung

- **Fokus 2 (Frameworks):** **Paradefall** — auto-sklearn / AutoGluon / PyCaret lösen diese
  überwachte Aufgabe out-of-the-box (CASH, Ensembling, Meta-Learning) und liefern oft mehr
  als ein handgetunter RF. Detail:
  [../automl-strategien/automl_frameworks.md](../automl-strategien/automl_frameworks.md).
- **Fokus 1 (HPO):** klassisches RF-Tuning als Vergleich zum Framework-Ergebnis.
- **Selektionsproblem:** entfällt — Labels vorhanden, Standard-CV. Genau das macht es zum
  **Kontrast** zur unüberwachten AD.

## 8. Referenz-Implementierung

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                            class_weight="balanced", n_jobs=-1, random_state=0)
rf.fit(X_train, y_train)            # y: 0..20 (mehrklassig) oder 0/1 (binär)
proba = rf.predict_proba(X_test)
print(classification_report(y_test, rf.predict(X_test)))
```

## 9. Eignung für TEP

- **Stärken:** stark und robust bei reichlich Labels; liefert **Fehlerdiagnose** (welcher
  Fehler), nicht nur „Anomalie". Sehr schnelle, gut interpretierbare Baseline (Feature
  Importances).
- **Schwächen:** braucht gelabelte Fehler **aller** Typen — im Realbetrieb selten gegeben;
  erkennt **keine neuen/ungesehenen** Fehlertypen (im Gegensatz zur AD). Genau dieser
  Unterschied ist die Kernaussage des Klassifikation-vs.-AD-Vergleichs.

## 10. Demonstrierte Capability

Der **Vergleichspol**: zeigt die Obergrenze bei voller Label-Verfügbarkeit und macht greifbar,
warum unüberwachte AD trotz schlechterer Zahlen praxisrelevant ist (neue Fehler, keine Labels).
Zugleich der sauberste Einstieg in **fertige AutoML-Frameworks**.

## 11. Referenzen

Breiman (2001), „Random Forests"; auto-sklearn (Feurer et al. 2015/2020). Siehe
[../referenzen.md](../referenzen.md).
