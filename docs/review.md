# Review & Ergebnis-Analyse

Kritische Einordnung der Ergebnisse des AutoML-für-Anomaliedetection-Projekts (TEP). Grundlage:
[reports/results.csv](../reports/results.csv) und die 24+ Figuren in `reports/`. Begleitet die
Abschluss-PowerPoint ([reports/automl_ad_praesentation.pptx](../reports/automl_ad_praesentation.pptx)).

## 1. Sind die Ergebnisse gut?

**Ja — im Rahmen des Settings, und vor allem ehrlich** (nicht auf hohe Zahlen optimiert). Die
absoluten ROC-AUC der unüberwachten AD wirken mit **0.78–0.855** mittelmäßig, sind aber durch
notorisch schwer detektierbare Fehler (IDV 3/9/15) **gedeckelt**. Die wissenschaftlichen
**Kernaussagen** sind dagegen stark und didaktisch wertvoll:

| Aussage | Beleg |
|---|---|
| **Label-freie Selektion ≈ Oracle** | select_internal (pca) **0.854** vs. select_oracle (ocsvm) **0.855** |
| **Zeitmodell schlägt punktweise** | LSTM-AE ROC-AUC **~0.99** inkl. Drift-Fehler IDV 13 |
| **Wenige Labels helfen massiv** | DeepSAD **0.985** vs. Deep SVDD **0.834** |
| **Klassifikation > unüberwachte AD** | FLAML binäre AD-AUC > unüberwachte AD; erkennt aber nur *bekannte* Fehler |

Die zentrale Botschaft — **AutoML kann AD-Modelle auf TEP praktisch ohne Labels nahezu optimal
auswählen** — ist robust belegt.

## 2. Warum sind sie so?

- **Fehler-Heterogenität.** Der Per-Fault-Deep-Dive ([reports/11_per_fault_heatmap.png](../reports/11_per_fault_heatmap.png))
  zeigt: leichte Fehler (1/2/6/13) erreichen ~1.0 Recall, harte (3/9/15) ~0. Der gemischte
  ROC-AUC ist ein **Durchschnitt** — „mittel" trotz exzellenter Teilleistung.
- **Starke Bibliotheks-Defaults** (`gamma="scale"`, sinnvolle `n_estimators` etc.) → HPO-Gewinn
  klein (iforest 0.803 → 0.813). Multi-Fidelity zeigt **Effizienz/Pruning**, keinen
  Qualitätssprung.
- **Punktweise Detektoren ignorieren die Zeit** → verpassen langsame Drifts; der LSTM-AE nutzt
  Sequenzfenster und fängt IDV 13 klar.
- **Konsens-/interne Selektion funktioniert**, weil die Kandidaten korreliert sind und der beste
  Detektor „zentral" (nahe am Median-Konsens) liegt.
- **Subsampling + moderate Budgets** (Tempo, Reproduzierbarkeit) → keine Maximal-Performance,
  aber stabile, schnell nachvollziehbare Demos.

## 3. Was kann man besser machen?

- **Fault-aware Ensemble/Routing.** Die Heatmap zeigt Komplementarität (Fehler 4 wird nur von
  ocsvm/pca/AE/som erkannt, nicht von ecod/iforest) → ein fehlerbewusstes Ensemble oder Routing
  könnte den Durchschnitt deutlich heben.
- **Zeitfenster für alle Detektoren** (nicht nur LSTM-AE) + Fokus auf Detection-Delay und
  Alarmmanagement (gleitender Alarm) statt nur punktweiser ROC-AUC.
- **Größere HPO-Budgets**, OCSVM gezielt tunen, Multi-Fidelity über AE-Epochen (Learning-Curve).
- **Bessere interne Metriken** (SIREOS korreliert laut Literatur oft schwach); **echtes MetaOD**
  in einem separaten Env (scheiterte unter py3.13 → Konsens-Fallback).
- **Mehr Labels / Active Learning** (ELECT) für Selektion und DeepSAD.
- **Volldaten-Läufe, mehrere Seeds, Konfidenzintervalle** statt einzelner Subsample-Läufe.

## 4. Warum haben wir AutoGluon (zunächst) nicht getestet?

Kurz: **bewusste Scope-Entscheidung — und eine harte technische Hürde.**

- **Scope.** FLAML genügte als *ein* robuster CASH-Vertreter, um Modellwahl + HPO + Ensembling
  + Meta-Learning zu demonstrieren. **auto-sklearn** und **AutoGluon** lösen dasselbe
  **überwachte** Problem und adressieren **nicht** den Projektkern (unüberwachte AD-Selektion
  ohne Labels).
- **Technische Hürde (real verifiziert).** Unter **Python 3.13** mit dem modernen Stack
  (numpy 2.x, pandas 3.x) ist **AutoGluon nicht installierbar** — weder die stabile Version
  (keine py3.13-Wheels) noch der Pre-Release (Resolution scheitert). auto-sklearn ist ohnehin
  Linux-only und auf ältere Python-Versionen beschränkt. Das ist exakt der Grund, warum FLAML
  (py3.13-tauglich) gewählt wurde.
- **Nachgeholt.** Für dieses Review wurde AutoGluon dennoch benchmarkt — **isoliert in einem
  separaten Python-3.12-venv** (siehe `scripts/_autogluon_bench.py`), die Ergebnisse fließen in
  den Framework-Vergleich ([reports/12_framework_comparison.png](../reports/12_framework_comparison.png)).

### AutoGluon-Benchmark (isoliert, py3.12)

Auf identischem Split (`load_supervised`, gleiche Daten für alle drei):

| Framework | Macro-F1 | binäre AD-ROC-AUC |
|---|---|---|
| Random Forest (Baseline) | 0.68 | 0.83 |
| FLAML (CASH, 60 s) | 0.68 | 0.83 |
| **AutoGluon** (WeightedEnsemble_L2, 180 s) | **0.66** | **0.83** |

**Überraschend ehrlich:** AutoGluon schlägt RF/FLAML hier **nicht** — alle drei liegen praktisch
gleichauf, der Macro-F1 ist sogar leicht niedriger. Bei 21 (teils sehr harten) Fehlerklassen und
begrenztem Zeitbudget bringt das tiefere Stacking keinen Vorteil. Das bestätigt die ursprüngliche
Scope-Entscheidung: Der Framework-Wechsel ändert das Ergebnis kaum, und keiner der Frameworks
adressiert das eigentliche (unüberwachte) Kernproblem.

**Einordnung:** AutoGluon investiert in tiefes Stacking/Bagging und erreicht auf der
Klassifikation tendenziell die beste Tabellen-Performance — **ändert aber die
AutoML-für-AD-Geschichte nicht**: Es bleibt überwacht und erkennt nur bekannte Fehlertypen.
Der Mehraufwand (separates Env, lange Laufzeit) steht in keinem Verhältnis zum konzeptionellen
Mehrwert im *unüberwachten* Kernthema des Projekts.

## 5. Fazit

Das Projekt liefert keine rekordverdächtigen AD-Zahlen — und das ist in Ordnung. Sein Wert
liegt in der **sauberen, reproduzierbaren Demonstration der AutoML-Möglichkeiten für AD** und in
einer ehrlichen, gut belegten Kernaussage: **Modellselektion ohne Labels funktioniert hier
nahezu so gut wie mit Labels.** Die größten Hebel für bessere Ergebnisse sind Zeitmodellierung,
wenige Labels und fehlerbewusste Ensembles — nicht mehr HPO.
