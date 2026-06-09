# Kernproblem: Modell- & Hyperparameterselektion ohne Labels

> Das konzeptionelle Herzstück des Projekts. Wer dieses Dokument verstanden hat, versteht,
> warum „AutoML für Anomaliedetection" nicht einfach „AutoML auf ein AD-Modell anwenden"
> ist.

> **Intuition zuerst** — dieser Block erklärt das Kernproblem in Alltagssprache. Die formale
> Behandlung folgt ab Abschnitt 1.

### In einem Satz

AutoML wählt sonst das beste Modell, indem es Kandidaten **gegen einen Lösungsschlüssel
(Labels)** benotet — in der unüberwachten AD gibt es zur Auswahlzeit **keinen Schlüssel**, und
genau das macht das Problem schwer.

### Das Bild im Kopf

Stell dir vor, du sollst aus **zehn Klausuren die beste auswählen — ohne Musterlösung**. Genau
darin steckt AD: Du hast zehn Detektoren, aber keine gelabelten Anomalien, an denen du „richtig"
von „falsch" ablesen könntest. Drei Auswege:

1. **Interne Indizien** — wähle die Klausur, die in sich am saubersten/selbstsichersten wirkt
   (interne Metriken wie EM/MV, SIREOS). *Vorsicht:* „sieht ordentlich aus" korreliert oft nur
   **schwach** mit „ist richtig".
2. **Erfahrung aus früheren Klausuren mit Musterlösung** — du erinnerst dich, welche Art Schüler
   bei ähnlichen Aufgaben gut war (Meta-Learning / MetaOD).
3. **Gar nicht auswählen** — lass *alle zehn* abstimmen und nimm den Konsens (Ensemble). Oft
   überraschend stark, weil es das Auswahlproblem **umgeht**.

Und der **TEP-Trick:** Wir *haben* hier heimlich doch die Musterlösung (`faultNumber`). Wir
benutzen sie **nicht zum Auswählen**, sondern nur, um zu *messen*, wie weit die schlüssel-freien
Methoden vom Optimum entfernt sind — das ist das **Oracle**.

### So liest und erklärst du das Ergebnis

- **Die Kernzahl ist der „Gap zum Oracle":** Wie viel ROC-AUC kostet der Verzicht auf Labels?
- **Das überraschende TEP-Ergebnis:** Die label-freie **Konsens-Selektion (~0.854)** kam dem
  **Oracle (~0.855)** praktisch gleich. Heißt: *Auf TEP hat „Schummeln mit Labels" bei der
  Auswahl kaum geholfen* — ein starkes, beruhigendes Resultat für die Praxis.
- **Faustregel zum Erklären:** „Wir konnten den besten Detektor fast genauso gut *ohne* Labels
  finden wie *mit* — und das ist die eigentliche Frage von AutoML-für-AD."
- **Ehrlich bleiben:** Interne Metriken sind ein **Werkzeug, keine Garantie** (Ma et al. 2023) —
  immer kritisch berichten, nicht als Wahrheit verkaufen.

## 1. Warum klassisches AutoML hier nicht direkt greift

Klassisches AutoML (HPO, CASH) ist ein Optimierungsproblem:

> Finde die Konfiguration λ aus dem Suchraum Λ, die eine **Verlustfunktion auf einem
> gelabelten Validierungsset** minimiert.

Die Acquisition Function einer Bayesian Optimization, das Pruning bei Hyperband, die
Ensemble Selection — **alle** brauchen ein Gütesignal: „Konfiguration A ist besser als B".
Im **unüberwachten** AD-Setting fehlt dieses Signal, denn:

- Modelle werden nur auf **Gutdaten** trainiert (Anomalien sind teuer/selten).
- Zur Selektionszeit gibt es **keine** anomalie-gelabelten Validierungsdaten.
- Der Trainings-Loss (z. B. Rekonstruktionsfehler eines Autoencoders) misst nur, wie gut
  Gutdaten rekonstruiert werden — **nicht**, wie gut Anomalien getrennt werden. Ein Modell
  kann Gutdaten perfekt rekonstruieren und trotzdem ein schlechter Detektor sein.

**Das Unsupervised Outlier Model Selection (UOMS) Problem:** Aus einer Menge von Detektoren
und/oder Hyperparametern den besten auswählen — ohne Labels.

## 2. Lösungsfamilien

### 2.1 Interne (unüberwachte) Validierungsmetriken

Idee: Bewerte eine Detektor-Ausgabe (Score-Vektor) **ohne Labels** anhand interner Kriterien.

| Metrik | Idee | Praxis-Eignung |
|---|---|---|
| **Excess-Mass (EM) / Mass-Volume (MV)** | Bewertet die Score-Funktion als Dichte-Level-Set über das Lebesgue-Maß; gute Detektoren haben „kompakte" Hochdichte-Regionen. | rechenintensiv, hochdim. schwierig |
| **IREOS** (Internal, Relative Evaluation of Outlier Solutions) | Misst, wie gut sich Top-Outlier per Maximum-Margin-Klassifikator vom Rest trennen lassen. | sehr teuer; **SIREOS**/linearer Klassifikator als schnellere Variante |
| **Model Centrality / Konsens** | Wähle die Lösung, die dem „Konsens" vieler Modelle am nächsten ist (kein einzelnes Modell ist Wahrheit, aber der Median-Konsens ist robust). | gut skalierbar, Basis vieler Ensembles |

**Wichtige Warnung (Ma et al. 2023, arXiv:2104.01422):** In großangelegten Studien
korrelieren interne Metriken oft **schwach** mit der echten (label-basierten) Performance
und skalieren schlecht. Sie sind ein Werkzeug, keine Garantie — immer kritisch berichten.

### 2.2 Meta-Learning (datengetrieben)

Idee: Lerne aus den **Performances vieler Detektoren auf vielen Benchmark-Datensätzen**,
welche Konfiguration für einen *neuen* Datensatz gut ist — anhand von **Meta-Features**
(Datensatz-Kennzahlen) statt anhand von Labels des neuen Datensatzes.

- **MetaOD** (Zhao et al. 2021) — wählt label-frei ein Detektor-Modell aus einer großen
  Modellbibliothek; Collaborative-Filtering-artige Matrix-Faktorisierung über
  (Datensatz × Modell)-Performance. Detail:
  [meta_learning_modellselektion.md](meta_learning_modellselektion.md).
- Verwandt: Warm-Start der Bayesian Optimization über Meta-Features (KNN-Ansatz, Greedy
  Portfolio) — siehe [hpo_bayesian_optimization.md](hpo_bayesian_optimization.md) und
  [automl_frameworks.md](automl_frameworks.md).

### 2.3 Konsens-Ensembles (Selektion vermeiden statt lösen)

Statt *ein* bestes Modell zu wählen, **kombiniere viele** robust (Score-Normalisierung +
Aggregation wie Mittelwert/Maximum, oder Ensemble Selection). Das umgeht das Selektionsproblem
teilweise und ist oft erstaunlich stark. Detail: [ensembling.md](ensembling.md).

### 2.4 ELECT / wenige Labels (semi-supervised Selektion)

Wenn ein **kleines** Label-Budget existiert: aktiv die informativsten Punkte labeln und
darüber selektieren (z. B. ELECT, Zhao et al. NeurIPS 2021). Brücke zum semi-supervised
Setting ([../ad-methoden/deep_svdd_deepsad.md](../ad-methoden/deep_svdd_deepsad.md)).

## 3. Der TEP-Sonderweg: das Oracle

TEP **hat** Labels (`faultNumber` + Onset-Logik). Das nutzen wir didaktisch:

1. **Oracle-Selektion:** Wähle Detektor/HP label-basiert (max. ROC-AUC auf gelabeltem
   Validierungsset). Das ist die **Obergrenze** — im echten unüberwachten Betrieb nicht
   verfügbar.
2. **Label-freie Selektion:** Wähle per interner Metrik / MetaOD / Konsens-Ensemble.
3. **Gap berichten:** Wie viel ROC-AUC kostet der Verzicht auf Labels? Das ist die
   **Kernaussage** des Projekts und eine starke PowerPoint-Folie.

```
Performance
  ^
  |   ● Oracle (label-basiert)            ← Obergrenze
  |   ◑ MetaOD / interne Metrik            ← realistisch, label-frei
  |   ○ Default-HP / Zufallswahl           ← Baseline
  +---------------------------------------> Methode
```

## 4. Konkrete Umsetzung im Projekt

- Einheitliches Detektor-Interface (`fit(X_good)` / `decision_function(X) -> score`), damit
  jede Selektionsstrategie austauschbar darauf operiert (siehe Phase B im Projektplan).
- Pro Strategie eine kleine `select(...)`-Funktion, die aus Kandidaten eine Wahl trifft:
  - `select_oracle(cands, X_val, y_val)`
  - `select_internal(cands, X)` (z. B. EM/MV oder Konsens)
  - `select_metaod(cands, X)`
- Alle drei auf denselben Kandidaten laufen lassen und im Reporting (
  [../02_evaluationsprotokoll.md](../02_evaluationsprotokoll.md)) gegenüberstellen.

## 5. Demonstrierte Capability

Zeigt den eigentlichen Forschungsbeitrag: **Kann AutoML AD-Modelle auch ohne Labels sinnvoll
auswählen, und wie groß ist der Abstand zum label-basierten Optimum?**

## 6. Referenzen

Ma et al. (2021/2023, arXiv:2104.01422); Zhao et al. MetaOD (2021, arXiv:2009.10606); Zhao
et al. ELECT (NeurIPS 2021); IREOS/SIREOS-Studie (TKDD 2024); Goix (EM/MV). Siehe
[../referenzen.md](../referenzen.md).
