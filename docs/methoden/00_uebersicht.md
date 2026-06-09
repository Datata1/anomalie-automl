# Übersicht: AutoML für Anomaliedetection in der Prozessindustrie

> Einstiegsdokument der Methodensammlung. Es ordnet das Projekt fachlich ein, erklärt
> die drei Lern-Settings und die vier AutoML-Strategien und verweist auf die
> Detaildokumente.

> **Intuition zuerst** — dieser Block erklärt in Alltagssprache, *worum es im ganzen Projekt
> geht*. Die fachliche Einordnung folgt ab Abschnitt 1.

### Die drei Begriffe in je einem Satz

- **Anomaliedetection (AD):** Ein Modell lernt, **wie „normal" aussieht**, und schlägt Alarm,
  sobald etwas davon abweicht — *ohne* vorher jeden möglichen Fehler kennen zu müssen.
- **AutoML:** Automatisiert die **langweiligen Experten-Entscheidungen** — *welches* Modell,
  *welche* Vorverarbeitung, *welche* Stellschrauben — die sonst ein Mensch mühsam von Hand trifft.
- **AutoML *für* AD:** Diese Automatisierung auf Anomaliedetection anwenden — und genau hier liegt
  der Knackpunkt, der dieses Projekt interessant macht.

### Das Bild im Kopf — und warum es hier hakt

Normales AutoML ist wie ein **Schüler, der mit Musterlösung übt**: Er probiert viele
Lösungswege und behält den, der die meisten Punkte gegen den **Lösungsschlüssel (die Labels)**
holt. So findet AutoML automatisch das beste Modell.

In der **unüberwachten Anomaliedetection gibt es zur Auswahlzeit aber keinen
Lösungsschlüssel** — echte Fehler sind selten und teuer, Modelle lernen nur den Normalbetrieb.
Damit fehlt AutoML das Kriterium, an dem es „besser" von „schlechter" unterscheidet. **Das ist
der rote Faden des Projekts:** Wie wählt man ein gutes AD-Modell aus, *ohne* nachschauen zu
können, ob es richtig liegt?

Der **Tennessee-Eastman-Datensatz** hat ausnahmsweise doch Labels. Wir nutzen sie **nicht zum
Auswählen**, sondern nur als **Oracle** — als heimliche Musterlösung, um zu *messen*, wie nah
eine label-freie Auswahl ans Optimum kommt.

> **Wenn du nur eine Sache mitnimmst:** „AutoML für AD ist schwer, weil im unüberwachten Fall der
> Lösungsschlüssel fehlt, an dem AutoML sonst Modelle benotet. Das Projekt misst, wie gut Auswahl
> *trotzdem* gelingt." Auf TEP war die label-freie Konsens-Auswahl (~0.854) fast so gut wie das
> Oracle (~0.855) — die ermutigende Kernbotschaft.

## 1. Fachlicher Rahmen: Prozessüberwachung → Anomaliedetection → AutoML

**Prozessüberwachung** beobachtet einen Produktionsprozess kontinuierlich anhand von
Sensormesswerten, um Soll-Ist-Abweichungen, Fehler und ineffiziente Zustände früh zu
erkennen. Sie ist eng verwandt mit *Predictive Maintenance* (Fehler erkennen, bevor ein
ungeplanter Stillstand entsteht).

**Anomaliedetection (AD)** ist der ML-Baustein dafür: Ein Modell lernt aus historischen
Daten das „normale" Prozessverhalten und entscheidet im Betrieb für jeden neuen Messwert,
ob der Prozess *normal* läuft oder *anormales* Verhalten vorliegt. Eine Anomalie ist ein
Datenpunkt mit Unregelmäßigkeiten gegenüber den übrigen Daten — sie kann sein:

- **univariat** — einzelne Werte weichen ab,
- **multivariat** — die *Struktur* zwischen Variablen ist gestört,
- **zeitlich** — die *Dynamik* der Daten ist gestört.

**AutoML** automatisiert die Erstellung von ML-Modellen: Auswahl der
Vorverarbeitung, des Modells (CASH — *Combined Algorithm Selection and Hyperparameter
optimization*), der Hyperparameter und ggf. der Architektur. Für die Prozessindustrie ist
das attraktiv, weil ähnliche Probleme (z. B. Soft-Sensoren, viele Anlagen/Linien) immer
wieder auftreten und der Entwicklungsprozess so skaliert werden kann.

Dieses Projekt verbindet beides: **Wie lassen sich AutoML-Methoden nutzen, um
Anomaliedetection-Modelle für einen Prozessdatensatz (Tennessee Eastman Process) zu bauen
und auszuwählen?**

## 2. Das zentrale Problem: AutoML für *unüberwachte* AD

Klassisches AutoML optimiert eine Verlustfunktion auf einem *gelabelten* Validierungsset.
In der unüberwachten AD gibt es zur Trainings-/Selektionszeit aber **keine Labels** —
Anomalien sind teuer und selten, und Modelle werden meist nur auf Gutdaten eingelernt.
Damit fehlt das Kriterium, an dem ein AutoML-System „besser" von „schlechter" unterscheidet.

> Dieses Spannungsfeld ist der rote Faden des Projekts. Es wird ausführlich in
> [automl-strategien/00_modellselektion_ohne_labels.md](automl-strategien/00_modellselektion_ohne_labels.md)
> behandelt.

Der TEP-Datensatz besitzt Labels (`faultNumber`), wir können also einen **„Oracle"-Vergleich**
ziehen: Wie nah kommt eine label-freie AutoML-Selektion an die label-basierte beste Wahl?

## 3. Die drei Lern-Settings

| Setting | Trainingsdaten | Typische Methoden | TEP-Bezug |
|---|---|---|---|
| **Unsupervised AD** | nur Gutdaten | Isolation Forest, OC-SVM, Autoencoder, statistische Baseline, SOM | `fault_free_training` |
| **Semi-supervised** | Gutdaten + wenige gelabelte Fehler | DeepSAD, Semi-Supervised-AE | Gutdaten + Teil von `faulty_training` |
| **Supervised** | viele gelabelte Fehler | Random-Forest-Klassifikation | `faulty_training` (Labels genutzt) |

Die Vorlesungs-Übung vergleicht explizit **Klassifikation vs. AD** — dieser Vergleich ist
Teil des Projekts.

## 4. Die vier AutoML-Strategien (querschnittlich)

Jede Strategie wird auf die AD-Methoden angewandt:

1. **HPO-Engines / Bayesian Optimization** — Optuna/SMAC tunen die Hyperparameter eines
   AD-Modells (Surrogate + Acquisition Function, Multi-Fidelity über Hyperband/BOHB).
   → [automl-strategien/hpo_bayesian_optimization.md](automl-strategien/hpo_bayesian_optimization.md)
2. **Fertige AutoML-Frameworks** — auto-sklearn, AutoGluon, PyCaret lösen CASH out-of-the-box
   (v. a. für die supervise Klassifikation), inkl. Meta-Learning und Ensembling.
   → [automl-strategien/automl_frameworks.md](automl-strategien/automl_frameworks.md)
3. **Meta-Learning-Modellselektion** — MetaOD wählt *ohne Labels* ein Detektor-Modell anhand
   gelernter Erfahrung auf vielen Benchmark-Datensätzen.
   → [automl-strategien/meta_learning_modellselektion.md](automl-strategien/meta_learning_modellselektion.md)
4. **Ensembling** — Kombination mehrerer Detektoren durch Score-Aggregation/Ensemble Selection,
   robust auch ohne perfektes Tuning.
   → [automl-strategien/ensembling.md](automl-strategien/ensembling.md)

## 5. Methoden-Matrix: Setting × AutoML-Strategie

|  | HPO (1) | Framework (2) | Meta-Learning (3) | Ensembling (4) |
|---|:---:|:---:|:---:|:---:|
| Statistische Baseline | ○ | – | – | ○ |
| Isolation Forest | ✓ | ○ | ✓ | ✓ |
| One-Class SVM | ✓ | ○ | ✓ | ✓ |
| Autoencoder / LSTM-AE | ✓ | – | ○ | ✓ |
| Deep SVDD / DeepSAD | ✓ | – | – | ○ |
| RF-Klassifikation | ✓ | ✓ | – | ✓ |

✓ = Kernkombination für die Demos · ○ = möglich/optional · – = nicht sinnvoll

## 6. Aufbau der Methodensammlung

```
docs/methoden/
  00_uebersicht.md                  ← dieses Dokument
  01_datensatz_tep.md               ← TEP-Steckbrief, Daten-Beschaffung, Label-Logik
  02_evaluationsprotokoll.md        ← Split, Metriken, Threshold, Alarmmanagement
  ad-methoden/                      ← 6 (+1) Basis-Detektoren, einheitliches Template
  automl-strategien/                ← Kernproblem + 4 Strategien
  referenzen.md                     ← zitierfähige Quellen
```

Jedes AD-Methodendokument folgt demselben Template (Grundidee → Funktionsweise/Mathe →
Annahmen → Hyperparameter → Score/Threshold → AutoML-Anbindung → Referenz-Implementierung →
TEP-Eignung → demonstrierte Capability → Referenzen), sodass es **ohne weitere Recherche**
implementierbar ist.

## 7. Praktischer Aspekt: vom Score zum Alarm

Im Betrieb springt eine AD oft zwischen „Anomalie"/„keine Anomalie". Ein einzelner
anormaler Zeitschritt ist meist ein Modellfehler; häufen sich Anomalien, liegt ein
Anlagenfehler vor. Übliches **Alarmmanagement**: gleitender Mittelwert der Anomalie-Flags
über die letzten *x* Zeitschritte; übersteigt er einen Schwellwert (z. B. 0.5), wird ein
Anlagenfehler gemeldet. Details in
[02_evaluationsprotokoll.md](02_evaluationsprotokoll.md).
