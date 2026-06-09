# AutoML-Strategie 3: Meta-Learning-Modellselektion (MetaOD)

> **Intuition zuerst** — dieser Block erklärt die Strategie in Alltagssprache. Die formale
> Behandlung folgt ab Abschnitt 1.

### In einem Satz

Empfiehl für einen **neuen, ungelabelten** Datensatz einen guten Detektor — nicht aus dessen
Labels, sondern aus **Erfahrung darüber, was auf ähnlichen Datensätzen gut war**.

### Das Bild im Kopf

Ein **Sommelier, der schon tausende Weine zu tausenden Gerichten probiert hat**. Du nennst ihm
ein neues Gericht, *er kostet es gar nicht* — er liest ein paar Merkmale ab („kräftig, fettig,
würzig") und empfiehlt aus seinem Erfahrungsschatz den passenden Wein. **MetaOD** ist dieser
Sommelier: Es hat viele Detektoren auf vielen Datensätzen gesehen, berechnet für deinen neuen
Datensatz einen **„Fingerabdruck" (Meta-Features)** und empfiehlt die Detektoren, die auf
ähnlich-fingerabdrückenden Datensätzen funktionierten — **ganz ohne Labels deines Datensatzes**.

### Wann sinnvoll – und wann nicht

| Stark, wenn … | Heikel/schwach, wenn … |
|---|---|
| du **prinzipiell label-frei** selektieren willst | dein Datensatz **untypisch** ist (außerhalb der Trainingsverteilung) |
| eine **sofortige** Empfehlung ohne Suchlauf gewünscht ist | **Zeitdynamik** entscheidend ist (Meta-Features ignorieren sie) |
| der State-of-the-Art-Kontrast zum Oracle gezeigt werden soll | das Tooling/Paket veraltet/inkompatibel ist |

### So liest und erklärst du das Ergebnis

- **Lies es als Dreikampf:** MetaOD-Empfehlung vs. **Oracle** (beste Wahl mit Labels) vs.
  **Default/Zufall**. Der **Gap zum Oracle** ist die Aussage.
- **Faustregel zum Erklären:** „Aus Erfahrung auf anderen Datensätzen können wir auch ohne Labels
  einen vernünftigen Detektor vorschlagen — die Frage ist nur, wie nah am Optimum."
- **Wichtige Einschränkung:** Die Qualität steht und fällt damit, **wie gut TEP durch die
  Benchmark-Datensätze repräsentiert** ist. TEP ist hochdimensional *und* zeitlich — gut möglich,
  dass es außerhalb dessen liegt, was MetaOD je gesehen hat. Im Projekt war MetaOD unter Python
  3.13 nicht lauffähig → **Konsens-Ensemble als Fallback** (siehe Kernproblem-Doku). Das offen
  benennen.

## 1. Idee & Problembezug

Direkte Antwort auf das **Kernproblem** (
[00_modellselektion_ohne_labels.md](00_modellselektion_ohne_labels.md)): Wähle für einen
**neuen, ungelabelten** Datensatz einen guten AD-Detektor (Modell + HP), indem man aus den
**Performances vieler Detektoren auf vielen Benchmark-Datensätzen** lernt. Statt auf Labels
des neuen Datensatzes stützt man sich auf dessen **Meta-Features** und gelerntes
Erfahrungswissen. Das ist „echtes" AutoML-für-AD ohne Labels.

## 2. Funktionsweise / Algorithmus

**MetaOD** (Zhao et al. 2021):
1. **Offline (vortrainiert):** Eine große **Performance-Matrix** P (Datensätze × Modelle) wird
   aus Benchmarks gebildet (Modell = Detektor-Typ + HP-Konfiguration). Per **Matrix-
   Faktorisierung** (Collaborative Filtering) werden latente Datensatz- und Modell-Faktoren
   gelernt; zusätzlich werden **Meta-Features** (Statistik-, Landmarking-Features) auf die
   latenten Datensatz-Faktoren abgebildet.
2. **Online (neuer Datensatz):** Meta-Features berechnen → latente Faktoren schätzen →
   erwartete Performance aller Modelle vorhersagen → **Top-n Modelle empfehlen**, ganz ohne
   Labels.

Verwandte/ergänzende Ansätze:
- **ELECT** (Zhao et al., NeurIPS 2021): Modellselektion mit **wenigem** aktivem Label-Budget.
- **Interne Metriken** (EM/MV, IREOS/SIREOS) als label-freie Alternative/Ergänzung (siehe
  Kernproblem-Doku). Achtung: oft schwache Korrelation zur echten Performance (Ma et al. 2023).

## 3. Anbindung an AD

- Kandidatenmenge = die Detektoren des Katalogs (Isolation Forest, OC-SVM, …) mit mehreren
  HP-Konfigurationen — exakt die Modellbibliothek, auf der MetaOD operiert.
- **Workflow im Projekt:** MetaOD-Empfehlung erzeugen → empfohlene Detektoren auf TEP-Gutdaten
  fitten → auf Testdaten evaluieren → gegen **Oracle** (label-basierte beste Wahl) und gegen
  **Default/Zufall** vergleichen. Der **Gap zum Oracle** ist die Kernaussage.

## 4. Tools / Libraries

- **MetaOD** — `pip install metaod` (`uv add metaod`), Repo `yzhao062/metaod`. Liefert
  vortrainierte Modelle + `select_model()`.
- Baut auf **PyOD** auf (gemeinsame Detektor-Bibliothek).
- **Kompatibilität in Phase B prüfen** (älteres Paket; Abhängigkeiten/Python-Version), ggf.
  fixierte Versionen oder separates Env.

Skizze:

```python
from metaod.models.predict_metaod import select_model
# X: neuer (ungelabelter) Datensatz; gibt Rangliste empfohlener Detektor-Konfigs zurück
recommended = select_model(X, n_selection=3)
# -> die empfohlenen PyOD-Modelle instanziieren, auf Gutdaten fitten, evaluieren
```

(Falls MetaOD nicht lauffähig: als **konzeptionelle Demo** eine kleine eigene
Performance-Matrix über wenige Datensätze + Meta-Feature-KNN nachbauen — derselbe Mechanismus
im Kleinen.)

## 5. Stärken / Grenzen für TEP

- **Stärken:** einziger Ansatz im Katalog, der **prinzipiell** label-frei selektiert; sehr
  schnelle Online-Empfehlung (kein Suchlauf nötig); didaktisch starker Kontrast zu Oracle.
- **Grenzen:** Qualität hängt davon ab, wie gut TEP durch die Benchmark-Datensätze
  „repräsentiert" ist (TEP ist hochdimensional + zeitlich — evtl. außerhalb der
  Trainingsverteilung); Tooling-/Wartungsstand des Pakets; Meta-Features ignorieren
  Zeitdynamik.

## 6. Demonstrierte Capability & Referenzen

**Capability:** der **State-of-the-Art-Baustein** für AutoML-AD ohne Labels; macht den Kern
des Projekts greifbar (label-frei selektieren und den Abstand zum Optimum messen).

**Referenzen:** Zhao et al. (2021, MetaOD, arXiv:2009.10606); Zhao et al. (NeurIPS 2021,
ELECT); Ma et al. (2021/2023, arXiv:2104.01422); PyOD (Zhao et al. 2019). Siehe
[../referenzen.md](../referenzen.md).
