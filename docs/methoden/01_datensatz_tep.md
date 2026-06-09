# Datensatz-Steckbrief: Tennessee Eastman Process (TEP)

> Alles, was man über die Daten wissen muss, um die Methoden zu implementieren und zu
> evaluieren — inkl. Beschaffung, Spaltenlogik und der für AD entscheidenden
> Fehler-Onset-Logik.

> **Intuition zuerst** — dieser Block fasst das Wichtigste in Alltagssprache zusammen. Die
> vollständigen Details folgen ab Abschnitt 1.

### Das Wichtigste in 60 Sekunden

Der **Tennessee Eastman Process (TEP)** ist eine **realitätsnah simulierte Chemieanlage**
(Reaktor, Kondensator, Separator, Stripper, Kompressor) mit **52 Sensoren/Stellgrößen**, die
alle 3 Minuten messen. Es gibt **Normalbetrieb** und **20 definierte Fehlertypen (IDV 1–20)** —
manche springen schlagartig (Step), manche schwanken zufällig, manche **driften langsam** (z. B.
IDV 13) oder zeigen klemmende Ventile. Weil es eine *Simulation* ist, gibt es keine fehlenden
Werte und eine saubere Balance — ideal zum Methoden-*Vergleichen*.

### Das Bild im Kopf — die Onset-Falle (der teuerste Anfängerfehler)

Stell dir eine **Patientenakte mit dem Etikett „Grippe-Patient"** vor. Trotzdem war der Patient
am Morgen *vor* Symptombeginn **gesund** — du darfst die Morgenmessungen nicht als „krank"
labeln. Genau so ist es hier: In den Test-Läufen mit Fehler sind die **ersten 160 Messungen noch
normal**, der Fehler wird **erst ab Sample 161** zugeschaltet.

> **`faultNumber != 0` heißt also NICHT, dass jede Zeile dieses Laufs anomal ist!** Das korrekte
> punktweise Label ist `(faultNumber != 0) & (sample > 160)`. Wer das ignoriert, *bestraft das
> Modell für korrektes Verhalten in der gesunden Anlaufphase* und verfälscht alle Zahlen.

### Worauf es beim Modellieren ankommt (und wie man es erklärt)

- **Skalierung ist Pflicht** — die Sensoren haben völlig verschiedene Einheiten (kPa, °C, kg/h,
  Mol-%). Scaler **nur auf Gutdaten** fitten (sonst Leakage).
- **Immer auf Lauf-Ebene (`simulationRun`) splitten/subsamplen**, nie zeilenweise — sonst zerreißt
  man die Zeitstruktur und schmuggelt Information von Train nach Test.
- **Manche Fehler sind leicht (IDV 1, 2, 6, 7), manche notorisch schwer (3, 9, 15).** Genau diese
  Spreizung macht den Datensatz zum guten *Schaukasten* für Methodenunterschiede: An den schweren
  Fehlern trennt sich die Spreu vom Weizen.

## 1. Herkunft & Charakter

- **Prozess:** Tennessee Eastman Process — ein realitätsnah simulierter chemischer Prozess
  (Reaktor, Kondensator, Separator, Stripper, Kompressor) mit 8 Komponenten (A–H).
  Ursprung: Downs & Vogel (1993).
- **Datensatz:** simulierte Sensor-Zeitreihen mit normalen und 20 Fehlerzuständen; hier die
  in der Community gängige Variante von **Rieth et al. (2017, Harvard Dataverse)**, gespiegelt
  u. a. auf [Kaggle (averkij)](https://www.kaggle.com/datasets/averkij/tennessee-eastman-process-simulation-dataset).
- **Format der Quelle:** `.RData`. Konvertierung nach Parquet via [export_rdata.R](../../export_rdata.R)
  (R-Paket `arrow`).
- **Task:** Anomaliedetection, Fehlerklassifikation, Fault Diagnosis.
- **Besonderheit:** synthetisch, **keine fehlenden Werte**, gleichmäßig balanciert.

## 2. Die vier Dateien

| RData-Variable / Datei | faultNumber | Runs | Samples/Run | Zeilen | Zweck |
|---|---|---|---|---|---|
| `fault_free_training` | 0 | 500 | 500 | 250 000 | **Gutdaten-Training** (unsupervised AD) |
| `fault_free_testing` | 0 | 500 | 960 | 480 000 | Gutdaten-Test (normale Klasse) |
| `faulty_training` | 1–20 | 500 je Fehler | 500 | 5 000 000 | gelabelte Fehler (supervised / semi-sup.) |
| `faulty_testing` | 1–20 | 500 je Fehler | 960 | **9 600 000** | Test mit Fehlern (Evaluierung AD) |

> Im Repo liegt bereits `data/TEP_Faulty_Testing.parquet` (9,6 Mio. Zeilen). Für das
> unüberwachte Setting werden zusätzlich **`fault_free_training`** (und für faire Tests
> `fault_free_testing`) benötigt — siehe Abschnitt 6.

## 3. Spaltenstruktur (55 Spalten)

| Block | Spalten | Anzahl | Bedeutung |
|---|---|---|---|
| Metadaten | `faultNumber`, `simulationRun`, `sample` | 3 | Fehlertyp, Lauf-ID, Zeitindex im Lauf |
| Manipulated Variables | `xmv_1` … `xmv_11` | 11 | Stellgrößen/Aktuatoren (Ventile, Feeds, Kühlwasser) |
| Continuous Measurements | `xmeas_1` … `xmeas_22` | 22 | kontinuierliche Sensoren (Drücke, Temperaturen, Flüsse) |
| Sampled Measurements | `xmeas_23` … `xmeas_41` | 19 | Gasanalysen (Mol-%), niedrigere Abtastrate (0.1 / 0.25 h) |

→ **52 Prozessvariablen** (41 XMEAS + 11 XMV) als Features + 3 Metadatenspalten.

> **Hinweis zur Spaltenzahl (bestätigt):** Das lokale Parquet hat **11 XMV** (`xmv_1…xmv_11`);
> XMV(12) „Agitator Speed" ist konstant und entfällt in der Rieth-Variante. Die Feature-Liste
> im Code (`automl_ad/config.py`) ist entsprechend auf 41 XMEAS + 11 XMV = 52 Features gesetzt.

Die `xmeas_23…41` haben eine geringere effektive Update-Rate (Mehrfachwiederholung der
Werte zwischen Abtastungen) — relevant für Feature-Engineering und Fenster.

## 4. Zeitstruktur & Fehler-Onset — **kritisch für AD-Labeling**

Jeder Lauf ist eine Zeitreihe (Abtastung alle 3 min):

- **`faulty_testing`:** 960 Samples/Lauf. **Samples 1–160 sind NORMAL**, der Fehler wird
  erst **ab Sample 161** eingeführt (8 Simulationsstunden) und bleibt bis zum Ende aktiv.
- **`faulty_training`:** 500 Samples/Lauf, Fehler **ab Sample 21**.
- **`fault_free_*`:** durchgehend normal.

**Konsequenz:** `faultNumber != 0` bedeutet **nicht**, dass jede Zeile dieses Laufs anormal
ist! Das punktweise Anomalie-Label lautet:

```python
# pointwise ground truth für einen faulty_testing-Lauf
y_true = ((df["faultNumber"] != 0) & (df["sample"] > 160)).astype(int)
```

Wer das ignoriert, „bestraft" Modelle für korrektes Verhalten in der normalen Anlaufphase
und überschätzt Detection-Delays. Details/Metriken in
[02_evaluationsprotokoll.md](02_evaluationsprotokoll.md).

## 5. Die 20 Fehlertypen (IDV)

| IDV | Beschreibung | Kategorie |
|---|---|---|
| 1 | A/C-Feed-Ratio, B konstant (Stream 4) | Step |
| 2 | B-Composition, A/C konstant (Stream 4) | Step |
| 3 | D-Feed-Temperatur (Stream 2) | Step |
| 4 | Reaktor-Kühlwasser-Eintrittstemperatur | Step |
| 5 | Kondensator-Kühlwasser-Eintrittstemperatur | Step |
| 6 | A-Feed-Verlust (Stream 1) | Step |
| 7 | C-Header-Druckverlust (Stream 4) | Step |
| 8 | A/B/C-Feed-Composition (Stream 4) | Random Variation |
| 9 | D-Feed-Temperatur (Stream 2) | Random Variation |
| 10 | C-Feed-Temperatur (Stream 4) | Random Variation |
| 11 | Reaktor-Kühlwasser-Eintrittstemperatur | Random Variation |
| 12 | Kondensator-Kühlwasser-Eintrittstemperatur | Random Variation |
| 13 | Reaktionskinetik | Slow Drift |
| 14 | Reaktor-Kühlwasser-Ventil | Sticking |
| 15 | Kondensator-Kühlwasser-Ventil | Sticking |
| 16–20 | unbekannt | Unknown |

**Schwierigkeit für AD:** Fehler 3, 9, 15 (und teils 16–20) sind notorisch schwer
detektierbar (geringe Signatur) — gut geeignet, um Methodenunterschiede zu zeigen. Fehler
1, 2, 6, 7 sind leicht.

## 6. Daten-Beschaffung & Konvertierung (Reproduzierbarkeit)

1. Download der vier `.RData`-Dateien (Kaggle/Harvard Dataverse) nach `data/`.
2. Konvertierung nach Parquet: `Rscript export_rdata.R` (exportiert **alle vier** Tabellen;
   das Skript liegt im Repo-Root).
3. Validierung von Vorhandensein + Schema (55 Spalten):
   `uv run python scripts/fetch_convert_data.py --check`.
4. Parquet ist git-ignored (`.gitignore`), bleibt also lokal.

## 7. Praktische Hinweise für die Modellierung

- **Skalierung zwingend:** Variablen haben sehr unterschiedliche Einheiten/Größenordnungen
  (kPa, °C, kg/h, Mol-%). `StandardScaler` **nur auf Gutdaten fitten**, dann auf Test
  anwenden (kein Leakage).
- **Größe:** 9,6 Mio. Zeilen — für schnelle, reproduzierbare Demos eine **Teilmenge der
  `simulationRun`s** ziehen (z. B. 20–50 Runs/Fehler). Subsampling immer **auf Run-Ebene**,
  nie zeilenweise (sonst Leakage zwischen Train/Test, Zeitstruktur zerstört).
- **Split nach `simulationRun`:** Train- und Test-Runs disjunkt halten.
- **Zeitreihen-Features:** für fensterbasierte Methoden (LSTM-AE) Sequenzen *innerhalb* eines
  Laufs bilden, nie über Lauf-Grenzen hinweg.
- **Klassenbalance:** Fehler sind gleichverteilt (500 Runs je Fehler) — kein
  Imbalance-Problem auf Fehlertyp-Ebene; auf *Punkt*-Ebene im Test sind ~17 % der faulty-
  Lauf-Samples normal (160/960).

## 8. Referenzen

Downs & Vogel (1993); Rieth et al. (2017, Harvard Dataverse). Siehe
[referenzen.md](referenzen.md).
