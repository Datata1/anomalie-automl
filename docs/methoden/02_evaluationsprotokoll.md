# Evaluationsprotokoll

> Ein einheitliches Protokoll für alle Methoden, damit Ergebnisse vergleichbar sind. Legt
> Split, Labeling, Metriken, Threshold-Wahl und Alarmmanagement fest.

## 1. Train/Test-Split (kein Leakage)

- **Split nach `simulationRun`**, nicht zeilenweise. Train- und Test-Läufe sind disjunkt.
- **Unsupervised AD:** Training **ausschließlich auf Gutdaten** (`fault_free_training`).
  Validierung/Test auf einer Mischung aus `fault_free_testing` (normal) und `faulty_testing`
  (Fehler).
- **Semi-supervised:** Gutdaten + ein kleiner, gelabelter Anteil aus `faulty_training`.
- **Supervised:** `faulty_training` (+ Gutdaten als Klasse 0) zum Training, `faulty_testing`
  zum Test.
- **Scaler/Vorverarbeitung** nur auf Trainingsdaten fitten.

Empfehlung Demo-Größe: pro Fehler 20–50 Test-Runs; Gutdaten entsprechend. Seed fixieren.

## 2. Punktweises Ground-Truth-Label

Wegen des Fehler-Onsets (siehe [01_datensatz_tep.md](01_datensatz_tep.md) §4):

```python
# faulty_testing: Fehler erst ab sample > 160
y_true = ((df["faultNumber"] != 0) & (df["sample"] > 160)).astype(int)
# fault_free_*: y_true = 0 für alle Zeilen
```

Optional **lauf-/segmentweise** Auswertung: Ein Lauf gilt als korrekt erkannt, wenn nach
dem Onset innerhalb eines Zeitfensters mindestens ein Alarm ausgelöst wird.

## 3. Anomaly-Score, Threshold & Richtungskonvention

Jeder Detektor liefert einen **Anomaly-Score** (höher = anomaler; Konvention vereinheitlichen!).
Die Entscheidung Anomalie/Normal entsteht durch einen **Threshold**:

- **Quantil-Regel (empfohlen, label-frei):** Threshold = z. B. 99 %-Quantil der Scores auf
  den *Gutdaten* (entspricht angenommener Kontaminationsrate). PyOD setzt analog
  `contamination` und liefert `threshold_`.
- **3σ / Z-Score-Regel:** für statistische Baselines (siehe
  [ad-methoden/statistische_baseline.md](ad-methoden/statistische_baseline.md)).
- **Label-basiert (nur als Oracle/Obergrenze):** Threshold, der F1 auf einem gelabelten
  Validierungsset maximiert — zeigt, was *mit* Labels möglich wäre.

## 4. Metriken

**Threshold-unabhängig** (bevorzugt für Methodenvergleich, da Threshold-frei):

- **ROC-AUC** — Rangqualität der Scores; robust, aber bei seltenen Anomalien optimistisch.
- **PR-AUC / Average Precision** — aussagekräftiger bei Klassen-Ungleichgewicht auf Punktebene.

**Threshold-abhängig** (Betriebssicht, nach Threshold-Wahl):

- **Precision, Recall, F1** (Anomalie = positive Klasse).
- **Confusion Matrix.**

**Zeit-/Betriebsspezifisch:**

- **Detection Delay / Time-to-Detect** — Anzahl Samples vom Fehler-Onset (Sample 161) bis
  zum ersten (stabilen) Alarm. Kleiner ist besser.
- **False Alarm Rate** auf reinen Gutdaten-Läufen (Fehlalarme pro Lauf/Stunde).
- **Missed Detection Rate** je Fehlertyp.

**Supervise Klassifikation:** zusätzlich Macro-F1 über die 21 Klassen (0 + 20 Fehler),
Per-Fault-Recall.

> Für den **AutoML-Vergleich** ist ROC-AUC/PR-AUC die primäre Zielgröße. Für die
> **Betriebsdarstellung** (PowerPoint) sind F1 + Detection Delay + False-Alarm-Rate
> anschaulicher.

## 5. Alarmmanagement (Post-Processing)

Rohe punktweise Flags sind verrauscht. Glättung wie in der Vorlesung:

```python
# y_flag: 0/1 pro Zeitschritt; window z. B. 10-30
alarm_rate = pd.Series(y_flag).rolling(window).mean()
plant_fault = (alarm_rate > 0.5).astype(int)   # Anlagenfehler-Signal
```

- Vereinzelte/zufällige Anomalien → eher **Modellfehler**.
- Anhaltend hohe Alarmrate → **Anlagenfehler**.
- `window` und Threshold (0.5) sind selbst Hyperparameter und können mitoptimiert werden.

## 6. Reporting-Schema (für Vergleichstabellen & PowerPoint)

Pro (Methode × AutoML-Strategie) mindestens:

| Methode | AutoML-Strategie | ROC-AUC | PR-AUC | F1 | Detection Delay | Trainingszeit |
|---|---|---|---|---|---|---|

Zusätzlich qualitativ: Score-Zeitreihe eines Beispiel-Laufs mit eingezeichnetem Onset und
Threshold; Per-Fault-Heatmap (welche Fehler werden von welcher Methode erkannt).

## 7. Fallstricke

- **Leakage** durch zeilenweises Splitten oder Scaler-Fit auf Testdaten — vermeiden.
- **Onset ignoriert** → falsche Labels (siehe §2).
- **ROC-AUC allein** kann täuschen — immer mit PR-AUC und Detection Delay zusammen berichten.
- **Modellselektion mit Labels** im unüberwachten Setting ist „Schummeln" — nur als
  explizites Oracle kennzeichnen (siehe
  [automl-strategien/00_modellselektion_ohne_labels.md](automl-strategien/00_modellselektion_ohne_labels.md)).
