# Statistische Baseline: Z-Score & PCA-Reconstruction

> **Intuition zuerst** — dieser Block erklärt die Methode in Alltagssprache und hilft dir,
> sie anderen zu erklären und Ergebnisse einzuordnen. Die formale Referenz folgt ab Abschnitt 1.

### In einem Satz

Beschreibe „normal" mit der **einfachsten sinnvollen Statistik** und markiere, was zu weit
herausfällt — der ehrliche Vergleichsmaßstab für alles Komplexere.

### Das Bild im Kopf

- **Z-Score (univariat):** „Dieser Wert ist *drei Standardabweichungen* vom Mittel entfernt"
  — wie ein **2,10-m-Mensch in einem Raum voller Durchschnittsgrößen**. Pro Sensor gemessen,
  ohne Rücksicht auf die anderen.
- **PCA-Reconstruction (multivariat):** Die Gutdaten leben in Wahrheit auf einer **dünnen,
  schräg im Raum liegenden Fläche** — denn die Sensoren bewegen sich gemeinsam (steigt der
  Reaktordruck, steigt auch die Temperatur). PCA lernt diese Fläche. Man **projiziert einen
  Punkt auf die Fläche und wieder zurück**; landet er weit von sich selbst, lag er *neben* der
  Normalfläche → anomal. Wie der Test „liegt dieser Punkt auf der bekannten Ebene?".

### Wann sinnvoll – und wann nicht

| Stark, wenn … | Heikel/schwach, wenn … |
|---|---|
| du **immer zuerst** einen ehrlichen Anker willst | Fehler **nichtlinear** sind (PCA sieht nur lineare Korrelation) |
| Geschwindigkeit & **Interpretierbarkeit** zählen | Fehler rein **zeitlich/driftend** sind |
| du erklären willst, *welche* Komponente abweicht | du den letzten Prozentpunkt AUC brauchst |

### So liest und erklärst du das Ergebnis

- **Der Score** ist der aggregierte |z| bzw. der Rekonstruktionsfehler — *höher = anomaler*.
- **Das ist dein Boden, nicht dein Ziel:** Die Baseline-Zahl ist der Maßstab, an dem sich alle
  komplexen Methoden messen lassen müssen. **Schlägt ein Autoencoder die PCA nur knapp, zahlt
  sich der Mehraufwand nicht aus** — eine der wichtigsten Aussagen, die du aus dem Vergleich
  ziehen kannst.
- **Faustregel zum Erklären:** „So weit kommt man mit Statistik in einer Zeile."
- **TEP-Einordnung:** PCA fängt die starke multivariate Sensor-Korrelation gut, *verpasst* aber
  die subtilen Fehler 3 / 9 / 15. Genau dieser **„Miss" motiviert** die teureren Methoden.
  Tipp: **ECOD** (parameterfrei) ist ein noch bequemerer Default-Anker.

---

## 1. Titel & Einordnung

- **Setting:** unsupervised
- **AD-Typ:** Z-Score = univariat; PCA-Reconstruction = multivariat
- **Reife:** sehr etabliert, einfachste sinnvolle Baseline. Pflicht als Vergleichsanker.

## 2. Grundidee

Beschreibe das normale Verhalten mit einfachen Statistiken und markiere Punkte, die zu weit
abweichen. **Z-Score** misst pro Feature, wie viele Standardabweichungen ein Wert vom
Mittel entfernt ist. **PCA-Reconstruction** lernt den linearen Unterraum der Gutdaten und
markiert Punkte, die sich daraus schlecht rekonstruieren lassen (großer Residualfehler).

## 3. Funktionsweise & Mathematik

**Z-Score (univariat):** Mit Mittel μ und Standardabweichung σ pro Feature (aus Gutdaten):

```
z_i = (x_i − μ_i) / σ_i
```

Anomalie, wenn `|z_i| > k` für ein Feature (oder Aggregation über Features, z. B. max/Norm).
Praxisüblich `k = 3` (bei Normalverteilung liegen 99,7 % zwischen ±3σ).

**PCA-Reconstruction (multivariat):** Standardisiere X, bestimme die ersten `q`
Hauptkomponenten `V_q` aus den Gutdaten. Rekonstruktion und Score:

```
x̂ = (x · V_q) · V_qᵀ           (Projektion in den Unterraum und zurück)
score(x) = ‖x − x̂‖²            (Reconstruction Error)
```

Anomalien liegen außerhalb des Gutdaten-Unterraums → großer Score. (Verwandt mit den
klassischen Prozess-Monitoring-Statistiken Hotelling's T² und SPE/Q.)

## 4. Annahmen & Datenanforderungen

- Features annähernd normalverteilt/stationär (Z-Score); PCA nimmt **lineare** Korrelationen
  an.
- **Skalierung zwingend** (StandardScaler auf Gutdaten gefittet).
- Keine Zeitdynamik modelliert (rein punktweise) — für TEP eine bewusst einfache Referenz.

## 5. Hyperparameter

| Name | Bedeutung | Typischer Bereich | Von AutoML getunt? |
|---|---|---|---|
| `k` (Z-Score) | Schwellwert in σ | 2.5 – 4 | ja (oder via Quantil) |
| `n_components` (PCA) | Größe des Unterraums | 1 – 51 (oder Varianzanteil 0.9–0.99) | ja |
| Aggregation (Z-Score) | max / L2-Norm über Features | – | optional |

## 6. Anomaly-Score & Threshold

- Score = aggregierter |z| bzw. Reconstruction Error.
- Threshold: 3σ-Regel **oder** Quantil der Gutdaten-Scores (z. B. 99 %) — siehe
  [../02_evaluationsprotokoll.md](../02_evaluationsprotokoll.md) §3.

## 7. AutoML-Anbindung

- **Fokus 1 (HPO):** `n_components` bzw. Varianzanteil und Threshold-Quantil tunen.
- **Fokus 4 (Ensembling):** als billiges Mitglied in ein Score-Ensemble.
- Selektionsproblem: minimal — dient v. a. als **untere Vergleichsgrenze** für die
  AutoML-Gewinne der stärkeren Methoden.

## 8. Referenz-Implementierung

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler().fit(X_good)
Xg = scaler.transform(X_good)

# PCA-Reconstruction
pca = PCA(n_components=10).fit(Xg)
def pca_score(X):
    Xs = scaler.transform(X)
    Xr = pca.inverse_transform(pca.transform(Xs))
    return ((Xs - Xr) ** 2).sum(axis=1)

thr = np.quantile(pca_score(X_good), 0.99)      # label-frei
y_pred = (pca_score(X_test) > thr).astype(int)
```

PyOD-Alternative: `pyod.models.pca.PCA`, `pyod.models.ecod.ECOD` (parameterfreie,
verteilungsbasierte Baseline — exzellent als Default-Anker).

## 9. Eignung für TEP

- **Stärken:** extrem schnell auf 9,6 Mio. Zeilen, interpretierbar (welches Feature/welche
  Komponente). PCA fängt die starke multivariate Korrelation der Sensoren gut ein.
- **Schwächen:** verpasst subtile, nichtlineare und zeitliche Fehler (z. B. IDV 3/9/15).
  Genau dieser „Miss" motiviert die komplexeren Methoden.

## 10. Demonstrierte Capability

Referenzpunkt: „So weit kommt man mit Statistik in einer Zeile." Macht die Mehrwerte von
AutoML + stärkeren Detektoren messbar.

## 11. Referenzen

Hotelling T²/SPE (Prozess-Monitoring-Klassiker); ECOD (Li et al. 2022); PyOD (Zhao et al.
2019). Siehe [../referenzen.md](../referenzen.md).
