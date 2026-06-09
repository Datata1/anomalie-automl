# Self-Organizing Map (SOM) — optional

> **Intuition zuerst** — dieser Block erklärt die Methode in Alltagssprache und hilft dir,
> sie anderen zu erklären und Ergebnisse einzuordnen. Die formale Referenz folgt ab Abschnitt 1.

### In einem Satz

Lege ein **flexibles 2D-Gitter** in die Datenwolke der Gutdaten; wer weit von jedem Gitterknoten
liegt, ist anomal — und du bekommst **gratis eine Landkarte** des Prozesszustands.

### Das Bild im Kopf

Stell dir vor, du **wirfst ein Fischernetz über die Landschaft aller normalen
Betriebszustände**. Jeder Knoten sinkt in eine gut besuchte Region und merkt sich, wie es dort
„aussieht". Ein neuer Messpunkt nah an einem Knoten = vertrautes Terrain; weit von *allen*
Knoten = unbekanntes Gebiet → anomal. Der Clou: Weil das Netz **zweidimensional** ist und
benachbarte Knoten ähnliche Zustände abbilden, kannst du es **einfärben und buchstäblich sehen**,
wo sich die Anlage gerade befindet. (Wichtiges Detail: Knoten, die im Training fast nie getroffen
wurden, werden entfernt — sonst „verstecken" sich Anomalien an leeren Knoten.)

### Wann sinnvoll – und wann nicht

| Stark, wenn … | Heikel/schwach, wenn … |
|---|---|
| du eine **anschauliche 2D-Karte** zum Erklären willst (PowerPoint!) | du maximale Detection-Leistung brauchst (→ AE, DeepSAD) |
| didaktischer Mehrwert wichtiger ist als Spitzenleistung | wenig Zeit ist (viele HP, mehr Tooling-Aufwand) |
| multivariate Struktur sichtbar gemacht werden soll | Fehler rein zeitlich sind (punktweise Methode) |

### So liest und erklärst du das Ergebnis

- **Der Score** ist die mittlere Distanz zu den nächsten Gitterknoten — *höher = anomaler*.
- **Das eigentliche Verkaufsargument ist die Karte**, nicht die Zahl: Du kannst zeigen, wie der
  Betriebspunkt im Normalfall in einer Region „wohnt" und bei einem Fehler in unbesiedeltes
  Gebiet wandert. Für ein Publikum ist das oft überzeugender als ein AUC-Wert.
- **Faustregel zum Erklären:** „Die Anlage ist in einen Zustand gewandert, den sie im Normalbetrieb
  nie eingenommen hat."
- **Einordnung:** bewusst als *optionales Anschauungsmodell* gedacht — nimm es rein, wenn
  Zeit-/Folienbudget es erlauben, nicht als Leistungsträger.

---

## 1. Titel & Einordnung

- **Setting:** unsupervised
- **AD-Typ:** multivariat (punktweise, distanzbasiert)
- **Reife:** klassisch, aus der Vorlesung; eher als didaktische Ergänzung gedacht (optional).

## 2. Grundidee

Eine SOM ist ein Gitter aus Neuronen, jedes mit einem **Gewichtsvektor** in
Datenpunkt-Dimension. Durch *kompetitives Lernen* ordnen sich die Neuronen so an, dass die
**topologische Struktur** der Trainingsdaten erhalten bleibt (ähnliche Punkte → benachbarte
Neuronen). Für AD: Liegt ein neuer Punkt weit von allen (relevanten) Neuronen entfernt, ist
er anomal.

## 3. Funktionsweise & Mathematik

Training (vereinfacht): Gewichtsvektoren zufällig initialisieren; für jeden Trainingspunkt das
**Best Matching Unit (BMU)** = nächstgelegenes Neuron bestimmen und BMU + Nachbarn in Richtung
Punkt anpassen. Anomaly-Score eines Punkts x:

```
score(x) = mean_{k-NN-Neuronen} ‖x − w_j‖      (mittlere Distanz zu k nächsten Neuronen)
```

**Rauschreduktion:** Neuronen, die für (fast) keinen Trainingspunkt BMU sind, werden entfernt
— sonst könnten Anomalien fälschlich nah an „leeren" Neuronen liegen und unerkannt bleiben.
Threshold = Quantil der Trainings-Scores.

## 4. Annahmen & Datenanforderungen

- **Skalierung zwingend** (distanzbasiert). Training nur auf Gutdaten.
- Punktweise (keine Zeitdynamik).

## 5. Hyperparameter

| Name | Bedeutung | Typischer Bereich | Von AutoML getunt? |
|---|---|---|---|
| `map_size` (m×n) | Gittergröße | 5×5 – 30×30 | ja |
| `sigma` | Nachbarschaftsradius | 0.5 – 3 | ja |
| `learning_rate` | Lernrate | 0.1 – 1.0 | ja |
| `num_iteration` | Trainingsschritte | 1e3 – 1e5 | ja (Multi-Fidelity) |
| `k` (Scoring) | Anzahl nächster Neuronen | 1 – 5 | ja |
| Pruning-Schwelle | Min. Treffer je Neuron | 1 – 10 | ja |

## 6. Anomaly-Score & Threshold

- Score = mittlere Distanz zu den k nächsten Neuronen (höher = anomaler).
- Threshold via Quantil der Gutdaten-Scores.

## 7. AutoML-Anbindung

- **Fokus 1 (HPO):** Gittergröße/`sigma`/`lr`/`k` tunen; Iterationen als Multi-Fidelity-Budget.
- Weniger Standard-Tooling als iForest/OC-SVM → eigener kleiner Wrapper nötig.

## 8. Referenz-Implementierung

```python
from minisom import MiniSom
import numpy as np

som = MiniSom(x=15, y=15, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X_good_scaled)
som.train_random(X_good_scaled, num_iteration=10000)

W = som.get_weights().reshape(-1, X.shape[1])       # ggf. Pruning seltener Neuronen
def som_score(X, k=3):
    d = np.linalg.norm(X[:, None, :] - W[None, :, :], axis=2)
    return np.sort(d, axis=1)[:, :k].mean(axis=1)   # höher = anomaler
thr = np.quantile(som_score(X_good_scaled), 0.99)
```

(Bibliothek `MiniSom`; via `uv add minisom`.)

## 9. Eignung für TEP

- **Stärken:** liefert zusätzlich eine **2D-Visualisierung** des Prozesszustands (gut für die
  PowerPoint); erfasst multivariate Struktur.
- **Schwächen:** punktweise, viele HP, weniger leistungsstark als AE/Deep SVDD; Tooling-Aufwand.

## 10. Demonstrierte Capability

Optionales „Anschauungsmodell": zeigt topologie-erhaltende AD und eine interpretierbare Karte
des Normalzustands. Nur einbinden, wenn Zeit/Folienbudget es erlauben.

## 11. Referenzen

Kohonen (SOM-Originalliteratur); `MiniSom`. Siehe [../referenzen.md](../referenzen.md).
