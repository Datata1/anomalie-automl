# Deep SVDD & DeepSAD

## 1. Titel & Einordnung

- **Setting:** Deep SVDD = unsupervised; **DeepSAD = semi-supervised** (Gutdaten + wenige
  gelabelte Fehler)
- **AD-Typ:** multivariat (mit Sequenz-Backbone auch zeitlich)
- **Reife:** State-of-the-art Deep-One-Class; bringt die **semi-supervised** Komponente des
  Projekts ein.

## 2. Grundidee

**Deep SVDD** ist die Deep-Learning-Erweiterung der One-Class-Idee: Ein neuronales Netz φ
bildet die Daten so in einen latenten Raum ab, dass alle Gutdaten möglichst nah an ein
festes **Zentrum c** rücken (kleinste umschließende Hypersphäre). Der Abstand zum Zentrum ist
der Anomaly-Score. **DeepSAD** erweitert das um wenige Labels: Gutpunkte sollen nah ans
Zentrum, **bekannte Anomalien weit weg**.

## 3. Funktionsweise & Mathematik

**Deep SVDD** (nur Gutdaten):

```
min_W  (1/n) Σ ‖φ(x_i; W) − c‖²  + (λ/2)·‖W‖²
Score: s(x) = ‖φ(x; W) − c‖²
```

c wird typischerweise als Mittel der initialen Netzausgaben fixiert; Regularisierung +
keine Bias-Terme verhindern den trivialen Kollaps (φ ≡ c).

**DeepSAD** (Gutdaten ungelabelt + gelabelte Punkte, ỹ ∈ {+1 normal, −1 anomal}):

```
min_W  (1/(n+m)) [ Σ_unlab ‖φ(x_i)−c‖²  +  η Σ_lab ‖φ(x̃_j)−c‖^{ ỹ_j } ] + (λ/2)‖W‖²
```

Für `ỹ=+1` wird der Abstand minimiert (nah ans Zentrum), für `ỹ=−1` der **inverse** Abstand
minimiert → Anomalien werden **weggedrückt**. `η` gewichtet gelabelte vs. ungelabelte Daten.

## 4. Annahmen & Datenanforderungen

- **Skalierung zwingend.** Häufig **AE-Pretraining** des Encoders, dann SVDD-Feintuning.
- DeepSAD braucht **einige** gelabelte Anomalien (auch wenige genügen) → passt zu TEP, wo aus
  `faulty_training` ein kleines Label-Budget gezogen werden kann.
- Backbone wählbar (MLP für punktweise, 1D-CNN/LSTM für Sequenzen).

## 5. Hyperparameter

| Name | Bedeutung | Typischer Bereich | Von AutoML getunt? |
|---|---|---|---|
| `latent_dim` (rep_dim) | Dimension des Bildraums | 8 – 128 | ja |
| Backbone-Architektur | Tiefe/Breite/Typ | MLP/CNN/LSTM | ja (NAS) |
| `learning_rate`, `weight_decay` | Optimierung/Reg. | 1e-4–1e-2 / 1e-6–1e-3 (log) | ja |
| `eta` (nur DeepSAD) | Gewicht gelabelter Daten | 0.1 – 100 (log) | ja |
| Pretraining-Epochen / `epochs` | Budget | 10 – 150 | ja (Multi-Fidelity) |

## 6. Anomaly-Score & Threshold

- Score = `‖φ(x) − c‖²` (höher = anomaler).
- Threshold: Quantil der Gutdaten-Scores; bei DeepSAD kann das kleine Label-Set zur
  Threshold-Kalibrierung dienen.

## 7. AutoML-Anbindung

- **Fokus 1 (HPO):** Architektur + `eta` + Optimierung; Multi-Fidelity über Epochen/Daten.
- **Verbindung zum Kernproblem:** DeepSAD ist selbst eine Antwort auf „wenige Labels" — und
  ein **kleines Label-Budget erlaubt zugleich label-(teil)basierte Modellselektion** (Brücke
  zu ELECT, siehe
  [../automl-strategien/00_modellselektion_ohne_labels.md](../automl-strategien/00_modellselektion_ohne_labels.md)).
- Frameworks (Fokus 2) decken diese Modelle i. d. R. **nicht** ab → eigener Trainings-/HPO-Code.

## 8. Referenz-Implementierung

Referenz: offizielles Repo `lukasruff/Deep-SAD-PyTorch` (enthält auch Deep SVDD). Skizze des
SVDD-Trainingsziels in PyTorch:

```python
# c: vorab fixiertes Zentrum (Mittel der initialen Encoder-Ausgaben)
z = net(x_batch)                       # φ(x)
loss = ((z - c) ** 2).sum(dim=1).mean()           # Deep SVDD
# DeepSAD-Zusatz für gelabelte Punkte (y in {+1,-1}):
# loss += eta * ((dist_lab) ** y_lab).mean()
score = ((net(X_test) - c) ** 2).sum(dim=1)        # höher = anomaler
```

Alternativen mit fertigen Klassen: **DeepOD** (`deepod.models.DeepSVDD`, `DeepSAD`),
`pyod.models.deep_svdd.DeepSVDD`.

## 9. Eignung für TEP

- **Stärken:** stärkster Detektor des Katalogs bei subtilen Fehlern; **DeepSAD** nutzt die in
  TEP vorhandenen Fehlerlabels gezielt → meist deutlich bessere Detection schwerer Fehler
  (IDV 3/9/15).
- **Schwächen:** höchster Implementierungs-/Tuning-Aufwand, GPU empfohlen, kollaps-anfällig
  bei falscher Regularisierung.

## 10. Demonstrierte Capability

Bringt das **semi-supervised** Setting ein und zeigt, wie schon **wenige Labels** sowohl die
Detektion als auch die (teil-)label-basierte Modellselektion verbessern.

## 11. Referenzen

Ruff et al. (2018), „Deep One-Class Classification" (Deep SVDD); Ruff et al. (2020), „Deep
Semi-Supervised Anomaly Detection" (arXiv:1906.02694); DeepOD. Siehe
[../referenzen.md](../referenzen.md).
