# Autoencoder & LSTM-Autoencoder

> **Intuition zuerst** — dieser Block erklärt die Methode in Alltagssprache und hilft dir,
> sie anderen zu erklären und Ergebnisse einzuordnen. Die formale Referenz folgt ab Abschnitt 1.

### In einem Satz

Ein Autoencoder lernt, **normalen Anlagenzustand fehlerfrei nachzubauen** — und was er
*schlecht* nachbauen kann, ist verdächtig.

### Das Bild im Kopf

Stell dir einen **Kunstfälscher vor, der nur echte Geldscheine kennt**. Er hat so viele
gesehen, dass er jeden echten Schein aus dem Gedächtnis perfekt nachzeichnen kann. Legt man
ihm eine Fälschung mit ungewohnten Merkmalen vor, gerät seine Kopie daneben — und genau die
**Größe dieses Kopierfehlers** verrät die Fälschung. Der Autoencoder ist dieser Fälscher: Der
**Encoder** presst den Input durch einen Flaschenhals (er muss das *Wesentliche* behalten, nicht
jedes Detail), der **Decoder** baut daraus wieder den vollen Zustand. Trainiert nur auf Gutdaten,
gelingt das bei Normalbetrieb gut und bei Fehlern schlecht → der **Rekonstruktionsfehler** ist
der Anomalie-Score. Der **LSTM-AE** denkt nicht in Einzelbildern, sondern in **Melodien**: Er
fragt nicht „ist dieser Ton normal?", sondern „ist diese *Tonfolge* normal?" — und fängt so
zeitliche Fehler, die punktweise unsichtbar sind.

### Wann sinnvoll – und wann nicht

| Stark, wenn … | Heikel/schwach, wenn … |
|---|---|
| viele Sensoren mit **nichtlinearen** Zusammenhängen | wenig Daten oder keine GPU/Rechenzeit |
| genug Gutdaten zum Lernen vorhanden | du eine **schnelle, interpretierbare** Baseline brauchst (→ PCA) |
| **zeitliche/driftende** Fehler wichtig sind (LSTM-AE) | du **label-frei das Modell auswählen** musst (Trainings-Loss täuscht!) |
| etwas Tuning-Aufwand vertretbar ist | der Threshold robust sitzen muss (heikel) |

### So liest und erklärst du das Ergebnis

- **Der Score** ist der Rekonstruktionsfehler pro Punkt/Fenster — *höher = anomaler*. Die
  überzeugendste Darstellung ist die **Score-Zeitreihe eines Laufs** mit eingezeichnetem
  Fehler-Onset und Threshold: Vor dem Onset bleibt der Fehler niedrig, danach springt er klar
  über die Schwelle.
- **Faustregel zum Erklären:** „Das Modell konnte diesen Zustand nicht aus dem Gelernten
  nachbauen." Das ist für ein Publikum sofort einleuchtend.
- **Häufigste Fehldeutung:** Ein *niedriger Trainings-Loss heißt nicht „guter Detektor"*. Ist
  der Flaschenhals zu weit, lernt der AE eine Abkürzung und rekonstruiert *auch Anomalien* gut
  → er erkennt nichts mehr. Deshalb ist der Trainings-Loss **kein** Selektionskriterium (siehe
  Abschnitt 7).
- **TEP-Einordnung:** Der LSTM-AE erreichte ~**0.99 ROC-AUC** und fing sogar den langsam
  driftenden Fehler IDV 13 — das ist die Pointe, dass *zeitliche* Modellierung punktweisen
  Methoden überlegen ist, wenn die Fehlersignatur in der Dynamik steckt.

---

## 1. Titel & Einordnung

- **Setting:** unsupervised (Deep Learning)
- **AD-Typ:** Vanilla-AE = multivariat (punktweise); LSTM-AE = **zeitlich** (Sequenzen)
- **Reife:** etabliert, sehr flexibel; höherer Implementierungs- und Tuning-Aufwand.

## 2. Grundidee

Ein Autoencoder lernt, seinen eigenen Input zu rekonstruieren — über einen **Encoder** in
eine niedrigdimensionale latente Darstellung `z` und einen **Decoder** zurück. Trainiert nur
auf Gutdaten, rekonstruiert er Normalverhalten gut und **Anomalien schlecht** → der
**Rekonstruktionsfehler** ist der Anomaly-Score. Der LSTM-AE erweitert dies auf Zeitfenster
und erfasst so die **Dynamik** des Prozesses.

## 3. Funktionsweise & Mathematik

```
z = f_enc(x);   x̂ = f_dec(z)
Loss / Score:  L(x) = ‖x − x̂‖²        (MSE-Rekonstruktionsfehler)
```

- **Vanilla-AE:** vollverbundene Schichten, Bottleneck `dim(z) ≪ dim(x)`. Training
  minimiert mittleren MSE über Gutdaten.
- **LSTM-AE:** Input = Sequenz `x_{t−w+1..t}`. RNN/LSTM-Encoder komprimiert die Sequenz nach
  `z`, LSTM-Decoder rekonstruiert sie. Score = Rekonstruktionsfehler des Fensters (oder des
  letzten Schritts).
- **VAE (Variante):** probabilistischer Bottleneck (μ, σ) + KL-Term gegen N(0,I); strukturiert
  den Latentraum, nützlich u. a. für Data Augmentation (siehe
  [../referenzen.md](../referenzen.md)).

## 4. Annahmen & Datenanforderungen

- **Skalierung zwingend.** Training nur auf Gutdaten.
- LSTM-AE braucht **Sequenzfenster innerhalb eines `simulationRun`** (nie über Lauf-Grenzen,
  siehe [../01_datensatz_tep.md](../01_datensatz_tep.md) §7).
- Benötigt mehr Daten/Rechenzeit; profitiert von GPU. Frühes Stoppen gegen Overfitting.

## 5. Hyperparameter

| Name | Bedeutung | Typischer Bereich | Von AutoML getunt? |
|---|---|---|---|
| `latent_dim` | Bottleneck-Größe | 2 – 32 | ja |
| `hidden_layers`/`units` | Encoder/Decoder-Tiefe & Breite | 1–4 Layer, 16–256 Units | ja |
| `learning_rate` | Lernrate | 1e-4 – 1e-2 (log) | ja |
| `batch_size` | Batchgröße | 64 – 1024 | ja |
| `epochs` / Early-Stopping | Trainingsbudget | 10 – 100 | ja (Multi-Fidelity!) |
| `window` (LSTM-AE) | Sequenzlänge | 10 – 100 | ja |
| `dropout`/`weight_decay` | Regularisierung | 0 – 0.5 / 1e-6 – 1e-3 | ja |

## 6. Anomaly-Score & Threshold

- Score = MSE-Rekonstruktionsfehler pro Punkt/Fenster.
- Threshold: Quantil der Gutdaten-Rekonstruktionsfehler (z. B. 99 %) oder μ+kσ der
  Trainingsfehler.

## 7. AutoML-Anbindung

- **Fokus 1 (HPO):** großer, gemischter Suchraum (Architektur + Optimierung) → Paradefall für
  **Multi-Fidelity** (Budget = Epochen oder Daten-Subset) via Hyperband/BOHB; auch **NAS**
  (Architektursuche) konzeptionell hier verortet.
- **Fokus 4 (Ensembling):** mehrere AEs unterschiedlicher Latentgrößen/Seeds aggregieren.
- **Selektionsproblem:** Trainings-Loss (Gutdaten-MSE) ist **kein** guter Selektor (siehe
  Kernproblem) → label-freie Validierung/interne Metrik nötig.

## 8. Referenz-Implementierung

```python
import torch, torch.nn as nn, numpy as np

class AE(nn.Module):
    def __init__(self, d_in, latent=8, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, d_in))
    def forward(self, x): return self.dec(self.enc(x))

model, opt, loss = AE(X.shape[1]), None, nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# ... Trainingsloop nur auf Gutdaten, Early-Stopping ...
def ae_score(X):
    with torch.no_grad():
        xb = torch.tensor(X, dtype=torch.float32)
        return ((model(xb) - xb) ** 2).mean(dim=1).numpy()   # höher = anomaler
thr = np.quantile(ae_score(X_good), 0.99)
```

Schneller Einstieg ohne eigenes Training: `pyod.models.auto_encoder.AutoEncoder` (Torch-basiert)
bzw. für Sequenzen `pyod.models.lstmod` / eigene LSTM-AE in PyTorch. Deep-AD-Bibliothek:
**DeepOD**.

## 9. Eignung für TEP

- **Stärken:** erfasst **nichtlineare** Zusammenhänge (Vanilla-AE) und **zeitliche** Dynamik
  (LSTM-AE) — kann die schweren Fehler (IDV 3/9/13/15) besser fangen als PCA/iForest.
- **Schwächen:** teuer (9,6 Mio. Zeilen → Subsampling/GPU), viele HP, Threshold-Wahl heikel;
  Trainings-Loss eignet sich nicht zur Modellselektion.

## 10. Demonstrierte Capability

Zeigt **AutoML auf Deep Learning**: Multi-Fidelity-HPO und NAS auf einem nichtlinearen/
zeitlichen Detektor. Brücke zu Deep SVDD/DeepSAD (
[deep_svdd_deepsad.md](deep_svdd_deepsad.md)) und VAE-Data-Augmentation.

## 11. Referenzen

Autoencoder-/LSTM-AE-Standardliteratur; Kingma & Welling (2014/2019, VAE, arXiv:1906.02691);
DeepOD; PyOD (Zhao et al. 2019). Siehe [../referenzen.md](../referenzen.md).
