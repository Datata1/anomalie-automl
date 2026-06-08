# TICKET-03: DeepSAD / Deep SVDD (semi-supervised)

**Stream:** A (Neue Detektoren) · **Priorität:** mittel · **Aufwand:** L · **Abhängigkeiten:** T01 · **Status:** offen

## Kontext
AutoML-für-Anomaliedetection auf dem TEP (`uv`, Python 3.13, PyOD-`BaseDetector`-Interface).
Bisher deckt das Projekt unsupervised AD und supervised Klassifikation ab — das **semi-
supervised Setting fehlt noch live**. TEP liefert über `faulty_training` (Fehler ab `sample>20`)
gelabelte Fehler, aus denen ein **kleines Label-Budget** gezogen werden kann. torch ist
installiert. Doku beschreibt Deep SVDD (unüberwacht) → DeepSAD (semi-supervised).

## Ziel
Einen Deep-One-Class-Detektor bereitstellen, der **wenige Labels** nutzt (DeepSAD): Gutpunkte
nah ans Zentrum, bekannte Anomalien weg. Demonstriert den Mehrwert weniger Labels gegenüber
rein unüberwacht (Deep SVDD).

## Zu erstellende/ändernde Dateien
- **NEU** `automl_ad/detectors/deep_sad.py` — bevorzugt `deepod` (`DeepSAD`, `DeepSVDD`);
  **Fallback** `pyod.models.deep_svdd.DeepSVDD` (unüberwacht), falls `deepod` unter Py 3.13
  nicht installierbar. `FACTORIES={"deep_svdd": …, "deep_sad": …}` (deep_sad nur, wenn
  semi-supervised verfügbar). Guarded Import (T01 toleriert Fehlschlag).
- **additiv** `automl_ad/data.py` — `load_semisupervised(faults=None, n_train_good_runs=20, n_labeled_fault_runs=3, seed=0)`
  liefert Gutdaten + kleines, gelabeltes Fehler-Set (Onset 20) + Test wie `load_split`.
- **NEU** `notebooks/06_deepsad_semisupervised.py` (marimo).
- Versuch: `uv add --optional dl deepod` (falls inkompatibel: im Ticket dokumentieren, Fallback
  nutzen).

## Schnittstellen-Kontrakt
- `load_semisupervised(...) -> SemiSupervisedData` (dataclass): `X_train_good`, `X_labeled`,
  `y_labeled` (+1 normal / -1 anomal **oder** 0/1, im Modul dokumentieren), `X_test`, `y_test`,
  `meta_test`, `scaler`.
- Detektor-Factories liefern PyOD-konforme Objekte; DeepSAD-`fit(X, y=None)` akzeptiert das
  Label-Budget (API je nach Lib im Modul kapseln, sodass nach außen `fit(X)` + optionales `y`).
- `decision_function` → Score (höher=anomaler).

## Umsetzungshinweise
- Scaler nur auf Gutdaten fitten (wie `data.load_split`); `load_supervised` als Vorlage für das
  Ziehen gelabelter Fehlerläufe.
- DeepSAD braucht ein Encoder-Backbone (MLP); kleines `epoch_num` für Tempo. `fit` gibt `self`
  zurück.
- Vergleich im Notebook: Deep SVDD (0 Labels) vs. DeepSAD (wenige Labels) → ROC-AUC-Delta.

## Akzeptanzkriterien
- Mindestens `deep_svdd` ist als Registry-Detektor verfügbar und liefert plausible Scores
  (ROC-AUC > 0.6 auf Demo-Split).
- Falls semi-supervised verfügbar: DeepSAD nutzt das Label-Budget und ist auf schweren Fehlern
  (z. B. IDV 3/9) ≥ Deep SVDD; sonst Fallback dokumentiert.
- `load_semisupervised` ist leakage-frei (Run-Level, Scaler nur auf Gutdaten).

## Verifikation
```bash
uv run python notebooks/06_deepsad_semisupervised.py   # Exit 0; reports/06_deepsad_*.png
```

## Konfliktvermeidung
`data.py` nur **additiv** (`load_semisupervised` anhängen; T02 hängt parallel `make_windows`
an). Detektor self-register via `FACTORIES` (kein base.py-Edit, dank T01). Eigenes Notebook `06`.

## Referenzen
[docs/methoden/ad-methoden/deep_svdd_deepsad.md](../../docs/methoden/ad-methoden/deep_svdd_deepsad.md),
[00_modellselektion_ohne_labels.md](../../docs/methoden/automl-strategien/00_modellselektion_ohne_labels.md)
(ELECT/wenige Labels).
