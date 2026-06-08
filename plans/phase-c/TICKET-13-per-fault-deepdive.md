# TICKET-13: Per-Fault-Deep-Dive (Methode × Fehlertyp)

**Stream:** D (Präsentation) · **Priorität:** mittel · **Aufwand:** M · **Abhängigkeiten:** weich A (nicht blockierend) · **Status:** offen

## Kontext
AutoML-für-Anomaliedetection auf dem TEP (`uv`, Python 3.13, PyOD-Interface). TEP hat 20
Fehlertypen mit sehr unterschiedlicher Detektierbarkeit (leicht: IDV 1/2/6/7; schwer: IDV
3/9/15). Bisher zeigt nur [eval/plots.py](../../automl_ad/eval/plots.py)
`per_fault_recall_heatmap` den Recall **eines** Detektors je Fehler. Eine vergleichende
Methode × Fehler-Sicht fehlt — sie ist für die Präsentation sehr aussagekräftig.

## Ziel
Eine Übersicht „welche Methode fängt welchen Fehler": Recall je (Detektor × Fehlertyp) über
**alle** Registry-Detektoren als Heatmap, plus kurze Analyse der schweren Fehler.

## Zu erstellende/ändernde Dateien
- **additiv** `automl_ad/eval/plots.py` — neue Funktion
  `method_fault_heatmap(per_method_recall: dict[str, dict[int, float]], save_as=None)`
  (Zeilen = Methoden, Spalten = Fehlertypen, Werte = Recall nach Onset).
- **NEU** `notebooks/11_per_fault_deepdive.py` (marimo) — fittet alle Registry-Detektoren,
  berechnet Recall je Fehler (nur Samples `>160`) und zeigt die Heatmap + Kurzfazit.

## Schnittstellen-Kontrakt
- `method_fault_heatmap(per_method_recall, save_as=None) -> matplotlib.figure.Figure`.
- Recall je Fehler: Anteil Punkte mit `score > threshold_` unter den anomalen Punkten
  (`faultNumber==f & sample>160`).
- Detektoren über `automl_ad.detectors.available_detectors()` (Stream-A-Detektoren automatisch
  inklusive).

## Umsetzungshinweise
- Reuse die Recall-Logik aus `per_fault_recall_heatmap` (gruppieren nach `faultNumber`).
- `data.load_split` mit allen oder den Demo-Fehlern (`config.DEMO_FAULTS`); je Detektor
  `threshold_` aus `contamination` nutzen.
- Subsampling für Tempo; teure Detektoren optional ausklammerbar.

## Akzeptanzkriterien
- Heatmap mit Zeilen = Detektoren, Spalten = Fehlertypen; schwere Fehler (IDV 3/9/15) sind
  sichtbar schwächer.
- Notebook läuft headless (Exit 0), schreibt die Figur nach `reports/`.

## Verifikation
```bash
uv run python notebooks/11_per_fault_deepdive.py    # Exit 0; reports/11_per_fault_heatmap.png
```

## Konfliktvermeidung
Neue Plot-Funktion nur **additiv** an `plots.py` anhängen (keine bestehende Funktion ändern).
Eigenes Notebook `11`.

## Referenzen
[docs/methoden/01_datensatz_tep.md](../../docs/methoden/01_datensatz_tep.md) (Fehlertypen),
[02_evaluationsprotokoll.md](../../docs/methoden/02_evaluationsprotokoll.md).
