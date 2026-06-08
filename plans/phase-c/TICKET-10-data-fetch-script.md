# TICKET-10: Daten-Fetch/Convert-Skript

**Stream:** C (Infrastruktur) · **Priorität:** niedrig · **Aufwand:** S · **Abhängigkeiten:** – · **Status:** offen

## Kontext
AutoML-für-Anomaliedetection auf dem TEP (`uv`, Python 3.13). Die 4 Parquet-Dateien unter
`data/` sind git-ignored und müssen lokal vorliegen. Aktuell konvertiert
[export_rdata.R](../../export_rdata.R) nur **eine** Tabelle (`faulty_testing`). Für
Reproduzierbarkeit fehlt ein vollständiger, dokumentierter Beschaffungs-/Konvertierungsweg.

## Ziel
Den Daten-Workflow reproduzierbar machen: alle vier `.RData`→Parquet konvertieren und eine
Python-Prüfung (Vorhandensein + Schema) bereitstellen; Doku entsprechend aktualisieren.

## Zu erstellende/ändernde Dateien
- **ändern** `export_rdata.R` — alle vier Tabellen laden/exportieren (`fault_free_training`,
  `fault_free_testing`, `faulty_training`, `faulty_testing` → entsprechende Parquet-Dateien),
  Pfade wie in `automl_ad/config.py`.
- **NEU** `scripts/fetch_convert_data.py` — prüft Vorhandensein der 4 Parquet-Dateien, gibt
  Download-Hinweis (Kaggle/Harvard Dataverse) + R-Konvertierungsanleitung aus, validiert das
  **Schema** (55 Spalten: 3 Meta + 41 `xmeas` + 11 `xmv`) via pyarrow.
- **ändern (additiv)** Doku-Abschnitt in
  [docs/methoden/01_datensatz_tep.md](../../docs/methoden/01_datensatz_tep.md) §6 (genauer
  Schritt-für-Schritt-Ablauf).

## Schnittstellen-Kontrakt
- `uv run python scripts/fetch_convert_data.py --check` → Exit 0, wenn alle 4 Dateien vorhanden
  und Schema korrekt; sonst klare Fehlermeldung + Anleitung (kein Crash).
- Keine automatischen Downloads ohne Nutzeraktion (Kaggle/Dataverse brauchen Auth) — nur
  Hinweise/Validierung. Reuse `automl_ad.config` für Pfade/Spalten.

## Akzeptanzkriterien
- `export_rdata.R` exportiert alle vier Parquet-Dateien.
- `scripts/fetch_convert_data.py --check` meldet bei vorhandenen Daten „OK" + Schema-Bestätigung,
  bei fehlenden Daten eine verständliche Anleitung (Exit ≠ 0, aber kein Traceback).
- Doku-Schritte sind vollständig und korrekt (11 XMV, Onset-Hinweis).

## Verifikation
```bash
uv run python scripts/fetch_convert_data.py --check
```

## Konfliktvermeidung
Eigene **neue** Datei `scripts/fetch_convert_data.py`; `export_rdata.R` und der Doku-Abschnitt
gehören thematisch allein diesem Ticket.

## Referenzen
[docs/methoden/01_datensatz_tep.md](../../docs/methoden/01_datensatz_tep.md),
[automl_ad/config.py](../../automl_ad/config.py).
