# TICKET-08: Lint (ruff) + CI

**Stream:** C (Infrastruktur) · **Priorität:** mittel · **Aufwand:** S · **Abhängigkeiten:** – · **Status:** offen

## Kontext
AutoML-für-Anomaliedetection auf dem TEP (`uv`-Projekt, Python 3.13, hatchling-Build, Package
`automl_ad/` + `notebooks/` + `tests/`). Aktuell gibt es keine Lint-Konfiguration und keine CI.
Für beitragsfähige, robuste Codequalität fehlen beide.

## Ziel
ruff als Linter/Formatter konfigurieren und eine schlanke GitHub-Actions-CI einrichten, die
**ohne die großen Parquet-Daten** läuft (Lint + Smoke-Import + schnelle Tests).

## Zu erstellende/ändernde Dateien
- **additiv** `pyproject.toml` — `[tool.ruff]` (line-length, select Regelsätze: E,F,I,UP,B;
  Ignorierungen für Notebooks/`__init__`-Re-Exports), `ruff` als dev-dependency
  (`uv add --dev ruff`).
- **NEU** `.github/workflows/ci.yml` — Job auf Python 3.13 mit `uv`: `uv sync`,
  `uv run ruff check .`, Smoke-Import (`uv run python -c "import automl_ad"`), `uv run pytest -m "not data"`
  (siehe T09: Marker `data` für datenabhängige Tests).

## Schnittstellen-Kontrakt
- CI ist grün **ohne** lokale Parquet-Daten (datenabhängige Tests via Marker übersprungen).
- `ruff check .` läuft lokal und in CI; Konfiguration tolerant genug, dass bestehender Code
  (Package + Notebooks) ohne große Umbauten passt (ggf. gezielte `per-file-ignores`).

## Umsetzungshinweise
- marimo-Notebooks haben Cell-Funktionen mit „ungenutzten" Argumenten (reaktive Deps) →
  `per-file-ignores` für `notebooks/*` (z. B. F401/F841/B008 lockern).
- CI minimal halten (ein Job, Cache optional). Keine GPU/torch-schweren Schritte erzwingen —
  ggf. nur Import statt Notebook-Ausführung.

## Akzeptanzkriterien
- `uv run ruff check .` läuft lokal mit Exit 0 (nach evtl. Auto-Fix `ruff check --fix`).
- CI-Workflow ist syntaktisch gültig und enthält Lint + Smoke-Import + schnelle Tests.

## Verifikation
```bash
uv run ruff check .
uv run python -c "import automl_ad; print('import ok')"
```

## Konfliktvermeidung
`pyproject.toml` nur **additiv** (`[tool.ruff]` + dev-dep). `.github/workflows/ci.yml` ist neu.
Koordination mit T09 nur über den Test-Marker `data` (in T09 definiert).

## Referenzen
README (Setup), [docs/methoden/00_uebersicht.md](../../docs/methoden/00_uebersicht.md).
