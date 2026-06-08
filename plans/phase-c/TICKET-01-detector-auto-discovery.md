# TICKET-01: Detector-Auto-Discovery

**Stream:** Enabler · **Priorität:** hoch (unblockt T02–T04) · **Aufwand:** S · **Abhängigkeiten:** – · **Status:** offen

## Kontext
AutoML-für-Anomaliedetection-Beispielprojekt auf dem Tennessee Eastman Process (TEP).
`uv`-Projekt (Python 3.13). Package `automl_ad/` standardisiert auf das PyOD-`BaseDetector`-
Interface (`fit(X)` / `decision_function(X)→score`, höher = anomaler). Detektoren werden über
eine **Registry** erzeugt: [automl_ad/detectors/base.py](../../automl_ad/detectors/base.py)
baut `REGISTRY` aktuell aus `classical.py` (`CLASSICAL_FACTORIES`) + guarded Import von
`autoencoder.py` (`AE_FACTORIES`). Jede neue Detektor-Art müsste derzeit base.py editieren —
das erzeugt Merge-Konflikte zwischen den parallelen Stream-A-Tickets.

## Ziel
base.py so refaktorieren, dass die Registry per **Auto-Discovery** entsteht: Jedes Modul in
`automl_ad/detectors/`, das ein Dict `FACTORIES: dict[str, Callable]` exportiert, wird
automatisch eingesammelt (guarded — fehlende optionale Deps wie torch dürfen den Import nicht
abbrechen). Danach ist ein neuer Detektor ein **reines Drop-in** (neue Datei mit `FACTORIES`),
ohne base.py zu ändern.

## Zu erstellende/ändernde Dateien
- **ändern** `automl_ad/detectors/base.py` — Registry via `pkgutil.iter_modules` über das
  `detectors`-Package aufbauen; pro Modul `FACTORIES` einsammeln; ImportErrors einzeln fangen.
- **ändern (minimal)** `automl_ad/detectors/classical.py` — `CLASSICAL_FACTORIES` zusätzlich
  als `FACTORIES` exportieren (Alias). `autoencoder.py` exportiert bereits `AE_FACTORIES` →
  Alias `FACTORIES = AE_FACTORIES` ergänzen.

## Schnittstellen-Kontrakt
- Öffentliche API **unverändert**: `REGISTRY: dict[str, Callable]`, `make_detector(name, **hp)`,
  `available_detectors() -> list[str]`, `AnomalyDetector`-Protocol bleiben bestehen.
- Neue Konvention: ein Detektor-Modul exportiert `FACTORIES = {"<name>": make_fn}`; `make_fn(**hp)`
  liefert einen PyOD-`BaseDetector`.
- Guarded: Module, deren Import scheitert (z. B. optionale Lib fehlt), werden übersprungen,
  nicht propagiert.

## Umsetzungshinweise
- `pkgutil.iter_modules(__path__)` + `importlib.import_module(f".{name}", package=__package__)`.
- `base.py` selbst ausschließen; nur Module mit Attribut `FACTORIES` berücksichtigen.
- Reihenfolge/Dedup: bei Namenskollision deterministisch (sortiert) und eindeutig halten.

## Akzeptanzkriterien
- `available_detectors()` liefert weiterhin mindestens `['ecod','iforest','ocsvm','pca']`
  (und `autoencoder`, falls torch installiert).
- Ein **temporäres** Dummy-Modul `detectors/_dummy.py` mit `FACTORIES={"dummy": …}` erscheint
  ohne Edit an base.py in `available_detectors()` (danach Dummy wieder entfernen).
- Fehlende optionale Dep bricht den Import von `automl_ad.detectors` nicht ab.

## Verifikation
```bash
uv run python -c "from automl_ad.detectors import available_detectors; print(available_detectors())"
uv run python notebooks/01_baselines.py      # muss weiterhin Exit 0 liefern (keine Regression)
```

## Konfliktvermeidung
**Eigentümer von `base.py`.** Dieses Ticket zuerst mergen; T02–T04 bauen darauf auf und
editieren base.py nicht mehr.

## Referenzen
[docs/methoden/00_uebersicht.md](../../docs/methoden/00_uebersicht.md) (Methoden-Matrix).
