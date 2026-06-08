"""Meta-Learning-Modellselektion (Strategie 3).

Versucht, mit **MetaOD** (Zhao et al. 2021) label-frei einen Detektor zu empfehlen. Da das
Paket unter Python 3.13 evtl. nicht installierbar ist, gibt es einen **Fallback** auf die
Konsens-Heuristik aus ``selection.select_internal`` — denselben Mechanismus (label-freie
Auswahl) im Kleinen. Siehe docs/methoden/automl-strategien/meta_learning_modellselektion.md.
"""

from __future__ import annotations

from .selection import DEFAULT_CANDIDATES, select_internal


def metaod_available() -> bool:
    try:
        import metaod  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def recommend(X_train, X_eval, candidates=DEFAULT_CANDIDATES, n_selection: int = 1) -> dict:
    """Label-freie Modellempfehlung.

    Bevorzugt MetaOD; fällt sonst auf die Konsens-/Centrality-Heuristik zurück. Rückgabe:
    ``{"source": "metaod"|"consensus", "choice": <name>, "detail": ...}``.
    """
    if metaod_available():
        try:  # pragma: no cover - abhängig von optionaler Installation
            from metaod.models.predict_metaod import select_model

            selected = select_model(X_eval, n_selection=n_selection)
            return {"source": "metaod", "choice": selected, "detail": None}
        except Exception as exc:  # noqa: BLE001 - bei Inkompatibilität Fallback
            fallback_reason = str(exc)
    else:
        fallback_reason = "metaod nicht installiert"

    best, centrality = select_internal(candidates, X_train, X_eval)
    return {"source": "consensus", "choice": best, "detail": centrality, "note": fallback_reason}
