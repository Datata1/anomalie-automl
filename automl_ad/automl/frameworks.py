"""Fertiges AutoML-Framework für die supervise Fehlerklassifikation (Strategie 2).

Nutzt **FLAML** (statt auto-sklearn — robust unter Python 3.13) für CASH: automatische
Modellauswahl + HPO. Dient als supervised Vergleichspol zur unüberwachten AD
(„Klassifikation vs. AD", siehe docs/methoden/ad-methoden/random_forest_klassifikation.md).
"""

from __future__ import annotations

from flaml.automl import AutoML

from .. import config


def run_flaml_classification(
    X_train,
    y_train,
    time_budget: int = 60,
    metric: str = "macro_f1",
    seed: int = config.RANDOM_SEED,
) -> AutoML:
    """Trainiert ein FLAML-AutoML-Modell (CASH) für Mehrklassen-Fehlerklassifikation.

    Gibt das gefittete ``AutoML``-Objekt zurück (``.predict`` / ``.predict_proba`` /
    ``.best_estimator`` / ``.best_config``).
    """
    automl = AutoML()
    automl.fit(
        X_train,
        y_train,
        task="classification",
        metric=metric,
        time_budget=time_budget,
        seed=seed,
        verbose=0,
    )
    return automl


# --- AutoGluon (optional, py3.13 ggf. nur als Pre-Release; guarded) --------------------
def autogluon_available() -> bool:
    try:
        import autogluon.tabular  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


class _AutoGluonClassifier:
    """Dünner Wrapper, der die DataFrame-Konvertierung kapselt (sklearn-artiges Interface)."""

    def __init__(self, predictor, feature_cols):
        self.predictor = predictor
        self.feature_cols = feature_cols
        self.classes_ = list(predictor.class_labels)

    def _df(self, X):
        import pandas as pd

        return pd.DataFrame(X, columns=self.feature_cols)

    def predict(self, X):
        return self.predictor.predict(self._df(X)).to_numpy()

    def predict_proba(self, X):
        return self.predictor.predict_proba(self._df(X)).to_numpy()


def run_autogluon_classification(
    X_train,
    y_train,
    time_budget: int = 120,
    seed: int = config.RANDOM_SEED,
) -> _AutoGluonClassifier:
    """Trainiert einen AutoGluon ``TabularPredictor`` (CASH + tiefes Stacking-Ensemble).

    Guarded: setzt voraus, dass ``autogluon.tabular`` installiert ist (siehe
    ``autogluon_available``).
    """
    import pandas as pd
    from autogluon.tabular import TabularPredictor

    feature_cols = [f"f{i}" for i in range(X_train.shape[1])]
    df = pd.DataFrame(X_train, columns=feature_cols)
    df["label"] = y_train
    predictor = TabularPredictor(
        label="label", problem_type="multiclass", eval_metric="f1_macro", verbosity=0
    ).fit(df, time_limit=time_budget)
    return _AutoGluonClassifier(predictor, feature_cols)
