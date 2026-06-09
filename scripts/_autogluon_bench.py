"""Isolierter AutoGluon-Benchmark (läuft im separaten py3.12-venv, NICHT im Haupt-venv).

Self-contained (nur numpy/pandas/sklearn/autogluon): lädt die per
``export_supervised_for_ag.py`` gedumpten Arrays, trainiert einen AutoGluon-TabularPredictor
und schreibt Macro-F1 + binäre AD-ROC-AUC nach ``reports/autogluon_result.json``.

Ausführen:  /tmp/ag-venv/bin/python scripts/_autogluon_bench.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import f1_score, roc_auc_score

DATA = Path("/tmp/ag_data")
REPORTS = Path(__file__).resolve().parent.parent / "reports"


def main():
    X_train = np.load(DATA / "X_train.npy")
    y_train = np.load(DATA / "y_train.npy")
    X_test = np.load(DATA / "X_test.npy")
    y_test = np.load(DATA / "y_test.npy")

    cols = [f"f{i}" for i in range(X_train.shape[1])]
    df = pd.DataFrame(X_train, columns=cols)
    df["label"] = y_train

    predictor = TabularPredictor(
        label="label", problem_type="multiclass", eval_metric="f1_macro", verbosity=1
    ).fit(df, time_limit=180)

    df_test = pd.DataFrame(X_test, columns=cols)
    y_pred = predictor.predict(df_test).to_numpy()
    proba = predictor.predict_proba(df_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    p_normal = proba[0].to_numpy() if 0 in proba.columns else np.zeros(len(df_test))
    ad_auc = roc_auc_score((y_test != 0).astype(int), 1.0 - p_normal)

    result = {
        "framework": "AutoGluon",
        "macro_f1": float(macro_f1),
        "ad_roc_auc": float(ad_auc),
        "best_model": str(predictor.model_best),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "autogluon_result.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
