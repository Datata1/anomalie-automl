"""marimo-App: Multi-Fidelity-HPO (TICKET-05).

Vergleicht **Random Search**, **Bayesian Optimization (TPE)** und **Hyperband** (Multi-Fidelity
über Trainings-Subsample-Budget) auf demselben Detektor. Botschaft: Hyperband prunt
aussichtslose Konfigurationen früh und erreicht vergleichbare Qualität effizienter.
Headless: ``uv run python notebooks/08_multifidelity_hpo.py``.
"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from automl_ad.automl.multifidelity import run_multifidelity
    from automl_ad.data import load_split, load_validation
    from automl_ad.eval.plots import grouped_bars

    return grouped_bars, load_split, load_validation, mo, np, run_multifidelity


@app.cell
def _(mo):
    mo.md(
        """
        # Multi-Fidelity-HPO: Random vs. BO vs. Hyperband

        Budget = Trainings-Subsample-Größe. Validierung test-disjunkt, auf ~4000 Punkte gekürzt.
        """
    )
    return


@app.cell
def _(load_split, load_validation, np):
    split = load_split(n_train_good_runs=12, n_test_good_runs=10, n_test_fault_runs=10)
    Xv, yv, _ = load_validation(split, n_good_runs=5, n_fault_runs=5, source="testing")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xv), size=min(4000, len(Xv)), replace=False)
    X_val, y_val = Xv[idx], yv[idx]
    split, X_val.shape
    return X_val, split, y_val


@app.cell
def _(X_val, run_multifidelity, split, y_val):
    common = dict(name="iforest", X_train_good=split.X_train_good, X_val=X_val, y_val=y_val, n_trials=24)
    _, info_random = run_multifidelity(**common, sampler="random", multifidelity=False)
    _, info_bo = run_multifidelity(**common, sampler="tpe", multifidelity=False)
    _, info_hb = run_multifidelity(**common, sampler="tpe", multifidelity=True)

    summary = {
        "Random": info_random,
        "BO (TPE)": info_bo,
        "Hyperband": info_hb,
    }
    {k: {"best": round(v["best_value"], 4), "elapsed_s": round(v["elapsed_s"], 1),
         "pruned": v["n_pruned"], "complete": v["n_complete"]} for k, v in summary.items()}
    return (summary,)


@app.cell
def _(grouped_bars, summary):
    grouped_bars(
        {k: {"best_val_auc": v["best_value"]} for k, v in summary.items()},
        ylabel="best_val_auc", title="Bester Val-ROC-AUC je Strategie",
        save_as="08_multifidelity_best_value.png",
    )
    return


@app.cell
def _(grouped_bars, summary):
    # Effizienz: Wandzeit + geprunte Trials (Hyperband prunt aussichtslose Configs).
    grouped_bars(
        {k: {"elapsed_s": v["elapsed_s"]} for k, v in summary.items()},
        ylabel="elapsed_s", title="Wandzeit je Strategie (s)",
        save_as="08_multifidelity_elapsed.png",
    )
    return


if __name__ == "__main__":
    app.run()
