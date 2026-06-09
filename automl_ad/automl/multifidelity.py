"""Multi-Fidelity-HPO (TICKET-05): Successive Halving / Hyperband via Optuna.

**Budget = Trainings-Subsample-Größe** (universell für alle Detektoren): Pro Trial wird die
Konfiguration auf wachsenden Subsamples ausgewertet; aussichtslose Trials werden früh geprunt.
So lassen sich Random-Search, Bayesian Optimization (TPE) und Hyperband direkt vergleichen.
Reuse: ``hpo.suggest_params`` (Suchraum), ``hpo``-Metriken.
"""

from __future__ import annotations

import time

import optuna
from sklearn.metrics import average_precision_score, roc_auc_score

from ..detectors import make_detector
from .hpo import suggest_params

optuna.logging.set_verbosity(optuna.logging.WARNING)

_METRICS = {"roc_auc": roc_auc_score, "pr_auc": average_precision_score}


def _budgets(full: int, n_rungs: int, reduction_factor: int) -> list[int]:
    """Geometrisch wachsende Budget-Stufen bis ``full`` (z. B. full/27, full/9, full/3, full)."""
    return [max(64, int(full / (reduction_factor ** (n_rungs - 1 - i)))) for i in range(n_rungs)]


def run_multifidelity(
    name: str,
    X_train_good,
    X_val,
    y_val,
    *,
    n_trials: int = 30,
    max_samples: int | None = None,
    n_rungs: int = 4,
    reduction_factor: int = 3,
    sampler: str = "tpe",
    multifidelity: bool = True,
    metric: str = "roc_auc",
    seed: int = 0,
) -> tuple[optuna.Study, dict]:
    """Optimiert ``name`` und gibt ``(study, info)`` zurück.

    - ``multifidelity=True`` + ``sampler="tpe"``  → **Hyperband/BOHB**-artig.
    - ``multifidelity=False`` + ``sampler="random"`` → Random Search (Baseline).
    - ``multifidelity=False`` + ``sampler="tpe"``  → reine Bayesian Optimization.

    ``info`` enthält ``best_value``, ``best_params``, ``elapsed_s``, ``n_pruned``, ``n_complete``.
    """
    score_fn = _METRICS[metric]
    full = max_samples if max_samples is not None else len(X_train_good)
    budgets = _budgets(full, n_rungs, reduction_factor) if multifidelity else [full]

    def objective(trial: optuna.Trial) -> float:
        hp = suggest_params(trial, name)
        last = 0.0
        for step, b in enumerate(budgets):
            det = make_detector(name, **hp).fit(X_train_good[:b])
            last = float(score_fn(y_val, det.decision_function(X_val)))
            trial.report(last, step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return last

    sampler_obj = (
        optuna.samplers.TPESampler(seed=seed)
        if sampler == "tpe"
        else optuna.samplers.RandomSampler(seed=seed)
    )
    pruner_obj = (
        optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=len(budgets), reduction_factor=reduction_factor
        )
        if multifidelity
        else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="maximize", sampler=sampler_obj, pruner=pruner_obj)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - t0

    info = {
        "best_value": float(study.best_value),
        "best_params": study.best_params,
        "elapsed_s": elapsed,
        "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "n_complete": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
    }
    return study, info
