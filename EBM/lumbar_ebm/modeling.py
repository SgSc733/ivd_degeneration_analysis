from __future__ import annotations

from typing import Any, Sequence

from interpret.glassbox import ExplainableBoostingRegressor


def build_ebm_regressor(
    *,
    cfg: dict[str, Any],
    feature_names: Sequence[str],
    feature_types: Sequence[str] | None,
) -> ExplainableBoostingRegressor:
    ebm_cfg = cfg.get("ebm", {}) or {}

    n_jobs = int(ebm_cfg.get("n_jobs", 1))

    return ExplainableBoostingRegressor(
        feature_names=list(feature_names),
        feature_types=list(feature_types) if feature_types is not None else None,
        max_bins=int(ebm_cfg.get("max_bins", 64)),
        max_interaction_bins=int(ebm_cfg.get("max_interaction_bins", 32)),
        interactions=ebm_cfg.get("interactions", 0),
        exclude=ebm_cfg.get("exclude"),
        validation_size=ebm_cfg.get("validation_size", 0.15),
        outer_bags=int(ebm_cfg.get("outer_bags", 8)),
        inner_bags=int(ebm_cfg.get("inner_bags", 0)),
        learning_rate=float(ebm_cfg.get("learning_rate", 0.04)),
        greedy_ratio=ebm_cfg.get("greedy_ratio", 10.0),
        cyclic_progress=ebm_cfg.get("cyclic_progress", False),
        smoothing_rounds=int(ebm_cfg.get("smoothing_rounds", 200)),
        interaction_smoothing_rounds=int(ebm_cfg.get("interaction_smoothing_rounds", 100)),
        max_rounds=int(ebm_cfg.get("max_rounds", 5000)),
        early_stopping_rounds=int(ebm_cfg.get("early_stopping_rounds", 100)),
        early_stopping_tolerance=float(ebm_cfg.get("early_stopping_tolerance", 1e-5)),
        min_samples_leaf=int(ebm_cfg.get("min_samples_leaf", 4)),
        min_hessian=float(ebm_cfg.get("min_hessian", 0.0)),
        reg_alpha=float(ebm_cfg.get("reg_alpha", 0.0)),
        reg_lambda=float(ebm_cfg.get("reg_lambda", 0.0)),
        max_delta_step=float(ebm_cfg.get("max_delta_step", 0.0)),
        max_leaves=int(ebm_cfg.get("max_leaves", 2)),
        monotone_constraints=ebm_cfg.get("monotone_constraints"),
        objective=str(ebm_cfg.get("objective", "rmse")),
        n_jobs=n_jobs,
        random_state=int(ebm_cfg.get("random_state", 0)),
    )


def make_feature_types(
    feature_names: Sequence[str],
    *,
    disc_level_feature_name: str,
    continuous_type: str = "continuous",
    disc_level_type: str = "nominal",
) -> list[str]:
    types: list[str] = []
    for n in feature_names:
        if n == disc_level_feature_name:
            types.append(disc_level_type)
        else:
            types.append(continuous_type)
    return types
