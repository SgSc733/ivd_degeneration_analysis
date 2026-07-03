from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lumbar_pnam.feature_selection_legacy import (
    SEGMENTS,
    RobustnessAnalysisResult,
    _align_and_clean_conditions,
    _bootstrap_stability_selection,
    _build_grade_series,
    _dedup_by_spearman_with_score,
    _elasticnet_cv_lambda,
    _feature_type_category,
    _fit_pca,
    _load_conditions,
    _normalize_feature_types,
    _normalize_force_include_features,
    _patient_weights,
    _pick_feature_groups,
    _spearman_corr_for_plot,
    _spearman_filter,
    _apply_pca,
    compute_icc_21,
    load_pfirrmann_grades,
    save_analysis_result,
)


@dataclass(frozen=True)
class FeatureSelectionDataset:
    X_split: pd.DataFrame
    conditions: dict[str, pd.DataFrame]
    gold_long: pd.DataFrame
    grade_long: pd.DataFrame
    meta: pd.DataFrame
    y: np.ndarray
    tasks: np.ndarray
    groups: np.ndarray
    task_names: list[str]
    id_col: str

    def subset_patient_ids(self, sample_indices: np.ndarray | list[int]) -> list[str]:
        idx = np.asarray(sample_indices, dtype=int)
        patient_ids = self.meta.iloc[idx]["patient_id"].astype(str)
        return patient_ids.drop_duplicates().tolist()

    def subset_conditions_by_patients(self, patient_ids: list[str]) -> dict[str, pd.DataFrame]:
        wanted = set(map(str, patient_ids))
        out: dict[str, pd.DataFrame] = {}
        for cond, df in self.conditions.items():
            mask = df.index.get_level_values(0).astype(str).isin(wanted)
            out[cond] = df.loc[mask].copy()
        return out

    def subset_grade_long_by_patients(self, patient_ids: list[str]) -> pd.DataFrame:
        wanted = set(map(str, patient_ids))
        mask = self.grade_long["case_id"].astype(str).isin(wanted)
        return self.grade_long.loc[mask].copy()


def load_feature_selection_dataset(
    *,
    unperturbed_csv: str | Path,
    perturbed_csv: str | Path,
    pfirrmann_csv: str | Path,
    segment_levels: list[str] | None = None,
    id_col: str = "case_id_层级",
) -> FeatureSelectionDataset:
    segments = list(segment_levels or SEGMENTS)
    conditions = _load_conditions(
        unperturbed_csv=unperturbed_csv,
        perturbed_csv=perturbed_csv,
        segments=segments,
    )
    gold_long = conditions["gold"].copy()
    grade_long = load_pfirrmann_grades(pfirrmann_csv=pfirrmann_csv, segments=segments)
    y_series = _build_grade_series(grade_long, gold_long.index).astype(int)

    case_id = gold_long.index.get_level_values(0).astype(str)
    disc_level = gold_long.index.get_level_values(1).astype(str)
    sample_ids = pd.Index(case_id + "_" + disc_level, name=id_col)

    meta = pd.DataFrame(
        {
            id_col: sample_ids.astype(str),
            "patient_id": case_id.to_numpy(),
            "disc_level": disc_level.to_numpy(),
        },
        index=sample_ids,
    )

    x_split = gold_long.copy()
    x_split.index = sample_ids

    level_to_task = {lvl: i for i, lvl in enumerate(segments)}
    tasks = np.asarray([level_to_task[str(v)] for v in disc_level.tolist()], dtype=int)
    groups = case_id.to_numpy(dtype=object)

    return FeatureSelectionDataset(
        X_split=x_split,
        conditions=conditions,
        gold_long=gold_long,
        grade_long=grade_long,
        meta=meta,
        y=y_series.to_numpy(dtype=int, copy=False),
        tasks=tasks,
        groups=groups,
        task_names=segments,
        id_col=id_col,
    )


@dataclass
class FoldFeatureSelector:
    feature_types: list[str]
    id_col: str = "case_id_层级"
    enable_stability_selection: bool = True
    force_include_features: list[str] = field(default_factory=list)
    enable_pca: bool = True
    pca_eta: float = 0.95
    pca_m_cap: int = 50
    deep_feature_prefixes: list[str] = field(default_factory=lambda: ["Deep_"])
    tensor_feature_prefixes: list[str] = field(default_factory=lambda: ["Tucker_", "Tensor_", "tensor_", "TENSOR_"])
    icc_threshold: float = 0.60
    alpha_fdr: float = 0.05
    rho_min: float = 0.0
    dup_corr_threshold: float = 0.95
    enet_l1_ratio: float = 0.8
    enable_auto_lambda_cv: bool = True
    lambda_value: float = 0.01
    lambda_cv_folds: int = 5
    lambda_cv_n_alphas: int = 100
    lambda_cv_epsilon: float = 0.01
    lambda_cv_use_1se: bool = False
    enable_lambda_size_tuning: bool = False
    bootstrap_B: int = 100
    stability_delta: float = 1e-4
    stability_tau: float = 0.30
    k_max: int | None = None
    segments: list[str] = field(default_factory=lambda: list(SEGMENTS))
    result_: RobustnessAnalysisResult | None = field(default=None, init=False, repr=False)
    cleaned_feature_columns_: list[str] = field(default_factory=list, init=False, repr=False)
    proc_feature_columns_: list[str] = field(default_factory=list, init=False, repr=False)
    deep_cols_: list[str] = field(default_factory=list, init=False, repr=False)
    tensor_cols_: list[str] = field(default_factory=list, init=False, repr=False)
    deep_scaler_: Any = field(default=None, init=False, repr=False)
    deep_pca_: Any = field(default=None, init=False, repr=False)
    tensor_scaler_: Any = field(default=None, init=False, repr=False)
    tensor_pca_: Any = field(default=None, init=False, repr=False)
    final_features_: list[str] = field(default_factory=list, init=False, repr=False)
    force_include_normalized_: list[str] = field(default_factory=list, init=False, repr=False)
    cleaned_feature_fill_values_: pd.Series = field(default_factory=lambda: pd.Series(dtype=float), init=False, repr=False)

    @classmethod
    def from_config(cls, feature_cfg: dict[str, Any], *, segments: list[str], id_col: str = "case_id_层级") -> FoldFeatureSelector:
        return cls(
            feature_types=list(feature_cfg.get("feature_types", ["classic", "pyradiomics", "deep", "tensor"])),
            id_col=str(id_col),
            enable_stability_selection=bool(feature_cfg.get("enable_stability_selection", True)),
            force_include_features=list(feature_cfg.get("force_include_features", []) or []),
            enable_pca=bool(feature_cfg.get("enable_pca", True)),
            pca_eta=float(feature_cfg.get("pca_eta", 0.95)),
            pca_m_cap=int(feature_cfg.get("pca_m_cap", 50)),
            deep_feature_prefixes=list(feature_cfg.get("deep_feature_prefixes", ["Deep_"])),
            tensor_feature_prefixes=list(
                feature_cfg.get("tensor_feature_prefixes", ["Tucker_", "Tensor_", "tensor_", "TENSOR_"])
            ),
            icc_threshold=float(feature_cfg.get("icc_threshold", 0.60)),
            alpha_fdr=float(feature_cfg.get("alpha_fdr", 0.05)),
            rho_min=float(feature_cfg.get("rho_min", 0.0)),
            dup_corr_threshold=float(feature_cfg.get("dup_corr_threshold", 0.95)),
            enet_l1_ratio=float(feature_cfg.get("enet_l1_ratio", 0.80)),
            enable_auto_lambda_cv=bool(feature_cfg.get("enable_auto_lambda_cv", True)),
            lambda_value=float(feature_cfg.get("lambda_value", 0.01)),
            lambda_cv_folds=int(feature_cfg.get("lambda_cv_folds", 5)),
            lambda_cv_n_alphas=int(feature_cfg.get("lambda_cv_n_alphas", 100)),
            lambda_cv_epsilon=float(feature_cfg.get("lambda_cv_epsilon", 0.01)),
            lambda_cv_use_1se=bool(feature_cfg.get("lambda_cv_use_1se", False)),
            enable_lambda_size_tuning=bool(feature_cfg.get("enable_lambda_size_tuning", False)),
            bootstrap_B=int(feature_cfg.get("bootstrap_B", 100)),
            stability_delta=float(feature_cfg.get("stability_delta", 1e-4)),
            stability_tau=float(feature_cfg.get("stability_tau", 0.30)),
            k_max=None if int(feature_cfg.get("k_max", 0) or 0) <= 0 else int(feature_cfg.get("k_max")),
            segments=list(segments),
        )

    def fit(
        self,
        *,
        conditions_raw: dict[str, pd.DataFrame],
        grade_long_df: pd.DataFrame,
    ) -> RobustnessAnalysisResult:
        feature_types_norm = _normalize_feature_types(self.feature_types)
        types_set = set(feature_types_norm)
        force_norm = _normalize_force_include_features(self.force_include_features, segments=self.segments)

        gold_full = conditions_raw["gold"].copy()
        conditions_proc_raw: dict[str, pd.DataFrame] = {}
        for cond, df in conditions_raw.items():
            keep_cols = [c for c in df.columns if _feature_type_category(c) in types_set]
            conditions_proc_raw[cond] = df.loc[:, keep_cols].copy()

        conditions, dropped_nan, dropped_const, n_features_initial = _align_and_clean_conditions(conditions_proc_raw)
        n_features_after_cleaning = int(conditions["gold"].shape[1])
        if n_features_after_cleaning <= 0:
            raise ValueError("按 feature_types 过滤并清洗后无可用特征列。")

        self.cleaned_feature_columns_ = list(conditions["gold"].columns)
        self.cleaned_feature_fill_values_ = conditions["gold"].median(axis=0).astype(float)
        cond_names = ["gold"] + [c for c in conditions.keys() if c != "gold"]

        deep_info: pd.DataFrame | None = None
        tensor_info: pd.DataFrame | None = None
        self.deep_cols_ = []
        self.tensor_cols_ = []
        self.deep_scaler_ = None
        self.deep_pca_ = None
        self.tensor_scaler_ = None
        self.tensor_pca_ = None

        if self.enable_pca:
            deep_cols, tensor_cols, _ = _pick_feature_groups(
                conditions["gold"].columns.tolist(),
                deep_prefixes=self.deep_feature_prefixes,
                tensor_prefixes=self.tensor_feature_prefixes,
            )
            self.deep_cols_ = list(deep_cols)
            self.tensor_cols_ = list(tensor_cols)
            self.deep_scaler_, self.deep_pca_, deep_info = _fit_pca(
                conditions["gold"].loc[:, deep_cols],
                eta=self.pca_eta,
                m_cap=self.pca_m_cap,
            )
            self.tensor_scaler_, self.tensor_pca_, tensor_info = _fit_pca(
                conditions["gold"].loc[:, tensor_cols],
                eta=self.pca_eta,
                m_cap=self.pca_m_cap,
            )
            for cond_name in cond_names:
                df = conditions[cond_name]
                if self.deep_scaler_ is not None and self.deep_pca_ is not None and self.deep_cols_:
                    df = _apply_pca(
                        df,
                        cols=self.deep_cols_,
                        scaler=self.deep_scaler_,
                        pca=self.deep_pca_,
                        out_prefix="DeepPCA_",
                    )
                if self.tensor_scaler_ is not None and self.tensor_pca_ is not None and self.tensor_cols_:
                    df = _apply_pca(
                        df,
                        cols=self.tensor_cols_,
                        scaler=self.tensor_scaler_,
                        pca=self.tensor_pca_,
                        out_prefix="TensorPCA_",
                    )
                conditions[cond_name] = df

            common_cols = None
            for df in conditions.values():
                common_cols = df.columns if common_cols is None else common_cols.intersection(df.columns)
            if common_cols is None or len(common_cols) == 0:
                raise RuntimeError("PCA 后未得到任何可用特征列。")
            for cond_name in cond_names:
                conditions[cond_name] = conditions[cond_name].loc[:, common_cols]

        gold_x = conditions["gold"]
        self.proc_feature_columns_ = list(gold_x.columns)
        y = _build_grade_series(grade_long_df, gold_x.index)

        mats = [conditions[c] for c in cond_names]
        icc = compute_icc_21(mats, condition_names=cond_names)
        icc_keep = icc.index[(~icc.isna()) & (icc >= self.icc_threshold)].tolist()
        if not icc_keep:
            raise ValueError("ICC 鲁棒性筛选后无特征保留。")

        rho_s, p_s, q_s, keep_by_spearman = _spearman_filter(
            gold_x,
            y,
            alpha_fdr=self.alpha_fdr,
            rho_min=self.rho_min,
        )
        if not keep_by_spearman:
            raise ValueError("Spearman+FDR 相关性预筛后无特征保留。")

        icc_set = set(icc_keep)
        s_rel = [f for f in keep_by_spearman if f in icc_set]
        if not s_rel:
            raise ValueError("ICC 与 Spearman 交集后无特征保留。")

        rel_x = gold_x.loc[:, s_rel]
        score = (rho_s.abs() * icc).rename("composite_score")
        dedup_audit = _dedup_by_spearman_with_score(rel_x, score=score, threshold=self.dup_corr_threshold)
        s_nr = dedup_audit.kept_features
        if not s_nr:
            raise ValueError("去冗余后无特征保留。")

        corr_pre_feats, corr_pre = _spearman_corr_for_plot(rel_x, features=s_rel, score=score, max_features=80)
        corr_post_feats, corr_post = _spearman_corr_for_plot(rel_x, features=s_nr, score=score, max_features=80)

        lambda_table: pd.DataFrame | None = None
        lambda_cv_selected: float | None = None
        lambda_size_tuning_table: pd.DataFrame | None = None

        if self.enable_stability_selection:
            groups = pd.Series(gold_x.index.get_level_values(0).astype(str), index=gold_x.index, name="case_id")
            seg_s = pd.Series(gold_x.index.get_level_values(1).astype(str), index=gold_x.index, name="segment")
            sample_weight = _patient_weights(gold_x.index)
            x_sel = gold_x.loc[:, s_nr]

            if self.enable_auto_lambda_cv:
                alpha_star, lambda_table = _elasticnet_cv_lambda(
                    x_sel,
                    y,
                    groups=groups,
                    segments=seg_s,
                    sample_weight=sample_weight,
                    l1_ratio=self.enet_l1_ratio,
                    n_folds=self.lambda_cv_folds,
                    n_alphas=self.lambda_cv_n_alphas,
                    epsilon=self.lambda_cv_epsilon,
                    use_1se=self.lambda_cv_use_1se,
                )
                lambda_mode = "auto_cv"
                lambda_cv_selected = float(alpha_star)
                lambda_used = float(alpha_star)
                cv_folds: int | None = int(self.lambda_cv_folds)
                cv_n_alphas: int | None = int(self.lambda_cv_n_alphas)
                cv_eps: float | None = float(self.lambda_cv_epsilon)
                cv_1se: bool | None = bool(self.lambda_cv_use_1se)
            else:
                lambda_mode = "manual"
                lambda_used = float(self.lambda_value)
                cv_folds = None
                cv_n_alphas = None
                cv_eps = None
                cv_1se = None

            lambda_size_tuning_enabled = bool(self.enable_lambda_size_tuning) and lambda_mode == "auto_cv"
            pi_s, mean_abs_s, stable_sorted = _bootstrap_stability_selection(
                x_sel,
                y,
                groups=groups,
                segments=seg_s,
                sample_weight=sample_weight,
                alpha=lambda_used,
                l1_ratio=self.enet_l1_ratio,
                B=self.bootstrap_B,
                delta=self.stability_delta,
                tau=self.stability_tau,
            )

            k_target = int(self.k_max) if self.k_max is not None and int(self.k_max) > 0 else None
            if lambda_size_tuning_enabled and k_target is not None and lambda_table is not None and not lambda_table.empty:
                alpha_grid = lambda_table["alpha"].astype(float).to_numpy()
                if "selected" in lambda_table.columns and bool(lambda_table["selected"].astype(bool).any()):
                    idx_star = int(lambda_table.index[lambda_table["selected"].astype(bool)].to_list()[0])
                else:
                    idx_star = int(np.argmin(np.abs(alpha_grid - float(lambda_used))))

                trace: list[dict[str, Any]] = []

                def _record(alpha: float, n_stable: int, note: str) -> None:
                    trace.append({"alpha": float(alpha), "n_stable": int(n_stable), "note": str(note)})

                _record(float(lambda_used), int(len(stable_sorted)), "start")
                best_alpha = float(lambda_used)
                best_pi = pi_s
                best_mean_abs = mean_abs_s
                best_stable = stable_sorted

                if len(best_stable) > int(k_target):
                    for idx in range(int(idx_star) - 1, -1, -1):
                        alpha_candidate = float(alpha_grid[idx])
                        pi_t, mean_abs_t, stable_t = _bootstrap_stability_selection(
                            x_sel,
                            y,
                            groups=groups,
                            segments=seg_s,
                            sample_weight=sample_weight,
                            alpha=alpha_candidate,
                            l1_ratio=self.enet_l1_ratio,
                            B=self.bootstrap_B,
                            delta=self.stability_delta,
                            tau=self.stability_tau,
                        )
                        _record(alpha_candidate, int(len(stable_t)), "increase_lambda")
                        if len(stable_t) <= int(k_target):
                            best_alpha, best_pi, best_mean_abs, best_stable = alpha_candidate, pi_t, mean_abs_t, stable_t
                            break
                elif len(best_stable) == 0:
                    for idx in range(int(idx_star) + 1, int(len(alpha_grid))):
                        alpha_candidate = float(alpha_grid[idx])
                        pi_t, mean_abs_t, stable_t = _bootstrap_stability_selection(
                            x_sel,
                            y,
                            groups=groups,
                            segments=seg_s,
                            sample_weight=sample_weight,
                            alpha=alpha_candidate,
                            l1_ratio=self.enet_l1_ratio,
                            B=self.bootstrap_B,
                            delta=self.stability_delta,
                            tau=self.stability_tau,
                        )
                        _record(alpha_candidate, int(len(stable_t)), "decrease_lambda")
                        if stable_t:
                            best_alpha, best_pi, best_mean_abs, best_stable = alpha_candidate, pi_t, mean_abs_t, stable_t
                            break

                lambda_used = float(best_alpha)
                pi_s, mean_abs_s, stable_sorted = best_pi, best_mean_abs, best_stable
                lambda_size_tuning_table = pd.DataFrame(trace, columns=["alpha", "n_stable", "note"])

            final_proc = list(stable_sorted)
            if self.k_max is not None and int(self.k_max) > 0 and len(final_proc) > int(self.k_max):
                final_proc = final_proc[: int(self.k_max)]

            stable_pi = pi_s
            stable_mean_abs = mean_abs_s
        else:
            lambda_mode = "skipped"
            lambda_used = float("nan")
            cv_folds = None
            cv_n_alphas = None
            cv_eps = None
            cv_1se = None
            lambda_size_tuning_enabled = False
            stable_pi = pd.Series(dtype=float)
            stable_mean_abs = pd.Series(dtype=float)
            final_proc = list(s_nr)

        gold_full_aligned = gold_full.reindex(gold_x.index)
        gold_out = gold_x
        force_to_add = [f for f in force_norm if f not in gold_out.columns and f in gold_full_aligned.columns]
        if force_to_add:
            gold_out = pd.concat([gold_out, gold_full_aligned.loc[:, force_to_add]], axis=1)

        final_all = list(final_proc)
        force_added: list[str] = []
        for feat in force_norm:
            if feat in final_all:
                continue
            if feat in gold_out.columns:
                final_all.append(feat)
                force_added.append(feat)

        x_out = gold_out.loc[:, final_all]
        case_id = gold_out.index.get_level_values(0).astype(str)
        segment = gold_out.index.get_level_values(1).astype(str)
        case_level = pd.Series(case_id + "_" + segment, index=gold_out.index, name=self.id_col)
        final_model_input = pd.concat([case_level, x_out], axis=1).reset_index(drop=True)

        pass_topk = pd.Series([pd.NA] * len(dedup_audit.pass_dedup), index=dedup_audit.pass_dedup.index, dtype="boolean")
        final_set = set(final_all)
        for feat in s_nr:
            pass_topk.loc[feat] = feat in final_set

        self.force_include_normalized_ = list(force_norm)
        self.final_features_ = list(final_all)
        self.result_ = RobustnessAnalysisResult(
            conditions=cond_names,
            n_samples=int(gold_out.shape[0]),
            segments=list(self.segments),
            feature_types=list(feature_types_norm),
            stability_selection_enabled=bool(self.enable_stability_selection),
            force_include_features=list(force_norm),
            force_included_added=list(force_added),
            n_features_initial=int(n_features_initial),
            n_features_after_cleaning=int(n_features_after_cleaning),
            dropped_nan_features=dropped_nan,
            dropped_constant_features=dropped_const,
            pca_enabled=bool(self.enable_pca),
            pca_eta=float(self.pca_eta),
            pca_m_cap=int(self.pca_m_cap),
            pca_deep_info=deep_info,
            pca_tensor_info=tensor_info,
            deep_feature_prefixes=list(map(str, self.deep_feature_prefixes)),
            tensor_feature_prefixes=list(map(str, self.tensor_feature_prefixes)),
            spearman_rho=rho_s,
            spearman_p=p_s,
            spearman_q=q_s,
            spearman_alpha_fdr=float(self.alpha_fdr),
            spearman_rho_min=float(self.rho_min),
            selected_by_spearman=keep_by_spearman,
            icc=icc,
            icc_threshold=float(self.icc_threshold),
            robust_features=icc_keep,
            composite_score=score,
            dup_corr_threshold=float(self.dup_corr_threshold),
            enet_l1_ratio=float(self.enet_l1_ratio),
            lambda_mode=lambda_mode,
            lambda_value=float(lambda_used),
            lambda_cv_folds=cv_folds,
            lambda_cv_n_alphas=cv_n_alphas,
            lambda_cv_epsilon=cv_eps,
            lambda_cv_use_1se=cv_1se,
            lambda_cv_table=lambda_table,
            lambda_cv_selected=(float(lambda_cv_selected) if lambda_cv_selected is not None else None),
            lambda_size_tuning_enabled=bool(lambda_size_tuning_enabled),
            lambda_size_tuning_table=lambda_size_tuning_table,
            bootstrap_B=int(self.bootstrap_B),
            stability_delta=float(self.stability_delta),
            stability_tau=float(self.stability_tau),
            stable_pi=stable_pi,
            stable_mean_abs_beta=stable_mean_abs,
            k_max=(int(self.k_max) if self.k_max is not None else None),
            final_features=final_all,
            final_model_input=final_model_input,
            pass_dedup=dedup_audit.pass_dedup,
            pass_topk=pass_topk,
            dedup_removed_by=dedup_audit.removed_by,
            dedup_removed_corr=dedup_audit.removed_corr,
            dedup_removed_abs_corr=dedup_audit.removed_abs_corr,
            dedup_removed_order=dedup_audit.removed_order,
            dedup_max_abs_corr=dedup_audit.max_abs_corr,
            dedup_n_corr_over_threshold=dedup_audit.n_corr_over_threshold,
            dedup_decisions=dedup_audit.decisions,
            corr_pre_dedup_features=corr_pre_feats,
            corr_post_dedup_features=corr_post_feats,
            corr_pre_dedup=corr_pre,
            corr_post_dedup=corr_post,
        )
        return self.result_

    def _apply_processing_to_gold(self, gold_subset: pd.DataFrame) -> pd.DataFrame:
        if self.result_ is None:
            raise RuntimeError("FoldFeatureSelector 尚未 fit。")
        df = gold_subset.copy()
        df = df.loc[:, [c for c in self.cleaned_feature_columns_ if c in df.columns]]
        df = df.replace([np.inf, -np.inf], np.nan)
        if not self.cleaned_feature_fill_values_.empty:
            df = df.fillna(self.cleaned_feature_fill_values_.reindex(df.columns))
        if self.enable_pca:
            if self.deep_scaler_ is not None and self.deep_pca_ is not None and self.deep_cols_:
                df = _apply_pca(df, cols=self.deep_cols_, scaler=self.deep_scaler_, pca=self.deep_pca_, out_prefix="DeepPCA_")
            if self.tensor_scaler_ is not None and self.tensor_pca_ is not None and self.tensor_cols_:
                df = _apply_pca(
                    df,
                    cols=self.tensor_cols_,
                    scaler=self.tensor_scaler_,
                    pca=self.tensor_pca_,
                    out_prefix="TensorPCA_",
                )
        return df.reindex(columns=self.proc_feature_columns_)

    def transform_gold(self, gold_subset: pd.DataFrame) -> pd.DataFrame:
        if self.result_ is None:
            raise RuntimeError("FoldFeatureSelector 尚未 fit。")

        gold_subset = gold_subset.copy()
        gold_proc = self._apply_processing_to_gold(gold_subset)
        gold_out = gold_proc

        force_to_add = [
            feat for feat in self.force_include_normalized_ if feat not in gold_out.columns and feat in gold_subset.columns
        ]
        if force_to_add:
            gold_out = pd.concat([gold_out, gold_subset.loc[:, force_to_add]], axis=1)

        x_out = gold_out.loc[:, self.final_features_]
        case_id = gold_subset.index.get_level_values(0).astype(str)
        segment = gold_subset.index.get_level_values(1).astype(str)
        case_level = pd.Series(case_id + "_" + segment, index=gold_subset.index, name=self.id_col)
        return pd.concat([case_level, x_out], axis=1).reset_index(drop=True)

    def save_audit(
        self,
        output_dir: str | Path,
        *,
        pfirrmann_csv: str | Path | None = None,
        statistics_csv: str | Path | None = None,
    ) -> None:
        if self.result_ is None:
            raise RuntimeError("FoldFeatureSelector 尚未 fit。")
        save_analysis_result(
            self.result_,
            output_dir=output_dir,
            pfirrmann_csv=pfirrmann_csv,
            statistics_csv=statistics_csv,
        )
