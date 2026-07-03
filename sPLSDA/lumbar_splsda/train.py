from __future__ import annotations

import itertools
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lumbar_splsda.data import load_model_input_csv
from lumbar_splsda.feature_selection import FeatureSelectionDataset, FoldFeatureSelector
from lumbar_splsda.metrics import OrdinalMetrics, compute_ordinal_metrics
from lumbar_splsda.modeling import SparsePLSDA
from lumbar_splsda.preprocess import RankGaussZScore


@dataclass(frozen=True)
class FoldResult:
    fold: int
    n_train: int
    n_val: int
    metrics: OrdinalMetrics
    checkpoint_dir: Path


@dataclass(frozen=True)
class CVResult:
    fold_results: list[FoldResult]
    best_fold: int
    mean_metrics: OrdinalMetrics
    overall_metrics: OrdinalMetrics


def _set_seed(seed: int) -> None:
    np.random.seed(seed)


def _softmax_rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a = a - np.max(a, axis=1, keepdims=True)
    e = np.exp(a)
    z = np.sum(e, axis=1, keepdims=True)
    z[z == 0] = 1.0
    return e / z


def _score_j_in(metrics: OrdinalMetrics, *, eta_kappa: float, eta_spearman: float, eta_acc_pm1: float) -> float:
    """
    Inner-CV objective (sPLSDA.md 4.3.3.3 example):
        J_in = -MAE + eta1*kappa + eta2*spearman + eta3*acc_pm1
    """
    return float(
        -metrics.mae
        + eta_kappa * metrics.kappa_quadratic
        + eta_spearman * metrics.spearman
        + eta_acc_pm1 * metrics.acc_pm1
    )


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    denom = float(np.sum(w))
    if denom <= 0:
        return float(np.mean(v)) if v.size > 0 else 0.0
    return float(np.sum(v * w) / denom)


def _iter_group_kfold_splits(*, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Deterministic GroupKFold-like splits with seed-controlled tie-breaking.

    Why:
    - sklearn's GroupKFold does not shuffle, so `seed` does not affect the split.
    - We want splits to be reproducible and independent of row order changes.
    """
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)  # uniq is sorted => stable
    group_counts = np.bincount(inv)

    rng = np.random.RandomState(int(seed))
    tie_break = rng.permutation(len(uniq))
    order = np.lexsort((tie_break, -group_counts))  # primary: -count, secondary: random tie-break

    fold_counts = np.zeros(int(n_splits), dtype=int)
    group_to_fold = np.empty(len(uniq), dtype=int)
    for g in order.tolist():
        min_count = int(fold_counts.min())
        candidates = np.where(fold_counts == min_count)[0]
        f = int(rng.choice(candidates))
        group_to_fold[g] = f
        fold_counts[f] += int(group_counts[g])

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for f in range(int(n_splits)):
        val_mask = group_to_fold[inv] == f
        val_idx = np.where(val_mask)[0]
        train_idx = np.where(~val_mask)[0]
        splits.append((train_idx, val_idx))
    return splits


def _fillna_with_train_median(X_train: pd.DataFrame, X_val: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fold-inner NaN imputation to avoid leakage:
    - fit median on train only
    - apply to train + val
    """
    med = X_train.median(numeric_only=True)
    X_train2 = X_train.fillna(med)
    X_val2 = X_val.fillna(med)
    return X_train2, X_val2


def _normalize_keepX(keepX: list[int], p: int) -> list[int]:
    out: list[int] = []
    for k in keepX:
        kk = int(k)
        kk = max(1, min(int(p), kk))
        out.append(kk)
    return out


def _get_axis_a(post_cfg: dict[str, Any], n_components: int) -> np.ndarray:
    """
    Get `a` used to construct the "severity axis":
        s = T a,  beta = R a

    With nested CV, H can vary by fold, so we adapt user-provided `axis_a` safely:
    - if missing: default to first component
    - if longer: truncate
    - if shorter: pad zeros
    """
    axis_a = post_cfg.get("axis_a", None)
    h = int(n_components)
    if axis_a is None:
        a = np.zeros((h,), dtype=np.float64)
        a[0] = 1.0
        return a

    a_in = np.asarray(axis_a, dtype=np.float64).reshape(-1)
    if a_in.size >= h:
        a = a_in[:h].copy()
    else:
        a = np.zeros((h,), dtype=np.float64)
        a[: a_in.size] = a_in
    if float(np.linalg.norm(a)) == 0.0:
        a[0] = 1.0
    return a


def _tuning_post_candidates(tune_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    versions = list(tune_cfg.get("version_candidates", ["segment-aligned", "global"]))
    dists = list(tune_cfg.get("dist_candidates", ["centroids.dist", "mahalanobis.dist", "max.dist"]))
    alphas = list(tune_cfg.get("alpha_candidates", [0.5, 1.0, 2.0, 4.0]))

    out: list[dict[str, Any]] = []
    for v in versions:
        for d in dists:
            if d == "max.dist":
                out.append({"version": v, "dist": d, "alpha": 1.0})
            else:
                for a in alphas:
                    out.append({"version": v, "dist": d, "alpha": float(a)})
    return out


def _tuning_fit_candidates(tune_cfg: dict[str, Any], *, p: int) -> list[dict[str, Any]]:
    h_cands = list(tune_cfg.get("H_candidates", [1, 2]))
    keepX_cands = list(tune_cfg.get("keepX_candidates", [5, 8, 10, 15, 20]))
    enforce_monotone = bool(tune_cfg.get("enforce_keepX_monotone", True))

    out: list[dict[str, Any]] = []
    for h in h_cands:
        h = int(h)
        if h <= 0:
            continue
        if h == 1:
            for k1 in keepX_cands:
                out.append({"n_components": 1, "keepX": _normalize_keepX([int(k1)], p)})
        elif h == 2:
            for k1 in keepX_cands:
                for k2 in keepX_cands:
                    if enforce_monotone and int(k2) > int(k1):
                        continue
                    out.append({"n_components": 2, "keepX": _normalize_keepX([int(k1), int(k2)], p)})
        else:
            # Generic (rare): full cartesian product, optionally monotone.
            for ks in itertools.product(keepX_cands, repeat=h):
                ks = [int(x) for x in ks]
                if enforce_monotone and any(ks[i] > ks[i - 1] for i in range(1, len(ks))):
                    continue
                out.append({"n_components": h, "keepX": _normalize_keepX(list(ks), p)})
    return out


def _nested_cv_select_params(
    *,
    X_df: pd.DataFrame,
    y: np.ndarray,
    tasks: np.ndarray,
    groups: np.ndarray,
    task_names: list[str],
    cfg: dict[str, Any],
    seed: int,
    global_classes: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """
    Nested CV selection on the *outer training fold*:
      - select psi_fit = (H, keepX) via inner GroupKFold
      - select psi_post = (version, dist, alpha) without re-fitting the model

    Returns:
      - best_fit: {"n_components": H, "keepX": [...], "eps": ...}
      - best_post: merged postprocess config (includes fixed fields + chosen version/dist/alpha)
      - grid_df: per-(psi_fit, psi_post) weighted mean inner score table for traceability
    """
    tune_cfg = cfg.get("tuning", {}) or {}
    cv_cfg = cfg.get("cv", {}) or {}
    n_in = int(tune_cfg.get("inner_n_splits", cv_cfg.get("inner_n_splits", 3)))
    if n_in < 2:
        raise ValueError("tuning.inner_n_splits must be >= 2 for nested CV.")

    eta_kappa = float(tune_cfg.get("eta_kappa", 1.0))
    eta_spearman = float(tune_cfg.get("eta_spearman", 1.0))
    eta_acc_pm1 = float(tune_cfg.get("eta_acc_pm1", 1.0))

    model_cfg = cfg.get("splsda", {}) or {}
    eps = float(model_cfg.get("eps", 1e-6))

    pre_cfg = cfg.get("preprocess", {}) or {}
    use_rankgauss = bool(pre_cfg.get("rank_gauss", True))

    base_post_cfg = cfg.get("postprocess", {}) or {}

    p = int(X_df.shape[1])
    fit_cands = _tuning_fit_candidates(tune_cfg, p=p)
    post_cands = _tuning_post_candidates(tune_cfg)

    inner_splits = _iter_group_kfold_splits(groups=groups, n_splits=n_in, seed=int(seed))

    grid_rows: list[dict[str, Any]] = []
    best_fit = None
    best_fit_score = None
    best_fit_best_post = None

    # Optional tie-break: prefer smaller H when scores are (almost) equal.
    tie_tol = float(tune_cfg.get("tie_tol", 1e-6))

    for fit in fit_cands:
        h = int(fit["n_components"])
        keepX = list(fit["keepX"])

        # Collect per post candidate scores across inner folds.
        post_scores: dict[str, list[float]] = {}
        post_weights: dict[str, list[int]] = {}

        for inner_fold, (tr_idx, va_idx) in enumerate(inner_splits):
            X_tr_df = X_df.iloc[tr_idx].reset_index(drop=True)
            X_va_df = X_df.iloc[va_idx].reset_index(drop=True)
            y_tr = np.asarray(y)[tr_idx].astype(int)
            y_va = np.asarray(y)[va_idx].astype(int)
            tasks_tr = np.asarray(tasks)[tr_idx].astype(int)
            tasks_va = np.asarray(tasks)[va_idx].astype(int)

            # Fold-inner NaN imputation (train only).
            X_tr_df, X_va_df = _fillna_with_train_median(X_tr_df, X_va_df)

            pre = RankGaussZScore(rank_gauss=use_rankgauss).fit(X_tr_df)
            X_tr = pre.transform(X_tr_df)
            X_va = pre.transform(X_va_df)

            model = SparsePLSDA(n_components=h, keepX=keepX, eps=eps).fit(X_tr, y_tr)

            for post in post_cands:
                post_key = json.dumps(post, ensure_ascii=False, sort_keys=True)
                merged_post = dict(base_post_cfg)
                merged_post.update(post)

                pred = _predict_with_postprocess(
                    model=model,
                    X_train=X_tr,
                    y_train=y_tr,
                    tasks_train=tasks_tr,
                    X_new=X_va,
                    tasks_new=tasks_va,
                    post_cfg=merged_post,
                    n_tasks=len(task_names),
                    global_classes=global_classes,
                )
                m = compute_ordinal_metrics(y_true=y_va, y_pred=pred["y_pred"], y_cont=pred["y_cont"])
                s = _score_j_in(m, eta_kappa=eta_kappa, eta_spearman=eta_spearman, eta_acc_pm1=eta_acc_pm1)

                post_scores.setdefault(post_key, []).append(float(s))
                post_weights.setdefault(post_key, []).append(int(len(va_idx)))

        # Aggregate each post candidate across inner folds (weighted by n_val).
        post_mean: dict[str, float] = {}
        for post_key, vals in post_scores.items():
            w = post_weights.get(post_key, [1] * len(vals))
            post_mean[post_key] = _weighted_mean(vals, w)

        # Pick best post for this fit.
        best_post_key = max(post_mean.keys(), key=lambda k: post_mean[k])
        fit_score = float(post_mean[best_post_key])
        best_post = json.loads(best_post_key)

        # Save grid rows (fit, post, score)
        for post_key, score in post_mean.items():
            post_cfg_row = json.loads(post_key)
            grid_rows.append(
                {
                    "H": h,
                    "keepX": json.dumps(keepX, ensure_ascii=False),
                    "version": post_cfg_row["version"],
                    "dist": post_cfg_row["dist"],
                    "alpha": float(post_cfg_row.get("alpha", 1.0)),
                    "J_in": float(score),
                }
            )

        # Select best fit across candidates.
        if best_fit is None:
            best_fit = {"n_components": h, "keepX": keepX, "eps": eps}
            best_fit_score = fit_score
            best_fit_best_post = best_post
        else:
            assert best_fit_score is not None
            if fit_score > best_fit_score + tie_tol:
                best_fit = {"n_components": h, "keepX": keepX, "eps": eps}
                best_fit_score = fit_score
                best_fit_best_post = best_post
            elif abs(fit_score - best_fit_score) <= tie_tol:
                # Tie-break: prefer smaller H (more conservative) as documented.
                if int(h) < int(best_fit["n_components"]):
                    best_fit = {"n_components": h, "keepX": keepX, "eps": eps}
                    best_fit_score = fit_score
                    best_fit_best_post = best_post

    if best_fit is None or best_fit_best_post is None or best_fit_score is None:
        raise RuntimeError("Nested CV tuning failed to select hyperparameters.")

    best_post_cfg = dict(base_post_cfg)
    best_post_cfg.update(best_fit_best_post)
    best_post_cfg["J_in_best"] = float(best_fit_score)

    grid_df = pd.DataFrame(grid_rows).sort_values(["H", "keepX", "J_in"], ascending=[True, True, False]).reset_index(drop=True)
    return best_fit, best_post_cfg, grid_df



def _save_confusion_png(cm: np.ndarray, out_path: Path, *, title: str) -> None:
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")

    # Write counts.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _safe_sign(x: float) -> float:
    return -1.0 if x < 0 else 1.0


def _align_components_to_ref(W: np.ndarray, ref_W: np.ndarray) -> np.ndarray:
    """
    Align per-component sign so cross-fold loadings are comparable.

    Only used for reporting/interpretation; does NOT change any fold's predictions.
    """
    W = np.asarray(W, dtype=np.float64).copy()
    h = W.shape[1]
    for j in range(h):
        s = float(np.dot(W[:, j], ref_W[:, j]))
        sign = _safe_sign(s)
        W[:, j] *= sign
    return W


def _compute_centroids(
    T_train: np.ndarray,
    y_train: np.ndarray,
    tasks_train: np.ndarray,
    *,
    classes: np.ndarray,
    n_tasks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      - global_centroids: (K, H)
      - task_centroids: (n_tasks, K, H) (undefined entries filled with 0)
      - task_counts: (n_tasks, K)
    """
    T_train = np.asarray(T_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)
    tasks_train = np.asarray(tasks_train, dtype=int)

    k = int(len(classes))
    h = int(T_train.shape[1])

    global_centroids = np.zeros((k, h), dtype=np.float64)
    for i, c in enumerate(classes.tolist()):
        mask = y_train == int(c)
        global_centroids[i] = T_train[mask].mean(axis=0)

    task_centroids = np.zeros((n_tasks, k, h), dtype=np.float64)
    task_counts = np.zeros((n_tasks, k), dtype=np.int64)
    for t in range(n_tasks):
        for i, c in enumerate(classes.tolist()):
            mask = (tasks_train == t) & (y_train == int(c))
            n = int(mask.sum())
            task_counts[t, i] = n
            if n > 0:
                task_centroids[t, i] = T_train[mask].mean(axis=0)
    return global_centroids, task_centroids, task_counts


def _mixed_centroids_for_tasks(
    global_centroids: np.ndarray,
    task_centroids: np.ndarray,
    task_counts: np.ndarray,
    tasks_new: np.ndarray,
    *,
    version: str,
    n_min: int,
) -> np.ndarray:
    """
    Build per-sample centroids used for distance computation.

    Returns centroids_new: (n_new, K, H)
    """
    global_centroids = np.asarray(global_centroids, dtype=np.float64)
    task_centroids = np.asarray(task_centroids, dtype=np.float64)
    task_counts = np.asarray(task_counts, dtype=np.int64)
    tasks_new = np.asarray(tasks_new, dtype=int)

    n_new = int(len(tasks_new))
    k, h = int(global_centroids.shape[0]), int(global_centroids.shape[1])
    out = np.zeros((n_new, k, h), dtype=np.float64)

    if version == "global":
        out[:] = global_centroids.reshape(1, k, h)
        return out

    if version != "segment-aligned":
        raise ValueError(f"Unknown splsda version: {version}. Expected 'global' or 'segment-aligned'.")

    n_min = int(n_min)
    for i in range(n_new):
        t = int(tasks_new[i])
        for c in range(k):
            n = int(task_counts[t, c])
            if n <= 0:
                lam = 0.0
                tc = global_centroids[c]
            else:
                lam = float(n / (n + n_min))
                tc = task_centroids[t, c]
            out[i, c] = (1.0 - lam) * global_centroids[c] + lam * tc
    return out


def _cov_inv_for_mahalanobis(
    T_train: np.ndarray,
    y_train: np.ndarray,
    *,
    classes: np.ndarray,
    use_pooled: bool,
    cov_reg: float,
) -> np.ndarray:
    T_train = np.asarray(T_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)
    classes = np.asarray(classes, dtype=int)

    h = int(T_train.shape[1])
    if use_pooled:
        # LDA-style pooled within-class covariance.
        k = int(len(classes))
        n = int(len(T_train))
        global_centroids = np.zeros((k, h), dtype=np.float64)
        counts = np.zeros((k,), dtype=np.int64)
        for i, c in enumerate(classes.tolist()):
            mask = y_train == int(c)
            counts[i] = int(mask.sum())
            global_centroids[i] = T_train[mask].mean(axis=0)

        s_w = np.zeros((h, h), dtype=np.float64)
        denom = float(max(1, n - k))
        for i, c in enumerate(classes.tolist()):
            mask = y_train == int(c)
            if int(mask.sum()) == 0:
                continue
            diff = T_train[mask] - global_centroids[i].reshape(1, -1)
            s_w += diff.T @ diff
        cov = s_w / denom
    else:
        cov = np.cov(T_train.T)

    cov = np.asarray(cov, dtype=np.float64)
    cov = cov + float(cov_reg) * np.eye(h, dtype=np.float64)
    return np.linalg.inv(cov)


def _predict_with_postprocess(
    *,
    model: SparsePLSDA,
    X_train: np.ndarray,
    y_train: np.ndarray,
    tasks_train: np.ndarray,
    X_new: np.ndarray,
    tasks_new: np.ndarray,
    post_cfg: dict[str, Any],
    n_tasks: int,
    global_classes: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Returns:
      y_pred: (n,)
      y_cont: (n,)
      proba: (n,K)
      distances: (n,K)  (only for centroid/mahalanobis; otherwise zeros)
      dummy: (n,K)      (only for max.dist; otherwise zeros)
      T_new: (n,H)
    """
    if model.classes_ is None:
        raise RuntimeError("Model is not fitted.")
    classes_fold = model.classes_.astype(int)
    global_classes = np.asarray(global_classes, dtype=int)
    k_global = int(len(global_classes))
    fold_to_global = {int(c): int(i) for i, c in enumerate(global_classes.tolist())}

    version = str(post_cfg.get("version", "segment-aligned"))
    dist = str(post_cfg.get("dist", "centroids.dist"))
    alpha = float(post_cfg.get("alpha", 1.0))
    n_min = int(post_cfg.get("n_min", 10))
    cov_reg = float(post_cfg.get("cov_reg", 1e-6))
    use_pooled = bool(post_cfg.get("use_pooled_cov", False))

    # --- max.dist branch ---
    if dist == "max.dist":
        dummy_fold = model.predict_dummy_scores(X_new)  # (n, K_fold)
        dummy_global = np.full((int(len(X_new)), k_global), -1e9, dtype=np.float64)
        for j, c in enumerate(classes_fold.tolist()):
            gi = fold_to_global.get(int(c))
            if gi is not None:
                dummy_global[:, gi] = dummy_fold[:, j]

        proba = _softmax_rows(dummy_global)
        y_cont = (proba * global_classes.reshape(1, -1)).sum(axis=1)
        y_pred = global_classes[np.argmax(dummy_global, axis=1)]

        return {
            "y_pred": y_pred.astype(int),
            "y_cont": y_cont.astype(float),
            "proba": proba.astype(float),
            "distances": np.zeros((int(len(X_new)), k_global), dtype=np.float64),
            "dummy": dummy_global.astype(float),
            "T_new": model.transform(X_new).astype(float),
            "classes_fold": classes_fold,
            "classes_global": global_classes,
        }

    # --- centroid / mahalanobis branch ---
    T_train = model.transform(X_train)
    T_new = model.transform(X_new)

    global_centroids, task_centroids, task_counts = _compute_centroids(
        T_train,
        y_train,
        tasks_train,
        classes=classes_fold,
        n_tasks=n_tasks,
    )

    centroids_new = _mixed_centroids_for_tasks(
        global_centroids,
        task_centroids,
        task_counts,
        tasks_new,
        version=version,
        n_min=n_min,
    )  # (n_new, K, H)

    diff = T_new[:, None, :] - centroids_new  # (n,K,H)
    if dist == "centroids.dist":
        distances_fold = np.linalg.norm(diff, axis=2)
    elif dist == "mahalanobis.dist":
        inv_cov = _cov_inv_for_mahalanobis(
            T_train, y_train, classes=classes_fold, use_pooled=use_pooled, cov_reg=cov_reg
        )
        distances_fold = np.sqrt(np.einsum("nkh,hh,nkh->nk", diff, inv_cov, diff))
    else:
        raise ValueError(
            f"Unknown dist: {dist}. Expected one of: max.dist, centroids.dist, mahalanobis.dist."
        )

    # Map fold distances to global K (missing classes -> huge distance -> ~0 prob).
    distances_global = np.full((int(len(X_new)), k_global), 1e9, dtype=np.float64)
    for j, c in enumerate(classes_fold.tolist()):
        gi = fold_to_global.get(int(c))
        if gi is not None:
            distances_global[:, gi] = distances_fold[:, j]

    # Soft probability from distances -> continuous output (defined on global classes).
    proba = np.exp(-alpha * distances_global)
    z = proba.sum(axis=1, keepdims=True)
    z[z == 0] = 1.0
    proba = proba / z
    y_cont = (proba * global_classes.reshape(1, -1)).sum(axis=1)

    idx = np.argmin(distances_global, axis=1)
    y_pred = global_classes[idx]

    return {
        "y_pred": y_pred.astype(int),
        "y_cont": y_cont.astype(float),
        "proba": proba.astype(float),
        "distances": distances_global.astype(float),
        "dummy": np.zeros((int(len(X_new)), k_global), dtype=np.float64),
        "T_new": T_new.astype(float),
        "classes_fold": classes_fold,
        "classes_global": global_classes,
    }


def train_group_kfold(
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    tasks: np.ndarray,
    task_names: list[str],
    groups: np.ndarray,
    classic_feature_names: list[str],
    run_dir: Path,
    cfg: dict[str, Any],
    meta: pd.DataFrame,
    id_col: str,
    feature_selection_dataset: FeatureSelectionDataset | None = None,
) -> CVResult:
    cv_cfg = cfg.get("cv", {}) or {}
    n_splits = int(cv_cfg.get("n_splits", 5))
    seed = int(cfg.get("training", {}).get("seed", 0))
    _set_seed(seed)

    model_cfg = cfg.get("splsda", {}) or {}
    eps = float(model_cfg.get("eps", 1e-6))

    pre_cfg = cfg.get("preprocess", {}) or {}
    use_rankgauss = bool(pre_cfg.get("rank_gauss", True))

    post_cfg_base = cfg.get("postprocess", {}) or {}

    tune_cfg = cfg.get("tuning", {}) or {}
    tuning_enabled = bool(tune_cfg.get("enabled", True))
    data_cfg = cfg.get("data", {}) or {}
    pooling_cfg = data_cfg.get("pooling", {}) or {}
    case_meta_cfg = data_cfg.get("case_meta_onehot", {}) or {}
    pooling_feature_types = pooling_cfg.get("feature_types", None)
    if pooling_feature_types is None:
        pooling_feature_types = None
    elif isinstance(pooling_feature_types, str):
        pooling_feature_types = [pooling_feature_types]
    else:
        pooling_feature_types = list(pooling_feature_types)
    label_col = str(data_cfg.get("label_col", "pfirrmann") or "pfirrmann")
    resolved_id_col = str(data_cfg.get("id_col", id_col))

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    fold_results: list[FoldResult] = []
    best_fold = 0
    best_kappa = -1e9

    # For cross-fold stability summaries.
    betas_by_fold: list[np.ndarray] = []
    selected_by_fold: list[np.ndarray] = []
    selected_names_by_fold: list[set[str]] = []
    beta_series_by_fold: list[pd.Series] = []
    w_by_fold: list[np.ndarray] = []
    r_by_fold: list[np.ndarray] = []
    a_by_fold: list[np.ndarray] = []
    h_by_fold: list[int] = []

    y_true_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    y_cont_all: list[np.ndarray] = []
    oof_pred_dfs: list[pd.DataFrame] = []

    classes = np.unique(y).astype(int)
    classes = np.sort(classes)

    outer_splits = _iter_group_kfold_splits(groups=groups, n_splits=n_splits, seed=seed)

    for fold, (train_idx, val_idx) in enumerate(outer_splits):
        fold_dir = run_dir / "checkpoints" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fs_summary: dict[str, Any] | None = None
        if feature_selection_dataset is not None:
            fs_dir = fold_dir / "feature_selection"
            fs_dir.mkdir(parents=True, exist_ok=True)
            fs_cfg = cfg.get("feature_selection", {}) or {}
            fs_enabled = bool(fs_cfg.get("enabled", True))

            if fs_enabled:
                selector = FoldFeatureSelector.from_config(
                    fs_cfg,
                    segments=feature_selection_dataset.task_names,
                    id_col=resolved_id_col,
                )
                train_patient_ids = feature_selection_dataset.subset_patient_ids(train_idx)
                train_conditions = feature_selection_dataset.subset_conditions_by_patients(train_patient_ids)
                train_grade_long = feature_selection_dataset.subset_grade_long_by_patients(train_patient_ids)

                fs_result = selector.fit(
                    conditions_raw=train_conditions,
                    grade_long_df=train_grade_long,
                )
                selector.save_audit(
                    fs_dir,
                    pfirrmann_csv=(cfg.get("labels", {}) or {}).get("xlsx_path"),
                    statistics_csv=fs_cfg.get("statistics_csv_path"),
                )
                with open(fs_dir / "selector.pkl", "wb") as f:
                    pickle.dump(selector, f, protocol=pickle.HIGHEST_PROTOCOL)
                selected_full_df = selector.transform_gold(feature_selection_dataset.gold_long)
                fs_summary = {
                    "enabled": True,
                    "n_features_initial": int(fs_result.n_features_initial),
                    "n_features_after_cleaning": int(fs_result.n_features_after_cleaning),
                    "n_robust_features": int(len(fs_result.robust_features)),
                    "n_spearman_features": int(len(fs_result.selected_by_spearman)),
                    "n_final_features": int(len(fs_result.final_features)),
                    "audit_dir": str(fs_dir),
                }
            else:
                gold_long = feature_selection_dataset.gold_long
                case_id_series = gold_long.index.get_level_values(0).astype(str)
                segment_series = gold_long.index.get_level_values(1).astype(str)
                sample_id = pd.Series(
                    case_id_series + "_" + segment_series,
                    index=gold_long.index,
                    name=resolved_id_col,
                )
                selected_full_df = pd.concat([sample_id, gold_long], axis=1).reset_index(drop=True)
                fs_summary = {"enabled": False, "audit_dir": str(fs_dir)}

            selected_full_df.insert(1, label_col, feature_selection_dataset.y.astype(int))
            selected_full_path = fs_dir / "fold_model_input_full.csv"
            selected_full_df.to_csv(selected_full_path, index=False, encoding="utf-8")

            loaded_fold = load_model_input_csv(
                selected_full_path,
                id_col=resolved_id_col,
                label_col=label_col,
                drop_cols=data_cfg.get("drop_cols", []),
                drop_patterns=data_cfg.get("drop_patterns", []),
                feature_order=str(data_cfg.get("feature_order", "csv")),
                classic_prefix=data_cfg.get("classic_prefix", "classic_"),
                patient_id_from_id_col=bool(data_cfg.get("patient_id_from_id_col", True)),
                patient_id_sep=str(data_cfg.get("patient_id_sep", "_")),
                level_from_id_col=bool(data_cfg.get("level_from_id_col", True)),
                level_sep=str(data_cfg.get("level_sep", "_")),
                segment_levels=list(data_cfg.get("segment_levels", feature_selection_dataset.task_names)),
                add_segment_onehot=bool(data_cfg.get("add_segment_onehot", False)),
                pooling_stats=list(pooling_cfg.get("stats", []) or []),
                pooling_feature_types=pooling_feature_types,
                pooling_pyr_prefix=str(pooling_cfg.get("pyr_prefix", "PyRadiomics_")),
                pooling_deep_prefix=str(pooling_cfg.get("deep_prefix", "DeepPCA_")),
                pooling_tensor_prefix=str(pooling_cfg.get("tensor_prefix", "TensorPCA_")),
                pooling_out_prefix=str(pooling_cfg.get("out_prefix", "pool_")),
                add_case_meta_onehot=bool(case_meta_cfg.get("enabled", False)),
                case_meta_csv_path=case_meta_cfg.get("statistics_csv_path"),
                case_meta_case_id_col=str(case_meta_cfg.get("case_id_col", "image_id")),
                case_meta_cols=list(case_meta_cfg.get("columns", []) or []),
                case_meta_onehot_prefix=str(case_meta_cfg.get("prefix", "meta_")),
                case_meta_onehot_drop_first=bool(case_meta_cfg.get("drop_first", False)),
                case_meta_on_missing=str(case_meta_cfg.get("on_missing", "error")),
            )
            loaded_fold.X.to_pickle(fold_dir / "raw_input_features.pkl")
            X_current = loaded_fold.X
            y_current = np.asarray(loaded_fold.y).astype(int)
            tasks_current = np.asarray(loaded_fold.tasks).astype(int)
            meta_current = loaded_fold.meta
            feature_names_current = list(loaded_fold.feature_names)
            classic_feature_names_current = list(loaded_fold.classic_feature_names)
        else:
            X_current = X
            y_current = np.asarray(y).astype(int)
            tasks_current = np.asarray(tasks).astype(int)
            meta_current = meta
            feature_names_current = list(X.columns)
            classic_feature_names_current = list(classic_feature_names)

        X_tr_df = X_current.iloc[train_idx].reset_index(drop=True)
        X_va_df = X_current.iloc[val_idx].reset_index(drop=True)
        y_tr = y_current[train_idx].astype(int)
        y_va = y_current[val_idx].astype(int)
        tasks_tr = tasks_current[train_idx].astype(int)
        tasks_va = tasks_current[val_idx].astype(int)
        groups_tr = np.asarray(groups)[train_idx]

        # --- Nested CV tuning on outer-train (4.3.3.3) ---
        if tuning_enabled:
            best_fit, best_post_cfg, grid_df = _nested_cv_select_params(
                X_df=X_tr_df,
                y=y_tr,
                tasks=tasks_tr,
                groups=groups_tr,
                task_names=task_names,
                cfg=cfg,
                seed=int(seed + 1000 + fold),
                global_classes=classes,
            )
            grid_df.to_csv(fold_dir / "inner_cv_grid.csv", index=False, encoding="utf-8")
            (fold_dir / "selected_hyperparams.json").write_text(
                json.dumps({"best_fit": best_fit, "best_postprocess": best_post_cfg}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        else:
            # Fallback: use fixed config (kept for fast debugging).
            h = int(model_cfg.get("n_components", 2))
            keepX_cfg = list(model_cfg.get("keepX", [10] * h))
            if len(keepX_cfg) >= h:
                keepX_cfg = keepX_cfg[:h]
            else:
                keepX_cfg = keepX_cfg + [int(keepX_cfg[-1])] * (h - len(keepX_cfg))
            best_fit = {"n_components": h, "keepX": _normalize_keepX(keepX_cfg, int(X_current.shape[1])), "eps": eps}
            best_post_cfg = dict(post_cfg_base)
            (fold_dir / "selected_hyperparams.json").write_text(
                json.dumps({"best_fit": best_fit, "best_postprocess": best_post_cfg}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # --- Preprocess (fit on outer-train only) ---
        # Fold-inner NaN imputation to avoid leakage (4.3.0 principle).
        # Save the imputer statistics for explanation/inference reproducibility.
        med = X_tr_df.median(numeric_only=True)
        pd.DataFrame({"feature": med.index.astype(str), "median": med.to_numpy(dtype=float)}).to_csv(
            fold_dir / "imputer_median.csv", index=False, encoding="utf-8"
        )
        X_tr_df = X_tr_df.fillna(med)
        X_va_df = X_va_df.fillna(med)
        pre = RankGaussZScore(rank_gauss=use_rankgauss).fit(X_tr_df)
        X_tr = pre.transform(X_tr_df)
        X_va = pre.transform(X_va_df)

        # --- Fit model ---
        n_components_fold = int(best_fit["n_components"])
        keepX_fold = list(best_fit["keepX"])
        model = SparsePLSDA(n_components=n_components_fold, keepX=keepX_fold, eps=eps).fit(X_tr, y_tr)

        # --- Predict val ---
        pred = _predict_with_postprocess(
            model=model,
            X_train=X_tr,
            y_train=y_tr,
            tasks_train=tasks_tr,
            X_new=X_va,
            tasks_new=tasks_va,
            post_cfg=best_post_cfg,
            n_tasks=len(task_names),
            global_classes=classes,
        )

        metrics = compute_ordinal_metrics(y_true=y_va, y_pred=pred["y_pred"], y_cont=pred["y_cont"])
        y_true_all.append(y_va.copy())
        y_pred_all.append(np.asarray(pred["y_pred"]).copy())
        y_cont_all.append(np.asarray(pred["y_cont"]).copy())

        # --- Save fold predictions ---
        pred_df = meta_current.iloc[val_idx].reset_index(drop=True).copy()
        pred_df["task"] = tasks_va
        pred_df["task_name"] = [task_names[t] for t in tasks_va.tolist()]
        pred_df["y_true"] = y_va
        pred_df["y_pred"] = pred["y_pred"]
        pred_df["y_cont"] = pred["y_cont"]
        for i, c in enumerate(classes.tolist()):
            pred_df[f"p_{c}"] = pred["proba"][:, i]
        if str(best_post_cfg.get("dist", "centroids.dist")) == "max.dist":
            for i, c in enumerate(classes.tolist()):
                pred_df[f"dummy_{c}"] = pred["dummy"][:, i]
        else:
            for i, c in enumerate(classes.tolist()):
                pred_df[f"dist_{c}"] = pred["distances"][:, i]
        pred_df.to_csv(fold_dir / "preds_val.csv", index=False, encoding="utf-8")

        # --- Save ProtoNAM-compatible fold predictions for decision-threshold calibration ---
        # NOTE: ProtoNAM's calibrate_decision_thresholds.py expects p_gt_1..p_gt_{K-1}.
        proba = np.asarray(pred["proba"], dtype=np.float64)
        K = int(proba.shape[1])
        expected = np.arange(1, K + 1, dtype=int)
        if not np.array_equal(classes.astype(int), expected):
            raise ValueError(
                "Decision-threshold calibration assumes ordinal labels are consecutive integers 1..K. "
                f"Got classes={classes.tolist()} (expected {expected.tolist()})."
            )

        val_pred_df = pd.DataFrame(
            {
                "orig_index": val_idx,
                "task": tasks_va,
                "y_true": y_va,
                "y_pred": pred["y_pred"],
                "y_cont": pred["y_cont"],
            }
        )
        for i, c in enumerate(classes.tolist()):
            val_pred_df[f"p_cls_{c}"] = proba[:, i]
        for k in range(1, K):
            # p_gt_k := P(y > k) = sum_{c=k+1..K} p_cls_c
            val_pred_df[f"p_gt_{k}"] = proba[:, int(k) :].sum(axis=1)

        if meta_current is not None and id_col and id_col in meta_current.columns:
            meta_val = meta_current.iloc[val_idx].reset_index(drop=True)
            val_pred_df.insert(0, id_col, meta_val[id_col])
            if "patient_id" in meta_val.columns:
                val_pred_df.insert(1, "patient_id", meta_val["patient_id"])
            if "disc_level" in meta_val.columns:
                val_pred_df.insert(2, "disc_level", meta_val["disc_level"])

        val_pred_df.to_csv(fold_dir / "val_predictions.csv", index=False, encoding="utf-8")
        oof_pred_dfs.append(val_pred_df)

        # Confusion matrix.
        cm = confusion_matrix(y_va, pred["y_pred"], labels=classes)
        pd.DataFrame(cm, index=[f"true_{c}" for c in classes], columns=[f"pred_{c}" for c in classes]).to_csv(
            fold_dir / "confusion_val.csv", encoding="utf-8"
        )
        _save_confusion_png(cm, fold_dir / "confusion_val.png", title=f"Fold {fold} confusion (val)")

        # Per-task metrics (segment-wise) for debugging.
        by_task_rows: list[dict[str, Any]] = []
        for t in range(len(task_names)):
            mask = pred_df["task"].to_numpy().astype(int) == t
            if int(mask.sum()) == 0:
                continue
            m = compute_ordinal_metrics(
                y_true=pred_df.loc[mask, "y_true"].to_numpy(),
                y_pred=pred_df.loc[mask, "y_pred"].to_numpy(),
                y_cont=pred_df.loc[mask, "y_cont"].to_numpy(),
            )
            by_task_rows.append(
                {
                    "fold": fold,
                    "task": t,
                    "task_name": task_names[t],
                    "n": int(mask.sum()),
                    "mae": m.mae,
                    "kappa_quadratic": m.kappa_quadratic,
                    "spearman": m.spearman,
                    "acc_pm1": m.acc_pm1,
                    "acc": m.acc,
                    "bacc": m.bacc,
                    "macro_f1": m.macro_f1,
                    "weighted_f1": m.weighted_f1,
                    "ccc": m.ccc,
                }
            )
        pd.DataFrame(by_task_rows).to_csv(fold_dir / "val_metrics_by_task.csv", index=False, encoding="utf-8")

        # --- Save fold artifacts ---
        with open(fold_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump(pre, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(fold_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        (fold_dir / "feature_names.json").write_text(
            json.dumps(
                {
                    "feature_names": feature_names_current,
                    "classic_feature_names": classic_feature_names_current,
                    "task_names": task_names,
                    "id_col": id_col,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Selected features per component (by non-zero w).
        W = np.asarray(model.W_, dtype=np.float64)
        selected = (np.abs(W) > 0).any(axis=1)
        selected_names = [name for name, m in zip(feature_names_current, selected.tolist()) if m]
        selected_by_fold.append(selected.astype(int))
        selected_names_by_fold.append(set(selected_names))
        w_by_fold.append(W.copy())

        (fold_dir / "selected_features.json").write_text(
            json.dumps(
                {
                    "n_components": int(n_components_fold),
                    "keepX": keepX_fold,
                    "selected_features": selected_names,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Save core matrices for inspection.
        pd.DataFrame(model.W_, index=feature_names_current).to_csv(fold_dir / "W_weights.csv", encoding="utf-8")
        pd.DataFrame(model.P_, index=feature_names_current).to_csv(fold_dir / "P_loadings.csv", encoding="utf-8")
        pd.DataFrame(model.R_, index=feature_names_current).to_csv(fold_dir / "R_projection.csv", encoding="utf-8")
        if model.classes_ is not None:
            fold_classes = model.classes_.astype(int).tolist()
            pd.DataFrame(model.B_, index=feature_names_current, columns=[f"dummy_{c}" for c in fold_classes]).to_csv(
                fold_dir / "B_dummy_coef_fold.csv", encoding="utf-8"
            )

        # beta for the "severity axis" score s = X * beta, where beta = R a.
        a = _get_axis_a(best_post_cfg, n_components_fold)
        R_mat = np.asarray(model.R_, dtype=np.float64)
        beta = (R_mat @ a.reshape(-1, 1)).reshape(-1)
        betas_by_fold.append(beta.copy())
        beta_series_by_fold.append(pd.Series(beta, index=pd.Index(feature_names_current, dtype=str), dtype=float))
        r_by_fold.append(R_mat.copy())
        a_by_fold.append(a.copy())
        h_by_fold.append(int(n_components_fold))

        pd.DataFrame({"feature": feature_names_current, "beta": beta}).to_csv(fold_dir / "beta_axis.csv", index=False, encoding="utf-8")

        (fold_dir / "fold_info.json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "metrics": {
                        "mae": metrics.mae,
                        "kappa_quadratic": metrics.kappa_quadratic,
                        "spearman": metrics.spearman,
                        "acc_pm1": metrics.acc_pm1,
                        "acc": metrics.acc,
                        "bacc": metrics.bacc,
                        "macro_f1": metrics.macro_f1,
                        "weighted_f1": metrics.weighted_f1,
                        "ccc": metrics.ccc,
                    },
                    "splsda": {"n_components": int(n_components_fold), "keepX": keepX_fold, "eps": eps},
                    "preprocess": {"rank_gauss": use_rankgauss},
                    "postprocess": best_post_cfg,
                    "tuning": {"enabled": bool(tuning_enabled)},
                    "feature_selection": fs_summary,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        fold_results.append(
            FoldResult(
                fold=fold,
                n_train=int(len(train_idx)),
                n_val=int(len(val_idx)),
                metrics=metrics,
                checkpoint_dir=fold_dir,
            )
        )

        if metrics.kappa_quadratic > best_kappa:
            best_kappa = metrics.kappa_quadratic
            best_fold = fold

    mae = float(np.mean([fr.metrics.mae for fr in fold_results]))
    kappa = float(np.mean([fr.metrics.kappa_quadratic for fr in fold_results]))
    spearman = float(np.mean([fr.metrics.spearman for fr in fold_results]))
    acc_pm1 = float(np.mean([fr.metrics.acc_pm1 for fr in fold_results]))
    acc = float(np.mean([fr.metrics.acc for fr in fold_results]))
    bacc = float(np.mean([fr.metrics.bacc for fr in fold_results]))
    macro_f1 = float(np.mean([fr.metrics.macro_f1 for fr in fold_results]))
    weighted_f1 = float(np.mean([fr.metrics.weighted_f1 for fr in fold_results]))
    ccc = float(np.mean([fr.metrics.ccc for fr in fold_results]))
    mean_metrics = OrdinalMetrics(
        mae=mae,
        kappa_quadratic=kappa,
        spearman=spearman,
        acc_pm1=acc_pm1,
        acc=acc,
        bacc=bacc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        ccc=ccc,
    )

    # Overall metrics computed on all out-of-fold predictions (preferred).
    if y_true_all:
        y_true_cat = np.concatenate(y_true_all)
        y_pred_cat = np.concatenate(y_pred_all)
        y_cont_cat = np.concatenate(y_cont_all)
        overall_metrics = compute_ordinal_metrics(y_true=y_true_cat, y_pred=y_pred_cat, y_cont=y_cont_cat)
    else:
        overall_metrics = mean_metrics

    # Save pooled OOF predictions (ProtoNAM-compatible), sorted by orig_index.
    if oof_pred_dfs:
        oof_df = pd.concat(oof_pred_dfs, axis=0, ignore_index=True)
        if "orig_index" not in oof_df.columns:
            raise RuntimeError("val_predictions missing orig_index; cannot pool OOF predictions.")
        if int(oof_df["orig_index"].nunique()) != int(len(y)):
            raise RuntimeError(
                "Unexpected pooled OOF size: each sample must appear exactly once across folds. "
                f"n_unique_orig_index={int(oof_df['orig_index'].nunique())} vs n_samples={int(len(y))}"
            )
        oof_df = oof_df.sort_values("orig_index").reset_index(drop=True)
        oof_df.to_csv(run_dir / "oof_predictions.csv", index=False, encoding="utf-8")

    # --- Cross-fold stability summaries (for interpretation) ---
    # 1) feature selection frequency. With fold-inner feature screening the feature
    # space can differ by fold, so summarize by feature-name union instead of a fixed matrix.
    if selected_names_by_fold:
        feature_union = sorted(set().union(*selected_names_by_fold))
        n_fold_sel = max(1, len(selected_names_by_fold))
        freq_rows = [
            {"feature": feat, "selected_freq": sum(feat in s for s in selected_names_by_fold) / n_fold_sel}
            for feat in feature_union
        ]
        pd.DataFrame(freq_rows).sort_values("selected_freq", ascending=False).to_csv(
            run_dir / "feature_selection_frequency.csv", index=False, encoding="utf-8"
        )

    # 2) beta mean/std for the chosen severity axis (full sign alignment on W/R per component).
    if beta_series_by_fold:
        beta_df = pd.concat(beta_series_by_fold, axis=1).fillna(0.0)
        beta_df.columns = [f"fold_{i}" for i in range(beta_df.shape[1])]
        pd.DataFrame(
            {
                "feature": beta_df.index.astype(str),
                "beta_mean": beta_df.mean(axis=1).to_numpy(dtype=float),
                "beta_std": beta_df.std(axis=1).to_numpy(dtype=float),
            }
        ).sort_values("beta_mean", ascending=False).to_csv(
            run_dir / "beta_axis_stability.csv", index=False, encoding="utf-8"
        )

    return CVResult(
        fold_results=fold_results,
        best_fold=best_fold,
        mean_metrics=mean_metrics,
        overall_metrics=overall_metrics,
    )
