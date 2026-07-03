from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize


def make_rank_targets_np(y: np.ndarray, n_classes: int) -> np.ndarray:
    """CORAL rank targets: t_k = 1 if y > k else 0, for k = 1..K-1."""
    y = np.asarray(y, dtype=int).reshape(-1)
    ks = np.arange(1, int(n_classes), dtype=int)[None, :]
    return (y[:, None] > ks).astype(np.float64)


def _theta_from_params(params: np.ndarray, n_classes: int) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    n_bounds = int(n_classes) - 1
    if params.shape != (n_bounds,):
        raise ValueError(f"params must have shape ({n_bounds},)")
    if n_bounds == 1:
        return params.copy()
    theta = np.empty(n_bounds, dtype=np.float64)
    theta[0] = params[0]
    theta[1:] = theta[0] + np.cumsum(np.exp(np.clip(params[1:], -30.0, 30.0)))
    return theta


def _params_from_theta(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta.size <= 1:
        return theta.copy()
    diffs = np.diff(theta)
    diffs = np.maximum(diffs, 1e-3)
    return np.concatenate([[theta[0]], np.log(diffs)])

@dataclass(frozen=True)
class CoralDecoded:
    theta: np.ndarray  # (K-1,)
    p_gt: np.ndarray  # (n, K-1)  P(y > k)
    class_probs: np.ndarray  # (n, K)
    y_pred: np.ndarray  # (n,) in {1..K}
    y_cont: np.ndarray  # (n,) in (1, K)


@dataclass(frozen=True)
class DecisionThresholds:
    """
    Decision thresholds used for CORAL decode (discrete y_pred).

    - global_thr: shape (K-1,), used when thresholds are shared across tasks
    - by_task_thr: shape (n_tasks, K-1), optional per-task per-boundary thresholds
    """

    global_thr: np.ndarray
    by_task_thr: np.ndarray | None = None

    def for_samples(self, task: np.ndarray) -> np.ndarray:
        """
        Args:
            task: (n,) int array in [0..n_tasks-1]
        Returns:
            thresholds: (K-1,) or (n, K-1) float array
        """
        if self.by_task_thr is None:
            return self.global_thr
        task = np.asarray(task).astype(int).reshape(-1)
        return self.by_task_thr[task]


def _thr_vec_from_any(v: Any, *, n_bounds: int) -> list[float]:
    if isinstance(v, (int, float)):
        return [float(v)] * int(n_bounds)
    if isinstance(v, (list, tuple)):
        if len(v) < int(n_bounds):
            raise ValueError(f"decision_threshold list must have length >= {int(n_bounds)} (got {len(v)})")
        return [float(x) for x in list(v)[: int(n_bounds)]]
    raise TypeError("decision_threshold must be a number, a list/tuple, or a dict with by_task/default.")


def build_decision_thresholds(
    *,
    ord_cfg: dict[str, Any],
    task_names: list[str],
    n_classes: int,
) -> DecisionThresholds:
    """
    Build decision thresholds from config. Supported formats:
    - float: shared scalar threshold
    - list[float]: shared per-boundary thresholds (use first K-1 entries)
    - list[list[float]]: per-task per-boundary thresholds aligned with task_names
    - dict:
        {
          "default": float | list[float],
          "by_task": { "L1-L2": [...], 0: [...], ... } | [ [...], [...], ... ]
        }

    Note: CORAL has K-1 rank probabilities p_gt_1..p_gt_{K-1}. Extra entries (e.g., thr5 when K=5)
    are ignored.
    """
    K = int(n_classes)
    n_bounds = int(K - 1)
    spec = ord_cfg.get("decision_threshold", 0.5)

    default_vec: list[float]
    by_task_mat: np.ndarray | None = None

    if isinstance(spec, dict):
        default_vec = _thr_vec_from_any(spec.get("default", 0.5), n_bounds=n_bounds)
        by_task = spec.get("by_task", None)
        if by_task is None:
            by_task_mat = None
        elif isinstance(by_task, (list, tuple)) and by_task and isinstance(by_task[0], (list, tuple)):
            if len(by_task) != len(task_names):
                raise ValueError(f"ordinal.decision_threshold.by_task must have {len(task_names)} rows (got {len(by_task)})")
            by_task_mat = np.asarray(
                [_thr_vec_from_any(row, n_bounds=n_bounds) for row in list(by_task)],
                dtype=np.float32,
            )
        elif isinstance(by_task, dict):
            by_task_mat = np.tile(np.asarray(default_vec, dtype=np.float32)[None, :], (len(task_names), 1))
            name_to_idx = {name: i for i, name in enumerate(list(task_names))}
            for k, v in by_task.items():
                if isinstance(k, str) and k in name_to_idx:
                    idx = int(name_to_idx[k])
                else:
                    idx = int(k)  # may raise ValueError (int conversion) which is fine
                if idx < 0 or idx >= len(task_names):
                    raise ValueError(f"decision_threshold.by_task key out of range: {k!r} -> idx={idx}")
                by_task_mat[idx, :] = np.asarray(_thr_vec_from_any(v, n_bounds=n_bounds), dtype=np.float32)
        else:
            raise TypeError("ordinal.decision_threshold.by_task must be a dict or a list of lists.")
    elif isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], (list, tuple)):
        # Direct per-task matrix (aligned with task_names).
        if len(spec) != len(task_names):
            raise ValueError(f"ordinal.decision_threshold must have {len(task_names)} rows (got {len(spec)})")
        default_vec = _thr_vec_from_any(0.5, n_bounds=n_bounds)
        by_task_mat = np.asarray(
            [_thr_vec_from_any(row, n_bounds=n_bounds) for row in list(spec)],
            dtype=np.float32,
        )
    else:
        default_vec = _thr_vec_from_any(spec, n_bounds=n_bounds)
        by_task_mat = None

    global_thr = np.asarray(default_vec, dtype=np.float32)
    by_task_thr = None if by_task_mat is None else np.asarray(by_task_mat, dtype=np.float32)
    return DecisionThresholds(global_thr=global_thr, by_task_thr=by_task_thr)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    # clip for numerical stability
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def decode_coral(
    *,
    s: np.ndarray,
    theta: np.ndarray,
    decision_threshold: float | np.ndarray = 0.5,
) -> CoralDecoded:
    s = np.asarray(s, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    K = int(theta.shape[0] + 1)

    logits = s[:, None] - theta[None, :]
    p_gt = _sigmoid_np(logits)

    thr = decision_threshold
    if isinstance(thr, (int, float, np.floating, np.integer)):
        thr_mat = float(thr)
        y_pred = 1 + (p_gt > thr_mat).sum(axis=1).astype(int)
    else:
        thr_arr = np.asarray(thr, dtype=float)
        if thr_arr.ndim == 1:
            if thr_arr.shape != (K - 1,):
                raise ValueError(f"decision_threshold vector must have shape ({K-1},)")
            y_pred = 1 + (p_gt > thr_arr[None, :]).sum(axis=1).astype(int)
        elif thr_arr.ndim == 2:
            if thr_arr.shape != (s.shape[0], K - 1):
                raise ValueError(f"decision_threshold matrix must have shape ({s.shape[0]}, {K-1})")
            y_pred = 1 + (p_gt > thr_arr).sum(axis=1).astype(int)
        else:
            raise TypeError("decision_threshold must be a scalar, (K-1,) array, or (batch, K-1) array.")

    y_cont = 1.0 + p_gt.sum(axis=1)

    class_probs = np.zeros((s.shape[0], K), dtype=float)
    class_probs[:, 0] = 1.0 - p_gt[:, 0]
    if K > 2:
        for k in range(2, K):
            class_probs[:, k - 1] = p_gt[:, k - 2] - p_gt[:, k - 1]
    class_probs[:, K - 1] = p_gt[:, K - 2]

    return CoralDecoded(
        theta=theta,
        p_gt=p_gt,
        class_probs=class_probs,
        y_pred=y_pred,
        y_cont=y_cont,
    )


def _init_theta_from_data(s_train: np.ndarray, n_classes: int, init: str) -> np.ndarray:
    s_train = np.asarray(s_train, dtype=float).reshape(-1)
    if init == "quantile":
        qs = [k / n_classes for k in range(1, n_classes)]
        theta = np.quantile(s_train, qs).astype(np.float32)
        return theta
    if init == "linear":
        return (np.arange(1, n_classes, dtype=np.float32) + 0.5).astype(np.float32)
    raise ValueError(f"Unknown coral.fit.init: {init!r}")


def fit_coral_thresholds(
    *,
    s_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    cfg_fit: dict[str, Any] | None = None,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fit CORAL thresholds theta on a 1-D latent score s_train.

    Uses the same ordered-threshold parameterization as the previous Torch implementation:
      theta_1 is free, theta_k = theta_{k-1} + exp(delta_k).
    The objective is rank-wise binary cross entropy on P(y>k), optionally sample-weighted.
    """
    cfg_fit = cfg_fit or {}
    enabled = bool(cfg_fit.get("enabled", True))
    s_train = np.asarray(s_train, dtype=np.float64).reshape(-1)
    y_train = np.asarray(y_train, dtype=int).reshape(-1)
    if s_train.shape[0] != y_train.shape[0]:
        raise ValueError("s_train/y_train length mismatch.")

    init = str(cfg_fit.get("init", "quantile"))
    theta0 = _init_theta_from_data(s_train, int(n_classes), init).astype(np.float64)
    if not enabled:
        return theta0.astype(np.float32)

    w = None
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.shape[0] != s_train.shape[0]:
            raise ValueError("sample_weight length mismatch.")
        w_sum = float(np.sum(w)) + 1e-12
    else:
        w_sum = float(s_train.shape[0])

    n_classes = int(n_classes)
    x0 = _params_from_theta(theta0)
    weight_decay = float(cfg_fit.get("weight_decay", 0.0))
    maxiter = int(cfg_fit.get("epochs", 2000))

    def objective(params: np.ndarray) -> float:
        theta = _theta_from_params(params, n_classes)
        loss_per = _coral_loss_per_sample(s=s_train, y_true=y_train, theta=theta, n_classes=n_classes)
        if w is not None:
            loss = float(np.sum(loss_per * w) / w_sum)
        else:
            loss = float(np.mean(loss_per))
        if weight_decay > 0:
            loss += 0.5 * weight_decay * float(np.sum(np.square(params)))
        return loss

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-10, "maxls": 50},
    )
    params = res.x if np.all(np.isfinite(res.x)) else x0
    theta = _theta_from_params(params, n_classes)
    return theta.astype(np.float32)


def _coral_loss_per_sample(
    *,
    s: np.ndarray,
    y_true: np.ndarray,
    theta: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    K = int(n_classes)
    logits = s[:, None] - theta[None, :]
    tgt = make_rank_targets_np(y_true, K)
    x = logits
    loss_mat = np.maximum(x, 0.0) - x * tgt + np.log1p(np.exp(-np.abs(x)))
    return loss_mat.mean(axis=1)


def coral_loss_np(
    *,
    s: np.ndarray,
    y_true: np.ndarray,
    theta: np.ndarray,
    n_classes: int,
    sample_weight: np.ndarray | None = None,
) -> float:
    """
    Compute CORAL loss (binary cross-entropy on rank logits) in NumPy.

    This mirrors PyTorch's `binary_cross_entropy_with_logits` averaged over (K-1)
    rank targets per sample, then averaged over samples (optionally weighted).
    """
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    K = int(n_classes)
    if K < 2:
        raise ValueError("n_classes must be >= 2")
    if theta.shape[0] != K - 1:
        raise ValueError(f"theta must have shape ({K-1},), got {theta.shape}")
    if s.shape[0] != y_true.shape[0]:
        raise ValueError("s/y_true length mismatch")

    loss_per = _coral_loss_per_sample(s=s, y_true=y_true, theta=theta, n_classes=K)

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if w.shape[0] != loss_per.shape[0]:
            raise ValueError("sample_weight length mismatch")
        denom = float(np.sum(w)) + 1e-12
        return float(np.sum(loss_per * w) / denom)

    return float(np.mean(loss_per))
