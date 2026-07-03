from __future__ import annotations

import itertools
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Dataset

from lumbar_pnam.harmonization import HarmonizedPreprocessor, build_harmonization_context, make_harmonizer
from lumbar_pnam.metrics import OrdinalMetrics, compute_ordinal_metrics
from lumbar_pnam.modeling import ProtoNAMMultiTaskCoral, ProtoN2AMMultiTaskCoral
from lumbar_pnam.preprocess import RankGaussZScore, ZScore

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lumbar_pnam.data import load_model_input_csv
from lumbar_pnam.feature_selection import FeatureSelectionDataset, FoldFeatureSelector


# Keep a reference so threadpool limits remain active for the process lifetime (if enabled).
_THREADPOOL_GUARD = None


def _write_threading_info(
    *,
    run_dir: Path,
    cfg: Dict[str, Any],
    device: str,
    multicore_enabled: bool,
    cpu_threads: int,
    blas_threads: int,
) -> None:
    """Persist effective CPU threading settings for reproducibility/debugging.

    This is intentionally best-effort: failures should never break training.
    """

    try:
        import os
        import platform
        import sys

        info: dict[str, Any] = {
            "python": {
                "executable": sys.executable,
                "version": sys.version,
            },
            "platform": platform.platform(),
            "pid": int(os.getpid()),
            "device": str(device),
            "os_cpu_count": int(os.cpu_count() or 1),
            "config": {
                "training.multicore_enabled": bool(multicore_enabled),
                "training.cpu_threads": int((cfg.get("training", {}) or {}).get("cpu_threads", 0)),
                "training.blas_threads": int((cfg.get("training", {}) or {}).get("blas_threads", 0)),
            },
            "effective": {
                "cpu_threads": int(cpu_threads),
                "blas_threads": int(blas_threads),
            },
            "env": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            },
        }

        # torch thread state
        try:
            info["torch"] = {
                "version": getattr(torch, "__version__", None),
                "num_threads": int(torch.get_num_threads()),
                "num_interop_threads": int(torch.get_num_interop_threads()),
            }
        except Exception:
            info["torch"] = None

        # BLAS/OpenMP threadpool state (if detectable)
        try:
            import numpy as _np
            from threadpoolctl import threadpool_info as _threadpool_info

            # Touch BLAS once to ensure libraries are loaded before inspection.
            _ = _np.ones((2, 2), dtype=float) @ _np.ones((2, 2), dtype=float)
            info["threadpool_info"] = _threadpool_info()
        except Exception:
            info["threadpool_info"] = None

        (run_dir / "threading_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        return


@dataclass(frozen=True)
class FoldResult:
    fold: int
    n_train: int
    n_val: int
    metrics: OrdinalMetrics
    checkpoint_dir: Path


@dataclass(frozen=True)
class CVResult:
    fold_results: List[FoldResult]
    best_fold: int
    # Per 方案1.md 4.4.2, the point estimate should be computed on the pooled OOF set
    # (each disc uses its validation-fold prediction), not by averaging per-fold metrics.
    mean_metrics: OrdinalMetrics  # alias for oof_metrics
    fold_mean_metrics: OrdinalMetrics
    oof_metrics: OrdinalMetrics
    oof_bootstrap_ci95: dict[str, tuple[float, float]] | None
    oof_bootstrap_n: int


@dataclass(frozen=True)
class DecisionThresholds:
    """
    Decision thresholds used for CORAL decode (discrete y_pred):

    - global_thr: shape (K-1,), used when thresholds are shared across tasks
    - by_task_thr: shape (n_tasks, K-1), optional per-task per-boundary thresholds
    """

    global_thr: torch.Tensor
    by_task_thr: torch.Tensor | None = None

    def for_batch(self, task: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task: (batch,) int64 tensor in [0..n_tasks-1]
        Returns:
            thresholds: (K-1,) or (batch, K-1) tensor, on the same device as stored thresholds
        """
        if self.by_task_thr is None:
            return self.global_thr
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
    device: str,
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
                raise ValueError(
                    f"ordinal.decision_threshold.by_task must have {len(task_names)} rows (got {len(by_task)})"
                )
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

    dev = torch.device(str(device))
    global_thr = torch.tensor(default_vec, dtype=torch.float32, device=dev)
    by_task_thr = None if by_task_mat is None else torch.tensor(by_task_mat, dtype=torch.float32, device=dev)
    return DecisionThresholds(global_thr=global_thr, by_task_thr=by_task_thr)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sigma(step: int, total_steps: int, tau: float) -> float:
    """
    Sigma schedule (方案1(新改).md 4.3.1 / 参数设置):
        sigma(T) = (1 + exp((T - 0.5*T_max)/tau))^{-1}

    We map ProtoNAM's "temperature" to 2*sigma(T)^2 to match:
        rho ~ exp(-(x-mu)^2 / (2*sigma^2))
    """
    return float(1.0 / (1.0 + math.exp((step - 0.5 * total_steps) / tau)))


def _proto_temperature(step: int, total_steps: int, tau: float) -> float:
    s = _sigma(step, total_steps, tau)
    return float(2.0 * s * s)


def _coral_pseudo_residual(*, y: np.ndarray, p_gt_oof: np.ndarray, n_classes: int) -> np.ndarray:
    """
    CORAL pseudo residual (negative gradient w.r.t. latent score s), evaluated at OOF predictions.

        r_tilde = sum_k (t_k - p_gt_k),  where t_k = 1[y > k], k=1..K-1.
    """
    y = np.asarray(y).astype(int)
    if y.ndim != 1:
        raise ValueError("y must be 1-D.")
    K = int(n_classes)
    if p_gt_oof.shape != (y.shape[0], K - 1):
        raise ValueError(f"p_gt_oof must have shape (n, K-1) = ({y.shape[0]}, {K-1}), got {p_gt_oof.shape}")

    ks = np.arange(1, K, dtype=int)
    t = (y[:, None] > ks[None, :]).astype(np.float64)
    return (t - p_gt_oof.astype(np.float64)).sum(axis=1)


def _topk_interaction_pairs_by_spearman(
    *,
    X: np.ndarray,
    residual: np.ndarray,
    max_pairs: int,
) -> tuple[list[tuple[int, int]], list[float]]:
    """
    Rank candidate interaction pairs (a,b) by |Spearman( X[:,a]*X[:,b], residual )| and return top-k.

    Args:
        X: (n, p) standardized features.
        residual: (n,) pseudo residual vector.
        max_pairs: k
    """
    X = np.asarray(X)
    residual = np.asarray(residual).astype(np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n, p).")
    if residual.ndim != 1 or residual.shape[0] != X.shape[0]:
        raise ValueError("residual must be 1-D with the same length as X.")

    n, p = X.shape
    k = int(max_pairs)
    if k <= 0 or p < 2 or n < 3:
        return [], []

    r_rank = rankdata(residual, method="average").astype(np.float64)

    scores: list[tuple[float, int, int]] = []
    for a, b in itertools.combinations(range(p), 2):
        z = X[:, a].astype(np.float64) * X[:, b].astype(np.float64)
        if np.all(z == z[0]):
            score = 0.0
        else:
            z_rank = rankdata(z, method="average").astype(np.float64)
            corr = float(np.corrcoef(z_rank, r_rank)[0, 1])
            score = abs(corr) if np.isfinite(corr) else 0.0
        scores.append((score, a, b))

    scores.sort(key=lambda t: t[0], reverse=True)
    top = scores[: min(k, len(scores))]
    pairs = [(a, b) for (_, a, b) in top]
    vals = [float(s) for (s, _, _) in top]
    return pairs, vals


def _metrics_to_dict(m: OrdinalMetrics) -> dict[str, float]:
    return {
        "mae": float(m.mae),
        "kappa_quadratic": float(m.kappa_quadratic),
        "spearman": float(m.spearman),
        "acc_pm1": float(m.acc_pm1),
        "acc": float(m.acc),
        "bacc": float(m.bacc),
        "macro_f1": float(m.macro_f1),
        "weighted_f1": float(m.weighted_f1),
        "ccc": float(m.ccc),
    }


def _patient_cluster_bootstrap_ci95(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_cont: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    n_resamples: int,
    alpha: float,
    seed: int,
    n_jobs: int = 1,
) -> dict[str, tuple[float, float]]:
    """
    Patient-level (clustered) bootstrap CI on the pooled OOF set (方案1.md 4.4.3).

    We resample patients (clusters) with replacement and include all discs for each sampled patient.
    CI is computed by percentile method at (alpha/2, 1-alpha/2).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_cont = np.asarray(y_cont).astype(float)
    groups = np.asarray(groups)
    if y_true.ndim != 1:
        raise ValueError("y_true must be 1-D.")
    if y_pred.shape != y_true.shape or y_cont.shape != y_true.shape or groups.shape != y_true.shape:
        raise ValueError("y_pred/y_cont/groups must have the same shape as y_true.")

    B = int(n_resamples)
    if B <= 0:
        return {}

    uniq_groups, inv = np.unique(groups, return_inverse=True)
    G = int(len(uniq_groups))
    if G <= 1:
        # Bootstrap is undefined/degenerate with <=1 cluster; return empty.
        return {}

    idx_lists: list[list[int]] = [[] for _ in range(G)]
    for i, gi in enumerate(inv.tolist()):
        idx_lists[int(gi)].append(int(i))
    idx_by_group = [np.asarray(lst, dtype=int) for lst in idx_lists]

    rng = np.random.default_rng(int(seed))
    metric_keys = [
        "mae",
        "kappa_quadratic",
        "spearman",
        "acc_pm1",
        "acc",
        "bacc",
        "macro_f1",
        "weighted_f1",
        "ccc",
    ]
    sampled_g_all = rng.integers(0, G, size=(B, G))

    # Optional multi-core bootstrap: keep determinism by pre-generating sampled groups.
    # Use threads to avoid large array copies on Windows.
    nj = int(n_jobs)
    if nj == 0:
        nj = -1  # joblib convention: all cores

    if nj == 1 or B <= 1:
        vals: dict[str, np.ndarray] = {k: np.zeros(B, dtype=np.float64) for k in metric_keys}
        for b in range(B):
            sampled_g = sampled_g_all[b]
            idx = np.concatenate([idx_by_group[int(g)] for g in sampled_g], axis=0)
            m = compute_ordinal_metrics(
                y_true=y_true[idx],
                y_pred=y_pred[idx],
                y_cont=y_cont[idx],
                n_classes=int(n_classes),
            )
            md = _metrics_to_dict(m)
            for k in metric_keys:
                vals[k][b] = float(md[k])
    else:
        from joblib import Parallel, delayed

        def _one(sampled_g: np.ndarray) -> list[float]:
            idx = np.concatenate([idx_by_group[int(g)] for g in sampled_g], axis=0)
            m = compute_ordinal_metrics(
                y_true=y_true[idx],
                y_pred=y_pred[idx],
                y_cont=y_cont[idx],
                n_classes=int(n_classes),
            )
            md = _metrics_to_dict(m)
            return [float(md[k]) for k in metric_keys]

        rows = Parallel(n_jobs=nj, backend="threading", prefer="threads")(
            delayed(_one)(sampled_g_all[b]) for b in range(B)
        )
        arr = np.asarray(rows, dtype=np.float64)
        vals = {k: arr[:, j] for j, k in enumerate(metric_keys)}

    lo_q = 100.0 * (float(alpha) / 2.0)
    hi_q = 100.0 * (1.0 - float(alpha) / 2.0)
    out: dict[str, tuple[float, float]] = {}
    for k in metric_keys:
        lo = float(np.percentile(vals[k], lo_q))
        hi = float(np.percentile(vals[k], hi_q))
        out[k] = (lo, hi)
    return out


class TaskDataset(Dataset):
    def __init__(self, X: np.ndarray, task: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32, copy=False)
        self.task = task.astype(np.int64, copy=False)
        self.y = y.astype(np.int64, copy=False)

    def __getitem__(self, i: int):
        return self.X[i], self.task[i], self.y[i]

    def __len__(self) -> int:
        return int(len(self.y))


@torch.no_grad()
def _eval_on_loader(
    model: Any,
    loader: DataLoader,
    device: str,
    *,
    n_classes: int,
    decision_thresholds: DecisionThresholds,
    proto_T: float,
) -> Tuple[float, OrdinalMetrics, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_loss_data = 0.0
    total_loss_out = 0.0
    total_r_out = 0.0
    total_loss_seg = 0.0
    total_r_seg = 0.0
    total_loss_task_smooth = 0.0
    total_r_task_smooth = 0.0
    total_loss_qwk = 0.0
    total_loss_acc = 0.0
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    y_cont_list: list[np.ndarray] = []

    for x, task, y in loader:
        x = x.float().to(device)
        task = task.to(device).to(torch.int64)
        y = y.to(device).to(torch.int64)

        _, stats = model(x, task, y, T=proto_T)
        bs = int(len(x))
        total_loss += float(stats["loss"].item()) * bs
        total_loss_data += float(stats["loss_data"].item()) * bs
        total_loss_out += float(stats["loss_out"].item()) * bs
        total_r_out += float(stats["r_out"].item()) * bs
        total_loss_seg += float(stats["loss_seg"].item()) * bs
        total_r_seg += float(stats["r_seg"].item()) * bs
        total_loss_task_smooth += float(stats["loss_task_smooth"].item()) * bs
        total_r_task_smooth += float(stats["r_task_smooth"].item()) * bs
        total_loss_qwk += float(stats["loss_qwk"].item()) * bs
        total_loss_acc += float(stats["loss_acc"].item()) * bs

        thr = decision_thresholds.for_batch(task)
        _, out, _ = model.predict_ordinal(x, task, T=proto_T, decision_threshold=thr)
        y_true_list.append(y.detach().cpu().numpy())
        y_pred_list.append(out.y_pred.detach().cpu().numpy())
        y_cont_list.append(out.y_cont.detach().cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_cont = np.concatenate(y_cont_list)

    metrics = compute_ordinal_metrics(y_true=y_true, y_pred=y_pred, y_cont=y_cont, n_classes=n_classes)
    n = float(len(y_true))
    aux = {
        "loss_total": float(total_loss / n),
        "loss_data": float(total_loss_data / n),
        "loss_out": float(total_loss_out / n),
        "r_out": float(total_r_out / n),
        "loss_seg": float(total_loss_seg / n),
        "r_seg": float(total_r_seg / n),
        "loss_task_smooth": float(total_loss_task_smooth / n),
        "r_task_smooth": float(total_r_task_smooth / n),
        "loss_qwk": float(total_loss_qwk / n),
        "loss_acc": float(total_loss_acc / n),
    }
    # Early stopping monitors L_data^{val} per the updated plan.
    return aux["loss_data"], metrics, aux


@torch.no_grad()
def _predict_on_loader(
    model: Any,
    loader: DataLoader,
    device: str,
    *,
    n_classes: int,
    decision_thresholds: DecisionThresholds,
    proto_T: float,
) -> dict[str, np.ndarray]:
    model.eval()
    y_true_list: list[np.ndarray] = []
    y_pred_list: list[np.ndarray] = []
    y_cont_list: list[np.ndarray] = []
    s_list: list[np.ndarray] = []
    s_layers_list: list[np.ndarray] = []
    task_list: list[np.ndarray] = []
    p_gt_list: list[np.ndarray] = []
    prob_list: list[np.ndarray] = []

    for x, task, y in loader:
        x = x.float().to(device)
        task = task.to(device).to(torch.int64)
        y = y.to(device).to(torch.int64)

        thr = decision_thresholds.for_batch(task)
        s, out, s_layers = model.predict_ordinal(x, task, T=proto_T, decision_threshold=thr)

        y_true_list.append(y.detach().cpu().numpy())
        y_pred_list.append(out.y_pred.detach().cpu().numpy())
        y_cont_list.append(out.y_cont.detach().cpu().numpy())
        s_list.append(s.detach().cpu().numpy())
        s_layers_list.append(s_layers.detach().cpu().numpy())
        task_list.append(task.detach().cpu().numpy())
        p_gt_list.append(out.p_gt.detach().cpu().numpy())
        prob_list.append(out.class_probs.detach().cpu().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    y_cont = np.concatenate(y_cont_list)
    s = np.concatenate(s_list)
    s_layers = np.concatenate(s_layers_list)
    task = np.concatenate(task_list)
    p_gt = np.concatenate(p_gt_list)
    class_probs = np.concatenate(prob_list)

    out: dict[str, np.ndarray] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_cont": y_cont,
        "s": s,
        "task": task,
    }
    for m in range(s_layers.shape[1]):
        out[f"s_layer_{m+1}"] = s_layers[:, m]
    for k in range(n_classes - 1):
        out[f"p_gt_{k+1}"] = p_gt[:, k]
    for k in range(n_classes):
        out[f"p_cls_{k+1}"] = class_probs[:, k]
    return out


def _fit_baseline_for_oof(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_val: np.ndarray,
    n_tasks: int,
    n_classes: int,
    decision_thresholds: DecisionThresholds,
    beta: float,
    qwk_loss_weight: float,
    acc_loss_weight: float,
    lambda_out: float,
    lambda_seg: float,
    lambda_task_smooth: float,
    device: str,
    seed: int,
    lr_max: float,
    lr_min: float,
    weight_decay: float,
    batch_size: int,
    max_epoch: int,
    patience: int,
    min_delta: float,
    stop_monitor: str,
    select_monitor: str,
    eval_T: float,
    p: int,
    h_dim: int,
    n_proto: int,
    n_layers: int,
    n_layers_pred: int,
    batch_norm: bool,
    dropout: float,
    dropout_output: float,
    tau: float,
    share_task_weights_across_layers: bool,
    class_weight_mode: str,
    num_workers: int = 0,
) -> ProtoNAMMultiTaskCoral:
    """
    Train a baseline (no-interaction) model for inner-fold OOF prediction.

    This is intentionally separate from the outer-fold training code path to avoid changing
    the default behavior when interaction is disabled.
    """
    _set_seed(int(seed))

    class_weight: list[float] | None = None
    if class_weight_mode == "balanced":
        counts = np.bincount((y_train - 1).astype(int), minlength=int(n_classes)).astype(np.float32)
        counts[counts == 0] = 1.0
        w = float(counts.sum()) / (float(n_classes) * counts)
        w = w / float(np.mean(w))
        class_weight = w.tolist()

    train_set = TaskDataset(X_train, task_train, y_train)
    val_set = TaskDataset(X_val, task_val, y_val)
    train_loader = DataLoader(train_set, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers))
    val_loader = DataLoader(val_set, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))

    model = ProtoNAMMultiTaskCoral(
        n_feat=int(X_train.shape[1]),
        n_classes=int(n_classes),
        n_tasks=int(n_tasks),
        p=int(p),
        h_dim=int(h_dim),
        n_proto=int(n_proto),
        n_layers=int(n_layers),
        n_layers_pred=int(n_layers_pred),
        batch_norm=bool(batch_norm),
        dropout=float(dropout),
        dropout_output=float(dropout_output),
        beta=float(beta),
        lambda_out=float(lambda_out),
        lambda_seg=float(lambda_seg),
        lambda_task_smooth=float(lambda_task_smooth),
        tau=float(tau),
        class_weight=class_weight,
        qwk_loss_weight=float(qwk_loss_weight),
        acc_loss_weight=float(acc_loss_weight),
        share_task_weights_across_layers=bool(share_task_weights_across_layers),
    ).to(str(device))

    model.initialize_prototypes(X_train)

    optim = torch.optim.AdamW(model.parameters(), lr=float(lr_max), weight_decay=float(weight_decay))
    total_steps = max(1, int(max_epoch) * max(1, len(train_loader)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps, eta_min=float(lr_min))

    best_stop_val = float("inf") if stop_monitor == "val_loss_data" else -1e9
    best_select_val = float("inf") if select_monitor == "val_loss_data" else -1e9
    best_state: dict[str, Any] | None = None
    best_epoch = -1
    no_improve = 0
    global_step = 0

    for epoch in range(int(max_epoch)):
        model.train()
        for xb, tb, yb in train_loader:
            xb = xb.float().to(device)
            tb = tb.to(device).to(torch.int64)
            yb = yb.to(device).to(torch.int64)

            global_step += 1
            T = max(_proto_temperature(global_step, total_steps, float(tau)), float(eval_T))

            optim.zero_grad()
            loss, _ = model(xb, tb, yb, T=T)
            loss.backward()
            optim.step()
            scheduler.step()

        # Validation for early stopping / best checkpoint selection.
        val_loss_data, val_metrics, _ = _eval_on_loader(
            model,
            val_loader,
            str(device),
            n_classes=int(n_classes),
            decision_thresholds=decision_thresholds,
            proto_T=float(eval_T),
        )
        cur_loss = float(val_loss_data)
        cur_kappa = float(val_metrics.kappa_quadratic)
        cur_acc = float(val_metrics.acc)

        def _mon_value(name: str) -> float:
            if name == "val_loss_data":
                return cur_loss
            if name == "val_kappa":
                return cur_kappa
            if name == "val_acc":
                return cur_acc
            raise ValueError(f"Unknown monitor: {name!r}")

        cur_stop_val = _mon_value(stop_monitor)
        cur_select_val = _mon_value(select_monitor)

        improved_stop = (
            (cur_stop_val < best_stop_val - float(min_delta))
            if stop_monitor == "val_loss_data"
            else (cur_stop_val > best_stop_val + float(min_delta))
        )
        improved_select = (
            (cur_select_val < best_select_val - float(min_delta))
            if select_monitor == "val_loss_data"
            else (cur_select_val > best_select_val + float(min_delta))
        )

        if improved_select:
            best_select_val = cur_select_val
            best_stop_val = cur_stop_val if improved_stop else best_stop_val
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if improved_stop:
            best_stop_val = cur_stop_val
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _oof_p_gt_baseline(
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    task: np.ndarray,
    groups: np.ndarray,
    harm_ctx: Any,
    n_tasks: int,
    n_classes: int,
    decision_thresholds: DecisionThresholds,
    beta: float,
    qwk_loss_weight: float,
    acc_loss_weight: float,
    lambda_out: float,
    lambda_seg: float,
    lambda_task_smooth: float,
    device: str,
    seed: int,
    lr_max: float,
    lr_min: float,
    weight_decay: float,
    batch_size: int,
    max_epoch: int,
    patience: int,
    min_delta: float,
    stop_monitor: str,
    select_monitor: str,
    eval_T: float,
    p: int,
    h_dim: int,
    n_proto: int,
    n_layers: int,
    n_layers_pred: int,
    batch_norm: bool,
    dropout: float,
    dropout_output: float,
    tau: float,
    share_task_weights_across_layers: bool,
    class_weight_mode: str,
    inner_splits: int,
    shuffle: bool,
    random_state: int,
    pre_method: str,
    pre_n_jobs: int = 1,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cross-fitted OOF (within the outer-fold training set) for interaction selection:
    - For each inner split, fit the preprocessor on inner-train ONLY (strictly no leakage).
    - Train a baseline (no-interaction) model on the transformed inner-train set.
    - Predict p_gt on transformed inner-val set and assemble OOF predictions.

    Returns:
        p_gt_oof: (n, K-1) OOF P(y>k) for k=1..K-1
        X_oof_t: (n, p) OOF-transformed features (each row transformed by its inner-train preprocessor)
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("_oof_p_gt_baseline expects X to be a pandas DataFrame.")
    y = np.asarray(y).astype(int)
    task = np.asarray(task).astype(int)
    groups = np.asarray(groups)

    splitter = StratifiedGroupKFold(
        n_splits=int(inner_splits),
        shuffle=bool(shuffle),
        random_state=(int(random_state) if shuffle else None),
    )

    n = int(len(y))
    p_dim = int(X.shape[1])
    oof = np.zeros((n, int(n_classes) - 1), dtype=np.float32)
    X_oof_t = np.zeros((n, p_dim), dtype=np.float32)
    seen = np.zeros(len(y), dtype=bool)

    pre_method = str(pre_method).strip().lower()
    if pre_method not in {"rankgauss_zscore", "zscore"}:
        raise ValueError(f"preprocess.method must be one of: rankgauss_zscore, zscore (got {pre_method!r})")
    pre_n_jobs = int(pre_n_jobs)

    for j, (tr_idx, va_idx) in enumerate(splitter.split(X, y, groups)):
        X_tr_raw = X.iloc[tr_idx]
        X_va_raw = X.iloc[va_idx]

        harm = make_harmonizer(harm_ctx)
        harm.fit(X_tr_raw)
        X_tr_h = harm.transform(X_tr_raw)
        X_va_h = harm.transform(X_va_raw)

        pre = (ZScore() if pre_method == "zscore" else RankGaussZScore(n_jobs=pre_n_jobs)).fit(X_tr_h)
        X_tr_t = pre.transform(X_tr_h)
        X_va_t = pre.transform(X_va_h)

        model = _fit_baseline_for_oof(
            X_train=X_tr_t,
            y_train=y[tr_idx],
            task_train=task[tr_idx],
            X_val=X_va_t,
            y_val=y[va_idx],
            task_val=task[va_idx],
            n_tasks=int(n_tasks),
            n_classes=int(n_classes),
            decision_thresholds=decision_thresholds,
            beta=float(beta),
            qwk_loss_weight=float(qwk_loss_weight),
            acc_loss_weight=float(acc_loss_weight),
            lambda_out=float(lambda_out),
            lambda_seg=float(lambda_seg),
            lambda_task_smooth=float(lambda_task_smooth),
            device=str(device),
            seed=int(seed) + 1000 + int(j),
            lr_max=float(lr_max),
            lr_min=float(lr_min),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
            max_epoch=int(max_epoch),
            patience=int(patience),
            min_delta=float(min_delta),
            stop_monitor=str(stop_monitor),
            select_monitor=str(select_monitor),
            eval_T=float(eval_T),
            p=int(p),
            h_dim=int(h_dim),
            n_proto=int(n_proto),
            n_layers=int(n_layers),
            n_layers_pred=int(n_layers_pred),
            batch_norm=bool(batch_norm),
            dropout=float(dropout),
            dropout_output=float(dropout_output),
            tau=float(tau),
            share_task_weights_across_layers=bool(share_task_weights_across_layers),
            class_weight_mode=str(class_weight_mode),
            num_workers=int(num_workers),
        )

        val_set = TaskDataset(X_va_t, task[va_idx], y[va_idx])
        val_loader = DataLoader(val_set, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))
        pred = _predict_on_loader(
            model,
            val_loader,
            str(device),
            n_classes=int(n_classes),
            decision_thresholds=decision_thresholds,
            proto_T=float(eval_T),
        )
        p_gt = np.stack([pred[f"p_gt_{k}"] for k in range(1, int(n_classes))], axis=1).astype(np.float32)
        oof[va_idx] = p_gt
        X_oof_t[va_idx] = X_va_t
        seen[va_idx] = True

    if not bool(seen.all()):
        raise RuntimeError("OOF prediction incomplete: some samples were never assigned an inner validation fold.")
    return oof, X_oof_t


def _safe_qwk_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import cohen_kappa_score

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        return 0.0


def _balanced_accuracy_np(y_true: np.ndarray, y_pred: np.ndarray, *, n_classes: int) -> float:
    labels = list(range(1, int(n_classes) + 1))
    cm = confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int), labels=labels)
    support = cm.sum(axis=1).astype(float)
    diag = np.diag(cm).astype(float)
    rec = np.divide(diag, support, out=np.zeros_like(support, dtype=float), where=support > 0)
    return float(rec.mean()) if rec.size else 0.0


def _threshold_metric_score(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    objective: str,
    tie_breakers: list[str] | tuple[str, ...],
) -> tuple[float, ...]:
    """Build a lexicographic maximization score for threshold calibration."""
    from sklearn.metrics import f1_score, mean_absolute_error

    y_true_i = np.asarray(y_true).astype(int)
    y_pred_i = np.asarray(y_pred).astype(int)
    labels = list(range(1, int(n_classes) + 1))
    metrics = {
        "acc": float(np.mean(y_true_i == y_pred_i)) if y_true_i.size else 0.0,
        "bacc": _balanced_accuracy_np(y_true_i, y_pred_i, n_classes=int(n_classes)),
        "qwk": _safe_qwk_np(y_true_i, y_pred_i),
        "mae": float(mean_absolute_error(y_true_i, y_pred_i)) if y_true_i.size else 0.0,
        "macro_f1": float(f1_score(y_true_i, y_pred_i, labels=labels, average="macro", zero_division=0))
        if y_true_i.size
        else 0.0,
    }

    score_parts: list[float] = []
    for raw_name in [str(objective), *[str(x) for x in tie_breakers]]:
        name = raw_name.strip().lower()
        if not name:
            continue
        sign = -1.0 if name.startswith("-") else 1.0
        metric_name = name[1:] if name.startswith("-") else name
        if metric_name not in metrics:
            allowed = ", ".join(sorted(metrics))
            raise ValueError(f"Unknown threshold calibration metric: {raw_name!r}; allowed: {allowed}")
        score_parts.append(sign * float(metrics[metric_name]))
    return tuple(score_parts)


def _decode_p_gt(p_gt: np.ndarray, thr_vec: np.ndarray) -> np.ndarray:
    """CORAL decode: y_pred = 1 + sum_k I(p_gt_k > thr_k), k=1..K-1."""
    p_gt = np.asarray(p_gt, dtype=float)
    thr_vec = np.asarray(thr_vec, dtype=float)
    return 1 + (p_gt > thr_vec[None, :]).sum(axis=1).astype(int)


def _search_thresholds_qwk_mae_f1(
    *,
    p_gt: np.ndarray,
    y_true: np.ndarray,
    n_classes: int,
    grid: np.ndarray,
    n_iter: int,
    objective: str = "qwk",
    tie_breakers: list[str] | tuple[str, ...] = ("-mae", "macro_f1"),
) -> np.ndarray:
    """
    Coordinate-descent search of CORAL decision thresholds on a calibration set.

    Objective is a lexicographic maximization tuple, e.g.:
    - qwk:  (QWK, -MAE, Macro-F1)
    - acc:  (Acc, BAcc, QWK, -MAE, Macro-F1)
    - bacc: (BAcc, Acc, QWK, -MAE, Macro-F1)
    """
    y_true_i = np.asarray(y_true).astype(int)
    n_bounds = int(n_classes) - 1
    thr = np.full((n_bounds,), 0.5, dtype=np.float64)
    if y_true_i.size == 0:
        return thr
    for _ in range(int(n_iter)):
        for k in range(n_bounds):
            best_score: tuple[float, ...] | None = None
            best_thr = float(thr[k])
            for cand in grid.tolist():
                thr_c = thr.copy()
                thr_c[k] = float(cand)
                yp = _decode_p_gt(p_gt, thr_c).astype(int)
                score = _threshold_metric_score(
                    y_true=y_true_i,
                    y_pred=yp,
                    n_classes=int(n_classes),
                    objective=str(objective),
                    tie_breakers=tie_breakers,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_thr = float(cand)
            thr[k] = best_thr
    return thr


def _nested_threshold_calibration(
    *,
    X_train_raw: pd.DataFrame,
    y_train: np.ndarray,
    task_train: np.ndarray,
    groups_train: np.ndarray,
    harm_ctx: Any,
    n_tasks: int,
    n_classes: int,
    decision_thresholds: DecisionThresholds,
    pre_method: str,
    pre_n_jobs: int,
    calib_splits: int,
    shuffle: bool,
    random_state: int,
    grid: np.ndarray,
    n_iter: int,
    objective: str,
    tie_breakers: list[str],
    beta: float,
    qwk_loss_weight: float,
    acc_loss_weight: float,
    lambda_out: float,
    lambda_seg: float,
    lambda_task_smooth: float,
    device: str,
    seed: int,
    lr_max: float,
    lr_min: float,
    weight_decay: float,
    batch_size: int,
    max_epoch: int,
    patience: int,
    min_delta: float,
    stop_monitor: str,
    select_monitor: str,
    eval_T: float,
    p: int,
    h_dim: int,
    n_proto: int,
    n_layers: int,
    n_layers_pred: int,
    batch_norm: bool,
    dropout: float,
    dropout_output: float,
    tau: float,
    share_task_weights_across_layers: bool,
    class_weight_mode: str,
    num_workers: int = 0,
) -> np.ndarray:
    """
    Learn per-task CORAL decision thresholds strictly inside the outer training fold (方案 0.2).

    A group-wise calibration split is carved out of the outer-train fold. A calibration model is
    trained on the train-subset ONLY, then used to predict P(y>k) on the held-out calibration-subset,
    where per-task thresholds are searched with the configured objective. The outer validation
    fold is never used for threshold learning, so the pooled calibrated OOF metric stays unbiased.

    Returns:
        thr_by_task: (n_tasks, K-1) thresholds; tasks without calibration samples fall back to 0.5.
    """
    y_train = np.asarray(y_train).astype(int)
    task_train = np.asarray(task_train).astype(int)
    groups_train = np.asarray(groups_train)

    thr_by_task = np.full((int(n_tasks), int(n_classes) - 1), 0.5, dtype=np.float64)

    # Feasibility guard: StratifiedGroupKFold needs enough groups and per-class samples.
    uniq_groups = np.unique(groups_train)
    class_counts = np.bincount((y_train - 1).astype(int), minlength=int(n_classes))
    nonzero = class_counts[class_counts > 0]
    if int(len(uniq_groups)) < int(calib_splits) or (nonzero.size and int(nonzero.min()) < int(calib_splits)):
        print(
            f"[ThresholdCalibration] skip: not enough groups/samples for a {int(calib_splits)}-way "
            f"group-wise calibration split (groups={int(len(uniq_groups))}); using default thresholds (0.5)."
        )
        return thr_by_task

    splitter = StratifiedGroupKFold(
        n_splits=int(calib_splits),
        shuffle=bool(shuffle),
        random_state=(int(random_state) if shuffle else None),
    )
    inner_tr_idx, calib_idx = next(iter(splitter.split(X_train_raw, y_train, groups_train)))

    X_inner_raw = X_train_raw.iloc[inner_tr_idx]
    X_calib_raw = X_train_raw.iloc[calib_idx]

    harm = make_harmonizer(harm_ctx)
    harm.fit(X_inner_raw)
    X_inner_h = harm.transform(X_inner_raw)
    X_calib_h = harm.transform(X_calib_raw)

    pre = (ZScore() if pre_method == "zscore" else RankGaussZScore(n_jobs=int(pre_n_jobs))).fit(X_inner_h)
    X_inner_t = pre.transform(X_inner_h)
    X_calib_t = pre.transform(X_calib_h)

    model = _fit_baseline_for_oof(
        X_train=X_inner_t,
        y_train=y_train[inner_tr_idx],
        task_train=task_train[inner_tr_idx],
        X_val=X_calib_t,
        y_val=y_train[calib_idx],
        task_val=task_train[calib_idx],
        n_tasks=int(n_tasks),
        n_classes=int(n_classes),
        decision_thresholds=decision_thresholds,
        beta=float(beta),
        qwk_loss_weight=float(qwk_loss_weight),
        acc_loss_weight=float(acc_loss_weight),
        lambda_out=float(lambda_out),
        lambda_seg=float(lambda_seg),
        lambda_task_smooth=float(lambda_task_smooth),
        device=str(device),
        seed=int(seed),
        lr_max=float(lr_max),
        lr_min=float(lr_min),
        weight_decay=float(weight_decay),
        batch_size=int(batch_size),
        max_epoch=int(max_epoch),
        patience=int(patience),
        min_delta=float(min_delta),
        stop_monitor=str(stop_monitor),
        select_monitor=str(select_monitor),
        eval_T=float(eval_T),
        p=int(p),
        h_dim=int(h_dim),
        n_proto=int(n_proto),
        n_layers=int(n_layers),
        n_layers_pred=int(n_layers_pred),
        batch_norm=bool(batch_norm),
        dropout=float(dropout),
        dropout_output=float(dropout_output),
        tau=float(tau),
        share_task_weights_across_layers=bool(share_task_weights_across_layers),
        class_weight_mode=str(class_weight_mode),
        num_workers=int(num_workers),
    )

    calib_set = TaskDataset(X_calib_t, task_train[calib_idx], y_train[calib_idx])
    calib_loader = DataLoader(calib_set, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers))
    pred = _predict_on_loader(
        model,
        calib_loader,
        str(device),
        n_classes=int(n_classes),
        decision_thresholds=decision_thresholds,
        proto_T=float(eval_T),
    )
    p_gt = np.stack([pred[f"p_gt_{k}"] for k in range(1, int(n_classes))], axis=1).astype(np.float64)
    task_calib = pred["task"].astype(int)
    y_calib = pred["y_true"].astype(int)

    for t in range(int(n_tasks)):
        m = task_calib == int(t)
        if int(m.sum()) == 0:
            continue
        thr_by_task[int(t), :] = _search_thresholds_qwk_mae_f1(
            p_gt=p_gt[m],
            y_true=y_calib[m],
            n_classes=int(n_classes),
            grid=grid,
            n_iter=int(n_iter),
            objective=str(objective),
            tie_breakers=list(tie_breakers),
        )
    return thr_by_task


def _save_confusion_matrix(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    out_path: Path,
    normalize: bool,
    title: str,
) -> None:
    labels = np.arange(1, n_classes + 1, dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm_plot = cm.astype(float)
        row_sum = cm_plot.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm_plot = cm_plot / row_sum
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels([str(i) for i in labels])
    ax.set_yticklabels([str(i) for i in labels])

    thresh = cm_plot.max() / 2.0 if cm_plot.size else 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            val = cm_plot[i, j]
            txt = format(val, fmt)
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontsize=8,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _save_ycont_scatter(*, y_true: np.ndarray, y_cont: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_cont, s=18, alpha=0.7)
    ax.set_xlabel("True grade")
    ax.set_ylabel("Continuous output (model)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def train_group_kfold(
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    tasks: np.ndarray,
    task_names: list[str],
    groups: np.ndarray,
    classic_feature_names: list[str],
    run_dir: Path,
    cfg: Dict[str, Any],
    meta: pd.DataFrame | None = None,
    id_col: str | None = None,
    feature_selection_dataset: FeatureSelectionDataset | None = None,
) -> CVResult:
    cv_cfg = cfg.get("cv", {}) or {}
    n_splits = int(cv_cfg["n_splits"])
    splitter = str(cv_cfg.get("splitter", "groupkfold"))
    splitter = splitter.strip().lower().replace("-", "_")
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = int(cv_cfg.get("random_state", cfg.get("training", {}).get("seed", 0)))

    ord_cfg = cfg.get("ordinal", {}) or {}
    n_classes = int(ord_cfg["n_classes"])
    beta = float(ord_cfg.get("beta", 0.5))
    qwk_loss_weight = float(ord_cfg.get("qwk_loss_weight", 0.0))
    acc_loss_weight = float(ord_cfg.get("acc_loss_weight", 0.1))
    reg_cfg = cfg.get("regularization", {}) or {}
    lambda_out = float(reg_cfg.get("lambda_out", 1e-3))
    lambda_seg = float(reg_cfg.get("lambda_seg", 0.0))
    lambda_task_smooth = float(reg_cfg.get("lambda_task_smooth", 0.0))

    # Soft monotonicity constraints (单调性约束方案). Disabled unless configured.
    mono_cfg = cfg.get("monotonicity", {}) or {}
    mono_enabled = bool(mono_cfg.get("enabled", False))
    mono_mode = str(mono_cfg.get("mode", "soft"))
    mono_lambda = float(mono_cfg.get("lambda_mono", 0.0))
    mono_alpha_w = float(mono_cfg.get("alpha_w", 1.0))
    mono_apply_to = str(mono_cfg.get("apply_to", "task_weighted"))
    mono_grid_size = int(mono_cfg.get("grid_size", 64))
    mono_grid_method = str(mono_cfg.get("grid_method", "quantile"))
    mono_eps_grid = float(mono_cfg.get("eps_grid", 1e-6))
    mono_w_pos_floor = float(mono_cfg.get("w_positive_floor", 0.0))
    mono_dir_by_feature: dict[str, int] = dict(mono_cfg.get("direction_by_feature", {}) or {})

    device = str(cfg["training"]["device"])
    seed = int(cfg["training"]["seed"])
    cpu_threads = int(cfg.get("training", {}).get("cpu_threads", 0))
    blas_threads = int(cfg.get("training", {}).get("blas_threads", 0))
    dataloader_num_workers = int(cfg.get("training", {}).get("dataloader_num_workers", 0))
    lr_max = float(cfg["training"]["lr_max"])
    lr_min = float(cfg["training"].get("lr_min", 0.0))
    weight_decay = float(reg_cfg.get("weight_decay", cfg["training"].get("weight_decay", 0.0)))
    batch_size = int(cfg["training"]["batch_size"])
    max_epoch = int(cfg["training"]["max_epoch"])
    patience = int(cfg["training"]["patience"])
    min_delta = float(cfg["training"]["min_delta"])
    stop_monitor = str(cfg.get("training", {}).get("stop_monitor", "val_loss_data"))
    select_monitor = str(cfg.get("training", {}).get("select_monitor", stop_monitor))
    for name, val in {"stop_monitor": stop_monitor, "select_monitor": select_monitor}.items():
        if val not in {"val_loss_data", "val_kappa", "val_acc"}:
            raise ValueError(f"training.{name} must be one of: val_loss_data, val_kappa, val_acc (got {val!r})")
    class_weight_mode = str(cfg.get("training", {}).get("class_weight_mode", "none"))
    if class_weight_mode not in {"none", "balanced"}:
        raise ValueError(f"training.class_weight_mode must be one of: none, balanced (got {class_weight_mode!r})")

    # Nested within-train threshold calibration (方案 0.2). When enabled, per-fold thresholds are
    # learned strictly inside the outer-train fold and applied to the outer val fold, then pooled
    # into an unbiased calibrated OOF estimate.
    tcal_cfg = cfg.get("threshold_calibration", {}) or {}
    tcal_enabled = bool(tcal_cfg.get("enabled", False))
    tcal_splits = int(tcal_cfg.get("calib_splits", 5))
    tcal_grid_min = float(tcal_cfg.get("grid_min", 0.25))
    tcal_grid_max = float(tcal_cfg.get("grid_max", 0.75))
    tcal_grid_step = float(tcal_cfg.get("grid_step", 0.01))
    tcal_n_iter = int(tcal_cfg.get("n_iter", 3))
    tcal_objective = str(tcal_cfg.get("objective", "acc")).strip().lower()
    default_ties = {
        "qwk": ["-mae", "macro_f1"],
        "acc": ["bacc", "qwk", "-mae", "macro_f1"],
        "bacc": ["acc", "qwk", "-mae", "macro_f1"],
    }
    tcal_tie_breakers = list(tcal_cfg.get("tie_breakers", default_ties.get(tcal_objective, ["qwk", "-mae"])))
    _threshold_metric_score(
        y_true=np.array([1, 2], dtype=int),
        y_pred=np.array([1, 2], dtype=int),
        n_classes=2,
        objective=tcal_objective,
        tie_breakers=tcal_tie_breakers,
    )
    tcal_grid = np.arange(tcal_grid_min, tcal_grid_max + 1e-12, tcal_grid_step, dtype=np.float64)

    prep_cfg = cfg.get("preprocess", {}) or {}
    pre_method = str(prep_cfg.get("method", "rankgauss_zscore")).strip().lower()
    if pre_method not in {"rankgauss_zscore", "zscore"}:
        raise ValueError(f"preprocess.method must be one of: rankgauss_zscore, zscore (got {pre_method!r})")
    pre_n_jobs = int(prep_cfg.get("n_jobs", 1))

    # Harmonization context (built once; fold-inner fit/transform is handled per fold).
    harm_ctx = build_harmonization_context(cfg=cfg, meta=meta)

    model_cfg = cfg["model"]
    p = int(model_cfg["p"])
    h_dim = int(model_cfg["h_dim"])
    n_proto = int(model_cfg["n_proto"])
    n_layers = int(model_cfg["n_layers"])
    n_layers_pred = int(model_cfg["n_layers_pred"])
    batch_norm = bool(model_cfg["batch_norm"])
    dropout = float(model_cfg["dropout"])
    dropout_output = float(model_cfg["dropout_output"])
    tau = float(model_cfg["tau"])
    share_task_weights_across_layers = bool(model_cfg.get("share_task_weights_across_layers", False))

    int_cfg = cfg.get("interaction", {}) or {}
    use_interaction = bool(int_cfg.get("enabled", False))
    int_max_pairs = int(int_cfg.get("max_pairs", 20))
    int_inner_splits = int(int_cfg.get("inner_splits", 5))
    int_sel_max_epoch = int(int_cfg.get("selection_max_epoch", 80))
    int_sel_patience = int(int_cfg.get("selection_patience", 20))
    int_mlp_hidden_dim_raw = int_cfg.get("mlp_hidden_dim")
    int_mlp_hidden_dim = None if int_mlp_hidden_dim_raw is None else int(int_mlp_hidden_dim_raw)
    int_mlp_dropout = float(int_cfg.get("mlp_dropout", 0.0))

    eval_cfg = cfg.get("evaluation", {}) or {}
    boot_cfg = eval_cfg.get("bootstrap", {}) or {}
    boot_enabled = bool(boot_cfg.get("enabled", True))
    boot_n = int(boot_cfg.get("n_resamples", 1000))
    boot_alpha = float(boot_cfg.get("alpha", 0.05))
    boot_seed = int(boot_cfg.get("seed", seed))
    boot_n_jobs = int(boot_cfg.get("n_jobs", 1))

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

    # Optional: force multi-core CPU execution for outer CV training (folds are still trained serially).
    multicore_enabled = bool((cfg.get("training", {}) or {}).get("multicore_enabled", False))
    if multicore_enabled and str(device).strip().lower() == "cpu":
        import os

        n_cores = int(os.cpu_count() or 1)
        if cpu_threads <= 0:
            cpu_threads = n_cores
        if blas_threads <= 0:
            blas_threads = n_cores

    if str(device).strip().lower() == "cpu" and blas_threads and int(blas_threads) > 0:
        # Control BLAS/OpenMP parallelism used by NumPy/SciPy/sklearn/combatlearn, etc.
        # Keep a global guard so limits stay active for the process lifetime.
        global _THREADPOOL_GUARD
        if _THREADPOOL_GUARD is None:
            try:
                from threadpoolctl import threadpool_limits

                _THREADPOOL_GUARD = threadpool_limits(limits=int(blas_threads))
                _THREADPOOL_GUARD.__enter__()
            except Exception:
                _THREADPOOL_GUARD = None

    if str(device).strip().lower() == "cpu" and cpu_threads and int(cpu_threads) > 0:
        # Control CPU parallelism (useful for reproducibility and for speeding up CPU training).
        torch.set_num_threads(int(cpu_threads))
        try:
            torch.set_num_interop_threads(max(1, min(4, int(cpu_threads))))
        except Exception:
            # set_num_interop_threads can only be called once and must be called early.
            pass

    _write_threading_info(
        run_dir=run_dir,
        cfg=cfg,
        device=device,
        multicore_enabled=multicore_enabled,
        cpu_threads=cpu_threads,
        blas_threads=blas_threads,
    )

    _set_seed(seed)

    # Decision thresholds for CORAL decode (supports per-task per-boundary).
    decision_thresholds = build_decision_thresholds(
        ord_cfg=ord_cfg,
        task_names=list(task_names),
        n_classes=int(n_classes),
        device=str(device),
    )

    # Normalize label range to {1..K} if needed.
    y = np.asarray(y).astype(int)
    tasks = np.asarray(tasks).astype(int)
    if y.min() == 0 and y.max() == n_classes - 1:
        y = y + 1
    if y.min() < 1 or y.max() > n_classes:
        raise ValueError(f"Labels must be in 1..{n_classes} (got min={y.min()}, max={y.max()}).")
    if tasks.min() < 0 or tasks.max() >= len(task_names):
        raise ValueError(
            f"tasks must be in [0..{len(task_names)-1}] (got min={tasks.min()}, max={tasks.max()})."
        )

    # Outer CV split (SSOT used by both training and analysis scripts).
    from lumbar_pnam.cv_split import get_cv_splits

    cv_splits, cv_split_info = get_cv_splits(X=X, y=y, groups=groups, cfg=cfg, meta=meta)

    # Persist split metadata for reproducibility / paper appendix.
    (run_dir / "cv_split_info.json").write_text(
        json.dumps(cv_split_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fold_results: list[FoldResult] = []
    best_fold = 0
    best_kappa = -1e9
    oof_pred_dfs: list[pd.DataFrame] = []
    nested_thr_by_fold: dict[int, np.ndarray] = {}

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        fold_dir = run_dir / "checkpoints" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fs_summary: dict[str, Any] | None = None
        fs_cfg = cfg.get("feature_selection", {}) or {}
        fs_enabled = bool(fs_cfg.get("enabled", True))
        resolved_id_col = str(data_cfg.get("id_col", id_col or "case_id_层级"))
        if feature_selection_dataset is not None:
            fs_dir = fold_dir / "feature_selection"
            fs_dir.mkdir(parents=True, exist_ok=True)

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
                    pfirrmann_csv=(cfg.get("labels", {}) or {}).get("xlsx_path") or data_cfg.get("pfirrmann_csv_path"),
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
                add_segment_onehot=bool(data_cfg.get("add_segment_onehot", True)),
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

            X_train_raw = loaded_fold.X.iloc[train_idx]
            y_train = loaded_fold.y[train_idx]
            task_train = loaded_fold.tasks[train_idx]
            X_val_raw = loaded_fold.X.iloc[val_idx]
            y_val = loaded_fold.y[val_idx]
            task_val = loaded_fold.tasks[val_idx]
            fold_feature_names = list(loaded_fold.feature_names)
            fold_classic_feature_names = list(loaded_fold.classic_feature_names)
        else:
            X_train_raw = X.iloc[train_idx]
            y_train = y[train_idx]
            task_train = tasks[train_idx]
            X_val_raw = X.iloc[val_idx]
            y_val = y[val_idx]
            task_val = tasks[val_idx]
            fold_feature_names = list(X.columns)
            fold_classic_feature_names = list(classic_feature_names)

        class_weight: list[float] | None = None
        if class_weight_mode == "balanced":
            # Inverse-frequency weights, normalized to mean=1.
            counts = np.bincount((y_train - 1).astype(int), minlength=n_classes).astype(np.float32)
            counts[counts == 0] = 1.0
            w = float(counts.sum()) / (float(n_classes) * counts)
            w = w / float(np.mean(w))
            class_weight = w.tolist()

        # Fold-inner harmonization + preprocessing, fit on training fold only.
        harm = make_harmonizer(harm_ctx)
        harm.fit(X_train_raw)
        X_train_h = harm.transform(X_train_raw)
        X_val_h = harm.transform(X_val_raw)

        pre = (ZScore() if pre_method == "zscore" else RankGaussZScore(n_jobs=pre_n_jobs)).fit(X_train_h)
        X_train_t = pre.transform(X_train_h)
        X_val_t = pre.transform(X_val_h)

        pre_combined = HarmonizedPreprocessor(harmonizer=harm, preprocessor=pre)

        train_set = TaskDataset(X_train_t, task_train, y_train)
        val_set = TaskDataset(X_val_t, task_val, y_val)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=int(dataloader_num_workers))
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=int(dataloader_num_workers))

        eval_T = float(cfg.get("training", {}).get("eval_T", 1e-8))

        interaction_pairs: list[tuple[int, int]] = []
        interaction_scores: list[float] = []
        if use_interaction:
            # Select interactions strictly within the outer-fold training set.
            groups_train = np.asarray(groups)[train_idx]
            p_gt_oof, X_oof_t = _oof_p_gt_baseline(
                X=X_train_raw,
                y=y_train,
                task=task_train,
                groups=groups_train,
                harm_ctx=harm_ctx,
                n_tasks=len(task_names),
                n_classes=n_classes,
                decision_thresholds=decision_thresholds,
                beta=beta,
                qwk_loss_weight=qwk_loss_weight,
                acc_loss_weight=acc_loss_weight,
                lambda_out=lambda_out,
                lambda_seg=lambda_seg,
                lambda_task_smooth=lambda_task_smooth,
                device=device,
                seed=seed + 10000 + fold,
                lr_max=lr_max,
                lr_min=lr_min,
                weight_decay=weight_decay,
                batch_size=batch_size,
                max_epoch=int_sel_max_epoch,
                patience=int_sel_patience,
                min_delta=min_delta,
                stop_monitor=stop_monitor,
                select_monitor=select_monitor,
                eval_T=eval_T,
                p=p,
                h_dim=h_dim,
                n_proto=n_proto,
                n_layers=n_layers,
                n_layers_pred=n_layers_pred,
                batch_norm=batch_norm,
                dropout=dropout,
                dropout_output=dropout_output,
                tau=tau,
                share_task_weights_across_layers=share_task_weights_across_layers,
                class_weight_mode=class_weight_mode,
                inner_splits=int_inner_splits,
                shuffle=shuffle,
                random_state=random_state,
                pre_method=pre_method,
                pre_n_jobs=pre_n_jobs,
                num_workers=int(dataloader_num_workers),
            )
            residual = _coral_pseudo_residual(y=y_train, p_gt_oof=p_gt_oof, n_classes=n_classes)
            interaction_pairs, interaction_scores = _topk_interaction_pairs_by_spearman(
                X=X_oof_t,
                residual=residual,
                max_pairs=int_max_pairs,
            )

            feat_names = list(X_train_raw.columns)
            pair_rows: list[dict[str, Any]] = []
            for (a, b), s in zip(interaction_pairs, interaction_scores):
                pair_rows.append(
                    {
                        "a": int(a),
                        "b": int(b),
                        "feature_a": feat_names[int(a)] if 0 <= int(a) < len(feat_names) else None,
                        "feature_b": feat_names[int(b)] if 0 <= int(b) < len(feat_names) else None,
                        "score_abs_spearman": float(s),
                    }
                )
            (fold_dir / "interaction_pairs.json").write_text(
                json.dumps(
                    {
                        "enabled": True,
                        "max_pairs": int(int_max_pairs),
                        "inner_splits": int(int_inner_splits),
                        "selection_training": {
                            "max_epoch": int(int_sel_max_epoch),
                            "patience": int(int_sel_patience),
                        },
                        "pairs": pair_rows,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        if use_interaction:
            model = ProtoN2AMMultiTaskCoral(
                n_feat=X_train_t.shape[1],
                n_classes=n_classes,
                n_tasks=len(task_names),
                pairs=interaction_pairs,
                p=p,
                h_dim=h_dim,
                n_proto=n_proto,
                n_layers=n_layers,
                n_layers_pred=n_layers_pred,
                batch_norm=batch_norm,
                dropout=dropout,
                dropout_output=dropout_output,
                beta=beta,
                lambda_out=lambda_out,
                lambda_seg=lambda_seg,
                lambda_task_smooth=lambda_task_smooth,
                tau=tau,
                class_weight=class_weight,
                qwk_loss_weight=qwk_loss_weight,
                acc_loss_weight=acc_loss_weight,
                share_task_weights_across_layers=share_task_weights_across_layers,
                interaction_mlp_hidden_dim=int_mlp_hidden_dim,
                interaction_mlp_dropout=float(int_mlp_dropout),
            ).to(device)
        else:
            model = ProtoNAMMultiTaskCoral(
                n_feat=X_train_t.shape[1],
                n_classes=n_classes,
                n_tasks=len(task_names),
                p=p,
                h_dim=h_dim,
                n_proto=n_proto,
                n_layers=n_layers,
                n_layers_pred=n_layers_pred,
                batch_norm=batch_norm,
                dropout=dropout,
                dropout_output=dropout_output,
                beta=beta,
                lambda_out=lambda_out,
                lambda_seg=lambda_seg,
                lambda_task_smooth=lambda_task_smooth,
                tau=tau,
                class_weight=class_weight,
                qwk_loss_weight=qwk_loss_weight,
                acc_loss_weight=acc_loss_weight,
                share_task_weights_across_layers=share_task_weights_across_layers,
            ).to(device)

        # Prototype quantile init on the (preprocessed) training fold.
        model.initialize_prototypes(X_train_t)

        # Soft monotonicity constraints: resolve constrained feature indices for THIS fold's
        # feature_names, then build the evaluation grid on the training fold. Features absent from
        # the fold's selected set are silently ignored (force_include is best-effort per 方案1).
        if mono_enabled and mono_dir_by_feature:
            name_to_idx = {n: i for i, n in enumerate(fold_feature_names)}
            mono_indices: list[int] = []
            mono_directions: list[float] = []
            for fname, d in mono_dir_by_feature.items():
                if fname in name_to_idx and int(d) in (1, -1):
                    mono_indices.append(int(name_to_idx[fname]))
                    mono_directions.append(float(d))
            active = model.configure_monotonicity(
                enabled=True,
                mode=mono_mode,
                lambda_mono=mono_lambda,
                alpha_w=mono_alpha_w,
                apply_to=mono_apply_to,
                grid_size=mono_grid_size,
                grid_method=mono_grid_method,
                eps_grid=mono_eps_grid,
                w_pos_floor=mono_w_pos_floor,
                indices=mono_indices,
                directions=mono_directions,
                X_train=X_train_t,
            )
            if active:
                print(
                    f"[monotonicity] fold {fold}: enabled | n_features={len(mono_indices)} "
                    f"indices={mono_indices} dirs={mono_directions} "
                    f"lambda={mono_lambda} apply_to={mono_apply_to}"
                )
            else:
                print(
                    f"[monotonicity] fold {fold}: no constrained feature matched feature_names "
                    f"(direction_by_feature has {len(mono_dir_by_feature)} entries) -> skipped."
                )

        optim = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=weight_decay)
        total_steps = max(1, int(max_epoch) * max(1, len(train_loader)))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps, eta_min=lr_min)

        best_stop_val = float("inf") if stop_monitor == "val_loss_data" else -1e9
        best_select_val = float("inf") if select_monitor == "val_loss_data" else -1e9
        best_val_loss_data = float("inf")
        best_val_kappa = -1e9
        best_state: dict[str, Any] | None = None
        best_epoch = -1
        no_improve = 0
        history_rows: list[dict[str, Any]] = []
        global_step = 0

        for epoch in range(max_epoch):
            model.train()
            total_train = {
                "loss": 0.0,
                "loss_data": 0.0,
                "loss_last": 0.0,
                "loss_aux_sum": 0.0,
                "r_out": 0.0,
                "loss_out": 0.0,
                "r_seg": 0.0,
                "loss_seg": 0.0,
                "r_task_smooth": 0.0,
                "loss_task_smooth": 0.0,
                "loss_qwk": 0.0,
                "loss_acc": 0.0,
                "r_mono": 0.0,
                "r_mono_w": 0.0,
                "loss_mono": 0.0,
                "mono_n_viol": 0.0,
            }
            n_seen = 0

            for xb, tb, yb in train_loader:
                xb = xb.float().to(device)
                tb = tb.to(device).to(torch.int64)
                yb = yb.to(device).to(torch.int64)

                global_step += 1
                # Avoid numerical issues when schedule drives temperature extremely close to 0.
                # We floor it at eval_T so training eventually matches the inference setting.
                T = max(_proto_temperature(global_step, total_steps, tau), eval_T)

                optim.zero_grad()
                loss, stats = model(xb, tb, yb, T=T)
                loss.backward()
                optim.step()
                scheduler.step()

                bs = int(len(xb))
                n_seen += bs
                for k in total_train.keys():
                    total_train[k] += float(stats[k].item()) * bs

            # Validation loss for early stopping.
            T_sched = max(_proto_temperature(global_step, total_steps, tau), eval_T)
            # Use the inference temperature for validation/early-stopping to keep the selection
            # consistent with the final evaluation/prediction setting.
            val_loss_data, val_metrics, val_aux = _eval_on_loader(
                model,
                val_loader,
                device,
                n_classes=n_classes,
                decision_thresholds=decision_thresholds,
                proto_T=eval_T,
            )

            lr_now = float(optim.param_groups[0]["lr"])
            row = {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "lr": lr_now,
                "proto_T_sched": float(T_sched),
                "proto_T_eval": float(eval_T),
                "train_n": int(n_seen),
                "train_loss": total_train["loss"] / max(1, n_seen),
                "train_loss_data": total_train["loss_data"] / max(1, n_seen),
                "train_loss_last": total_train["loss_last"] / max(1, n_seen),
                "train_loss_aux_sum": total_train["loss_aux_sum"] / max(1, n_seen),
                "train_r_out": total_train["r_out"] / max(1, n_seen),
                "train_loss_out": total_train["loss_out"] / max(1, n_seen),
                "train_r_seg": total_train["r_seg"] / max(1, n_seen),
                "train_loss_seg": total_train["loss_seg"] / max(1, n_seen),
                "train_r_task_smooth": total_train["r_task_smooth"] / max(1, n_seen),
                "train_loss_task_smooth": total_train["loss_task_smooth"] / max(1, n_seen),
                "train_loss_qwk": total_train["loss_qwk"] / max(1, n_seen),
                "train_loss_acc": total_train["loss_acc"] / max(1, n_seen),
                "train_r_mono": total_train["r_mono"] / max(1, n_seen),
                "train_r_mono_w": total_train["r_mono_w"] / max(1, n_seen),
                "train_loss_mono": total_train["loss_mono"] / max(1, n_seen),
                "train_mono_n_viol": total_train["mono_n_viol"] / max(1, n_seen),
                "val_loss_data": float(val_loss_data),
                "val_kappa": float(val_metrics.kappa_quadratic),
                "val_mae": float(val_metrics.mae),
                "val_spearman": float(val_metrics.spearman),
                "val_acc_pm1": float(val_metrics.acc_pm1),
                "val_acc": float(val_metrics.acc),
                "val_bacc": float(val_metrics.bacc),
                "val_macro_f1": float(val_metrics.macro_f1),
                "val_weighted_f1": float(val_metrics.weighted_f1),
                "val_ccc": float(val_metrics.ccc),
                "val_loss_total": float(val_aux["loss_total"]),
                "val_loss_out": float(val_aux["loss_out"]),
                "val_r_out": float(val_aux["r_out"]),
                "val_loss_seg": float(val_aux["loss_seg"]),
                "val_r_seg": float(val_aux["r_seg"]),
                "val_loss_task_smooth": float(val_aux["loss_task_smooth"]),
                "val_r_task_smooth": float(val_aux["r_task_smooth"]),
                "val_loss_qwk": float(val_aux["loss_qwk"]),
                "val_loss_acc": float(val_aux["loss_acc"]),
            }
            history_rows.append(row)

            # Best-checkpoint selection / early stopping MUST run every epoch (方案 0.1).
            cur_loss = float(val_loss_data)
            cur_kappa = float(val_metrics.kappa_quadratic)
            cur_acc = float(val_metrics.acc)

            def _mon_value(name: str) -> float:
                if name == "val_loss_data":
                    return cur_loss
                if name == "val_kappa":
                    return cur_kappa
                if name == "val_acc":
                    return cur_acc
                raise ValueError(f"Unknown monitor: {name!r}")

            cur_stop_val = _mon_value(stop_monitor)
            cur_select_val = _mon_value(select_monitor)

            improved_stop = (
                (cur_stop_val < best_stop_val - min_delta)
                if stop_monitor == "val_loss_data"
                else (cur_stop_val > best_stop_val + min_delta)
            )
            improved_select = (
                (cur_select_val < best_select_val - min_delta)
                if select_monitor == "val_loss_data"
                else (cur_select_val > best_select_val + min_delta)
            )

            if improved_select:
                best_select_val = cur_select_val
                best_val_loss_data = cur_loss
                best_val_kappa = cur_kappa
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if improved_stop:
                best_stop_val = cur_stop_val
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        pd.DataFrame(history_rows).to_csv(fold_dir / "training_history.csv", index=False, encoding="utf-8")

        # Restore best.
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation on val.
        val_loss_data, metrics, val_aux = _eval_on_loader(
            model,
            val_loader,
            device,
            n_classes=n_classes,
            decision_thresholds=decision_thresholds,
            proto_T=eval_T,
        )

        # Save validation predictions + plots to make results inspectable.
        pred = _predict_on_loader(
            model,
            val_loader,
            device,
            n_classes=n_classes,
            decision_thresholds=decision_thresholds,
            proto_T=eval_T,
        )

        # --- 方案 0.2: nested within-train threshold calibration ---
        fold_thr_by_task: np.ndarray | None = None
        if tcal_enabled:
            groups_train = np.asarray(groups)[train_idx]
            fold_thr_by_task = _nested_threshold_calibration(
                X_train_raw=X_train_raw,
                y_train=y_train,
                task_train=task_train,
                groups_train=groups_train,
                harm_ctx=harm_ctx,
                n_tasks=len(task_names),
                n_classes=n_classes,
                decision_thresholds=decision_thresholds,
                pre_method=pre_method,
                pre_n_jobs=pre_n_jobs,
                calib_splits=tcal_splits,
                shuffle=shuffle,
                random_state=random_state,
                grid=tcal_grid,
                n_iter=tcal_n_iter,
                objective=tcal_objective,
                tie_breakers=tcal_tie_breakers,
                beta=beta,
                qwk_loss_weight=qwk_loss_weight,
                acc_loss_weight=acc_loss_weight,
                lambda_out=lambda_out,
                lambda_seg=lambda_seg,
                lambda_task_smooth=lambda_task_smooth,
                device=device,
                seed=seed + 50000 + fold,
                lr_max=lr_max,
                lr_min=lr_min,
                weight_decay=weight_decay,
                batch_size=batch_size,
                max_epoch=max_epoch,
                patience=patience,
                min_delta=min_delta,
                stop_monitor=stop_monitor,
                select_monitor=select_monitor,
                eval_T=eval_T,
                p=p,
                h_dim=h_dim,
                n_proto=n_proto,
                n_layers=n_layers,
                n_layers_pred=n_layers_pred,
                batch_norm=batch_norm,
                dropout=dropout,
                dropout_output=dropout_output,
                tau=tau,
                share_task_weights_across_layers=share_task_weights_across_layers,
                class_weight_mode=class_weight_mode,
                num_workers=int(dataloader_num_workers),
            )
            nested_thr_by_fold[int(fold)] = fold_thr_by_task

        pred_df = pd.DataFrame(
            {
                "orig_index": val_idx,
                "task": pred["task"],
                "y_true": pred["y_true"],
                "y_pred": pred["y_pred"],
                "y_cont": pred["y_cont"],
                "s": pred["s"],
            }
        )
        for m in range(1, n_layers + 1):
            pred_df[f"s_layer_{m}"] = pred[f"s_layer_{m}"]
        for k in range(1, n_classes):
            pred_df[f"p_gt_{k}"] = pred[f"p_gt_{k}"]
        for k in range(1, n_classes + 1):
            pred_df[f"p_cls_{k}"] = pred[f"p_cls_{k}"]

        if meta is not None and id_col and id_col in meta.columns:
            meta_val = meta.iloc[val_idx].reset_index(drop=True)
            pred_df.insert(0, id_col, meta_val[id_col])
            if "patient_id" in meta_val.columns:
                pred_df.insert(1, "patient_id", meta_val["patient_id"])
            if "disc_level" in meta_val.columns:
                pred_df.insert(2, "disc_level", meta_val["disc_level"])

        if fold_thr_by_task is not None:
            p_gt_val = np.stack([pred[f"p_gt_{k}"] for k in range(1, n_classes)], axis=1).astype(float)
            task_val_arr = pred["task"].astype(int)
            y_pred_cal = np.zeros((len(task_val_arr),), dtype=int)
            for t in range(len(task_names)):
                mt = task_val_arr == int(t)
                if int(mt.sum()) == 0:
                    continue
                y_pred_cal[mt] = _decode_p_gt(p_gt_val[mt], fold_thr_by_task[int(t)])
            pred_df["y_pred_calibrated"] = y_pred_cal

        pred_df.to_csv(fold_dir / "val_predictions.csv", index=False, encoding="utf-8")
        oof_pred_dfs.append(pred_df)

        _save_confusion_matrix(
            y_true=pred["y_true"],
            y_pred=pred["y_pred"],
            n_classes=n_classes,
            out_path=fold_dir / "confusion_matrix_val.png",
            normalize=False,
            title=f"Fold {fold} confusion matrix (val)",
        )
        _save_confusion_matrix(
            y_true=pred["y_true"],
            y_pred=pred["y_pred"],
            n_classes=n_classes,
            out_path=fold_dir / "confusion_matrix_val_norm.png",
            normalize=True,
            title=f"Fold {fold} confusion matrix (val, row-normalized)",
        )
        _save_ycont_scatter(
            y_true=pred["y_true"],
            y_cont=pred["y_cont"],
            out_path=fold_dir / "ycont_scatter_val.png",
            title=f"Fold {fold} continuous output vs true (val)",
        )

        # Per-task metrics for debugging (segment-wise).
        by_task_rows: list[dict[str, Any]] = []
        for t in range(len(task_names)):
            mask = pred_df["task"].to_numpy().astype(int) == t
            if int(mask.sum()) == 0:
                continue
            m = compute_ordinal_metrics(
                y_true=pred_df.loc[mask, "y_true"].to_numpy(),
                y_pred=pred_df.loc[mask, "y_pred"].to_numpy(),
                y_cont=pred_df.loc[mask, "y_cont"].to_numpy(),
                n_classes=n_classes,
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

        # Save fold artifacts.
        torch.save(model.state_dict(), fold_dir / "best_model.pt")
        with open(fold_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump(pre_combined, f, protocol=pickle.HIGHEST_PROTOCOL)
        (fold_dir / "feature_names.json").write_text(
            json.dumps(
                {
                    "feature_names": fold_feature_names,
                    "classic_feature_names": fold_classic_feature_names,
                    "task_names": task_names,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        theta = model.coral.thresholds.detach().cpu().numpy().tolist()

        multitask_payload: dict[str, Any] = {"task_names": task_names}
        interaction_payload: dict[str, Any] | None = None
        if isinstance(model, ProtoNAMMultiTaskCoral):
            task_w = model.task_w.detach().cpu().numpy().tolist()
            task_b = model.task_b.detach().cpu().numpy().tolist()
            multitask_payload["task_w"] = task_w
            multitask_payload["task_b"] = task_b
        elif isinstance(model, ProtoN2AMMultiTaskCoral):
            task_w_main = model.task_w_main.detach().cpu().numpy().tolist()
            task_w_int = model.task_w_int.detach().cpu().numpy().tolist()
            task_b = model.task_b.detach().cpu().numpy().tolist()
            multitask_payload["task_w_main"] = task_w_main
            multitask_payload["task_w_int"] = task_w_int
            multitask_payload["task_b"] = task_b
            interaction_payload = {
                "enabled": True,
                "n_pairs": int(model.n_pairs),
                "pairs_file": "interaction_pairs.json",
            }
        else:
            raise TypeError(f"Unexpected model type: {type(model)}")
        best_row = None
        for r in history_rows:
            if int(r.get("epoch", -1)) == int(best_epoch):
                best_row = r
                break
        (fold_dir / "fold_info.json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "best_select": {"name": select_monitor, "value": float(best_select_val)},
                    "best_stop": {"name": stop_monitor, "value": float(best_stop_val)},
                    "best_val_loss": float(best_val_loss_data),
                    "best_val_loss_data": float(best_val_loss_data),
                    "best_val_kappa": float(best_val_kappa),
                    "early_stopping": {
                        "stop_monitor": stop_monitor,
                        "select_monitor": select_monitor,
                        "eval_T": float(eval_T),
                        "best_epoch": int(best_epoch),
                        "best_stop_value": float(best_stop_val),
                        "best_select_value": float(best_select_val),
                        "best_val_loss_total": None if best_row is None else float(best_row.get("val_loss_total", 0.0)),
                        "best_val_loss_out": None if best_row is None else float(best_row.get("val_loss_out", 0.0)),
                        "best_val_r_out": None if best_row is None else float(best_row.get("val_r_out", 0.0)),
                        "best_val_loss_seg": None if best_row is None else float(best_row.get("val_loss_seg", 0.0)),
                        "best_val_r_seg": None if best_row is None else float(best_row.get("val_r_seg", 0.0)),
                        "best_val_loss_task_smooth": None
                        if best_row is None
                        else float(best_row.get("val_loss_task_smooth", 0.0)),
                        "best_val_r_task_smooth": None if best_row is None else float(best_row.get("val_r_task_smooth", 0.0)),
                        "best_val_loss_qwk": None if best_row is None else float(best_row.get("val_loss_qwk", 0.0)),
                        "best_val_loss_acc": None if best_row is None else float(best_row.get("val_loss_acc", 0.0)),
                    },
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
                    "val_losses_evalT": {
                        "eval_T": float(eval_T),
                        "loss_data": float(val_loss_data),
                        "loss_total": float(val_aux["loss_total"]),
                        "loss_out": float(val_aux["loss_out"]),
                        "r_out": float(val_aux["r_out"]),
                        "loss_seg": float(val_aux["loss_seg"]),
                        "r_seg": float(val_aux["r_seg"]),
                        "loss_task_smooth": float(val_aux["loss_task_smooth"]),
                        "r_task_smooth": float(val_aux["r_task_smooth"]),
                        "loss_qwk": float(val_aux["loss_qwk"]),
                        "loss_acc": float(val_aux["loss_acc"]),
                    },
                    "coral": {"thresholds_theta": theta},
                    "multitask": multitask_payload,
                    **({} if fs_summary is None else {"feature_selection": fs_summary}),
                    **({} if interaction_payload is None else {"interaction": interaction_payload}),
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

    # --- 4.4.2 pooled OOF metrics (point estimate) ---
    if not oof_pred_dfs:
        raise RuntimeError("Missing fold validation predictions; cannot compute pooled OOF metrics.")

    oof_df = pd.concat(oof_pred_dfs, axis=0, ignore_index=True)
    if "orig_index" not in oof_df.columns:
        raise RuntimeError("val_predictions missing orig_index; cannot pool OOF predictions.")
    if int(oof_df["orig_index"].nunique()) != int(len(y)):
        raise RuntimeError(
            "OOF pooled predictions do not cover all samples exactly once: "
            f"n_unique_orig_index={int(oof_df['orig_index'].nunique())} vs n_samples={int(len(y))}"
        )
    oof_df = oof_df.sort_values("orig_index").reset_index(drop=True)
    oof_df.to_csv(run_dir / "oof_predictions.csv", index=False, encoding="utf-8")

    # --- 方案 0.2: pooled nested-calibrated OOF (unbiased: thresholds learned within-train only) ---
    if tcal_enabled and "y_pred_calibrated" in oof_df.columns and nested_thr_by_fold:
        n_bounds = int(n_classes) - 1
        oof_df.to_csv(run_dir / "oof_predictions_nested_calibrated.csv", index=False, encoding="utf-8")

        m_nested = compute_ordinal_metrics(
            y_true=oof_df["y_true"].to_numpy(),
            y_pred=oof_df["y_pred_calibrated"].to_numpy(),
            y_cont=oof_df["y_cont"].to_numpy(),
            n_classes=n_classes,
        )

        thr_stack = np.full((len(nested_thr_by_fold), len(task_names), n_bounds), np.nan, dtype=np.float64)
        per_fold_thr: dict[str, dict[str, list[float]]] = {}
        for i, f in enumerate(sorted(nested_thr_by_fold)):
            thr_stack[i] = nested_thr_by_fold[f]
            per_fold_thr[str(f)] = {
                str(task_names[t]): [float(x) for x in nested_thr_by_fold[f][t, :].tolist()]
                for t in range(len(task_names))
            }
        thr_median = np.nanmedian(thr_stack, axis=0)
        median_thr = {
            str(task_names[t]): [float(x) for x in thr_median[t, :].tolist()] for t in range(len(task_names))
        }

        (run_dir / "decision_thresholds_nested.json").write_text(
            json.dumps(
                {
                    "method": "nested_within_train_calibration",
                    "objective": {"primary": str(tcal_objective), "tie_breakers": list(tcal_tie_breakers)},
                    "calib_splits": int(tcal_splits),
                    "grid": {
                        "min": float(tcal_grid_min),
                        "max": float(tcal_grid_max),
                        "step": float(tcal_grid_step),
                        "n_iter": int(tcal_n_iter),
                    },
                    "n_classes": int(n_classes),
                    "note": "CORAL decode: y_pred = 1 + sum_k I(p_gt_k > thr_k), k=1..K-1.",
                    "thresholds_by_fold": per_fold_thr,
                    "thresholds_median_by_task": median_thr,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (run_dir / "oof_metrics_nested_calibrated.json").write_text(
            json.dumps(
                {
                    "method": "nested_within_train_calibration",
                    "objective": {"primary": "qwk", "tie_breakers": ["-mae", "macro_f1"]},
                    "note": "Unbiased calibrated OOF: thresholds learned strictly inside each outer-train fold.",
                    "point_estimate": _metrics_to_dict(m_nested),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print("\n[Nested Threshold Calibration] (方案 0.2, unbiased within-train)")
        print(f"- OOF MAE (nested-calibrated): {m_nested.mae:.4f}")
        print(f"- OOF Kappa(q) (nested-calibrated): {m_nested.kappa_quadratic:.4f}")
        print(f"- OOF Acc (nested-calibrated): {m_nested.acc:.4f}")
        print(f"- OOF Macro-F1 (nested-calibrated): {m_nested.macro_f1:.4f}")
        print(f"- OOF BAcc (nested-calibrated): {m_nested.bacc:.4f}")

    oof_metrics = compute_ordinal_metrics(
        y_true=oof_df["y_true"].to_numpy(),
        y_pred=oof_df["y_pred"].to_numpy(),
        y_cont=oof_df["y_cont"].to_numpy(),
        n_classes=n_classes,
    )

    _save_confusion_matrix(
        y_true=oof_df["y_true"].to_numpy(),
        y_pred=oof_df["y_pred"].to_numpy(),
        n_classes=n_classes,
        out_path=run_dir / "confusion_matrix_oof.png",
        normalize=False,
        title="OOF pooled confusion matrix",
    )
    _save_confusion_matrix(
        y_true=oof_df["y_true"].to_numpy(),
        y_pred=oof_df["y_pred"].to_numpy(),
        n_classes=n_classes,
        out_path=run_dir / "confusion_matrix_oof_norm.png",
        normalize=True,
        title="OOF pooled confusion matrix (row-normalized)",
    )
    _save_ycont_scatter(
        y_true=oof_df["y_true"].to_numpy(),
        y_cont=oof_df["y_cont"].to_numpy(),
        out_path=run_dir / "ycont_scatter_oof.png",
        title="OOF pooled continuous output vs true",
    )

    # --- 4.4.3 patient-level clustered bootstrap CI on pooled OOF ---
    oof_bootstrap_ci95: dict[str, tuple[float, float]] | None = None
    oof_bootstrap_n = 0
    if boot_enabled and int(boot_n) > 0:
        idx_oof = oof_df["orig_index"].to_numpy().astype(int, copy=False)
        groups_oof = np.asarray(groups)[idx_oof]
        oof_bootstrap_ci95 = _patient_cluster_bootstrap_ci95(
            y_true=y[idx_oof],
            y_pred=oof_df["y_pred"].to_numpy(),
            y_cont=oof_df["y_cont"].to_numpy(),
            groups=groups_oof,
            n_classes=n_classes,
            n_resamples=int(boot_n),
            alpha=float(boot_alpha),
            seed=int(boot_seed),
            n_jobs=int(boot_n_jobs),
        )
        oof_bootstrap_n = int(boot_n) if oof_bootstrap_ci95 else 0

        (run_dir / "oof_bootstrap_ci95.json").write_text(
            json.dumps(
                {
                    "enabled": True,
                    "n_resamples": int(boot_n),
                    "alpha": float(boot_alpha),
                    "seed": int(boot_seed),
                    "n_jobs": int(boot_n_jobs),
                    "ci95": oof_bootstrap_ci95,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    else:
        (run_dir / "oof_bootstrap_ci95.json").write_text(
            json.dumps({"enabled": False}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Save pooled OOF point estimate as a single JSON (easy to parse for paper tables).
    (run_dir / "oof_metrics.json").write_text(
        json.dumps(
            {
                "point_estimate": _metrics_to_dict(oof_metrics),
                "bootstrap_ci95": oof_bootstrap_ci95,
                "bootstrap_n": int(oof_bootstrap_n),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --- Fold-mean metrics kept for reference (NOT the plan's point estimate) ---
    mae = float(np.mean([fr.metrics.mae for fr in fold_results]))
    kappa = float(np.mean([fr.metrics.kappa_quadratic for fr in fold_results]))
    spearman = float(np.mean([fr.metrics.spearman for fr in fold_results]))
    acc_pm1 = float(np.mean([fr.metrics.acc_pm1 for fr in fold_results]))
    acc = float(np.mean([fr.metrics.acc for fr in fold_results]))
    bacc = float(np.mean([fr.metrics.bacc for fr in fold_results]))
    macro_f1 = float(np.mean([fr.metrics.macro_f1 for fr in fold_results]))
    weighted_f1 = float(np.mean([fr.metrics.weighted_f1 for fr in fold_results]))
    ccc = float(np.mean([fr.metrics.ccc for fr in fold_results]))
    fold_mean_metrics = OrdinalMetrics(
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

    return CVResult(
        fold_results=fold_results,
        best_fold=best_fold,
        mean_metrics=oof_metrics,
        fold_mean_metrics=fold_mean_metrics,
        oof_metrics=oof_metrics,
        oof_bootstrap_ci95=oof_bootstrap_ci95,
        oof_bootstrap_n=int(oof_bootstrap_n),
    )
