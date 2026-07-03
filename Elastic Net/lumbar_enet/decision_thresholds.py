from __future__ import annotations

from typing import Any

import numpy as np


def _thr_vec_from_any(v: Any, *, n_bounds: int) -> list[float]:
    """
    Convert a threshold spec to a length-(K-1) float vector.

    Supported:
      - scalar: replicate to all bounds
      - list/tuple: take first K-1 entries
    """
    if isinstance(v, (int, float)):
        return [float(v)] * int(n_bounds)
    if isinstance(v, (list, tuple)):
        if len(v) < int(n_bounds):
            raise ValueError(f"threshold list must have length >= {int(n_bounds)} (got {len(v)})")
        return [float(x) for x in list(v)[: int(n_bounds)]]
    raise TypeError("threshold must be a number or a list/tuple.")


def parse_threshold_matrix(
    *,
    ord_cfg: dict[str, Any],
    task_names: list[str],
    n_classes: int,
) -> tuple[np.ndarray, float]:
    """
    Parse config.ordinal.decision_threshold into a (n_tasks, K-1) matrix.

    This follows ProtoNAM's config semantics, but for Elastic Net we interpret the
    thresholds as *fractional offsets* used to map y_cont -> y_pred:
        y_pred = 1 + sum_k I(y_cont > k + thr_k),  k=1..K-1

    Returns:
        thr_by_task: shape (n_tasks, K-1)
        thr5_placeholder: a value to fill the 5th column in output CSV (unused when K=5)
    """
    K = int(n_classes)
    n_bounds = int(K - 1)
    spec = ord_cfg.get("decision_threshold", 0.5)

    default_vec = _thr_vec_from_any(0.5, n_bounds=n_bounds)
    thr5_placeholder = 0.5

    if isinstance(spec, dict):
        default_raw = spec.get("default", 0.5)
        default_vec = _thr_vec_from_any(default_raw, n_bounds=n_bounds)
        if isinstance(default_raw, (list, tuple)) and len(default_raw) >= 5:
            thr5_placeholder = float(default_raw[4])
        else:
            thr5_placeholder = float(default_vec[-1]) if default_vec else 0.5

        by_task = spec.get("by_task", None)
        thr = np.tile(np.asarray(default_vec, dtype=np.float64)[None, :], (len(task_names), 1))
        if by_task is None:
            return thr, thr5_placeholder

        name_to_idx = {name: i for i, name in enumerate(list(task_names))}
        if isinstance(by_task, dict):
            for k, v in by_task.items():
                if isinstance(k, str) and k in name_to_idx:
                    idx = int(name_to_idx[k])
                else:
                    idx = int(k)
                thr[idx, :] = np.asarray(_thr_vec_from_any(v, n_bounds=n_bounds), dtype=np.float64)
            return thr, thr5_placeholder

        if isinstance(by_task, (list, tuple)) and by_task and isinstance(by_task[0], (list, tuple)):
            if len(by_task) != len(task_names):
                raise ValueError(f"by_task rows must equal n_tasks={len(task_names)} (got {len(by_task)})")
            thr = np.asarray([_thr_vec_from_any(row, n_bounds=n_bounds) for row in list(by_task)], dtype=np.float64)
            return thr, thr5_placeholder

        raise TypeError("decision_threshold.by_task must be a dict or a list of lists.")

    if isinstance(spec, (list, tuple)) and spec and isinstance(spec[0], (list, tuple)):
        if len(spec) != len(task_names):
            raise ValueError(f"decision_threshold rows must equal n_tasks={len(task_names)} (got {len(spec)})")
        thr = np.asarray([_thr_vec_from_any(row, n_bounds=n_bounds) for row in list(spec)], dtype=np.float64)
        return thr, thr5_placeholder

    default_vec = _thr_vec_from_any(spec, n_bounds=n_bounds)
    if isinstance(spec, (list, tuple)) and len(spec) >= 5:
        thr5_placeholder = float(spec[4])
    else:
        thr5_placeholder = float(default_vec[-1]) if default_vec else 0.5
    thr = np.tile(np.asarray(default_vec, dtype=np.float64)[None, :], (len(task_names), 1))
    return thr, thr5_placeholder


def predict_from_y_cont(y_cont: np.ndarray, thr_vec: np.ndarray) -> np.ndarray:
    """
    Map continuous outputs to ordinal grades using fractional offsets:
        y_pred = 1 + sum_k I(y_cont > k + thr_k), k=1..K-1
    """
    y_cont = np.asarray(y_cont, dtype=float).reshape(-1)
    thr_vec = np.asarray(thr_vec, dtype=float).reshape(-1)
    K = int(thr_vec.size + 1)
    ks = np.arange(1, K, dtype=np.float64)
    bounds = ks + thr_vec  # (K-1,)
    return 1 + (y_cont[:, None] > bounds[None, :]).sum(axis=1).astype(int)


def predict_from_y_cont_by_task(*, y_cont: np.ndarray, task: np.ndarray, thr_by_task: np.ndarray) -> np.ndarray:
    """
    Per-task variant of predict_from_y_cont (each task has its own thr vector).
    """
    y_cont = np.asarray(y_cont, dtype=float).reshape(-1)
    task = np.asarray(task, dtype=int).reshape(-1)
    thr_by_task = np.asarray(thr_by_task, dtype=float)
    if y_cont.shape[0] != task.shape[0]:
        raise ValueError(f"y_cont/task length mismatch: {y_cont.shape[0]} vs {task.shape[0]}")

    out = np.zeros((y_cont.shape[0],), dtype=int)
    for t in np.unique(task).tolist():
        m = task == int(t)
        if int(m.sum()) == 0:
            continue
        out[m] = predict_from_y_cont(y_cont[m], thr_by_task[int(t)])
    return out.astype(int)

