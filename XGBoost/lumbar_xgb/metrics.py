from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score


@dataclass(frozen=True)
class OrdinalMetrics:
    mae: float
    kappa_quadratic: float
    spearman: float
    acc_pm1: float
    acc: float
    bacc: float
    macro_f1: float
    weighted_f1: float
    ccc: float


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """
    Numpy-only rankdata (average ranks for ties), 1..n.
    """
    x = np.asarray(x)
    n = int(x.size)
    if n == 0:
        return x.astype(float)

    order = np.argsort(x, kind="mergesort")  # stable for reproducibility
    xs = x[order]
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        # average rank for [i..j], ranks are 1-based
        r = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = r
        i = j + 1
    return ranks


def _spearmanr_safe(y_true: np.ndarray, y_cont: np.ndarray) -> float:
    """
    Spearman correlation computed without pandas/scipy to avoid platform-specific crashes.
    """
    y_true = np.asarray(y_true).astype(float).reshape(-1)
    y_cont = np.asarray(y_cont).astype(float).reshape(-1)
    m = np.isfinite(y_true) & np.isfinite(y_cont)
    y_true = y_true[m]
    y_cont = y_cont[m]
    if y_true.size < 2:
        return 0.0
    if np.unique(y_true).size < 2 or np.unique(y_cont).size < 2:
        return 0.0

    r1 = _rankdata_average(y_true)
    r2 = _rankdata_average(y_cont)
    r1 = r1 - r1.mean()
    r2 = r2 - r2.mean()
    denom = float(np.sqrt(np.sum(r1 * r1) * np.sum(r2 * r2)))
    if denom <= 0:
        return 0.0
    return float(np.sum(r1 * r2) / denom)


def _lin_ccc(*, y_true: np.ndarray, y_cont: np.ndarray) -> float:
    """
    Lin's Concordance Correlation Coefficient (CCC) between u=y_true and v=y_cont.
    """
    u = np.asarray(y_true, dtype=float).reshape(-1)
    v = np.asarray(y_cont, dtype=float).reshape(-1)
    m = np.isfinite(u) & np.isfinite(v)
    u = u[m]
    v = v[m]
    if u.size == 0:
        return 0.0

    mu_u = float(u.mean())
    mu_v = float(v.mean())
    var_u = float(u.var())
    var_v = float(v.var())
    std_u = float(np.sqrt(var_u))
    std_v = float(np.sqrt(var_v))

    if std_u <= 0.0 or std_v <= 0.0:
        rho = 0.0
    else:
        rho_val = float(np.corrcoef(u, v)[0, 1])
        rho = rho_val if np.isfinite(rho_val) else 0.0

    denom = var_u + var_v + (mu_u - mu_v) ** 2
    if denom <= 0.0 or not np.isfinite(denom):
        return 0.0

    ccc_val = (2.0 * rho * std_u * std_v) / denom
    return float(ccc_val) if np.isfinite(ccc_val) else 0.0


def compute_ordinal_metrics(
    *, y_true: np.ndarray, y_pred: np.ndarray, y_cont: np.ndarray, n_classes: int | None = None
) -> OrdinalMetrics:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_cont = np.asarray(y_cont).astype(float)
    if n_classes is None:
        # Labels are in {1..K} in this project.
        max_true = int(y_true.max()) if y_true.size else 1
        max_pred = int(y_pred.max()) if y_pred.size else 1
        n_classes = int(max(max_true, max_pred))
    labels = np.arange(1, int(n_classes) + 1, dtype=int)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    acc_pm1 = float(np.mean(np.abs(y_true - y_pred) <= 1))
    acc = float(np.mean(y_true == y_pred))

    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        kappa = 0.0
    else:
        kappa_val = cohen_kappa_score(y_true, y_pred, weights="quadratic")
        kappa = float(kappa_val) if np.isfinite(kappa_val) else 0.0

    spearman = _spearmanr_safe(y_true, y_cont)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    support = cm.sum(axis=1).astype(float)  # true count per class
    diag = np.diag(cm).astype(float)
    rec = np.divide(diag, support, out=np.zeros_like(support, dtype=float), where=support > 0)
    bacc = float(rec.mean()) if rec.size else 0.0

    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    ccc = _lin_ccc(y_true=y_true, y_cont=y_cont)

    return OrdinalMetrics(
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
