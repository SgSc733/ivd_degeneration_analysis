from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class CDResult:
    coef: np.ndarray  # shape (p,)
    intercept: float
    n_iter: int
    converged: bool


def _soft_threshold(x: float, lam: float) -> float:
    if x > lam:
        return x - lam
    if x < -lam:
        return x + lam
    return 0.0


def compute_lambda_max(
    *,
    X: np.ndarray,
    y: np.ndarray,
    penalty_factors: np.ndarray,
    l1_ratio: float,
    fit_intercept: bool,
) -> float:
    """
    Compute lambda_max such that penalized coefficients become all zeros.

    For v_j=0 (unpenalized columns), we first regress them out (together with intercept),
    then compute correlations on the residual.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    v = np.asarray(penalty_factors, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n, p = X.shape
    if v.shape[0] != p:
        raise ValueError("penalty_factors length must match X.shape[1]")
    if l1_ratio <= 0:
        raise ValueError("l1_ratio must be > 0 to define lambda_max.")

    unpen = v <= 0
    if fit_intercept or np.any(unpen):
        cols = []
        if fit_intercept:
            cols.append(np.ones((n, 1), dtype=np.float64))
        if np.any(unpen):
            cols.append(X[:, unpen])
        U = np.concatenate(cols, axis=1) if cols else None
        if U is None or U.size == 0:
            r = y - y.mean()
        else:
            # Least squares for unpenalized part.
            w, *_ = np.linalg.lstsq(U, y, rcond=None)
            r = y - U @ w
    else:
        r = y.copy()

    pen = v > 0
    if not np.any(pen):
        return 0.0

    # (1/n) * |X_j^T r| / v_j
    corr = np.abs((X[:, pen].T @ r) / float(n)) / v[pen]
    lam_max = float(np.max(corr)) / float(l1_ratio)
    if not np.isfinite(lam_max) or lam_max <= 0:
        lam_max = 0.0
    return lam_max


def fit_weighted_elastic_net_cd(
    *,
    X: np.ndarray,
    y: np.ndarray,
    penalty_factors: np.ndarray,
    l1_ratio: float,
    lambda_: float,
    fit_intercept: bool,
    selection: Literal["cyclic", "random"],
    max_iter: int,
    tol: float,
    random_state: int,
    coef_init: np.ndarray | None = None,
    intercept_init: float | None = None,
) -> CDResult:
    """
    Weighted Elastic Net coordinate descent for the objective:

      (1/2n)||y - (b0 + X beta)||^2
        + lambda * [ alpha * sum_j v_j |beta_j| + (1-alpha)/2 * sum_j v_j beta_j^2 ]

    Where v_j can be 0 (unpenalized; e.g., segment intercept dummies).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    v = np.asarray(penalty_factors, dtype=np.float64).reshape(-1)

    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("y length must match X rows.")
    if v.shape[0] != p:
        raise ValueError("penalty_factors length must match X columns.")
    if max_iter <= 0:
        raise ValueError("max_iter must be > 0.")
    if tol <= 0:
        raise ValueError("tol must be > 0.")
    if lambda_ < 0:
        raise ValueError("lambda_ must be >= 0.")
    if not (0 <= l1_ratio <= 1):
        raise ValueError("l1_ratio must be in [0, 1].")
    if selection not in ("cyclic", "random"):
        raise ValueError("selection must be 'cyclic' or 'random'.")

    coef = np.zeros(p, dtype=np.float64) if coef_init is None else np.asarray(coef_init, dtype=np.float64).copy()
    if coef.shape[0] != p:
        raise ValueError("coef_init length must match X columns.")

    if fit_intercept:
        intercept = float(y.mean()) if intercept_init is None else float(intercept_init)
    else:
        intercept = 0.0 if intercept_init is None else float(intercept_init)

    y_pred = intercept + X @ coef
    z = (X * X).mean(axis=0)  # (1/n) * sum x^2

    rng = np.random.RandomState(int(random_state))
    order = np.arange(p, dtype=int)

    for it in range(int(max_iter)):
        max_delta = 0.0

        if selection == "random":
            rng.shuffle(order)
            idx_iter = order
        else:
            idx_iter = range(p)

        for j in idx_iter:
            zj = float(z[j])
            if zj == 0.0:
                continue
            bj_old = float(coef[j])

            # Partial residual update.
            r = y - y_pred + X[:, j] * bj_old
            rho = float((X[:, j] @ r) / float(n))  # (1/n) * x_j^T r

            vj = float(v[j])
            if vj <= 0.0 or lambda_ == 0.0:
                # Unpenalized coordinate (or no penalty at all).
                bj_new = rho / zj
            else:
                thr = float(lambda_ * l1_ratio * vj)
                num = _soft_threshold(rho, thr)
                den = zj + float(lambda_ * (1.0 - l1_ratio) * vj)
                bj_new = num / den if den != 0.0 else 0.0

            delta = bj_new - bj_old
            if delta != 0.0:
                coef[j] = bj_new
                y_pred += X[:, j] * delta
                if abs(delta) > max_delta:
                    max_delta = abs(delta)

        if fit_intercept:
            # Closed-form intercept update.
            resid_mean = float((y - y_pred).mean())
            if resid_mean != 0.0:
                intercept += resid_mean
                y_pred += resid_mean
                if abs(resid_mean) > max_delta:
                    max_delta = abs(resid_mean)

        if max_delta < tol:
            return CDResult(coef=coef, intercept=float(intercept), n_iter=it + 1, converged=True)

    return CDResult(coef=coef, intercept=float(intercept), n_iter=int(max_iter), converged=False)


def fit_weighted_elastic_net_path(
    *,
    X: np.ndarray,
    y: np.ndarray,
    penalty_factors: np.ndarray,
    l1_ratio: float,
    lambdas: np.ndarray,
    fit_intercept: bool,
    selection: Literal["cyclic", "random"],
    max_iter: int,
    tol: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a lambda path with warm starts.

    Returns:
      coefs: (n_lambdas, p)
      intercepts: (n_lambdas,)
      n_iters: (n_lambdas,)
      converged: (n_lambdas,)
    """
    lambdas = np.asarray(lambdas, dtype=np.float64).reshape(-1)
    if lambdas.size == 0:
        raise ValueError("lambdas must be non-empty")
    # Ensure descending (from strong to weak regularization) for warm starts.
    if np.any(np.diff(lambdas) > 0):
        lambdas = np.sort(lambdas)[::-1]

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    v = np.asarray(penalty_factors, dtype=np.float64).reshape(-1)

    n, p = X.shape
    coefs = np.zeros((len(lambdas), p), dtype=np.float64)
    intercepts = np.zeros(len(lambdas), dtype=np.float64)
    n_iters = np.zeros(len(lambdas), dtype=np.int32)
    conv = np.zeros(len(lambdas), dtype=bool)

    coef_init = np.zeros(p, dtype=np.float64)
    intercept_init = float(y.mean()) if fit_intercept else 0.0

    for i, lam in enumerate(lambdas.tolist()):
        res = fit_weighted_elastic_net_cd(
            X=X,
            y=y,
            penalty_factors=v,
            l1_ratio=float(l1_ratio),
            lambda_=float(lam),
            fit_intercept=bool(fit_intercept),
            selection=selection,
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=int(random_state),
            coef_init=coef_init,
            intercept_init=intercept_init,
        )
        coefs[i, :] = res.coef
        intercepts[i] = res.intercept
        n_iters[i] = int(res.n_iter)
        conv[i] = bool(res.converged)

        coef_init = res.coef
        intercept_init = res.intercept

    return coefs, intercepts, n_iters, conv

