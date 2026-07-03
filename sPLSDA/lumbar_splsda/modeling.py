from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _one_hot(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    classes = np.asarray(classes).astype(int)
    n = int(len(y))
    k = int(len(classes))
    m = np.zeros((n, k), dtype=np.float64)
    cls_to_idx = {int(c): i for i, c in enumerate(classes.tolist())}
    for i, yi in enumerate(y.tolist()):
        j = cls_to_idx.get(int(yi))
        if j is None:
            raise ValueError(f"Unknown class in y: {yi}. Known: {classes.tolist()}")
        m[i, j] = 1.0
    return m


def _softmax_rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    a = a - np.max(a, axis=1, keepdims=True)
    e = np.exp(a)
    z = np.sum(e, axis=1, keepdims=True)
    z[z == 0] = 1.0
    return e / z


@dataclass
class SparsePLSDA:
    """
    A lightweight (sparse) PLS-DA implementation tailored for this project.

    - Multiclass is handled via dummy-encoded Y.
    - Sparsity is induced by keepX_h: keep only top-|w| features per component.
    - Projection uses R = W (P^T W + eps I)^{-1}.
    """

    n_components: int
    keepX: list[int]
    eps: float = 1e-6

    classes_: np.ndarray | None = None
    x_mean_: np.ndarray | None = None
    y_mean_: np.ndarray | None = None  # dummy-Y mean
    W_: np.ndarray | None = None
    P_: np.ndarray | None = None
    Q_: np.ndarray | None = None
    R_: np.ndarray | None = None
    B_: np.ndarray | None = None  # coefficients for dummy scores

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SparsePLSDA":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2:
            raise ValueError("X must be 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be 1D array.")
        if int(len(X)) != int(len(y)):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

        h = int(self.n_components)
        if h <= 0:
            raise ValueError("n_components must be positive.")
        if len(self.keepX) != h:
            raise ValueError(f"keepX must have length n_components ({h}). Got: {self.keepX}")

        classes = np.unique(y)
        classes = np.sort(classes)
        self.classes_ = classes
        Y = _one_hot(y, classes)

        # Center X and Y (on the training fold only).
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = Y.mean(axis=0)
        Xh = X - self.x_mean_
        Yh = Y - self.y_mean_

        n, p = Xh.shape
        k = Yh.shape[1]

        W = np.zeros((p, h), dtype=np.float64)
        P = np.zeros((p, h), dtype=np.float64)
        Q = np.zeros((k, h), dtype=np.float64)

        for comp in range(h):
            # Cross-covariance: p x k
            M = Xh.T @ Yh
            # SVD for the leading singular vectors.
            # M = U S V^T -> w = U[:,0]
            U, _, _ = np.linalg.svd(M, full_matrices=False)
            w = U[:, 0].copy()

            keep = int(self.keepX[comp])
            keep = min(max(1, keep), p)
            # Hard threshold: keep top-|w| entries.
            idx = np.argsort(np.abs(w))[::-1]
            mask = idx[:keep]
            w_sparse = np.zeros_like(w)
            w_sparse[mask] = w[mask]
            w = w_sparse

            norm = float(np.linalg.norm(w))
            if norm == 0.0:
                raise ValueError(f"Component {comp}: sparse weight vector is all zeros. keepX={keep}")
            w = w / norm

            t = Xh @ w  # n
            denom = float(t.T @ t)
            if denom == 0.0:
                raise ValueError(f"Component {comp}: zero variance score vector.")

            p_vec = (Xh.T @ t) / denom
            q_vec = (Yh.T @ t) / denom

            W[:, comp] = w
            P[:, comp] = p_vec
            Q[:, comp] = q_vec

            # Deflation
            Xh = Xh - np.outer(t, p_vec)
            Yh = Yh - np.outer(t, q_vec)

        self.W_ = W
        self.P_ = P
        self.Q_ = Q

        ptw = P.T @ W
        ptw_reg = ptw + float(self.eps) * np.eye(h, dtype=np.float64)
        self.R_ = W @ np.linalg.inv(ptw_reg)
        self.B_ = self.R_ @ Q.T
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.R_ is None or self.x_mean_ is None:
            raise RuntimeError("SparsePLSDA is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.x_mean_
        return Xc @ self.R_

    def predict_dummy_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Predict dummy-coded scores (n x K). Used by max.dist.
        """
        if self.B_ is None or self.x_mean_ is None or self.y_mean_ is None:
            raise RuntimeError("SparsePLSDA is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        Xc = X - self.x_mean_
        return self.y_mean_ + Xc @ self.B_

    def predict_max_dist(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """
        max.dist (dummy-score argmax) + softmax probabilities for continuous output.
        """
        if self.classes_ is None:
            raise RuntimeError("SparsePLSDA is not fitted.")
        scores = self.predict_dummy_scores(X)
        proba = _softmax_rows(scores)
        idx = np.argmax(scores, axis=1)
        y_pred = self.classes_[idx]
        y_cont = (proba * self.classes_.reshape(1, -1)).sum(axis=1)
        return {"y_pred": y_pred.astype(int), "y_cont": y_cont.astype(float), "proba": proba.astype(float), "dummy": scores}

