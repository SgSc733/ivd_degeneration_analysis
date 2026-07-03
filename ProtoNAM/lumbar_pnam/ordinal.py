from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_rank_targets(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """
    CORAL rank targets:
      t_k = 1 if y > k else 0, for k = 1..K-1.

    Args:
        y: shape (batch,), integer labels in {1..K}
        n_classes: K
    Returns:
        t: shape (batch, K-1) float tensor in {0,1}
    """
    if y.ndim != 1:
        raise ValueError("y must be 1-D (batch,)")
    # k = 1..K-1
    ks = torch.arange(1, n_classes, device=y.device, dtype=y.dtype)
    return (y[:, None] > ks[None, :]).to(torch.float32)


@dataclass(frozen=True)
class CoralOutputs:
    p_gt: torch.Tensor  # (batch, K-1)
    y_pred: torch.Tensor  # (batch,) in {1..K}
    y_cont: torch.Tensor  # (batch,) continuous output in [1,K]
    class_probs: torch.Tensor  # (batch, K)


class CoralHead(nn.Module):
    """
    Updated CORAL ordinal regression head (方案1(新改).md 4.3.2):

        p_gt_k = sigmoid(s - theta_k)
        theta_1 = tilde_theta_1
        theta_k = theta_{k-1} + exp(delta_k)  (k=2..K-1)

    This learns ordered thresholds only (no alpha, no per-k bias b_k).
    """

    def __init__(self, n_classes: int):
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = n_classes

        # theta_1 is free; subsequent increments are exp(delta_k) to guarantee monotonicity.
        self._theta1 = nn.Parameter(torch.tensor(0.0))
        if n_classes > 2:
            self._delta_raw = nn.Parameter(torch.zeros(n_classes - 2))
        else:
            self._delta_raw = nn.Parameter(torch.zeros(0))

    @property
    def thresholds(self) -> torch.Tensor:
        if self.n_classes == 2:
            return self._theta1[None]  # (1,)
        inc = torch.exp(self._delta_raw)  # (K-2,)
        theta = [self._theta1]
        for i in range(inc.shape[0]):
            theta.append(theta[-1] + inc[i])
        return torch.stack(theta, dim=0)  # (K-1,)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: shape (batch,)
        Returns:
            logits: shape (batch, K-1)
        """
        if s.ndim != 1:
            raise ValueError("s must be 1-D (batch,)")
        return s[:, None] - self.thresholds[None, :]

    def decode(self, logits: torch.Tensor, decision_threshold: float | torch.Tensor = 0.5) -> CoralOutputs:
        """
        Decode logits to discrete prediction and continuous output.
        """
        if logits.ndim != 2 or logits.shape[1] != self.n_classes - 1:
            raise ValueError("logits must have shape (batch, K-1)")

        p_gt = torch.sigmoid(logits)  # P(y > k)

        thr = decision_threshold
        if isinstance(thr, (int, float)):
            thr_t: float | torch.Tensor = float(thr)
        elif isinstance(thr, torch.Tensor):
            thr_t = thr
        else:
            raise TypeError("decision_threshold must be a float or torch.Tensor.")

        if isinstance(thr_t, torch.Tensor):
            # Allow:
            # - scalar tensor
            # - (K-1,) per-boundary thresholds
            # - (batch, K-1) per-sample per-boundary thresholds
            if thr_t.ndim == 0:
                pass
            elif thr_t.ndim == 1:
                if thr_t.shape[0] != self.n_classes - 1:
                    raise ValueError("decision_threshold vector must have shape (K-1,)")
            elif thr_t.ndim == 2:
                if thr_t.shape[0] != logits.shape[0] or thr_t.shape[1] != self.n_classes - 1:
                    raise ValueError("decision_threshold matrix must have shape (batch, K-1)")
            else:
                raise ValueError("decision_threshold tensor must be a scalar, (K-1,), or (batch, K-1)")
            thr_t = thr_t.to(device=logits.device, dtype=p_gt.dtype)

        # Discrete prediction (CORAL rule).
        y_pred = 1 + (p_gt > thr_t).to(torch.int64).sum(dim=1)

        # Continuous output (plan-defined):
        #   y_cont = 1 + sum_k p_gt_k
        y_cont = 1.0 + p_gt.sum(dim=1)

        # Class probs from cumulative probs (for debugging / optional downstream use).
        K = self.n_classes
        class_probs = torch.zeros((logits.shape[0], K), device=logits.device, dtype=p_gt.dtype)
        class_probs[:, 0] = 1.0 - p_gt[:, 0]
        if K > 2:
            for k in range(2, K):
                class_probs[:, k - 1] = p_gt[:, k - 2] - p_gt[:, k - 1]
        class_probs[:, K - 1] = p_gt[:, K - 2]

        return CoralOutputs(
            p_gt=p_gt,
            class_probs=class_probs,
            y_pred=y_pred,
            y_cont=y_cont,
        )
