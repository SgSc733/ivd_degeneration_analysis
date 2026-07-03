from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _thr_vec_from_any(v: Any, *, n_bounds: int) -> list[float]:
    if isinstance(v, (int, float)):
        return [float(v)] * int(n_bounds)
    if isinstance(v, (list, tuple, np.ndarray)):
        vv = list(v)
        if len(vv) < int(n_bounds):
            raise ValueError(f"threshold list must have length >= {int(n_bounds)} (got {len(vv)})")
        return [float(x) for x in vv[: int(n_bounds)]]
    raise TypeError("threshold must be a number or a list/tuple/ndarray.")


def parse_decision_threshold_default(*, cfg: dict[str, Any], n_classes: int) -> float | list[float]:
    """
    Match ProtoNAM config schema for ordinal.decision_threshold:

      - number: use as scalar threshold for all bounds
      - dict: {"default": number|list, "by_task": ...} (we only use default here)

    Returns a scalar or a length-(K-1) list.
    """
    ord_cfg = cfg.get("ordinal", {}) or {}
    spec = ord_cfg.get("decision_threshold", 0.5)
    default_raw = spec.get("default", 0.5) if isinstance(spec, dict) else spec

    n_bounds = int(n_classes) - 1
    vec = _thr_vec_from_any(default_raw, n_bounds=n_bounds)
    # Keep the historical behavior for scalar defaults.
    if isinstance(default_raw, (int, float)):
        return float(default_raw)
    return vec


def make_rank_targets(y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """CORAL rank targets.

    t_k = 1 if y > k else 0, for k = 1..K-1.
    y is expected to be in {1..K}.
    """
    if y.ndim != 1:
        raise ValueError("y must be 1-D (batch,)")
    ks = torch.arange(1, n_classes, device=y.device, dtype=y.dtype)
    return (y[:, None] > ks[None, :]).to(torch.float32)


@dataclass(frozen=True)
class CalibOutputs:
    p_gt: torch.Tensor  # (batch, K-1)
    y_pred: torch.Tensor  # (batch,) in {1..K}
    y_cont: torch.Tensor  # (batch,) in [1,K]
    class_probs: torch.Tensor  # (batch, K)


class CoralCalibrator(nn.Module):
    """CORAL-style calibrator with ordered thresholds and optional learned alpha.

    We follow the cumulative-logit form in `EBM.md`:

        p_gt_k = sigmoid(alpha * (s_raw - theta_k))

    where theta_k are *directly* the grade boundaries on the s_raw axis.
    """

    def __init__(self, n_classes: int, *, learn_alpha: bool = True):
        super().__init__()
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = int(n_classes)
        self.learn_alpha = bool(learn_alpha)

        self._theta1 = nn.Parameter(torch.tensor(0.0))
        if self.n_classes > 2:
            self._delta_raw = nn.Parameter(torch.zeros(self.n_classes - 2))
        else:
            self._delta_raw = nn.Parameter(torch.zeros(0))

        if self.learn_alpha:
            self._gamma = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("_gamma", torch.tensor(0.0))

    @property
    def thresholds(self) -> torch.Tensor:
        if self.n_classes == 2:
            return self._theta1[None]
        inc = torch.exp(self._delta_raw)  # (K-2,)
        theta = [self._theta1]
        for i in range(inc.shape[0]):
            theta.append(theta[-1] + inc[i])
        return torch.stack(theta, dim=0)  # (K-1,)

    @property
    def alpha(self) -> torch.Tensor:
        # alpha > 0
        return torch.exp(self._gamma)

    def forward(self, s_raw: torch.Tensor) -> torch.Tensor:
        if s_raw.ndim != 1:
            raise ValueError("s_raw must be 1-D (batch,)")
        return self.alpha * (s_raw[:, None] - self.thresholds[None, :])

    def decode(
        self,
        logits: torch.Tensor,
        *,
        decision_threshold: float = 0.5,
        decision_rule: str = "threshold",  # "threshold" | "argmax"
    ) -> CalibOutputs:
        if logits.ndim != 2 or logits.shape[1] != self.n_classes - 1:
            raise ValueError("logits must have shape (batch, K-1)")
        p_gt = torch.sigmoid(logits)
        y_cont = 1.0 + p_gt.sum(dim=1)

        K = self.n_classes
        class_probs = torch.zeros((logits.shape[0], K), device=logits.device, dtype=p_gt.dtype)
        class_probs[:, 0] = 1.0 - p_gt[:, 0]
        if K > 2:
            for k in range(2, K):
                class_probs[:, k - 1] = p_gt[:, k - 2] - p_gt[:, k - 1]
        class_probs[:, K - 1] = p_gt[:, K - 2]

        rule = str(decision_rule or "threshold").lower().strip()
        if rule in {"threshold", "thresh"}:
            # Support scalar (historical) and vector thresholds (ProtoNAM schema).
            # - scalar: apply to every bound for every sample
            # - 1-D (K-1): per-bound thresholds shared across samples
            # - 2-D (batch, K-1): per-sample per-bound thresholds (used for by_task decoding)
            if isinstance(decision_threshold, (int, float)):
                thr = float(decision_threshold)
                y_pred = 1 + (p_gt > thr).to(torch.int64).sum(dim=1)
            else:
                thr_t = torch.as_tensor(decision_threshold, device=p_gt.device, dtype=p_gt.dtype)
                if thr_t.ndim == 1:
                    if int(thr_t.shape[0]) != int(K - 1):
                        raise ValueError(
                            f"decision_threshold 1-D must have length K-1={int(K-1)} (got {tuple(thr_t.shape)})"
                        )
                    y_pred = 1 + (p_gt > thr_t[None, :]).to(torch.int64).sum(dim=1)
                elif thr_t.ndim == 2:
                    if tuple(thr_t.shape) != tuple(p_gt.shape):
                        raise ValueError(
                            f"decision_threshold 2-D must have shape {tuple(p_gt.shape)} (got {tuple(thr_t.shape)})"
                        )
                    y_pred = 1 + (p_gt > thr_t).to(torch.int64).sum(dim=1)
                else:
                    raise ValueError("decision_threshold must be a scalar, (K-1,) or (batch, K-1).")
        elif rule in {"argmax", "map"}:
            y_pred = 1 + class_probs.argmax(dim=1).to(torch.int64)
        else:
            raise ValueError(f"Unsupported decision_rule: {decision_rule!r}. Supported: 'threshold', 'argmax'.")

        return CalibOutputs(p_gt=p_gt, class_probs=class_probs, y_pred=y_pred, y_cont=y_cont)


@dataclass(frozen=True)
class CalibResult:
    calibrator: CoralCalibrator
    best_epoch: int
    best_val_loss: float
    history: list[dict[str, float]]


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _eval_loss(model: CoralCalibrator, s: torch.Tensor, y: torch.Tensor) -> float:
    logits = model(s)
    t = make_rank_targets(y, model.n_classes)
    return float(F.binary_cross_entropy_with_logits(logits, t).item())


def fit_calibrator(
    *,
    s_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> CalibResult:
    calib_cfg = cfg.get("calibration", {}) or {}
    n_classes = int(cfg["ordinal"]["n_classes"])
    decision_threshold = parse_decision_threshold_default(cfg=cfg, n_classes=n_classes)
    decision_rule = str(cfg["ordinal"].get("decision_rule", "threshold"))
    device = str(calib_cfg.get("device", "cpu"))

    learn_alpha = bool(calib_cfg.get("learn_alpha", True))
    lr = float(calib_cfg.get("lr", 1e-2))
    weight_decay = float(calib_cfg.get("weight_decay", 0.0))
    max_epoch = int(calib_cfg.get("max_epoch", 2000))
    patience = int(calib_cfg.get("patience", 200))
    min_delta = float(calib_cfg.get("min_delta", 0.0))
    val_size = calib_cfg.get("validation_size", 0.2)

    _set_seed(seed)

    s_arr = np.asarray(s_train).astype(np.float32)
    y_arr = np.asarray(y_train).astype(int)

    if y_arr.min() == 0 and y_arr.max() == n_classes - 1:
        y_arr = y_arr + 1
    if y_arr.min() < 1 or y_arr.max() > n_classes:
        raise ValueError(f"Labels must be in 1..{n_classes} (got min={y_arr.min()}, max={y_arr.max()}).")

    n = int(len(y_arr))
    idx = np.arange(n)
    np.random.shuffle(idx)
    if isinstance(val_size, float) and val_size < 1.0:
        n_val = int(math.ceil(val_size * n))
    else:
        n_val = int(val_size)
    n_val = max(0, min(n - 1, n_val))

    val_idx = idx[:n_val]
    tr_idx = idx[n_val:] if n_val > 0 else idx

    s_tr = torch.from_numpy(s_arr[tr_idx]).to(device)
    y_tr = torch.from_numpy(y_arr[tr_idx]).to(device).to(torch.int64)
    if n_val > 0:
        s_val = torch.from_numpy(s_arr[val_idx]).to(device)
        y_val = torch.from_numpy(y_arr[val_idx]).to(device).to(torch.int64)
    else:
        s_val = None
        y_val = None

    model = CoralCalibrator(n_classes=n_classes, learn_alpha=learn_alpha).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epoch + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(s_tr)
        t = make_rank_targets(y_tr, n_classes)
        loss = F.binary_cross_entropy_with_logits(logits, t)
        loss.backward()
        opt.step()

        model.eval()
        tr_loss = float(loss.detach().cpu().item())
        if s_val is not None and y_val is not None:
            val_loss = _eval_loss(model, s_val, y_val)
        else:
            val_loss = tr_loss

        history.append({"epoch": float(epoch), "train_loss": tr_loss, "val_loss": float(val_loss)})

        if val_loss + min_delta < best_val:
            best_val = float(val_loss)
            best_epoch = int(epoch)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Materialize once to ensure valid shapes.
    _ = model.decode(
        model(torch.from_numpy(s_arr[: min(8, len(s_arr))]).to(device)),
        decision_threshold=decision_threshold,
        decision_rule=decision_rule,
    )

    return CalibResult(calibrator=model, best_epoch=best_epoch, best_val_loss=best_val, history=history)
