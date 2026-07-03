"""Soft monotonicity constraints for ProtoNAM shape functions (单调性约束方案).

This module provides a mixin that adds an optional, *soft* monotonicity penalty to the
multi-task ProtoNAM/ProtoN2AM models. For a small set of clinically-motivated features
(e.g. ``classic_dhi_dhi``, ``classic_asi_asi``) we know the shape function should trend
monotonically with the Pfirrmann grade. The penalty nudges those shape functions toward
the prescribed direction without enforcing strict monotonicity, so data noise / small-sample
effects can still produce minor local non-monotonic segments.

Design notes (see ``单调性约束方案.md``):
- The penalty is evaluated on a fixed grid built from training-fold quantiles. ProtoNAM is
  additive, so ``f_cum_j(x_j)`` depends only on ``x_j``; we therefore reuse one shared global
  x-grid across all constrained features and read off each feature's column.
- ``apply_to="task_weighted"`` constrains the segment-weighted contribution
  ``g_{l,j}(x_j) = w_{l,j} * f_cum_j(x_j)`` for every segment (stricter, default);
  ``apply_to="raw_fcum"`` constrains only ``f_cum_j`` (looser).
- Only main effects are constrained; ProtoN2AM interaction terms are left free.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch


_VALID_APPLY_TO = ("raw_fcum", "task_weighted")
_VALID_MODE = ("off", "soft")


class MonotonicityMixin:
    """Adds soft monotonicity penalty hooks to a multi-task ProtoNAM model.

    The host class must provide: ``backbone`` (with ``encode``), ``_layer_feature_contrib``,
    ``n_feat``, ``n_layers``, ``share_task_weights_across_layers`` and a per-feature
    last-layer task weight tensor exposed via :meth:`_mono_task_weight_last`.
    """

    def _init_monotonicity(self) -> None:
        """Initialize monotonicity state (disabled by default). Call from ``__init__``."""
        self.mono_enabled: bool = False
        self.mono_mode: str = "off"
        self.mono_lambda: float = 0.0
        self.mono_alpha_w: float = 1.0
        self.mono_apply_to: str = "task_weighted"
        self.mono_grid_size: int = 64
        self.mono_grid_method: str = "quantile"
        self.mono_eps_grid: float = 1e-6
        self.mono_w_pos_floor: float = 0.0
        # Temperature used when evaluating the shape on the grid (match inference sharpness).
        self.mono_grid_T: float = 1e-8
        # Injected at build time (kept on the model's device).
        self.mono_indices: torch.Tensor | None = None
        self.mono_directions: torch.Tensor | None = None
        # Buffer so it follows .to(device); reassigned in build_mono_grid.
        self.register_buffer("mono_grid_x", torch.empty(0), persistent=False)

    def _mono_task_weight_last(self) -> torch.Tensor:
        """Return last-layer per-feature task weights, shape (n_tasks, n_feat).

        Subclasses with a differently named weight parameter (e.g. ProtoN2AM uses
        ``task_w_main``) should override this.
        """
        idx = 0 if self.share_task_weights_across_layers else (self.n_layers - 1)
        return self.task_w[idx]  # type: ignore[attr-defined]

    def configure_monotonicity(
        self,
        *,
        enabled: bool,
        mode: str,
        lambda_mono: float,
        alpha_w: float,
        apply_to: str,
        grid_size: int,
        grid_method: str,
        eps_grid: float,
        w_pos_floor: float,
        indices: Sequence[int],
        directions: Sequence[float],
        X_train,
    ) -> bool:
        """Inject monotonicity configuration and build the evaluation grid.

        Returns whether the constraint is effectively enabled (``False`` when disabled,
        ``mode == 'off'``, or no constrained features resolved).
        """
        mode = str(mode)
        apply_to = str(apply_to)
        if apply_to not in _VALID_APPLY_TO:
            raise ValueError(f"monotonicity.apply_to must be one of {_VALID_APPLY_TO} (got {apply_to!r})")
        if mode not in _VALID_MODE:
            raise ValueError(f"monotonicity.mode must be one of {_VALID_MODE} (got {mode!r})")

        self.mono_mode = mode
        self.mono_lambda = float(lambda_mono)
        self.mono_alpha_w = float(alpha_w)
        self.mono_apply_to = apply_to
        self.mono_grid_size = int(grid_size)
        self.mono_grid_method = str(grid_method)
        self.mono_eps_grid = float(eps_grid)
        self.mono_w_pos_floor = float(w_pos_floor)

        idx_list = [int(i) for i in indices]
        self.mono_enabled = bool(enabled) and mode == "soft" and len(idx_list) > 0
        if not self.mono_enabled:
            self.mono_indices = None
            self.mono_directions = None
            return False

        device = next(self.parameters()).device
        X_train_t = torch.as_tensor(X_train, dtype=torch.float32, device=device)
        self.build_mono_grid(X_train_t, idx_list, [float(d) for d in directions])
        return self.mono_indices is not None and self.mono_indices.numel() > 0

    def build_mono_grid(self, X_train: torch.Tensor, indices: list[int], directions: list[float]) -> None:
        """Build the (G, n_feat) monotonicity grid from training-fold quantiles.

        Only the constrained columns are populated with the shared global grid; the other
        columns are 0 (irrelevant, since ``f_cum_j`` only depends on ``x_j``).
        """
        if not indices:
            self.mono_indices = None
            self.mono_directions = None
            return

        n_feat = int(self.n_feat)
        for j in indices:
            if not (0 <= j < n_feat):
                raise ValueError(f"monotonicity feature index out of range: {j} (n_feat={n_feat})")

        device = X_train.device
        idx = torch.as_tensor(indices, dtype=torch.int64, device=device)
        d = torch.as_tensor(directions, dtype=torch.float32, device=device)
        self.mono_indices = idx
        self.mono_directions = d

        g = int(self.mono_grid_size)
        flat = X_train.detach().reshape(-1)
        if self.mono_grid_method == "quantile":
            qs = torch.linspace(self.mono_eps_grid, 1.0 - self.mono_eps_grid, g, device=device)
            grid_global = torch.quantile(flat, qs)
        else:  # linspace
            grid_global = torch.linspace(float(flat.min()), float(flat.max()), g, device=device)

        grid = torch.zeros((g, n_feat), dtype=torch.float32, device=device)
        grid[:, idx] = grid_global[:, None]
        self.mono_grid_x = grid  # (G, n_feat) registered buffer

    def _mono_grid_fcum(self) -> torch.Tensor:
        """Cumulative per-feature shape contributions on the grid, shape (G, n_feat).

        The grid encode is run with the backbone in eval mode so that BatchNorm uses its
        accumulated running statistics and, crucially, does NOT update them from the
        artificial grid distribution (which would corrupt the trained model). Dropout is
        also disabled here, giving a stable shape estimate. Gradients still flow to the
        learnable parameters, so the penalty remains trainable.
        """
        xg = self.mono_grid_x
        backbone_was_training = self.backbone.training  # type: ignore[attr-defined]
        self.backbone.eval()  # type: ignore[attr-defined]
        try:
            Zg = self.backbone.encode(xg, T=self.mono_grid_T)  # type: ignore[attr-defined]
            f_cum_g: torch.Tensor | None = None
            for m in range(len(Zg)):
                f_layer = self._layer_feature_contrib(Zg[m], m)  # type: ignore[attr-defined]
                f_cum_g = f_layer if f_cum_g is None else (f_cum_g + f_layer)
        finally:
            self.backbone.train(backbone_was_training)  # type: ignore[attr-defined]
        assert f_cum_g is not None
        return f_cum_g

    def _compute_monotonicity_penalty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compute the soft monotonicity penalty.

        Args:
            x: a batch tensor (only used for device/dtype of the zero fallbacks).
        Returns:
            (r_mono, r_mono_w, n_violations)
        """
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        if (not self.mono_enabled) or (self.mono_indices is None) or (self.mono_indices.numel() == 0):
            return zero, zero, 0

        f_cum_g = self._mono_grid_fcum()          # (G, n_feat)
        f_sub = f_cum_g[:, self.mono_indices]      # (G, M)
        d = self.mono_directions                   # (M,)
        diffs = f_sub[1:] - f_sub[:-1]             # (G-1, M)

        if self.mono_apply_to == "raw_fcum":
            # d_j * Δf_j should be >= 0.
            viol = torch.relu(-d.unsqueeze(0) * diffs)        # (G-1, M)
            r_mono = viol.sum()
        else:  # task_weighted
            w = self._mono_task_weight_last()                 # (n_tasks, n_feat)
            w_sub = w[:, self.mono_indices]                   # (n_tasks, M)
            g_diff = w_sub.unsqueeze(1) * diffs.unsqueeze(0)  # (n_tasks, G-1, M)
            viol = torch.relu(-d.view(1, 1, -1) * g_diff)     # (n_tasks, G-1, M)
            r_mono = viol.sum()

        n_viol = int((viol > 1e-6).sum().item())

        # Task-weight direction penalty: encourage sign(w_{l,j}) == d_j.
        r_mono_w = zero
        if self.mono_alpha_w > 0:
            w = self._mono_task_weight_last()                 # (n_tasks, n_feat)
            w_sub = w[:, self.mono_indices]                   # (n_tasks, M)
            d_b = d.view(1, -1)                               # (1, M)
            if self.mono_w_pos_floor > 0:
                target = self.mono_w_pos_floor * d_b          # same-sign floor
                r_mono_w = torch.relu(target - d_b * w_sub).sum()
            else:
                r_mono_w = torch.relu(-d_b * w_sub).sum()

        return r_mono, r_mono_w, n_viol
