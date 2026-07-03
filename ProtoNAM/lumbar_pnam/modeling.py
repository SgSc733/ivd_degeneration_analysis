from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn

from lumbar_pnam.losses import soft_acc_loss, soft_qwk_loss
from lumbar_pnam.monotonicity import MonotonicityMixin
from lumbar_pnam.ordinal import CoralHead, make_rank_targets
from lumbar_pnam.protonam_imports import ensure_protonam_src_on_path


class ProtoNAMMultiTaskCoral(MonotonicityMixin, nn.Module):
    """ProtoNAM (shared shape functions) + multi-task (segment) weights + CORAL thresholds head.

    Key changes vs the previous implementation (per 方案1(新改).md):
    - Segment is NOT concatenated into X; it is a separate task index.
    - Each layer produces per-feature contributions; cumulative (hierarchical) shape functions
      are supervised with CORAL loss (last layer + beta * auxiliary layers).
    - CORAL head learns ordered thresholds only: p_gt = sigmoid(s - theta_k).
    - Output regularization: R_out = mean(sum_j f_j(x)^2) on the final cumulative per-feature contributions.
    """

    def __init__(
        self,
        *,
        n_feat: int,
        n_classes: int,
        n_tasks: int,
        p: int,
        h_dim: int,
        n_proto: int,
        n_layers: int,
        n_layers_pred: int,
        batch_norm: bool,
        dropout: float,
        dropout_output: float,
        beta: float,
        lambda_out: float,
        lambda_seg: float = 0.0,
        lambda_task_smooth: float = 0.0,
        tau: float,
        class_weight: list[float] | None = None,
        qwk_loss_weight: float = 0.0,
        acc_loss_weight: float = 0.1,
        share_task_weights_across_layers: bool = False,
    ):
        super().__init__()
        ensure_protonam_src_on_path()
        # Imported after sys.path is set.
        from model import ProtoNAM  # type: ignore

        self.n_classes = n_classes
        self.n_tasks = int(n_tasks)
        self.tau = float(tau)
        self.beta = float(beta)
        self.lambda_out = float(lambda_out)
        self.lambda_seg = float(lambda_seg)
        self.lambda_task_smooth = float(lambda_task_smooth)
        self.qwk_loss_weight = float(qwk_loss_weight)
        self.acc_loss_weight = float(acc_loss_weight)
        self.share_task_weights_across_layers = bool(share_task_weights_across_layers)

        if class_weight is None:
            cw = torch.ones(int(n_classes), dtype=torch.float32)
        else:
            cw = torch.tensor(class_weight, dtype=torch.float32)
            if cw.numel() != int(n_classes):
                raise ValueError(f"class_weight must have length {int(n_classes)} (got {cw.numel()})")
        # Per-class weights for ordinal loss (buffer so it follows device but is not trainable).
        self.register_buffer("class_weight", cw, persistent=False)

        p = int(p)
        if p not in {1, -1}:
            raise ValueError("This implementation currently supports p=1 (main effects) or p=-1 (full interaction) only.")

        # ProtoNAM is used as a hierarchical per-feature shape function generator.
        # We do NOT use its aggregator as the final score; segment differences are modeled via task weights.
        self.backbone = ProtoNAM(
            problem="regression",
            n_feat=n_feat,
            h_dim=h_dim,
            n_proto=n_proto,
            n_layers=n_layers,
            n_class=1,
            dropout=dropout,
            dropout_output=dropout_output,
            output_penalty=0.0,
            p=p,
            n_layers_pred=n_layers_pred,
            batch_norm=batch_norm,
        )
        self.coral = CoralHead(n_classes=n_classes)
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

        # Segment/task-specific linear weights for combining shared shape functions:
        #   s_{task,m}(x) = b_{task,m} + sum_j w_{task,j,m} * f'_{j,m}(x_j)
        n_task_layers = 1 if self.share_task_weights_across_layers else int(n_layers)
        self.task_w = nn.Parameter(torch.ones(int(n_task_layers), self.n_tasks, int(self.backbone.n_comp)))
        self.task_b = nn.Parameter(torch.zeros(int(n_task_layers), self.n_tasks))

        # Optional soft monotonicity constraints (disabled until configure_monotonicity is called).
        self._init_monotonicity()

    @property
    def n_feat(self) -> int:
        return int(self.backbone.n_feat)

    @property
    def n_layers(self) -> int:
        return int(self.backbone.n_layers)

    @property
    def n_comp(self) -> int:
        return int(self.backbone.n_comp)

    def initialize_prototypes(self, X_train: Any) -> None:
        """Quantile initialization of prototypes on the (preprocessed) training fold."""
        self.backbone.initialize(X_train)

    def _layer_feature_contrib(self, z: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Args:
            z: (batch, n_feat, h_dim)
        Returns:
            contrib: (batch, n_feat) per-feature contribution at this layer.
        """
        # (batch, n_comp, n_feat, h_dim) where:
        # - p=1  -> n_comp == n_feat (main effects)
        # - p=-1 -> n_comp == 1      (full interaction component)
        z_comp = z[:, None] * self.backbone.mask[None, :, :, None]
        res = self.backbone.clfs[layer_idx](z_comp.flatten(start_dim=-2))  # (batch, n_comp, 1)
        res = self.backbone.dropout_output(res)
        res = res.squeeze(-1)  # (batch, n_comp)
        return res

    def _apply_task_weights(self, *, f_cum: torch.Tensor, task: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Args:
            f_cum: (batch, n_feat) cumulative per-feature shape contributions
            task: (batch,) int64 in [0..n_tasks-1]
        Returns:
            s: (batch,) score for this layer and task
        """
        task = task.to(torch.int64)
        idx = 0 if self.share_task_weights_across_layers else int(layer_idx)
        w = self.task_w[idx][task]  # (batch, n_comp)
        b = self.task_b[idx][task]  # (batch,)
        return b + (w * f_cum).sum(dim=1)

    def forward(self, x: torch.Tensor, task: torch.Tensor, y: torch.Tensor, *, T: float) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: (batch, n_feat) float
            task: (batch,) int64 task index (segment), 0..n_tasks-1
            y: (batch,) int labels in {1..K}
            T: ProtoNAM temperature (controls prototype assignment sharpness)
        Returns:
            loss: scalar tensor
            stats: dict of scalar tensors for logging
        """
        y = y.to(torch.int64)
        rank_tgt = make_rank_targets(y, self.n_classes)  # (batch, K-1)

        Z = self.backbone.encode(x, T=T)  # list[(batch, n_feat, h_dim)]

        f_cum = torch.zeros((x.shape[0], self.n_comp), device=x.device, dtype=x.dtype)
        layer_losses: list[torch.Tensor] = []
        logits_last: torch.Tensor | None = None

        for m in range(len(Z)):
            f_layer = self._layer_feature_contrib(Z[m], m)
            f_cum = f_cum + f_layer

            s_m = self._apply_task_weights(f_cum=f_cum, task=task, layer_idx=m)
            logits = self.coral(s_m)  # (batch, K-1)
            if m == (len(Z) - 1):
                logits_last = logits
            # Weighted mean over samples (helps with class imbalance).
            loss_vec = self._bce(logits, rank_tgt).sum(dim=1)  # (batch,)
            w = self.class_weight[(y - 1).clamp(min=0, max=self.n_classes - 1)]  # (batch,)
            loss_m = (loss_vec * w).sum() / w.sum().clamp(min=1.0)
            layer_losses.append(loss_m)

        loss_last = layer_losses[-1]
        loss_aux = torch.stack(layer_losses[:-1]).sum() if len(layer_losses) > 1 else torch.tensor(0.0, device=x.device)
        loss_data = loss_last + self.beta * loss_aux

        # Output regularization on final cumulative per-feature contributions.
        r_out = (f_cum.pow(2).sum(dim=1)).mean()
        loss_out = self.lambda_out * r_out

        r_seg = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        loss_seg = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.lambda_seg != 0.0:
            w_bar = self.task_w.mean(dim=1, keepdim=True)
            b_bar = self.task_b.mean(dim=1, keepdim=True)
            r_seg = (self.task_w - w_bar).pow(2).mean() + (self.task_b - b_bar).pow(2).mean()
            loss_seg = self.lambda_seg * r_seg

        r_task_smooth = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        loss_task_smooth = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.lambda_task_smooth != 0.0 and self.n_tasks > 1:
            dw = (self.task_w[:, 1:, :] - self.task_w[:, :-1, :]).pow(2).mean()
            db = (self.task_b[:, 1:] - self.task_b[:, :-1]).pow(2).mean()
            r_task_smooth = dw + db
            loss_task_smooth = self.lambda_task_smooth * r_task_smooth

        loss_qwk = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        loss_acc = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.qwk_loss_weight != 0.0:
            if logits_last is None:
                raise RuntimeError("logits_last is missing (unexpected: no backbone layers).")
            out_last = self.coral.decode(logits_last, decision_threshold=0.5)
            loss_qwk = soft_qwk_loss(y_true=y, class_probs=out_last.class_probs, n_classes=self.n_classes)
        else:
            out_last = self.coral.decode(logits_last, decision_threshold=0.5) if logits_last is not None else None
        if self.acc_loss_weight != 0.0:
            if logits_last is None:
                raise RuntimeError("logits_last is missing (unexpected: no backbone layers).")
            if out_last is None:
                out_last = self.coral.decode(logits_last, decision_threshold=0.5)
            loss_acc = soft_acc_loss(y_true=y, class_probs=out_last.class_probs, n_classes=self.n_classes)

        # Soft monotonicity penalty on selected features' shape functions (单调性约束方案).
        r_mono, r_mono_w, n_mono_viol = self._compute_monotonicity_penalty(x)
        loss_mono = self.mono_lambda * (r_mono + self.mono_alpha_w * r_mono_w)

        loss = (
            loss_data
            + loss_out
            + loss_seg
            + loss_task_smooth
            + self.qwk_loss_weight * loss_qwk
            + self.acc_loss_weight * loss_acc
            + loss_mono
        )
        stats = {
            "loss": loss.detach(),
            "loss_data": loss_data.detach(),
            "loss_last": loss_last.detach(),
            "loss_aux_sum": loss_aux.detach(),
            "r_out": r_out.detach(),
            "loss_out": loss_out.detach(),
            "r_seg": r_seg.detach(),
            "loss_seg": loss_seg.detach(),
            "r_task_smooth": r_task_smooth.detach(),
            "loss_task_smooth": loss_task_smooth.detach(),
            "loss_qwk": loss_qwk.detach(),
            "loss_acc": loss_acc.detach(),
            "r_mono": r_mono.detach(),
            "r_mono_w": r_mono_w.detach(),
            "loss_mono": loss_mono.detach(),
            "mono_n_viol": torch.tensor(float(n_mono_viol), device=x.device),
        }
        return loss, stats

    @torch.no_grad()
    def predict_ordinal(
        self,
        x: torch.Tensor,
        task: torch.Tensor,
        *,
        T: float = 1e-8,
        decision_threshold: float | torch.Tensor = 0.5,
    ):
        """Predict using the last layer score."""
        Z = self.backbone.encode(x, T=T)
        f_cum = torch.zeros((x.shape[0], self.n_comp), device=x.device, dtype=x.dtype)
        s_layers: list[torch.Tensor] = []
        for m in range(len(Z)):
            f_cum = f_cum + self._layer_feature_contrib(Z[m], m)
            s_m = self._apply_task_weights(f_cum=f_cum, task=task, layer_idx=m)
            s_layers.append(s_m)
        s_last = s_layers[-1]
        logits = self.coral(s_last)
        out = self.coral.decode(logits, decision_threshold=decision_threshold)
        s_all = torch.stack(s_layers, dim=1)  # (batch, n_layers)
        return s_last, out, s_all

    @torch.no_grad()
    def feature_contributions(self, x: torch.Tensor, *, T: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            final_cum: (batch, n_comp) final cumulative contributions (n_comp depends on p)
            cum_by_layer: (batch, n_layers, n_comp) cumulative contributions per layer
        """
        Z = self.backbone.encode(x, T=T)
        layer = []
        f_cum = torch.zeros((x.shape[0], self.n_comp), device=x.device, dtype=x.dtype)
        for m in range(len(Z)):
            f_cum = f_cum + self._layer_feature_contrib(Z[m], m)
            layer.append(f_cum)
        cum_by_layer = torch.stack(layer, dim=1)  # (batch, n_layers, n_comp)
        return cum_by_layer[:, -1, :], cum_by_layer


class ProtoN2AMMultiTaskCoral(MonotonicityMixin, nn.Module):
    """
    ProtoN2AM (ProtoNA^2M): ProtoNAM main effects + a limited set of pairwise interactions (GA^2M),
    with multi-task (segment) weights and CORAL thresholds head.

    This is an opt-in extension (USE_INTERACTION=true). When disabled, the code path should use
    ProtoNAMMultiTaskCoral unchanged to preserve exact reproducibility of previous runs.
    """

    def __init__(
        self,
        *,
        n_feat: int,
        n_classes: int,
        n_tasks: int,
        pairs: list[tuple[int, int]],
        p: int,
        h_dim: int,
        n_proto: int,
        n_layers: int,
        n_layers_pred: int,
        batch_norm: bool,
        dropout: float,
        dropout_output: float,
        beta: float,
        lambda_out: float,
        lambda_seg: float = 0.0,
        lambda_task_smooth: float = 0.0,
        tau: float,
        class_weight: list[float] | None = None,
        qwk_loss_weight: float = 0.0,
        acc_loss_weight: float = 0.1,
        share_task_weights_across_layers: bool = False,
        interaction_mlp_hidden_dim: int | None = None,
        interaction_mlp_dropout: float = 0.0,
    ):
        super().__init__()
        ensure_protonam_src_on_path()
        from model import ProtoNAM  # type: ignore

        if int(p) != 1:
            raise ValueError("ProtoN2AM requires model.p=1 (main effects backbone).")

        self.n_classes = int(n_classes)
        self.n_tasks = int(n_tasks)
        self.tau = float(tau)
        self.beta = float(beta)
        self.lambda_out = float(lambda_out)
        self.lambda_seg = float(lambda_seg)
        self.lambda_task_smooth = float(lambda_task_smooth)
        self.qwk_loss_weight = float(qwk_loss_weight)
        self.acc_loss_weight = float(acc_loss_weight)
        self.share_task_weights_across_layers = bool(share_task_weights_across_layers)

        if class_weight is None:
            cw = torch.ones(int(n_classes), dtype=torch.float32)
        else:
            cw = torch.tensor(class_weight, dtype=torch.float32)
            if cw.numel() != int(n_classes):
                raise ValueError(f"class_weight must have length {int(n_classes)} (got {cw.numel()})")
        self.register_buffer("class_weight", cw, persistent=False)

        # Main-effects ProtoNAM backbone (shared per-feature hierarchical encoders).
        self.backbone = ProtoNAM(
            problem="regression",
            n_feat=int(n_feat),
            h_dim=int(h_dim),
            n_proto=int(n_proto),
            n_layers=int(n_layers),
            n_class=1,
            dropout=float(dropout),
            dropout_output=float(dropout_output),
            output_penalty=0.0,
            p=1,
            n_layers_pred=int(n_layers_pred),
            batch_norm=bool(batch_norm),
        )
        self.coral = CoralHead(n_classes=int(n_classes))
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

        # Interaction pairs are fold-specific and provided externally.
        pairs_norm = []
        seen: set[tuple[int, int]] = set()
        for a, b in pairs:
            a_i = int(a)
            b_i = int(b)
            if a_i == b_i:
                raise ValueError(f"Invalid interaction pair (a==b): {(a_i, b_i)}")
            if a_i < 0 or b_i < 0 or a_i >= int(n_feat) or b_i >= int(n_feat):
                raise ValueError(f"Interaction pair out of range for n_feat={int(n_feat)}: {(a_i, b_i)}")
            if a_i > b_i:
                a_i, b_i = b_i, a_i
            if (a_i, b_i) in seen:
                continue
            seen.add((a_i, b_i))
            pairs_norm.append((a_i, b_i))
        self.pairs = pairs_norm
        pair_idx = torch.tensor(self.pairs, dtype=torch.int64)
        self.register_buffer("pair_index", pair_idx, persistent=False)

        # Per-layer shared interaction MLP: MLP_m_int(concat(v_a, v_b)) -> scalar.
        hid = int(interaction_mlp_hidden_dim) if interaction_mlp_hidden_dim is not None else int(h_dim)
        drop_int = float(interaction_mlp_dropout)
        self.int_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * int(h_dim), hid),
                    nn.ReLU(),
                    nn.Dropout(p=drop_int),
                    nn.Linear(hid, 1),
                )
                for _ in range(int(n_layers))
            ]
        )

        # Task-specific weights for combining:
        #   s_{task,m} = b_{task,m} + sum_j w_main * f_main + sum_pairs w_int * f_int
        n_task_layers = 1 if self.share_task_weights_across_layers else int(n_layers)
        self.task_w_main = nn.Parameter(torch.ones(n_task_layers, self.n_tasks, int(self.backbone.n_feat)))
        self.task_w_int = nn.Parameter(torch.zeros(n_task_layers, self.n_tasks, int(len(self.pairs))))
        self.task_b = nn.Parameter(torch.zeros(n_task_layers, self.n_tasks))

        # Optional soft monotonicity constraints on main effects only (interactions stay free).
        self._init_monotonicity()

    def _mono_task_weight_last(self) -> torch.Tensor:
        """Main-effect last-layer task weights, shape (n_tasks, n_feat)."""
        idx = 0 if self.share_task_weights_across_layers else (self.n_layers - 1)
        return self.task_w_main[idx]

    @property
    def n_feat(self) -> int:
        return int(self.backbone.n_feat)

    @property
    def n_layers(self) -> int:
        return int(self.backbone.n_layers)

    @property
    def n_pairs(self) -> int:
        return int(len(self.pairs))

    def initialize_prototypes(self, X_train: Any) -> None:
        self.backbone.initialize(X_train)

    def _layer_feature_contrib(self, z: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # Same as ProtoNAMMultiTaskCoral for p=1 backbone.
        z_comp = z[:, None] * self.backbone.mask[None, :, :, None]
        res = self.backbone.clfs[layer_idx](z_comp.flatten(start_dim=-2))  # (batch, n_comp, 1)
        res = self.backbone.dropout_output(res)
        return res.squeeze(-1)  # (batch, n_feat)

    def _layer_interaction_contrib(self, z: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Args:
            z: (batch, n_feat, h_dim)
        Returns:
            u: (batch, n_pairs) interaction per-pair contribution at this layer.
        """
        if self.n_pairs == 0:
            return torch.zeros((z.shape[0], 0), device=z.device, dtype=z.dtype)
        a = self.pair_index[:, 0]
        b = self.pair_index[:, 1]
        za = z[:, a, :]  # (batch, n_pairs, h_dim)
        zb = z[:, b, :]  # (batch, n_pairs, h_dim)
        x_int = torch.cat([za, zb], dim=-1)  # (batch, n_pairs, 2*h_dim)
        x_int2 = x_int.reshape(-1, x_int.shape[-1])
        u = self.int_mlps[layer_idx](x_int2).reshape(z.shape[0], self.n_pairs)
        return u

    def _apply_task_weights(
        self,
        *,
        f_main_cum: torch.Tensor,
        f_int_cum: torch.Tensor,
        task: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        task = task.to(torch.int64)
        idx = 0 if self.share_task_weights_across_layers else int(layer_idx)
        w_main = self.task_w_main[idx][task]  # (batch, n_feat)
        b = self.task_b[idx][task]  # (batch,)
        s = b + (w_main * f_main_cum).sum(dim=1)
        if self.n_pairs > 0:
            w_int = self.task_w_int[idx][task]  # (batch, n_pairs)
            s = s + (w_int * f_int_cum).sum(dim=1)
        return s

    def forward(self, x: torch.Tensor, task: torch.Tensor, y: torch.Tensor, *, T: float) -> Tuple[torch.Tensor, dict]:
        y = y.to(torch.int64)
        rank_tgt = make_rank_targets(y, self.n_classes)  # (batch, K-1)

        Z = self.backbone.encode(x, T=T)  # list[(batch, n_feat, h_dim)]
        f_main_cum = torch.zeros((x.shape[0], self.n_feat), device=x.device, dtype=x.dtype)
        f_int_cum = torch.zeros((x.shape[0], self.n_pairs), device=x.device, dtype=x.dtype)

        layer_losses: list[torch.Tensor] = []
        logits_last: torch.Tensor | None = None

        for m in range(len(Z)):
            f_main_cum = f_main_cum + self._layer_feature_contrib(Z[m], m)
            if self.n_pairs > 0:
                f_int_cum = f_int_cum + self._layer_interaction_contrib(Z[m], m)

            s_m = self._apply_task_weights(
                f_main_cum=f_main_cum,
                f_int_cum=f_int_cum,
                task=task,
                layer_idx=m,
            )
            logits = self.coral(s_m)  # (batch, K-1)
            if m == (len(Z) - 1):
                logits_last = logits

            loss_vec = self._bce(logits, rank_tgt).sum(dim=1)  # (batch,)
            w = self.class_weight[(y - 1).clamp(min=0, max=self.n_classes - 1)]
            loss_m = (loss_vec * w).sum() / w.sum().clamp(min=1.0)
            layer_losses.append(loss_m)

        loss_last = layer_losses[-1]
        loss_aux = torch.stack(layer_losses[:-1]).sum() if len(layer_losses) > 1 else torch.tensor(0.0, device=x.device)
        loss_data = loss_last + self.beta * loss_aux

        r_out = (f_main_cum.pow(2).sum(dim=1) + f_int_cum.pow(2).sum(dim=1)).mean()
        loss_out = self.lambda_out * r_out

        r_seg = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        loss_seg = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.lambda_seg != 0.0:
            w_main_bar = self.task_w_main.mean(dim=1, keepdim=True)
            b_bar = self.task_b.mean(dim=1, keepdim=True)
            r_seg = (self.task_w_main - w_main_bar).pow(2).mean() + (self.task_b - b_bar).pow(2).mean()
            if self.n_pairs > 0:
                w_int_bar = self.task_w_int.mean(dim=1, keepdim=True)
                r_seg = r_seg + (self.task_w_int - w_int_bar).pow(2).mean()
            loss_seg = self.lambda_seg * r_seg

        r_task_smooth = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        loss_task_smooth = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.lambda_task_smooth != 0.0 and self.n_tasks > 1:
            dw_main = (self.task_w_main[:, 1:, :] - self.task_w_main[:, :-1, :]).pow(2).mean()
            db = (self.task_b[:, 1:] - self.task_b[:, :-1]).pow(2).mean()
            r_task_smooth = dw_main + db
            if self.n_pairs > 0:
                dw_int = (self.task_w_int[:, 1:, :] - self.task_w_int[:, :-1, :]).pow(2).mean()
                r_task_smooth = r_task_smooth + dw_int
            loss_task_smooth = self.lambda_task_smooth * r_task_smooth

        loss_qwk = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        loss_acc = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.qwk_loss_weight != 0.0:
            if logits_last is None:
                raise RuntimeError("logits_last is missing (unexpected: no backbone layers).")
            out_last = self.coral.decode(logits_last, decision_threshold=0.5)
            loss_qwk = soft_qwk_loss(y_true=y, class_probs=out_last.class_probs, n_classes=self.n_classes)
        else:
            out_last = self.coral.decode(logits_last, decision_threshold=0.5) if logits_last is not None else None
        if self.acc_loss_weight != 0.0:
            if logits_last is None:
                raise RuntimeError("logits_last is missing (unexpected: no backbone layers).")
            if out_last is None:
                out_last = self.coral.decode(logits_last, decision_threshold=0.5)
            loss_acc = soft_acc_loss(y_true=y, class_probs=out_last.class_probs, n_classes=self.n_classes)

        # Soft monotonicity penalty on main-effect shape functions (interactions stay free).
        r_mono, r_mono_w, n_mono_viol = self._compute_monotonicity_penalty(x)
        loss_mono = self.mono_lambda * (r_mono + self.mono_alpha_w * r_mono_w)

        loss = (
            loss_data
            + loss_out
            + loss_seg
            + loss_task_smooth
            + self.qwk_loss_weight * loss_qwk
            + self.acc_loss_weight * loss_acc
            + loss_mono
        )
        stats = {
            "loss": loss.detach(),
            "loss_data": loss_data.detach(),
            "loss_last": loss_last.detach(),
            "loss_aux_sum": loss_aux.detach(),
            "r_out": r_out.detach(),
            "loss_out": loss_out.detach(),
            "r_seg": r_seg.detach(),
            "loss_seg": loss_seg.detach(),
            "r_task_smooth": r_task_smooth.detach(),
            "loss_task_smooth": loss_task_smooth.detach(),
            "loss_qwk": loss_qwk.detach(),
            "loss_acc": loss_acc.detach(),
            "r_mono": r_mono.detach(),
            "r_mono_w": r_mono_w.detach(),
            "loss_mono": loss_mono.detach(),
            "mono_n_viol": torch.tensor(float(n_mono_viol), device=x.device),
        }
        return loss, stats

    @torch.no_grad()
    def predict_ordinal(
        self,
        x: torch.Tensor,
        task: torch.Tensor,
        *,
        T: float = 1e-8,
        decision_threshold: float | torch.Tensor = 0.5,
    ):
        Z = self.backbone.encode(x, T=T)
        f_main_cum = torch.zeros((x.shape[0], self.n_feat), device=x.device, dtype=x.dtype)
        f_int_cum = torch.zeros((x.shape[0], self.n_pairs), device=x.device, dtype=x.dtype)
        s_layers: list[torch.Tensor] = []
        for m in range(len(Z)):
            f_main_cum = f_main_cum + self._layer_feature_contrib(Z[m], m)
            if self.n_pairs > 0:
                f_int_cum = f_int_cum + self._layer_interaction_contrib(Z[m], m)
            s_m = self._apply_task_weights(
                f_main_cum=f_main_cum,
                f_int_cum=f_int_cum,
                task=task,
                layer_idx=m,
            )
            s_layers.append(s_m)
        s_last = s_layers[-1]
        logits = self.coral(s_last)
        out = self.coral.decode(logits, decision_threshold=decision_threshold)
        s_all = torch.stack(s_layers, dim=1)  # (batch, n_layers)
        return s_last, out, s_all

    @torch.no_grad()
    def feature_contributions(self, x: torch.Tensor, *, T: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns main-effects contributions only (compatible with existing explain pipeline):
            final_cum: (batch, n_feat)
            cum_by_layer: (batch, n_layers, n_feat)
        """
        Z = self.backbone.encode(x, T=T)
        layer = []
        f_main_cum = torch.zeros((x.shape[0], self.n_feat), device=x.device, dtype=x.dtype)
        for m in range(len(Z)):
            f_main_cum = f_main_cum + self._layer_feature_contrib(Z[m], m)
            layer.append(f_main_cum)
        cum_by_layer = torch.stack(layer, dim=1)
        return cum_by_layer[:, -1, :], cum_by_layer
