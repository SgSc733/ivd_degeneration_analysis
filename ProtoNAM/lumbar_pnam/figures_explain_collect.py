from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from lumbar_pnam.figures_explain_types import FoldContrib
from lumbar_pnam.modeling import ProtoNAMMultiTaskCoral, ProtoN2AMMultiTaskCoral


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_interaction_pairs(fold_dir: Path) -> list[tuple[int, int]] | None:
    pair_path = fold_dir / "interaction_pairs.json"
    if not pair_path.exists():
        return None
    meta = _load_json(pair_path)
    rows = list(meta.get("pairs", []) or [])
    pairs = []
    for r in rows:
        if "a" in r and "b" in r:
            pairs.append((int(r["a"]), int(r["b"])))
    return pairs if pairs else None


def _build_model(
    *,
    n_feat: int,
    task_names: list[str],
    cfg: dict[str, Any],
    interaction_pairs: list[tuple[int, int]] | None,
) -> torch.nn.Module:
    n_classes = int(cfg["ordinal"]["n_classes"])
    beta = float(cfg["ordinal"].get("beta", 0.5))
    reg_cfg = cfg.get("regularization", {}) or {}
    lambda_out = float(reg_cfg.get("lambda_out", 1e-3))
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
    int_mlp_hidden_dim_raw = int_cfg.get("mlp_hidden_dim")
    int_mlp_hidden_dim = None if int_mlp_hidden_dim_raw is None else int(int_mlp_hidden_dim_raw)
    int_mlp_dropout = float(int_cfg.get("mlp_dropout", 0.0))

    if interaction_pairs is not None:
        return ProtoN2AMMultiTaskCoral(
            n_feat=int(n_feat),
            n_classes=int(n_classes),
            n_tasks=int(len(task_names)),
            pairs=interaction_pairs,
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
            tau=float(tau),
            share_task_weights_across_layers=bool(share_task_weights_across_layers),
            interaction_mlp_hidden_dim=int_mlp_hidden_dim,
            interaction_mlp_dropout=float(int_mlp_dropout),
        )

    return ProtoNAMMultiTaskCoral(
        n_feat=int(n_feat),
        n_classes=int(n_classes),
        n_tasks=int(len(task_names)),
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
        tau=float(tau),
        share_task_weights_across_layers=bool(share_task_weights_across_layers),
    )


def collect_fold_contributions(
    *,
    run_dir: str | Path,
    X_raw: pd.DataFrame | None,
    meta: pd.DataFrame | None,
    task_names: list[str],
    cfg: dict[str, Any],
) -> list[FoldContrib]:
    """Compute per-fold per-feature contributions (shared shape) for the full dataset."""
    run_dir = Path(run_dir)
    device = torch.device(str(cfg["training"]["device"]))

    fold_dirs = sorted([p for p in (run_dir / "checkpoints").glob("fold_*") if p.is_dir()])
    out: list[FoldContrib] = []
    for fold_dir in fold_dirs:
        try:
            fold = int(fold_dir.name.split("_")[-1])
        except Exception:
            continue

        ckpt_path = fold_dir / "best_model.pt"
        pre_path = fold_dir / "preprocessor.pkl"
        feat_path = fold_dir / "feature_names.json"
        raw_input_path = fold_dir / "raw_input_features.pkl"
        if not ckpt_path.exists() or not pre_path.exists() or not feat_path.exists():
            continue

        feat_meta = _load_json(feat_path)
        feature_names = list(feat_meta.get("feature_names") or [])
        if not feature_names:
            continue

        with open(pre_path, "rb") as f:
            pre = pickle.load(f)

        if raw_input_path.exists():
            X_raw_fold = pd.read_pickle(raw_input_path)
        else:
            if X_raw is None:
                raise FileNotFoundError(f"Missing X_raw and fold raw_input_features.pkl: {fold_dir}")
            X_raw_fold = X_raw.copy()
            if list(X_raw_fold.columns) != feature_names:
                X_raw_fold = X_raw_fold.reindex(columns=feature_names)

        if meta is not None and hasattr(pre, "transform_with_meta") and not raw_input_path.exists():
            X_trans = pre.transform_with_meta(X_raw_fold, cfg=cfg, meta=meta)  # type: ignore[attr-defined]
        else:
            X_trans = pre.transform(X_raw_fold)
        x_tensor = torch.from_numpy(np.asarray(X_trans)).float().to(device)

        interaction_pairs = _load_interaction_pairs(fold_dir)
        model = _build_model(
            n_feat=int(x_tensor.shape[1]),
            task_names=task_names,
            cfg=cfg,
            interaction_pairs=interaction_pairs,
        ).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        with torch.no_grad():
            f_raw_t, _ = model.feature_contributions(x_tensor, T=1e-8)  # type: ignore[attr-defined]
        f_raw = f_raw_t.detach().cpu().numpy().astype(float)

        # Last-layer task weights/bias for main effects.
        if hasattr(model, "task_w"):
            w_last = model.task_w.detach().cpu().numpy()[-1]  # type: ignore[attr-defined]
            b_last = model.task_b.detach().cpu().numpy()[-1]  # type: ignore[attr-defined]
        else:
            w_last = model.task_w_main.detach().cpu().numpy()[-1]  # type: ignore[attr-defined]
            b_last = model.task_b.detach().cpu().numpy()[-1]  # type: ignore[attr-defined]

        if f_raw.shape[1] != len(feature_names):
            raise ValueError(f"Unexpected f_raw shape: {f_raw.shape} vs n_features={len(feature_names)}")
        if w_last.shape[1] != len(feature_names):
            raise ValueError(f"Unexpected w_last shape: {w_last.shape} vs n_features={len(feature_names)}")

        f_mean = f_raw.mean(axis=0, keepdims=False)
        f_centered = f_raw - f_mean[None, :]

        out.append(
            FoldContrib(
                fold=int(fold),
                feature_names=feature_names,
                task_names=list(task_names),
                X_raw=X_raw_fold,
                f_raw=f_raw,
                f_mean=f_mean,
                f_centered=f_centered,
                w_last=w_last.astype(float),
                b_last=b_last.astype(float),
            )
        )

    if not out:
        raise RuntimeError(f"No fold contributions computed under: {run_dir / 'checkpoints'}")
    return sorted(out, key=lambda x: x.fold)
