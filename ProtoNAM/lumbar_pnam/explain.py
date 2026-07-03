from __future__ import annotations

import json
import itertools
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as patches  # noqa: E402

from lumbar_pnam.modeling import ProtoNAMMultiTaskCoral
from lumbar_pnam.modeling import ProtoN2AMMultiTaskCoral


def _density_blocks(ax, x_all: np.ndarray, *, y_min: float, y_max: float, n_blocks: int = 20) -> None:
    x_all = x_all.astype(float)
    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))
    if x_max == x_min:
        return

    x_n_blocks = int(min(n_blocks, len(np.unique(x_all))))
    hist, bin_edges = np.histogram(x_all, bins=x_n_blocks)
    hist = hist.astype(float)
    if hist.max() > 0:
        hist = hist / hist.max()

    length = (x_max - x_min) / x_n_blocks
    for j in range(x_n_blocks):
        x_start = x_min + length * j
        x_end = x_start + length
        alpha = float(min(1.0, 0.01 + hist[j]))
        rect = patches.Rectangle(
            (x_start, y_min),
            x_end - x_start,
            y_max - y_min,
            linewidth=0.01,
            edgecolor=[0.9, 0.5, 0.5],
            facecolor=[0.9, 0.5, 0.5],
            alpha=alpha,
        )
        ax.add_patch(rect)


def _plot_single_curve(
    *,
    x_unique: np.ndarray,
    y_unique: np.ndarray,
    x_all: np.ndarray,
    title: str | None,
    xlabel: str,
    out_path: Path,
    dpi: int,
) -> None:
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_unique.astype(float), y_unique.astype(float), color="blue", linewidth=3)

    # In the original ProtoNAM demo scripts the y-range is clamped to [-15, 15].
    # For small-sample medical datasets, learned contributions are often much smaller;
    # clamping makes curves look like a horizontal line. Use a robust data-driven range.
    y_lo = float(np.percentile(y_unique, 1))
    y_hi = float(np.percentile(y_unique, 99))
    if y_hi == y_lo:
        # Fallback if the function is (almost) constant.
        y_hi = y_lo + 1.0
    margin = 0.15 * (y_hi - y_lo)
    y_min = y_lo - margin
    y_max = y_hi + margin

    _density_blocks(ax, x_all, y_min=y_min, y_max=y_max)
    ax.set_xlim(float(np.min(x_all)), float(np.max(x_all)))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(xlabel, fontsize="x-large")
    ax.set_ylabel("Centered contribution", fontsize="x-large")
    if title:
        ax.set_title(title, fontsize=18)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def explain_features(
    *,
    X_raw: pd.DataFrame | None = None,
    meta: pd.DataFrame | None = None,
    tasks: np.ndarray,
    task_names: List[str],
    feature_names_to_plot: List[str] | None = None,
    fold_dir: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
) -> None:
    """
    Generate global + layer-wise shape function plots for specified features.
    """
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

    device = str(cfg["training"]["device"])
    dpi = int(cfg["explain"]["dpi"])
    seg_quantiles = list(cfg.get("explain", {}).get("segment_quantiles", [0.1, 0.3, 0.5, 0.7, 0.9]))
    seg_n_segments = int(cfg.get("explain", {}).get("segment_n_segments", 3))
    lambda_seg = float(cfg.get("explain", {}).get("lambda_seg", 1e-2))

    ckpt_path = fold_dir / "best_model.pt"
    pre_path = fold_dir / "preprocessor.pkl"
    feat_path = fold_dir / "feature_names.json"
    pair_path = fold_dir / "interaction_pairs.json"
    raw_input_path = fold_dir / "raw_input_features.pkl"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not pre_path.exists():
        raise FileNotFoundError(f"Preprocessor not found: {pre_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_names.json not found: {feat_path}")

    with open(pre_path, "rb") as f:
        pre = pickle.load(f)

    feat_meta = json.loads(feat_path.read_text(encoding="utf-8"))
    feature_names = feat_meta["feature_names"]
    if raw_input_path.exists():
        X_raw_use = pd.read_pickle(raw_input_path)
    else:
        if X_raw is None:
            raise FileNotFoundError(f"Neither raw_input_features.pkl nor X_raw was provided for fold: {fold_dir}")
        X_raw_use = X_raw.copy()
        if list(X_raw_use.columns) != list(feature_names):
            X_raw_use = X_raw_use.reindex(columns=feature_names)

    if meta is not None and hasattr(pre, "transform_with_meta") and not raw_input_path.exists():
        X_trans = pre.transform_with_meta(X_raw_use, cfg=cfg, meta=meta)  # type: ignore[attr-defined]
    else:
        X_trans = pre.transform(X_raw_use)
    x_tensor = torch.from_numpy(X_trans).float().to(device)

    share_task_weights_across_layers = bool(model_cfg.get("share_task_weights_across_layers", False))

    int_cfg = cfg.get("interaction", {}) or {}
    int_mlp_hidden_dim_raw = int_cfg.get("mlp_hidden_dim")
    int_mlp_hidden_dim = None if int_mlp_hidden_dim_raw is None else int(int_mlp_hidden_dim_raw)
    int_mlp_dropout = float(int_cfg.get("mlp_dropout", 0.0))

    interaction_pairs: list[tuple[int, int]] | None = None
    if pair_path.exists():
        pair_meta = json.loads(pair_path.read_text(encoding="utf-8"))
        pairs_rows = list(pair_meta.get("pairs", []) or [])
        interaction_pairs = [(int(r["a"]), int(r["b"])) for r in pairs_rows if "a" in r and "b" in r]

    if interaction_pairs is not None:
        model = ProtoN2AMMultiTaskCoral(
            n_feat=X_trans.shape[1],
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
            tau=tau,
            share_task_weights_across_layers=share_task_weights_across_layers,
            interaction_mlp_hidden_dim=int_mlp_hidden_dim,
            interaction_mlp_dropout=int_mlp_dropout,
        ).to(device)
    else:
        model = ProtoNAMMultiTaskCoral(
            n_feat=X_trans.shape[1],
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
            tau=tau,
            share_task_weights_across_layers=share_task_weights_across_layers,
        ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Global and per-layer contributions for all features.
    final_c, cum_by_layer = model.feature_contributions(x_tensor, T=1e-8)
    final_c = final_c.detach().cpu().numpy()
    cum_by_layer = cum_by_layer.detach().cpu().numpy()

    # Centering (per-feature) for inspectable shape functions.
    final_c = final_c - final_c.mean(axis=0, keepdims=True)
    cum_by_layer = cum_by_layer - cum_by_layer.mean(axis=0, keepdims=True)

    # Index mapping by feature name to match the training-time feature order.
    name_to_idx = {name: i for i, name in enumerate(list(feature_names))}

    shared_global_dir = out_dir / "shared" / "global"
    shared_layer_dir = out_dir / "shared" / "layerwise"
    task_global_dir = out_dir / "task_weighted" / "global"
    shared_global_dir.mkdir(parents=True, exist_ok=True)
    shared_layer_dir.mkdir(parents=True, exist_ok=True)
    task_global_dir.mkdir(parents=True, exist_ok=True)

    feats = feature_names_to_plot or list(feature_names)
    tasks = np.asarray(tasks).astype(int)
    if tasks.shape[0] != X_raw_use.shape[0]:
        raise ValueError(f"tasks length mismatch: tasks={tasks.shape[0]} vs X_raw={X_raw_use.shape[0]}")

    # Export last-layer task weights as a table for debugging.
    if hasattr(model, "task_w"):
        w_last = model.task_w.detach().cpu().numpy()[-1]  # type: ignore[attr-defined]
    else:
        w_last = model.task_w_main.detach().cpu().numpy()[-1]  # type: ignore[attr-defined]
    theta = model.coral.thresholds.detach().cpu().numpy().tolist()
    (out_dir / "coral_thresholds.json").write_text(
        json.dumps({"thresholds_theta": theta}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    w_rows: list[dict[str, Any]] = []
    for t in range(len(task_names)):
        for feat in feature_names:
            w_rows.append({"task": t, "task_name": task_names[t], "feature": feat, "w": float(w_last[t, name_to_idx[feat]])})
    pd.DataFrame(w_rows).to_csv(out_dir / "task_weights_last_layer.csv", index=False, encoding="utf-8")

    # --- Monotonicity diagnostic (单调性约束方案 §5.2): violation rate per feature x task. ---
    # Read-only on the trained model; does not affect any metric. Records how well each
    # constrained shape function follows its prescribed direction on a quantile grid.
    mono_cfg = cfg.get("monotonicity", {}) or {}
    mono_dir_by_feature = dict(mono_cfg.get("direction_by_feature", {}) or {})
    if mono_dir_by_feature:
        g = int(mono_cfg.get("grid_size", 64))
        eps = float(mono_cfg.get("eps_grid", 1e-6))
        grid_method = str(mono_cfg.get("grid_method", "quantile"))
        flat = x_tensor.detach().reshape(-1)
        if grid_method == "quantile":
            qs = torch.linspace(eps, 1.0 - eps, g, device=x_tensor.device)
            grid_global = torch.quantile(flat, qs)
        else:
            grid_global = torch.linspace(float(flat.min()), float(flat.max()), g, device=x_tensor.device)
        mono_rows: list[dict[str, Any]] = []
        for fname, d in mono_dir_by_feature.items():
            if fname not in name_to_idx or int(d) not in (1, -1):
                continue
            j = int(name_to_idx[fname])
            grid_in = torch.zeros((g, len(feature_names)), dtype=torch.float32, device=x_tensor.device)
            grid_in[:, j] = grid_global
            with torch.no_grad():
                fc, _ = model.feature_contributions(grid_in, T=1e-8)
            fj = fc[:, j].detach().cpu().numpy()
            diffs = np.diff(fj)
            dj = float(d)
            # raw f_cum_j direction
            raw_viol_rate = float(np.mean((dj * diffs) < 0.0)) if diffs.size else 0.0
            # Net signed change along increasing x, aligned with the prior direction:
            # >=0 means the overall trend matches d_j (f rises for d=+1, falls for d=-1).
            net_dir = float((fj[-1] - fj[0]) * dj) if fj.size else 0.0
            # per-task weighted g_{l,j}
            for t in range(len(task_names)):
                w_tj = float(w_last[t, j])
                gdiff = w_tj * diffs
                viol_rate = float(np.mean((dj * gdiff) < 0.0)) if diffs.size else 0.0
                mono_rows.append(
                    {
                        "feature": fname,
                        "feature_index": j,
                        "direction": int(d),
                        "task": t,
                        "task_name": task_names[t],
                        "w_last": w_tj,
                        "raw_fcum_viol_rate": raw_viol_rate,
                        "task_weighted_viol_rate": viol_rate,
                        "net_direction_signed": net_dir,
                    }
                )
        if mono_rows:
            pd.DataFrame(mono_rows).to_csv(out_dir / "monotonicity_diag.csv", index=False, encoding="utf-8")

    # If interaction model is used, also export last-layer interaction weights.
    if interaction_pairs is not None:
        w_int_last = model.task_w_int.detach().cpu().numpy()[-1]  # type: ignore[attr-defined]  # (n_tasks, n_pairs)
        int_rows: list[dict[str, Any]] = []
        for idx, (a, b) in enumerate(interaction_pairs):
            fa = feature_names[int(a)] if 0 <= int(a) < len(feature_names) else str(a)
            fb = feature_names[int(b)] if 0 <= int(b) < len(feature_names) else str(b)
            for t in range(len(task_names)):
                int_rows.append(
                    {
                        "task": t,
                        "task_name": task_names[t],
                        "pair_index": int(idx),
                        "a": int(a),
                        "b": int(b),
                        "feature_a": fa,
                        "feature_b": fb,
                        "w": float(w_int_last[t, idx]),
                    }
                )
        pd.DataFrame(int_rows).to_csv(out_dir / "task_weights_interactions_last_layer.csv", index=False, encoding="utf-8")

    # Segment-wise threshold search outputs (4.7.2): per (task, feature) best partition.
    seg_rows: list[dict[str, Any]] = []

    for feat in feats:
        if feat not in name_to_idx:
            continue
        i = name_to_idx[feat]

        x_series = pd.to_numeric(X_raw_use[feat], errors="coerce")
        if x_series.isna().all():
            continue

        # Use unique x positions for a clean curve.
        x_all = x_series.to_numpy()
        valid = ~np.isnan(x_all)
        x_all_v = x_all[valid]
        if x_all_v.size == 0:
            continue

        # np.unique returns unique values sorted; keep that order for a monotone x-axis.
        x_unique_vals, unique_pos = np.unique(x_all_v, return_index=True)
        order = np.argsort(x_unique_vals.astype(float))
        unique_pos = unique_pos[order]

        x_unique = x_all_v[unique_pos]
        y_unique = final_c[valid, i][unique_pos]

        _plot_single_curve(
            x_unique=x_unique,
            y_unique=y_unique,
            x_all=x_all_v,
            title=None,
            xlabel=feat,
            out_path=shared_global_dir / f"{feat}.png",
            dpi=dpi,
        )

        for l in range(cum_by_layer.shape[1]):
            y_unique_l = cum_by_layer[valid, l, i][unique_pos]
            _plot_single_curve(
                x_unique=x_unique,
                y_unique=y_unique_l,
                x_all=x_all_v,
                title=f"Layer {l + 1}",
                xlabel=feat,
                out_path=shared_layer_dir / f"{feat}_layer{l + 1}.png",
                dpi=dpi,
            )

        # Task-weighted contribution curves + segment-wise partition search.
        for t in range(len(task_names)):
            mask_t = (tasks == t) & valid
            if int(mask_t.sum()) == 0:
                continue

            x_t = x_all[mask_t].astype(float)
            # g_{t,j} = w_{t,j,last} * f_j(x_j)
            g_t = float(w_last[t, i]) * final_c[mask_t, i].astype(float)

            # Plot task-weighted curve.
            x_unique_vals_t, unique_pos_t = np.unique(x_t, return_index=True)
            order_t = np.argsort(x_unique_vals_t.astype(float))
            unique_pos_t = unique_pos_t[order_t]
            x_unique_t = x_t[unique_pos_t]
            g_unique_t = g_t[unique_pos_t]

            _plot_single_curve(
                x_unique=x_unique_t,
                y_unique=g_unique_t,
                x_all=x_t,
                title=f"{task_names[t]}",
                xlabel=feat,
                out_path=task_global_dir / task_names[t] / f"{feat}.png",
                dpi=dpi,
            )

            # Segment-wise partition search (default: 3 segments) on g_{t,j}(x_j).
            if seg_n_segments >= 2:
                cand = []
                for q in seg_quantiles:
                    try:
                        cand.append(float(np.quantile(x_t, q)))
                    except Exception:
                        continue
                cand = sorted(set(cand))
                if len(cand) >= seg_n_segments - 1:
                    best = None
                    for bounds in itertools.combinations(cand, seg_n_segments - 1):
                        bounds = list(bounds)
                        bounds_sorted = sorted(bounds)
                        # build segment masks: (-inf,b1], (b1,b2], ..., (b_last, inf)
                        seg_masks = []
                        lo = -float("inf")
                        for b in bounds_sorted:
                            seg_masks.append((x_t > lo) & (x_t <= b))
                            lo = b
                        seg_masks.append(x_t > lo)
                        if any(int(m.sum()) == 0 for m in seg_masks):
                            continue

                        means = np.array([float(np.mean(g_t[m])) for m in seg_masks], dtype=float)
                        vars_ = np.array([float(np.var(g_t[m])) for m in seg_masks], dtype=float)
                        diff_sum = 0.0
                        for a in range(len(means)):
                            for b in range(a + 1, len(means)):
                                diff_sum += float((means[a] - means[b]) ** 2)
                        J = diff_sum - lambda_seg * float(vars_.sum())
                        if best is None or J > best["J"]:
                            best = {
                                "bounds": bounds_sorted,
                                "J": J,
                                "means": means.tolist(),
                                "vars": vars_.tolist(),
                                "ns": [int(m.sum()) for m in seg_masks],
                            }

                    if best is not None:
                        seg_rows.append(
                            {
                                "task": t,
                                "task_name": task_names[t],
                                "feature": feat,
                                "bounds": json.dumps(best["bounds"], ensure_ascii=False),
                                "J": best["J"],
                                "ns": json.dumps(best["ns"], ensure_ascii=False),
                                "means": json.dumps(best["means"], ensure_ascii=False),
                                "vars": json.dumps(best["vars"], ensure_ascii=False),
                            }
                        )

    if seg_rows:
        pd.DataFrame(seg_rows).to_csv(out_dir / "segment_thresholds.csv", index=False, encoding="utf-8")
