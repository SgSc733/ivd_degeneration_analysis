from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


_INVALID_WIN_CHARS = re.compile(r'[<>:"/\\\\|?*]+')


def _safe_filename(name: str, *, max_len: int = 160) -> str:
    name = _INVALID_WIN_CHARS.sub("_", name)
    name = name.strip().strip(".")
    if len(name) > max_len:
        name = name[:max_len]
    if not name:
        return "unnamed"
    return name


def _require_xgboost():
    try:
        import xgboost as xgb  # type: ignore
    except Exception as e:  # pragma: no cover
        msg = (
            "Missing dependency: xgboost.\n\n"
            "Install it into your conda env (recommended):\n"
            "  conda install -n pnam -c conda-forge xgboost\n\n"
            "Or via pip:\n"
            "  pip install xgboost\n\n"
            f"Original error: {type(e).__name__}: {e}"
        )
        raise RuntimeError(msg) from e
    return xgb


def _bin_summary_quantile(x: np.ndarray, y: np.ndarray, *, bins: int) -> pd.DataFrame:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size == 0:
        return pd.DataFrame(columns=["bin", "x_median", "y_mean", "n"])

    uniq = np.unique(x)
    if uniq.size <= 1:
        return pd.DataFrame(
            [
                {
                    "bin": 0,
                    "x_median": float(np.median(x)),
                    "y_mean": float(np.mean(y)),
                    "n": int(x.size),
                }
            ]
        )

    bins = int(max(2, bins))
    q = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(x, q)
    edges = np.unique(edges)
    if edges.size <= 2:
        # Fallback to equal-width if quantiles collapse.
        edges = np.linspace(float(np.min(x)), float(np.max(x)), min(bins, uniq.size) + 1)

    n_bins = int(edges.size - 1)
    # digitize by interior edges
    bin_idx = np.digitize(x, edges[1:-1], right=True)

    rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        mb = bin_idx == b
        if not bool(np.any(mb)):
            continue
        rows.append(
            {
                "bin": int(b),
                "x_median": float(np.median(x[mb])),
                "y_mean": float(np.mean(y[mb])),
                "n": int(mb.sum()),
            }
        )
    return pd.DataFrame(rows)


def _moving_average(v: np.ndarray, win: int = 3) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.size < win or win <= 1:
        return v
    pad = win // 2
    vv = np.pad(v, (pad, pad), mode="edge")
    k = np.ones(win, dtype=float) / float(win)
    return np.convolve(vv, k, mode="valid")


def _plot_shape_function(
    *,
    x: np.ndarray,
    phi: np.ndarray,
    feature_name: str,
    out_png: Path,
    out_csv: Path,
    dpi: int,
    bins: int,
    smooth: bool,
) -> None:
    df_bins = _bin_summary_quantile(x, phi, bins=bins)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_bins.to_csv(out_csv, index=False, encoding="utf-8")

    plt.close()
    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.asarray(x, dtype=float).reshape(-1)
    phi = np.asarray(phi, dtype=float).reshape(-1)
    m = np.isfinite(x) & np.isfinite(phi)

    ax.scatter(x[m], phi[m], s=10, alpha=0.25, linewidths=0, color="#1f77b4")
    if not df_bins.empty:
        xs = df_bins["x_median"].to_numpy(dtype=float)
        ys = df_bins["y_mean"].to_numpy(dtype=float)
        if smooth:
            ys = _moving_average(ys, win=3)
        ax.plot(xs, ys, color="#d62728", linewidth=2.0, marker="o", markersize=3)

    ax.set_title(f"Shape-like curve (TreeSHAP): {feature_name}", fontsize=12)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Contribution to latent score s (phi)")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def explain_features(
    *,
    X_raw: pd.DataFrame,
    tasks: np.ndarray,
    task_names: List[str],
    meta: pd.DataFrame,
    id_col: str,
    feature_names_to_plot: List[str] | None = None,
    fold_dir: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
) -> None:
    xgb = _require_xgboost()

    explain_cfg = cfg.get("explain", {}) or {}
    dpi = int(explain_cfg.get("dpi", 300))
    top_n = int(explain_cfg.get("top_n", 20))
    bins = int(explain_cfg.get("bins", 30))
    smooth = bool(explain_cfg.get("smooth", False))
    by_task = bool(explain_cfg.get("by_task", False))
    x_axis = str(explain_cfg.get("x_axis", "raw"))

    feat_path = fold_dir / "feature_names.json"
    model_path = fold_dir / "xgb_model.json"
    info_path = fold_dir / "fold_info.json"
    pre_path = fold_dir / "preprocessor.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {model_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"feature_names.json not found: {feat_path}")

    feat_meta = json.loads(feat_path.read_text(encoding="utf-8"))
    feature_names = list(feat_meta["feature_names"])
    cont_feature_names = list(feat_meta.get("cont_feature_names") or [])
    segment_feature_names = list(feat_meta.get("segment_feature_names") or [])
    if not cont_feature_names and not segment_feature_names:
        # Backward compatibility: older runs only stored "feature_names".
        cont_feature_names = feature_names
        segment_feature_names = []

    best_iter = None
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if "xgb_selected_iteration" in info:
            best_iter = int(info.get("xgb_selected_iteration"))
        elif "xgb_best_iteration_reg" in info:
            best_iter = int(info.get("xgb_best_iteration_reg"))
        elif "xgb_best_iteration" in info:
            best_iter = int(info.get("xgb_best_iteration"))

    pre = None
    preprocess_enabled = bool(feat_meta.get("preprocess_enabled", False))
    if preprocess_enabled and pre_path.exists():
        with open(pre_path, "rb") as f:
            pre = pickle.load(f)

    # Align X_raw to training-time feature order. For runs with segment one-hot,
    # we apply preprocessing only to continuous features and passthrough segment {0,1}.
    if list(X_raw.columns) != feature_names:
        X_raw = X_raw.reindex(columns=feature_names)

    X_cont = X_raw[cont_feature_names] if cont_feature_names else pd.DataFrame(index=X_raw.index)
    X_seg = X_raw[segment_feature_names] if segment_feature_names else pd.DataFrame(index=X_raw.index)

    if pre is not None and cont_feature_names:
        X_cont_model = pre.transform(X_cont)
    elif cont_feature_names:
        X_cont_model = X_cont.to_numpy(dtype=np.float32, copy=True)
    else:
        X_cont_model = np.zeros((len(X_raw), 0), dtype=np.float32)

    if segment_feature_names:
        X_seg_model = X_seg.to_numpy(dtype=np.float32, copy=True)
    else:
        X_seg_model = np.zeros((len(X_raw), 0), dtype=np.float32)

    X_model = np.concatenate([X_cont_model, X_seg_model], axis=1)

    booster = xgb.Booster()
    booster.load_model(model_path)

    # Training-time order is cont + segment (see lumbar_xgb.train).
    feature_names = cont_feature_names + segment_feature_names
    dmat = xgb.DMatrix(X_model, feature_names=feature_names)
    iter_range = None
    if best_iter is not None:
        iter_range = (0, best_iter + 1)

    s_pred = booster.predict(dmat, iteration_range=iter_range)
    contribs = booster.predict(dmat, pred_contribs=True, iteration_range=iter_range)
    # contribs shape: (n, n_features + 1), last column is bias.
    if contribs.shape[1] != len(feature_names) + 1:
        raise ValueError("Unexpected pred_contribs shape. Check xgboost version / model.")

    phi = contribs[:, :-1]
    bias = contribs[:, -1]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-row contributions for traceability/debugging.
    shap_df = meta.copy()
    shap_df["s_pred"] = s_pred.astype(float)
    shap_df["phi_bias"] = bias.astype(float)
    for j, name in enumerate(feature_names):
        shap_df[f"phi_{name}"] = phi[:, j].astype(float)
    shap_df.to_csv(out_dir / "shap_contribs.csv", index=False, encoding="utf-8")

    # Global importance: mean(|phi_j|) over all samples.
    imp = np.mean(np.abs(phi), axis=0)
    imp_df = pd.DataFrame({"feature": feature_names, "mean_abs_phi": imp.astype(float)})
    imp_df = imp_df.sort_values("mean_abs_phi", ascending=False).reset_index(drop=True)
    imp_df.to_csv(out_dir / "global_importance.csv", index=False, encoding="utf-8")

    # Global importance plot.
    top_plot = imp_df.head(top_n) if top_n > 0 else imp_df
    plt.close()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(top_plot))))
    ax.barh(top_plot["feature"][::-1], top_plot["mean_abs_phi"][::-1], color="#2ca02c", alpha=0.8)
    ax.set_xlabel("mean(|phi|) on latent score s")
    ax.set_title("Global feature importance (TreeSHAP)")
    fig.savefig(out_dir / "global_importance.png", dpi=dpi)
    plt.close(fig)

    # Shape-like plots for each feature.
    feats = feature_names_to_plot or feature_names
    shape_dir = out_dir / "shape_functions"
    shape_data_dir = out_dir / "shape_functions_data"
    shape_dir.mkdir(parents=True, exist_ok=True)
    shape_data_dir.mkdir(parents=True, exist_ok=True)

    # Optional x-axis: raw or model-input (preprocessed).
    if x_axis not in {"raw", "preprocessed"}:
        raise ValueError("explain.x_axis must be 'raw' or 'preprocessed'")

    if x_axis == "raw":
        X_cont_axis = X_cont.to_numpy(dtype=float, copy=True) if cont_feature_names else np.zeros((len(X_raw), 0))
    else:
        X_cont_axis = X_cont_model.astype(float, copy=False)
    X_seg_axis = X_seg.to_numpy(dtype=float, copy=True) if segment_feature_names else np.zeros((len(X_raw), 0))
    X_axis = np.concatenate([X_cont_axis, X_seg_axis], axis=1)

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    for feat in feats:
        if feat not in name_to_idx:
            continue
        j = name_to_idx[feat]
        safe = _safe_filename(feat)
        _plot_shape_function(
            x=X_axis[:, j],
            phi=phi[:, j],
            feature_name=feat,
            out_png=shape_dir / f"{safe}.png",
            out_csv=shape_data_dir / f"{safe}.csv",
            dpi=dpi,
            bins=bins,
            smooth=smooth,
        )

    # Optional: per-task shape-like plots (disc level subsets).
    if by_task:
        tasks = np.asarray(tasks).astype(int)
        if tasks.shape[0] != X_model.shape[0]:
            raise ValueError("tasks length mismatch for explanation.")

        by_task_dir = out_dir / "shape_functions_by_task"
        for t, t_name in enumerate(task_names):
            m = tasks == t
            if not bool(np.any(m)):
                continue
            td = by_task_dir / _safe_filename(t_name)
            td_data = out_dir / "shape_functions_by_task_data" / _safe_filename(t_name)
            td.mkdir(parents=True, exist_ok=True)
            td_data.mkdir(parents=True, exist_ok=True)
            for feat in feats:
                if feat not in name_to_idx:
                    continue
                j = name_to_idx[feat]
                safe = _safe_filename(feat)
                _plot_shape_function(
                    x=X_axis[m, j],
                    phi=phi[m, j],
                    feature_name=f"{feat} ({t_name})",
                    out_png=td / f"{safe}.png",
                    out_csv=td_data / f"{safe}.csv",
                    dpi=dpi,
                    bins=bins,
                    smooth=smooth,
                )

    # A small metadata snapshot for the explanation output.
    (out_dir / "explain_meta.json").write_text(
        json.dumps(
            {
                "fold_dir": str(fold_dir),
                "model_path": str(model_path),
                "preprocess_enabled": bool(preprocess_enabled),
                "x_axis": x_axis,
                "best_iteration": best_iter,
                "n_rows": int(X_model.shape[0]),
                "n_features": int(len(feature_names)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
