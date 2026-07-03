from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def export_feature_importance(ebm, *, out_path: Path) -> pd.DataFrame:
    """Export global term importance from InterpretML EBM."""
    glob = ebm.explain_global()
    term_names = list(glob.data()["names"])
    scores = list(glob.data()["scores"])
    df = pd.DataFrame({"term": term_names, "importance": scores})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return df


def build_g_contrib_df(
    ebm,
    *,
    contrib_df: pd.DataFrame,
    feature_names_num: list[str],
    disc_level_feature_name: str = "disc_level",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build g_{ell,j} contribution table.

    For each numeric feature j, we start from its main effect contribution f_j(x_j).
    If the fitted EBM contains a segment interaction term (disc_level, j), we add
    the corresponding interaction contribution f_{j,ell}(x_j, ell).

    Returns:
        g_df: DataFrame with columns = feature_names_num, values = g_{ell,j} per row
        mapping: {feature_name -> interaction_term_name} for features that got an added term
    """
    feat_names = list(getattr(ebm, "feature_names_in_", []))
    if disc_level_feature_name not in feat_names:
        raise ValueError(
            f"disc_level_feature_name '{disc_level_feature_name}' not found in ebm.feature_names_in_: {feat_names}"
        )
    disc_idx = feat_names.index(disc_level_feature_name)

    term_features = list(getattr(ebm, "term_features_", []))
    term_names = list(getattr(ebm, "term_names_", []))
    if len(term_features) != len(term_names):
        raise ValueError("Unexpected EBM term metadata: term_features_ and term_names_ length mismatch.")

    mapping: dict[str, str] = {}
    for feats, tname in zip(term_features, term_names):
        if not isinstance(feats, tuple) or len(feats) != 2:
            continue
        if disc_idx not in feats:
            continue
        other_idx = feats[0] if feats[1] == disc_idx else feats[1]
        other_name = feat_names[int(other_idx)]
        if other_name in feature_names_num:
            mapping[other_name] = str(tname)

    n = int(len(contrib_df))
    zeros = pd.Series(np.zeros(n, dtype=float), index=contrib_df.index)

    g_cols: dict[str, pd.Series] = {}
    for feat in feature_names_num:
        base = contrib_df[feat].astype(float) if feat in contrib_df.columns else zeros
        inter_term = mapping.get(feat)
        if inter_term and inter_term in contrib_df.columns:
            g_cols[feat] = base + contrib_df[inter_term].astype(float)
        else:
            g_cols[feat] = base

    g_df = pd.DataFrame(g_cols, index=contrib_df.index)

    return g_df, mapping


def _plot_continuous_term(*, x: np.ndarray, y: np.ndarray, title: str, out_path: Path, dpi: int) -> None:
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x.astype(float), y.astype(float), color="blue", linewidth=3)
    ax.set_xlabel(title, fontsize="x-large")
    ax.set_ylabel("Term contribution", fontsize="x-large")
    ax.set_title(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _plot_categorical_term(*, names: list[str], scores: np.ndarray, title: str, out_path: Path, dpi: int) -> None:
    plt.close()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(range(len(names)), scores.astype(float))
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel(title, fontsize="x-large")
    ax.set_ylabel("Term contribution", fontsize="x-large")
    ax.set_title(title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def explain_global_terms(
    ebm,
    *,
    out_dir: Path,
    dpi: int,
) -> None:
    """Save global term shape functions (main effects + interactions if any)."""
    glob = ebm.explain_global()
    term_names = list(glob.data()["names"])

    # Determine feature types from the fitted model if available.
    feature_type_map: dict[str, str] = {}
    if hasattr(ebm, "feature_names_in_") and hasattr(ebm, "feature_types_in_"):
        feature_type_map = {n: t for n, t in zip(ebm.feature_names_in_, ebm.feature_types_in_)}

    main_dir = out_dir / "term_main"
    inter_dir = out_dir / "term_interactions"

    for i, term in enumerate(term_names):
        d = glob.data(i)
        term_type = str(d.get("type", ""))

        # Skip non-term items defensively.
        if "scores" not in d or "names" not in d:
            continue

        scores = np.asarray(d["scores"])
        names = d["names"]

        if term_type == "interaction":
            # names is a pair of axes; scores is 2D.
            try:
                xnames = list(names[0])
                ynames = list(names[1])
                z = np.asarray(scores)
            except Exception:
                continue

            plt.close()
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(z.astype(float), aspect="auto", origin="lower")
            ax.set_title(term, fontsize=14)
            ax.set_xticks(range(len(xnames)))
            ax.set_xticklabels([str(x) for x in xnames], rotation=45, ha="right")
            ax.set_yticks(range(len(ynames)))
            ax.set_yticklabels([str(y) for y in ynames])
            fig.colorbar(im, ax=ax, shrink=0.8, label="Term contribution")
            out_path = inter_dir / f"{term}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
            plt.close(fig)
            continue

        # Univariate term.
        ftype = feature_type_map.get(term, "continuous")
        if ftype in ("nominal", "ordinal"):
            cat_names = [str(x) for x in names]
            _plot_categorical_term(
                names=cat_names,
                scores=scores,
                title=term,
                out_path=main_dir / f"{term}.png",
                dpi=dpi,
            )
        else:
            # Interpret returns numeric bin labels for continuous.
            try:
                x = np.asarray(names, dtype=float)
            except Exception:
                # Fallback: treat as categorical if conversion fails.
                cat_names = [str(x) for x in names]
                _plot_categorical_term(
                    names=cat_names,
                    scores=scores,
                    title=term,
                    out_path=main_dir / f"{term}.png",
                    dpi=dpi,
                )
                continue
            y = scores.astype(float).reshape(-1)
            if x.shape[0] == y.shape[0] + 1:
                # InterpretML continuous term names are often bin edges; convert to bin centers.
                x = (x[:-1] + x[1:]) / 2.0
            if x.shape[0] != y.shape[0]:
                m = int(min(x.shape[0], y.shape[0]))
                x = x[:m]
                y = y[:m]

            _plot_continuous_term(
                x=x,
                y=y,
                title=term,
                out_path=main_dir / f"{term}.png",
                dpi=dpi,
            )


def _local_contrib_matrix(local_exp) -> tuple[list[str], np.ndarray]:
    # Interpret's local explanation has per-sample 'scores' list in term order.
    internal = getattr(local_exp, "_internal_obj", {}) or {}
    specific = internal.get("specific")
    if not isinstance(specific, list) or len(specific) == 0:
        raise ValueError("Unexpected interpret local explanation format: missing internal_obj['specific'].")
    names = list(specific[0]["names"])
    mat = np.asarray([row["scores"] for row in specific], dtype=float)
    return names, mat


def export_local_contributions(
    ebm,
    *,
    X_ebm: pd.DataFrame,
    out_path: Path,
    include_intercept: bool = True,
) -> pd.DataFrame:
    """Export per-sample term contributions using explain_local."""
    df = local_contributions_df(ebm, X_ebm=X_ebm, include_intercept=include_intercept)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return df


def local_contributions_df(
    ebm,
    *,
    X_ebm: pd.DataFrame,
    include_intercept: bool = False,
) -> pd.DataFrame:
    loc = ebm.explain_local(X_ebm)
    term_names, mat = _local_contrib_matrix(loc)
    df = pd.DataFrame(mat, columns=term_names)
    df.insert(0, "pred", ebm.predict(X_ebm))
    if include_intercept and hasattr(ebm, "intercept_"):
        df.insert(0, "intercept", float(ebm.intercept_))
    return df


def segment_threshold_search(
    *,
    X_num: pd.DataFrame,
    tasks: np.ndarray,
    task_names: list[str],
    contrib_df: pd.DataFrame,
    feature_names_num: list[str],
    out_path: Path,
    segment_quantiles: list[float],
    segment_n_segments: int,
    lambda_seg: float,
) -> pd.DataFrame:
    """Search piecewise thresholds per segment based on per-sample term contribution."""
    if segment_n_segments < 2:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    tasks = np.asarray(tasks).astype(int)

    for feat in feature_names_num:
        if feat not in contrib_df.columns:
            continue

        x_all = X_num[feat].to_numpy(dtype=float)
        g_all = contrib_df[feat].to_numpy(dtype=float)

        for t, tname in enumerate(task_names):
            mask = tasks == t
            if int(mask.sum()) == 0:
                continue

            x_t = x_all[mask]
            g_t = g_all[mask]
            if np.unique(x_t).size < segment_n_segments:
                continue

            cand = []
            for q in segment_quantiles:
                try:
                    cand.append(float(np.quantile(x_t, q)))
                except Exception:
                    continue
            cand = sorted(set(cand))
            if len(cand) < segment_n_segments - 1:
                continue

            best = None
            for bounds in itertools.combinations(cand, segment_n_segments - 1):
                bounds_sorted = sorted(list(bounds))
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
                # Document (`EBM.md`) uses sum_{r!=r'} which is 2x the unordered-pair sum.
                diff_sum *= 2.0
                J = diff_sum - lambda_seg * float(vars_.sum())
                if best is None or J > best["J"]:
                    best = {
                        "bounds": bounds_sorted,
                        "J": float(J),
                        "means": means.tolist(),
                        "vars": vars_.tolist(),
                        "ns": [int(m.sum()) for m in seg_masks],
                    }

            if best is not None:
                rows.append(
                    {
                        "task_idx0": int(t),
                        "task_idx1": int(t) + 1,
                        "task_name": tname,
                        "feature": feat,
                        "bounds": json.dumps(best["bounds"], ensure_ascii=False),
                        "J": best["J"],
                        "ns": json.dumps(best["ns"], ensure_ascii=False),
                        "means": json.dumps(best["means"], ensure_ascii=False),
                        "vars": json.dumps(best["vars"], ensure_ascii=False),
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8")
    return df
