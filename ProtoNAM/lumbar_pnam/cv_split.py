from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _norm_id(v: object) -> str:
    s = str(v).strip()
    # Handle common "Excel numeric" artifacts like "1.0" -> "1".
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def _read_statistics_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"statistics.csv not found: {path}")
    return pd.read_csv(path, encoding="utf-8")


def _infer_scanner_cfg(cfg: dict[str, Any], balance_cfg: dict[str, Any]) -> tuple[str, str, str]:
    """
    Returns (statistics_csv_path, case_id_col, scanner_col).

    Priority:
      1) cv.balance_pfirrmann_scanner.scanner.{...} overrides
      2) harmonization.{statistics_csv_path, case_id_col, batch_col}
      3) defaults: statistics.csv / image_id / batch_scanner
    """

    sc = (balance_cfg.get("scanner", {}) or {}) if isinstance(balance_cfg, dict) else {}
    stats_path = sc.get("statistics_csv_path", None)
    case_id_col = sc.get("case_id_col", None)
    scanner_col = sc.get("scanner_col", None)

    if stats_path or case_id_col or scanner_col:
        return (
            str(stats_path or "statistics.csv"),
            str(case_id_col or "image_id"),
            str(scanner_col or "batch_scanner"),
        )

    harm = cfg.get("harmonization", {}) or {}
    return (
        str(harm.get("statistics_csv_path", "statistics.csv")),
        str(harm.get("case_id_col", "image_id")),
        str(harm.get("batch_col", "batch_scanner")),
    )


def load_scanner_for_samples(
    *,
    cfg: dict[str, Any],
    meta: pd.DataFrame,
    balance_cfg: dict[str, Any],
) -> pd.Series:
    """
    Load scanner/batch label for each sample (disc), aligned to meta.index.

    Notes:
    - Scanner is case-level in statistics.csv. We map by meta["patient_id"].
    - Missing mappings are treated as an error (fail-fast for split reproducibility).
    """

    if "patient_id" not in meta.columns:
        raise ValueError("meta missing required column: 'patient_id'")

    stats_csv_path, case_id_col, scanner_col = _infer_scanner_cfg(cfg, balance_cfg)
    stats_df = _read_statistics_csv(stats_csv_path)
    if case_id_col not in stats_df.columns:
        raise ValueError(f"statistics.csv missing case_id_col={case_id_col!r}")
    if scanner_col not in stats_df.columns:
        raise ValueError(f"statistics.csv missing scanner_col={scanner_col!r}")

    df = stats_df[[case_id_col, scanner_col]].copy()
    df[case_id_col] = df[case_id_col].map(_norm_id)
    df[scanner_col] = df[scanner_col].astype(str).str.strip()
    df = df.dropna(subset=[case_id_col])
    df = df.drop_duplicates(subset=[case_id_col], keep="first")

    case_to_scanner = pd.Series(df[scanner_col].to_numpy(), index=df[case_id_col].to_numpy(), dtype=object)
    case_ids = meta["patient_id"].map(_norm_id)
    scanner = case_ids.map(case_to_scanner)
    if scanner.isna().any():
        missing = sorted({str(x) for x in case_ids[scanner.isna()].tolist()})[:10]
        raise ValueError(f"statistics.csv does not cover all cases. Missing examples: {missing}")
    return pd.Series(scanner.to_numpy(), index=meta.index, name="scanner", dtype=object)


def _safe_soft_target(total: np.ndarray, n_splits: int) -> np.ndarray:
    total = np.asarray(total).astype(np.float64)
    return total / float(n_splits)


def _rel_sq_err(counts: np.ndarray, target: np.ndarray) -> float:
    """
    Squared relative error: sum(((c - t) / t)^2) over t>0.

    This makes rare classes contribute more, which is desirable for balancing.
    """

    counts = np.asarray(counts).astype(np.float64)
    target = np.asarray(target).astype(np.float64)
    mask = target > 0
    if not np.any(mask):
        return 0.0
    d = (counts[mask] - target[mask]) / target[mask]
    return float(np.sum(d * d))


@dataclass(frozen=True)
class BalancedSplitResult:
    splits: list[tuple[np.ndarray, np.ndarray]]
    score: float
    scanner_levels: list[str]


def split_balanced_pfirrmann_scanner(
    *,
    y: np.ndarray,
    groups: np.ndarray,
    scanner_by_sample: np.ndarray,
    n_splits: int,
    n_classes: int,
    random_state: int,
    n_trials: int,
    w_grade: float,
    w_scanner: float,
    w_samples: float,
    w_groups: float,
) -> BalancedSplitResult:
    """
    Grouped K-fold split that balances BOTH:
      - disc-level Pfirrmann grade distribution (y)
      - patient-level scanner distribution (scanner_by_sample, constant within group)

    Implementation:
      - build per-patient grade count vector and scanner label
      - greedy assignment with multiple randomized restarts
      - objective = sum_f (weighted squared relative errors)
    """

    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    scanner_by_sample = np.asarray(scanner_by_sample).astype(str)

    if y.ndim != 1 or groups.ndim != 1 or scanner_by_sample.ndim != 1:
        raise ValueError("y/groups/scanner_by_sample must be 1-D.")
    if not (y.shape[0] == groups.shape[0] == scanner_by_sample.shape[0]):
        raise ValueError("y/groups/scanner_by_sample length mismatch.")
    if int(n_splits) < 2:
        raise ValueError("n_splits must be >=2.")

    uniq_groups, inv = np.unique(groups, return_inverse=True)
    G = int(len(uniq_groups))
    if G < int(n_splits):
        raise ValueError(f"Not enough groups for n_splits={n_splits}: n_groups={G}")

    # Scanner is case-level, so it must be constant within each group.
    scanner_levels: list[str] = sorted({str(x) for x in np.unique(scanner_by_sample).tolist()})
    scanner_to_idx = {s: i for i, s in enumerate(scanner_levels)}
    S = int(len(scanner_levels))

    idx_lists: list[list[int]] = [[] for _ in range(G)]
    for i, gi in enumerate(inv.tolist()):
        idx_lists[int(gi)].append(int(i))
    idx_by_group = [np.asarray(lst, dtype=int) for lst in idx_lists]

    grade_counts_by_group = np.zeros((G, int(n_classes)), dtype=np.int64)
    scanner_idx_by_group = np.full((G,), -1, dtype=np.int64)
    group_sizes = np.zeros((G,), dtype=np.int64)

    for gi in range(G):
        idx = idx_by_group[gi]
        group_sizes[gi] = int(len(idx))
        yy = y[idx]
        c = np.bincount((yy - 1).astype(int), minlength=int(n_classes)).astype(np.int64)
        grade_counts_by_group[gi] = c

        ss = pd.Series(scanner_by_sample[idx]).astype(str)
        uniq = ss.unique().tolist()
        if len(uniq) != 1:
            raise ValueError(f"scanner_by_sample is not constant within group={uniq_groups[gi]!r}: {uniq[:5]}")
        scanner_idx_by_group[gi] = int(scanner_to_idx[str(uniq[0])])

    grade_total = grade_counts_by_group.sum(axis=0).astype(np.float64)
    scanner_total = np.bincount(scanner_idx_by_group.astype(int), minlength=S).astype(np.float64)
    samples_total = float(group_sizes.sum())
    groups_total = float(G)

    grade_target = _safe_soft_target(grade_total, int(n_splits))
    scanner_target = _safe_soft_target(scanner_total, int(n_splits))
    samples_target = samples_total / float(n_splits)
    groups_target = groups_total / float(n_splits)

    rng = np.random.default_rng(int(random_state))

    def _fold_cost(
        grade_counts_f: np.ndarray,
        scanner_counts_f: np.ndarray,
        n_samples_f: float,
        n_groups_f: float,
    ) -> float:
        cost = 0.0
        if w_grade:
            cost += float(w_grade) * _rel_sq_err(grade_counts_f, grade_target)
        if w_scanner:
            cost += float(w_scanner) * _rel_sq_err(scanner_counts_f, scanner_target)
        if w_samples and samples_target > 0:
            d = (float(n_samples_f) - float(samples_target)) / float(samples_target)
            cost += float(w_samples) * float(d * d)
        if w_groups and groups_target > 0:
            d = (float(n_groups_f) - float(groups_target)) / float(groups_target)
            cost += float(w_groups) * float(d * d)
        return float(cost)

    # Priority: put "rare scanner" and "rare grade contribution" groups earlier.
    scanner_rarity = 1.0 / np.maximum(1.0, scanner_total[scanner_idx_by_group.astype(int)])
    grade_rarity = np.zeros((G,), dtype=np.float64)
    for gi in range(G):
        contrib = grade_counts_by_group[gi].astype(np.float64)
        denom = np.maximum(1.0, grade_total)
        grade_rarity[gi] = float(np.max(contrib / denom))

    base_priority = (
        5.0 * scanner_rarity
        + 3.0 * grade_rarity
        + (group_sizes.astype(np.float64) / max(1.0, samples_total))
    )

    best_score = float("inf")
    best_fold_groups: list[list[int]] | None = None

    T = int(max(1, n_trials))
    for _ in range(T):
        # Randomized but deterministic (rng is seeded). We permute once per trial.
        order = np.arange(G, dtype=int)
        rng.shuffle(order)
        # Stable sort by priority so shuffle only breaks ties among similar groups.
        order = order[np.argsort(-base_priority[order], kind="stable")]

        fold_grade = np.zeros((int(n_splits), int(n_classes)), dtype=np.float64)
        fold_scanner = np.zeros((int(n_splits), S), dtype=np.float64)
        fold_samples = np.zeros((int(n_splits),), dtype=np.float64)
        fold_groups = np.zeros((int(n_splits),), dtype=np.float64)
        fold_costs = np.zeros((int(n_splits),), dtype=np.float64)

        fold_group_lists: list[list[int]] = [[] for _ in range(int(n_splits))]

        # IMPORTANT: empty folds have a non-zero cost (relative to target).
        empty_cost = _fold_cost(
            np.zeros((int(n_classes),), dtype=np.float64),
            np.zeros((S,), dtype=np.float64),
            0.0,
            0.0,
        )
        fold_costs[:] = float(empty_cost)
        total_cost = float(np.sum(fold_costs))

        for gi in order.tolist():
            g_grade = grade_counts_by_group[gi].astype(np.float64)
            s_idx = int(scanner_idx_by_group[gi])
            g_size = float(group_sizes[gi])

            best_f = 0
            best_total = float("inf")

            # Evaluate placing this group into each fold.
            for f in range(int(n_splits)):
                new_grade = fold_grade[f] + g_grade
                new_scanner = fold_scanner[f].copy()
                new_scanner[s_idx] += 1.0  # patient-level
                new_samples = fold_samples[f] + g_size
                new_groups = fold_groups[f] + 1.0

                new_cost_f = _fold_cost(new_grade, new_scanner, new_samples, new_groups)
                cand_total = float(total_cost - fold_costs[f] + new_cost_f)

                if cand_total < best_total - 1e-12:
                    best_total = cand_total
                    best_f = int(f)
                elif abs(cand_total - best_total) <= 1e-12:
                    # Tie-break: prefer the fold with fewer samples so far (keeps sizes even).
                    if fold_samples[f] < fold_samples[best_f] - 1e-12:
                        best_f = int(f)

            # Commit to best fold.
            f = best_f
            fold_grade[f] += g_grade
            fold_scanner[f, s_idx] += 1.0
            fold_samples[f] += g_size
            fold_groups[f] += 1.0

            new_cost = _fold_cost(fold_grade[f], fold_scanner[f], fold_samples[f], fold_groups[f])
            total_cost = float(total_cost - fold_costs[f] + new_cost)
            fold_costs[f] = float(new_cost)
            fold_group_lists[f].append(int(gi))

        if total_cost < best_score:
            best_score = float(total_cost)
            best_fold_groups = fold_group_lists

    assert best_fold_groups is not None

    # Build indices for each fold.
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    all_groups_set = set(range(G))
    for f in range(int(n_splits)):
        val_groups = set(best_fold_groups[f])
        train_groups = all_groups_set - val_groups
        val_idx = (
            np.concatenate([idx_by_group[gi] for gi in sorted(val_groups)], axis=0)
            if val_groups
            else np.zeros((0,), dtype=int)
        )
        train_idx = (
            np.concatenate([idx_by_group[gi] for gi in sorted(train_groups)], axis=0)
            if train_groups
            else np.zeros((0,), dtype=int)
        )
        splits.append((train_idx.astype(int), val_idx.astype(int)))

    return BalancedSplitResult(splits=splits, score=float(best_score), scanner_levels=scanner_levels)


def get_cv_splits(
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: dict[str, Any],
    meta: pd.DataFrame | None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """
    Single SSOT for outer CV splitting used by both train.py and tools scripts.
    """

    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    cv_cfg = cfg.get("cv", {}) or {}
    n_splits = int(cv_cfg["n_splits"])
    splitter = str(cv_cfg.get("splitter", "groupkfold")).strip().lower().replace("-", "_")
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = int(cv_cfg.get("random_state", cfg.get("training", {}).get("seed", 0)))

    ord_cfg = cfg.get("ordinal", {}) or {}
    n_classes = int(ord_cfg.get("n_classes", int(np.max(np.asarray(y).astype(int)))))

    bal_cfg = cv_cfg.get("balance_pfirrmann_scanner", {}) or {}
    bal_enabled = bool(bal_cfg.get("enabled", False))

    if bal_enabled:
        if meta is None:
            raise ValueError("cv.balance_pfirrmann_scanner.enabled requires meta (with patient_id). meta=None was provided.")
        scanner = load_scanner_for_samples(cfg=cfg, meta=meta, balance_cfg=bal_cfg).to_numpy()

        weights = bal_cfg.get("weights", {}) or {}
        res = split_balanced_pfirrmann_scanner(
            y=np.asarray(y).astype(int),
            groups=np.asarray(groups),
            scanner_by_sample=scanner,
            n_splits=int(n_splits),
            n_classes=int(n_classes),
            random_state=int(random_state),
            n_trials=int(bal_cfg.get("n_trials", 200)),
            w_grade=float(weights.get("grade", 1.0)),
            w_scanner=float(weights.get("scanner", 1.0)),
            w_samples=float(weights.get("n_samples", 0.2)),
            w_groups=float(weights.get("n_groups", 0.2)),
        )

        # Quick verification metrics (used by tools + saved to cv_split_info.json).
        y_int = np.asarray(y).astype(int)
        groups_arr = np.asarray(groups)
        scanner_arr = np.asarray(scanner).astype(str)
        S = int(len(res.scanner_levels))
        scanner_to_idx = {s: i for i, s in enumerate(res.scanner_levels)}

        grade_fold_max: list[float] = []
        scanner_fold_max: list[float] = []
        for tr_idx, va_idx in res.splits:
            # Grade balance (disc-level)
            c_tr = np.bincount((y_int[tr_idx] - 1).astype(int), minlength=int(n_classes)).astype(np.float64)
            c_va = np.bincount((y_int[va_idx] - 1).astype(int), minlength=int(n_classes)).astype(np.float64)
            p_tr = c_tr / max(1.0, float(c_tr.sum()))
            p_va = c_va / max(1.0, float(c_va.sum()))
            grade_fold_max.append(float(np.max(np.abs(p_va - p_tr))))

            # Scanner balance (patient-level)
            pat_tr = np.unique(groups_arr[tr_idx])
            pat_va = np.unique(groups_arr[va_idx])
            # Each patient has a single scanner label (scanner is constant within group by construction).
            sc_by_pat = pd.Series(scanner_arr, index=groups_arr).groupby(level=0).first()

            def _scanner_pct(pat: np.ndarray) -> np.ndarray:
                sc = sc_by_pat.loc[pat].astype(str).to_numpy()
                out = np.zeros((S,), dtype=np.float64)
                for v in sc.tolist():
                    out[int(scanner_to_idx[str(v)])] += 1.0
                tot = float(out.sum())
                return out / max(1.0, tot)

            p_sc_tr = _scanner_pct(pat_tr)
            p_sc_va = _scanner_pct(pat_va)
            scanner_fold_max.append(float(np.max(np.abs(p_sc_va - p_sc_tr))))

        grade_max_abs_diff_pctpt = 100.0 * float(np.max(grade_fold_max)) if grade_fold_max else 0.0
        scanner_max_abs_diff_pctpt = 100.0 * float(np.max(scanner_fold_max)) if scanner_fold_max else 0.0

        info = {
            "splitter": "balanced_pfirrmann_scanner",
            "n_splits": int(n_splits),
            "random_state": int(random_state),
            "score": float(res.score),
            "grade_max_abs_diff_pctpt": float(grade_max_abs_diff_pctpt),
            "scanner_max_abs_diff_pctpt": float(scanner_max_abs_diff_pctpt),
            "scanner_levels": res.scanner_levels,
        }
        return list(res.splits), info

    if splitter == "groupkfold":
        cv_splitter = GroupKFold(n_splits=n_splits)
    elif splitter in {"stratified_groupkfold", "stratified_group_kfold"}:
        cv_splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=(random_state if shuffle else None),
        )
    else:
        raise ValueError("cv.splitter must be one of: groupkfold, stratified_groupkfold")

    splits = [(tr.astype(int), va.astype(int)) for (tr, va) in cv_splitter.split(X, y, groups)]
    info = {
        "splitter": str(splitter),
        "n_splits": int(n_splits),
        "shuffle": bool(shuffle),
        "random_state": int(random_state),
    }

    # Optional verification metrics (helps detect distribution drift early).
    y_int = np.asarray(y).astype(int)
    grade_fold_max: list[float] = []
    for tr_idx, va_idx in splits:
        c_tr = np.bincount((y_int[tr_idx] - 1).astype(int), minlength=int(n_classes)).astype(np.float64)
        c_va = np.bincount((y_int[va_idx] - 1).astype(int), minlength=int(n_classes)).astype(np.float64)
        p_tr = c_tr / max(1.0, float(c_tr.sum()))
        p_va = c_va / max(1.0, float(c_va.sum()))
        grade_fold_max.append(float(np.max(np.abs(p_va - p_tr))))
    info["grade_max_abs_diff_pctpt"] = float(100.0 * float(np.max(grade_fold_max)) if grade_fold_max else 0.0)

    if meta is not None:
        try:
            scanner = load_scanner_for_samples(cfg=cfg, meta=meta, balance_cfg=bal_cfg).to_numpy()
            groups_arr = np.asarray(groups)
            scanner_arr = np.asarray(scanner).astype(str)
            scanner_levels = sorted({str(x) for x in np.unique(scanner_arr).tolist()})
            S = int(len(scanner_levels))
            scanner_to_idx = {s: i for i, s in enumerate(scanner_levels)}

            sc_by_pat = pd.Series(scanner_arr, index=groups_arr).groupby(level=0).first()

            def _scanner_pct(pat: np.ndarray) -> np.ndarray:
                sc = sc_by_pat.loc[pat].astype(str).to_numpy()
                out = np.zeros((S,), dtype=np.float64)
                for v in sc.tolist():
                    out[int(scanner_to_idx[str(v)])] += 1.0
                tot = float(out.sum())
                return out / max(1.0, tot)

            scanner_fold_max: list[float] = []
            for tr_idx, va_idx in splits:
                pat_tr = np.unique(groups_arr[tr_idx])
                pat_va = np.unique(groups_arr[va_idx])
                p_sc_tr = _scanner_pct(pat_tr)
                p_sc_va = _scanner_pct(pat_va)
                scanner_fold_max.append(float(np.max(np.abs(p_sc_va - p_sc_tr))))

            info["scanner_max_abs_diff_pctpt"] = float(
                100.0 * float(np.max(scanner_fold_max)) if scanner_fold_max else 0.0
            )
        except Exception:
            # If statistics.csv is missing/unreadable, keep grade metrics only.
            pass

    return splits, info
