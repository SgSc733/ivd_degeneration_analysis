from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Ensure project root is on sys.path when running as a script (sys.path[0] == tools/).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lumbar_pnam.data import DEFAULT_LEVEL_TO_COL, DEFAULT_SEGMENT_LEVELS, attach_pfirrmann_from_xlsx, load_model_input_csv
from lumbar_pnam.cv_split import get_cv_splits, load_scanner_for_samples


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_id(v: object) -> str:
    s = str(v).strip()
    # Handle Excel-like numeric id "1.0"
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def _scanner_cfg_for_print(cfg: dict, balance_cfg: dict) -> tuple[str, str, str]:
    sc = (balance_cfg.get("scanner", {}) or {}) if isinstance(balance_cfg, dict) else {}
    stats_path = sc.get("statistics_csv_path", None)
    case_id_col = sc.get("case_id_col", None)
    scanner_col = sc.get("scanner_col", None)
    if stats_path or case_id_col or scanner_col:
        return (str(stats_path or "statistics.csv"), str(case_id_col or "image_id"), str(scanner_col or "batch_scanner"))

    harm = cfg.get("harmonization", {}) or {}
    return (
        str(harm.get("statistics_csv_path", "statistics.csv")),
        str(harm.get("case_id_col", "image_id")),
        str(harm.get("batch_col", "batch_scanner")),
    )


def _grade_table(*, y: np.ndarray, idx: np.ndarray, n_classes: int) -> pd.DataFrame:
    y = np.asarray(y).astype(int)
    idx = np.asarray(idx).astype(int)
    yy = y[idx]
    # y is 1..K
    counts = np.bincount((yy - 1).astype(int), minlength=int(n_classes)).astype(int)
    total = int(len(idx))
    pct = (counts / max(1, total)).astype(float)
    return pd.DataFrame({"grade": np.arange(1, int(n_classes) + 1), "count": counts, "pct": pct})


def _value_table(values: np.ndarray) -> pd.DataFrame:
    s = pd.Series(values, dtype="string")
    vc = s.value_counts(dropna=False)
    total = int(vc.sum())
    df = vc.reset_index()
    df.columns = ["value", "count"]
    df["pct"] = df["count"].astype(float) / max(1.0, float(total))
    return df


def _merge_train_val(train: pd.DataFrame, val: pd.DataFrame, *, key: str) -> pd.DataFrame:
    out = train.merge(val, on=key, how="outer", suffixes=("_train", "_val")).fillna(0)
    out["count_train"] = out["count_train"].astype(int)
    out["count_val"] = out["count_val"].astype(int)
    out["pct_train"] = out["pct_train"].astype(float)
    out["pct_val"] = out["pct_val"].astype(float)
    out["pct_diff"] = (out["pct_val"] - out["pct_train"]).astype(float)
    return out.sort_values(key).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze per-fold balance of Pfirrmann grades and scanner batches for the current CV split. "
            "Uses the same loader + splitter settings as run.py / lumbar_pnam.train.train_group_kfold."
        )
    )
    ap.add_argument("--config", default="config.json", help="Path to config.json")
    ap.add_argument("--stats-csv", default=None, help="Override statistics.csv path (default: inferred from config)")
    ap.add_argument("--case-id-col", default=None, help="Override case id column in statistics CSV")
    ap.add_argument("--scanner-col", default=None, help="Override scanner/batch column in statistics CSV")
    ap.add_argument("--out-dir", default=None, help="If set, write CSV reports to this directory")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_json(cfg_path)

    data_cfg = cfg["data"]
    pooling_cfg = data_cfg.get("pooling", {}) or {}
    case_meta_cfg = data_cfg.get("case_meta_onehot", {}) or {}

    pooling_feature_types = pooling_cfg.get("feature_types", None)
    if pooling_feature_types is None:
        pooling_feature_types = None
    elif isinstance(pooling_feature_types, str):
        pooling_feature_types = [pooling_feature_types]
    else:
        pooling_feature_types = list(pooling_feature_types)

    loaded = load_model_input_csv(
        data_cfg["csv_path"],
        id_col=data_cfg["id_col"],
        label_col=data_cfg.get("label_col"),
        drop_cols=data_cfg.get("drop_cols", []),
        drop_patterns=data_cfg.get("drop_patterns", []),
        feature_order=str(data_cfg.get("feature_order", "csv")),
        classic_prefix=data_cfg.get("classic_prefix", "classic_"),
        patient_id_from_id_col=bool(data_cfg.get("patient_id_from_id_col", True)),
        patient_id_sep=str(data_cfg.get("patient_id_sep", "_")),
        level_from_id_col=bool(data_cfg.get("level_from_id_col", True)),
        level_sep=str(data_cfg.get("level_sep", "_")),
        segment_levels=list(data_cfg.get("segment_levels", DEFAULT_SEGMENT_LEVELS)),
        add_segment_onehot=bool(data_cfg.get("add_segment_onehot", True)),
        pooling_stats=list(pooling_cfg.get("stats", []) or []),
        pooling_feature_types=pooling_feature_types,
        pooling_pyr_prefix=str(pooling_cfg.get("pyr_prefix", "PyRadiomics_")),
        pooling_deep_prefix=str(pooling_cfg.get("deep_prefix", "DeepPCA_")),
        pooling_tensor_prefix=str(pooling_cfg.get("tensor_prefix", "TensorPCA_")),
        pooling_out_prefix=str(pooling_cfg.get("out_prefix", "pool_")),
        add_case_meta_onehot=bool(case_meta_cfg.get("enabled", False)),
        case_meta_csv_path=case_meta_cfg.get("statistics_csv_path"),
        case_meta_case_id_col=str(case_meta_cfg.get("case_id_col", "image_id")),
        case_meta_cols=list(case_meta_cfg.get("columns", []) or []),
        case_meta_onehot_prefix=str(case_meta_cfg.get("prefix", "meta_")),
        case_meta_onehot_drop_first=bool(case_meta_cfg.get("drop_first", False)),
        case_meta_on_missing=str(case_meta_cfg.get("on_missing", "error")),
    )

    if loaded.y is None:
        lab_cfg = cfg.get("labels", {}) or {}
        loaded = attach_pfirrmann_from_xlsx(
            loaded,
            xlsx_path=lab_cfg.get("xlsx_path", "pfirr_data.xlsx"),
            sheet_name=lab_cfg.get("sheet_name"),
            patient_id_col=str(lab_cfg.get("patient_id_col", "序号")),
            level_to_col=dict(
                lab_cfg.get(
                    "level_to_col",
                    DEFAULT_LEVEL_TO_COL,
                )
            ),
            on_missing=str(lab_cfg.get("on_missing", "error")),
        )
    if loaded.y is None:
        raise RuntimeError("Missing labels (pfirrmann).")

    ord_cfg = cfg.get("ordinal", {}) or {}
    n_classes = int(ord_cfg.get("n_classes", int(np.max(np.asarray(loaded.y).astype(int)))))

    # Apply optional overrides for scanner metadata in this script run.
    cv_cfg = cfg.get("cv", {}) or {}
    bal_cfg = cv_cfg.get("balance_pfirrmann_scanner", {}) or {}
    sc = (bal_cfg.get("scanner", {}) or {}) if isinstance(bal_cfg, dict) else {}
    if args.stats_csv is not None:
        sc["statistics_csv_path"] = str(args.stats_csv)
    if args.case_id_col is not None:
        sc["case_id_col"] = str(args.case_id_col)
    if args.scanner_col is not None:
        sc["scanner_col"] = str(args.scanner_col)
    bal_cfg["scanner"] = sc
    cv_cfg["balance_pfirrmann_scanner"] = bal_cfg
    cfg["cv"] = cv_cfg

    # Use the SSOT splitter (same logic as training).
    cv_splits, cv_split_info = get_cv_splits(X=loaded.X, y=loaded.y, groups=loaded.groups, cfg=cfg, meta=loaded.meta)

    scanner_disc = load_scanner_for_samples(cfg=cfg, meta=loaded.meta, balance_cfg=bal_cfg).to_numpy()
    stats_csv_path, case_id_col, scanner_col = _scanner_cfg_for_print(cfg, bal_cfg)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    y = np.asarray(loaded.y).astype(int)
    groups = np.asarray(loaded.groups)

    print("[CV Split Balance]")
    print(f"- cfg: {cfg_path}")
    print(f"- n_samples (discs): {len(y)}")
    print(f"- n_groups (patients): {int(len(np.unique(groups)))}")
    print(f"- splitter: {cv_split_info}")
    print(f"- scanner stats: {stats_csv_path} ({case_id_col=} {scanner_col=})")

    all_grade_rows: list[dict] = []
    all_scanner_rows: list[dict] = []
    all_scanner_patient_rows: list[dict] = []

    for fold, (tr_idx, va_idx) in enumerate(cv_splits):
        tr_idx = np.asarray(tr_idx).astype(int)
        va_idx = np.asarray(va_idx).astype(int)

        n_train = int(len(tr_idx))
        n_val = int(len(va_idx))
        n_train_pat = int(len(np.unique(groups[tr_idx])))
        n_val_pat = int(len(np.unique(groups[va_idx])))

        print(f"\n[Fold {fold}] n_train={n_train} n_val={n_val} | patients: train={n_train_pat} val={n_val_pat}")

        gt_tr = _grade_table(y=y, idx=tr_idx, n_classes=n_classes).rename(
            columns={"count": "count_train", "pct": "pct_train"}
        )
        gt_va = _grade_table(y=y, idx=va_idx, n_classes=n_classes).rename(
            columns={"count": "count_val", "pct": "pct_val"}
        )
        gt = _merge_train_val(gt_tr, gt_va, key="grade")
        gt_show = gt.copy()
        gt_show["pct_train"] = (100.0 * gt_show["pct_train"]).round(2)
        gt_show["pct_val"] = (100.0 * gt_show["pct_val"]).round(2)
        gt_show["pct_diff"] = (100.0 * gt_show["pct_diff"]).round(2)
        print("[Pfirrmann Grade] (count, %)")
        print(gt_show.to_string(index=False))

        for _, r in gt.iterrows():
            all_grade_rows.append(
                {
                    "fold": int(fold),
                    "split": "train",
                    "grade": int(r["grade"]),
                    "count": int(r["count_train"]),
                    "pct": float(r["pct_train"]),
                }
            )
            all_grade_rows.append(
                {
                    "fold": int(fold),
                    "split": "val",
                    "grade": int(r["grade"]),
                    "count": int(r["count_val"]),
                    "pct": float(r["pct_val"]),
                }
            )

        st_tr = _value_table(scanner_disc[tr_idx]).rename(columns={"value": "scanner"}).rename(
            columns={"count": "count_train", "pct": "pct_train"}
        )
        st_va = _value_table(scanner_disc[va_idx]).rename(columns={"value": "scanner"}).rename(
            columns={"count": "count_val", "pct": "pct_val"}
        )
        st = _merge_train_val(st_tr, st_va, key="scanner")
        st_show = st.copy()
        st_show["pct_train"] = (100.0 * st_show["pct_train"]).round(2)
        st_show["pct_val"] = (100.0 * st_show["pct_val"]).round(2)
        st_show["pct_diff"] = (100.0 * st_show["pct_diff"]).round(2)
        print(f"[{scanner_col}] (disc-level) (count, %)")
        print(st_show.to_string(index=False))

        for _, r in st.iterrows():
            all_scanner_rows.append(
                {
                    "fold": int(fold),
                    "split": "train",
                    "scanner": str(r["scanner"]),
                    "count": int(r["count_train"]),
                    "pct": float(r["pct_train"]),
                }
            )
            all_scanner_rows.append(
                {
                    "fold": int(fold),
                    "split": "val",
                    "scanner": str(r["scanner"]),
                    "count": int(r["count_val"]),
                    "pct": float(r["pct_val"]),
                }
            )

        # Patient-level scanner distribution (each patient counted once)
        pid_tr = pd.Series(groups[tr_idx]).map(_norm_id)
        pid_va = pd.Series(groups[va_idx]).map(_norm_id)
        scanner_by_patient = pd.Series(scanner_disc, index=pd.Series(groups).map(_norm_id)).groupby(level=0).first()
        sp_tr = _value_table(scanner_by_patient.loc[pid_tr.unique()].to_numpy()).rename(columns={"value": "scanner"})
        sp_va = _value_table(scanner_by_patient.loc[pid_va.unique()].to_numpy()).rename(columns={"value": "scanner"})
        sp = _merge_train_val(
            sp_tr.rename(columns={"count": "count_train", "pct": "pct_train"}),
            sp_va.rename(columns={"count": "count_val", "pct": "pct_val"}),
            key="scanner",
        )
        sp_show = sp.copy()
        sp_show["pct_train"] = (100.0 * sp_show["pct_train"]).round(2)
        sp_show["pct_val"] = (100.0 * sp_show["pct_val"]).round(2)
        sp_show["pct_diff"] = (100.0 * sp_show["pct_diff"]).round(2)
        print(f"[{scanner_col}] (patient-level) (count, %)")
        print(sp_show.to_string(index=False))

        for _, r in sp.iterrows():
            all_scanner_patient_rows.append(
                {
                    "fold": int(fold),
                    "split": "train",
                    "scanner": str(r["scanner"]),
                    "count": int(r["count_train"]),
                    "pct": float(r["pct_train"]),
                }
            )
            all_scanner_patient_rows.append(
                {
                    "fold": int(fold),
                    "split": "val",
                    "scanner": str(r["scanner"]),
                    "count": int(r["count_val"]),
                    "pct": float(r["pct_val"]),
                }
            )

    if out_dir is not None:
        pd.DataFrame(all_grade_rows).to_csv(out_dir / "fold_balance_grades.csv", index=False, encoding="utf-8")
        pd.DataFrame(all_scanner_rows).to_csv(out_dir / "fold_balance_scanner_discs.csv", index=False, encoding="utf-8")
        pd.DataFrame(all_scanner_patient_rows).to_csv(
            out_dir / "fold_balance_scanner_patients.csv", index=False, encoding="utf-8"
        )
        print(f"\n[Saved] {out_dir}")
        print("- fold_balance_grades.csv")
        print("- fold_balance_scanner_discs.csv")
        print("- fold_balance_scanner_patients.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
