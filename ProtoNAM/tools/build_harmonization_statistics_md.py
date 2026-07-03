from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _list_image_ids(image_dir: Path) -> dict[int, str]:
    """
    Returns:
        Mapping: image_id -> file_ext (e.g. "nii.gz", "mha")
    """
    out: dict[int, str] = {}
    for p in image_dir.iterdir():
        if p.is_dir():
            continue

        name = p.name
        m = re.match(r"^(\d+)\.nii\.gz$", name, flags=re.IGNORECASE)
        if m:
            out[int(m.group(1))] = "nii.gz"
            continue

        m = re.match(r"^(\d+)\.(mha|nii|nrrd|mhd)$", name, flags=re.IGNORECASE)
        if m:
            out[int(m.group(1))] = m.group(2).lower()
            continue

    return out


def _normalize_str(s: object) -> str:
    if s is None:
        return ""
    v = str(s).replace("\r", " ").replace("\n", " ").strip()
    return v


def _batch_scanner(manufacturer: object, model: object, field_strength: object) -> str:
    return f"{_normalize_str(manufacturer)}__{_normalize_str(model)}__{_normalize_str(field_strength)}"


def _select_1_124_from_dicom_series_csv(
    dicom_csv: Path,
    *,
    image_ids_1_124: set[int],
) -> pd.DataFrame:
    df = pd.read_csv(dicom_csv)
    df["case_id"] = df["zip_stem"].astype(str).astype(int)
    df = df[(df["case_id"] >= 1) & (df["case_id"] <= 124)].copy()

    # Keep only the sagittal T2 sequences we used.
    patts = ["t2_tse_sag", "t2_fse_sag", "osag t2"]
    for c in ["series_description", "protocol_name"]:
        df[c] = df[c].fillna("").astype(str)

    mask = False
    for p in patts:
        mask = mask | df["series_description"].str.lower().str.contains(p) | df["protocol_name"].str.lower().str.contains(p)
    df = df[mask].copy()

    df["n_instances_in_series_seen"] = pd.to_numeric(df["n_instances_in_series_seen"], errors="coerce").fillna(0).astype(int)
    # If multiple series match for a case, choose the one with most instances.
    df = df.sort_values(["case_id", "n_instances_in_series_seen"], ascending=[True, False])
    df = df.groupby("case_id", as_index=False).head(1)

    # Intersect with converted images directory.
    df = df[df["case_id"].isin(sorted(image_ids_1_124))].copy()

    df["image_id"] = df["case_id"]
    df["source_case_id"] = df["case_id"]
    df["cohort"] = "1-124"
    df["source_table"] = "dicom_zip_series_headers"

    # Ensure columns exist (keep unified schema).
    for c in ["institution_name", "station_name"]:
        if c not in df.columns:
            df[c] = ""

    return df


def _select_ge200_from_overview_csv(
    overview_csv: Path,
    *,
    image_ids_ge200: set[int],
) -> pd.DataFrame:
    df = pd.read_csv(overview_csv)
    df["new_file_name"] = df["new_file_name"].astype(str)

    # Only keep rows like "2_t2" (exclude t1, t2_SPACE, etc.).
    df = df[df["new_file_name"].str.lower().str.endswith("_t2")].copy()
    df["source_case_id"] = df["new_file_name"].str.split("_").str[0].astype(int)
    df["image_id"] = df["source_case_id"] + 200

    # Intersect with converted images directory.
    df = df[df["image_id"].isin(sorted(image_ids_ge200))].copy()

    # Rename to unified snake_case used by dicom_zip_series_headers.csv.
    rename = {
        "BodyPartExamined": "body_part_examined",
        "DeviceSerialNumber": "device_serial_number",
        "EchoNumbers": "echo_numbers",
        "EchoTime": "echo_time",
        "EchoTrainLength": "echo_train_length",
        "FlipAngle": "flip_angle",
        "InPlanePhaseEncodingDirection": "in_plane_phase_encoding_direction",
        "MRAcquisitionType": "mr_acquisition_type",
        "MagneticFieldStrength": "magnetic_field_strength",
        "Manufacturer": "manufacturer",
        "ManufacturerModelName": "manufacturer_model_name",
        "RepetitionTime": "repetition_time",
        "ScanningSequence": "scanning_sequence",
        "SequenceName": "sequence_name",
        "SeriesDescription": "series_description",
        "SliceThickness": "slice_thickness",
        "SoftwareVersions": "software_versions",
        "SpacingBetweenSlices": "spacing_between_slices",
        "PixelSpacing": "pixel_spacing",
        "InversionTime": "inversion_time",
    }
    for k in rename:
        if k not in df.columns:
            # Some columns (e.g., InversionTime) may be missing; add as empty.
            df[k] = ""
    df = df.rename(columns=rename)

    # Columns that exist in DICOM-derived table but not in overview.csv:
    df["institution_name"] = ""
    df["station_name"] = ""
    df["protocol_name"] = ""
    df["rows"] = ""
    df["columns"] = ""

    df["cohort"] = ">=200"
    df["source_table"] = "overview"
    return df


def _to_markdown(md_path: Path, *, df_all: pd.DataFrame, image_ext: dict[int, str]) -> None:
    ids = sorted(image_ext.keys())
    ids_1_124 = [x for x in ids if 1 <= x <= 124]
    ids_ge200 = [x for x in ids if x >= 200]

    # Unified output columns (keep only harmonization-relevant fields).
    out_cols = [
        "image_id",
        "cohort",
        "series_description",
        "protocol_name",
        "manufacturer",
        "manufacturer_model_name",
        "magnetic_field_strength",
        "software_versions",
        "device_serial_number",
        "institution_name",
        "station_name",
        "repetition_time",
        "echo_time",
        "inversion_time",
        "flip_angle",
        "echo_train_length",
        "pixel_spacing",
        "slice_thickness",
        "spacing_between_slices",
        "batch_scanner",
        "source_table",
        "source_case_id",
    ]

    df = df_all.copy()
    # Make sure all output columns exist.
    for c in out_cols:
        if c not in df.columns:
            df[c] = ""
    df = df[out_cols].copy()
    df = df.sort_values(["image_id"]).reset_index(drop=True)

    # Summaries
    df_scanner = (
        df.groupby("batch_scanner")["image_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="n_images")
    )
    df_manu = (
        df.groupby("manufacturer")["image_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="n_images")
    )
    df_field = (
        df.groupby("magnetic_field_strength")["image_id"]
        .nunique()
        .sort_values(ascending=False)
        .reset_index(name="n_images")
    )

    # Compose markdown
    lines: list[str] = []
    lines.append("# 纳入研究图像统计信息（harmonization 用）")
    lines.append("")
    lines.append("## 1. 数据源与筛选规则（复现说明）")
    lines.append("")
    lines.append("- 转换后图像目录：`E:\\image`（文件名为序号）")
    lines.append(f"- 目录内数值型序号总数：{len(ids)}（1-124：{len(ids_1_124)}；>=200：{len(ids_ge200)}）")
    lines.append("- 1-124：来自 `E:\\ProtoNAM\\outputs\\dicom_zip_series_headers.csv`，筛选 `series_description/protocol_name` 包含：`t2_tse_sag` / `t2_fse_sag` / `OSag T2`，并与 `E:\\image` 取交集")
    lines.append("- >=200：来自 `E:\\data\\overview.csv`，筛选 `new_file_name` 以 `_t2` 结尾，映射 `image_id = (overview_id + 200)`，并与 `E:\\image` 取交集")
    lines.append("")
    lines.append("## 2. 推荐 batch 定义（用于特征层 harmonization）")
    lines.append("")
    lines.append("- 推荐起步：`batch_scanner = manufacturer__manufacturer_model_name__magnetic_field_strength`（文件中已给出该列）")
    lines.append("- 可选增强：当 `institution_name/station_name` 非空且每个中心样本数足够时，再考虑按中心拆分 batch（否则 batch 过碎会导致校正不稳定）")
    lines.append("")
    lines.append("## 3. 批次统计（按纳入的 277 例图像）")
    lines.append("")
    lines.append("### 3.1 batch_scanner 计数")
    lines.append("")
    lines.append("```csv")
    lines.append(df_scanner.to_csv(index=False, lineterminator="\n"))
    lines.append("```")
    lines.append("")
    lines.append("### 3.2 Manufacturer 计数")
    lines.append("")
    lines.append("```csv")
    lines.append(df_manu.to_csv(index=False, lineterminator="\n"))
    lines.append("```")
    lines.append("")
    lines.append("### 3.3 MagneticFieldStrength 计数")
    lines.append("")
    lines.append("```csv")
    lines.append(df_field.to_csv(index=False, lineterminator="\n"))
    lines.append("```")
    lines.append("")
    lines.append("## 4. 纳入图像的 harmonization 相关元数据（统一列名）")
    lines.append("")
    lines.append("- 说明：以下表格仅保留对 harmonization 有用的字段；`source_table/source_case_id` 用于追溯来源。")
    lines.append("")
    lines.append("```csv")
    lines.append(df.to_csv(index=False, lineterminator="\n"))
    lines.append("```")

    # Force LF newlines so markdown code blocks don't show stray '\r' in some viewers.
    with md_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="Build harmonization-ready metadata statistics.md for included images.")
    ap.add_argument("--image-dir", default=r"E:/image")
    ap.add_argument("--dicom-series-csv", default=r"E:/ProtoNAM/outputs/dicom_zip_series_headers.csv")
    ap.add_argument("--overview-csv", default=r"E:/data/overview.csv")
    ap.add_argument("--out-md", default=r"outputs/statistics.md")
    args = ap.parse_args()

    image_dir = Path(args.image_dir)
    dicom_csv = Path(args.dicom_series_csv)
    overview_csv = Path(args.overview_csv)
    out_md = Path(args.out_md)

    image_ext = _list_image_ids(image_dir)
    if not image_ext:
        raise FileNotFoundError(f"No numeric images found under: {image_dir}")

    ids_1_124 = {k for k in image_ext if 1 <= k <= 124}
    ids_ge200 = {k for k in image_ext if k >= 200}

    df_a = _select_1_124_from_dicom_series_csv(dicom_csv, image_ids_1_124=ids_1_124)
    df_b = _select_ge200_from_overview_csv(overview_csv, image_ids_ge200=ids_ge200)

    df_all = pd.concat([df_a, df_b], ignore_index=True)
    df_all["batch_scanner"] = df_all.apply(
        lambda r: _batch_scanner(r.get("manufacturer"), r.get("manufacturer_model_name"), r.get("magnetic_field_strength")),
        axis=1,
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    _to_markdown(out_md, df_all=df_all, image_ext=image_ext)
    print(f"[OK] wrote: {out_md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
