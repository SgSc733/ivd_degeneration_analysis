from __future__ import annotations

import argparse
import csv
import datetime as _dt
from pathlib import Path
import zipfile

import pydicom


# NOTE:
# - We intentionally avoid reading or exporting patient-identifying tags (PatientName, PatientID, etc.).
# - We only scan DICOM headers (stop_before_pixels=True) for batch/protocol harmonization.


_TAG_MAP: list[tuple[str, str]] = [
    ("StudyInstanceUID", "study_instance_uid"),
    ("SeriesInstanceUID", "series_instance_uid"),
    ("SeriesNumber", "series_number"),
    ("Modality", "modality"),
    ("BodyPartExamined", "body_part_examined"),
    ("InstitutionName", "institution_name"),
    ("StationName", "station_name"),
    ("Manufacturer", "manufacturer"),
    ("ManufacturerModelName", "manufacturer_model_name"),
    ("MagneticFieldStrength", "magnetic_field_strength"),
    ("SoftwareVersions", "software_versions"),
    ("DeviceSerialNumber", "device_serial_number"),
    ("ProtocolName", "protocol_name"),
    ("SeriesDescription", "series_description"),
    ("SequenceName", "sequence_name"),
    ("ScanningSequence", "scanning_sequence"),
    ("MRAcquisitionType", "mr_acquisition_type"),
    ("InPlanePhaseEncodingDirection", "in_plane_phase_encoding_direction"),
    ("RepetitionTime", "repetition_time"),
    ("EchoTime", "echo_time"),
    ("InversionTime", "inversion_time"),
    ("FlipAngle", "flip_angle"),
    ("EchoTrainLength", "echo_train_length"),
    ("EchoNumbers", "echo_numbers"),
    ("PixelSpacing", "pixel_spacing"),
    ("SliceThickness", "slice_thickness"),
    ("SpacingBetweenSlices", "spacing_between_slices"),
    ("Rows", "rows"),
    ("Columns", "columns"),
]

_SPECIFIC_TAGS = [k for k, _ in _TAG_MAP]


def _to_csv_cell(v: object) -> str:
    if v is None:
        return ""

    # pydicom's MultiValue / DS / IS / etc.
    try:
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(_to_csv_cell(x) for x in v) + "]"
    except Exception:
        pass

    # Some pydicom types can be stringified safely; keep it compact.
    s = str(v)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    return s


def _read_minimal_ds(z: zipfile.ZipFile, name: str):
    with z.open(name) as f:
        # force=True is helpful for some vendor/private variants.
        return pydicom.dcmread(
            f,
            stop_before_pixels=True,
            force=True,
            specific_tags=_SPECIFIC_TAGS,
        )


def scan_zip_series(
    zip_path: Path,
    *,
    max_instances: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """
    Scan a zip that contains DICOM files, return 1 row per (Study, Series).

    Args:
        zip_path: Path to a .zip file.
        max_instances: Max DICOM instances to read in this zip (0 means no limit).
    """
    series_rows: dict[str, dict[str, object]] = {}
    n_dcm_total = 0
    n_read = 0
    n_bad = 0

    with zipfile.ZipFile(zip_path, "r") as z:
        members = [n for n in z.namelist() if not n.endswith("/")]
        # Heuristic: prefer common DICOM extensions first.
        members.sort(key=lambda x: (0 if x.lower().endswith((".dcm", ".ima")) else 1, x))

        for name in members:
            if max_instances and n_read >= max_instances:
                break

            # Try to avoid obviously non-DICOM files.
            if not name.lower().endswith((".dcm", ".ima")) and "." in Path(name).name:
                continue

            try:
                ds = _read_minimal_ds(z, name)
            except Exception:
                n_bad += 1
                continue

            n_dcm_total += 1
            n_read += 1

            series_uid = _to_csv_cell(getattr(ds, "SeriesInstanceUID", None))
            if not series_uid:
                continue
            study_uid = _to_csv_cell(getattr(ds, "StudyInstanceUID", None))
            key = f"{study_uid}|{series_uid}"

            row = series_rows.get(key)
            if row is None:
                row = {
                    "zip_name": zip_path.name,
                    "zip_stem": zip_path.stem,
                    "dicom_path_in_zip": name,
                    "n_instances_in_series_seen": 0,
                }
                for dicom_key, out_key in _TAG_MAP:
                    row[out_key] = _to_csv_cell(getattr(ds, dicom_key, None))
                series_rows[key] = row

            row["n_instances_in_series_seen"] = int(row["n_instances_in_series_seen"]) + 1

    meta = {
        "zip_name": zip_path.name,
        "zip_stem": zip_path.stem,
        "n_instances_read": int(n_read),
        "n_instances_counted_as_dicom": int(n_dcm_total),
        "n_bad_instances": int(n_bad),
        "n_series": int(len(series_rows)),
        "max_instances": int(max_instances),
    }
    return list(series_rows.values()), meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan DICOM headers inside zip files and export 1 row per series.")
    ap.add_argument("--zip-dir", required=True, help="Directory containing *.zip DICOM archives (e.g. E:/dicom)")
    ap.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path. If empty, save under outputs/ with timestamp.",
    )
    ap.add_argument(
        "--max-instances-per-zip",
        type=int,
        default=0,
        help="Max DICOM instances to read per zip (0 = no limit). Use to speed up on very large zips.",
    )
    args = ap.parse_args()

    zip_dir = Path(args.zip_dir)
    if not zip_dir.exists():
        raise FileNotFoundError(f"zip-dir not found: {zip_dir}")

    out_csv = Path(args.out_csv) if str(args.out_csv).strip() else None
    if out_csv is None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = Path("outputs") / f"dicom_zip_series_headers_{ts}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    zip_paths = sorted(zip_dir.glob("*.zip"), key=lambda p: p.name)
    if not zip_paths:
        raise FileNotFoundError(f"No *.zip found under: {zip_dir}")

    fieldnames = [
        "zip_name",
        "zip_stem",
        "dicom_path_in_zip",
        "n_instances_in_series_seen",
    ] + [out_key for _, out_key in _TAG_MAP]

    # Write as we go to avoid keeping everything in memory.
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        totals = {"zip": 0, "series": 0, "instances_read": 0, "bad": 0}
        for i, zp in enumerate(zip_paths, start=1):
            rows, meta = scan_zip_series(zp, max_instances=int(args.max_instances_per_zip))
            totals["zip"] += 1
            totals["series"] += int(meta["n_series"])
            totals["instances_read"] += int(meta["n_instances_read"])
            totals["bad"] += int(meta["n_bad_instances"])

            for r in rows:
                # Ensure all keys exist for DictWriter.
                for k in fieldnames:
                    if k not in r:
                        r[k] = ""
                w.writerow(r)

            print(
                f"[{i}/{len(zip_paths)}] {zp.name}: series={meta['n_series']}, "
                f"instances_read={meta['n_instances_read']}, bad={meta['n_bad_instances']}"
            )

    print("\n[Done]")
    print(f"- out_csv: {out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

