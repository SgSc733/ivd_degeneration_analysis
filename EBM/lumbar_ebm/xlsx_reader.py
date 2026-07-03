from __future__ import annotations

"""
Minimal .xlsx reader (stdlib-only).

Why:
- The training environment might not always have openpyxl installed.
- We still need to read `pfirr_data.xlsx` (small, simple sheet) without extra deps.

Supported subset:
- shared strings (t="s")
- numeric cells (default)
- first row as header
"""

from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd


_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        data = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(data)
    out: list[str] = []
    for si in root.findall("main:si", _NS):
        texts: list[str] = []
        for t in si.findall(".//main:t", _NS):
            texts.append(t.text or "")
        out.append("".join(texts))
    return out


def _sheet_name_to_path(zf: zipfile.ZipFile) -> dict[str, str]:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))

    rid_to_target: dict[str, str] = {}
    for rel in rels.findall("pr:Relationship", _NS):
        rid = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rid and target:
            rid_to_target[rid] = target

    out: dict[str, str] = {}
    r_id_attr = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    for sh in wb.findall("main:sheets/main:sheet", _NS):
        name = sh.attrib.get("name")
        rid = sh.attrib.get(r_id_attr)
        target = rid_to_target.get(rid or "")
        if not (name and target):
            continue
        if not target.startswith("xl/"):
            target = "xl/" + target
        out[name] = target
    return out


def _cell_ref_to_rc(ref: str) -> tuple[int, int]:
    col = 0
    i = 0
    while i < len(ref) and ref[i].isalpha():
        col = col * 26 + (ord(ref[i].upper()) - 64)
        i += 1
    row = int(ref[i:])
    return row - 1, col - 1


def _parse_sheet_matrix(zf: zipfile.ZipFile, sheet_path: str, shared: list[str]) -> list[list[str | None]]:
    root = ET.fromstring(zf.read(sheet_path))
    sheetData = root.find("main:sheetData", _NS)
    if sheetData is None:
        return []

    rows: dict[int, dict[int, str | None]] = {}
    max_r = 0
    max_c = 0

    for row in sheetData.findall("main:row", _NS):
        r_idx = int(row.attrib.get("r", "1")) - 1
        max_r = max(max_r, r_idx)
        row_dict: dict[int, str | None] = {}

        for c in row.findall("main:c", _NS):
            ref = c.attrib.get("r")
            if not ref:
                continue
            rr, cc = _cell_ref_to_rc(ref)
            max_c = max(max_c, cc)

            t = c.attrib.get("t")
            v_el = c.find("main:v", _NS)
            if v_el is None:
                val: str | None = None
            else:
                v = v_el.text or ""
                if t == "s":
                    try:
                        val = shared[int(v)]
                    except Exception:
                        val = v
                else:
                    val = v
            row_dict[cc] = val

        rows[r_idx] = row_dict

    mat: list[list[str | None]] = []
    for r in range(max_r + 1):
        row_dict = rows.get(r, {})
        mat.append([row_dict.get(c) for c in range(max_c + 1)])
    return mat


def read_xlsx_sheet_to_df(xlsx_path: str | Path, sheet_name: str | None = None) -> pd.DataFrame:
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"xlsx not found: {xlsx_path}")

    with zipfile.ZipFile(xlsx_path) as zf:
        shared = _read_shared_strings(zf)
        name_to_path = _sheet_name_to_path(zf)
        if not name_to_path:
            raise ValueError("No sheets found in workbook.")

        if sheet_name is None:
            sheet_name = next(iter(name_to_path.keys()))
        if sheet_name not in name_to_path:
            raise ValueError(f"sheet '{sheet_name}' not found. Available: {list(name_to_path.keys())}")

        mat = _parse_sheet_matrix(zf, name_to_path[sheet_name], shared)

    if not mat:
        return pd.DataFrame()

    header = [str(x).strip() if x is not None else "" for x in mat[0]]
    rows = mat[1:]
    df = pd.DataFrame(rows, columns=header)
    df = df.dropna(how="all").reset_index(drop=True)
    return df

