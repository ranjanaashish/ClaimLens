"""
core/tsv_loader.py
==================
Utility for loading images from dataset files (TSV / CSV / Excel).

Supports three image-column modes:
  1. **base64** — column contains a raw JPEG/PNG base64 string (INS-MMBench format)
  2. **url**    — column contains an HTTP/HTTPS URL
  3. **path**   — column contains a local file path

Usage
-----
    from core.tsv_loader import load_dataset_file, detect_image_column

    images = load_dataset_file(
        file_or_path="data/INS-MMBench/dataset/multi_step_claim.tsv",
        max_rows=10,
    )
    # returns list[tuple[str, PIL.Image.Image, dict]]
    #   label, PIL image, metadata dict from other columns
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import IO, Union

from PIL import Image

# ---------------------------------------------------------------------------
# Column name heuristics
# ---------------------------------------------------------------------------

_BASE64_PREFIXES = ("/9j/", "iVBOR", "R0lGOD", "UklGR", "AAAAB")  # JPEG, PNG, GIF, WEBP, RIFF
_URL_PREFIXES = ("http://", "https://")
_PATH_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif"}

_KNOWN_IMAGE_COLS = ["image", "img", "photo", "picture", "screenshot"]
_KNOWN_URL_COLS = ["url", "image_url", "img_url", "link", "href", "image_link"]
_KNOWN_PATH_COLS = ["path", "image_path", "img_path", "file", "filename", "filepath", "file_path"]


def _sniff_value(val: str) -> str:
    """Return 'base64', 'url', 'path', or 'unknown' for a sample value."""
    val = str(val).strip()
    if any(val.startswith(p) for p in _BASE64_PREFIXES):
        return "base64"
    if any(val.startswith(p) for p in _URL_PREFIXES):
        return "url"
    ext = Path(val).suffix.lower()
    if ext in _PATH_EXTENSIONS and (os.sep in val or "/" in val):
        return "path"
    # Try decoding as base64 if long enough
    if len(val) > 200:
        try:
            decoded = base64.b64decode(val[:64])
            if decoded[:3] == b"\xff\xd8\xff" or decoded[:4] == b"\x89PNG":
                return "base64"
        except Exception:
            pass
    return "unknown"


def detect_image_column(df) -> tuple[str | None, str]:
    """
    Detect which column contains images and what encoding mode they use.

    Returns
    -------
    (column_name, mode)  where mode ∈ {"base64", "url", "path", "unknown"}
    """
    import pandas as pd

    cols_lower = {c.lower(): c for c in df.columns}

    # Priority 1: known image-column names → sniff their values
    for candidate in _KNOWN_IMAGE_COLS + _KNOWN_URL_COLS + _KNOWN_PATH_COLS:
        if candidate in cols_lower:
            col = cols_lower[candidate]
            sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
            mode = _sniff_value(str(sample))
            if mode != "unknown":
                return col, mode

    # Priority 2: sniff every column
    for col in df.columns:
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else ""
        mode = _sniff_value(str(sample))
        if mode != "unknown":
            return col, mode

    return None, "unknown"


def available_columns(df) -> list[str]:
    """Return all column names for manual override dropdown."""
    return list(df.columns)


# ---------------------------------------------------------------------------
# Image decoders
# ---------------------------------------------------------------------------

def _decode_base64(val: str) -> Image.Image:
    raw = base64.b64decode(val.strip())
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _decode_url(url: str, timeout: int = 10) -> Image.Image:
    import requests

    resp = requests.get(url.strip(), timeout=timeout, stream=True)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return img


def _decode_path(path_str: str, base_dir: str | None = None) -> Image.Image:
    p = Path(path_str.strip())
    if not p.is_absolute() and base_dir:
        p = Path(base_dir) / p
    return Image.open(p).convert("RGB")


def _decode_image(val: str, mode: str, base_dir: str | None = None) -> Image.Image | None:
    """Decode a single image value. Returns None on failure."""
    try:
        if mode == "base64":
            img = _decode_base64(val)
        elif mode == "url":
            img = _decode_url(val)
        elif mode == "path":
            img = _decode_path(val, base_dir)
        else:
            return None
        # Resize to max 1024×1024 for API budget
        img.thumbnail((1024, 1024), Image.LANCZOS)
        return img
    except Exception as e:
        print(f"[tsv_loader] Failed to decode image ({mode}): {e}")
        return None


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_dataset_file(
    file_or_path: Union[str, Path, IO],
    max_rows: int = 10,
    image_col: str | None = None,
    mode: str | None = None,
    label_col: str | None = None,
    base_dir: str | None = None,
    sep: str | None = None,
) -> list[tuple[str, Image.Image, dict]]:
    """
    Load images from a dataset TSV / CSV / Excel file.

    Parameters
    ----------
    file_or_path : str | Path | file-like
        Path to (or uploaded bytes of) the dataset file.
    max_rows : int
        Maximum number of rows to decode (default 10).
    image_col : str | None
        Column name containing images. Auto-detected if None.
    mode : str | None
        "base64" | "url" | "path". Auto-detected if None.
    label_col : str | None
        Column to use as image label (default: row index).
    base_dir : str | None
        Base directory for resolving relative paths (path mode only).
    sep : str | None
        Separator override. Auto-detected from extension if None.

    Returns
    -------
    list of (label, PIL.Image, metadata_dict)
        metadata_dict contains all non-image columns for that row.
    """
    import pandas as pd

    # --- Determine file format ---
    fname = ""
    if isinstance(file_or_path, (str, Path)):
        fname = str(file_or_path)
    elif hasattr(file_or_path, "name"):
        fname = getattr(file_or_path, "name", "")

    ext = Path(fname).suffix.lower()

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_or_path, nrows=max_rows)
    else:
        if sep is None:
            sep = "\t" if ext == ".tsv" else ","
        # Use chunksize to avoid loading huge files into memory
        chunk_iter = pd.read_csv(
            file_or_path,
            sep=sep,
            nrows=max_rows,
            on_bad_lines="skip",
            encoding="utf-8",
            low_memory=False,
        )
        df = chunk_iter if isinstance(chunk_iter, pd.DataFrame) else pd.DataFrame(chunk_iter)

    df = df.head(max_rows).reset_index(drop=True)

    # --- Detect image column and mode ---
    if image_col is None or mode is None:
        auto_col, auto_mode = detect_image_column(df)
        if image_col is None:
            image_col = auto_col
        if mode is None:
            mode = auto_mode

    if image_col is None:
        raise ValueError(
            "Could not auto-detect an image column. Please specify image_col explicitly."
        )
    if image_col not in df.columns:
        raise ValueError(f"Column '{image_col}' not found in file. Available: {list(df.columns)}")

    # --- Build result list ---
    results: list[tuple[str, Image.Image, dict]] = []

    for idx, row in df.iterrows():
        val = row.get(image_col, "")
        if not isinstance(val, str) or not val.strip():
            continue

        img = _decode_image(val, mode, base_dir=base_dir)
        if img is None:
            continue

        # Label
        if label_col and label_col in row:
            label = f"row{idx}_{row[label_col]}"
        else:
            label = f"row_{idx}"

        # Metadata: all columns except image
        meta = {k: v for k, v in row.items() if k != image_col}

        results.append((label, img, meta))

    return results


def sniff_columns(
    file_or_path: Union[str, Path, IO],
    max_rows: int = 5,
    sep: str | None = None,
) -> tuple[list[str], str | None, str]:
    """
    Quick preview: return (column_names, detected_image_col, detected_mode)
    without fully decoding any images.
    """
    import pandas as pd

    fname = ""
    if isinstance(file_or_path, (str, Path)):
        fname = str(file_or_path)
    elif hasattr(file_or_path, "name"):
        fname = getattr(file_or_path, "name", "")

    ext = Path(fname).suffix.lower()

    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_or_path, nrows=max_rows)
    else:
        if sep is None:
            sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(
            file_or_path,
            sep=sep,
            nrows=max_rows,
            on_bad_lines="skip",
            encoding="utf-8",
            low_memory=False,
        )

    col, mode = detect_image_column(df)
    return list(df.columns), col, mode
