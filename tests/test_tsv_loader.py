"""
tests/test_tsv_loader.py
Unit tests for core/tsv_loader.py
"""
import base64
import io
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.tsv_loader import (
    _sniff_value,
    detect_image_column,
    load_dataset_file,
    sniff_columns,
)


def _make_base64_row() -> str:
    """Create a tiny 10×10 red JPEG encoded as base64."""
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# 1. Sniff value
# ---------------------------------------------------------------------------

def test_sniff_base64():
    b64 = _make_base64_row()
    assert _sniff_value(b64) == "base64", "Should detect base64"

def test_sniff_url():
    assert _sniff_value("https://example.com/car.jpg") == "url"
    assert _sniff_value("http://images.example.com/test.png") == "url"

def test_sniff_path():
    assert _sniff_value("/home/user/images/car.jpg") == "path"
    assert _sniff_value("C:\\Users\\test\\car.png") in ("path", "unknown")

print("[1] _sniff_value tests...")
test_sniff_base64()
test_sniff_url()
test_sniff_path()
print("    PASS")

# ---------------------------------------------------------------------------
# 2. detect_image_column
# ---------------------------------------------------------------------------

def test_detect_base64_column():
    import pandas as pd
    b64 = _make_base64_row()
    df = pd.DataFrame({"index": [0], "image": [b64], "label": ["minor"]})
    col, mode = detect_image_column(df)
    assert col == "image", f"Expected 'image', got {col!r}"
    assert mode == "base64", f"Expected 'base64', got {mode!r}"

def test_detect_url_column():
    import pandas as pd
    df = pd.DataFrame({
        "index": [0],
        "image_url": ["https://example.com/car.jpg"],
        "label": ["minor"],
    })
    col, mode = detect_image_column(df)
    assert col == "image_url", f"Expected 'image_url', got {col!r}"
    assert mode == "url", f"Expected 'url', got {mode!r}"

print("[2] detect_image_column tests...")
test_detect_base64_column()
test_detect_url_column()
print("    PASS")

# ---------------------------------------------------------------------------
# 3. load_dataset_file — in-memory TSV with base64
# ---------------------------------------------------------------------------

def test_load_base64_tsv():
    import pandas as pd
    b64_1 = _make_base64_row()
    b64_2 = _make_base64_row()
    df = pd.DataFrame({
        "index": [0, 1],
        "image": [b64_1, b64_2],
        "Severity": ["minor", "moderate"],
        "claim": [0, 1],
    })
    buf = io.StringIO()
    df.to_csv(buf, sep="\t", index=False)
    buf.seek(0)

    # Pretend it's a .tsv file-like
    class FakeFile:
        name = "test.tsv"
        def read(self):
            return buf.getvalue().encode()

    # Load via StringIO directly
    buf.seek(0)
    results = load_dataset_file(buf, max_rows=10, sep="\t")
    assert len(results) == 2, f"Expected 2 rows, got {len(results)}"
    label, img, meta = results[0]
    assert isinstance(img, Image.Image), "Expected PIL Image"
    assert img.size[0] <= 1024 and img.size[1] <= 1024
    assert "Severity" in meta
    assert meta["Severity"] == "minor"
    print(f"    Loaded {len(results)} rows, first label={label!r}, meta keys={list(meta.keys())}")

print("[3] load_dataset_file (base64 TSV) tests...")
test_load_base64_tsv()
print("    PASS")

# ---------------------------------------------------------------------------
# 4. sniff_columns — against real INS-MMBench TSV (if present)
# ---------------------------------------------------------------------------

REAL_TSV = Path(__file__).parent.parent / "data" / "INS-MMBench" / "dataset" / "multi_step_claim.tsv"

def test_sniff_real_tsv():
    if not REAL_TSV.exists():
        print("    SKIP (TSV not present)")
        return
    cols, img_col, mode = sniff_columns(REAL_TSV, max_rows=3)
    print(f"    Columns: {cols}")
    print(f"    Detected image col: {img_col!r}, mode: {mode!r}")
    assert img_col is not None, "Should detect image column"
    assert mode == "base64", f"Expected base64 for INS-MMBench, got {mode!r}"

print("[4] sniff_columns (real TSV) test...")
test_sniff_real_tsv()
print("    PASS")

# ---------------------------------------------------------------------------
# 5. load_dataset_file — real TSV, 2 rows
# ---------------------------------------------------------------------------

def test_load_real_tsv():
    if not REAL_TSV.exists():
        print("    SKIP (TSV not present)")
        return
    results = load_dataset_file(REAL_TSV, max_rows=2)
    assert len(results) == 2, f"Expected 2, got {len(results)}"
    label, img, meta = results[0]
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size[0] <= 1024 and img.size[1] <= 1024
    print(f"    Row 0: label={label!r}, size={img.size}, meta_keys={list(meta.keys())}")
    print(f"    Severity={meta.get('Severity')!r}, claim={meta.get('claim')!r}")

print("[5] load_dataset_file (real TSV, 2 rows) test...")
test_load_real_tsv()
print("    PASS")

print()
print("=" * 50)
print("  All tsv_loader tests passed!")
print("=" * 50)
