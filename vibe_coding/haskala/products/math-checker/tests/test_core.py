"""טסטים לפונקציות הטהורות של core.py — בלי קריאות AI ובלי Flask."""
import io
import os
import sys

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402


# ─── resolve_model ────────────────────────────────────────────────────────────

def test_resolve_model_known_key():
    key, label = core.resolve_model("sonnet")
    assert key == "sonnet"
    assert "Sonnet" in label


def test_resolve_model_unknown_falls_back_to_default():
    key, _ = core.resolve_model("no-such-model")
    assert key == core.DEFAULT_MODEL


def test_resolve_model_none_falls_back_to_default():
    key, _ = core.resolve_model(None)
    assert key == core.DEFAULT_MODEL


def test_every_model_key_has_a_provider():
    """כל מפתח ב-MODELS חייב להיות ממופה לספק — אחרת analyze_page נופל לדיפולט בשקט."""
    mapped = (set(core._ANTHROPIC_IDS) | set(core._GEMINI_IDS)
              | set(core._GROQ_IDS) | set(core._AZURE_KEYS))
    assert set(core.MODELS) == mapped


# ─── image utils ──────────────────────────────────────────────────────────────

def _img(w, h):
    return Image.new("RGB", (w, h), (255, 255, 255))


def test_downscale_caps_long_edge():
    out = core._downscale(_img(4400, 2200), 2200)
    assert max(out.size) == 2200
    assert out.size == (2200, 1100)  # שומר יחס


def test_downscale_noop_when_small_enough():
    img = _img(800, 600)
    assert core._downscale(img, 2200) is img


def test_apply_rotation_90_swaps_dimensions():
    out = core.apply_rotation(_img(200, 100), 90)
    assert out.size == (100, 200)


def test_apply_rotation_0_is_noop():
    img = _img(200, 100)
    assert core.apply_rotation(img, 0) is img


def test_jpeg_roundtrip_and_b64():
    b = core._jpeg_bytes(_img(100, 50))
    assert b[:2] == b"\xff\xd8"  # JPEG magic
    s = core._img_to_b64_jpeg(_img(100, 50))
    import base64
    assert base64.standard_b64decode(s)[:2] == b"\xff\xd8"


def test_file_to_pages_png():
    buf = io.BytesIO()
    _img(120, 80).save(buf, format="PNG")
    pages = core.file_to_pages(buf.getvalue(), ".png")
    assert len(pages) == 1
    assert pages[0].size == (120, 80)


def test_file_to_pages_pdf():
    """PDF בן עמוד אחד דרך fitz — מוודא את מסלול ה-PDF כולו."""
    import fitz
    doc = fitz.open()
    doc.new_page(width=200, height=300)
    pdf_bytes = doc.tobytes()
    doc.close()
    pages = core.file_to_pages(pdf_bytes, ".pdf")
    assert len(pages) == 1


# ─── rubric + schema helpers ─────────────────────────────────────────────────

def test_rubric_block_empty():
    assert core._rubric_block("") == ""
    assert core._rubric_block(None) == ""
    assert core._rubric_block("   ") == ""


def test_rubric_block_wraps_text():
    out = core._rubric_block("סעיף א: 5 נק'")
    assert "סעיף א: 5 נק'" in out
    assert "רובריקה" in out


def test_strip_additional_props():
    node = {"type": "object", "additionalProperties": False,
            "items": [{"additionalProperties": False, "x": 1}]}
    out = core._strip_additional_props(node)
    assert "additionalProperties" not in out
    assert "additionalProperties" not in out["items"][0]
    assert out["items"][0]["x"] == 1


def test_gemini_schema_is_clean():
    def walk(n):
        if isinstance(n, dict):
            assert "additionalProperties" not in n
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(core._GEMINI_SCHEMA)


# ─── humanize_error ───────────────────────────────────────────────────────────

def test_humanize_error_json_decode():
    import json
    try:
        json.loads("{bad")
    except json.JSONDecodeError as e:
        out = core.humanize_error(e)
    assert out["message"] == "המודל החזיר תשובה לא תקפה"
    assert "technical" in out


def test_humanize_error_unknown():
    out = core.humanize_error(RuntimeError("boom"))
    assert out["message"] == "תקלה לא צפויה"
    assert "boom" in out["technical"]


# ─── lazy page iteration ─────────────────────────────────────────────────────

def test_count_pages_image():
    buf = io.BytesIO()
    _img(50, 50).save(buf, format="PNG")
    assert core.count_pages(buf.getvalue(), ".png") == 1


def test_count_pages_and_iter_pdf():
    import fitz
    doc = fitz.open()
    for _ in range(3):
        doc.new_page(width=200, height=300)
    pdf = doc.tobytes()
    doc.close()
    assert core.count_pages(pdf, ".pdf") == 3
    pages = list(core.iter_file_pages(pdf, ".pdf"))
    assert len(pages) == 3
    assert all(p.mode == "RGB" for p in pages)
