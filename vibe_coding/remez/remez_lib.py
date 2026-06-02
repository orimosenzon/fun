"""Shared utilities for remez."""
import base64
import io
import json
import os
import re
from pathlib import Path

from PIL import Image, ImageOps
from anthropic import Anthropic

MODEL = "claude-opus-4-8"
MAX_TOKENS = 4096
CLAUDE_MAX_DIM = 2000


def load_image_normalized(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def normalize_for_claude(img: Image.Image, max_dim: int = CLAUDE_MAX_DIM) -> Image.Image:
    """Resize image so its longest edge is `max_dim`. Coordinates returned by Claude
    are in this resized space, so all crops must also happen on this normalized image."""
    if max(img.size) <= max_dim:
        return img
    scale = max_dim / max(img.size)
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)


def image_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_block(img: Image.Image) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": image_to_b64(img),
        },
    }


def extract_json(text: str) -> dict | list:
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    raw = fence.group(1) if fence else text
    start = raw.find("{")
    if start == -1:
        start = raw.find("[")
    end = max(raw.rfind("}"), raw.rfind("]"))
    return json.loads(raw[start:end + 1])


def client() -> Anthropic:
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def crop_line(img: Image.Image, y_top: int, y_bottom: int, pad: int = 20) -> Image.Image:
    h = img.height
    y0 = max(0, y_top - pad)
    y1 = min(h, y_bottom + pad)
    return img.crop((0, y0, img.width, y1))


def crop_word(img: Image.Image, box: dict, pad: int = 10) -> Image.Image:
    """box: {x, y, w, h} in img's coordinate space."""
    x0 = max(0, int(box["x"]) - pad)
    y0 = max(0, int(box["y"]) - pad)
    x1 = min(img.width, int(box["x"] + box["w"]) + pad)
    y1 = min(img.height, int(box["y"] + box["h"]) + pad)
    return img.crop((x0, y0, x1, y1))


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate via Levenshtein distance."""
    ref = reference.strip()
    hyp = hypothesis.strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    m, n = len(ref), len(hyp)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n] / len(ref)
