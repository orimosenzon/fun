"""Shared utilities for remez."""
import base64
import io
import json
import os
import re
from pathlib import Path

from PIL import Image, ImageOps
from anthropic import Anthropic

# Model registry. Key is what the UI / cache key uses; value is the human
# label shown in the dropdown. Adding a model = one entry + one branch in
# `transcribe()` (or reuse an existing provider branch).
MODELS: dict[str, str] = {
    "claude-opus-4-8": "Claude Opus 4.8 (בתשלום)",
    "gemini-2.5-flash": "Gemini 2.5 Flash (חינם)",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash-Lite (חינם)",
}
DEFAULT_MODEL = "claude-opus-4-8"

# Back-compat for annotate.py / experiment.py which still import MODEL.
MODEL = DEFAULT_MODEL
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


def image_to_jpeg_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


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
    # strip(): an accidentally-pasted trailing newline in the HF Space
    # secret value makes the Authorization header malformed, which the
    # SDK surfaces as a generic "Connection error" — strip defensively.
    return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"].strip())


_gemini_client = None


def _gemini():
    """Lazily built so a missing GEMINI_API_KEY only bites if a Gemini model
    is actually selected — Claude-only flows keep working without the key."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


def is_gemini(model_key: str) -> bool:
    return model_key.startswith("gemini")


def transcribe(parts: list, model_key: str = DEFAULT_MODEL, max_tokens: int = 4000) -> str:
    """Run one transcription call against the chosen model and return the raw response text.

    parts: ordered list of ('image', PIL.Image) and ('text', str) tuples. The provider
    branch converts to the SDK's native multi-modal representation.
    """
    if is_gemini(model_key):
        return _gemini_transcribe(parts, model_key, max_tokens)
    return _claude_transcribe(parts, model_key, max_tokens)


def _claude_transcribe(parts: list, model_key: str, max_tokens: int) -> str:
    content = []
    for kind, val in parts:
        if kind == "image":
            content.append(image_block(val))
        elif kind == "text":
            content.append({"type": "text", "text": val})
        else:
            raise ValueError(f"unknown part kind: {kind}")
    resp = client().messages.create(
        model=model_key,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    return resp.content[0].text


def _gemini_transcribe(parts: list, model_key: str, max_tokens: int) -> str:
    from google.genai import types

    gem_parts = []
    for kind, val in parts:
        if kind == "image":
            gem_parts.append(
                types.Part.from_bytes(data=image_to_jpeg_bytes(val), mime_type="image/jpeg")
            )
        elif kind == "text":
            gem_parts.append(val)
        else:
            raise ValueError(f"unknown part kind: {kind}")

    # Disable thinking for pro (default-on) so the call doesn't burn the
    # tight free-tier RPD on internal CoT we don't need for OCR. Flash
    # variants ignore the field cleanly.
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    resp = _gemini().models.generate_content(
        model=model_key,
        contents=gem_parts,
        config=cfg,
    )
    return resp.text


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
