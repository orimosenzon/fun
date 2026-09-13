#!/usr/bin/env python3
"""יצירת תמונות מקומות דרך Azure AI Foundry.

המפתח של Azure יושב בצד השרת בלבד ולעולם לא מגיע לדפדפן, ולכן כל בקשת
תמונה עוברת דרך כאן. הקרדיט של Founders Hub מכסה את משפחת gpt-image;
FLUX ו-Stable נמכרים כפריט Marketplace ולכן הם לא ברשימה.
"""

import base64
import os
import pathlib
import urllib.request

import ai_azure

# לפי סדר העדפה. הראשון שיש לו deployment הוא זה שינצח.
CANDIDATES = [
    "gpt-image-2",
    "gpt-image-1.5",
    "gpt-image-1",
    "dall-e-3",
]

_resolved: str | None = None


class NoImageModel(RuntimeError):
    """אין אף מודל תמונות פרוס ברסורס."""


def _client():
    return ai_azure.client()


def available_model(force: bool = False) -> str | None:
    """שם המודל הפרוס בפועל, או None אם אין אף אחד. התוצאה נשמרת במטמון."""
    global _resolved
    if _resolved and not force:
        return _resolved
    override = os.environ.get("PIPIN_IMAGE_MODEL")
    candidates = [override, *CANDIDATES] if override else CANDIDATES
    client = _client()
    for model in candidates:
        try:
            # גודל לא חוקי בכוונה: אם ה-deployment לא קיים נקבל 404 עוד לפני
            # בדיקת הפרמטרים, ואם הוא קיים נקבל 400 בלי לצייר ובלי לשלם.
            client.images.generate(model=model, prompt="probe", n=1, size="1x1")
        except Exception as e:
            if "DeploymentNotFound" in str(e):
                continue
        _resolved = model
        return model
    return None


def generate(prompt: str, size: str = "1024x1024", model: str | None = None) -> bytes:
    """מחזיר PNG כ-bytes. זורק NoImageModel אם אין מודל פרוס."""
    model = model or available_model()
    if not model:
        raise NoImageModel(
            "אין מודל תמונות פרוס ברסורס של Azure. "
            "הרץ: python3 -c \"import ai_azure; ai_azure.deploy('gpt-image-2')\""
        )
    resp = _client().images.generate(model=model, prompt=prompt, n=1, size=size)
    datum = resp.data[0]
    if getattr(datum, "b64_json", None):
        return base64.b64decode(datum.b64_json)
    return urllib.request.urlopen(datum.url).read()


def generate_to_file(prompt: str, path: str | pathlib.Path,
                     size: str = "1024x1024", model: str | None = None) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(generate(prompt, size=size, model=model))
    return path


if __name__ == "__main__":
    m = available_model()
    print("מודל פרוס:", m or "אין")
    if m:
        out = generate_to_file("a small stone bridge over a misty river at dawn, "
                               "Alan Lee watercolor", "/tmp/pipin_probe.png")
        print("נכתב:", out, out.stat().st_size, "bytes")
