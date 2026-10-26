"""Seed the riddles collection in Firestore (project carmit-riddles) from re.zip and WORDPLAY-1-001.zip.

Since 26/10/2026 the riddles live in Firestore and Ori and Carmit edit them from app/admin.html.
This script loaded the first set. Fixed document ids (seed-NN) make re-running it overwrite
those documents rather than duplicate them, so don't re-run it after edits in the admin page,
or the seed versions will overwrite those edits.

Writes go through the REST API with Ori's gcloud token (a project owner is not bound by the rules).
"""
import base64
import io
import json
import pathlib
import subprocess
import urllib.request
import zipfile

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROJECT = "carmit-riddles"
GCLOUD = pathlib.Path.home() / "google-cloud-sdk/bin/gcloud"
MAX_SIDE = 1280
MAX_B64 = 380_000  # two images must fit together in one 1 MiB Firestore document

OLD = "re.zip"
NEW = "WORDPLAY-1-001.zip"

# (zip, file name inside it, question, hint, answers (the first is the primary one))
SEED = [
    (OLD, "מי אני_איתמר_ילד שמטייל על קירות", "מי אני", "ילד שמטייל על קירות", ["איתמר"]),
    (NEW, "WORDPLAY/הורדה.jfif", "", "תהפכו מספרים לאותיות באנגלית 3 בעברית 4", ["LEO", "אריה", "LEO אריה"]),
    (NEW, "WORDPLAY/איזה מספר.jfif", "", "באנגלית", ["4", "ארבע", "four", "4 ever", "forever"]),
    (NEW, "WORDPLAY/ביטוי לשניים.jfif", "", "", ["ONE ON ONE", "1 on 1", "וואן און וואן"]),
    (NEW, "WORDPLAY/ארוע.jfif", "", "מייל", ["MILESTONE", "מיילסטון", "מילסטון", "אבן דרך"]),
    (NEW, "WORDPLAY/מאסטר.jfif", "", "", ["MASTERPIECE", "מאסטרפיס", "יצירת מופת"]),
    (OLD, "ברכת הצלחה_ בהצלחה_ זה לק.jpg", "ברכת הצלחה", "זה לק", ["GOOD LUCK", "בהצלחה", "גוד לאק"]),
    (OLD, "האות שהופכת את אלוהים לזהב_ L_ היא בעצמה אל.jpg", "האות שהופכת את אלוהים לזהב", "היא בעצמה אל", ["L", "אל"]),
    (NEW, "WORDPLAY/זה כמו עוד חרוזז.jfif", "", "לכתוב את זה", ["HONEY MONEY", "האני מאני"]),
    (OLD, "מי אנחנו_מספרים בינארים_שפה של מחשב.jpg", "מי אנחנו", "שפה של מחשב",
     ["מספרים בינאריים", "מספרים בינארים", "בינארי", "בינאריים", "binary"]),
    (OLD, "לחש קסם_שומשום היפתח_אין מי שלא מכיר.jpg", "לחש קסם", "אין מי שלא מכיר", ["שומשום היפתח", "open sesame"]),
    (NEW, "WORDPLAY/TIME.jfif", "", "", ["TIME IS MONEY", "זמן זה כסף", "time = money"]),
    (NEW, "WORDPLAY/lineage.jfif", "", "", ["LINEAGE", "שושלת", "ליניג'"]),
    (NEW, "WORDPLAY/אל חשש.jfif", "", "", ["HALLOWEEN", "הלווין", "האלווין", "הלואין"]),
    (NEW, "WORDPLAY/הכל בשבילנו.jfif", "", "", ["FOUR ACES", "4 aces", "for us", "all for us", "all four aces"]),
    (NEW, "WORDPLAY/תעשה את זה טוב.jfif", "", "", ["GOOD", "טוב"]),
]


def data_url(raw):
    """Shrink an image the same way the admin page does: longest side 1280, JPEG, under MAX_B64."""
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    for quality in (85, 78, 70, 62, 55, 48, 40):
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        if len(b64) < MAX_B64:
            return "data:image/jpeg;base64," + b64
        im.thumbnail((int(im.width * 0.85), int(im.height * 0.85)))
    raise ValueError("image too large")


def fs_value(v):
    if isinstance(v, bool):
        return {"booleanValue": v}
    if isinstance(v, int):
        return {"integerValue": str(v)}
    if isinstance(v, list):
        return {"arrayValue": {"values": [fs_value(x) for x in v]}}
    if v == "__now__":
        return {"timestampValue": "2026-10-26T00:00:00Z"}
    return {"stringValue": v}


def main():
    token = subprocess.check_output(
        [str(GCLOUD), "auth", "print-access-token", "--account=orimosenzon@gmail.com"], text=True).strip()
    zips = {name: zipfile.ZipFile(ROOT / name) for name in (OLD, NEW)}
    for i, (zname, fname, question, hint, answers) in enumerate(SEED, start=1):
        doc = {
            "image": data_url(zips[zname].read(fname)),
            "question": question,
            "hint": hint,
            "answerType": "text",
            "answers": answers,
            "primary": 0,
            "answerImage": "",
            "by": "בוטי",
            "created": "__now__",
        }
        body = json.dumps({"fields": {k: fs_value(v) for k, v in doc.items()}}).encode()
        url = (f"https://firestore.googleapis.com/v1/projects/{PROJECT}/databases/(default)"
               f"/documents/riddles/seed-{i:02d}")
        req = urllib.request.Request(url, data=body, method="PATCH", headers={
            "Authorization": f"Bearer {token}", "x-goog-user-project": PROJECT,
            "Content-Type": "application/json"})
        urllib.request.urlopen(req).read()
        print(f"seed-{i:02d}  {answers[0]}  ({len(doc['image']) // 1024} KB)")


if __name__ == "__main__":
    main()
