#!/usr/bin/env python3
"""שרת "מה אמרו במועצה": חיפוש ושאלות על תמלולי ישיבות המליאה.

    .venv/bin/python rag/app.py            # http://localhost:5077

/api/search  חיפוש היברידי, בלי מודל שפה (חינם)
/api/ask     תשובה מנוסחת עם הפניות, דרך Azure (עולה כסף, ולכן מוגבל)
/api/share/<id>  תשובה שנשמרה, לקישור ששולחים לאחרים. בלי מודל ובלי חיפוש

כל תשובה נשמרת כקובץ JSON בתיקיית הנתונים (RAG_DATA). ב-Space הדיסק נמחק בכל הפעלה,
ולכן אם מוגדרים HF_TOKEN ו-RAG_DATASET, התיקייה מועלית כל כמה דקות ל-Dataset פרטי,
וקישור שלא נמצא מקומית נמשך משם. יומן השאלות נשמר באותה תיקייה ובאותה דרך.
"""
import datetime
import json
import os
import re
import secrets
import threading
from collections import defaultdict

from flask import Flask, jsonify, request, send_from_directory

from engine import Engine

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("RAG_DATA", os.path.join(HERE, "..", "private", "rag_data"))
DATASET = os.environ.get("RAG_DATASET")   # למשל orimosenzon/phk-council-data
HF_TOKEN = os.environ.get("HF_TOKEN")
# קובץ יומן חדש בכל הפעלה: ההעלאה דורסת קובץ באותו שם, ואחרי הפעלה מחדש היומן המקומי ריק
QUERY_LOG = os.path.join(DATA, "logs", datetime.datetime.now().strftime("queries-%Y%m%d-%H%M%S.jsonl"))
PER_IP_DAILY = 20    # שאלות ביום לכל כתובת
GLOBAL_DAILY = 300   # תקרה לכל האתר: ~300 שאלות * ~1 סנט = ~3$ ביום לכל היותר
SHARE_ID = re.compile(r"^[a-z0-9]{8}$")

os.makedirs(os.path.join(DATA, "logs"), exist_ok=True)
os.makedirs(os.path.join(DATA, "shares"), exist_ok=True)
if DATASET and HF_TOKEN:
    from huggingface_hub import CommitScheduler
    CommitScheduler(repo_id=DATASET, repo_type="dataset", folder_path=DATA, every=3,
                    private=True, token=HF_TOKEN)

app = Flask(__name__, static_folder=None)
engine = Engine(device=os.environ.get("RAG_DEVICE", "cpu"))
lock = threading.Lock()
usage = {"day": None, "total": 0, "per_ip": defaultdict(int)}


def save_share(rec):
    alphabet = "abcdefghijkmnpqrstuvwxyz23456789"   # בלי l/1/o/0 שמתבלבלים
    sid = "".join(secrets.choice(alphabet) for _ in range(8))
    rec = dict(rec, id=sid, t=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(DATA, "shares", sid + ".json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    return sid


def load_share(sid):
    path = os.path.join(DATA, "shares", sid + ".json")
    if not os.path.exists(path) and DATASET and HF_TOKEN:
        from huggingface_hub import hf_hub_download
        try:
            path = hf_hub_download(DATASET, f"shares/{sid}.json", repo_type="dataset", token=HF_TOKEN)
        except Exception:
            return None
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def allow(ip):
    with lock:
        today = datetime.date.today().isoformat()
        if usage["day"] != today:
            usage.update(day=today, total=0, per_ip=defaultdict(int))
        if usage["total"] >= GLOBAL_DAILY:
            return "הגענו למכסת השאלות היומית של האתר. החיפוש הרגיל עדיין עובד, ושאלות יחזרו מחר."
        if usage["per_ip"][ip] >= PER_IP_DAILY:
            return f"אפשר לשאול עד {PER_IP_DAILY} שאלות ביום. החיפוש הרגיל עדיין עובד."
        usage["total"] += 1
        usage["per_ip"][ip] += 1
    return None


def log_query(kind, q, extra=None):
    rec = {"t": datetime.datetime.now().isoformat(timespec="seconds"), "kind": kind, "q": q}
    rec.update(extra or {})
    with lock, open(QUERY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def filters():
    return {"date_from": request.values.get("from") or None,
            "date_to": request.values.get("to") or None}


@app.get("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.get("/api/meta")
def meta():
    return jsonify(meetings=len(engine.meetings), first=engine.meetings[0],
                   last=engine.meetings[-1], chunks=len(engine.chunks))


@app.get("/api/health")
def health():
    """בדיקת הגדרות בלי לחשוף דבר: האם הוגדרו כתובת ומפתח, ואם יש בהם רווח מיותר."""
    ep = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    return jsonify(endpoint_set=bool(ep.strip()), key_set=bool(key.strip()),
                   has_whitespace=(ep != ep.strip()) or (key != key.strip()))


@app.get("/api/search")
def search():
    q = (request.values.get("q") or "").strip()[:300]
    if not q:
        return jsonify(results=[])
    log_query("search", q)
    return jsonify(results=engine.search(q, k=20, **filters()))


@app.post("/api/ask")
def ask():
    q = (request.json or {}).get("q", "").strip()[:500]
    if not q:
        return jsonify(error="כתבו שאלה"), 400
    ip = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0]
    denied = allow(ip)
    if denied:
        return jsonify(error=denied), 429
    try:
        res = engine.answer(q, **{k: (request.json or {}).get(k) for k in ("date_from", "date_to")})
    except Exception as e:  # שגיאה של השירות לא אמורה להפיל את הדף
        app.logger.exception("answer failed")
        return jsonify(error=f"השירות לא הצליח לענות כרגע ({type(e).__name__}). נסו שוב."), 502
    filt = {k: (request.json or {}).get(k) for k in ("date_from", "date_to")}
    res["id"] = save_share({"q": q, **filt, "answer": res["answer"], "sources": res["sources"]})
    log_query("ask", q, {"usage": res["usage"], "id": res["id"]})
    return jsonify(res)


@app.get("/api/share/<sid>")
def share(sid):
    rec = SHARE_ID.match(sid) and load_share(sid)
    if not rec:
        return jsonify(error="הקישור לא נמצא. אולי הוא שגוי, או שהתשובה עוד לא נשמרה."), 404
    return jsonify(rec)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5077)), threaded=True)
