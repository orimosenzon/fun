#!/usr/bin/env python3
"""שרת "מה אמרו במועצה": חיפוש ושאלות על תמלולי ישיבות המליאה.

    .venv/bin/python rag/app.py            # http://localhost:5077

/api/search  חיפוש היברידי, בלי מודל שפה (חינם)
/api/ask     תשובה מנוסחת עם הפניות, דרך Azure (עולה כסף, ולכן מוגבל)
"""
import datetime
import json
import os
import threading
from collections import defaultdict

from flask import Flask, jsonify, request, send_from_directory

from engine import Engine

HERE = os.path.dirname(os.path.abspath(__file__))
QUERY_LOG = os.environ.get("RAG_QUERY_LOG", os.path.join(HERE, "..", "private", "queries.jsonl"))
PER_IP_DAILY = 20    # שאלות ביום לכל כתובת
GLOBAL_DAILY = 300   # תקרה לכל האתר: ~300 שאלות * ~1 סנט = ~3$ ביום לכל היותר

app = Flask(__name__, static_folder=None)
engine = Engine(device=os.environ.get("RAG_DEVICE", "cpu"))
lock = threading.Lock()
usage = {"day": None, "total": 0, "per_ip": defaultdict(int)}


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
    log_query("ask", q, {"usage": res["usage"]})
    return jsonify(res)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5077)), threaded=True)
