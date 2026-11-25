"""Claude's side of the "גבר גבר" tab: reads the requests Carmit (or Ori) wrote, and writes the answers.

Goes through the Firestore REST API with Ori's gcloud token (a project owner is not bound by the rules, which let
the page only create requests). Used by tools/gever_cron.sh and by the agent it starts:

  python3 tools/gever.py count              # COUNT=<requests with status 'new'>
  python3 tools/gever.py pending            # the new requests, each with the thread it continues, as JSON;
                                            # screenshots are saved as files and their paths given instead
  python3 tools/gever.py status <id>... --to working|new
  python3 tools/gever.py reply <id> --status done|question --file answer.txt
  python3 tools/gever.py release <id>...    # after a run: what is still 'working' goes back to 'new' for the next
                                            # hour; after MAX_TRIES runs it gets an answer that Ori will look at it
"""
import argparse
import base64
import json
import pathlib
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

PROJECT = "carmit-riddles"
GCLOUD = pathlib.Path.home() / "google-cloud-sdk/bin/gcloud"
DOCS = f"https://firestore.googleapis.com/v1/projects/{PROJECT}/databases/(default)/documents"
SHOTS = pathlib.Path.home() / ".cache/carmit_gever/shots"
STATUSES = ("new", "working", "done", "question")
TRIES = pathlib.Path.home() / ".cache/carmit_gever/tries.json"
MAX_TRIES = 2
STUCK = ("לא הצלחתי לסיים את זה לבד, גם אחרי שני ניסיונות. העברתי את הבקשה לאורי, והוא יבדוק מה קרה. "
         "אם רוצים לנסות שוב, אפשר לכתוב אותה מחדש, אולי במילים אחרות.")


def headers():
    token = subprocess.check_output(
        [str(GCLOUD), "auth", "print-access-token", "--account=orimosenzon@gmail.com"], text=True).strip()
    return {"Authorization": f"Bearer {token}", "x-goog-user-project": PROJECT, "Content-Type": "application/json"}


def call(url, method="GET", body=None):
    req = urllib.request.Request(url, method=method, headers=headers(),
                                 data=json.dumps(body).encode() if body is not None else None)
    return json.loads(urllib.request.urlopen(req).read() or b"{}")


def plain(v):
    if "stringValue" in v: return v["stringValue"]
    if "timestampValue" in v: return v["timestampValue"]
    if "integerValue" in v: return int(v["integerValue"])
    if "booleanValue" in v: return v["booleanValue"]
    if "nullValue" in v: return None
    return v


def load_all():
    out, token = [], ""
    while True:
        page = call(f"{DOCS}/gever?pageSize=300" + (f"&pageToken={token}" if token else ""))
        for d in page.get("documents", []):
            g = {k: plain(v) for k, v in d.get("fields", {}).items()}
            g["id"] = d["name"].rsplit("/", 1)[1]
            out.append(g)
        token = page.get("nextPageToken")
        if not token:
            return sorted(out, key=lambda g: g.get("created") or "")


def shot(g):
    """A screenshot as a file Claude can look at (the data URL itself would only fill the prompt)."""
    data = g.pop("image", "") or ""
    if not data.startswith("data:image/"):
        return
    SHOTS.mkdir(parents=True, exist_ok=True)
    ext = data[len("data:image/"):data.index(";")].replace("jpeg", "jpg")
    path = SHOTS / f"{g['id']}.{ext}"
    path.write_bytes(base64.b64decode(data.split(",", 1)[1]))
    g["screenshot"] = str(path)


def pending():
    all_ = load_all()
    by_id = {g["id"]: g for g in all_}
    out = []
    for g in all_:
        if g.get("status") != "new":
            continue
        thread, cur, seen = [], g, set()
        while cur.get("followUp") in by_id and cur["id"] not in seen:   # the earlier requests it continues
            seen.add(cur["id"])
            cur = by_id[cur["followUp"]]
            thread.insert(0, {k: cur.get(k) for k in ("id", "by", "text", "created", "status", "reply")})
        item = dict(g)
        shot(item)
        item["continues"] = thread
        out.append(item)
    return out


def patch(doc_id, fields):
    mask = "&".join(f"updateMask.fieldPaths={k}" for k in fields)
    body = {"fields": {k: ({"timestampValue": v} if k == "replied" else {"stringValue": v}) for k, v in fields.items()}}
    # currentDocument.exists: a request Carmit withdrew in the meantime is not brought back as a shell.
    call(f"{DOCS}/gever/{doc_id}?{mask}&currentDocument.exists=true", "PATCH", body)


def release(ids):
    tries = json.loads(TRIES.read_text()) if TRIES.exists() else {}
    status = {g["id"]: g.get("status") for g in load_all()}
    for i in ids:
        if status.get(i) != "working":
            tries.pop(i, None)
            continue
        tries[i] = tries.get(i, 0) + 1
        if tries[i] >= MAX_TRIES:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            patch(i, {"reply": STUCK, "status": "question", "replied": now})
            print(f"STUCK {i}")
            tries.pop(i)
        else:
            patch(i, {"status": "new"})
            print(f"RETRY {i} (try {tries[i]})")
    TRIES.parent.mkdir(parents=True, exist_ok=True)
    TRIES.write_text(json.dumps(tries))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("count")
    sub.add_parser("pending")
    s = sub.add_parser("status"); s.add_argument("ids", nargs="+"); s.add_argument("--to", choices=STATUSES, required=True)
    r = sub.add_parser("reply"); r.add_argument("id"); r.add_argument("--status", choices=("done", "question"), required=True)
    r.add_argument("--file", required=True)
    rl = sub.add_parser("release"); rl.add_argument("ids", nargs="*")
    a = ap.parse_args()

    if a.cmd == "count":
        print(f"COUNT={sum(1 for g in load_all() if g.get('status') == 'new')}")
    elif a.cmd == "pending":
        print(json.dumps(pending(), ensure_ascii=False, indent=2))
    elif a.cmd == "status":
        for i in a.ids:
            patch(i, {"status": a.to})
            print(f"{i} -> {a.to}")
    elif a.cmd == "reply":
        text = pathlib.Path(a.file).read_text(encoding="utf-8").strip()
        if not text:
            sys.exit("empty answer, nothing written")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        patch(a.id, {"reply": text, "status": a.status, "replied": now})
        print(f"{a.id} answered ({a.status})")
    elif a.cmd == "release":
        release(a.ids)


if __name__ == "__main__":
    main()
