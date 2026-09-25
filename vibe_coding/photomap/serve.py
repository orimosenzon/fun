#!/usr/bin/env python3
"""שרת מקומי למפת התמונות: הדף עצמו + מדיה שנקראת ישירות מתוך ה-zip-ים של ה-Takeout.

שימוש:
    python3 serve.py [פורט]        (ברירת מחדל 8797)
    python3 serve.py --set-password

/media/thumb/<id>.jpg    תמונה ממוזערת 300x300. נוצרת בפעם הראשונה ונשמרת במטמון
/media/poster/<id>.jpg   פריים מתוך סרטון
/media/view/<id>         המקור: תמונה כמו שהיא, או סרטון עם תמיכה ב-Range

כשיש סיסמה (ראו auth.py), כל בקשה צריכה עוגיית כניסה, והשרת מגיש רק את מה שהאפליקציה צריכה.

סרטון שהדפדפן לא יודע לנגן (3gp ישן, HEVC וכו') עובר המרה ל-H.264 בצפייה הראשונה.
תהליך רקע מכין מראש תמונות ממוזערות לכל הפריטים הממוקמים.
"""
import getpass
import io
import json
import posixpath
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageOps

import auth

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
VIDEO_CACHE_BYTES = 8 << 30
PLAYABLE = {"h264", "vp8", "vp9", "av1"}
# רק אלה נגישים מבחוץ. כל השאר (auth.json, takeout_members.json, קוד) מחזיר 404
PUBLIC_FILES = {"/", "/index.html", "/app.js", "/style.css", "/data/library.json", "/data/trips.json"}
PUBLIC_DIRS = ("/data/thumbs/", "/data/large/", "/media/")
TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif",
         ".webp": "image/webp", ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/mp4"}

for sub in ("thumbs", "posters", "video"):
    (CACHE / sub).mkdir(parents=True, exist_ok=True)

_members, _members_mtime = {}, 0
_local = threading.local()
_locks, _locks_guard = {}, threading.Lock()


def members():
    global _members, _members_mtime
    p = DATA / "takeout_members.json"
    if p.exists() and p.stat().st_mtime != _members_mtime:
        _members, _members_mtime = json.loads(p.read_text()), p.stat().st_mtime
    return _members


def lock_for(key):
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def zopen(iid):
    zp, member, kind = members()[iid][:3]
    zs = getattr(_local, "zips", None)
    if zs is None:
        zs = _local.zips = {}
    if zp not in zs:
        zs[zp] = zipfile.ZipFile(zp)
    return zs[zp], member, kind


def ext_of(iid):
    return Path(members()[iid][1]).suffix.lower()


# ---------- תמונות ----------

def square(im, path):
    im = ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (300, 300))
    tmp = path.with_suffix(".tmp")
    im.save(tmp, "JPEG", quality=82)
    tmp.replace(path)


def photo_thumb(iid):
    path = CACHE / "thumbs" / f"{iid}.jpg"
    with lock_for(path):
        if not path.exists():
            z, member, _ = zopen(iid)
            im = Image.open(io.BytesIO(z.read(member)))
            im.draft("RGB", (600, 600))       # פענוח JPEG מוקטן, פי כמה מהר יותר
            square(im, path)
    return path


# ---------- סרטונים ----------

def extract(iid, dest):
    z, member, _ = zopen(iid)
    tmp = dest.with_suffix(".part")
    with z.open(member) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)
    tmp.replace(dest)
    return dest


def codec(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return r.stdout.strip()


def make_poster(video_file, iid):
    poster = CACHE / "posters" / f"{iid}.jpg"
    for ss in ("1", "0"):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", ss, "-i", str(video_file), "-frames:v", "1",
                        "-vf", "scale='min(1600,iw)':-2", str(poster)], capture_output=True)
        if poster.exists() and poster.stat().st_size:
            break
    if poster.exists():
        square(Image.open(poster), CACHE / "thumbs" / f"{iid}.jpg")
    return poster


def video_poster(iid):
    poster = CACHE / "posters" / f"{iid}.jpg"
    with lock_for(poster):
        if poster.exists():
            return poster
        cached = CACHE / "video" / f"{iid}{ext_of(iid)}"
        if cached.exists():
            return make_poster(cached, iid)
        with tempfile.TemporaryDirectory(dir=CACHE) as td:
            return make_poster(extract(iid, Path(td) / f"v{ext_of(iid)}"), iid)


def playable_video(iid):
    """קובץ שהדפדפן יודע לנגן, במטמון. מחלץ, ואם צריך ממיר."""
    raw = CACHE / "video" / f"{iid}{ext_of(iid)}"
    play = CACHE / "video" / f"{iid}.play.mp4"
    with lock_for(raw):
        if play.exists():
            return play
        if not raw.exists():
            extract(iid, raw)
        if ext_of(iid) in (".mp4", ".m4v", ".mov") and codec(raw) in PLAYABLE:
            out = raw
        else:
            tmp = play.with_suffix(".part.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-vf", "scale=-2:'min(ih,1080)'",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac",
                            "-movflags", "+faststart", str(tmp)], capture_output=True)
            tmp.replace(play)
            raw.unlink()
            out = play
    trim_video_cache()
    return out


def trim_video_cache():
    files = sorted((f for f in (CACHE / "video").iterdir() if f.is_file()), key=lambda f: f.stat().st_atime)
    total = sum(f.stat().st_size for f in files)
    for f in files[:-1]:
        if total <= VIDEO_CACHE_BYTES:
            break
        total -= f.stat().st_size
        f.unlink(missing_ok=True)


# ---------- HTTP ----------

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = posixpath.normpath(urllib.parse.unquote(self.path.split("?")[0]))
        if path == "/login":
            return self.send_html(auth.login_page(urllib.parse.parse_qs(self.path.partition("?")[2]).get("e", [""])[0]))
        if path == "/logout":
            return self.redirect("/login", auth.logout_header())
        if auth.enabled() and not auth.valid(self.headers.get("Cookie")):
            return self.redirect("/login") if path in ("/", "/index.html") else self.send_error(401)
        if path not in PUBLIC_FILES and not path.startswith(PUBLIC_DIRS):
            return self.send_error(404)
        parts = path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "media":
            return super().do_GET()
        kind, iid = parts[1], parts[2].removesuffix(".jpg")
        if iid not in members():
            return self.send_error(404)
        try:
            if kind == "thumb":
                if members()[iid][2] == "video":
                    video_poster(iid)             # יוצר גם את הממוזערת
                    return self.send_file(CACHE / "thumbs" / f"{iid}.jpg", "image/jpeg")
                return self.send_file(photo_thumb(iid), "image/jpeg")
            if kind == "poster":
                return self.send_file(video_poster(iid), "image/jpeg")
            if kind == "view":
                if members()[iid][2] == "video":
                    return self.send_file(playable_video(iid), "video/mp4")
                z, member, _ = zopen(iid)
                return self.send_bytes(z.read(member), TYPES.get(ext_of(iid), "application/octet-stream"))
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            print(f"! {kind} {iid[:12]}: {e}", flush=True)
            return self.send_error(500)
        self.send_error(404)

    def do_POST(self):
        if self.path != "/login" or not auth.enabled():
            return self.send_error(404)
        if auth.locked():
            return self.redirect("/login?e=locked")
        n = min(int(self.headers.get("Content-Length") or 0), 4096)
        pw = urllib.parse.parse_qs(self.rfile.read(n).decode("utf-8", "replace")).get("password", [""])[0]
        if auth.check(pw):
            return self.redirect("/", auth.cookie_header())
        self.redirect("/login?e=" + ("locked" if auth.locked() else "bad"))

    def redirect(self, where, cookie=None):
        self.send_response(303)
        self.send_header("Location", where)
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def send_html(self, html):
        data = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_bytes(self, data, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path, ctype):
        if not path or not path.exists():
            return self.send_error(404)
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            a, _, b = rng[6:].split(",")[0].partition("-")
            if a:
                start, end = int(a), int(b) if b else size - 1
            else:
                start = max(0, size - int(b))
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def prewarm():
    """מכין מראש תמונות ממוזערות לכל מה שעל המפה."""
    try:
        items = json.loads((DATA / "library.json").read_text())["items"].values()
    except Exception:
        return
    # לפי הסדר הפיזי בתוך ה-zip (הדיסק מכני), קודם תמונות ואחר כך סרטונים
    m = members()
    todo = sorted((i for i in items if i["lat"] is not None and i["id"] in m),
                  key=lambda i: (i["type"] == "video", m[i["id"]][0], m[i["id"]][3]))
    done = 0
    for it in todo:
        if (CACHE / "thumbs" / f"{it['id']}.jpg").exists():
            continue
        try:
            video_poster(it["id"]) if it["type"] == "video" else photo_thumb(it["id"])
            done += 1
        except Exception as e:
            print(f"! prewarm {it['id'][:12]}: {e}", flush=True)
    if done:
        print(f"הכנה מוקדמת הסתיימה: {done} תמונות ממוזערות חדשות", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--set-password"]:
        pw = getpass.getpass("סיסמה: ") if sys.stdin.isatty() else sys.stdin.readline().strip()
        if len(pw) < 4:
            sys.exit("סיסמה קצרה מדי")
        auth.set_password(pw)
        sys.exit("הסיסמה נקבעה. כל הכניסות הקודמות בוטלו.")
    if not auth.enabled():
        print("אזהרה: אין סיסמה. להגדרה: python3 serve.py --set-password", flush=True)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8797
    threading.Thread(target=prewarm, daemon=True).start()
    print(f"http://127.0.0.1:{port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
