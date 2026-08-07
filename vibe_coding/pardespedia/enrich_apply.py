#!/usr/bin/env python3
"""Stage 3 of the enrichment pipeline: actually publish what stage 2 proposed.

Stages 1–2 (enrich_scan.py + the headless drafting agent in cron_enrich.sh) find
new articles and write proposals. They cannot touch the wiki — the drafting stage
runs without Bash on purpose. This script closes the loop, but never unattended:
every entry needs an explicit human approval before it is published.

Why the approval gate: choosing an image is not a mechanical step. Two judgment
calls recur and neither can be automated away — whether the licence/attribution
is honest, and whether the frame contains people (especially children) who did
not consent. A live example: the only good photo of פארק הנינג'ה is its opening
ceremony, packed with identifiable kids. The right answer there was "publish
nothing", and no heuristic would have chosen that.

    # cron does this, read-only against the wiki:
    python3 enrich_apply.py --prepare  enrich_plans/plan-....json

    # you do this:
    python3 enrich_apply.py --list     enrich_plans/plan-....json
    python3 enrich_apply.py --approve  enrich_plans/plan-....json --article "ערך"
    python3 enrich_apply.py --apply    enrich_plans/plan-....json [--dry-run]

Plan format: see enrich_prompt.txt, which tells the drafting agent what to emit.
"""
import argparse
import io
import json
import os
import re
import sys

import requests

from enrich_images import insert_image  # same splice rules as the Commons path
from wiki_client import WikiClient, API_URL

HERE = os.path.dirname(os.path.abspath(__file__))
H = {"User-Agent": "Pardespedia-Bot/1.0 (orimosenzon@gmail.com)"}
CACHE = os.path.join(HERE, "enrich_candidates")
MAX_W = 500          # fair-use copies stay small; that is part of the rationale
MAX_EDITS = 10       # blast radius per run


# --------------------------------------------------------------------------
# plan I/O
# --------------------------------------------------------------------------

def load_plan(path):
    with open(path, encoding="utf-8") as f:
        plan = json.load(f)
    if not isinstance(plan, list):
        sys.exit(f"plan must be a JSON list, got {type(plan).__name__}")
    return plan


def save_plan(path, plan):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)


def cache_path(entry):
    safe = re.sub(r"[^\w֐-׿-]", "_", entry["article"])[:60]
    return os.path.join(CACHE, f"{safe}.img")


# --------------------------------------------------------------------------
# checks (stage 3a — no wiki writes)
# --------------------------------------------------------------------------

def count_faces(path):
    """Faces detected in the candidate, or None if we cannot tell.

    Deliberately over-eager: a false positive costs a human glance, a false
    negative publishes someone's face. cv2 is optional so cron never breaks on
    a missing dependency, but 'unknown' is reported, not silently treated as 0.
    """
    try:
        import cv2
    except ImportError:
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    n = 0
    for name in ("haarcascade_frontalface_default.xml",
                 "haarcascade_frontalface_alt2.xml",
                 "haarcascade_profileface.xml"):
        casc = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, name))
        n = max(n, len(casc.detectMultiScale(gray, 1.05, 4, minSize=(24, 24))))
    return n


def check_video(video_id):
    """Confirm the id resolves, and return its real title/channel.

    The drafting agent works from search snippets and has hallucinated before,
    so we ask YouTube itself rather than trusting the plan.
    """
    r = requests.get("https://www.youtube.com/oembed",
                     params={"url": f"https://www.youtube.com/watch?v={video_id}",
                             "format": "json"},
                     headers=H, timeout=30)
    if r.status_code != 200:
        return None
    d = r.json()
    return {"title": d.get("title", ""), "channel": d.get("author_name", "")}


def prepare(plan, path):
    """Download candidates and record what we found. Never touches the wiki."""
    os.makedirs(CACHE, exist_ok=True)
    for entry in plan:
        art = entry["article"]
        checks = entry.setdefault("checks", {})
        print(f"\n== {art}")

        img = entry.get("image")
        if img and img.get("source_url"):
            dest = cache_path(entry)
            try:
                r = requests.get(img["source_url"], headers=H, timeout=120)
                r.raise_for_status()
                with open(dest, "wb") as f:
                    f.write(r.content)
                from PIL import Image
                with Image.open(dest) as im:
                    checks["image_size"] = list(im.size)
                checks["image_local"] = dest
                checks["image_error"] = None
                faces = count_faces(dest)
                checks["faces"] = faces
                print(f"  image: {checks['image_size']} faces="
                      f"{'unknown' if faces is None else faces}  -> {dest}")
                if faces:
                    print(f"  ⚠ {faces} face(s) detected — needs faces_ok before it can publish")
            except Exception as e:                       # network, 403, bad image
                checks["image_error"] = str(e)
                print(f"  image FAILED: {e}")

        vid = entry.get("video")
        if vid and vid.get("video_id"):
            real = check_video(vid["video_id"])
            checks["video_real"] = real
            if not real:
                print(f"  video {vid['video_id']}: DOES NOT RESOLVE")
            else:
                print(f"  video {vid['video_id']}: {real['title']} | {real['channel']}")
                claimed = (vid.get("title") or "").strip()
                if claimed and claimed not in real["title"] and real["title"] not in claimed:
                    print(f"  ⚠ claimed title differs: {claimed!r}")

    save_plan(path, plan)
    print(f"\nprepared {len(plan)} entr(ies) -> {path}")


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

def gate(entry, kind):
    """Why this image/video may not be published yet. Empty list == clear."""
    item = entry.get(kind)
    if not item:
        return ["nothing proposed"]
    checks = entry.get("checks", {})
    bad = []

    if not entry.get("approved"):
        bad.append("not approved")
    if not item.get("verified"):
        bad.append("drafting stage did not mark it verified")

    if kind == "image":
        if checks.get("image_error"):
            bad.append(f"download failed: {checks['image_error']}")
        elif not checks.get("image_local"):
            bad.append("not prepared yet (run --prepare)")
        if not item.get("filename") or not item.get("caption"):
            bad.append("missing filename/caption")
        if not item.get("page_url"):
            bad.append("missing attribution page_url")
        if item.get("kind") == "commons":
            if not item.get("license"):
                bad.append("commons entry without a licence")
        elif not item.get("rationale"):
            bad.append("fair-use entry without a rationale")
        faces = checks.get("faces")
        if faces and not entry.get("faces_ok"):
            bad.append(f"{faces} face(s) detected, faces_ok not set")
    else:
        real = checks.get("video_real")
        if real is None and "video_real" not in checks:
            bad.append("not prepared yet (run --prepare)")
        elif not real:
            bad.append("video id does not resolve")
        if not item.get("caption"):
            bad.append("missing caption")
    return bad


def has_image(text):
    return bool(re.search(r"\[\[\s*(קובץ|File|תמונה)\s*:", text))


def has_video(text):
    return "<youtube" in text


# --------------------------------------------------------------------------
# publishing
# --------------------------------------------------------------------------

def file_exists(client, filename):
    r = client.session.get(API_URL, params={
        "action": "query", "titles": f"קובץ:{filename}",
        "prop": "imageinfo", "format": "json",
    })
    return "missing" not in next(iter(r.json()["query"]["pages"].values()))


def describe(item):
    """The file page: what it shows, where it came from, under what terms."""
    if item.get("kind") == "commons":
        lic = item.get("license", "")
        if item.get("license_url"):
            lic = f"{lic} ([{item['license_url']} הרישיון])"
        return (f"{item['caption']}\n\n"
                f"מקור: [[commons:{item.get('commons_title','')}|ויקישיתוף]].\n\n"
                f"צלם/יוצר: {item.get('author') or 'לא צוין'}.\n\n"
                f"רישיון: {lic}.\n\n"
                "[[קטגוריה: תמונות מוויקישיתוף]]\n")
    return (f"{item['caption']}\n\n"
            f"מקור: [{item['page_url']} {item.get('source_name') or item['page_url']}].\n\n"
            f"{item['rationale']}\n\n"
            "[[קטגוריה:תמונות בשימוש הוגן]]\n")


def upload(client, entry, dry_run):
    item, checks = entry["image"], entry["checks"]
    fn = item["filename"]
    if file_exists(client, fn):
        print(f"  upload skipped (exists): {fn}")
        return True
    if dry_run:
        print(f"  [dry-run] would upload {fn} from {item['source_url']}")
        return True

    from PIL import Image
    with Image.open(checks["image_local"]) as im:
        im = im.convert("RGB")
        if im.width > MAX_W:
            im = im.resize((MAX_W, int(im.height * MAX_W / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        data = buf.getvalue()

    r = client.session.post(API_URL, data={
        "action": "upload", "filename": fn,
        "comment": "העשרת ערך חדש: העלאת תמונה",
        "text": describe(item), "token": client._csrf_token(),
        "ignorewarnings": "1", "format": "json",
    }, files={"file": (fn, io.BytesIO(data), "image/jpeg")})
    r.raise_for_status()
    if r.json().get("upload", {}).get("result") == "Success":
        print(f"  UPLOADED: {fn}")
        return True
    print(f"  UPLOAD FAILED: {fn} -> {r.json()}", file=sys.stderr)
    return False


# Where a video goes when the plan does not name a section: above the
# back-matter, so it reads as part of the body rather than an afterthought.
BACK_MATTER = ["== ראו גם ==", "== קישורים חיצוניים ==", "== מקורות ==",
               "== הערות שוליים ==", "==ראו גם==", "==קישורים חיצוניים==",
               "==מקורות==", "==הערות שוליים=="]


def insert_video(text, item):
    """Splice the player in. Returns (new_text, note) — note explains a skip."""
    block = (f"=== סרטון ===\n"
             f'<youtube width="300" height="180">{item["video_id"]}</youtube>\n'
             f"{item['caption']}\n")
    if item["video_id"] in text:
        return text, "already placed"

    if item.get("section"):
        i = text.find(item["section"])
        if i == -1:
            return text, f"section not found: {item['section']}"
        j = i + len(item["section"])
        return text[:j] + "\n" + block + text[j:], None

    cuts = [text.find(h) for h in BACK_MATTER]
    cuts = [c for c in cuts if c != -1]
    if cuts:
        i = min(cuts)
        return text[:i] + block + "\n" + text[i:], None

    m = re.search(r"\n\[\[קטגוריה", text)
    if m:
        return text[:m.start()] + "\n" + block + text[m.start():], None
    return text.rstrip() + "\n\n" + block, None


def apply_plan(plan, path, dry_run):
    client = WikiClient()
    client.login()
    edits = done = blocked = 0

    for entry in plan:
        art = entry["article"]
        print(f"\n== {art}")
        if edits >= MAX_EDITS:
            print(f"  stopping: hit the {MAX_EDITS}-edit cap for one run")
            break

        page = client.get_page(art)
        if not page["exists"]:
            print("  ARTICLE MISSING", file=sys.stderr)
            blocked += 1
            continue
        text = page["wikitext"]
        original = text
        summary = []

        for kind in ("image", "video"):
            if not entry.get(kind):
                continue
            why = gate(entry, kind)
            if why:
                print(f"  {kind} blocked: {'; '.join(why)}")
                blocked += 1
                continue
            if kind == "image" and has_image(text):
                print("  image skipped: the article already has one")
                continue
            if kind == "video" and has_video(text):
                print("  video skipped: the article already has one")
                continue

            if kind == "image":
                if not upload(client, entry, dry_run):
                    blocked += 1
                    continue
                text, note = insert_image(text, {
                    "filename": entry["image"]["filename"],
                    "caption": entry["image"]["caption"],
                    "width": entry["image"].get("width", 300),
                    "align": entry["image"].get("align", "שמאל"),
                    "section": entry["image"].get("section"),
                })
                if note:
                    print(f"  image not inserted ({note})")
                    continue
                summary.append("תמונה")
            else:
                text, note = insert_video(text, entry["video"])
                if note:
                    print(f"  video not inserted ({note})")
                    continue
                summary.append("סרטון")

        if text == original:
            continue
        if dry_run:
            print(f"  [dry-run] would edit {art}: {' + '.join(summary)}")
            done += 1
            edits += 1
            continue
        client.edit_page(art, text, summary="עיבוי אוטומטי (באישור ידני): " + " + ".join(summary))
        entry["applied"] = True
        done += 1
        edits += 1

    if not dry_run:
        save_plan(path, plan)
    print(f"\nedited={done} blocked={blocked}")


# --------------------------------------------------------------------------

def show(plan):
    for entry in plan:
        flags = []
        if entry.get("applied"):
            flags.append("APPLIED")
        if entry.get("approved"):
            flags.append("approved")
        if entry.get("faces_ok"):
            flags.append("faces_ok")
        print(f"\n== {entry['article']}  {' '.join(flags)}")
        for kind in ("image", "video"):
            item = entry.get(kind)
            if not item:
                continue
            why = gate(entry, kind)
            state = "READY" if not why else "blocked: " + "; ".join(why)
            label = item.get("filename") or item.get("video_id")
            print(f"  {kind:5} {label}  [{state}]")
            if kind == "image" and entry.get("checks", {}).get("faces"):
                print(f"        faces detected: {entry['checks']['faces']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("--prepare", action="store_true", help="download + check candidates (no wiki writes)")
    ap.add_argument("--list", action="store_true", help="show proposals and why each is or is not ready")
    ap.add_argument("--approve", action="store_true", help="mark entries approved for publishing")
    ap.add_argument("--faces-ok", action="store_true", help="with --approve: accept the detected faces")
    ap.add_argument("--article", action="append", default=[], help="limit --approve to these articles")
    ap.add_argument("--all", action="store_true", help="with --approve: every entry in the plan")
    ap.add_argument("--apply", action="store_true", help="publish approved entries")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = load_plan(args.plan)

    if args.prepare:
        prepare(plan, args.plan)
    elif args.approve:
        if not args.article and not args.all:
            sys.exit("--approve needs --article <name> (repeatable) or --all")
        n = 0
        for entry in plan:
            if args.all or entry["article"] in args.article:
                entry["approved"] = True
                if args.faces_ok:
                    entry["faces_ok"] = True
                n += 1
                print(f"approved: {entry['article']}")
        save_plan(args.plan, plan)
        print(f"{n} entr(ies) approved")
    elif args.apply:
        apply_plan(plan, args.plan, args.dry_run)
    else:
        show(plan)


if __name__ == "__main__":
    main()
