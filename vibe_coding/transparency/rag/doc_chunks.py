"""קטעי חיפוש מהמסמכים הכתובים, לצד קטעי ההקלטות שבונה build_index.

שני מקורות:
- פרוטוקולים של הוועדה המקומית לתכנון ובנייה (tools/fetch_vaada.py): ועדת משנה, מליאת
  הוועדה, רשות רישוי. כל פרוטוקול בנוי מסעיפים, סעיף לכל בקשה או תוכנית, ולכן הקטעים
  נחתכים לפי סעיף. לכל קטע כותרת קצרה (מספר הסעיף, הבקשה או התוכנית, כתובת, גוש וחלקה),
  כדי שחיפוש של כתובת או של מספר תוכנית ינחת על הסעיף הנכון גם מעמוד ההמשך שלו.
- מסמכי המועצה מ-goinfo (tools/fetch_goinfo.py): פרוטוקולים, תמלולים וסדרי יום של המליאה,
  ופרוטוקולים של ועדות המועצה.
- פרוטוקולי המליאה מהארכיון של המועצה (tools/fetch_council_protocols.py), עם OCR: פרוטוקולים
  ש-goinfo לא מחזיק, ועמוד ההצבעות הסרוק שבסוף כל פרוטוקול דיגיטלי.

כל קטע: date, src ("planning" / "council_doc"), body, kind, title, url, page, head, text.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PRIV = os.path.join(HERE, "..", "private")
MAX_CHARS = 1400
OVERLAP_CHARS = 200

# שורות שחוזרות בכל עמוד ולא אומרות כלום
NOISE = re.compile(r"^(עמוד\s*\d+|פרוטוקול (לישיבת )?ישיבת .{0,40}מס(פר|')\s*\d+.*|תאריך\s*:.*|ת\. עברי.*)$")
SECTION = re.compile(r"^סעיף\s*(\d+)$")
ENTITY = re.compile(r"^(בקשה להיתר|תכנית מתאר מקומית|תכנית מפורטת|תוכנית|תכנית|תשריט חלוקה|דיון כללי|"
                    r"היתר|בקשה לשימוש חורג)\s*:?\s*(\S*)")


def spacing(s):
    """pdftotext מדביק מספר למילה שאחריו ("4מבנים", "בגוש10072"). שם תוכנית כמו ש17/ נשאר."""
    s = re.sub(r"(\d)([א-ת]{2})", r"\1 \2", s)
    return re.sub(r"([א-ת]{2})(\d)", r"\1 \2", s)


def windows(text, size=MAX_CHARS, overlap=OVERLAP_CHARS):
    """חלונות של עד size תווים, שנחתכים בסוף שורה ככל האפשר."""
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):
            k = text.rfind("\n", i + size // 2, j)
            j = k if k > 0 else j
        out.append(text[i:j].strip())
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return [w for w in out if len(w) > 40]


def after(lines, label):
    for i, l in enumerate(lines):
        if l.startswith(label) and i + 1 < len(lines):
            rest = l[len(label):].strip(" :")
            val = (rest or lines[i + 1]).strip(" :,")
            # שדה ריק: השורה הבאה היא כבר התווית של השדה הבא ("גוש וחלקה:")
            nxt = lines[i + 1].strip()
            is_label = nxt.endswith(":") and len(nxt) < 20 and not re.search(r"\d", nxt)
            return "" if is_label and not rest else val
    return ""


def tidy(s):
    # "בקשה להיתר20220425 :" -> "בקשה להיתר 20220425"
    s = re.sub(r"([א-ת])(\d)", r"\1 \2", s)
    return re.sub(r"\s*:\s*$|^\s*:\s*", "", s).strip(" :")


def section_head(lines, n):
    ent = next((tidy(l) for l in lines[:12] if ENTITY.match(l) and not l.startswith("תיק")), "")
    parts = [f"סעיף {n}", ent]
    name = after(lines, "שם התכנית")
    addr = after(lines, "כתובת")
    gush = after(lines, "גוש וחלקה")
    what = after(lines, "מהות הבקשה") or after(lines, "מטרת התכנית") or after(lines, "מהות")
    for p in (name, addr, gush):
        if p:
            parts.append(tidy(p)[:90])
    if what and what not in name:
        parts.append(tidy(what)[:110])
    return " · ".join(p for p in parts if p)


def planning_chunks():
    path = os.path.join(PRIV, "vaada", "meetings.json")
    if not os.path.exists(path):
        return []
    out = []
    for m in json.load(open(path, encoding="utf-8")):
        tpath = os.path.join(PRIV, "vaada", "text", m["file"] + ".json")
        if not m["date"] or not os.path.exists(tpath):
            continue
        pages = json.load(open(tpath, encoding="utf-8"))
        title = f"{m['body']}, ישיבה {m['number']}"
        kind = {"פרוטוקול": "פרוטוקול דיון והחלטות"}.get(m["kind"], m["kind"])
        base = {"date": m["date"], "src": "planning", "body": m["body"], "kind": kind,
                "title": title, "url": m["url"], "number": m["number"]}
        # צוברים עמודים לסעיפים; עמוד בלי "סעיף N" ממשיך את הסעיף הקודם
        sections, cur = [], {"head": "פתיחה, נוכחים וסדר היום", "pages": []}
        for pno, pg in enumerate(pages, 1):
            lines = [l.strip() for l in pg.split("\n") if l.strip() and not NOISE.match(l.strip())]
            n = next((SECTION.match(l).group(1) for l in lines[:8] if SECTION.match(l)), None)
            if n:
                sections.append(cur)
                cur = {"head": spacing(section_head(lines, n)), "pages": []}
            cur["pages"].append((pno, spacing("\n".join(l for l in lines if not SECTION.match(l)))))
        sections.append(cur)
        for s in sections:
            # חלון אחד יכול לעבור בין עמודים; העמוד שלו הוא העמוד שבו הוא מתחיל
            text, starts = "", []
            for pno, t in s["pages"]:
                starts.append((len(text), pno))
                text += t + "\n"
            pos = 0
            for w in windows(text):
                at = text.find(w[:60], max(0, pos - OVERLAP_CHARS - 5))
                pos = at if at >= 0 else pos
                page = max((p for off, p in starts if off <= pos + 1), default=s["pages"][0][0] if s["pages"] else 1)
                out.append(dict(base, page=page, head=s["head"], text=w))
    return out


# החלטות שאדם קרא בעין מהעמוד האחרון של פרוטוקול, שהוא סריקה בלי טקסט. לשם נופלת לעתים קרובות
# ההצבעה של הישיבה (למשל התקציב לשנת 2026), ולכן בלי זה התשובה היא "לא נמצאה הצבעה". הקובץ
# מתוחזק בפרדספדיה (council_protocols.py), המפתח שלו הוא תאריך + hash.
OVERRIDES = os.path.join(HERE, "..", "..", "pardespedia", "protocol_overrides.json")


def override_chunks(docs):
    if not os.path.exists(OVERRIDES):
        return []
    out = []
    for key, v in json.load(open(OVERRIDES, encoding="utf-8")).items():
        if key == "_" or not (v.get("resolutions") or v.get("note")):
            continue
        date = key[:10]
        prot = next((d for d in docs if d["date"] == date and d["body"] == "מליאת המועצה"
                     and d["kind"] == "פרוטוקול"), None)
        text = "\n".join(v.get("resolutions", []) + ([v["note"]] if v.get("note") else []))
        out.append({"date": date, "src": "council_doc", "body": "מליאת המועצה",
                    "kind": "החלטות מהעמוד הסרוק", "title": prot["title"] if prot else "פרוטוקול",
                    "url": prot["url"] if prot else "", "page": v.get("page"),
                    "head": f"מליאת המועצה: החלטות שהתקבלו בהצבעה (עמוד {v.get('page')} בפרוטוקול, נקרא מהסריקה)",
                    "text": text})
    return out


def council_chunks():
    path = os.path.join(PRIV, "goinfo", "docs.json")
    if not os.path.exists(path):
        return []
    out = []
    docs = json.load(open(path, encoding="utf-8"))
    for d in docs:
        if not d["date"]:
            continue
        for w in windows(d["text"]):
            out.append({"date": d["date"], "src": "council_doc", "body": d["body"], "kind": d["kind"],
                        "title": d["title"], "url": d["url"], "page": None,
                        "head": f"{d['body']}: {d['kind']}", "text": w})
    return out + override_chunks(docs) + council_site_chunks()


SCANNED_KIND = "פרוטוקול (עמוד ההצבעות הסרוק)"


def council_site_chunks():
    """מסמכים שנשמרו עמוד-עמוד: פרוטוקולי המליאה מארכיון המועצה (fetch_council_protocols), מסמכי
    אתר המועצה (fetch_council_media), דוחות הפעילות של הוועדה המקומית (fetch_vaada_reports), והוראות
    תוכניות והחלטות מוסדות התכנון מ-מבא"ת (fetch_plans)."""
    out = []
    for sub in ("council_site", "council_media", "vaada_reports", "plans"):
        path = os.path.join(PRIV, sub, "docs.json")
        if not os.path.exists(path):
            continue
        for d in json.load(open(path, encoding="utf-8")):
            if not d["date"]:
                continue
            scanned = d["kind"] == "עמוד סרוק"
            kind = SCANNED_KIND if scanned else d["kind"]
            base = {"date": d["date"], "src": "council_doc", "body": d["body"], "kind": kind,
                    "title": d["title"], "url": d["url"]}
            head = ("מליאת המועצה: הצבעות והחלטות מהעמוד האחרון של הפרוטוקול (נקרא בזיהוי טקסט מהסריקה)"
                    if scanned else f"{d['body']}: {kind} · {d['title']}")
            for p in d["pages"]:
                for w in windows(p["text"]):
                    out.append(dict(base, page=p["page"], head=head, text=w))
    return out


# רישיונות כריתה מ"יעלה" (tools/fetch_trees.py). רשומה לכל מין עץ בבקשה, ולכן מקבצים לפי בקשה:
# קטע אחד לכל בקשה, עם כל העצים שבה. אין קישור קבוע לרישיון, אז הקישור הוא לדף הפרסומים.
TREES_PAGE = "https://yeela-trees.moag.gov.il/FoPublic/FoLicence"


def tree_chunks():
    path = os.path.join(PRIV, "trees", "licenses.json")
    if not os.path.exists(path):
        return []
    reqs = {}
    for r in json.load(open(path, encoding="utf-8")):
        reqs.setdefault(r["requestId"], []).append(r)
    out = []
    for rid, rows in reqs.items():
        r0, x = rows[0], (rows[0].get("expandRows") or [{}])[0]
        date = (r0.get("approvedDate") or x.get("dateOLicensePublic") or "")[:10]
        if not date:
            continue
        trees = []
        for r in rows:
            acts = [f"{n} {w}" for n, w in ((r.get("unproot"), "לכריתה"), (r.get("copying"), "להעתקה"),
                                            (r.get("conservation"), "לשימור")) if n]
            trees.append(f"{r.get('treeName') or 'עץ'}: {', '.join(acts) or 'בלי פירוט כמות'}")
        where = " · ".join(p for p in (x.get("street"), x.get("block") and f"גוש {x['block']} חלקה {x.get('parcel')}") if p)
        appeal = (r0.get("appealLastDate") or "")[:10]
        lines = [f"רישיון כריתה/העתקה של עצים מס' {r0.get('licenseId')}, {x.get('shortHebDesc', '')}",
                 f"מקום: {where}", f"סיבה: {x.get('requestReason', '')}", "העצים: " + "; ".join(trees),
                 f"סטטוס: {r0.get('licenseStatusDesc', '')}", f"מבקש: {(x.get('customerName') or '').strip()}",
                 f"אישר: {x.get('approvedBy') or ''}"]
        if appeal:
            lines.append(f"מועד אחרון להגשת השגה: {appeal}")
        out.append({"date": date, "src": "trees", "body": "פקיד היערות", "kind": "רישיון כריתה",
                    "title": f"רישיון {r0.get('licenseId')}", "url": TREES_PAGE, "page": None,
                    "head": f"כריתת עצים · {where}", "text": "\n".join(lines)})
    return out


# רשומה לכל תוכנית מ-Xplan (fetch_plans.py): מספר, שם, מטרות, סטטוס ותאריכים. קטע קצר, כדי שחיפוש
# של מספר תוכנית או של מטרה ("תוספת יח"ד", "מבנה ציבור") יגיע לתוכנית גם בלי שהורדו ההוראות שלה.
AUTH = {1: "הוועדה המחוזית", 2: "הוועדה הארצית", 3: "הוועדה המקומית"}


def plan_chunks():
    path = os.path.join(PRIV, "plans", "xplan.json")
    if not os.path.exists(path):
        return []
    import time
    day = lambda ms: time.strftime("%Y-%m-%d", time.gmtime(ms / 1000)) if ms else ""  # noqa: E731
    out = []
    for p in json.load(open(path, encoding="utf-8")):
        date = day(p.get("last_update_date")) or day(p.get("receiving_date"))
        if not date:
            continue
        obj = (p.get("pl_objectives") or "").replace("^", "\n").replace("\r", "")
        lines = [f"תוכנית {p['pl_number']}: {p.get('pl_name') or ''}",
                 f"סטטוס: {p.get('internet_short_status') or ''}; שלב: {p.get('station_desc') or ''}",
                 f"בסמכות: {AUTH.get(int(p['pl_by_auth_of'] or 0), '')}; סוג: {p.get('entity_subtype_desc') or ''}; "
                 f"שטח: {p.get('pl_area_dunam') or '?'} דונם",
                 f"ייעודי קרקע: {p.get('pl_landuse_string') or ''}"]
        for k, label in (("receiving_date", "נקלטה"), ("depositing_date", "הופקדה"),
                         ("pl_last_deposit_date", "מועד אחרון להתנגדות")):
            if p.get(k):
                lines.append(f"{label}: {day(p[k])}")
        lines.append("מטרות: " + obj.strip())
        out.append({"date": date, "src": "council_doc", "body": "מוסדות התכנון", "kind": "תוכנית (Xplan)",
                    "title": f"תוכנית {p['pl_number']}: {p.get('pl_name') or ''}", "url": p.get("pl_url") or "",
                    "page": None, "head": f"תוכנית {p['pl_number']} · {p.get('pl_name') or ''}",
                    "text": "\n".join(lines)[:MAX_CHARS * 2]})
    return out
