"""קטעי חיפוש מהמסמכים הכתובים, לצד קטעי ההקלטות שבונה build_index.

שני מקורות:
- פרוטוקולים של הוועדה המקומית לתכנון ובנייה (tools/fetch_vaada.py): ועדת משנה, מליאת
  הוועדה, רשות רישוי. כל פרוטוקול בנוי מסעיפים, סעיף לכל בקשה או תוכנית, ולכן הקטעים
  נחתכים לפי סעיף. לכל קטע כותרת קצרה (מספר הסעיף, הבקשה או התוכנית, כתובת, גוש וחלקה),
  כדי שחיפוש של כתובת או של מספר תוכנית ינחת על הסעיף הנכון גם מעמוד ההמשך שלו.
- מסמכי המועצה מ-goinfo (tools/fetch_goinfo.py): פרוטוקולים, תמלולים וסדרי יום של המליאה,
  ופרוטוקולים של ועדות המועצה.

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
    return out + override_chunks(docs)
