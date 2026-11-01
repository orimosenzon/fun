#!/usr/bin/env python3
""""שפ"ע במספרים": פסולת, גזם, גינון, שטח ציבורי, תנועה והוצאה לתושב, בעמוד אחד.

נולד מרשימת שאלות של פעיל ועדי שכונות (1/11/2026). כל שאלה מקבלת את המספר הכי טוב שיש לו מקור
פתוח, ולצידו מה חסר ומי מחזיק בו. המקורות:

- **ספרי התקציב של המועצה** (web/data/budget_<שנה>.json, series.json): מחלקות אסוף אשפה, ניקוי רחובות,
  גנים ונטיעות ומינהל שפ"ע, ושלוש שורות: פינוי גזם בקבלנות, גיזום עצים ומים לגנים ציבוריים.
  "מתוכנן" = הספר של אותה שנה. "צפי ביצוע" = עמודת השנה הקודמת בספר של השנה שאחריה.
- **קובצי הנתונים לעיבוד של הלמ"ס** (src/benchmark/cbs/libud_*): הוצאה לנפש, פסולת לנפש ומחזור,
  מים לגינון ציבורי, כלי רכב, תאונות דרכים, ושימושי קרקע (שלושה סקרים: 2013, 2021, 2024).
- **דוח מבקר המועצה על 2022**, כפי שהוצג במליאה ב-6/3/2025 (פרוטוקול באתר המועצה): הטונות היחידים
  של אשפה ביתית ושל גזם שפורסמו. מוזן ידנית, עם קישור.
- **ספירות תנועה** (data.gov.il, "sfirot", src/shefa/sfirot.json): איפה ומתי נספרו כלי רכב בתוך גבול
  השיפוט (src/shefa/boundary_itm.json, govmap muni_il). הערכים עצמם בקבצים שדורשים התחברות.
- **שטחים ציבוריים בתוכניות** (../../dereh_kitzur/web/data/public.json): דונמים של שצ"פ ומבני ציבור
  לפי ייעודי הקרקע של התוכניות בתוקף (Xplan).

    python3 tools/build_shefa.py   ->   web/data/shefa.json
"""
import glob
import json
import os
import re
import statistics

import openpyxl

from build_benchmark import headers, is_code, norm, num

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "web", "data")
OUT = os.path.join(DATA, "shefa.json")
PUBLIC = os.path.join(ROOT, "..", "..", "dereh_kitzur", "web", "data", "public.json")
HOME = "פרדס חנה-כרכור"

# שדה: (ביטוי בכותרת המשורשרת, גיליון). אותו רעיון כמו ב-build_benchmark.
FIELDS = {
    "pop": (r"^סה\"כ ?אוכלוסייה בסוף השנה", "phys"),
    "status": (r"^מעמד מוניציפלי", "phys"),
    "waste": (r"פסולת.*ליום לנפש", "phys"),
    "recycle": (r"אחוז מחזור", "phys"),
    "water_garden": (r"צריכת מים ביתית \(אלפי מ\"ק\).*לגינון ציבורי$", "phys"),
    "vehicles": (r"כלי רכב מנועיים סה\"כ", "phys"),
    "private": (r"כלי רכב פרטיים סה\"כ", "phys"),
    "accidents": (r"תאונות דרכים עם נפגעים סה\"כ", "phys"),
    "per_res": (r"הוצאה לנפש בתקציב הרגיל", "budget"),
    "expense": (r"^סה\"כ הוצאות בתקציב הרגיל", "budget"),
}
# שימושי קרקע: שטח בקמ"ר לכל קטגוריה, וסך שטח השיפוט
LAND = {
    "total": r"שטח (ה)?שיפוט \| בקמ\"ר",
    "residential": r"מגורים \| שטח בקמ\"ר",
    "education": r"חינוך והשכלה \| שטח בקמ\"ר",
    "welfare": r"בריאות ורווחה \| שטח בקמ\"ר",
    "public": r"שירותים ציבוריים \| שטח בקמ\"ר",
    "culture": r"תרבות, פנאי, נופש וספורט \| שטח בקמ\"ר",
    "parks": r"גינון לנוי,? פארק ציבורי \| שטח בקמ\"ר",
    "forest": r"יער וחורש \| שטח בקמ\"ר",
}

# הצירוף בעמודת המים: "צריכת מים ביתית (אלפי מ"ק) | למגורים" ואחריו עמודות שכותרתן רק "לגינון ציבורי".
# לכן מחפשים את העמודה הראשונה "לגינון ציבורי" אחרי עמודת "צריכת מים ביתית (אלפי מ"ק)".
WATER_ANCHOR = r"צריכת מים ביתית \(אלפי מ\"ק\)"


def land_year(hdr, col, file_year):
    """שנת הסקר של שימושי הקרקע: מופיעה בכותרת (2013, 2021), ובקובץ 2024 לא (שכבת 2024)."""
    m = re.findall(r"\b(20\d\d)\b", hdr[col])
    return int(m[-1]) if m else file_year


def read_cbs(path, year):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheets = {"phys": next(n for n in wb.sheetnames if n.startswith("נתונים פיזיים")),
              "budget": next((n for n in wb.sheetnames if n.startswith("נתוני תקציב")), None)}
    out = {}
    for kind, sn in sheets.items():
        if not sn:
            continue
        rows = [list(r) for r in wb[sn].iter_rows(values_only=True)]
        hdr = headers(rows)
        cols = {}
        for f, (rx, where) in FIELDS.items():
            if where != kind or f == "water_garden":
                continue
            hit = [c for c, h in enumerate(hdr) if re.search(rx, h)]
            if hit:
                cols[f] = hit[0]
        land = {}
        if kind == "phys":
            anchor = next((c for c, h in enumerate(hdr) if re.search(WATER_ANCHOR, h)), None)
            if anchor is not None:
                cols["water_garden"] = next(c for c in range(anchor, len(hdr)) if hdr[c].endswith("לגינון ציבורי"))
            for f, rx in LAND.items():
                hit = [c for c, h in enumerate(hdr) if re.search(rx, h)]
                if hit:
                    land[f] = hit[0]
            ly = land_year(hdr, land["total"], year) if "total" in land else None
        for r in rows[4:]:
            if not r or len(r) < 2 or not r[0] or not is_code(r[1]):
                continue
            rec = out.setdefault(norm(r[0]), {"name": str(r[0]).strip()})
            for f, c in cols.items():
                v = r[c] if c < len(r) else None
                rec[f] = v.strip() if f == "status" and isinstance(v, str) else num(v)
            if land:
                rec["land"] = {"year": ly, **{f: num(r[c]) for f, c in land.items() if c < len(r)}}
    return out


def cbs_rows(peers):
    rows, land = [], {}
    for path in sorted(glob.glob(os.path.join(ROOT, "src", "benchmark", "cbs", "libud_*.xlsx"))):
        y = int(re.search(r"(\d{2,4})\.xlsx$", path).group(1))
        y = y + 2000 if y < 100 else y
        if y < 2016:
            continue
        for r in read_cbs(path, y).values():
            if r["name"] != HOME and r["name"] not in peers:
                continue
            if not r.get("pop"):
                continue
            pop = r["pop"] * (1000 if r["pop"] < 1000 else 1)  # 2016–2017: באלפים
            rec = {"y": y, "n": r["name"], "pop": round(pop)}
            if r.get("per_res"):
                rec["per_res"] = round(r["per_res"])
            if r.get("expense"):
                rec["expense"] = round(r["expense"])  # אלפי ש"ח
            if r.get("waste"):
                rec["kg"] = round(r["waste"], 2)
                rec["tons"] = round(r["waste"] * 365 * pop / 1000)
            if r.get("recycle") is not None:
                rec["rec"] = round(r["recycle"], 1)
            if r.get("water_garden") is not None:
                rec["water_garden"] = round(r["water_garden"], 1)  # אלפי מ"ק
                rec["water_garden_res"] = round(r["water_garden"] * 1000 / pop, 2)  # מ"ק לתושב
            for k in ("vehicles", "private", "accidents"):
                if r.get(k) is not None:
                    rec[k] = round(r[k])
            if r.get("private"):
                rec["private_k"] = round(r["private"] * 1000 / pop)
            if r.get("accidents") is not None:
                rec["acc_10k"] = round(r["accidents"] * 10000 / pop, 1)
            rows.append(rec)
            lu = r.get("land")
            if lu and lu.get("total") and lu["year"] not in land.get(r["name"], {}):
                land.setdefault(r["name"], {})[lu["year"]] = lu
    return rows, land


def budget_lines():
    """מחלקות ושורות של שפ"ע לאורך השנים: מתוכנן וצפי ביצוע, בש"ח."""
    s = json.load(open(os.path.join(DATA, "series.json"), encoding="utf-8"))
    years = s["years"]
    depts = {}
    for key, label in (("אסוף אשפה", "איסוף אשפה (כולל גזם)"), ("ניקוי רחובות", "ניקוי רחובות"),
                       ("גנים ונטיעות", "גנים ונטיעות"), ('מינהל שפ"ע', 'מינהל שפ"ע')):
        n = s["nodes"][f"e|P|{key}"]
        depts[key] = {"label": label, "plan": n["v"], "actual": n["f"]}
    total = s["nodes"]["e"]["v"]

    codes = {"1712300751": "פינוי גזם וגרוטאות (קבלן)", "1746001750": "גיזום עצים (קבלן)",
             "1746000432": "מים לגנים ציבוריים"}
    lines = {c: {"label": l, "plan": [], "actual": [None] * len(years)} for c, l in codes.items()}
    for i, y in enumerate(years):
        b = json.load(open(os.path.join(DATA, f"budget_{y}.json"), encoding="utf-8"))
        items = {it["code"]: it for d in b["domains"] for dp in d["depts"] for it in dp["items"]
                 if it["section"] == "expense"}
        for c in codes:
            it = items.get(c)
            lines[c]["plan"].append(it["p"] if it else None)
            # צפי הביצוע של השנה הקודמת, מתוך הספר של השנה הזו
            if it and i > 0 and it.get("f_prev"):
                lines[c]["actual"][i - 1] = it["f_prev"]
    return {"years": years, "total": total, "depts": depts, "lines": lines}


def inside(poly, x, y):
    c = False
    for part in poly:
        for ring in part:
            for (x1, y1), (x2, y2) in zip(ring, ring[1:]):
                if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                    c = not c
    return c


def traffic_counts():
    poly = json.load(open(os.path.join(ROOT, "src", "shefa", "boundary_itm.json")))
    out = []
    for s in json.load(open(os.path.join(ROOT, "src", "shefa", "sfirot.json"), encoding="utf-8")):
        if inside(poly, s["X"], s["Y"]):
            out.append({"date": s["START_DATE"][:10], "name": s["COUNT_NAME"].replace('""', '"'),
                        "by": s["CUSTOMER"], "hours": f'{s["START_TIME"][:5]}–{s["END_TIME"][:5]}',
                        "id": s["COUNT_ID"]})
    return sorted(out, key=lambda r: r["date"], reverse=True)


def statutory_public():
    d = json.load(open(PUBLIC, encoding="utf-8"))
    return {"dunam": d["stats"]["dunam"], "updated": d["updated"][:10], "source": d["source"]}


def main():
    peers = json.load(open(os.path.join(DATA, "benchmark.json"), encoding="utf-8"))["peers"]
    rows, land = cbs_rows(set(peers))
    years = sorted({r["y"] for r in rows})

    def med(y, key):
        vals = [r[key] for r in rows if r["y"] == y and r["n"] != HOME and r.get(key) is not None]
        return round(statistics.median(vals), 2) if len(vals) >= 4 else None

    keys = ("per_res", "kg", "rec", "water_garden_res", "private_k", "acc_10k")
    summary = {y: {k: med(y, k) for k in keys} for y in years}

    # שטח ציבורי בתוכניות, למ"ר לתושב לפי האוכלוסייה האחרונה
    pop = max((r for r in rows if r["n"] == HOME), key=lambda r: r["y"])["pop"]
    stat = statutory_public()
    stat["pop"] = pop

    audit = {
        "year": 2022, "presented": "2025-03-06",
        "url": "https://www.pardes-hanna-karkur.muni.il/wp-content/uploads/2025/03/1742804192.1746.pdf",
        "pages": "16–24",
        "household_tons": 16000, "household_priced_tons": 17000, "household_paid_2023": 14_000_000,
        "yard_tons": 11000, "yard_cost": 6_500_000,
    }

    json.dump({"home": HOME, "peers": peers, "years": years, "rows": rows, "summary": summary,
               "land": land, "budget": budget_lines(), "audit": audit, "traffic": traffic_counts(),
               "statutory": stat},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
    for r in rows:
        if r["n"] == HOME:
            print(r)
    print("שימושי קרקע:", json.dumps(land.get(HOME), ensure_ascii=False))
    print("ספירות תנועה בתוך הגבול:", len(json.load(open(OUT))["traffic"]))


if __name__ == "__main__":
    main()
