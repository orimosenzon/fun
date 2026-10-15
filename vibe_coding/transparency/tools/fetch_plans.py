#!/usr/bin/env python3
"""תוכניות בניין עיר בפרדס חנה-כרכור, ל"מה אמרו במועצה": רשומה לכל תוכנית מ-Xplan, ומ-מבא"ת
הוראות התוכנית והחלטות מוסדות התכנון (הוועדה המחוזית, הוועדה המקומית, ועדות ערר) עליה.

שלושה שלבים, כל אחד נשמר ומדלג על מה שכבר יש:
1. Xplan (שירות ArcGIS פתוח): כל התוכניות שהשטח שלהן בפרדס חנה, עם מטרות, סטטוס ותאריכים.
2. מבא"ת בכרום: ה-JSON של דף התוכנית, ואז הקבצים עצמם (הוראות התוכנית, ומסמכי ההחלטות). ה-API
   דורש טוקן reCAPTCHA v3 בלתי נראה שהדף מנפיק לעצמו, ולכן הכל קורה מתוך הדף, כמו
   data/merkaz/fetch_docs.py ו-dereh_kitzur/mavat.py. תוכנית ב"סיום טיפול" לא נכנסת לשלב הזה.
3. טקסט: pdftotext, ו-OCR לעמודים סרוקים (תקנונים ישנים), עד MAX_OCR עמודים לקובץ.

    python3 tools/fetch_plans.py   ->   private/plans/{xplan.json, docs.json}
"""
import base64
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "dereh_kitzur"))
import build_plans  # noqa: E402  אותה שאילתה ל-Xplan שמזינה את שכבת התוכניות בדרך קיצור
from fetch_council_protocols import ocr_page, text_pages  # noqa: E402

OUT = os.path.join(HERE, "..", "private", "plans")
OLD_MAVAT = os.path.join(HERE, "..", "data", "merkaz", "mavat")  # מטמון מתחקיר מרכז המושבה
KEY = "6LeUKkMoAAAAAH4UacB4zewg4ult8Rcriv-ce0Db"
MAX_OCR = 30
EMPTY = 40
FIELDS = ["pl_number", "pl_name", "pl_objectives", "internet_short_status", "station_desc",
          "entity_subtype_desc", "pl_url", "pl_area_dunam", "pl_landuse_string", "pl_by_auth_of",
          "depositing_date", "pl_last_deposit_date", "receiving_date", "last_update_date", "mp_id"]
AUTH = {1: "הוועדה המחוזית", 2: "הוועדה הארצית", 3: "הוועדה המקומית"}

FETCH_JS = """async ([key, url]) => {
  const tok = await new Promise(r => grecaptcha.ready(() => grecaptcha.execute(key, {action: 'importantAction'}).then(r)));
  const res = await fetch(url, {headers: {Authorization: tok}});
  const buf = new Uint8Array(await res.arrayBuffer());
  let s = ''; for (let i = 0; i < buf.length; i += 8192) s += String.fromCharCode(...buf.subarray(i, i + 8192));
  return [res.status, btoa(s)];
}"""


def day(ms):
    return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000)) if ms else ""


def xplan():
    doc = build_plans.get(build_plans.XPLAN, {"where": build_plans.AREA, "outFields": ",".join(FIELDS),
                                              "returnGeometry": "false", "f": "json", "resultRecordCount": "2000"})
    plans = [f["attributes"] for f in doc["features"]]
    json.dump(plans, open(os.path.join(OUT, "xplan.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return plans


def wanted_files(d):
    """הוראות התוכנית (הגרסה העדכנית ביותר: מאושרות, ואם אין, להפקדה), ומסמכי ההחלטות."""
    pnum = d["planDetails"]["NUMB"]
    docs = [r for r in (d.get("rsPlanDocs") or []) + (d.get("rsPlanDocsAdd") or [])
            if r.get("FILE_DATA") and "הוראות" in (r.get("DOC_NAME") or "")]
    docs.sort(key=lambda r: ("מאושר" not in r["DOC_NAME"], "חתום" not in r["DOC_NAME"]))
    out = [(int(r["ID"]), r.get("PLAN_ENTITY_DOC_NUM") or r["FILE_DATA"]["edNum"], r["DOC_NAME"],
            r["FILE_TYPE"].strip(), "הוראות")
           for r in docs[:1]]
    for r in d.get("rsDes") or []:
        for suf in ("", "1", "_10"):
            eid = r.get("ENTITY_DOC_ID" + suf)
            if eid:
                out.append((int(eid), "temp-default", f"החלטה {r['MEETING_DATE'].replace('/', '-')}{suf}",
                            (r.get("FILE_TYPE" + suf) or "pdf").strip(), "החלטה"))
    return pnum, out


def mavat(plans):
    from playwright.sync_api import sync_playwright
    os.makedirs(os.path.join(OUT, "mavat"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "files"), exist_ok=True)
    todo = [p for p in plans if p.get("mp_id") and p.get("internet_short_status") != "סיום טיפול"]
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        ctx = b.new_context()
        ctx.route(re.compile(r"govmap\.gov\.il|fonts\.(googleapis|gstatic)"), lambda r: r.abort())
        pg = ctx.new_page()
        for n, p in enumerate(todo, 1):
            mid = str(int(p["mp_id"]))
            jpath = os.path.join(OUT, "mavat", mid + ".json")
            old = os.path.join(OLD_MAVAT, mid + ".json")
            if not os.path.exists(jpath) and os.path.exists(old):
                jpath = old
            d = json.load(open(jpath, encoding="utf-8")) if os.path.exists(jpath) else None
            if d:
                pnum, files = wanted_files(d)
                if all(os.path.exists(fname(pnum, f)) for f in files):
                    continue
            try:
                with pg.expect_response(lambda r: "rest/api/SV4/1?mid=" in r.url, timeout=60000) as info:
                    pg.goto(f"https://mavat.iplan.gov.il/SV4/1/{mid}/310", wait_until="commit", timeout=60000)
                d = info.value.json()
                json.dump(d, open(os.path.join(OUT, "mavat", mid + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
                pg.wait_for_timeout(5000)  # עד שהדף טוען את reCAPTCHA
            except Exception as e:
                print("נכשל דף", p["pl_number"], str(e)[:100], flush=True)
                continue
            pnum, files = wanted_files(d)
            for f in files:
                path = fname(pnum, f)
                if os.path.exists(path):
                    continue
                eid, edn, name, ft, _ = f
                url = f"https://mavat.iplan.gov.il/rest/api/Attacments/?eid={eid}&fn={name}.{ft}&edn={edn}&pn={pnum}"
                try:
                    st, data = pg.evaluate(FETCH_JS, [KEY, url])
                    raw = base64.b64decode(data)
                    if st == 200 and len(raw) > 500:
                        open(path, "wb").write(raw)
                    else:
                        print("קובץ לא ירד", st, pnum, name, flush=True)
                except Exception as e:
                    print("נכשל קובץ", pnum, name, str(e)[:100], flush=True)
                time.sleep(1.5)
            print(f"{n}/{len(todo)} {pnum}: {len(files)} קבצים", flush=True)
            time.sleep(1)
        b.close()


def fname(pnum, f):
    eid, _, name, ft, _ = f
    safe = re.sub(r"[^\w.-]+", "_", f"{pnum}__{name}__{eid}")
    return os.path.join(OUT, "files", f"{safe}.{ft.lower()}")


def plan_pages(pages, pnum):
    """מסמך החלטה הוא לרוב פרוטוקול שלם של ישיבה, עם תוכניות מיישובים אחרים. נשארים העמודים שבהם
    מופיע מספר התוכנית (גם בלי הקידומת או עם רווחים), והעמוד שאחריהם, כי הדיון גולש."""
    tail = pnum.split("-")[-1].lstrip("0")
    rx = re.compile(r"(?<!\d)0*" + r"\s?".join(tail) + r"(?!\d)")
    hit = {i for i, (_, t) in enumerate(pages) if rx.search(t)}
    keep = sorted(hit | {i + 1 for i in hit if i + 1 < len(pages)})
    return [pages[i] for i in keep]


def texts(plans):
    import hashlib
    docs, seen = [], set()
    for p in plans:
        mid = str(int(p["mp_id"])) if p.get("mp_id") else ""
        jpath = next((x for x in (os.path.join(OUT, "mavat", mid + ".json"), os.path.join(OLD_MAVAT, mid + ".json"))
                      if mid and os.path.exists(x)), None)
        if not jpath:
            continue
        d = json.load(open(jpath, encoding="utf-8"))
        pnum, files = wanted_files(d)
        for f in files:
            path = fname(pnum, f)
            if not os.path.exists(path) or not path.endswith(".pdf"):
                continue
            digest = hashlib.md5(open(path, "rb").read()).hexdigest()
            if digest in seen:  # אותו פרוטוקול צורף לכמה תוכניות
                continue
            seen.add(digest)
            try:
                pages = text_pages(path)
            except Exception:
                continue
            got, n_ocr = [], 0
            for i, t in enumerate(pages, 1):
                if len(t.strip()) < EMPTY and n_ocr < MAX_OCR:
                    t = ocr_page(path, i, "p" + os.path.basename(path)[:-4][-60:])
                    n_ocr += 1
                if len(t.strip()) >= EMPTY:
                    got.append((i, t))
            if f[4] == "החלטה":
                got = plan_pages(got, pnum)
            out = [{"page": i, "text": t} for i, t in got]
            if not out:
                continue
            kind = f[4]
            date = day(p.get("last_update_date")) if kind == "הוראות" else \
                "-".join(reversed(re.search(r"(\d\d)-(\d\d)-(\d{4})", f[2]).groups())) if re.search(r"\d\d-\d\d-\d{4}", f[2]) else ""
            docs.append({"date": date, "body": "מוסדות התכנון", "kind": "הוראות התוכנית" if kind == "הוראות" else "החלטת מוסד תכנון",
                         "title": f"תוכנית {p['pl_number']}: {p.get('pl_name') or ''}".strip(),
                         "url": p.get("pl_url") or "", "pages": out})
    json.dump(docs, open(os.path.join(OUT, "docs.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(docs), "מסמכי תוכניות")


def main():
    os.makedirs(OUT, exist_ok=True)
    plans = xplan()
    print(len(plans), "תוכניות ב-Xplan", flush=True)
    if "--no-mavat" not in sys.argv:
        mavat(plans)
    texts(plans)


if __name__ == "__main__":
    main()
