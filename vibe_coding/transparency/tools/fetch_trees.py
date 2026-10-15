#!/usr/bin/env python3
"""רישיונות כריתה והעתקה של עצים בפרדס חנה-כרכור, ממערכת "יעלה" של פקיד היערות (משרד החקלאות).

מאז 1/6/2026 כל בקשה לכריתה עוברת ב"יעלה", וכל רישיון מתפרסם שם לשבועיים להגשת השגה לפני
שהוא נכנס לתוקף. ה-API פתוח בלי הרשמה, אבל השרת חוסם בקשות שלא מגיעות מדפדפן (403 גם
לכותרות של כרום), ולכן הבקשות יוצאות מתוך דף של "יעלה" בכרום אמיתי, כמו mavat.py בדרך קיצור.

כל רישיון: מין העץ, כמה עצים לכריתה/העתקה/שימור, הסיבה, כתובת, גוש וחלקה, מי אישר, מועד אחרון
להשגה, ושם המבקש (פרסום רשמי לפי פקודת היערות).

    python3 tools/fetch_trees.py   ->   private/trees/licenses.json
"""
import json
import os

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "private", "trees")
PAGE = "https://yeela-trees.moag.gov.il/FoPublic/FoLicence"
API = "/api/Fo/FOServiceRequest/getFOGridPublicityLicenses"
CITY_ID = 7800  # פרדס חנה-כרכור, סמל הישוב של הלמ"ס

FETCH_JS = """async ([api, city, page]) => {
  const r = await fetch(api, {method: 'POST', headers: {'content-type': 'application/json', accept: 'application/json', userorgroleid: ''},
    body: JSON.stringify({orderDetails: null, pageDetails: {pageNumber: page, pageSize: 100},
      parameters: {zoneId: null, cityId: city, appealLastDate: null, licenseId: null, licenseStatusId: null}})});
  return await r.json();
}"""


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    with sync_playwright() as p:
        # בלי ראש השרת מחזיר 403; החלון נפתח מחוץ למסך
        b = p.chromium.launch(channel="chrome", headless=False, args=["--window-position=-2000,0"])
        pg = b.new_page()
        pg.goto(PAGE, wait_until="networkidle", timeout=90000)
        page = 1
        while True:
            d = pg.evaluate(FETCH_JS, [API, CITY_ID, page])
            rows += d["result"]
            if page >= d["pagination"]["totalPages"]:
                break
            page += 1
        b.close()
    rows = [r for r in rows if r.get("cityId") == CITY_ID]
    json.dump(rows, open(os.path.join(OUT, "licenses.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(rows), "רישיונות", min((r["approvedDate"] for r in rows if r.get("approvedDate")), default=""), "עד",
          max((r["approvedDate"] for r in rows if r.get("approvedDate")), default=""))


if __name__ == "__main__":
    main()
