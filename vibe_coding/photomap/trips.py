#!/usr/bin/env python3
"""זיהוי טיולים בספריית התמונות.

שימוש:
    python3 trips.py [תיקיית-נתונים]      (ברירת מחדל: data/)

קורא את library.json וכותב trips.json. fetch_album.py ו-import_takeout.py מריצים את זה לבד בסוף.

האלגוריתם:
0. ניקוי שיבושי GPS (ראו drop_spoofed). פריט משובש מאבד את המיקום, והמקורי נשמר ב-"spoof".
1. ממיינים את הפריטים הממוקמים לפי זמן, ומפצלים ל"יציאות" בכל פער של יותר מ-6 שעות.
2. "בית" = התא (בערך 1 ק"מ) שמופיע בהכי הרבה ימים שונים, אם הוא מספיק דומיננטי.
   יציאה שמרכזה במרחק של פחות מ-20 ק"מ מהבית היא לא טיול.
3. יציאות רחוקות מהבית שביניהן פחות מ-30 שעות מתאחדות (טיול של כמה ימים). חזרה הביתה סוגרת.
4. סוג: "trip" = כמה ימים או חו"ל, "outing" = יציאה של יום בארץ.
5. שם לפי המקום שבו צולם הכי הרבה, דרך Nominatim של OpenStreetMap. עם מטמון ב-places.json,
   כך שכל נקודה נשלחת רק פעם אחת. הבית עצמו לא נשלח אף פעם.
"""
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

H = 3600 * 1000
SESSION_GAP = 6 * H
TRIP_GAP = 30 * H
AWAY_KM = 20
MIN_ITEMS = 3
HOME_CELL = 0.01          # מעלות, בערך 1 ק"מ
HOME_MIN_DAYS = 5
HOME_MIN_SHARE = 0.2
WIDE_TRIP_KM = 300        # טיול בחו"ל רחב מזה נקרא בשם המדינה

# לאן שיבושי ה-GPS בישראל "מעבירים" טלפונים מאז אוקטובר 2023: נמלי התעופה של ביירות, עמאן וקהיר
SPOOF_POINTS = [(33.8209, 35.4884), (31.7226, 35.9932), (30.1219, 31.4056)]
SPOOF_SINCE = datetime(2023, 10, 1, tzinfo=timezone.utc).timestamp() * 1000
SPOOF_RADIUS_KM = 3
JUMP_KM = 150

UA = "photomap/0.1 (personal photo map)"
TOWN_KEYS = ("city", "town", "village", "hamlet", "suburb")
POI_CATEGORIES = {"tourism", "historic", "natural", "leisure", "waterway", "boundary", "place", "amenity", "building"}
COUNCIL = "מועצה אזורית "


def km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def pos(it):
    return (it["lat"], it["lon"])


def median_point(items):
    lats = sorted(i["lat"] for i in items)
    lons = sorted(i["lon"] for i in items)
    return (lats[len(lats) // 2], lons[len(lons) // 2])


def day(it):
    return datetime.fromtimestamp((it["taken"] + it["tz"]) / 1000, timezone.utc).date()


# ---------- ניקוי ----------

def drop_spoofed(items):
    """מסיר מיקומים משובשים. מחזיר כמה הוסרו."""
    def drop(it):
        it["spoof"] = [it["lat"], it["lon"]]
        it["lat"] = it["lon"] = None

    located = sorted((i for i in items.values() if i["lat"] is not None), key=lambda i: i["taken"])
    n = 0
    for it in located:
        if it["taken"] >= SPOOF_SINCE and any(km(pos(it), p) < SPOOF_RADIUS_KM for p in SPOOF_POINTS):
            drop(it)
            n += 1
    # קפיצה בלתי אפשרית: רחוק מהשכנים משני הצדדים, כשהשכנים עצמם קרובים זה לזה
    located = [i for i in located if i["lat"] is not None]
    for prev, it, nxt in zip(located, located[1:], located[2:]):
        if nxt["taken"] - prev["taken"] > 6 * H:
            continue
        if km(pos(prev), pos(it)) > JUMP_KM and km(pos(it), pos(nxt)) > JUMP_KM and km(pos(prev), pos(nxt)) < 50:
            drop(it)
            n += 1
    return n


# ---------- זיהוי ----------

def find_home(items):
    days = {}
    for it in items:
        cell = (round(it["lat"] / HOME_CELL), round(it["lon"] / HOME_CELL))
        days.setdefault(cell, set()).add(day(it))
    if not days:
        return None
    all_days = len({day(i) for i in items})
    cell, ds = max(days.items(), key=lambda kv: len(kv[1]))
    if len(ds) < HOME_MIN_DAYS or len(ds) < HOME_MIN_SHARE * all_days:
        return None
    return (cell[0] * HOME_CELL, cell[1] * HOME_CELL)


def split(items, gap):
    groups = []
    for it in items:
        if groups and it["taken"] - groups[-1][-1]["taken"] <= gap:
            groups[-1].append(it)
        else:
            groups.append([it])
    return groups


def detect(items, home):
    trips, current = [], None
    for s in split(items, SESSION_GAP):
        if home is not None and km(median_point(s), home) < AWAY_KM:
            current = None          # חזרה הביתה סוגרת את הטיול
        elif current and s[0]["taken"] - current[-1]["taken"] <= TRIP_GAP:
            current += s
        else:
            current = list(s)
            trips.append(current)
    return [t for t in trips if len(t) >= MIN_ITEMS]


# ---------- שמות מקומות ----------

class Places:
    def __init__(self, path):
        self.path = path
        self.cache = json.loads(path.read_text()) if path.exists() else {}
        self.last = 0

    def _get(self, lat, lon, zoom):
        time.sleep(max(0, 1.1 - (time.time() - self.last)))   # מדיניות Nominatim: בקשה לשנייה
        q = urllib.parse.urlencode({"lat": lat, "lon": lon, "zoom": zoom, "format": "jsonv2",
                                    "accept-language": "he,en"})
        try:
            req = urllib.request.Request(f"https://nominatim.openstreetmap.org/reverse?{q}", headers={"User-Agent": UA})
            return json.load(urllib.request.urlopen(req, timeout=30))
        finally:
            self.last = time.time()

    def _cached(self, key, fetch):
        if key not in self.cache:
            try:
                self.cache[key] = fetch()
            except Exception as e:
                print(f"  ! שם מקום ל-{key}: {e}")
                return None
            self.path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=1))
        return self.cache[key]

    def lookup(self, lat, lon):
        """יישוב (או מועצה/אזור אם אין יישוב), מדינה, והאם זה יישוב אמיתי."""
        def fetch():
            a = self._get(lat, lon, 14).get("address", {})
            town = next((a[k] for k in TOWN_KEYS if k in a and not a[k].startswith(COUNCIL)), None)
            area = next((a[k] for k in ("city", "municipality", "region", "county", "state") if k in a), None)
            return {"place": town or area, "town": bool(town), "country": a.get("country")}
        return self._cached(f"{lat:.2f},{lon:.2f}", fetch) or {"place": None, "town": False, "country": None}

    def poi(self, lat, lon):
        """שם של נקודת עניין (מצפור, שמורה, חורבה) ליד הנקודה, או None."""
        def fetch():
            d = self._get(lat, lon, 18)
            return d.get("name") if d.get("category") in POI_CATEGORIES and d.get("name") else None
        return self._cached(f"poi:{lat:.3f},{lon:.3f}", fetch)


def trip_place(trip, places):
    """(שם, מדינה) לפי המקום שצולם בו הכי הרבה, ולא המרכז הגאומטרי שיכול ליפול באמצע הים."""
    cell = lambda i: (round(i["lat"], 1), round(i["lon"], 1))
    found = []
    for c, _ in Counter(map(cell, trip)).most_common(3):     # המוביל עלול ליפול בשטח פתוח
        pt = median_point([i for i in trip if cell(i) == c])
        found.append((pt, places.lookup(*pt)))
        if found[-1][1]["town"]:
            break
    pt, p = next((f for f in found if f[1]["town"]), found[0])
    name = p["place"]
    if not p["town"]:
        # בטבע: עדיף "מצפור הר ברקן" על "מועצה אזורית גלבוע"
        name = places.poi(*pt) or (name or "").removeprefix(COUNCIL) or None
    return name or p["country"] or "טיול", p["country"]


def route_km(trip):
    return sum(km(pos(a), pos(b)) for a, b in zip(trip, trip[1:]))


def build(data_dir):
    data_dir = Path(data_dir)
    lib_path = data_dir / "library.json"
    lib = json.loads(lib_path.read_text())
    dropped = drop_spoofed(lib["items"])
    if dropped:
        lib_path.write_text(json.dumps(lib, ensure_ascii=False))
        print(f"שיבושי GPS: הוסר מיקום מ-{dropped} פריטים")
    items = sorted((i for i in lib["items"].values() if i["lat"] is not None), key=lambda i: i["taken"])
    home = find_home(items)
    places = Places(data_dir / "places.json")
    trips = detect(items, home)
    named = [trip_place(t, places) for t in trips]
    # המדינה של הבית = הנפוצה בטיולים, כדי לא לשלוח את הבית עצמו
    home_country = Counter(c for _, c in named if c).most_common(1)[0][0] if named else None

    out = []
    for t, (name, country) in zip(trips, named):
        abroad = bool(country and home_country and country != home_country)
        dist = route_km(t)
        if abroad:
            name = country if dist > WIDE_TRIP_KM else f"{name}, {country}"
        days = len({day(i) for i in t})
        out.append({
            "id": t[0]["id"],
            "name": name,
            "kind": "trip" if days > 1 or abroad else "outing",
            "start": t[0]["taken"] + t[0]["tz"],
            "end": t[-1]["taken"] + t[-1]["tz"],
            "days": days,
            "count": len(t),
            "km": round(dist),
            "cover": t[len(t) // 2]["id"],
            "items": [i["id"] for i in t],
        })
    out.reverse()   # החדש ראשון
    (data_dir / "trips.json").write_text(json.dumps({"home": bool(home), "trips": out}, ensure_ascii=False, indent=1))
    kinds = Counter(t["kind"] for t in out)
    print(f"טיולים: {kinds['trip']}, יציאות יום: {kinds['outing']}"
          + ("" if home else " (בית לא זוהה עדיין, כל יציאה נחשבת טיול)"))
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent / "data")
