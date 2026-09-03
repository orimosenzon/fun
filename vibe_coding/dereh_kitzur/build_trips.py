"""הצעות לטיולים: משרשר דרכי קיצור קיימות להליכות שלמות.

למה זה קיים
-----------
במפה יש 62 דרכי קיצור ושבעה קילומטרים וחצי, ועד היום היה עליהן טיול אחד.
דרך קיצור בודדת באורך 130 מטר היא לא משהו שיוצאים אליו מהבית. מה שתושב באמת
שואל הוא "תן לי הליכה של ארבעים דקות", והתשובה לזה היא שרשרת של כמה קיצורים
עם קטעי רחוב ביניהם. הכלי הזה מייצר את השרשראות האלה.

למה צריך ניתוב ולא רק חיבור נקודות
----------------------------------
נמדד 2/9/2026: רק 9 זוגות קצוות מתוך 124 נמצאים במרחק 25 מטר זה מזה, שזה הסף
שהעורך באפליקציה מרשה לשרשר בלעדיו. החציון הוא 67 מטר. כלומר הרשת כמעט מחוברת
אבל כמעט אף פעם לא נוגעת, ובין קיצור לקיצור צריך ללכת קצת ברחוב. חיבור בקו ישר
היה מעביר את המסלול דרך חצרות אחוריות, ולכן קטעי הקישור מנותבים על רשת הרחובות
של OpenStreetMap.

מה נשמר
-------
טיול נשמר כמתכון ולא כקו, בדיוק כמו טיול שנבנה באפליקציה:

    parts: [ {trail:'p52'}, {draw:[[lat,lng],…]}, {trail:'p13', reversed:true} ]

כך שתיקון גאומטריה של דרך קיצור מתקן כל טיול שעובר בה. הקטעים המנותבים הם
`draw`, בדיוק כמו קטע שמישהו היה מצייר באצבע.

שימוש
-----
    python3 build_trips.py                    # מדפיס מועמדים, לא כותב כלום
    python3 build_trips.py --json cands.json  # שומר את המועמדים לקובץ
    python3 build_trips.py --plan plan.json --out ready.json

הכלי לא כותב לריפו הנתונים בעצמו. הוא מייצר טיולים מוכנים, והוספתם למפה היא
צעד נפרד ומכוון, כי טיול שמתפרסם הוא הליכה שתושב יצא אליה.

קובץ התוכנית הוא רשימה של
    {"key": "a1b2c3d", "name": "…", "note": "…", "difficulty": "קל|בינוני|מאתגר"}
המזהה הוא מה שמודפס בסוגריים ליד כל מועמד. הוא נגזר מהקיצורים שההליכה עוברת
בהם ולא ממקומה בדירוג, ולכן הוא יציב בין הרצות: את המסלול הכלי יודע למצוא,
ואת השם רק אדם יודע לתת, ושתי הידיעות האלה חייבות להיפגש במזהה שלא זז.
"""

import argparse
import hashlib
import heapq
import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timezone

import requests

# השרת הראשי של Overpass מחזיר 504 די הרבה על שאילתה בגודל הזה, ואז אין רשת
# הליכה ואין טיולים. שני המראות עונים על אותה שאילתה בדיוק.
OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
UA = {"User-Agent": "derech-kitzur-trips/1.0 "
                    "(https://github.com/orimosenzon/fun; orimosenzon@gmail.com)"}

PUBLISHED = ("https://raw.githubusercontent.com/orimosenzon/"
             "derech-kitzur-data/main/data/trails.json")

# השם שנרשם כמי שהוסיף, כדי שיהיה אפשר להבדיל בין מה שאדם הלך וסימן לבין מה
# שנבנה כאן. מופיע במסך הטיול תחת "מופה בידי".
BOT = "בוטי"

BOX = (32.435, 34.925, 32.515, 35.005)          # south, west, north, east
WALK_CACHE = ".cache/walk_network.json"

# הרחובות שהולכים בהם. כביש מהיר וכביש ראשי בין-עירוני לא נכנסים, כל השאר כן:
# במושבה גם רחוב "ראשי" הוא רחוב עם מדרכה.
WALKABLE = ("residential|living_street|pedestrian|footway|path|track|service|"
            "unclassified|tertiary|secondary|steps|cycleway")

# עד כמה רחוק מותר ללכת ברחוב בין קיצור לקיצור. מעבר לזה זה כבר לא "טיול בין
# קיצורי דרך" אלא טיול ברחובות שבמקרה נוגע בקיצור.
MAX_LINK_M = 700

# הטווח שבו טיול הוא הליכה של אחר צהריים. מתחת לזה זו סיבוב הבלוק, מעל זה כבר
# צריך להתארגן.
MIN_TRIP_M = 1500
MAX_TRIP_M = 5000

# קצב הליכה, לחישוב הזמן שמוצג. 4 קמ"ש זה קצב של אדם שמסתכל סביב ולא ממהר.
KMH = 4.0

# כמה קרוב צריך להיות סוף המסלול להתחלה כדי שייחשב מעגלי. אותו כלל כמו
# באפליקציה: רבע מהאורך, עד תקרה.
LOOP_M = 150


def metres(a, b):
    """מרחק על פני כדור הארץ, במטרים."""
    radius = 6371000.0
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlat = lat2 - lat1
    dlng = math.radians(b[1] - a[1])
    h = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2)
    return 2 * radius * math.asin(math.sqrt(h))


def path_length(path):
    return sum(metres(path[i - 1], path[i]) for i in range(1, len(path)))


def load_json(path, default=None):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    return default


def save_json(path, doc):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False)


# ------------------------------------------------------------- רשת ההליכה

def fetch_ways():
    """הדרכים שאפשר ללכת בהן במושבה, עם הגאומטריה שלהן."""
    ways = load_json(WALK_CACHE)
    if ways is not None:
        return ways

    south, west, north, east = BOX
    query = f"""[out:json][timeout:300];
way["highway"~"^({WALKABLE})$"]({south},{west},{north},{east});
out geom;"""
    print("  מושך את רשת ההליכה מ-OpenStreetMap…", file=sys.stderr)
    res = None
    for host in OVERPASS:
        try:
            # Overpass עונה 406 לבקשה בלי User-Agent.
            res = requests.post(host, data={"data": query}, headers=UA, timeout=320)
            res.raise_for_status()
            break
        except Exception as err:                   # noqa: BLE001 - כל כשל, מראה הבא
            print(f"  {host.split('/')[2]}: {err}", file=sys.stderr)
            res = None
    if res is None:
        raise SystemExit("כל המראות של Overpass נכשלו. נסה שוב בעוד כמה דקות.")

    ways = [{"name": el.get("tags", {}).get("name", ""),
             "highway": el.get("tags", {}).get("highway", ""),
             "geom": [[p["lat"], p["lon"]] for p in el.get("geometry", [])]}
            for el in res.json()["elements"] if el.get("geometry")]
    save_json(WALK_CACHE, ways)
    return ways


class Streets:
    """רשת הרחובות כגרף, עם ניתוב הליכה בין נקודות.

    הצמתים הם קואורדינטות מעוגלות לשבע ספרות. שתי דרכים ב-OSM שחולקות צומת
    חולקות גם את הקודקוד המדויק, ולכן העיגול מחבר אותן בלי לחבר בטעות שתי
    דרכים שרק חוצות זו את זו בלי צומת. זו בדיוק ההתנהגות הנכונה: גשר מעל כביש
    אינו מקום שאפשר לרדת בו.
    """

    def __init__(self, ways):
        self.adj = {}
        self.streets = {}
        for way in ways:
            geom = way["geom"]
            for i in range(1, len(geom)):
                a, b = self.node(geom[i - 1]), self.node(geom[i])
                if a == b:
                    continue
                d = metres(geom[i - 1], geom[i])
                self.adj.setdefault(a, []).append((b, d))
                self.adj.setdefault(b, []).append((a, d))
                if way["name"]:
                    self.streets.setdefault(a, way["name"])
                    self.streets.setdefault(b, way["name"])
        self.nodes = list(self.adj)

    @staticmethod
    def node(point):
        return (round(point[0], 7), round(point[1], 7))

    def name_at(self, point):
        """שם הרחוב בנקודה, לפי אותו עיגול שבו נשמרים קטעי הקישור.

        הצמתים נשמרים בשבע ספרות ואילו קטע קישור נכתב בשש, כמו כל קואורדינטה
        אחרת באפליקציה. בלי האינדקס הזה שום חיפוש לא היה מוצא כלום, וזה נראה
        בדיוק כמו רשת בלי שמות רחובות.
        """
        if not hasattr(self, "_by6"):
            self._by6 = {}
            for node, name in self.streets.items():
                self._by6.setdefault((round(node[0], 6), round(node[1], 6)), name)
        return self._by6.get((round(point[0], 6), round(point[1], 6)))

    def edges(self):
        """כל קטע ברשת, פעם אחת. מחושב פעם אחת ולא משתנה אחרי הצמדות."""
        if not hasattr(self, "_edges"):
            seen = set()
            self._edges = []
            for a, neighbours in self.adj.items():
                for b, _ in neighbours:
                    key = (a, b) if a <= b else (b, a)
                    if key not in seen:
                        seen.add(key)
                        self._edges.append(key)
        return self._edges

    def attach(self, point):
        """מצמיד נקודה לרשת, ומחזיר את הצומת שנוצר ואת מרחק ההצמדה.

        לא לצומת הקרוב ביותר אלא לנקודה הקרובה ביותר על הקטע הקרוב ביותר,
        ואז מפצל שם את הקטע. ההבדל אינו טכני: קצה של דרך קיצור יושב באמצע
        רחוב, ולרחוב ישר ב-OSM יש לפעמים שני צמתים במרחק מאות מטרים זה מזה.
        הצמדה לצומת הייתה מותחת קו ישר של מאתיים מטר מהקצה אל הצומת, כלומר
        דרך חצרות, וזה בדיוק מה שקטעי הקישור אמורים למנוע.

        נמדד לפני התיקון: עשר הצמדות מתוך 94 היו מעל 60 מטר, והגדולה 213.
        """
        # השלכה מישורית מקומית. בגודל של מושבה השגיאה זניחה, והיא חוסכת
        # חישוב טריגונומטרי לכל אחד מעשרות אלפי הקטעים.
        kx = 111320.0 * math.cos(math.radians(point[0]))
        ky = 110540.0
        px, py = point[1] * kx, point[0] * ky

        best, best_d2, best_t = None, float("inf"), 0.0
        for a, b in self.edges():
            ax, ay, bx, by = a[1] * kx, a[0] * ky, b[1] * kx, b[0] * ky
            dx, dy = bx - ax, by - ay
            span = dx * dx + dy * dy
            t = 0.0 if span == 0 else max(0.0, min(
                1.0, ((px - ax) * dx + (py - ay) * dy) / span))
            qx, qy = ax + t * dx, ay + t * dy
            d2 = (px - qx) ** 2 + (py - qy) ** 2
            if d2 < best_d2:
                best, best_d2, best_t = (a, b), d2, t

        a, b = best
        if best_t <= 1e-9:
            return a, metres(point, a)
        if best_t >= 1 - 1e-9:
            return b, metres(point, b)

        node = self.node((a[0] + best_t * (b[0] - a[0]),
                          a[1] + best_t * (b[1] - a[1])))
        if node not in self.adj:
            da, db = metres(node, a), metres(node, b)
            self.adj[node] = [(a, da), (b, db)]
            self.adj[a].append((node, da))
            self.adj[b].append((node, db))
            self.nodes.append(node)
            name = self.streets.get(a) or self.streets.get(b)
            if name:
                self.streets[node] = name
                # שמות הרחובות נשמרים במפתח מעוגל, ואינדקס שכבר נבנה לא יכיר
                # צומת שנוצר אחריו.
                self.__dict__.pop("_by6", None)
        return node, math.sqrt(best_d2)

    def routes_from(self, source, targets, cap):
        """דייקסטרה אחת שמחזירה מסלול לכל יעד שנמצא בתוך `cap` מטרים."""
        dist = {source: 0.0}
        prev = {}
        seen = set()
        want = set(targets)
        out = {}
        heap = [(0.0, source)]
        while heap:
            d, node = heapq.heappop(heap)
            if node in seen:
                continue
            seen.add(node)
            if node in want:
                path = [node]
                while path[-1] in prev:
                    path.append(prev[path[-1]])
                out[node] = (d, path[::-1])
            if d > cap:
                continue
            for nxt, w in self.adj.get(node, ()):
                nd = d + w
                if nd < dist.get(nxt, float("inf")):
                    dist[nxt] = nd
                    prev[nxt] = node
                    heapq.heappush(heap, (nd, nxt))
        return out


# ------------------------------------------------------------- שרשור

class Network:
    """דרכי הקיצור, הקצוות שלהן, והמרחקים ברחוב ביניהן."""

    def __init__(self, segments, streets):
        self.segs = {s["id"]: s for s in segments}
        self.streets = streets
        self.snap = {}          # (id, קצה) -> (צומת, מרחק הצמדה)
        for seg in segments:
            for end, point in (("a", seg["path"][0]), ("b", seg["path"][-1])):
                self.snap[(seg["id"], end)] = streets.attach(point)
        far = sorted(d for _, d in self.snap.values())
        print(f"  הצמדה לרשת: חציון {round(far[len(far) // 2])} מ׳, "
              f"מקסימום {round(far[-1])} מ׳", file=sys.stderr)

        print(f"  מנתב בין {len(self.snap)} קצוות…", file=sys.stderr)
        # דייקסטרה אחת לכל קצה, ולא אחת לכל זוג.
        node_of = {k: v[0] for k, v in self.snap.items()}
        targets = set(node_of.values())
        cache = {}
        for key, node in node_of.items():
            if node not in cache:
                cache[node] = streets.routes_from(node, targets, MAX_LINK_M)
        self.link = {}
        for src, s_node in node_of.items():
            for dst, d_node in node_of.items():
                if src[0] == dst[0]:
                    continue
                hit = cache[s_node].get(d_node)
                if hit and hit[0] <= MAX_LINK_M:
                    self.link[(src, dst)] = hit

    def connector(self, src, dst):
        """הקו שמחבר קצה לקצה: מנקודת הקצה עצמה, דרך הרחוב, אל הקצה הבא.

        המסלול מהניתוב מתחיל ונגמר בצמתים של הרחוב, שיושבים כמה מטרים מהקצה
        עצמו. הקצוות האמיתיים נדחפים בהתחלה ובסוף כדי שהקו יהיה רציף ובלי
        קפיצה, שזה מה ש-resolveTrip מצפה לו.
        """
        hit = self.link.get((src, dst))
        if not hit:
            return None
        _, nodes = hit
        start = self.end_point(src)
        finish = self.end_point(dst)
        line = [start] + [list(n) for n in nodes] + [finish]
        # קודקודים כפולים רצופים מנפחים את האורך ולא מוסיפים כלום לצורה.
        clean = [line[0]]
        for point in line[1:]:
            if metres(clean[-1], point) > 0.5:
                clean.append(point)
        return clean

    def end_point(self, key):
        seg = self.segs[key[0]]
        return seg["path"][0] if key[1] == "a" else seg["path"][-1]

    def other(self, key):
        return (key[0], "b" if key[1] == "a" else "a")


def drawn(link):
    return {"draw": [[round(p[0], 6), round(p[1], 6)] for p in link]}


def chains(net, seeds=None):
    """שרשראות מועמדות, בחיפוש חמדני מכל קיצור ובכל כיוון.

    מכל קצה מתחילים ללכת: לוקחים את הקיצור שהחיבור אליו הוא הקצר ביותר, ואז
    שוב. חמדני ולא ממצה בכוונה: המטרה היא כמה הצעות טובות שאדם יבחר מהן, לא
    המסלול האופטימלי.

    השרשרת לא נמדדת רק בסופה אלא בכל צעד. גרסה מוקדמת עצרה רק כשנגמרו
    האפשרויות והחזירה הליכות של 16 קיצורים ושעה וחצי, שזה טיול שאף אחד לא
    יוצא אליו. עכשיו כל צעד שנמצא בטווח נרשם כמועמד בפני עצמו, וגם, אם אפשר
    לחזור מהראש אל נקודת ההתחלה דרך הרחוב, נרשמת גם הגרסה המעגלית שלו.
    """
    out = []
    for seg_id in (seeds or net.segs):
        for start_end in ("a", "b"):
            tail = (seg_id, start_end)               # לשם חוזרים אם סוגרים מעגל
            used = [seg_id]
            head = net.other(tail)
            parts = [{"trail": seg_id, **({"reversed": True} if start_end == "b" else {})}]
            total = net.segs[seg_id]["length"]
            street = 0.0
            while total < MAX_TRIP_M:
                options = [(dist, dst) for (src, dst), (dist, _) in net.link.items()
                           if src == head and dst[0] not in used]
                if not options:
                    break
                dist, nxt = min(options)
                link = net.connector(head, nxt)
                if not link:
                    break
                nxt_seg = net.segs[nxt[0]]
                if total + dist + nxt_seg["length"] > MAX_TRIP_M:
                    break
                parts = parts + [drawn(link),
                                 {"trail": nxt[0],
                                  **({"reversed": True} if nxt[1] == "b" else {})}]
                used = used + [nxt[0]]
                street += dist
                total += dist + nxt_seg["length"]
                head = net.other(nxt)

                if len(used) >= 2 and total >= MIN_TRIP_M:
                    out.append({"parts": parts, "uses": used,
                                "street": round(street)})
                    # וגם, אם יש דרך חזרה, אותה הליכה כשהיא מעגלית. הליכה
                    # שמסתיימת במקום שהתחילה היא מה שאפשר לצאת אליה מהבית
                    # בלי לחשוב איך חוזרים.
                    back = net.connector(head, tail)
                    if back:
                        home = net.link[(head, tail)][0]
                        if MIN_TRIP_M <= total + home <= MAX_TRIP_M:
                            out.append({"parts": parts + [drawn(back)],
                                        "uses": used,
                                        "street": round(street + home)})
    return out


def resolve(parts, segs):
    """אותה הרכבה שהאפליקציה עושה, כדי שהאורך והצורה כאן יהיו מה שיוצג שם."""
    path = []
    for part in parts:
        if "trail" in part:
            seg = segs[part["trail"]]
            piece = seg["path"][::-1] if part.get("reversed") else seg["path"]
        else:
            piece = part["draw"]
        start = 1 if path and metres(path[-1], piece[0]) < 1e-4 else 0
        path.extend(piece[start:])
    return path


# הטווח שבו טיול הוא "אחר צהריים", כלומר בערך 30 עד 60 דקות הליכה. מועמד
# מחוץ לטווח לא נפסל, רק מפסיד נקודות ככל שהוא רחוק ממנו.
SWEET_M = (2000, 4000)


def score(cand, segs):
    """כמה הטיול הזה שווה.

    שלושה דברים, לפי סדר החשיבות: כמה מההליכה היא דרכי קיצור ולא רחוב, כמה
    קיצורים היא מחברת, והאם היא חוזרת לנקודת ההתחלה. ומעליהם קנס על אורך
    שיוצא מהטווח, כי הליכה של תשעים דקות היא לא מה שנשאלה עליה השאלה.
    """
    shortcut = sum(segs[i]["length"] for i in cand["uses"])
    share = shortcut / max(cand["length"], 1)
    low, high = SWEET_M
    if cand["length"] < low:
        off = (low - cand["length"]) / low
    elif cand["length"] > high:
        off = (cand["length"] - high) / high
    else:
        off = 0.0
    return ((share * 100)
            + (len(cand["uses"]) * 6)
            + (30 if cand["loop"] else 0)
            - (off * 60))


def dedupe(cands, keep):
    """הצעות מגוונות: לא שתי גרסאות של אותה הליכה.

    שתי שרשראות שחולקות רוב מהקיצורים הן אותו טיול בכיוון אחר או עם קיצור
    אחד פחות, ורשימה של חמש כאלה לא שווה יותר מאחת.
    """
    chosen = []
    for cand in sorted(cands, key=lambda c: -c["score"]):
        pool = set(cand["uses"])
        if any(len(pool & set(other["uses"])) / min(len(pool), len(other["uses"])) > 0.4
               for other in chosen):
            continue
        chosen.append(cand)
        if len(chosen) >= keep:
            break
    return chosen


def where(cand, net):
    """באילו רחובות ההליכה הזאת עוברת, לפי שמות הצמתים שהיא נוגעת בהם."""
    names = []
    for part in cand["parts"]:
        if "draw" not in part:
            continue
        for point in part["draw"]:
            name = net.streets.name_at(point)
            if name and name not in names:
                names.append(name)
    return names


def build(args):
    doc = json.loads(urllib.request.urlopen(PUBLISHED).read().decode())
    segments = doc["segments"]
    print(f"  {len(segments)} דרכי קיצור", file=sys.stderr)

    streets = Streets(fetch_ways())
    print(f"  {len(streets.nodes)} צמתים ברשת ההליכה", file=sys.stderr)

    net = Network(segments, streets)
    print(f"  {len(net.link)} חיבורים אפשריים בין קצוות", file=sys.stderr)

    cands = chains(net)
    for cand in cands:
        cand["path"] = resolve(cand["parts"], net.segs)
        cand["length"] = round(path_length(cand["path"]))
        cand["minutes"] = max(5, round(cand["length"] / 1000 / KMH * 60))
        cand["loop"] = metres(cand["path"][0], cand["path"][-1]) <= min(
            LOOP_M, cand["length"] / 4)
        cand["score"] = score(cand, net.segs)
    print(f"  {len(cands)} שרשראות אפשריות", file=sys.stderr)

    best = dedupe(cands, args.keep)
    for i, cand in enumerate(best, 1):
        cand["rank"] = i
        # מזהה שנגזר מהקיצורים שההליכה עוברת בהם, ולכן יציב בין הרצות. הדירוג
        # אינו יציב: שינוי בניקוד או ברשת מזיז מסלול ממקום שלישי לחמישי, וקובץ
        # תוכנית שמצביע על "מקום 3" היה מלביש שם של הליכה אחת על הליכה אחרת.
        cand["key"] = key_of(cand["uses"])
        cand["streets"] = where(cand, net)
        cand["names"] = [net.segs[u]["name"] for u in cand["uses"]]
        print(f"\n{i}. [{cand['key']}] {cand['length']} מ׳ · {cand['minutes']} דק׳ · "
              f"{len(cand['uses'])} קיצורים · "
              f"{'מעגלי' if cand['loop'] else 'מנקודה לנקודה'} · "
              f"רחוב {cand['street']} מ׳")
        print("   קיצורים: " + " ← ".join(cand["names"]))
        print("   רחובות: " + ", ".join(cand["streets"][:8]))

    if args.json:
        save_json(args.json, best)
        print(f"\nנשמר ל-{args.json}", file=sys.stderr)

    if args.plan:
        save_json(args.out, dress(best, load_json(args.plan, [])))
        print(f"נכתבו טיולים מוכנים ל-{args.out}", file=sys.stderr)


def key_of(uses):
    return hashlib.sha1("|".join(uses).encode()).hexdigest()[:7]


def dress(best, plan):
    """מלביש על המועמדים את מה שרק אדם יכול לתת להם: שם, תיאור, דרגת קושי.

    הכלי יודע למצוא מסלול טוב, ולא יודע לקרוא לו בשם. קובץ התוכנית הוא
    רשימה של {"rank", "name", "note", "difficulty"}, ורק המועמדים שמופיעים בו
    יוצאים מכאן. הזמן והאורך לא נלקחים מהתוכנית אלא מהגאומטריה, כדי שלא ייווצר
    מצב שכתוב "ארבעים דקות" על הליכה של שעה.
    """
    by_key = {c["key"]: c for c in best}
    out = []
    for entry in plan:
        cand = by_key.get(entry["key"])
        if not cand:
            print(f"  אזהרה: אין מועמד עם המזהה {entry['key']}", file=sys.stderr)
            continue
        out.append({
            # מזהה יציב שנגזר מהתוכן, כדי שהרצה חוזרת של אותה תוכנית לא תיצור
            # עותקים נוספים של אותו טיול. hashlib ולא hash() המובנה, שמערבב
            # מחדש בכל הרצה של פייתון ולכן היה מייצר מזהה חדש בכל פעם.
            "id": "trip-b" + cand["key"],
            "name": entry["name"],
            "note": entry.get("note", ""),
            "photos": [],
            "links": [],
            "parts": cand["parts"],
            **({"difficulty": entry["difficulty"]} if entry.get("difficulty") else {}),
            "minutes": cand["minutes"],
            "origin": "build_trips",
            "added": entry.get("added") or datetime.now(timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            # מי שהוסיף. חמשת הראשונים נדחפו בלי השדה הזה, והם נראו באפליקציה
            # בדיוק כמו טיול שתושב הוסיף, עד שאורי שאל מי הוסיף אותם. מסלול
            # שנבנה בכלי צריך להגיד את זה בעצמו, במסך, ולא רק ביומן הקומיטים.
            "by": entry.get("by") or BOT,
            # מסלול שהכלי צייר סביר בכל מקום ומאומת באף מקום: שער נעול, גדר
            # שהוקמה, וקטע בוצי אחרי גשם נראים לו זהים. הדגל מוצג לכל מי
            # שפותח את הטיול, ומי שילך בו בפועל יוריד אותו.
            "unwalked": True,
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", type=int, default=5,
                        help="כמה הצעות להחזיר")
    parser.add_argument("--json", help="קובץ לשמירת ההצעות")
    parser.add_argument("--plan", help="קובץ שם/תיאור/קושי לכל מועמד נבחר")
    parser.add_argument("--out", default="trips_ready.json",
                        help="לאן לכתוב את הטיולים המוכנים")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
