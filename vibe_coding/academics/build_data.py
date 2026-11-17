"""בונה את data.js מנתוני המפקד ב-raw/.

הלמ"ס מפרסמת את אחוז האקדמאים (בני 15+) לפי אזור סטטיסטי. כאן מחברים
לרמת יישוב בממוצע משוקלל. המשקל של אזור הוא קירוב למספר בני ה-15 ומעלה בו:
תושבים × (100 - 0.75 × אחוז בני 0-19) / 100, כלומר מניחים שרבע מבני 0-19 הם 15-19.
ביישוב שהוא אזור אחד התוצאה זהה למספר של הלמ"ס.
"""
import glob
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def load_areas():
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, "raw", "census_*.json"))):
        rows += [x["attributes"] for x in json.load(open(f)).get("features", [])]
    if not rows:
        raise SystemExit("אין נתונים ב-raw/. להריץ קודם ./fetch_raw.sh")
    return rows


def aggregate(areas):
    agg = defaultdict(lambda: {"name": "", "pop": 0.0, "wsum": 0.0, "w": 0.0, "areas": 0})
    for r in areas:
        g = agg[r["SEMEL_YISHUV"]]
        g["name"] = r["SHEM_YISHUV_HEB"]
        p = r["pop_approx"] or 0
        g["pop"] += p
        a, y = r["AcadmCert_pcnt"], r["age0_19_pcnt"]
        if a is None or a < 0 or y is None or p <= 0:
            continue
        w = p * (100 - 0.75 * y) / 100
        g["wsum"] += w * a
        g["w"] += w
        g["areas"] += 1
    out = []
    for code, g in agg.items():
        if g["w"] <= 0:
            continue
        name = g["name"].replace("*", "").strip().replace(" -", "-")
        out.append([name, int(round(g["pop"])), round(g["wsum"] / g["w"], 1), g["areas"], code])
    out.sort(key=lambda t: (-t[2], t[0]))
    return out, len(agg)


def main():
    areas = load_areas()
    rows, total = aggregate(areas)
    path = os.path.join(HERE, "data.js")
    with open(path, "w") as f:
        f.write("// נוצר ע\"י build_data.py, לא לערוך ביד. [שם, תושבים, אחוז אקדמאים 15+, מספר אזורים, סמל יישוב]\n")
        f.write(f"const META = {json.dumps({'areas': len(areas), 'localities': total, 'with_data': len(rows)})};\n")
        f.write("const RAW = " + json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"{len(areas)} אזורים, {total} יישובים, {len(rows)} עם נתון השכלה → data.js")


if __name__ == "__main__":
    main()
