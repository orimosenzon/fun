#!/usr/bin/env python3
"""הצמדת שמות לדוברים (SPEAKER_NN) בתמלולי ישיבות המליאה, עם מודל שפה ב-Azure.

לכל ישיבה ב-private/transcripts_spk שולחים את התמלול המסומן ואת רשימת חברי המועצה
המוכרים, ומבקשים מיפוי דובר -> שם, תפקיד, רמת ביטחון וציטוט שמבסס את הזיהוי.
הפלט: private/speakers/<שם הישיבה>.json. הזיהוי נשען על מה שנאמר בישיבה עצמה
(יו"ר שנותן רשות דיבור בשם, פנייה ישירה, הצגה עצמית), לא על ניחוש.

    python3 tools/name_speakers.py              # כל הישיבות שעוד לא עברו
    python3 tools/name_speakers.py 8ow4RDfJRoM  # ישיבה אחת
"""
import glob
import json
import os
import sys
import time

import ai_azure

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PRIV = os.path.join(ROOT, "private")
LOG = os.path.join(PRIV, "speakers.log")
MODEL = "gpt-5.4-mini"

# רשימת עזר לאיות של חברי מועצה ובעלי תפקידים. נשמרת ב-private/ ולא בקוד, כי הריפו
# ציבורי ולתיקייה לא נכנסים שמות. זו לא רשימה סגורה: המודל מזהה גם אחרים מהתמלול.
KNOWN_FILE = os.path.join(PRIV, "known_names.txt")
KNOWN = open(KNOWN_FILE, encoding="utf-8").read() if os.path.exists(KNOWN_FILE) else ""

SYSTEM = """אתה עוזר מחקר שמזהה דוברים בתמלול של ישיבת מליאת מועצה מקומית בישראל.
התמלול אוטומטי: יש בו שגיאות כתיב, והחלוקה לדוברים אוטומטית ולפעמים מילה בודדת
משויכת לדובר הלא נכון. כל דובר מסומן SPEAKER_NN.

זהה כל דובר רק על סמך ראיות מהתמלול: יו"ר שנותן רשות דיבור בשם ("דנה, בבקשה")
ואחריו מדבר דובר מסוים, פנייה ישירה בשם, הצגה עצמית, או תפקיד שמשתמע בבירור
(מי שמנהל את הישיבה ומעמיד להצבעה הוא היו"ר). עבור על כל התמלול: כל פעם שהיו"ר
או מישהו אחר פונה לאדם בשמו ("יוסי", "רונית, בבקשה", "דנה, תסיימי"), בדוק מי הדובר
שמדבר מיד אחרי הפנייה או שאליו מופנית התשובה. ראיה חוזרת בכמה מקומות מחזקת את הזיהוי.
גם ציטוט של מה שדובר אומר על עצמו ("אני כיו"ר ועדת המשנה", "הצעה לסדר שלי") הוא ראיה.
אל תמציא: אם אין ראיה, השאר name ריק. confidence=high רק כשיש שתי ראיות או יותר.

החזר JSON בלבד, בצורה:
{"speakers": {"SPEAKER_00": {"name": "...", "role": "...", "confidence": "high|medium|low",
  "evidence": "ציטוט קצר עם חותמת הזמן [mm:ss] שמבסס את הזיהוי"}}}
role: ראש המועצה / חבר מועצה / יועמ"ש / גזבר / מהנדס / מנכ"ל / מבקר / תושב / אחר / לא ידוע.
כלול כל SPEAKER שמופיע בתמלול."""


def fmt(t):
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}"


def compact(segs):
    lines = []
    for s in segs:
        if lines and lines[-1][1] == s["speaker"]:
            lines[-1][2] += " " + s["text"]
        else:
            lines.append([s["start"], s["speaker"], s["text"]])
    return "\n".join(f"[{fmt(t)}] {spk}: {txt}" for t, spk, txt in lines)


def main():
    only = set(sys.argv[1:])
    os.makedirs(os.path.join(PRIV, "speakers"), exist_ok=True)
    client = ai_azure.client()
    tot_in = tot_out = 0
    for path in sorted(glob.glob(os.path.join(PRIV, "transcripts_spk", "*.json")), reverse=True):
        name = os.path.basename(path)
        vid = name.split("_", 1)[1][:-5]
        out = os.path.join(PRIV, "speakers", name)
        if (only and vid not in only) or (not only and os.path.exists(out)):
            continue
        d = json.load(open(path, encoding="utf-8"))
        m = d["meeting"]
        prompt = (f"ישיבה: {m['title']} (תאריך {m['date']}).\n"
                  f"שמות מוכרים לעזרה באיות:\n{KNOWN}\n\nהתמלול:\n{compact(d['segments'])}")
        t0 = time.time()
        try:
            # בישיבה של 94 דקות חשיבה ברמה high מיצתה 32K טוקנים בלי להחזיר תשובה (29/9).
            # לכן medium עם תקרה גבוהה, ואם עדיין נחתך, ניסיון נוסף ב-low.
            for effort in ("medium", "low"):
                resp = client.chat.completions.create(
                    model=MODEL, messages=[{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}, max_completion_tokens=64000,
                    reasoning_effort=effort)
                u = resp.usage
                tot_in += u.prompt_tokens
                tot_out += u.completion_tokens
                if resp.choices[0].finish_reason != "length" and resp.choices[0].message.content:
                    break
            res = json.loads(resp.choices[0].message.content)
            json.dump({"meeting": m, "model": MODEL, **res}, open(out, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
            named = sum(1 for v in res["speakers"].values() if v.get("name"))
            line = (f"done {name}: {named}/{len(res['speakers'])} named, "
                    f"{u.prompt_tokens} in / {u.completion_tokens} out, effort {effort}, "
                    f"{time.time()-t0:.0f}s")
        except Exception as e:
            line = f"FAIL {name}: {e}"
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}"
        print(line, flush=True)
        open(LOG, "a", encoding="utf-8").write(line + "\n")
    print(f"total tokens: {tot_in} in, {tot_out} out")


if __name__ == "__main__":
    main()
