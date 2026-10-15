"""מנוע החיפוש והתשובות על תמלולי ישיבות המליאה ועל הפרוטוקולים הכתובים.

search(): חיפוש היברידי. מילים (BM25, עם הסרת אותיות שימוש) + משמעות (bge-m3),
ממוזגים ב-Reciprocal Rank Fusion. קטע שמופיע גבוה בשניהם עולה לראש.
answer(): שולח את הקטעים המובילים למודל ב-Azure ומבקש תשובה עם הפניות [n] בלבד מהם.
"""
import json
import os
import re
import sys
import time

# ai_azure יושב ב-~/.config/ai ונטען בפייתון הגלובלי דרך קובץ .pth; ה-venv לא רואה אותו
sys.path.append(os.path.expanduser("~/.config/ai"))

import numpy as np
from rank_bm25 import BM25Okapi

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.environ.get("RAG_INDEX", os.path.join(HERE, "..", "private", "index"))
ANSWER_MODEL = "gpt-5.4-mini"
PREFIXES = ("וש", "וה", "וב", "ול", "ומ", "שה", "שב", "של", "מה", "כש", "לכ", "ו", "ה", "ב", "ל", "מ", "ש", "כ")
# קבוצות המקורות שאפשר לסנן לפיהן. src של קטע: video (הקלטה), council_doc (מסמך מ-goinfo),
# planning (פרוטוקול של הוועדה המקומית לתכנון ובנייה), trees (רישיון כריתה של פקיד היערות)
# דוחות הפעילות של הוועדה, ותוכניות והחלטות מ-מבא"ת, הם council_doc אבל שייכים לתכנון
PLANNING_BODIES = ("הוועדה המקומית לתכנון ובנייה", "מוסדות התכנון")
GROUPS = {
    "council": lambda c: c["src"] == "video" or (c["src"] == "council_doc" and c["body"] == "מליאת המועצה"),
    "committees": lambda c: c["src"] == "council_doc" and c["body"] not in ("מליאת המועצה",) + PLANNING_BODIES,
    "planning": lambda c: c["src"] == "planning" or c.get("body") in PLANNING_BODIES,
    "trees": lambda c: c["src"] == "trees",
}
STOP = set("של את על עם זה זו לא כן גם אבל או כי אם אני אתה את הוא היא אנחנו הם יש אין מה מי "
           "כל עוד רק כבר אז פה שם היה היו יהיה אחד אחת טוב בסדר רגע אוקיי".split())


def llm_client():
    """ב-Space: משתני סביבה (סודות של HF). במחשב של אורי: המודול המשותף ai_azure."""
    if os.environ.get("AZURE_OPENAI_API_KEY"):
        from openai import OpenAI
        # strip: מפתח שהודבק מהלוח הגיע עם ירידת שורה בסוף, וכל קריאה נכשלה ב-APIConnectionError (30/9)
        return OpenAI(base_url=os.environ["AZURE_OPENAI_ENDPOINT"].strip().rstrip("/") + "/",
                      api_key=os.environ["AZURE_OPENAI_API_KEY"].strip())
    import ai_azure
    return ai_azure.client()


def tokens(text):
    """מילים עבריות, כל אחת גם כמו שהיא וגם בלי אותיות שימוש בתחילתה."""
    out = []
    for w in re.findall(r"[֐-׿a-zA-Z0-9\"']+", text.lower()):
        w = w.replace('"', "").replace("'", "")
        if len(w) < 2 or w in STOP:
            continue
        out.append(w)
        for p in PREFIXES:
            if w.startswith(p) and len(w) - len(p) >= 2:
                out.append(w[len(p):])
                break
    return out


def describe(h):
    """שורת הכותרת של קטע בהקשר שנשלח למודל."""
    if h.get("src") == "planning":
        return f"פרוטוקול {h['body']}, ישיבה {h['number']}, {h['date']}, עמוד {h['page']}. {h['head']}"
    if h.get("src") == "trees":
        return f"רישיון כריתה שפרסם פקיד היערות במערכת יעלה, {h['date']}"
    if h.get("src") == "council_doc":
        return f"מסמך רשמי, {h['body']}, {h['kind']}, {h['date']} ({h['title']})"
    return f"תמלול אוטומטי של הקלטת ישיבת המליאה {h['date']}, דקה {h['start'] // 60}"


class Engine:
    def __init__(self, device="cpu"):
        d = json.load(open(os.path.join(INDEX, "chunks.json"), encoding="utf-8"))
        self.chunks = d["chunks"]
        for c in self.chunks:
            c.setdefault("src", "video")   # אינדקס מלפני 15/10/2026, רק הקלטות
        self.emb = np.load(os.path.join(INDEX, "emb.npy")).astype(np.float32)
        toks = [tokens(c.get("head", "") + " " + c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(toks)
        self.token_sets = [set(self._base(t)) for t in toks]
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(d["model"], device=device)
        self.boost = np.array([2.0 if c.get("kind") in ("החלטות מהעמוד הסרוק", "פרוטוקול (עמוד ההצבעות הסרוק)") else 1.0 for c in self.chunks])
        self.meetings = sorted({c["date"] for c in self.chunks if c["src"] == "video"})
        self.dates = sorted({c["date"] for c in self.chunks})
        self.counts = {
            "video": len({c["video"] for c in self.chunks if c["src"] == "video"}),
            "council_doc": len({c["url"] for c in self.chunks if c["src"] == "council_doc"}),
            "planning": len({c["url"] for c in self.chunks if c["src"] == "planning"}),
            "trees": sum(1 for c in self.chunks if c["src"] == "trees"),
        }

    @staticmethod
    def _base(toks):
        """צורות בסיס בלבד: כל מילה בלי אות שימוש, כדי ש"ובמגרש" ו"מגרש" ייחשבו אותו דבר."""
        out = []
        for w in toks:
            for p in PREFIXES:
                if w.startswith(p) and len(w) - len(p) >= 2:
                    w = w[len(p):]
                    break
            out.append(w)
        return out

    def search(self, query, k=10, date_from=None, date_to=None, src=None):
        n = len(self.chunks)
        ok = np.ones(n, bool)
        group = GROUPS.get(src)
        if date_from or date_to or group:
            ok = np.array([(not date_from or c["date"] >= date_from) and
                           (not date_to or c["date"] <= date_to) and
                           (not group or group(c)) for c in self.chunks])
        q = self.model.encode([query], normalize_embeddings=True)[0]
        dense = np.where(ok, self.emb @ q, -1)
        sparse = np.where(ok, self.bm25.get_scores(tokens(query)), -1)
        score = np.zeros(n)
        for arr in (dense, sparse):
            order = np.argsort(-arr)[:200]
            score[order] += 1.0 / (60 + np.arange(1, len(order) + 1))
        # כיסוי: קטע שמכיל את כל מילות השאלה עולה על קטע שמכיל רק חלק. בלי זה, "ניגוד
        # עניינים שי חי" החזיר קודם דיון ארוך מ-2021 על ניגוד עניינים בלי שי חי (30/9).
        qset = set(self._base(tokens(query)))
        if len(qset) > 1:
            cov = np.array([len(qset & s) / len(qset) for s in self.token_sets])
            score *= 0.4 + 0.6 * cov
        # החלטה שנקראה מהעמוד הסרוק היא משפט אחד ("מליאת המועצה מאשרת..."), ולכן כמעט אין לה
        # מילים משותפות עם שאלה כמו "האם אושר תקציב 2026?", אף שהמשמעות שלה הכי קרובה (15/10)
        score *= self.boost
        top = [i for i in np.argsort(-score) if ok[i]][:k]
        return [dict(self.chunks[i], score=round(float(score[i]), 4),
                     sim=round(float(dense[i]), 3)) for i in top]

    def answer(self, question, k=14, **filters):
        hits = self.search(question, k=k, **filters)
        ctx = "\n\n".join(f"[{i+1}] {describe(h)}:\n{h['text']}" for i, h in enumerate(hits))
        system = (
            "אתה עוזר שמסביר לתושבי פרדס חנה-כרכור מה נאמר והוחלט בישיבות של המועצה ושל "
            "הוועדה המקומית לתכנון ובנייה. ענה בעברית פשוטה וברורה, רק על סמך הקטעים שקיבלת. "
            "אחרי כל טענה הוסף הפניה לקטע בסוגריים מרובעים, למשל [3]. "
            "יש שלושה סוגי קטעים, וכותרת כל קטע אומרת מאיזה סוג הוא: תמלול אוטומטי של הקלטת "
            "ישיבת מליאה (יש בו שגיאות, והשמות של הדוברים זוהו אוטומטית); מסמך רשמי של המועצה "
            "(פרוטוקול, תמלול רשמי, סדר יום, ועדה; הטקסט עבר OCR ולפעמים סדר המילים משתבש); "
            "ופרוטוקול של הוועדה המקומית לתכנון ובנייה, שבו לכל סעיף יש בקשה או תוכנית, מהלך "
            "הדיון והחלטה. החלטות והצבעות לוקחים מהפרוטוקולים כשיש. אם משהו לא ברור, אמור זאת. "
            "אם הקטעים לא עונים על השאלה, אמור בפשטות שלא נמצא מידע, והצע ניסוח אחר לחיפוש. "
            "אל תמציא ואל תשתמש בידע חיצוני. ציין תאריכים וגוף (מליאה, ועדת משנה וכו') כשזה עוזר. "
            "הבחן בין דבר שנאמר לבין החלטה שהתקבלה, ובין המלצה של הצוות המקצועי להחלטת הוועדה. "
            "כשיש הצבעה, ציין מי הצביע בעד ומי נגד אם זה כתוב. אין המשך שיחה, ולכן אל תציע "
            "לענות על עוד שאלות.")
        t0 = time.time()
        resp = llm_client().chat.completions.create(
            model=ANSWER_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": f"שאלה: {question}\n\nקטעים מהישיבות:\n\n{ctx}"}],
            max_completion_tokens=6000, reasoning_effort="low")
        u = resp.usage
        return {"answer": resp.choices[0].message.content, "sources": hits,
                "usage": {"in": u.prompt_tokens, "out": u.completion_tokens,
                          "seconds": round(time.time() - t0, 1)}}
