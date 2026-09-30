"""מנוע החיפוש והתשובות על תמלולי ישיבות המליאה.

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
INDEX = os.path.join(HERE, "..", "private", "index")
ANSWER_MODEL = "gpt-5.4-mini"
PREFIXES = ("וש", "וה", "וב", "ול", "ומ", "שה", "שב", "של", "מה", "כש", "לכ", "ו", "ה", "ב", "ל", "מ", "ש", "כ")
STOP = set("של את על עם זה זו לא כן גם אבל או כי אם אני אתה את הוא היא אנחנו הם יש אין מה מי "
           "כל עוד רק כבר אז פה שם היה היו יהיה אחד אחת טוב בסדר רגע אוקיי".split())


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


class Engine:
    def __init__(self, device="cpu"):
        d = json.load(open(os.path.join(INDEX, "chunks.json"), encoding="utf-8"))
        self.chunks = d["chunks"]
        self.emb = np.load(os.path.join(INDEX, "emb.npy")).astype(np.float32)
        toks = [tokens(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(toks)
        self.token_sets = [set(self._base(t)) for t in toks]
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(d["model"], device=device)
        self.meetings = sorted({c["date"] for c in self.chunks})

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

    def search(self, query, k=10, date_from=None, date_to=None):
        n = len(self.chunks)
        ok = np.ones(n, bool)
        if date_from or date_to:
            ok = np.array([(not date_from or c["date"] >= date_from) and
                           (not date_to or c["date"] <= date_to) for c in self.chunks])
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
        top = [i for i in np.argsort(-score) if ok[i]][:k]
        return [dict(self.chunks[i], score=round(float(score[i]), 4),
                     sim=round(float(dense[i]), 3)) for i in top]

    def answer(self, question, k=12, **filters):
        hits = self.search(question, k=k, **filters)
        ctx = "\n\n".join(f"[{i+1}] ישיבה {h['date']}, דקה {h['start']//60}:\n{h['text']}"
                          for i, h in enumerate(hits))
        system = (
            "אתה עוזר שמסביר לתושבי פרדס חנה-כרכור מה נאמר בישיבות מליאת המועצה. "
            "ענה בעברית פשוטה וברורה, רק על סמך הקטעים שקיבלת. אחרי כל טענה הוסף הפניה "
            "לקטע בסוגריים מרובעים, למשל [3]. התמלול אוטומטי ויש בו שגיאות; אם משהו לא ברור, "
            "אמור זאת. אם הקטעים לא עונים על השאלה, אמור בפשטות שלא נמצא מידע בישיבות, "
            "והצע ניסוח אחר לחיפוש. אל תמציא ואל תשתמש בידע חיצוני. ציין תאריכים כשזה עוזר. "
            "הבחן בין דבר שנאמר על ידי חבר מועצה לבין החלטה שהתקבלה בהצבעה.")
        import ai_azure
        t0 = time.time()
        resp = ai_azure.client().chat.completions.create(
            model=ANSWER_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": f"שאלה: {question}\n\nקטעים מהישיבות:\n\n{ctx}"}],
            max_completion_tokens=6000, reasoning_effort="low")
        u = resp.usage
        return {"answer": resp.choices[0].message.content, "sources": hits,
                "usage": {"in": u.prompt_tokens, "out": u.completion_tokens,
                          "seconds": round(time.time() - t0, 1)}}
