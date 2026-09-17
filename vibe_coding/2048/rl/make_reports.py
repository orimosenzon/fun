"""
בניית דו"חות ה-HTML מהלוגים ומתוצאות ההערכה.

קלט:  logs/<run>_train.jsonl  (עקומות למידה),  reports/data/eval_<name>.json  (הערכה סופית)
פלט:  reports/dqn_report.html, reports/ac_report.html, reports/comparison.html
       וטבלת התוצאות ב-README.md (בין הסמנים RESULTS_TABLE).

הרצה:  python rl/make_reports.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "logs")
DATA = os.path.join(ROOT, "reports", "data")
OUT = os.path.join(ROOT, "reports")
HARDWARE = "NVIDIA GeForce GTX 1060 6GB · PyTorch 2.10 (CUDA 12.6) · 4 ליבות CPU"
CHARTJS = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"

AGENTS = {
    "dqn": {"run": "dqn", "label": "DQN", "color": "s1", "report": "dqn_report.html", "long": "Deep Q-Network"},
    "ac": {"run": "a2c", "label": "Actor-Critic", "color": "s2", "report": "ac_report.html", "long": "Advantage Actor-Critic (A2C)"},
}


# ----------------------------------------------------------------------------
# טעינת נתונים
# ----------------------------------------------------------------------------

def load_log(run: str):
    train, evals = [], []
    path = os.path.join(LOGS, f"{run}_train.jsonl")
    if not os.path.exists(path):
        return train, evals
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            (evals if "eval" in rec else train).append(rec)
    return train, evals


def load_eval(name: str):
    path = os.path.join(DATA, f"eval_{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def j(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def f0(v) -> str:
    return f"{v:,.0f}"


def pct(v) -> str:
    return f"{100 * v:.1f}%"


def hist(values, width, max_value=None):
    values = np.asarray(values)
    top = max_value if max_value is not None else values.max()
    edges = np.arange(0, top + width, width)
    counts, _ = np.histogram(values, bins=edges)
    labels = [f"{int(edges[i] / 1000)}–{int(edges[i + 1] / 1000)}K" for i in range(len(counts))]
    return labels, counts.tolist()


TILE_ORDER = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def tile_dist(max_tiles):
    t = np.asarray(max_tiles)
    return [int((t == v).sum()) for v in TILE_ORDER]


def thresholds_table(evals, thresholds):
    """באיזה מספר מעברים ובאיזו דקה ההערכה החמדנית עברה לראשונה כל סף."""
    rows = []
    for th in thresholds:
        hit = next((e for e in evals if e["eval"]["score_mean"] >= th), None)
        rows.append((th, hit["transitions"] if hit else None, hit["elapsed_sec"] / 60 if hit else None))
    return rows


# ----------------------------------------------------------------------------
# תבנית דף
# ----------------------------------------------------------------------------

def page(title: str, subtitle: str, meta: str, toc: list[tuple[str, str]], body: str, crumbs: str = "") -> str:
    toc_html = " ".join(f'<a href="#{a}">{t}</a>' for a, t in toc)
    today = dt.date.today().strftime("%d.%m.%Y")
    return f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="assets/report.css">
<script src="{CHARTJS}"></script>
<script src="assets/report.js"></script>
</head>
<body>
<div class="wrap">
<div class="crumbs">{crumbs}</div>
<header class="hero">
  <h1>{title}</h1>
  <p class="sub">{subtitle}</p>
  <div class="meta">{meta}</div>
  <nav class="toc">{toc_html}</nav>
</header>
{body}
<footer>נוצר אוטומטית ב-{today} על ידי <code>rl/make_reports.py</code> (בוטי). הגרפים מצוירים עם Chart.js; לכל גרף יש תצוגת טבלה.</footer>
</div>
<script>R.init();</script>
</body>
</html>
"""


def kpi(label, value, sub="", cls="", delta="", delta_cls=""):
    sub_html = f"<small>{sub}</small>" if sub else ""
    delta_html = f'<div class="delta {delta_cls}">{delta}</div>' if delta else ""
    return f'<div class="kpi {cls}"><div class="label">{label}</div><div class="value">{value}{sub_html}</div>{delta_html}</div>'


def card(cid, title, desc, cls="chart", script=""):
    return f'<div class="card"><h3>{title}</h3><p class="desc">{desc}</p><div class="{cls}" id="{cid}"></div></div>\n<script>{script}</script>\n'


def hp_list(args: dict, keys: list[tuple[str, str]]) -> str:
    def fmt_v(v):
        return f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) and abs(v) >= 1000 else str(v)
    items = "".join(f"<div><span>{label}</span><span>{fmt_v(args.get(k, ''))}</span></div>" for k, label in keys if k in args)
    return f'<div class="hp">{items}</div>'


DQN_HP = [
    ("total_transitions", "תקציב מעברים"), ("n_envs", "סביבות במקביל"), ("gamma", "gamma (היוון)"),
    ("n_step", "n-step"), ("lr", "קצב למידה (Adam)"), ("batch_size", "גודל אצווה"), ("grad_steps", "צעדי למידה לצעד סביבה"),
    ("buffer_size", "גודל זיכרון החוויות"), ("learning_starts", "התחלת למידה אחרי"), ("eps_start", "epsilon התחלתי"),
    ("eps_end", "epsilon סופי"), ("eps_decay_frac", "חלק האימון לדעיכת epsilon"), ("tau", "tau (עדכון רך של רשת המטרה)"),
    ("grad_clip", "חיתוך גרדיאנט"), ("reward_scale", "סקאלת תגמול"), ("filters", "פילטרים בקונבולוציה"), ("hidden", "שכבה נסתרת"),
    ("eval_every", "הערכה כל"), ("eval_games", "משחקים בהערכה"), ("seed", "זרע"),
]
AC_HP = [
    ("time_limit_min", "מגבלת זמן (דקות)"), ("n_envs", "סביבות במקביל"), ("n_steps", "אורך רולאאוט"), ("gamma", "gamma (היוון)"),
    ("gae_lambda", "lambda (GAE)"), ("lr", "קצב למידה (Adam)"), ("vf_coef", "משקל הפסד המבקר"), ("ent_coef", "מקדם אנטרופיה"),
    ("grad_clip", "חיתוך גרדיאנט"), ("norm_adv", "נרמול יתרונות"), ("reward_scale", "סקאלת תגמול"), ("filters", "פילטרים בקונבולוציה"),
    ("hidden", "שכבה נסתרת"), ("eval_every", "הערכה כל"), ("eval_games", "משחקים בהערכה"), ("seed", "זרע"),
]


# ----------------------------------------------------------------------------
# טקסטים של ניתוח (נכתבים ידנית אחרי צפייה בתוצאות; נטענים מקובץ אם קיים)
# ----------------------------------------------------------------------------

def analysis_text(name: str) -> str:
    path = os.path.join(OUT, "data", f"analysis_{name}.html")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return "<p class='note'>(הניתוח המילולי ייכתב אחרי סיום הריצה.)</p>"


# ----------------------------------------------------------------------------
# דו"ח לאלגוריתם יחיד
# ----------------------------------------------------------------------------

def algo_report(kind: str, baselines: dict):
    A = AGENTS[kind]
    train, evals = load_log(A["run"])
    ev = load_eval(kind)
    if ev is None or not train:
        print(f"skip {kind}: missing data")
        return None
    S = ev["summary"]
    args = ev.get("train_args", {})
    rnd, grd = baselines["random"]["summary"], baselines["greedy"]["summary"]
    total_min = train[-1]["elapsed_sec"] / 60
    total_trans = train[-1]["transitions"]
    sps = train[-1]["sps"]
    color = A["color"]

    # --- KPI ---
    kpis = "".join([
        kpi("ניקוד ממוצע (1,000 משחקים)", f0(S["score_mean"]), cls=color,
            delta=f"פי {S['score_mean'] / grd['score_mean']:.1f} מהחמדן, פי {S['score_mean'] / rnd['score_mean']:.1f} מאקראי", delta_cls="good"),
        kpi("חציון", f0(S["score_median"]), cls=color),
        kpi("המשחק הטוב ביותר", f0(S["score_max"]), cls=color),
        kpi("הגיע ל-1024", pct(S["reach_1024"]), cls=color, delta=f"החמדן: {pct(grd['reach_1024'])}"),
        kpi("הגיע ל-2048", pct(S["reach_2048"]), cls=color, delta=(f"ול-4096: {pct(S['reach_4096'])}" if S["reach_4096"] > 0 else "")),
        kpi("זמן אימון", f"{total_min:.0f}", "דקות", cls=color, delta=f"{total_trans / 1e6:.1f} מיליון מעברים, {sps:,} בשנייה"),
    ])

    # --- עקומת למידה ---
    x_eval = [e["transitions"] / 1e6 for e in evals]
    y_eval = [e["eval"]["score_mean"] for e in evals]
    x_tr = [r["transitions"] / 1e6 for r in train]
    y_tr = [r["recent_score_mean"] for r in train]
    lc = {
        "x": x_tr,
        "series": [
            {"label": "הערכה חמדנית (100 משחקים, בלי חקירה)", "x": x_eval, "y": y_eval, "color": color, "width": 2.5, "direct": False},
            {"label": "ניקוד באימון (200 המשחקים האחרונים, עם חקירה)", "x": x_tr, "y": y_tr, "color": "s3", "width": 1.5},
        ],
        "refs": [
            {"label": f"חמדן צעד אחד ({f0(grd['score_mean'])})", "value": grd["score_mean"], "offset": -8},
            {"label": f"אקראי ({f0(rnd['score_mean'])})", "value": rnd["score_mean"], "offset": 8},
        ],
        "xTitle": "מעברים (מיליונים)", "yTitle": "ניקוד ממוצע",
        "table": {"columns": ["מעברים (מיליונים)", "הערכה חמדנית", "ניקוד באימון"],
                  "rows": [[round(e["transitions"] / 1e6, 2), round(e["eval"]["score_mean"]), ""] for e in evals]},
    }

    # --- התפלגות האריח המקסימלי לאורך האימון (ערימה, סולם סדור) ---
    tiles_keys = ["256", "512", "1024", "2048", "4096"]
    labels_ev = [f"{e['transitions'] / 1e6:.0f}M" for e in evals]
    dist_rows = []
    for e in evals:
        d = e["eval"]["max_tile_dist"]
        n = e["eval"]["games"]
        row = {"≤128": sum(v for k, v in d.items() if int(k) <= 128) / n}
        for k in tiles_keys:
            row[k] = d.get(k, 0) / n
        dist_rows.append(row)
    stacked_series = []
    for i, k in enumerate(["≤128"] + tiles_keys):
        stacked_series.append({"label": k, "y": [round(100 * r[k], 1) for r in dist_rows], "color": ["de", "seq0", "seq1", "seq2", "seq3", "seq4"][i]})

    # --- הפסדים ---
    if kind == "dqn":
        loss_cfg = {"x": x_tr, "series": [{"label": "הפסד Huber", "y": [r["loss"] for r in train], "color": color}], "xTitle": "מעברים (מיליונים)", "yTitle": "הפסד"}
        q_cfg = {"x": x_tr, "series": [{"label": "ערך Q ממוצע של הפעולות שנבחרו", "y": [r["q_mean"] for r in train], "color": color}], "xTitle": "מעברים (מיליונים)", "yTitle": "Q (יחידות של 1,000 נקודות)"}
        loss_title, loss_desc = "הפסד TD במהלך האימון", "ההפסד גדל עם הזמן, וזה צפוי: ככל שהסוכן מגיע לאריחים גדולים יותר, התגמולים והערכים גדולים יותר, ולכן גם השגיאות המוחלטות."
        q_title, q_desc = "ערכי Q במהלך האימון", "ה-Q הממוצע של הפעולות שנבחרו באימון. עלייה מונוטונית פירושה שהסוכן צופה יותר ויותר נקודות מהמצבים שהוא מבקר בהם."
        third_cfg = {"x": x_tr, "series": [{"label": "epsilon", "y": [r["epsilon"] for r in train], "color": color}], "xTitle": "מעברים (מיליונים)", "yTitle": "epsilon", "yMax": 1}
        third_title, third_desc = "לוח זמנים של החקירה", "epsilon יורד לינארית מ-1 ל-0.02 ברבע הראשון של האימון. אחרי זה 2% מהמהלכים אקראיים."
    else:
        loss_cfg = {"x": x_tr, "series": [{"label": "הפסד המבקר (ערך)", "y": [r["v_loss"] for r in train], "color": color}], "xTitle": "מעברים (מיליונים)", "yTitle": "הפסד"}
        q_cfg = {"x": x_tr, "series": [{"label": "אנטרופיית המדיניות", "y": [r["entropy"] for r in train], "color": color}], "xTitle": "מעברים (מיליונים)", "yTitle": "אנטרופיה (nats)", "yMax": 1.4}
        loss_title, loss_desc = "הפסד המבקר במהלך האימון", "ריבוע הפער בין V(s) להחזר המשוער. גדל כשהערכים גדלים, כמו ב-DQN."
        q_title, q_desc = "אנטרופיית המדיניות", "אנטרופיה מקסימלית (ln 4 ≈ 1.39) היא מדיניות אחידה לגמרי. ירידה פירושה שהמדיניות נעשית בטוחה יותר בבחירותיה. בונוס האנטרופיה מונע ממנה לרדת לאפס."
        third_cfg = {"x": x_tr, "series": [{"label": "הפסד המדיניות", "y": [r["pg_loss"] for r in train], "color": color}], "xTitle": "מעברים (מיליונים)", "yTitle": "הפסד", "yZero": False}
        third_title, third_desc = "הפסד המדיניות", "עם יתרונות מנורמלים הערך הזה מרכז סביב אפס ואינו מדד לאיכות; הוא מוצג לשלמות."

    # --- הערכה סופית ---
    width = 2000
    max_score = max(max(ev["scores"]), max(baselines["greedy"]["scores"]))
    labels_h, counts_h = hist(ev["scores"], width, max_score)
    h_cfg = {"labels": labels_h, "series": [{"label": "משחקים", "y": counts_h, "color": color}], "yTitle": "מספר משחקים", "xTitle": "ניקוד (אלפים)",
             "table": {"columns": ["טווח ניקוד", "משחקים"], "rows": [[l, c] for l, c in zip(labels_h, counts_h)]}}
    td = tile_dist(ev["max_tiles"])
    td_cfg = {"labels": [str(t) for t in TILE_ORDER], "series": [{"label": "משחקים", "y": td, "color": color}], "yTitle": "מספר משחקים", "xTitle": "האריח הגדול ביותר במשחק",
              "table": {"columns": ["אריח", "משחקים", "אחוז"], "rows": [[str(t), c, f"{100 * c / len(ev['max_tiles']):.1f}%"] for t, c in zip(TILE_ORDER, td)]}}
    pts = [{"x": m, "y": s} for m, s in zip(ev["moves"], ev["scores"])][:600]
    sc_cfg = {"series": [{"label": A["label"], "points": pts, "color": color}], "xTitle": "מהלכים", "yTitle": "ניקוד"}
    ac_counts = ev.get("action_counts", [0, 0, 0, 0])
    tot = max(1, sum(ac_counts))
    act_cfg = {"labels": ["↑ למעלה", "→ ימינה", "↓ למטה", "← שמאלה"], "series": [{"label": "אחוז מהמהלכים", "y": [round(100 * c / tot, 1) for c in ac_counts], "color": color}], "yTitle": "אחוז מהמהלכים", "yMax": 100}
    base_cfg = {"labels": ["אקראי", "חמדן צעד אחד", A["label"]],
                "series": [{"label": "ניקוד ממוצע", "y": [round(rnd["score_mean"]), round(grd["score_mean"]), round(S["score_mean"])], "color": ["de", "de", color]}],
                "yTitle": "ניקוד ממוצע ב-1,000 משחקים", "legend": False}

    # --- ספים ---
    th_rows = thresholds_table(evals, [3000, 5000, 8000, 12000, 16000, 20000])
    th_html = "".join(
        f"<tr><td>{f0(th)}</td><td class='num'>{(f'{t / 1e6:.1f}M' if t else 'לא הושג')}</td><td class='num'>{(f'{m:.0f}' if m else '')}</td></tr>"
        for th, t, m in th_rows)

    ablation_html = ""
    if kind == "dqn":
        ab = {}
        for nm, lbl in (("ablation_dqn_1step_2.5M", "צעד אחד (הגרסה שנבחרה)"), ("ablation_dqn_3step_2.5M", "3 צעדים")):
            t_ab, _ = load_log(nm)
            if t_ab:
                ab[lbl] = t_ab
        if len(ab) == 2:
            xs = [r["transitions"] / 1e6 for r in list(ab.values())[0]]
            ab_cfg = {"x": xs, "series": [{"label": lbl, "x": [r["transitions"] / 1e6 for r in rows], "y": [r["recent_score_mean"] for r in rows], "color": c, "direct": True}
                                            for (lbl, rows), c in zip(ab.items(), (color, "s3"))],
                      "xTitle": "מעברים (מיליונים)", "yTitle": "ניקוד באימון (200 משחקים אחרונים)", "padRight": 130}
            ablation_html = card("c_ab", "ניסוי הסרה: החזר של צעד אחד מול 3 צעדים", "שתי ריצות קצרות (2.5 מיליון מעברים, epsilon יורד ל-0.02 אחרי מיליון) שזהות בכל דבר חוץ מאורך ההחזר. ניקוד האימון (כולל חקירה) אינו מוטה. ההסבר בסעיף הניתוח.", "chart", f"R.lineChart('c_ab', {j(ab_cfg)});")

    toc = [("summary", "תקציר"), ("learning", "עקומת הלמידה"), ("final", "ההערכה הסופית"), ("replay", "צפייה במשחק"), ("analysis", "ניתוח"), ("hp", "היפר-פרמטרים")]
    body = f"""
<section id="summary">
<h2>תקציר</h2>
<p class="lead">{A['long']} אומן {total_min:.0f} דקות על {total_trans / 1e6:.0f} מיליון מהלכים. בהערכה סופית של 1,000 משחקים (בלי חקירה) הוא הגיע לניקוד ממוצע של <b>{f0(S['score_mean'])}</b>,
הגיע לאריח 1024 ב-<b>{pct(S['reach_1024'])}</b> מהמשחקים ול-2048 ב-<b>{pct(S['reach_2048'])}</b>. לשם השוואה, מדיניות אקראית מגיעה ל-{f0(rnd['score_mean'])} וחמדן של צעד אחד ל-{f0(grd['score_mean'])}.</p>
<div class="kpis">{kpis}</div>
<p class="note">חומרה: {HARDWARE}. הניקוד הוא ניקוד המשחק המקורי (סכום כל האריחים שנוצרו במיזוג).</p>
</section>

<section id="learning">
<h2>עקומת הלמידה</h2>
{card("c_lc", "ניקוד לאורך האימון", "הקו העבה: הערכה חמדנית של 100 משחקים בכל נקודת ביקורת (המדד האמיתי). הקו הדק: ממוצע 200 המשחקים האחרונים באימון עצמו, כולל מהלכי החקירה. הקווים האפורים: קווי הבסיס.", "chart tall", f"R.lineChart('c_lc', {j(lc)});")}
{card("c_tiles", "האריח הגדול ביותר, לפי נקודת ביקורת", "אחוז המשחקים בהערכה שהסתיימו עם כל אריח מקסימלי. כחול כהה יותר = אריח גדול יותר.", "chart", f"R.barChart('c_tiles', {j({'labels': labels_ev, 'series': stacked_series, 'stacked': True, 'yTitle': 'אחוז מהמשחקים', 'yMax': 100, 'xTitle': 'מעברים'})});")}
<div class="grid2">
{card("c_loss", loss_title, loss_desc, "chart short", f"R.lineChart('c_loss', {j(loss_cfg)});")}
{card("c_q", q_title, q_desc, "chart short", f"R.lineChart('c_q', {j(q_cfg)});")}
</div>
{card("c_third", third_title, third_desc, "chart short", f"R.lineChart('c_third', {j(third_cfg)});")}
<div class="card"><h3>מתי הושג כל סף</h3><p class="desc">הנקודה הראשונה שבה ההערכה החמדנית עברה את הסף (ממוצע 100 משחקים).</p>
<table><thead><tr><th>ניקוד ממוצע</th><th class="num">מעברים</th><th class="num">דקות</th></tr></thead><tbody>{th_html}</tbody></table></div>
{ablation_html}
</section>

<section id="final">
<h2>ההערכה הסופית: 1,000 משחקים</h2>
<p>נקודת הביקורת הטובה ביותר (<code>{ev.get('checkpoint', '')}</code>) שיחקה 1,000 משחקים עם זרע קבוע, בלי חקירה. אותם 1,000 משחקים (אותם אריחים אקראיים) שימשו גם לקווי הבסיס.</p>
{card("c_base", "מול קווי הבסיס", "ניקוד ממוצע ב-1,000 משחקים.", "chart short", f"R.barChart('c_base', {j(base_cfg)});")}
<div class="grid2">
{card("c_hist", "התפלגות הניקוד", f"רוחב כל עמודה {width:,} נקודות. חציון {f0(S['score_median'])}, סטיית תקן {f0(S['score_std'])}.", "chart", f"R.barChart('c_hist', {j(h_cfg)});")}
{card("c_td", "האריח הגדול ביותר", "עם איזה אריח מקסימלי נגמר כל משחק.", "chart", f"R.barChart('c_td', {j(td_cfg)});")}
</div>
{card("c_sc", "ניקוד מול אורך המשחק", "כל נקודה היא משחק (600 הראשונים). משחק ארוך יותר הוא כמעט תמיד משחק טוב יותר: הניקוד נצבר ממיזוגים, וכל מיזוג הוא מהלך.", "chart", f"R.scatterChart('c_sc', {j(sc_cfg)});")}
<h3>מה האסטרטגיה שנלמדה?</h3>
<p>שחקנים טובים ב-2048 שומרים את האריח הגדול בפינה ומשתמשים בעיקר בשני כיוונים, ומשתמשים בכיוון השלישי רק כשאין ברירה. אפשר לבדוק אם הסוכן גילה את זה לבד, מתוך 40 המשחקים המוקלטים.</p>
<div class="grid2">
{card("c_act", "אילו כיוונים הסוכן בוחר", "התפלגות המהלכים ב-40 משחקים מוקלטים.", "chart short", f"R.barChart('c_act', {j(act_cfg)});")}
<div class="card"><h3>האריח הגדול בפינה</h3><p class="desc">אחוז המהלכים (מרגע שיש אריח 32 ומעלה) שבהם האריח הגדול ביותר נמצא באחת מארבע הפינות.</p>
<div class="kpis"><div class="kpi {color}"><div class="label">{A['label']}</div><div class="value">{pct(ev.get('corner_rate', 0))}</div></div>
<div class="kpi"><div class="label">חמדן צעד אחד</div><div class="value">{pct(baselines['greedy'].get('corner_rate', 0))}</div></div>
<div class="kpi"><div class="label">אקראי</div><div class="value">{pct(baselines['random'].get('corner_rate', 0))}</div></div></div></div>
</div>
</section>

<section id="replay">
<h2>צפייה במשחק</h2>
<p>שני משחקים מוקלטים של הסוכן. אפשר לנגן, לגרור את הסרגל או להתקדם צעד-צעד.</p>
<div class="card"><h3>המשחק הטוב ביותר מתוך 40 מוקלטים ({f0(ev['best_game']['score'])} נקודות, אריח {f0(ev['best_game']['max_tile'])})</h3><div id="rp_best"></div></div>
<script>R.replay('rp_best', {j(ev['best_game'])});</script>
<div class="card"><h3>משחק טיפוסי (חציוני, {f0(ev['typical_game']['score'])} נקודות, אריח {f0(ev['typical_game']['max_tile'])})</h3><div id="rp_typ"></div></div>
<script>R.replay('rp_typ', {j(ev['typical_game'])});</script>
</section>

<section id="analysis">
<h2>ניתוח</h2>
{analysis_text(kind)}
</section>

<section id="hp">
<h2>היפר-פרמטרים</h2>
{hp_list(args, DQN_HP if kind == 'dqn' else AC_HP)}
<p class="note" style="margin-top:12px">הרשת: קלט one-hot (16, 4, 4) → שתי שכבות Conv 3x3 עם {args.get('filters', 128)} פילטרים → Linear → {args.get('hidden', 256)} → ראש. כ-690 אלף פרמטרים. הפירוט המלא ב-<a href="../README.md">README</a>.</p>
</section>
"""
    other = "ac" if kind == "dqn" else "dqn"
    crumbs = f'<a href="../README.md">README</a> · <a href="../game/index.html">המשחק</a> · <a href="{AGENTS[other]["report"]}">דו"ח {AGENTS[other]["label"]}</a> · <a href="comparison.html">השוואה</a>'
    html = page(f"{A['label']} לומד לשחק 2048", f"""דו"ח ביצועים: {A['long']}, אימון מקומי על GPU""",
                f"אומן ב-{dt.date.today().strftime('%d.%m.%Y')} · {total_min:.0f} דקות · {total_trans / 1e6:.0f} מיליון מעברים", toc, body, crumbs)
    path = os.path.join(OUT, A["report"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", os.path.relpath(path, ROOT))
    return {"summary": S, "train": train, "evals": evals, "eval": ev, "minutes": total_min, "transitions": total_trans, "sps": sps, "args": args}


# ----------------------------------------------------------------------------
# דו"ח השוואה
# ----------------------------------------------------------------------------

def comparison_report(R: dict, baselines: dict):
    if "dqn" not in R or "ac" not in R:
        print("skip comparison: need both agents")
        return
    D, C = R["dqn"], R["ac"]
    rnd, grd = baselines["random"]["summary"], baselines["greedy"]["summary"]

    def two_kpis(label, fd, fc, sub=""):
        return kpi(label, fd, sub, "s1") + kpi(label, fc, sub, "s2")

    kpis = "".join([
        kpi("DQN: ניקוד ממוצע", f0(D["summary"]["score_mean"]), cls="s1"),
        kpi("Actor-Critic: ניקוד ממוצע", f0(C["summary"]["score_mean"]), cls="s2"),
        kpi("DQN: הגיע ל-2048", pct(D["summary"]["reach_2048"]), cls="s1", delta=f"ל-1024: {pct(D['summary']['reach_1024'])}"),
        kpi("Actor-Critic: הגיע ל-2048", pct(C["summary"]["reach_2048"]), cls="s2", delta=f"ל-1024: {pct(C['summary']['reach_1024'])}"),
        kpi("DQN: מעברים", f"{D['transitions'] / 1e6:.0f}", "מיליון", cls="s1", delta=f"{D['minutes']:.0f} דקות, {D['sps']:,} בשנייה"),
        kpi("Actor-Critic: מעברים", f"{C['transitions'] / 1e6:.0f}", "מיליון", cls="s2", delta=f"{C['minutes']:.0f} דקות, {C['sps']:,} בשנייה"),
    ])

    def curve(res, key):
        return [e[key] for e in res["evals"]]

    lc_trans = {
        "x": [], "series": [
            {"label": "DQN", "x": [t / 1e6 for t in curve(D, "transitions")], "y": [e["eval"]["score_mean"] for e in D["evals"]], "color": "s1", "width": 2.5, "direct": True},
            {"label": "Actor-Critic", "x": [t / 1e6 for t in curve(C, "transitions")], "y": [e["eval"]["score_mean"] for e in C["evals"]], "color": "s2", "width": 2.5, "direct": True},
        ],
        "xTitle": "מעברים (מיליונים, סולם לוגריתמי)", "yTitle": "ניקוד ממוצע בהערכה", "xLog": True, "padRight": 90,
        "table": {"columns": ["אלגוריתם", "מעברים (מיליונים)", "ניקוד"],
                  "rows": [["DQN", round(e["transitions"] / 1e6, 1), round(e["eval"]["score_mean"])] for e in D["evals"]] +
                          [["Actor-Critic", round(e["transitions"] / 1e6, 1), round(e["eval"]["score_mean"])] for e in C["evals"]]},
    }
    lc_trans["x"] = sorted(lc_trans["series"][0]["x"] + lc_trans["series"][1]["x"])
    lc_time = {
        "x": [], "series": [
            {"label": "DQN", "x": [e["elapsed_sec"] / 60 for e in D["evals"]], "y": [e["eval"]["score_mean"] for e in D["evals"]], "color": "s1", "width": 2.5, "direct": True},
            {"label": "Actor-Critic", "x": [e["elapsed_sec"] / 60 for e in C["evals"]], "y": [e["eval"]["score_mean"] for e in C["evals"]], "color": "s2", "width": 2.5, "direct": True},
        ],
        "refs": [{"label": f"חמדן ({f0(grd['score_mean'])})", "value": grd["score_mean"]}],
        "xTitle": "זמן אימון (דקות)", "yTitle": "ניקוד ממוצע בהערכה", "padRight": 90,
    }
    lc_time["x"] = sorted(lc_time["series"][0]["x"] + lc_time["series"][1]["x"])
    reach = {
        "x": lc_time["x"], "series": [
            {"label": "DQN: הגיע ל-1024", "x": [e["elapsed_sec"] / 60 for e in D["evals"]], "y": [100 * e["eval"]["reach_1024"] for e in D["evals"]], "color": "s1", "width": 2.5},
            {"label": "Actor-Critic: הגיע ל-1024", "x": [e["elapsed_sec"] / 60 for e in C["evals"]], "y": [100 * e["eval"]["reach_1024"] for e in C["evals"]], "color": "s2", "width": 2.5},
            {"label": "DQN: הגיע ל-2048", "x": [e["elapsed_sec"] / 60 for e in D["evals"]], "y": [100 * e["eval"]["reach_2048"] for e in D["evals"]], "color": "s1", "width": 1.2},
            {"label": "Actor-Critic: הגיע ל-2048", "x": [e["elapsed_sec"] / 60 for e in C["evals"]], "y": [100 * e["eval"]["reach_2048"] for e in C["evals"]], "color": "s2", "width": 1.2},
        ],
        "xTitle": "זמן אימון (דקות)", "yTitle": "אחוז מהמשחקים", "yMax": 100,
    }

    width = 2000
    max_score = max(max(D["eval"]["scores"]), max(C["eval"]["scores"]))
    lh, ch_d = hist(D["eval"]["scores"], width, max_score)
    _, ch_c = hist(C["eval"]["scores"], width, max_score)
    h_cfg = {"labels": lh, "series": [{"label": "DQN", "y": ch_d, "color": "s1"}, {"label": "Actor-Critic", "y": ch_c, "color": "s2"}],
             "yTitle": "מספר משחקים", "xTitle": "ניקוד (אלפים)",
             "table": {"columns": ["טווח", "DQN", "Actor-Critic"], "rows": [[l, a, b] for l, a, b in zip(lh, ch_d, ch_c)]}}
    td_d, td_c = tile_dist(D["eval"]["max_tiles"]), tile_dist(C["eval"]["max_tiles"])
    td_cfg = {"labels": [str(t) for t in TILE_ORDER], "series": [{"label": "DQN", "y": td_d, "color": "s1"}, {"label": "Actor-Critic", "y": td_c, "color": "s2"}],
              "yTitle": "מספר משחקים", "xTitle": "האריח הגדול ביותר במשחק",
              "table": {"columns": ["אריח", "DQN", "Actor-Critic"], "rows": [[str(t), a, b] for t, a, b in zip(TILE_ORDER, td_d, td_c)]}}
    base_cfg = {"labels": ["אקראי", "חמדן צעד אחד", "DQN", "Actor-Critic"],
                "series": [{"label": "ניקוד ממוצע", "y": [round(rnd["score_mean"]), round(grd["score_mean"]), round(D["summary"]["score_mean"]), round(C["summary"]["score_mean"])], "color": ["de", "de", "s1", "s2"]}],
                "yTitle": "ניקוד ממוצע ב-1,000 משחקים", "legend": False}

    def act_pct(ev):
        c = ev.get("action_counts", [0, 0, 0, 0]); t = max(1, sum(c))
        return [round(100 * v / t, 1) for v in c]
    act_cmp = {"labels": ["↑ למעלה", "→ ימינה", "↓ למטה", "← שמאלה"],
               "series": [{"label": "DQN", "y": act_pct(D["eval"]), "color": "s1"}, {"label": "Actor-Critic", "y": act_pct(C["eval"]), "color": "s2"}],
               "yTitle": "אחוז מהמהלכים", "yMax": 100}

    # טבלת סיכום
    def row(label, fd, fc, hl=False):
        return f"<tr class='{'hl' if hl else ''}'><td>{label}</td><td class='num'>{fd}</td><td class='num'>{fc}</td></tr>"
    SD, SC = D["summary"], C["summary"]
    summary_rows = "".join([
        row("ניקוד ממוצע", f0(SD["score_mean"]), f0(SC["score_mean"]), True),
        row("חציון", f0(SD["score_median"]), f0(SC["score_median"])),
        row("סטיית תקן", f0(SD["score_std"]), f0(SC["score_std"])),
        row("המשחק הטוב ביותר", f0(SD["score_max"]), f0(SC["score_max"])),
        row("המשחק הגרוע ביותר", f0(SD["score_min"]), f0(SC["score_min"])),
        row("אורך משחק ממוצע (מהלכים)", f0(SD["moves_mean"]), f0(SC["moves_mean"])),
        row("הגיע ל-512", pct(SD["reach_512"]), pct(SC["reach_512"])),
        row("הגיע ל-1024", pct(SD["reach_1024"]), pct(SC["reach_1024"]), True),
        row("הגיע ל-2048", pct(SD["reach_2048"]), pct(SC["reach_2048"]), True),
        row("הגיע ל-4096", pct(SD["reach_4096"]), pct(SC["reach_4096"])),
        row("זמן אימון (דקות)", f"{D['minutes']:.0f}", f"{C['minutes']:.0f}"),
        row("מעברים באימון", f"{D['transitions'] / 1e6:.1f}M", f"{C['transitions'] / 1e6:.1f}M"),
        row("מעברים בשנייה", f"{D['sps']:,}", f"{C['sps']:,}"),
        row("דגימות אימון לכל מעבר", f"{D['args'].get('grad_steps', 2) * D['args'].get('batch_size', 256) / D['args'].get('n_envs', 64):.0f}", "1"),
    ])
    thresholds = [3000, 5000, 8000, 12000, 16000, 20000]
    thd, thc = thresholds_table(D["evals"], thresholds), thresholds_table(C["evals"], thresholds)
    th_rows = "".join(
        f"<tr><td>{f0(th)}</td><td class='num'>{(f'{a[1] / 1e6:.1f}M' if a[1] else 'לא הושג')}</td><td class='num'>{(f'{a[2]:.0f}' if a[2] else '')}</td>"
        f"<td class='num'>{(f'{b[1] / 1e6:.1f}M' if b[1] else 'לא הושג')}</td><td class='num'>{(f'{b[2]:.0f}' if b[2] else '')}</td></tr>"
        for th, a, b in zip(thresholds, thd, thc))

    diff_rows = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in [
        ("מה הרשת לומדת", "ערכי Q(s, a) לכל כיוון", "התפלגות על הכיוונים (שחקן) + ערך המצב (מבקר)"),
        ("איך נבחרת פעולה", "argmax על Q (עם epsilon לחקירה)", "דגימה מההתפלגות (הערכה: argmax)"),
        ("חקירה", "epsilon-greedy, יורד ל-0.02", "בונוס אנטרופיה 0.01"),
        ("שימוש בנתונים", "off-policy: זיכרון חוויות של 500K, כל מעבר נדגם ~8 פעמים", "on-policy: כל אצווה פעם אחת ונזרקת"),
        ("יעד הלמידה", "r + γ·Q_target(s', argmax Q)", "יתרון GAE(λ=0.95) למדיניות, החזר למבקר"),
        ("עדכונים", "2 צעדי גרדיאנט (אצווה 256) על כל 64 מעברים", "צעד גרדיאנט אחד על כל 1,024 מעברים"),
        ("יציבות", "רשת מטרה, Double DQN, Huber", "נרמול יתרונות, חיתוך גרדיאנט 0.5"),
    ])

    toc = [("summary", "תקציר"), ("curves", "עקומות למידה"), ("final", "ההערכה הסופית"), ("table", "טבלת השוואה"), ("replay", "משחקים"), ("discussion", "דיון")]
    body = f"""
<section id="summary">
<h2>תקציר</h2>
<p class="lead">שני אלגוריתמים, אותה רשת, אותה סביבה, אותו תגמול, אותם 1,000 משחקי מבחן. DQN אומן {D['minutes']:.0f} דקות ({D['transitions'] / 1e6:.0f} מיליון מעברים) ו-Actor-Critic {C['minutes']:.0f} דקות ({C['transitions'] / 1e6:.0f} מיליון מעברים).
ניקוד ממוצע: DQN <b>{f0(SD['score_mean'])}</b>, Actor-Critic <b>{f0(SC['score_mean'])}</b>. הגעה ל-2048: DQN <b>{pct(SD['reach_2048'])}</b>, Actor-Critic <b>{pct(SC['reach_2048'])}</b>.</p>
<div class="kpis">{kpis}</div>
<p class="note">הדו"חות הנפרדים: <a href="dqn_report.html">DQN</a> · <a href="ac_report.html">Actor-Critic</a>. חומרה: {HARDWARE}.</p>
</section>

<section id="curves">
<h2>עקומות למידה</h2>
<p>ההשוואה נעשית על שני צירים, כי לשני האלגוריתמים יש יחס שונה בין חישוב לנתונים. DQN מבצע 8 דגימות אימון על כל מעבר, Actor-Critic דגימה אחת. לכן Actor-Critic צורך מעברים מהר בהרבה, אבל מפיק מכל מעבר פחות.</p>
{card("c_time", "ניקוד מול זמן אימון", "מה מקבלים מכל דקה של GPU. ההערכה החמדנית של 100 משחקים בכל נקודת ביקורת.", "chart tall", f"R.lineChart('c_time', {j(lc_time)});")}
{card("c_trans", "ניקוד מול מספר מעברים", "יעילות דגימה: כמה ניסיון צריך כל אלגוריתם כדי להגיע לאותה רמה. ציר ה-x לוגריתמי.", "chart tall", f"R.lineChart('c_trans', {j(lc_trans)});")}
{card("c_reach", "הגעה ל-1024 ול-2048 לאורך האימון", "קו עבה: 1024. קו דק: 2048.", "chart", f"R.lineChart('c_reach', {j(reach)});")}
<div class="card"><h3>מתי הושג כל סף</h3><p class="desc">הנקודה הראשונה שבה ההערכה החמדנית עברה את הסף.</p>
<table><thead><tr><th>ניקוד ממוצע</th><th class="num"><span class="swatch s1"></span>DQN מעברים</th><th class="num">DQN דקות</th><th class="num"><span class="swatch s2"></span>A2C מעברים</th><th class="num">A2C דקות</th></tr></thead><tbody>{th_rows}</tbody></table></div>
</section>

<section id="final">
<h2>ההערכה הסופית: 1,000 משחקים לכל סוכן</h2>
{card("c_base", "ניקוד ממוצע מול קווי הבסיס", "אותם 1,000 משחקים לכל המדיניויות.", "chart short", f"R.barChart('c_base', {j(base_cfg)});")}
<div class="grid2">
{card("c_hist", "התפלגות הניקוד", f"רוחב עמודה {width:,} נקודות.", "chart", f"R.barChart('c_hist', {j(h_cfg)});")}
{card("c_td", "האריח הגדול ביותר", "עם איזה אריח מקסימלי נגמר כל משחק.", "chart", f"R.barChart('c_td', {j(td_cfg)});")}
</div>
<h3>האסטרטגיה שכל סוכן למד</h3>
<div class="grid2">
{card("c_act", "העדפת כיוונים", "התפלגות המהלכים ב-40 משחקים מוקלטים לכל סוכן.", "chart short", f"R.barChart('c_act', {j(act_cmp)});")}
<div class="card"><h3>האריח הגדול בפינה</h3><p class="desc">אחוז המהלכים (מאריח 32 ומעלה) שבהם האריח הגדול ביותר יושב בפינה.</p>
<div class="kpis"><div class="kpi s1"><div class="label">DQN</div><div class="value">{pct(D['eval'].get('corner_rate', 0))}</div></div>
<div class="kpi s2"><div class="label">Actor-Critic</div><div class="value">{pct(C['eval'].get('corner_rate', 0))}</div></div>
<div class="kpi"><div class="label">חמדן צעד אחד</div><div class="value">{pct(baselines['greedy'].get('corner_rate', 0))}</div></div></div></div>
</div>
</section>

<section id="table">
<h2>טבלת השוואה</h2>
<table><thead><tr><th>מדד</th><th class="num"><span class="swatch s1"></span>DQN</th><th class="num"><span class="swatch s2"></span>Actor-Critic</th></tr></thead><tbody>{summary_rows}</tbody></table>
<h3>ההבדלים האלגוריתמיים</h3>
<table><thead><tr><th>היבט</th><th>DQN</th><th>Actor-Critic (A2C)</th></tr></thead><tbody>{diff_rows}</tbody></table>
</section>

<section id="replay">
<h2>המשחקים הטובים ביותר</h2>
<div class="grid2">
<div class="card"><h3><span class="swatch s1"></span>DQN: {f0(D['eval']['best_game']['score'])} נקודות, אריח {f0(D['eval']['best_game']['max_tile'])}</h3><div id="rp_d"></div></div>
<div class="card"><h3><span class="swatch s2"></span>Actor-Critic: {f0(C['eval']['best_game']['score'])} נקודות, אריח {f0(C['eval']['best_game']['max_tile'])}</h3><div id="rp_c"></div></div>
</div>
<script>R.replay('rp_d', {j(D['eval']['best_game'])}); R.replay('rp_c', {j(C['eval']['best_game'])});</script>
</section>

<section id="discussion">
<h2>דיון</h2>
{analysis_text("comparison")}
</section>
"""
    crumbs = '<a href="../README.md">README</a> · <a href="../game/index.html">המשחק</a> · <a href="dqn_report.html">דו"ח DQN</a> · <a href="ac_report.html">דו"ח Actor-Critic</a>'
    html = page("DQN מול Actor-Critic ב-2048", "השוואת ביצועים של שני אלגוריתמי למידת חיזוק על אותו משחק, אותה רשת ואותה חומרה",
                f"נוצר ב-{dt.date.today().strftime('%d.%m.%Y')}", toc, body, crumbs)
    path = os.path.join(OUT, "comparison.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", os.path.relpath(path, ROOT))


def update_readme(R: dict, baselines: dict):
    path = os.path.join(ROOT, "README.md")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    rows = ["| מדיניות | ניקוד ממוצע | חציון | הטוב ביותר | הגיע ל-1024 | הגיע ל-2048 | אימון |", "|---|---|---|---|---|---|---|"]
    for name, label in (("random", "אקראי"), ("greedy", "חמדן צעד אחד")):
        s = baselines[name]["summary"]
        rows.append(f"| {label} | {f0(s['score_mean'])} | {f0(s['score_median'])} | {f0(s['score_max'])} | {pct(s['reach_1024'])} | {pct(s['reach_2048'])} | |")
    for kind in ("dqn", "ac"):
        if kind in R:
            s = R[kind]["summary"]
            rows.append(f"| **{AGENTS[kind]['label']}** | **{f0(s['score_mean'])}** | {f0(s['score_median'])} | {f0(s['score_max'])} | {pct(s['reach_1024'])} | {pct(s['reach_2048'])} | {R[kind]['minutes']:.0f} דק', {R[kind]['transitions'] / 1e6:.0f}M מעברים |")
    table = "\n".join(rows)
    new = re.sub(r"<!-- RESULTS_TABLE -->.*?(?=\n---)", f"<!-- RESULTS_TABLE -->\n{table}\n", text, flags=re.S)
    if "<!-- RESULTS_TABLE -->" in text and new == text:
        new = text.replace("<!-- RESULTS_TABLE -->", f"<!-- RESULTS_TABLE -->\n{table}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
    print("updated README results table")


def main():
    baselines = {n: load_eval(n) for n in ("random", "greedy")}
    if any(v is None for v in baselines.values()):
        raise SystemExit("run rl/evaluate.py --all first (baselines missing)")
    R = {}
    for kind in ("dqn", "ac"):
        r = algo_report(kind, baselines)
        if r:
            R[kind] = r
    comparison_report(R, baselines)
    update_readme(R, baselines)


if __name__ == "__main__":
    main()
