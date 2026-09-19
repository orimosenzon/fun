"""
בניית דו"חות ה-HTML מהלוגים ומתוצאות ההערכה.

קלט:  logs/<run>_train.jsonl  (עקומות למידה),  reports/data/eval_<name>.json  (הערכה סופית)
       reports/data/analysis_<name>.html (ניתוח מילולי), reports/data/method_<name>.html (הסבר השיטה:
       אינטואיציה, היסטוריה, מתמטיקה), methods_intro.html ו-methods_outro.html (המסגרת המשותפת וההבדלים)
פלט:  reports/{dqn,ac,ppo,ntuple}_report.html, reports/comparison.html
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
HARDWARE_CPU = "ליבת CPU אחת של Intel Core i5-7400 (3.0GHz) · NumPy + numba, בלי GPU"
CHARTJS = "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

# הסדר כאן הוא סדר ההצגה בכל הדו"חות. הצבעים: DQN כחול, A2C כתום, PPO ירוק, N-Tuple צהוב.
AGENTS = {
    "dqn": {"run": "dqn", "label": "DQN", "color": "s1", "report": "dqn_report.html", "long": "Deep Q-Network", "hw": HARDWARE},
    "ac": {"run": "a2c", "label": "Actor-Critic", "color": "s2", "report": "ac_report.html", "long": "Advantage Actor-Critic (A2C)", "hw": HARDWARE},
    "ppo": {"run": "ppo", "label": "PPO", "color": "s3", "report": "ppo_report.html", "long": "Proximal Policy Optimization (PPO)", "hw": HARDWARE},
    "ntuple": {"run": "ntuple", "label": "N-Tuple TD", "color": "s4", "report": "ntuple_report.html",
               "long": "TD(0) על afterstates עם רשת n-tuple", "hw": HARDWARE_CPU},
    "asnet": {"run": "asnet", "label": "Afterstate TD (רשת)", "color": "s5", "report": "asnet_report.html",
              "long": "TD(0) על afterstates עם רשת הקונבולוציה (הניסוי המפריד: האלגוריתם של רשת ה-n-tuple, הייצוג של הרשתות העצביות)", "hw": HARDWARE},
}
KINDS = list(AGENTS)
NEURAL = ("dqn", "ac", "ppo", "asnet")
COUNT_WORDS = {2: "שני", 3: "שלושה", 4: "ארבעה", 5: "חמישה"}
COUNT_WORDS_F = {2: "שתי", 3: "שלוש", 4: "ארבע", 5: "חמש"}
DIRECTIONS = ["↑ למעלה", "→ ימינה", "↓ למטה", "← שמאלה"]
THRESHOLDS = [3000, 5000, 8000, 12000, 16000, 20000, 30000, 40000, 60000, 100000]


# ----------------------------------------------------------------------------
# טעינת נתונים
# ----------------------------------------------------------------------------

def load_log(run: str):
    train, evals = [], []
    path = os.path.join(LOGS, f"{run}_train.jsonl")
    if not os.path.exists(path):  # ריצות ההסרה נשמרו בלי הסיומת _train
        path = os.path.join(LOGS, f"{run}.jsonl")
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
    # סימן LRM בתחילת התווית כדי שהטווח לא יתהפך בציור על קנבס בדף RTL
    labels = [f"‎{int(edges[i] / 1000)}–{int(edges[i + 1] / 1000)}K" for i in range(len(counts))]
    return labels, counts.tolist()


TILE_ORDER = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def tile_order_for(*max_tile_lists) -> list[int]:
    """עמודות האריח המקסימלי: מ-64 עד האריח הגדול ביותר שמישהו הגיע אליו."""
    top = max(max(t) for t in max_tile_lists)
    return [v for v in TILE_ORDER if v <= max(top, 256)]


def tile_dist(max_tiles, order):
    t = np.asarray(max_tiles)
    return [int((t == v).sum()) for v in order]


def hist_width(max_score: float) -> int:
    if max_score < 40000:
        return 2000
    if max_score < 150000:
        return 5000
    return 20000


def thresholds_table(evals, thresholds):
    """באיזה מספר מעברים ובאיזו דקה ההערכה החמדנית עברה לראשונה כל סף."""
    rows = []
    for th in thresholds:
        hit = next((e for e in evals if e["eval"]["score_mean"] >= th), None)
        rows.append((th, hit["transitions"] if hit else None, hit["elapsed_sec"] / 60 if hit else None))
    return rows


def th_cells(row):
    """שני תאים לטבלת הספים: מעברים ודקות (או 'לא הושג')."""
    _, t, m = row
    t_str = (f"{t / 1e9:.2f}B" if t >= 1e9 else f"{t / 1e6:.1f}M") if t else "לא הושג"
    m_str = (f"{m:.1f}" if m < 10 else f"{m:.0f}") if m else ""
    return f"<td class='num'>{t_str}</td><td class='num'>{m_str}</td>"


# ----------------------------------------------------------------------------
# תבנית דף
# ----------------------------------------------------------------------------

def page(title: str, subtitle: str, meta: str, toc: list[tuple[str, str]], body: str, crumbs: str = "", math: bool = False) -> str:
    toc_html = " ".join(f'<a href="#{a}">{t}</a>' for a, t in toc)
    today = dt.date.today().strftime("%d.%m.%Y")
    katex = ""
    if math:
        # KaTeX לנוסחאות. auto-render מחפש \( \) לנוסחה בתוך שורה ו-$$ $$ לנוסחה בשורה נפרדת.
        katex = f"""<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js" onload="renderMathInElement(document.body, {{delimiters: [{{left: '$$', right: '$$', display: true}}, {{left: '\\\\(', right: '\\\\)', display: false}}], throwOnError: false}});"></script>
"""
    return f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="assets/report.css">
{katex}<script src="{CHARTJS}"></script>
<script src="assets/report.js"></script>
<script src="assets/ntuple_sim.js"></script>
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


def heatmap_html(pos, label: str) -> str:
    """מפת חום 4x4 של מיקום האריח הגדול (אחוזים), בסולם כחול רציף."""
    cells = []
    for r in range(4):
        for c in range(4):
            v = pos[r][c]
            cells.append(f'<div class="hm-cell" style="--v:{v:.3f}" title="{100 * v:.1f}%">{(f"{100 * v:.0f}" if v >= 0.005 else "")}</div>')
    return f'<div class="hm"><div class="hm-grid">{"".join(cells)}</div><div class="hm-label">{label}</div></div>'


def hp_list(args: dict, keys: list[tuple[str, str]]) -> str:
    def fmt_v(v):
        return f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) and abs(v) >= 1000 else str(v)
    items = "".join(f"<div><span>{label}</span><span>{fmt_v(args.get(k, ''))}</span></div>" for k, label in keys if k in args)
    return f'<div class="hp">{items}</div>'


def trans_fmt(n: float) -> tuple[str, str]:
    """מספר מעברים כ-(מספר, מילה): 144 מיליון, 2.86 מיליארד."""
    return (f"{n / 1e9:.2f}", "מיליארד") if n >= 1e9 else (f"{n / 1e6:.0f}", "מיליון")


def swatch(kind: str) -> str:
    return f'<span class="swatch {AGENTS[kind]["color"]}"></span>'


def crumbs_for(current: str | None, R: dict | None = None) -> str:
    """שורת הניווט העליונה: README, המשחק, דו"חות הסוכנים האחרים וההשוואה."""
    items = ['<a href="../README.md">README</a>', '<a href="../game/index.html">המשחק</a>']
    for k in KINDS:
        if k != current and (R is None or k in R):
            items.append(f'<a href="{AGENTS[k]["report"]}">דו"ח {AGENTS[k]["label"]}</a>')
    if current != "comparison":
        items.append('<a href="comparison.html">השוואה</a>')
    return " · ".join(items)


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
PPO_HP = AC_HP[:6] + [
    ("ppo_epochs", "מעברים על כל אצווה (epochs)"), ("n_minibatches", "מיני-אצוות בכל מעבר"), ("clip_eps", "epsilon של הקטימה"),
    ("clip_vloss", "קטימת הפסד המבקר"),
] + AC_HP[6:]
NTUPLE_HP = [
    ("time_limit_min", "מגבלת זמן (דקות)"), ("alpha", "קצב למידה alpha (מתחלק ב-32 משקלים)"), ("chunk", "מהלכים בכל קריאה ללולאה המקומפלת"),
    ("eval_every", "הערכה כל"), ("eval_games", "משחקים בהערכה"), ("seed", "זרע"),
]
ASNET_HP = [
    ("time_limit_min", "מגבלת זמן (דקות)"), ("n_envs", "סביבות במקביל"), ("gamma", "gamma (היוון)"), ("lr", "קצב למידה (Adam)"),
    ("grad_clip", "חיתוך גרדיאנט"), ("reward_scale", "סקאלת תגמול"), ("filters", "פילטרים בקונבולוציה"), ("hidden", "שכבה נסתרת"),
    ("eval_every", "הערכה כל"), ("eval_games", "משחקים בהערכה"), ("seed", "זרע"),
]
HP_LISTS = {"dqn": DQN_HP, "ac": AC_HP, "ppo": PPO_HP, "ntuple": NTUPLE_HP, "asnet": ASNET_HP}


def network_note(kind: str, args: dict) -> str:
    if kind == "ntuple":
        return ("הרשת: 4 טבלאות חיפוש של 16⁶ כניסות (67 מיליון משקלים, 268MB), אחת לכל חלון של 6 משבצות "
                "(השורה הראשונה עם שתי משבצות מהשנייה, השורה השנייה עם שתיים מהשלישית, מלבן 2×3 בפינה, מלבן 2×3 באמצע), "
                "כל חלון נקרא על 8 הסימטריות של הלוח. V(afterstate) = סכום 32 קריאות. בלי היוון, ניקוד גולמי. "
                "הפירוט המלא ב-<a href=\"../README.md\">README</a>.")
    head = "ראש ערך אחד (V של afterstate; הקלט הוא הלוח אחרי ההחלקה)" if kind == "asnet" else "ראש"
    return (f"הרשת: קלט one-hot (16, 4, 4) → שתי שכבות Conv 3x3 עם {args.get('filters', 128)} פילטרים → Linear → "
            f"{args.get('hidden', 256)} → {head}. כ-690 אלף פרמטרים. הפירוט המלא ב-<a href=\"../README.md\">README</a>.")


# ----------------------------------------------------------------------------
# טקסטים של ניתוח (נכתבים ידנית אחרי צפייה בתוצאות; נטענים מקובץ אם קיים)
# ----------------------------------------------------------------------------

def data_text(name: str, fallback: str) -> str:
    path = os.path.join(DATA, f"{name}.html")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return f"<p class='note'>({fallback})</p>"


def analysis_text(name: str) -> str:
    return data_text(f"analysis_{name}", "הניתוח המילולי ייכתב אחרי סיום הריצה.")


# ----------------------------------------------------------------------------
# דו"ח לאלגוריתם יחיד
# ----------------------------------------------------------------------------

def training_charts(kind: str, train: list, x_tr: list, color: str) -> str:
    """גרפי הפנים של האימון (הפסדים, אנטרופיה וכדומה), שונים לכל אלגוריתם."""
    x = "מעברים (מיליונים)"

    def lc(cid, title, desc, series, cls="chart short", **extra):
        cfg = {"x": x_tr, "series": series, "xTitle": x, **extra}
        return card(cid, title, desc, cls, f"R.lineChart('{cid}', {j(cfg)});")

    if kind == "dqn":
        return f"""<div class="grid2">
{lc("c_loss", "הפסד TD במהלך האימון", "ההפסד גדל עם הזמן, וזה צפוי: ככל שהסוכן מגיע לאריחים גדולים יותר, התגמולים והערכים גדולים יותר, ולכן גם השגיאות המוחלטות.", [{"label": "הפסד Huber", "y": [r["loss"] for r in train], "color": color}], yTitle="הפסד")}
{lc("c_q", "ערכי Q במהלך האימון", "ה-Q הממוצע של הפעולות שנבחרו באימון. עלייה מונוטונית פירושה שהסוכן צופה יותר ויותר נקודות מהמצבים שהוא מבקר בהם.", [{"label": "ערך Q ממוצע של הפעולות שנבחרו", "y": [r["q_mean"] for r in train], "color": color}], yTitle="Q (יחידות של 1,000 נקודות)")}
</div>
{lc("c_third", "לוח זמנים של החקירה", "epsilon יורד לינארית מ-1 ל-0.02 ברבע הראשון של האימון. אחרי זה 2% מהמהלכים אקראיים.", [{"label": "epsilon", "y": [r["epsilon"] for r in train], "color": color}], yTitle="epsilon", yMax=1)}
"""
    if kind == "ntuple":
        return f"""<div class="grid2">
{lc("c_loss", "שגיאת ה-TD הממוצעת", "ממוצע |delta| של העדכונים, בנקודות. גדל עם הזמן מאותה סיבה כמו ההפסד של DQN: הערכים גדלים ככל שהסוכן מגיע רחוק יותר. שגיאה של 400 נקודות על ערכים של עשרות אלפים היא כאחוז.", [{"label": "|delta| ממוצע", "y": [r["td_abs"] for r in train], "color": color}], yTitle="נקודות")}
{lc("c_q", "כמה מהטבלה בשימוש", "אחוז המשקלים (מתוך 67 מיליון) שכבר נגעו בהם. רוב הכניסות מייצגות חלונות שלא יכולים להופיע במשחק אמיתי (למשל שישה אריחים גדולים צמודים), ולכן הטבלה לא תתמלא לעולם.", [{"label": "משקלים שנגעו בהם", "y": [100 * r["visited"] for r in train], "color": color}], yTitle="אחוז מהטבלה")}
</div>
"""
    if kind == "asnet":
        return f"""<div class="grid2">
{lc("c_loss", "שגיאת ה-TD הממוצעת", "ממוצע |delta| של העדכונים, בנקודות משחק (אותו מדד כמו בדוח של רשת ה-n-tuple). גדל כשהערכים גדלים.", [{"label": "|delta| ממוצע", "y": [r["td_abs"] for r in train], "color": color}], yTitle="נקודות")}
{lc("c_q", "V הממוצע של ה-afterstates שעודכנו", "כמה נקודות הרשת צופה לצבור עוד מהלוחות שהיא מבקרת בהם. ברשת עצבית הערך יכול לזנק בתחילת האימון ואז להתייצב; בטבלה הוא מתחיל מאפס ורק עולה.", [{"label": "V ממוצע", "y": [r["v_mean"] for r in train], "color": color}], yTitle="נקודות")}
</div>
"""
    common = f"""<div class="grid2">
{lc("c_loss", "הפסד המבקר במהלך האימון", "ריבוע הפער בין V(s) להחזר המשוער. גדל כשהערכים גדלים, כמו ב-DQN.", [{"label": "הפסד המבקר (ערך)", "y": [r["v_loss"] for r in train], "color": color}], yTitle="הפסד")}
{lc("c_q", "אנטרופיית המדיניות", "אנטרופיה מקסימלית (ln 4 ≈ 1.39) היא מדיניות אחידה לגמרי. ירידה פירושה שהמדיניות נעשית בטוחה יותר בבחירותיה. בונוס האנטרופיה מונע ממנה לרדת לאפס.", [{"label": "אנטרופיית המדיניות", "y": [r["entropy"] for r in train], "color": color}], yTitle="אנטרופיה (nats)", yMax=1.4)}
</div>
"""
    if kind == "ac":
        return common + lc("c_third", "הפסד המדיניות", "עם יתרונות מנורמלים הערך הזה מרכז סביב אפס ואינו מדד לאיכות; הוא מוצג לשלמות.", [{"label": "הפסד המדיניות", "y": [r["pg_loss"] for r in train], "color": color}], yTitle="הפסד", yZero=False)
    # PPO: שני מדדי הבריאות של הקטימה
    return common + f"""<div class="grid2">
{lc("c_kl", "מרחק KL בין המדיניות הישנה לחדשה", "כמה המדיניות זזה במהלך 16 צעדי הגרדיאנט על כל אצווה (אומדן k3, ממוצע על המיני-אצוות). PPO נחשב בריא באזור 0.01 עד 0.03; קפיצות פירושן עדכונים אגרסיביים מדי.", [{"label": "KL משוער", "y": [r["approx_kl"] for r in train], "color": color}], yTitle="KL (nats)")}
{lc("c_clip", "חלק הדגימות שנקטמו", "אחוז הדגימות במיני-אצווה שיחס ההסתברויות שלהן יצא מהחלון [1-ε, 1+ε] ולכן הגרדיאנט שלהן אופס. במיני-אצווה הראשונה של כל אצווה הוא תמיד אפס, כך שהממוצע כאן הוא על כל 16 הצעדים.", [{"label": "חלק שנקטם", "y": [100 * r["clip_frac"] for r in train], "color": color}], yTitle="אחוז מהדגימות")}
</div>
"""



# ----------------------------------------------------------------------------
# ניסויים נוספים על רשת ה-n-tuple: expectimax מעל הטבלה, וריצה ארוכה עם דעיכה מאוחרת
# ----------------------------------------------------------------------------

def ntuple_experiments(evals_main: list, color: str) -> str:
    html = ""
    em = load_eval("ntuple_expectimax")
    if em and em.get("depths"):
        depths = sorted(em["depths"], key=int)
        D = {d: em["depths"][d] for d in depths}
        labels = {"1": "עומק 1 (חמדן, כמו בדו\"ח)", "2": "עומק 2", "3": "עומק 3"}
        colors = ["de", color, color]
        bar = {"labels": [labels.get(d, f"עומק {d}") for d in depths],
               "series": [{"label": "ניקוד ממוצע", "y": [round(D[d]["summary"]["score_mean"]) for d in depths], "color": [colors[i] if i < len(colors) else color for i in range(len(depths))]}],
               "yTitle": "ניקוד ממוצע", "legend": False}
        max_score = max(max(D[d]["scores"]) for d in depths)
        width = hist_width(max_score)
        h_series, lh = [], None
        seq = ["de", "seq2", "seq4"]
        for i, d in enumerate(depths):
            lh, counts = hist(D[d]["scores"], width, max_score)
            h_series.append({"label": labels.get(d, d), "y": [100 * c / D[d]["games"] for c in counts], "color": seq[i] if i < len(seq) else color})
        h_cfg = {"labels": lh, "series": h_series, "yTitle": "אחוז מהמשחקים", "xTitle": "ניקוד (אלפים)",
                 "table": {"columns": ["טווח"] + [labels.get(d, d) for d in depths], "rows": [[l] + [round(sr["y"][i], 1) for sr in h_series] for i, l in enumerate(lh)]}}
        order = tile_order_for(*(D[d]["max_tiles"] for d in depths))
        td_cfg = {"labels": [str(t) for t in order],
                  "series": [{"label": labels.get(d, d), "y": [round(100 * c / D[d]["games"], 1) for c in tile_dist(D[d]["max_tiles"], order)], "color": seq[i] if i < len(seq) else color} for i, d in enumerate(depths)],
                  "yTitle": "אחוז מהמשחקים", "xTitle": "האריח הגדול ביותר במשחק"}
        rows = ""
        for d in depths:
            S, mt = D[d]["summary"], np.asarray(D[d]["max_tiles"])
            rows += (f"<tr class='{'hl' if d != '1' else ''}'><td>{labels.get(d, d)}</td><td class='num'>{D[d]['games']:,}</td><td class='num'>{f0(S['score_mean'])}</td>"
                     f"<td class='num'>{f0(S['score_median'])}</td><td class='num'>{f0(S['score_max'])}</td><td class='num'>{pct(S['reach_2048'])}</td>"
                     f"<td class='num'>{pct(S['reach_4096'])}</td><td class='num'>{pct((mt >= 8192).mean())}</td><td class='num'>{pct((mt >= 16384).mean())}</td>"
                     f"<td class='num'>{f0(S['moves_mean'])}</td><td class='num'>{f0(D[d]['moves_per_sec'])}</td></tr>")
        rec = next((D[d].get("recorded_game") for d in depths if D[d].get("recorded_game")), None)
        rec_d = next((d for d in depths if D[d].get("recorded_game")), None)
        rec_html = ""
        if rec:
            rec_html = (f'<div class="card"><h3>משחק מוקלט של עומק {rec_d} ({f0(rec["score"])} נקודות, אריח {f0(rec["max_tile"])}, {f0(rec["moves"])} מהלכים)</h3>'
                        f'<div id="rp_em"></div></div><script>R.replay(\'rp_em\', {j(rec)}, {{start: \'end\'}});</script>')
        html += f"""
<h3>ניסוי 1: expectimax מעל הטבלה, בלי אימון נוסף</h3>
<p>אותה טבלה בדיוק, אבל במקום לבחור לפי \(r + V(s')\) של מהלך אחד קדימה, הסוכן מחשב לכל afterstate את התוחלת על האריח האקראי (עד 15 משבצות × שני ערכים) ובכל לוח שנוצר בוחר שוב את המהלך הטוב ביותר, וכן הלאה עד העומק. עומק 1 הוא הסוכן של הדו\"ח; כל רמת עומק נוספת עולה פי ~100 בחישוב (30 המשכים × 4 מהלכים), ולכן ההערכה רצה על ארבע ליבות במקביל, ועומק 3 על {D[depths[-1]]['games']:,} משחקים. הקוד ב-<code>rl/expectimax.py</code>.</p>
<div class="grid2">
{card("c_em_bar", "ניקוד ממוצע לפי עומק החיפוש", "אותה טבלה, אותם זרעים להתחלות המשחקים.", "chart short", f"R.barChart('c_em_bar', {j(bar)});")}
{card("c_em_td", "האריח הגדול ביותר, לפי עומק", "אחוז המשחקים שהסתיימו עם כל אריח מקסימלי.", "chart short", f"R.barChart('c_em_td', {j(td_cfg)});")}
</div>
{card("c_em_hist", "התפלגות הניקוד לפי עומק", f"רוחב עמודה {width:,} נקודות, באחוזים מהמשחקים של כל עומק (מספר המשחקים שונה).", "chart", f"R.barChart('c_em_hist', {j(h_cfg)});")}
<div class="card"><h3>הטבלה המלאה</h3>
<div class="scroll"><table><thead><tr><th>חיפוש</th><th class="num">משחקים</th><th class="num">ממוצע</th><th class="num">חציון</th><th class="num">הטוב ביותר</th><th class="num">2048</th><th class="num">4096</th><th class="num">8192</th><th class="num">16384</th><th class="num">מהלכים למשחק</th><th class="num">מהלכים/שנייה</th></tr></thead><tbody>{rows}</tbody></table></div></div>
{rec_html}
"""
    train_long, evals_long = load_log("ntuple_long")
    if evals_long:
        ev_long = load_eval("ntuple_long")
        t_main = [e["elapsed_sec"] / 60 for e in evals_main]
        ab_cfg = {"series": [
                      {"label": "הריצה הארוכה", "x": [e["elapsed_sec"] / 60 for e in evals_long], "y": [e["eval"]["score_mean"] for e in evals_long], "color": color, "width": 2.5},
                      {"label": "הריצה של הדו\"ח (73 דקות, קצב קבוע)", "x": t_main, "y": [e["eval"]["score_mean"] for e in evals_main], "color": "de", "width": 2.5},
                  ],
                  "xTitle": "זמן אימון (דקות)", "yTitle": "ניקוד ממוצע בהערכה (100 משחקים)",
                  "table": {"columns": ["דקות", "הריצה הארוכה", "הריצה של הדו\"ח"],
                            "rows": [[round(e["elapsed_sec"] / 60, 1), round(e["eval"]["score_mean"]), ""] for e in evals_long] +
                                    [[round(e["elapsed_sec"] / 60, 1), "", round(e["eval"]["score_mean"])] for e in evals_main]}}
        ab_cfg["x"] = sorted(set(ab_cfg["series"][0]["x"]) | set(ab_cfg["series"][1]["x"]))
        alpha_cfg = {"x": [r["elapsed_sec"] / 60 for r in train_long], "series": [{"label": "alpha", "y": [r["alpha"] for r in train_long], "color": color}],
                     "xTitle": "זמן אימון (דקות)", "yTitle": "alpha", "yMax": 0.11}
        last = train_long[-1]
        kp = ""
        if ev_long:
            S = ev_long["summary"]
            kp = '<div class="kpis cols4">' + "".join([
                kpi("ניקוד ממוצע (1,000 משחקים)", f0(S["score_mean"]), cls=color, delta=f"הריצה של הדו\"ח: 116,716"),
                kpi("חציון", f0(S["score_median"]), cls=color),
                kpi("הגיע ל-2048 / 4096", f"{pct(S['reach_2048'])} / {pct(S['reach_4096'])}", cls=color),
                kpi("הגיע ל-8192 / 16384", f"{pct(np.mean(np.asarray(ev_long['max_tiles']) >= 8192))} / {pct(np.mean(np.asarray(ev_long['max_tiles']) >= 16384))}", cls=color),
            ]) + "</div>"
        long_title = "ההערכה החמדנית לאורך הזמן: הריצה הארוכה מול הריצה של הדו\"ח"
        html += f"""
<h3>ניסוי 2: ריצה ארוכה, עם דעיכה מתונה רק בסופה</h3>
<p>{last['elapsed_sec'] / 60:.0f} דקות במקום 73, {trans_fmt(last['transitions'])[0]} {trans_fmt(last['transitions'])[1]} מעברים, {f0(last.get('episodes', 0))} משחקים. קצב הלמידה נשאר 0.1 בשלושת הרבעים הראשונים של הזמן (ניסוי ההסרה הראה שדעיכה מוקדמת מזיקה), ירד ל-0.03 ברבע האחרון ול-0.01 בעשירית האחרונה. השאלה: האם המישור של 117 אלף אחרי 43 דקות הוא של הקצב הקבוע (ואז זמן ודעיכה יעברו אותו) או של גודל הרשת (ואז לא).</p>
{kp}
<div class="grid2">
{card("c_long", long_title, "100 משחקים בכל נקודת ביקורת.", "chart", f"R.lineChart('c_long', {j(ab_cfg)});")}
{card("c_long_alpha", "לוח הזמנים של קצב הלמידה בריצה הארוכה", "0.1 → 0.03 ב-75% מהזמן → 0.01 ב-90%.", "chart short", f"R.lineChart('c_long_alpha', {j(alpha_cfg)});")}
</div>
"""
    if html:
        html += data_text("analysis_ntuple_experiments", "הניתוח של הניסויים ייכתב בקובץ reports/data/analysis_ntuple_experiments.html.")
    return html



def asnet_plain_cards(evals_main: list, train_main: list, color: str) -> str:
    """הריצה הראשונה של ניסוי 3, האלגוריתם כלשונו בלי מייצבים (asnet_plain): קרסה בדקה 18. מוצגת מול הריצה המיוצבת."""
    train_p, evals_p = load_log("asnet_plain")
    if not evals_p:
        return ""
    ev_plain = load_eval("asnet_plain")
    t = lambda e: e["elapsed_sec"] / 60
    sc = {"series": [
              {"label": "הריצה המיוצבת (gamma 0.99, רשת מטרה, Huber)", "x": [t(e) for e in evals_main], "y": [e["eval"]["score_mean"] for e in evals_main], "color": color, "width": 2.5},
              {"label": "האלגוריתם כלשונו (gamma 1, בלי רשת מטרה, ריבועי)", "x": [t(e) for e in evals_p], "y": [e["eval"]["score_mean"] for e in evals_p], "color": "de", "width": 2.5},
          ],
          "xTitle": "זמן אימון (דקות)", "yTitle": "ניקוד ממוצע בהערכה (100 משחקים)",
          "table": {"columns": ["דקות", "מיוצבת", "כלשונו"],
                    "rows": [[round(t(e), 1), round(e["eval"]["score_mean"]), ""] for e in evals_main] + [[round(t(e), 1), "", round(e["eval"]["score_mean"])] for e in evals_p]}}
    sc["x"] = sorted(set(sc["series"][0]["x"]) | set(sc["series"][1]["x"]))
    vm = {"series": [
              {"label": "מיוצבת", "x": [t(r) for r in train_main], "y": [r["v_mean"] for r in train_main], "color": color, "width": 2},
              {"label": "כלשונו", "x": [t(r) for r in train_p], "y": [r["v_mean"] for r in train_p], "color": "de", "width": 2},
          ], "xTitle": "זמן אימון (דקות)", "yTitle": "V ממוצע (נקודות)"}
    vm["x"] = sorted(set(vm["series"][0]["x"]) | set(vm["series"][1]["x"]))
    best_p = max(e["eval"]["score_mean"] for e in evals_p)
    best_min = next(t(e) for e in evals_p if e["eval"]["score_mean"] == best_p)
    plain_1000 = f0(ev_plain["summary"]["score_mean"]) if ev_plain else ""
    return f"""<div class="card"><h3>הריצה הראשונה: האלגוריתם כלשונו, וקריסה</h3>
<p class="desc">הניסוי התחיל בדיוק כמו בטבלה: gamma = 1, הפסד ריבועי, בלי רשת מטרה. הריצה הזאת (<code>asnet_plain</code>) הגיעה ל-{f0(best_p)} אחרי {best_min:.0f} דקות{(' (' + plain_1000 + ' על 1,000 משחקים בנקודת הביקורת הטובה ביותר)') if plain_1000 else ''}, ואז V זינק, והרשת קרסה לפונקציה קבועה: אותו ערך לכל לוח, ולכן בחירה לפי הניקוד המיידי בלבד (3,130, כמו החמדן), בלי התאוששות עד סוף 73 הדקות. הריצה השנייה מוסיפה את שלושת המייצבים הסטנדרטיים של השיטות העצביות. ההסבר בסעיף הניתוח.</p>
<div class="grid2">
<div><div class="chart" id="c_ab"></div></div>
<div><div class="chart" id="c_ab_v"></div></div>
</div></div>
<script>R.lineChart('c_ab', {j(sc)}); R.lineChart('c_ab_v', {j(vm)});</script>
"""


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
    updates = train[-1].get("updates") or train[-1].get("grad_steps")  # DQN קורא לזה grad_steps

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
            {"label": "ניקוד באימון (200 המשחקים האחרונים, עם חקירה)", "x": x_tr, "y": y_tr, "color": "de", "width": 1.5},
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
    # חמש העמודות הגדולות שמופיעות ב-1% לפחות באיזושהי נקודת ביקורת, וכל מה שמתחתן בעמודה אחת
    present = sorted({int(k) for e in evals for k, v in e["eval"]["max_tile_dist"].items() if v / e["eval"]["games"] >= 0.01})
    tiles_int = present[-5:] if len(present) > 5 else present
    if tiles_int and tiles_int[0] <= 128:
        tiles_int = [t for t in tiles_int if t > 128] or [256]
    tiles_keys = [str(t) for t in tiles_int]
    low_label = f"≤{tiles_int[0] // 2}"
    labels_ev = [f"{e['transitions'] / 1e6:.0f}M" for e in evals]
    dist_rows = []
    for e in evals:
        d = e["eval"]["max_tile_dist"]
        n = e["eval"]["games"]
        row = {low_label: sum(v for k, v in d.items() if int(k) < tiles_int[0]) / n}
        for k in tiles_keys:
            row[k] = d.get(k, 0) / n
        dist_rows.append(row)
    stacked_series = []
    for i, k in enumerate([low_label] + tiles_keys):
        stacked_series.append({"label": k, "y": [round(100 * r[k], 1) for r in dist_rows], "color": ["de", "seq0", "seq1", "seq2", "seq3", "seq4"][i]})

    # --- הערכה סופית ---
    max_score = max(max(ev["scores"]), max(baselines["greedy"]["scores"]))
    width = hist_width(max_score)
    labels_h, counts_h = hist(ev["scores"], width, max_score)
    h_cfg = {"labels": labels_h, "series": [{"label": "משחקים", "y": counts_h, "color": color}], "yTitle": "מספר משחקים", "xTitle": "ניקוד (אלפים)",
             "table": {"columns": ["טווח ניקוד", "משחקים"], "rows": [[l, c] for l, c in zip(labels_h, counts_h)]}}
    order = tile_order_for(ev["max_tiles"])
    td = tile_dist(ev["max_tiles"], order)
    td_cfg = {"labels": [str(t) for t in order], "series": [{"label": "משחקים", "y": td, "color": color}], "yTitle": "מספר משחקים", "xTitle": "האריח הגדול ביותר במשחק",
              "table": {"columns": ["אריח", "משחקים", "אחוז"], "rows": [[str(t), c, f"{100 * c / len(ev['max_tiles']):.1f}%"] for t, c in zip(order, td)]}}
    pts = [{"x": m, "y": s} for m, s in zip(ev["moves"], ev["scores"])][:600]
    sc_cfg = {"series": [{"label": A["label"], "points": pts, "color": color}], "xTitle": "מהלכים", "yTitle": "ניקוד"}
    ac_counts = ev.get("action_counts", [0, 0, 0, 0])
    tot = max(1, sum(ac_counts))
    act_cfg = {"labels": DIRECTIONS, "series": [{"label": "אחוז מהמהלכים", "y": [round(100 * c / tot, 1) for c in ac_counts], "color": color}], "yTitle": "אחוז מהמהלכים", "yMax": 100}
    base_cfg = {"labels": ["אקראי", "חמדן צעד אחד", A["label"]],
                "series": [{"label": "ניקוד ממוצע", "y": [round(rnd["score_mean"]), round(grd["score_mean"]), round(S["score_mean"])], "color": ["de", "de", color]}],
                "yTitle": "ניקוד ממוצע ב-1,000 משחקים", "legend": False}

    # --- ספים ---
    th_rows = thresholds_table(evals, THRESHOLDS)
    th_html = "".join(f"<tr><td>{f0(r[0])}</td>{th_cells(r)}</tr>" for r in th_rows)

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
                                            for (lbl, rows), c in zip(ab.items(), (color, "de"))],
                      "xTitle": "מעברים (מיליונים)", "yTitle": "ניקוד באימון (200 משחקים אחרונים)", "padRight": 130}
            ablation_html = card("c_ab", "ניסוי הסרה: החזר של צעד אחד מול 3 צעדים", "שתי ריצות קצרות (2.5 מיליון מעברים, epsilon יורד ל-0.02 אחרי מיליון) שזהות בכל דבר חוץ מאורך ההחזר. ניקוד האימון (כולל חקירה) אינו מוטה. ההסבר בסעיף הניתוח.", "chart", f"R.lineChart('c_ab', {j(ab_cfg)});")

    if kind == "ntuple":
        _, ev_decay = load_log("ablation_ntuple_decay")
        if ev_decay:
            t_max = ev_decay[-1]["elapsed_sec"] / 60 + 0.5
            ev_const = [e for e in evals if e["elapsed_sec"] / 60 <= t_max]
            ab_cfg = {"series": [
                          {"label": "קצב קבוע 0.1 (הריצה המלאה, 20 הדקות הראשונות)", "x": [e["elapsed_sec"] / 60 for e in ev_const], "y": [e["eval"]["score_mean"] for e in ev_const], "color": color, "width": 2.5},
                          {"label": "דעיכה: 0.1 → 0.01 בדקה 10 → 0.001 בדקה 15", "x": [e["elapsed_sec"] / 60 for e in ev_decay], "y": [e["eval"]["score_mean"] for e in ev_decay], "color": "de", "width": 2.5},
                      ],
                      "xTitle": "זמן אימון (דקות)", "yTitle": "ניקוד ממוצע בהערכה (100 משחקים)",
                      "table": {"columns": ["דקות", "קבוע", "דעיכה"],
                                "rows": [[round(e["elapsed_sec"] / 60, 1), round(e["eval"]["score_mean"]), ""] for e in ev_const] +
                                        [[round(e["elapsed_sec"] / 60, 1), "", round(e["eval"]["score_mean"])] for e in ev_decay]}}
            ab_cfg["x"] = sorted(set(ab_cfg["series"][0]["x"]) | set(ab_cfg["series"][1]["x"]))
            ablation_html = card("c_ab", "ניסוי הסרה: קצב למידה קבוע מול דעיכה, בתקציב של 20 דקות",
                                 "ריצה נפרדת של 20 דקות עם לוח הזמנים המקובל בספרות (alpha יורד פי 10 בחצי התקציב ופי 100 בשלושה רבעים), מול 20 הדקות הראשונות של הריצה המלאה עם alpha קבוע. ההסבר בסעיף הניתוח.",
                                 "chart", f"R.lineChart('c_ab', {j(ab_cfg)});")

    if kind == "asnet":
        ablation_html = asnet_plain_cards(evals, train, color)

    updates_note = ""
    if kind == "ppo" and updates:
        updates_note = f" ({f0(updates)} צעדי גרדיאנט, {args.get('ppo_epochs', 4)} מעברים על כל אצווה)"
    elif kind == "ntuple":
        updates_note = f" ({f0(train[-1].get('episodes', 0))} משחקים; כל מהלך הוא עדכון אחד של הטבלה)"
    elif kind == "asnet" and updates:
        updates_note = f" ({f0(updates)} צעדי גרדיאנט, אחד על כל צעד של {args.get('n_envs', 64)} הסביבות)"
    elif updates:
        updates_note = f" ({f0(updates)} צעדי גרדיאנט)"
    trans_num, trans_word = trans_fmt(total_trans)
    n_methods = len(KINDS)

    experiments_html = ntuple_experiments(evals, color) if kind == "ntuple" else ""
    toc = [("summary", "תקציר"), ("method", "האלגוריתם"), ("learning", "עקומת הלמידה"), ("final", "ההערכה הסופית"), ("replay", "צפייה במשחק")]
    if experiments_html:
        toc.append(("experiments", "ניסויים נוספים"))
        experiments_html = f'<section id="experiments"><h2>ניסויים נוספים: חיפוש מעל הטבלה, וריצה ארוכה</h2>{experiments_html}</section>'
    toc += [("analysis", "ניתוח"), ("hp", "היפר-פרמטרים")]
    body = f"""
<section id="summary">
<h2>תקציר</h2>
<p class="lead">{A['long']} אומן {total_min:.0f} דקות על {trans_num} {trans_word} מהלכים{updates_note}. בהערכה סופית של 1,000 משחקים (בלי חקירה) הוא הגיע לניקוד ממוצע של <b>{f0(S['score_mean'])}</b>,
הגיע לאריח 1024 ב-<b>{pct(S['reach_1024'])}</b> מהמשחקים ול-2048 ב-<b>{pct(S['reach_2048'])}</b>. לשם השוואה, מדיניות אקראית מגיעה ל-{f0(rnd['score_mean'])} וחמדן של צעד אחד ל-{f0(grd['score_mean'])}.</p>
<div class="kpis">{kpis}</div>
<p class="note">חומרה: {A['hw']}. הניקוד הוא ניקוד המשחק המקורי (סכום כל האריחים שנוצרו במיזוג). ההסבר על האלגוריתם בסעיף הבא; המסגרת המשותפת (MDP, החזר, V ו-Q, הרשת) וההשוואה בין {COUNT_WORDS_F[n_methods]} השיטות בסעיף <a href="comparison.html#methods">{COUNT_WORDS_F[n_methods]} השיטות</a> של דו"ח ההשוואה.</p>
</section>

<section id="method">
<h2>האלגוריתם: אינטואיציה, היסטוריה ומתמטיקה</h2>
<p class="note">הסימונים (\\(s, a, r, \\gamma, G_t, V, Q, A\\)) מוגדרים ב<a href="comparison.html#methods">מסגרת המשותפת</a> של דו"ח ההשוואה. הנוסחאות כאן הן בדיוק מה ש-<code>rl/{'actor_critic' if kind == 'ac' else 'ntuple_td' if kind == 'ntuple' else kind}.py</code> מחשב.</p>
{data_text(f"method_{kind}", "ההסבר על השיטה ייכתב בקובץ reports/data/method_" + kind + ".html.")}
</section>

<section id="learning">
<h2>עקומת הלמידה</h2>
{card("c_lc", "ניקוד לאורך האימון", "הקו העבה: הערכה חמדנית של 100 משחקים בכל נקודת ביקורת (המדד האמיתי). הקו הדק: ממוצע 200 המשחקים האחרונים באימון עצמו, כולל מהלכי החקירה. הקווים האפורים: קווי הבסיס.", "chart tall", f"R.lineChart('c_lc', {j(lc)});")}
{card("c_tiles", "האריח הגדול ביותר, לפי נקודת ביקורת", "אחוז המשחקים בהערכה שהסתיימו עם כל אריח מקסימלי. כחול כהה יותר = אריח גדול יותר.", "chart", f"R.barChart('c_tiles', {j({'labels': labels_ev, 'series': stacked_series, 'stacked': True, 'yTitle': 'אחוז מהמשחקים', 'yMax': 100, 'xTitle': 'מעברים'})});")}
{training_charts(kind, train, x_tr, color)}
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
<div class="card"><h3>איפה יושב האריח הגדול ביותר</h3><p class="desc">אחוז המהלכים (מרגע שיש אריח 32 ומעלה) שבהם האריח הגדול ביותר נמצא בכל משבצת. כחול כהה = לעיתים קרובות. שחקן טוב מחזיק אותו במקום קבוע.</p>
<div class="hm-row">{heatmap_html(ev.get('max_tile_pos', [[0]*4]*4), A['label'] + f" (בקצה {pct(ev.get('edge_rate', 0))}, בפינה {pct(ev.get('corner_rate', 0))})")}
{heatmap_html(baselines['greedy'].get('max_tile_pos', [[0]*4]*4), "חמדן צעד אחד")}
{heatmap_html(baselines['random'].get('max_tile_pos', [[0]*4]*4), "אקראי")}</div></div>
</div>
</section>

<section id="replay">
<h2>צפייה במשחק</h2>
<p>שני משחקים מוקלטים של הסוכן. אפשר לנגן, לגרור את הסרגל או להתקדם צעד-צעד.</p>
<div class="card"><h3>המשחק הטוב ביותר מתוך 40 מוקלטים ({f0(ev['best_game']['score'])} נקודות, אריח {f0(ev['best_game']['max_tile'])})</h3><div id="rp_best"></div></div>
<script>R.replay('rp_best', {j(ev['best_game'])}, {{start: 'end'}});</script>
<div class="card"><h3>משחק טיפוסי (חציוני, {f0(ev['typical_game']['score'])} נקודות, אריח {f0(ev['typical_game']['max_tile'])})</h3><div id="rp_typ"></div></div>
<script>R.replay('rp_typ', {j(ev['typical_game'])}, {{start: 'end'}});</script>
</section>
{experiments_html}
<section id="analysis">
<h2>ניתוח</h2>
{analysis_text(kind)}
</section>

<section id="hp">
<h2>היפר-פרמטרים</h2>
{hp_list(args, HP_LISTS[kind])}
<p class="note" style="margin-top:12px">{network_note(kind, args)}</p>
</section>
"""
    html = page(f"{A['label']} לומד לשחק 2048", f"""דו"ח ביצועים: {A['long']}, אימון מקומי על {'ליבת CPU אחת' if kind == 'ntuple' else 'GPU'}""",
                f"הדו\"ח נוצר ב-{dt.date.today().strftime('%d.%m.%Y')} · אימון של {total_min:.0f} דקות · {trans_num} {trans_word} מעברים", toc, body, crumbs_for(kind), math=True)
    path = os.path.join(OUT, A["report"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", os.path.relpath(path, ROOT))
    return {"summary": S, "train": train, "evals": evals, "eval": ev, "minutes": total_min, "transitions": total_trans, "sps": sps, "args": args, "updates": updates}


# ----------------------------------------------------------------------------
# דו"ח השוואה (כל הסוכנים שיש להם נתונים)
# ----------------------------------------------------------------------------

def samples_per_transition(kind: str, args: dict) -> float:
    """כמה פעמים כל מעבר נכנס לצעד גרדיאנט, בממוצע."""
    if kind == "dqn":
        return args.get("grad_steps", 2) * args.get("batch_size", 256) / args.get("n_envs", 64)
    if kind == "ppo":
        return float(args.get("ppo_epochs", 4))
    return 1.0  # A2C ורשת ה-n-tuple: כל מעבר משמש פעם אחת


def comparison_report(R: dict, baselines: dict):
    kinds = [k for k in KINDS if k in R]
    if len(kinds) < 2:
        print("skip comparison: need at least two agents")
        return
    rnd, grd = baselines["random"]["summary"], baselines["greedy"]["summary"]
    n = len(kinds)

    def L(k):
        return AGENTS[k]["label"]

    def C(k):
        return AGENTS[k]["color"]

    # --- KPI: עמודה לכל סוכן ---
    kpis = "".join(kpi(f"{L(k)}: ניקוד ממוצע", f0(R[k]["summary"]["score_mean"]), cls=C(k),
                       delta=f"הגיע ל-2048: {pct(R[k]['summary']['reach_2048'])}, ל-1024: {pct(R[k]['summary']['reach_1024'])}") for k in kinds)
    kpis += "".join(kpi(f"{L(k)}: מעברים", *trans_fmt(R[k]["transitions"]), cls=C(k),
                        delta=f"{R[k]['minutes']:.0f} דקות, {R[k]['sps']:,} בשנייה") for k in kinds)

    # --- עקומות ---
    def series(xkey, ykey, width=2.5, direct=True, label_suffix=""):
        out = []
        for k in kinds:
            ev = R[k]["evals"]
            out.append({"label": L(k) + label_suffix, "x": [xkey(e) for e in ev], "y": [ykey(e) for e in ev], "color": C(k), "width": width, "direct": direct})
        return out

    def all_x(ser):
        return sorted(set(x for s in ser for x in s["x"]))

    trans_x = lambda e: e["transitions"] / 1e6
    time_x = lambda e: e["elapsed_sec"] / 60
    score_y = lambda e: e["eval"]["score_mean"]

    lc_trans = {"series": series(trans_x, score_y), "xTitle": "מעברים (מיליונים, סולם לוגריתמי)", "yTitle": "ניקוד ממוצע בהערכה", "xLog": True, "padRight": 90,
                "table": {"columns": ["אלגוריתם", "מעברים (מיליונים)", "ניקוד"],
                          "rows": [[L(k), round(trans_x(e), 1), round(score_y(e))] for k in kinds for e in R[k]["evals"]]}}
    lc_trans["x"] = all_x(lc_trans["series"])
    lc_time = {"series": series(time_x, score_y), "refs": [{"label": f"חמדן ({f0(grd['score_mean'])})", "value": grd["score_mean"]}],
               "xTitle": "זמן אימון (דקות)", "yTitle": "ניקוד ממוצע בהערכה", "padRight": 90,
               "table": {"columns": ["אלגוריתם", "דקות", "ניקוד"], "rows": [[L(k), round(time_x(e), 1), round(score_y(e))] for k in kinds for e in R[k]["evals"]]}}
    lc_time["x"] = all_x(lc_time["series"])
    reach = {"series": series(time_x, lambda e: 100 * e["eval"]["reach_1024"], 2.5, False, ": הגיע ל-1024") +
                       series(time_x, lambda e: 100 * e["eval"]["reach_2048"], 1.2, False, ": הגיע ל-2048"),
             "xTitle": "זמן אימון (דקות)", "yTitle": "אחוז מהמשחקים", "yMax": 100}
    reach["x"] = lc_time["x"]

    # --- הערכה סופית ---
    max_score = max(max(R[k]["eval"]["scores"]) for k in kinds)
    width = hist_width(max_score)
    order = tile_order_for(*(R[k]["eval"]["max_tiles"] for k in kinds))
    lh = None
    h_series, td_series, act_series = [], [], []
    for k in kinds:
        lh, counts = hist(R[k]["eval"]["scores"], width, max_score)
        h_series.append({"label": L(k), "y": counts, "color": C(k)})
        td_series.append({"label": L(k), "y": tile_dist(R[k]["eval"]["max_tiles"], order), "color": C(k)})
        ac = R[k]["eval"].get("action_counts", [0, 0, 0, 0])
        tot = max(1, sum(ac))
        act_series.append({"label": L(k), "y": [round(100 * v / tot, 1) for v in ac], "color": C(k)})
    h_cfg = {"labels": lh, "series": h_series, "yTitle": "מספר משחקים", "xTitle": "ניקוד (אלפים)",
             "table": {"columns": ["טווח"] + [L(k) for k in kinds], "rows": [[l] + [s["y"][i] for s in h_series] for i, l in enumerate(lh)]}}
    td_cfg = {"labels": [str(t) for t in order], "series": td_series, "yTitle": "מספר משחקים", "xTitle": "האריח הגדול ביותר במשחק",
              "table": {"columns": ["אריח"] + [L(k) for k in kinds], "rows": [[str(t)] + [s["y"][i] for s in td_series] for i, t in enumerate(order)]}}
    base_cfg = {"labels": ["אקראי", "חמדן צעד אחד"] + [L(k) for k in kinds],
                "series": [{"label": "ניקוד ממוצע", "y": [round(rnd["score_mean"]), round(grd["score_mean"])] + [round(R[k]["summary"]["score_mean"]) for k in kinds],
                            "color": ["de", "de"] + [C(k) for k in kinds]}],
                "yTitle": "ניקוד ממוצע ב-1,000 משחקים", "legend": False}
    act_cmp = {"labels": DIRECTIONS, "series": act_series, "yTitle": "אחוז מהמהלכים", "yMax": 100}
    heatmaps = "".join(heatmap_html(R[k]["eval"].get("max_tile_pos", [[0] * 4] * 4), f"{L(k)} (בקצה {pct(R[k]['eval'].get('edge_rate', 0))})") for k in kinds)

    # --- טבלת סיכום ---
    def row(label, cells, hl=False):
        return f"<tr class='{'hl' if hl else ''}'><td>{label}</td>" + "".join(f"<td class='num'>{c}</td>" for c in cells) + "</tr>"

    S = {k: R[k]["summary"] for k in kinds}
    summary_rows = "".join([
        row("ניקוד ממוצע", [f0(S[k]["score_mean"]) for k in kinds], True),
        row("חציון", [f0(S[k]["score_median"]) for k in kinds]),
        row("סטיית תקן", [f0(S[k]["score_std"]) for k in kinds]),
        row("המשחק הטוב ביותר", [f0(S[k]["score_max"]) for k in kinds]),
        row("המשחק הגרוע ביותר", [f0(S[k]["score_min"]) for k in kinds]),
        row("אורך משחק ממוצע (מהלכים)", [f0(S[k]["moves_mean"]) for k in kinds]),
        row("הגיע ל-512", [pct(S[k]["reach_512"]) for k in kinds]),
        row("הגיע ל-1024", [pct(S[k]["reach_1024"]) for k in kinds], True),
        row("הגיע ל-2048", [pct(S[k]["reach_2048"]) for k in kinds], True),
        row("הגיע ל-4096", [pct(S[k]["reach_4096"]) for k in kinds]),
        row("הגיע ל-8192", [pct(np.mean(np.asarray(R[k]["eval"]["max_tiles"]) >= 8192)) for k in kinds]),
        row("זמן אימון (דקות)", [f"{R[k]['minutes']:.0f}" for k in kinds]),
        row("חומרה", [("ליבת CPU" if k == "ntuple" else "GPU") for k in kinds]),
        row("ייצוג V / המדיניות", [{"dqn": "רשת: Q(s,a)", "ac": "רשת: π ו-V(s)", "ppo": "רשת: π ו-V(s)", "ntuple": "טבלאות: V(afterstate)", "asnet": "רשת: V(afterstate)"}[k] for k in kinds]),
        row("מעברים באימון", [(f"{R[k]['transitions'] / 1e9:.2f}B" if R[k]["transitions"] >= 1e9 else f"{R[k]['transitions'] / 1e6:.1f}M") for k in kinds]),
        row("משחקים באימון", [f0(R[k]["train"][-1].get("episodes", 0)) for k in kinds]),
        row("מעברים בשנייה", [f"{R[k]['sps']:,}" for k in kinds]),
        row("צעדי עדכון", [(f0(R[k]["updates"]) if R[k].get("updates") else "") for k in kinds]),
        row("דגימות אימון לכל מעבר", [f"{samples_per_transition(k, R[k]['args']):.0f}" for k in kinds]),
    ])
    header = "".join(f"<th class='num'>{swatch(k)}{L(k)}</th>" for k in kinds)

    th_rows_by_kind = {k: thresholds_table(R[k]["evals"], THRESHOLDS) for k in kinds}
    th_rows = "".join(f"<tr><td>{f0(th)}</td>" + "".join(th_cells(th_rows_by_kind[k][i]) for k in kinds) + "</tr>" for i, th in enumerate(THRESHOLDS))
    th_header = "".join(f"<th class='num'>{swatch(k)}{L(k)} מעברים</th><th class='num'>דקות</th>" for k in kinds)

    diff_table = {
        "dqn": ["ערכי Q(s, a) לכל כיוון", "argmax על Q (עם epsilon לחקירה)", "epsilon-greedy, יורד ל-0.02",
                "off-policy: זיכרון חוויות של 500K, כל מעבר נדגם ~8 פעמים", "r + γ·Q_target(s', argmax Q)",
                "2 צעדי גרדיאנט (אצווה 256) על כל 64 מעברים", "רשת מטרה, Double DQN, Huber"],
        "ac": ["התפלגות על הכיוונים (שחקן) + ערך המצב (מבקר)", "דגימה מההתפלגות (הערכה: argmax)", "בונוס אנטרופיה 0.01",
               "on-policy: כל אצווה פעם אחת ונזרקת", "יתרון GAE(λ=0.95) למדיניות, החזר למבקר",
               "צעד גרדיאנט אחד על כל 1,024 מעברים", "נרמול יתרונות, חיתוך גרדיאנט 0.5"],
        "ppo": ["כמו A2C: התפלגות + ערך המצב", "כמו A2C", "בונוס אנטרופיה 0.01",
                "on-policy עם שימוש חוזר: כל אצווה 4 מעברים ואז נזרקת", "יתרון GAE, משוקלל ביחס ההסתברויות וקטום ב-±0.2",
                "16 צעדי גרדיאנט (4 מעברים × 4 מיני-אצוות של 256) על כל 1,024 מעברים", "קטימת היחס (חלון אמון), נרמול יתרונות, חיתוך גרדיאנט 0.5"],
        "ntuple": ["V(afterstate): ערך הלוח אחרי ההחלקה, סכום 4 טבלאות × 8 סימטריות", "argmax על r + V(afterstate), צעד אחד קדימה", "אין (חמדן; האריח האקראי מספיק)",
                   "on-policy מקוון: כל מעבר מיד, פעם אחת", "r + V(afterstate הבא), בלי היוון",
                   "עדכון של 32 משקלים אחרי כל מהלך (alpha/32 לכל אחד)", "לא נדרש; קירוב לינארי"],
        "asnet": ["V(afterstate) ברשת הקונבולוציה (ראש ערך אחד)", "כמו רשת ה-n-tuple: argmax על r + V(afterstate)", "אין (חמדן)",
                  "on-policy מקוון: כל מעבר פעם אחת, מיד", "r + V(afterstate הבא), בלי היוון; חצי-גרדיאנט",
                  "צעד גרדיאנט אחד (Adam) על כל 64 מעברים", "חיתוך גרדיאנט 0.5; בלי רשת מטרה ובלי זיכרון חוויות"],
    }
    aspects = ["מה נלמד", "איך נבחרת פעולה", "חקירה", "שימוש בנתונים", "יעד הלמידה", "עדכונים", "יציבות"]
    diff_rows = "".join(f"<tr><td>{a}</td>" + "".join(f"<td>{diff_table[k][i]}</td>" for k in kinds) + "</tr>" for i, a in enumerate(aspects))
    diff_header = "".join(f"<th>{swatch(k)}{L(k)}</th>" for k in kinds)

    # --- תקציר מילולי ---
    best = max(kinds, key=lambda k: S[k]["score_mean"])
    lead_parts = [f"{L(k)} <b>{f0(S[k]['score_mean'])}</b>" for k in kinds]
    reach_parts = [f"{L(k)} <b>{pct(S[k]['reach_2048'])}</b>" for k in kinds]
    budget_parts = [f"{L(k)} {R[k]['minutes']:.0f} דקות ({' '.join(trans_fmt(R[k]['transitions']))} מעברים)" for k in kinds]
    replays = "".join(
        f'<div class="card"><h3>{swatch(k)}{L(k)}: {f0(R[k]["eval"]["best_game"]["score"])} נקודות, אריח {f0(R[k]["eval"]["best_game"]["max_tile"])}</h3><div id="rp_{k}"></div></div>'
        for k in kinds)
    replay_js = " ".join(f"R.replay('rp_{k}', {j(R[k]['eval']['best_game'])}, {{start: 'end'}});" for k in kinds)
    reports_links = " · ".join(f'<a href="{AGENTS[k]["report"]}">{L(k)}</a>' for k in kinds)

    n_word = COUNT_WORDS[n]
    n_word_f = COUNT_WORDS_F[n]
    has_nt = "ntuple" in kinds
    n_neural = sum(1 for k in kinds if k in NEURAL)
    same_net = "אותה סביבה, אותו תגמול, אותם 1,000 משחקי מבחן" + (f" ({COUNT_WORDS_F[n_neural]} רשתות עצביות זהות, ורשת n-tuple בלי רשת עצבית)" if has_nt and n_neural >= 2 else ", אותה רשת")
    methods_html = data_text("methods_intro", "המסגרת המשותפת תיכתב בקובץ reports/data/methods_intro.html.") + \
        "".join(data_text(f"method_{k}", f"ההסבר על {L(k)} ייכתב בקובץ reports/data/method_{k}.html.") for k in kinds) + \
        data_text("methods_outro", "טבלת ההבדלים תיכתב בקובץ reports/data/methods_outro.html.")
    curves_intro = ("ההשוואה נעשית על שני צירים, כי לאלגוריתמים יש יחס שונה בין חישוב לנתונים. DQN מבצע 8 דגימות אימון ברשת על כל מעבר, A2C דגימה אחת, PPO ארבע"
                    + (", ורשת ה-n-tuple עדכון אחד של 32 מספרים בטבלה (בלי רשת, על ליבת CPU אחת)" if has_nt else "")
                    + (", ו-Afterstate TD ברשת דגימה אחת, אבל ארבע הרצות רשת לכל מעבר בבחירת המהלך" if "asnet" in kinds else "")
                    + ". לכן קצב צריכת המעברים שונה בסדרי גודל, וכל אלגוריתם מפיק מכל מעבר כמות אחרת. ציר הזמן עונה על \"מה מקבלים מדקת חישוב\", ציר המעברים על \"כמה ניסיון צריך\"."
                    + (" שימו לב שלרשת ה-n-tuple ציר הזמן מודד דקת CPU ולאחרים דקת GPU; זה חלק מהסיפור, לא פגם בהשוואה." if has_nt else ""))
    hw_note = HARDWARE + (f"; {L('ntuple')}: {HARDWARE_CPU}" if has_nt else "")

    toc = [("summary", "תקציר"), ("methods", f"{n_word_f} השיטות"), ("curves", "עקומות למידה"), ("final", "ההערכה הסופית"), ("table", "טבלת השוואה"), ("replay", "משחקים"), ("discussion", "דיון")]
    body = f"""
<section id="summary">
<h2>תקציר</h2>
<p class="lead">{n_word} אלגוריתמים, {same_net}. תקציב האימון: {"; ".join(budget_parts)}.
ניקוד ממוצע: {", ".join(lead_parts)}. הגעה ל-2048: {", ".join(reach_parts)}. המנצח: <b>{L(best)}</b>.</p>
<div class="kpis cols{n}">{kpis}</div>
<p class="note">הדו"חות הנפרדים: {reports_links}. חומרה: {hw_note}.</p>
</section>

<section id="methods">
<h2>{n_word_f} השיטות: מה כל אלגוריתם לומד, ואיך</h2>
{methods_html}
</section>

<section id="curves">
<h2>עקומות למידה</h2>
<p>{curves_intro}</p>
{card("c_time", "ניקוד מול זמן אימון", "מה מקבלים מכל דקה של חישוב. ההערכה החמדנית של 100 משחקים בכל נקודת ביקורת.", "chart tall", f"R.lineChart('c_time', {j(lc_time)});")}
{card("c_trans", "ניקוד מול מספר מעברים", "יעילות דגימה: כמה ניסיון צריך כל אלגוריתם כדי להגיע לאותה רמה. ציר ה-x לוגריתמי.", "chart tall", f"R.lineChart('c_trans', {j(lc_trans)});")}
{card("c_reach", "הגעה ל-1024 ול-2048 לאורך האימון", "קו עבה: 1024. קו דק: 2048.", "chart", f"R.lineChart('c_reach', {j(reach)});")}
<div class="card"><h3>מתי הושג כל סף</h3><p class="desc">הנקודה הראשונה שבה ההערכה החמדנית עברה את הסף.</p>
<div class="scroll"><table><thead><tr><th>ניקוד ממוצע</th>{th_header}</tr></thead><tbody>{th_rows}</tbody></table></div></div>
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
<div class="card"><h3>איפה יושב האריח הגדול ביותר</h3><p class="desc">אחוז המהלכים (מאריח 32 ומעלה) שבהם האריח הגדול ביותר נמצא בכל משבצת.</p>
<div class="hm-row">{heatmaps}</div></div>
</div>
</section>

<section id="table">
<h2>טבלת השוואה</h2>
<div class="scroll"><table><thead><tr><th>מדד</th>{header}</tr></thead><tbody>{summary_rows}</tbody></table></div>
<h3>ההבדלים האלגוריתמיים</h3>
<div class="scroll"><table><thead><tr><th>היבט</th>{diff_header}</tr></thead><tbody>{diff_rows}</tbody></table></div>
</section>

<section id="replay">
<h2>המשחקים הטובים ביותר</h2>
<p>המשחק הטוב ביותר של כל סוכן מתוך 40 משחקים מוקלטים. אפשר לנגן, לגרור את הסרגל או להתקדם צעד-צעד.</p>
<div class="grid{n}">{replays}</div>
<script>{replay_js}</script>
</section>

<section id="discussion">
<h2>דיון</h2>
{analysis_text("comparison")}
</section>
"""
    title = (f"{n_word} אלגוריתמי למידת חיזוק ב-2048" if n >= 4 else " מול ".join(L(k) for k in kinds) + " ב-2048")
    html = page(title, f"השוואת ביצועים של {n_word} אלגוריתמי למידת חיזוק על אותו משחק ואותו תקציב זמן ({', '.join(L(k) for k in kinds)}), עם הסבר מפורט על כל שיטה: אינטואיציה, היסטוריה ומתמטיקה",
                f"נוצר ב-{dt.date.today().strftime('%d.%m.%Y')}", toc, body, crumbs_for("comparison", R), math=True)
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
    for kind in KINDS:
        if kind in R:
            s = R[kind]["summary"]
            t = R[kind]["transitions"]
            t_str = f"{t / 1e9:.2f}B" if t >= 1e9 else f"{t / 1e6:.0f}M"
            hw = " (CPU)" if kind == "ntuple" else ""
            rows.append(f"| **{AGENTS[kind]['label']}** | **{f0(s['score_mean'])}** | {f0(s['score_median'])} | {f0(s['score_max'])} | {pct(s['reach_1024'])} | {pct(s['reach_2048'])} | {R[kind]['minutes']:.0f} דק', {t_str} מעברים{hw} |")
    table = "\n".join(rows)
    pattern = r"<!-- RESULTS_TABLE -->.*?(?=\n---)"
    if re.search(pattern, text, flags=re.S):
        new = re.sub(pattern, f"<!-- RESULTS_TABLE -->\n{table}\n", text, flags=re.S)
    else:
        new = text.replace("<!-- RESULTS_TABLE -->", f"<!-- RESULTS_TABLE -->\n{table}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
    print("updated README results table")


def main():
    baselines = {n: load_eval(n) for n in ("random", "greedy")}
    if any(v is None for v in baselines.values()):
        raise SystemExit("run rl/evaluate.py --all first (baselines missing)")
    R = {}
    for kind in KINDS:
        r = algo_report(kind, baselines)
        if r:
            R[kind] = r
    comparison_report(R, baselines)
    update_readme(R, baselines)


if __name__ == "__main__":
    main()
