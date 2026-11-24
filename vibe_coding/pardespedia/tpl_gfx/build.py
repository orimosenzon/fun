#!/usr/bin/env python3
"""
בונה את הגרסאות הגרפיות של התבניות.

  python3 tpl_gfx/build.py sandbox   -> tpl_gfx/out_sandbox/  (דפי משנה של הבוט, בלי קטגוריות)
  python3 tpl_gfx/build.py live      -> tpl_gfx/out_live/     (מרחב תבנית, עם הקטגוריות)

באתר אין ParserFunctions, ולכן "אתם כאן" והמדדים נשענים על שם פרמטר מחושב:
התבנית החיצונית קוראת לתבנית פנימית עם  {{PAGENAME}}=visible , והפנימית
כותבת בכל תא  visibility:{{{<שם הדף של התא>|hidden}}} . רק התא שהדף שלו
הוא הדף הנוכחי מקבל visible. אותו רעיון משמש לדרגת השימור ולנרטיב.

הקובץ הזה רק כותב קבצים. את העלאה עושה publish.py.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = "משתמש:אורי מוסנזון בוט/ארגז חול/"

# (מספר, שם הדף, שם קצר, שנים, אורך בשנים, רקע, מסגרת, אייקון)
# הצבעים והאייקונים הם של חוה אהרון, מתבניות "תקופה 1" עד "תקופה 6".
PERIODS = [
    (1, "ימי הבראשית והאחוזה", "בראשית ואחוזה", "1913–1928", 16, "#CAE9AE", "#3B5E1C", "🌳"),
    (2, "הקמת פרדס חנה וביסוס המושבות", "הקמה וביסוס", "1929–1938", 10, "#F8D7A5", "#D86B00", "🍊"),
    (3, "מלחמה, מחנות ומאבק לעצמאות", "מלחמה ועצמאות", "1939–1948", 10, "#B0D0F4", "#1A5296", "🛡️"),
    (4, "קליטת עלייה והקמת השכונות", "קליטת עלייה", "1949–1968", 20, "#E8CBB2", "#6E4423", "⛺"),
    (5, "המועצה המאוחדת והתפתחות", "המועצה המאוחדת", "1969–1994", 26, "#D2B6EE", "#5B2C86", "🏛️"),
    (6, "צמיחה, תרבות והמועצה הגדולה", "צמיחה ותרבות", "1995–היום", 35, "#B0EAE2", "#0E695E", "🎨"),
]
LEAD = 13  # 1900–1912: השנים שלפני התקופה הראשונה, על אותו סרגל


def decade_page(y):
    c = y // 100 + 1
    d = y % 100
    if d == 0:
        return f"העשור הראשון של המאה ה-{c}"
    return f"שנות ה-{d} של המאה ה-{c}"


# דפים כפולים/שמות חלופיים שגם עליהם צריך לסמן "אתם כאן"
ALT = {1940: ["שנות ה-40 של המאה העשרים"]}

HERE_COLOR = "#222"


def vis(names):
    """visibility:{{{a|{{{b|hidden}}}}}}"""
    expr = "hidden"
    for n in reversed(names):
        expr = "{{{" + n + "|" + expr + "}}}"
    return expr


def period_seg(p):
    n, full, short, years, w, bg, fr, icon = p
    v = vis([full])
    return (
        f'<div style="flex:{w} 1 0; min-width:0; position:relative; background:{bg}; '
        f'border-top:5px solid {fr}; border-left:1px solid #fff; box-sizing:border-box" '
        f'title="תקופה {n}: {full} ({years})">'
        f'<div style="overflow:hidden; text-align:center; padding:4px 1px 5px; line-height:1.25">'
        f'[[{full}|<span style="display:block; color:#1A1A1A">'
        f'<span style="display:block; font-size:140%; line-height:1.2">{icon}</span>'
        f'<span class="d-none d-md-block" style="font-weight:bold; font-size:88%">{short}</span>'
        f'<span class="d-none d-md-block" style="font-size:82%; color:#444; font-weight:normal">{years}</span>'
        f'</span>]]</div>'
        f'<span style="position:absolute; top:-5px; right:0; bottom:0; left:0; border:3px solid {HERE_COLOR}; '
        f'pointer-events:none; visibility:{v}"></span>'
        f'<span style="position:absolute; top:-20px; right:0; left:0; text-align:center; font-size:13px; '
        f'line-height:1; color:{HERE_COLOR}; pointer-events:none; visibility:{v}">▼</span>'
        f'</div>'
    )


def decade_cell(y, muted=False):
    page = decade_page(y)
    v = vis([page] + ALT.get(y, []))
    bg = "#f1ede4" if muted else "#fff"
    return (
        f'<div style="flex:1 1 0; min-width:0; position:relative; text-align:center; padding:6px 0 3px; '
        f'background:linear-gradient(#777,#777) right top / 1px 6px no-repeat {bg}; '
        f'line-height:1.3; white-space:nowrap" title="{page}">'
        f'[[{page}|{y}]]'
        f'<span style="position:absolute; top:1px; right:1px; bottom:0; left:0; border:2px solid {HERE_COLOR}; '
        f'border-radius:4px; pointer-events:none; visibility:{v}"></span>'
        f'<span style="position:absolute; bottom:-13px; right:0; left:0; text-align:center; font-size:11px; '
        f'line-height:1; color:{HERE_COLOR}; pointer-events:none; visibility:{v}">▲</span>'
        f'</div>'
    )


def milestone(year, label):
    pct = round((year - 1900) / 130 * 100, 2)
    return (
        f'<span style="position:absolute; top:0; right:{pct}%; white-space:nowrap; font-size:82%; color:#444; '
        f'border-right:2px solid #444; padding:0 5px 0 0; line-height:1.4">{year} · {label}</span>'
    )


def legend_row(p):
    n, full, short, years, w, bg, fr, icon = p
    return (
        f'<div style="display:flex; align-items:center; gap:6px; padding:3px 6px; margin-top:3px; '
        f'background:{bg}; border-right:5px solid {fr}; border-radius:3px">'
        f'<span>{icon}</span><span style="flex:1 1 auto">[[{full}|<span style="color:#1A1A1A">{full}</span>]]</span>'
        f'<span style="white-space:nowrap; color:#444; font-size:90%">{years}</span></div>'
    )


def strip():
    """הציר המלא: סרגל המאה ה-19, פס שש התקופות, סרגל העשורים 1900–היום."""
    cap = 'style="font-size:85%; color:#555; margin:0 0 2px"'
    out = []
    out.append('<div style="direction:rtl; text-align:right">')
    out.append(f'<div {cap}>המאה ה-19 · לפני ייסוד המושבות</div>')
    out.append('<div style="display:flex; border-top:2px solid #999; font-size:min(92%, 2.7vw); margin-bottom:16px">'
               + "".join(decade_cell(y, muted=True) for y in range(1800, 1900, 10)) + '</div>')
    out.append(f'<div {cap}>המאה ה-20 וה-21 · שש התקופות בחיי המושבה</div>')
    out.append('<div style="display:flex; align-items:stretch; margin-top:18px">'
               f'<div style="flex:{LEAD} 1 0; min-width:0"></div>'
               + "".join(period_seg(p) for p in PERIODS) + '</div>')
    out.append('<div style="display:flex; border-top:2px solid #555; font-size:min(92%, 2.7vw)">'
               + "".join(decade_cell(y) for y in range(1900, 2030, 10)) + '</div>')
    out.append('<div class="d-none d-lg-block" style="position:relative; height:1.5em; margin-top:14px">'
               + milestone(1913, "ייסוד כרכור") + milestone(1929, "ייסוד פרדס חנה")
               + milestone(1969, "איחוד המושבות") + '</div>')
    out.append('<div class="d-md-none" style="margin-top:16px">'
               + "".join(legend_row(p) for p in PERIODS) + '</div>')
    out.append('</div>')
    return "\n".join(out)


def mini_seg(p):
    n, full, short, years, w, bg, fr, icon = p
    v = vis([full])
    return (
        f'<div style="flex:{w} 1 0; min-width:0; position:relative; background:{bg}; border-top:4px solid {fr}; '
        f'border-left:1px solid #fff; box-sizing:border-box; text-align:center; font-size:14px; line-height:24px" '
        f'title="תקופה {n}: {full} ({years})">'
        f'[[{full}|<span style="display:block; overflow:hidden; height:24px">{icon}</span>]]'
        f'<span style="position:absolute; top:-6px; right:-2px; bottom:-2px; left:-1px; border:2px solid {HERE_COLOR}; '
        f'border-radius:4px; pointer-events:none; visibility:{v}"></span>'
        f'<span style="position:absolute; top:-18px; right:0; left:0; text-align:center; font-size:11px; '
        f'line-height:1; color:{HERE_COLOR}; pointer-events:none; visibility:{v}">▼</span>'
        f'</div>'
    )


def mini():
    """פס קטן של שש התקופות, לתיבת התקופה שבראש ערך."""
    return (
        '<div style="display:flex; direction:rtl; align-items:flex-end; gap:6px; font-size:80%; color:#444; '
        'padding-top:14px">'
        '<span style="line-height:28px">1913</span>'
        '<div style="flex:1 1 auto; min-width:0; display:flex; background:#fff; border-radius:3px">'
        + "".join(mini_seg(p) for p in PERIODS) +
        '</div><span style="line-height:28px">היום</span></div>'
    )


NAV = """<includeonly><div style="clear:both; margin-top:1.5em; border:1px solid #aaa; background:#f9f9f9; font-size:90%; direction:rtl; text-align:right; border-radius:6px; overflow:hidden">
<div style="background:#d8e8c8; color:#1b3610; text-align:center; padding:6px; font-size:105%; font-weight:bold; border-bottom:1px solid #aaa">[[תקופות בחיי פרדס חנה-כרכור|ציר הזמן ההיסטורי של פרדס חנה-כרכור]]</div>
<div style="padding:8px 10px 6px">{{@@ציר זמן פרדס חנה-כרכור/פס|{{{כאן|{{PAGENAME}}}}}=visible}}</div>
<div style="padding:5px 10px; border-top:1px solid #ddd">'''נושאי מפתח:''' [[העלייה הייקית|עליית היקים]] • [[תקופות בחיי פרדס חנה-כרכור|הטבלה המלאה]]</div>
<div style="padding:5px 10px; border-top:1px solid #ddd; font-size:90%; color:#444">'''מקורות:''' [[בית הראשונים]] – המרכז לתולדות פרדס חנה-כרכור • אריה פיאלקוב (עורך), '''ספר פרדס חנה''' • פרסומי [[מועצה מקומית פרדס חנה-כרכור]] • דוד גרוסמן, '''הכפר הערבי ובנותיו''' • [[סקר ארץ ישראל המערבית]] (PEF) • [[ארגון יוצאי מרכז אירופה]] • ויקיפדיה העברית ונתוני הלמ"ס</div>
</div></includeonly><noinclude>
== תיעוד התבנית ==
תבנית ניווט תחתונה המקשרת בין דפי ההיסטוריה של [[פרדס חנה-כרכור]] ומפרטת את המקורות המרכזיים. לשימוש יש להוסיף בתחתית הערך: <code><nowiki>{{ציר זמן פרדס חנה-כרכור}}</nowiki></code>

הציר עצמו (סרגל העשורים ופס שש התקופות) נמצא בתבנית המשנה <code>/פס</code>. הדף שבו התבנית מוצגת מסומן על הציר במסגרת ובחץ, בלי שצריך לכתוב דבר בערך. כדי לסמן תקופה או עשור בערך ששמו שונה משם הדף שעל הציר, כותבים את שם הדף שעל הציר בפרמטר <code>כאן</code>, למשל <code><nowiki>{{ציר זמן פרדס חנה-כרכור|כאן=הקמת פרדס חנה וביסוס המושבות}}</nowiki></code>.
@@CAT[[קטגוריה:תבניות ניווט]]@@
</noinclude>"""

STRIP_DOC = """<noinclude>
תבנית משנה של [[תבנית:ציר זמן פרדס חנה-כרכור]]: הציר הגרפי. אין צורך לקרוא לה ישירות.

'''איך מסומן "אתם כאן":''' התבנית הראשית קוראת לתבנית הזאת עם שם הדף הנוכחי כשם פרמטר (<code><nowiki>{{PAGENAME}}=visible</nowiki></code>), וכל תא בציר מציג את המסגרת שלו רק אם קיבל פרמטר בשם הדף שלו. כך אין צורך בהרחבה ParserFunctions. כדי לסמן דף אחר לצורך בדיקה: <code><nowiki>{{ציר זמן פרדס חנה-כרכור/פס|שנות ה-30 של המאה ה-20=visible}}</nowiki></code>

רוחב כל תקופה בפס יחסי למספר השנים שלה. הצבעים והאייקונים זהים לאלה של תבניות התקופה (תקופה 1 עד תקופה 6).
@@CAT[[קטגוריה:תבניות ניווט]]@@
</noinclude>"""

MINI_DOC = """<noinclude>
תבנית משנה: פס קטן של שש התקופות, שמוצג בתוך [[תבנית:תקופה היסטורית]]. התקופה שהערך שייך אליה מסומנת במסגרת ובחץ, והשאר מקשרות לדפי התקופות.
@@CAT[[קטגוריה:תבניות ניווט]]@@
</noinclude>"""

PERIOD_BASE = """<includeonly><div style="background-color: {{{רקע|#E8F0DA}}}; border: 1.5px solid {{{מסגרת|#4A5D23}}}; border-right: 8px solid {{{מסגרת|#4A5D23}}}; color: #1A1A1A; padding: 12px 16px; margin-bottom: 16px; border-radius: 6px; box-shadow: 0 2px 5px rgba(0,0,0,0.08);">
{{{אייקון|}}} '''[[תקופות בחיי פרדס חנה-כרכור|{{{כותרת|}}}]]''' – {{{תיאור|}}}
{{@@ציר זמן פרדס חנה-כרכור/מיני|{{{קטגוריה|}}}=visible}}
</div></includeonly>"""

EVENT = """<includeonly>{| class="infobox" style="width: 22em; font-size: 90%; text-align: right; border: 1px solid #aaa; background-color: #f9f9f9; float: left; margin: 0 1em 1em 0; border-collapse: collapse;"
|-
! colspan="2" style="text-align: center; font-size: 120%; background-color: #d8e8c8; color: #1b3610; padding: 6px; border-bottom: 1px solid #aaa;" | {{{שם|{{PAGENAME}}}}}
|-
! style="background-color: #eef5e6; padding: 5px; border-top: 1px solid #ddd; width: 38%; vertical-align: top;" | 📅 תאריך

| style="padding: 5px; border-top: 1px solid #ddd;" | {{{תאריך|לא צוין}}}
|-
! style="background-color: #eef5e6; padding: 5px; border-top: 1px solid #ddd; vertical-align: top;" | 📍 מיקום

| style="padding: 5px; border-top: 1px solid #ddd;" | {{{מיקום|לא צוין}}}
|-
! style="background-color: #eef5e6; padding: 5px; border-top: 1px solid #ddd; vertical-align: top;" | 🧭 רקע / הקשר

| style="padding: 5px; border-top: 1px solid #ddd;" | {{{רקע|לא צוין}}}
|-
! style="background-color: #eef5e6; padding: 5px; border-top: 1px solid #ddd; vertical-align: top;" | 👥 גורמים מעורבים

| style="padding: 5px; border-top: 1px solid #ddd;" | {{{גורמים מעורבים|לא צוין}}}
|-
! style="background-color: #eef5e6; padding: 5px; border-top: 1px solid #ddd; vertical-align: top;" | 🎯 תוצאה / השפעה

| style="padding: 5px; border-top: 1px solid #ddd;" | {{{תוצאה|לא צוין}}}
|}</includeonly>"""

# תיבת מידע בנויה מ-div, כדי שבטלפון לא תצוף ותמעך את הטקסט שלידה
BOX_OPEN = ('<div class="float-md-right ml-md-3" style="clear:left; box-sizing:border-box; '
            'width:max(290px, min(100%, calc((640px - 100%) * 999))); max-width:100%; margin-bottom:1em; '
            'border:1px solid {border}; border-radius:6px; overflow:hidden; background:{bg}; font-size:90%; '
            'direction:rtl; text-align:right">')
BOX_HEAD = ('<div style="background:{head}; color:#fff; padding:7px 10px; display:flex; align-items:center; gap:8px">'
            '<span style="font-size:150%; line-height:1">{icon}</span>'
            '<span style="flex:1 1 auto; font-weight:bold; font-size:110%">[[{link}|<span style="color:#fff">{title}</span>]]</span>'
            '<span style="background:#fff; color:{head}; border-radius:12px; padding:1px 9px; font-weight:bold; '
            'white-space:nowrap" title="{numtitle}">מס\' {{{{{{מספר}}}}}}</span></div>')
TH = 'style="width:40%; text-align:right; vertical-align:top; padding:5px 8px; background:{th}; border-top:1px solid {line}; font-weight:bold"'
TD = 'style="vertical-align:top; padding:5px 8px; border-top:1px solid {line}"'


def box(colors, head, rows, cats):
    th = TH.format(**colors)
    td = TD.format(**colors)
    out = ["<includeonly>" + BOX_OPEN.format(**colors), BOX_HEAD.format(**colors, **head),
           '{| style="width:100%; border-collapse:collapse; margin:0; background:transparent"']
    for label, value in rows:
        out += ["|-", f"! {th} | {label}", f"| {td} | {value}"]
    out += ["|}", "</div>" + cats + "</includeonly>"]
    return "\n".join(out)


SHIMUR_COLORS = dict(border="#b9a48a", bg="#fbf8f3", head="#7a4b2a", th="#f1e8dc", line="#e0d3c1")
SHIMUR = box(
    SHIMUR_COLORS,
    dict(icon="🏛️", link="נספח השימור לתוכנית המתאר", title="אתר לשימור",
         numtitle="מספר האתר ברשימת השימור"),
    [
        ("דרגת שימור", "{{@@אתר לשימור/דרגה|{{{דרגה}}}=inline-flex}} {{{דרגה}}}"),
        ("נרטיב", "{{@@אתר לשימור/נרטיב|{{{נרטיב}}}=inline}} {{{נרטיב}}}"),
        ("📍 כתובת", "{{{כתובת}}}"),
        ("📐 גוש וחלקה", "{{{גוש}}}"),
        ("🏗️ הקמה", "{{{הקמה}}}"),
        ("🗺️ מיקום", "{{{מיקום}}}"),
    ],
    "@@CAT[[קטגוריה:מבנים לשימור]]@@",
) + """<noinclude>
תבנית לערכים על אתרים מתוך [[נספח השימור לתוכנית המתאר]] של פרדס חנה-כרכור (2015).
התבנית מציגה את מספר האתר ברשימת השימור ואת דרגת השימור שלו, ומוסיפה את הערך ל[[:קטגוריה:מבנים לשימור]].

כל השדות חובה: הרחבה ParserFunctions אינה מותקנת באתר, ולכן שדה ריק יוצג כטקסט גולמי. כשאין מידע כותבים "—".

ליד דרגת השימור מוצג מד של שתי משבצות (משבצת אחת ל"שימור רגיל", שתיים ל"שימור עם הגבלות מחמירות"), וליד הנרטיב אייקון. שניהם מופיעים רק כשהערך כתוב בדיוק כמו ברשימת הנספח; בכל ניסוח אחר מוצג הטקסט בלבד.

<pre>
{{אתר לשימור
| מספר   = 3.3.18
| דרגה   = שימור רגיל
| נרטיב  = נרטיב המושבה
| כתובת  = רחוב הנחילות, כרכור
| גוש    = גוש 10071, חלקה 12
| הקמה   = שנות ה-40
| מיקום  = [https://www.google.com/maps/search/?api=1&query=32.48,34.98&hl=iw 32.48, 34.98]
}}
</pre>
</noinclude>"""


def gauge(param, lit, color, tip):
    bars = "".join(
        f'<span style="display:inline-block; width:16px; height:9px; border-radius:2px; '
        f'background:{color if i < lit else "#d9d2c7"}"></span>' for i in range(2))
    return (f'<span style="display:{{{{{{{param}|none}}}}}}; gap:2px; vertical-align:middle; margin-left:4px" '
            f'title="{tip}">{bars}</span>')


GRADE = ('<includeonly>'
         + gauge("שימור רגיל", 1, "#c98a2b", "דרגת שימור רגילה (1 מתוך 2)")
         + gauge("שימור עם הגבלות מחמירות", 2, "#9c2f1e", "דרגת השימור המחמירה (2 מתוך 2)")
         + '</includeonly><noinclude>תבנית משנה של [[תבנית:אתר לשימור]]: מד דרגת השימור. '
           'נקראת עם הדרגה כשם פרמטר, ומציגה משבצת אחת או שתיים.@@CAT[[קטגוריה:תבניות מידע]]@@</noinclude>')

NARRATIVES = [
    ("נרטיב המושבה", "🏡"), ("נרטיב המים", "💧"), ("נרטיב החקלאות", "🌾"),
    ("נרטיב המבנים הביטחוניים", "🛡️"), ("נרטיב המוסדות הייחודיים", "🏫"),
    ("נרטיב המנדט הבריטי", "👑"), ("נרטיב קליטת העלייה", "⛺"),
]
NARR = ('<includeonly>'
        + "".join(f'<span style="display:{{{{{{{n}|none}}}}}}">{i}</span>' for n, i in NARRATIVES)
        + '</includeonly><noinclude>תבנית משנה של [[תבנית:אתר לשימור]]: אייקון לכל אחד משבעת הנרטיבים '
          'של נספח השימור. נקראת עם שם הנרטיב כשם פרמטר.@@CAT[[קטגוריה:תבניות מידע]]@@</noinclude>')

TEVA_COLORS = dict(border="#9db88a", bg="#f7faf3", head="#3f6b2a", th="#e6f0dc", line="#d3e2c5")
TEVA = box(
    TEVA_COLORS,
    dict(icon="🌿", link="סקר תשתיות טבע עירוני", title="אתר בסקר הטבע העירוני",
         numtitle="מספר האתר בסקר"),
    [
        ("🏷️ שם בסקר", "{{{שם}}}"),
        ("📏 שטח", "{{{שטח}}}"),
        ("🌳 אופי השטח", "{{{אופי}}}"),
        ("🌼 מיני דגל", "{{{דגל}}}"),
        ("🗺️ ייעוד בתוכנית המתאר", "{{{ייעוד}}}"),
    ],
    "@@CAT[[קטגוריה:טבע עירוני]][[קטגוריה:אתרי סקר הטבע העירוני]]@@",
) + """<noinclude>
תבנית לערכים על אתרים מתוך [[סקר תשתיות טבע עירוני]] של פרדס חנה-כרכור (2014). היא מציגה את מספר האתר בסקר ואת עיקרי הכרטיס שלו, ומוסיפה את הערך ל[[:קטגוריה:טבע עירוני]] ול[[:קטגוריה:אתרי סקר הטבע העירוני]].

כל השדות חובה: ההרחבה ParserFunctions אינה מותקנת באתר, ולכן שדה ריק יוצג כטקסט גולמי. כשאין מידע כותבים "אין".

<pre>
{{אתר טבע עירוני
| מספר  = 31
| שם    = שמורת אלון
| שטח   = 22 דונם
| אופי  = יער פארק של אלון התבור
| דגל   = אלון התבור, צבעוני ההרים
| ייעוד = שמורת טבע, קרקע חקלאית ודרך
}}
</pre>
</noinclude>"""

STUB = """<div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px 12px; border:2px solid #c8a44d; background-color:#fbf6e9; padding:10px 12px; border-radius:5px; direction:rtl; text-align:right">
<div style="font-size:24px; line-height:1">✍️</div>
<div style="flex:1 1 240px"><span style="color:#7a5c00">'''ערך זה הוא קצרמר ומחכה להרחבה.'''</span><br>יש לכם מה להוסיף — זיכרון, תמונה, פרט שחסר? מוזמנים לערוך ולהשלים.</div>
<div class="plainlinks" style="flex:0 0 auto"><span style="display:inline-block; background:#c8a44d; border-radius:20px; padding:4px 14px; font-weight:bold; white-space:nowrap">[{{fullurl:{{FULLPAGENAME}}|action=edit}} <span style="color:#3d2e00">✏️ להרחבת הערך</span>]</span></div>
</div>
@@CAT<includeonly>[[קטגוריה:ערכים להרחבה]]</includeonly>@@

<noinclude>
תבנית לסימון ערכים קצרמריים. שימוש: <code><nowiki>{{הרחבה}}</nowiki></code>

הכפתור פותח את דף העריכה של הערך שבו התבנית מוצגת.
@@CAT
[[קטגוריה:תבניות]]@@
</noinclude>"""

WIKIPEDIA = """<div style="display:flex; align-items:center; gap:12px; border:1px solid #c8ccd1; background-color:#f8f9fa; padding:10px 12px; border-radius:4px; margin:15px 0; direction:rtl; text-align:right">
<div style="flex:0 0 auto; width:38px; height:38px; box-sizing:border-box; border-radius:50%; border:1px solid #a2a9b1; background:#fff; font-family:'Linux Libertine','Times New Roman',serif; font-size:26px; line-height:36px; text-align:center; color:#222" title="ויקיפדיה">W</div>
<div style="flex:1 1 auto">חלק מהתוכן בדף זה מקורו בוויקיפדיה העברית, מהערך [https://he.wikipedia.org/wiki/{{PAGENAMEE:{{{1|}}}}} {{{1|}}}], והוא זמין תחת רישיון [https://creativecommons.org CC-BY-SA 4.0].</div>
</div>
<noinclude>
תבנית זו נועדה להצגת קרדיט לערך המבוסס על ויקיפדיה העברית.

יש להוסיף בתוך התבנית את שם ערך היעד בוויקיפדיה בצורה הזו <nowiki>{{ויקיפדיה|שם הערך הרצוי}}</nowiki>
@@CAT[[קטגוריה:תבניות]]@@
</noinclude>"""

LETTERS = "אבגדהוזחטיכלמנסעפצקרשת"
ALPHA = ('<div style="display:flex; flex-wrap:wrap; align-items:center; justify-content:center; gap:4px; '
         'border:1px solid #8888aa; background:#DBD9F8; padding:8px; border-radius:6px; direction:rtl">'
         '<span style="font-weight:bold; margin-left:6px">ערכים על פי א"ב:</span>'
         + "".join(
             f'<span style="display:inline-block; background:#fff; border:1px solid #8888aa; border-radius:5px; '
             f'font-weight:bold; text-align:center">[[מיוחד:דפים המתחילים ב/{c}|'
             f'<span style="display:block; min-width:2em; padding:3px 0">{c}</span>]]</span>' for c in LETTERS)
         + '</div>')

# שם התבנית -> (תוכן, האם דורסים תבנית קיימת)
TEMPLATES = {
    "ציר זמן פרדס חנה-כרכור": NAV,
    "ציר זמן פרדס חנה-כרכור/פס": "<includeonly>" + strip() + "</includeonly>" + STRIP_DOC,
    "ציר זמן פרדס חנה-כרכור/מיני": "<includeonly>" + mini() + "</includeonly>" + MINI_DOC,
    "תקופה היסטורית": PERIOD_BASE,
    "אירוע היסטורי": EVENT,
    "אתר לשימור": SHIMUR,
    "אתר לשימור/דרגה": GRADE,
    "אתר לשימור/נרטיב": NARR,
    "אתר טבע עירוני": TEVA,
    "הרחבה": STUB,
    "ויקיפדיה": WIKIPEDIA,
    "ניווט אלפביתי": ALPHA,
}


def render(text, mode):
    import re
    if mode == "sandbox":
        text = text.replace("{{@@", "{{" + SANDBOX)
        text = re.sub(r"@@CAT.*?@@", "", text, flags=re.S)
    else:
        text = text.replace("{{@@", "{{")
        text = re.sub(r"@@CAT(.*?)@@", r"\1", text, flags=re.S)
    assert "@@" not in text, text[text.find("@@") - 80:text.find("@@") + 80]
    return text


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sandbox"
    out = os.path.join(HERE, "out_" + mode)
    os.makedirs(out, exist_ok=True)
    for name, text in TEMPLATES.items():
        fn = os.path.join(out, name.replace("/", "__") + ".wiki")
        with open(fn, "w") as f:
            f.write(render(text, mode))
        print(f"{len(text):6d}  {fn}")


if __name__ == "__main__":
    main()
