"""מי יקים את הממשלה? תשעה תרחישי קואליציה אחרי הבחירות לכנסת ה-26, בסרטון מדובב.

כל המספרים (מנדטים, ספירת קולות, ציון יציבות, הבטחות שנשברות, איפה נוחתת המדיניות) מגיעים מ-data/coalition26.json,
שנוצר מהאפליקציה עצמה (coalition26/data.js ו-app.js) עם: node compute/coalition_data.mjs
הקריינות כתובה ביד, ולכן בטעינה נבדק שהמספרים שבה תואמים לנתונים.
רינדור: ./render.sh --hd scenes/coalition.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403

DATA = json.loads((ROOT / "data" / "coalition26.json").read_text())
P = DATA["parties"]
SC = DATA["scenarios"]

OPP = "#3a3f4b"
MUTED = GREY_B
GOOD, WARN, SERIOUS, CRIT = "#3fb27f", "#e3b23c", "#ec8a3c", "#e5534b"
CLS_COLOR = dict(good=GOOD, warn=WARN, serious=SERIOUS, crit=CRIT)
SEV = {3: ("קשה", CRIT), 2: ("בינונית", SERIOUS), 1: ("קלה", WARN)}
ROLE_NAME = dict(**{"in": "בקואליציה"}, support="תומכות מבחוץ", abstain="נמנעות", opp="אופוזיציה")
PM_NAME = dict(N="בנימין נתניהו", E="גדי איזנקוט")
NEUTRAL = ["L", "O", "RZ", "S", "U", "R", "J", "YB", "D", "B", "Y"]  # מימין לשמאל, כמו באפליקציה
RIGHT_EDGE = 6.8

# כאן אין נוסחאות כמו f(x), ולכן סוגר שאחרי מספר ("ב-2021)") נשאר מחוץ לבידוד, אחרת הוא מתהפך
_RUN = re.compile(r"[A-Za-z0-9](?:[^֐-׿()]*[A-Za-z0-9])?")


def heb(text, size=36, color=WHITE, **kwargs):  # noqa: F811
    text = _RUN.sub(lambda m: "⁦" + m.group(0) + "⁩", text)
    return Text(text, font=FONT, font_size=size, fill_color=color, **kwargs)


def bold(text, size=36, color=WHITE):
    return heb(text, size, color, weight="BOLD")


def wrap(text, n):
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > n:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur]


def para(text, size=24, color=GREY_A, n=58, buff=0.12):
    """פסקה עברית מיושרת לימין, שורה-שורה (Pango לא עוטף לבד)."""
    return VGroup(*(heb(l, size, color) for l in wrap(text, n))).arrange(DOWN, buff=buff, aligned_edge=RIGHT)


def name(pid, sc=None):
    """שם המפלגה לתצוגה. בתרחישי איזנקוט הליכוד באופוזיציה הוא עדיין הליכוד של נתניהו."""
    if pid == "Lx" and (sc is None or sc["id"] != "likud-after"):
        pid = "L"
    return P[pid]["name"]


def short(pid, sc=None):
    if pid == "Lx" and (sc is None or sc["id"] != "likud-after"):
        pid = "L"
    return P[pid]["short"]


# ---------- חצי העיגול של הכנסת (אותו פריסה כמו באפליקציה) ----------

def seat_layout():
    rows, r0 = 6, 0.42
    radii = [r0 + (1 - r0) * i / (rows - 1) for i in range(rows)]
    counts = [round(120 * r / sum(radii)) for r in radii]
    counts[-1] += 120 - sum(counts)
    seats = [(r, PI * j / (k - 1)) for r, k in zip(radii, counts) for j in range(k)]
    seats.sort(key=lambda s: (s[1], -s[0]))
    return seats


SEATS = seat_layout()
BOUNDARY_61 = (SEATS[59][1] + SEATS[60][1]) / 2


def style_seat(dot, color, role):
    if role == "in" or role == "neutral":
        dot.set_fill(color, 1).set_stroke(width=0)
    elif role == "support":
        dot.set_fill(color, 0.35).set_stroke(color, 2.5, 1)
    elif role == "abstain":
        dot.set_fill(color, 0).set_stroke(color, 2.5, 0.9)
    else:
        dot.set_fill(OPP, 1).set_stroke(width=0)
    return dot


class Hemicycle(VGroup):
    """120 נקודות, אחת לכל חבר כנסת. כל נקודה שייכת למפלגה, וכשהסדר משתנה הנקודות עוברות למושבים החדשים."""

    def __init__(self, center=ORIGIN, radius=3.0):
        self.c = np.array(center, dtype=float)
        self.R = radius
        self.dots = {}
        dots = []
        i = 0
        for pid in NEUTRAL:
            self.dots[pid] = []
            for _ in range(P[pid]["seats"]):
                d = style_seat(Dot(self.pos(i), radius=radius * 0.034), P[pid]["color"], "neutral")
                self.dots[pid].append(d)
                dots.append(d)
                i += 1
        super().__init__(*dots)

    def pos(self, i):
        r, th = SEATS[i]
        return self.c + self.R * r * np.array([np.cos(th), np.sin(th), 0])

    def arrange_anims(self, order):
        """order: רשימת (מפלגה, תפקיד) מימין לשמאל. מחזיר אנימציות שמעבירות כל נקודה למושב ולסגנון החדשים."""
        anims, i = [], 0
        for pid, role in order:
            key = "L" if pid == "Lx" else pid
            for d in self.dots[key]:
                target = style_seat(d.copy().move_to(self.pos(i)), P[key]["color"], role)
                anims.append(Transform(d, target, path_arc=-0.6))
                i += 1
        return anims

    def range_angles(self, a, b):
        """הזוויות של המושבים a עד b-1."""
        return SEATS[a][1], SEATS[b - 1][1]

    def marker_61(self):
        th = BOUNDARY_61
        u = np.array([np.cos(th), np.sin(th), 0])
        line = DashedLine(self.c + 0.3 * self.R * u, self.c + 1.08 * self.R * u, dash_length=0.08)
        line.set_stroke(WHITE, 2, 0.7)
        lab = heb("61", 22, WHITE).move_to(self.c + 1.16 * self.R * u)
        return VGroup(line, lab)


def bloc_arc(hc, a, b, color, label, size=26):
    t0, t1 = hc.range_angles(a, b)
    arc = Arc(start_angle=t0, angle=t1 - t0, radius=hc.R * 1.1, arc_center=hc.c).set_stroke(color, 6)
    mid = (t0 + t1) / 2
    u = np.array([np.cos(mid), np.sin(mid), 0])
    lab = bold(label, size, color).next_to(hc.c + hc.R * 1.1 * u, u, buff=0.2)
    return VGroup(arc, lab)


def legend_dot(role, color=GREY_A, r=0.11):
    return style_seat(Dot(radius=r), color, role)


def gauge(score, cls, radius=1.5):
    bg = Arc(start_angle=PI, angle=-PI, radius=radius).set_stroke(OPP, 22)
    v = ValueTracker(0)
    # הקשת והמספר נגזרים מקשת הרקע, כדי שיתאימו גם אחרי שהפאנל מוגדל
    fg = always_redraw(lambda: Arc(start_angle=PI, angle=-PI * max(v.get_value(), 0.5) / 100, radius=bg.get_width() / 2)
                       .set_stroke(CLS_COLOR[cls], 22 * bg.get_width() / (2 * radius))
                       .move_arc_center_to(bg.get_arc_center()))
    num = always_redraw(lambda: bold(str(round(v.get_value())), 60 * bg.get_width() / (2 * radius), WHITE)
                        .move_to(bg.get_arc_center() + 0.3 * bg.get_width() / 2 * UP))
    return VGroup(bg, fg, num), v


# ---------- קריינות ----------

N = {
    "change-minority": dict(
        intro='התרחיש הראשון: גוש השינוי, כלומר ישר!, ביחד, הדמוקרטים וישראל ביתנו, מכניס את רע"מ לקואליציה. '
              "הרשימה המשותפת לא נכנסת, אבל נמנעת בהצבעת האמון.",
        vote="56 בעד, 53 נגד, ו-11 נמנעים. הנמנעים לא נספרים, ולכן הממשלה קמה. "
             "כך בדיוק קמה ממשלת רבין ב-1993.",
        stab="ציון היציבות: 33 מתוך 100. שבירה. "
             "לטובתה: המתנגדים יגיעו לכל היותר ל-53, ובלי 61 הם לא יכולים להחליף אותה בלי בחירות. "
             "לרעתה: זו ממשלת מיעוט, וכל חוק, ובראשם התקציב, תלוי בכך שהרשימה המשותפת לא תצביע נגד.",
        pledges="שלוש הבטחות נשברות, כולן באותו נושא. איזנקוט וליברמן הבטיחו ממשלה ציונית שלא נשענת על המפלגות הערביות, "
                "ובנט הבטיח בלי מפלגות ערביות.",
        issues=[("draft", "בגיוס, כל השותפות באותו צד. המדיניות נוחתת על סוף הפטור ושירות לכולם."),
                ("arab", 'ובזכות רע"מ, תוכנית ממשלתית רחבה נגד הפשיעה ולתשתיות ביישובים הערביים.')],
        needs="כדי שזה יקרה, איזנקוט, בנט וליברמן צריכים לוותר על ההבטחה שלהם, "
              "והרשימה המשותפת צריכה להסתפק בהימנעות, בלי דרישות שאי אפשר לעמוד בהן.",
    ),
    "change-67": dict(
        intro="התרחיש השני: ממשלת ה-67. אותה קואליציה, אבל הפעם הרשימה המשותפת לא רק נמנעת. היא תומכת מבחוץ.",
        vote="67 בעד, 53 נגד. זה הרוב הגדול ביותר שאפשר להשיג בכנסת הזאת בלי הליכוד.",
        stab="ובכל זאת, ציון היציבות כמעט לא זז: 32. בתוך הקואליציה עדיין יש רק 56, "
             "ויש שש מפלגות שכל אחת מהן יכולה להפיל את הממשלה לבד. ומי שתומך מבחוץ, לא מחויב לשום דבר.",
        pledges="אותן שלוש הבטחות נשברות, והפעם באופן בולט יותר: הממשלה נשענת באופן פעיל על הרשימה המשותפת.",
        issues=[("diplomacy", "במדיניות החוץ: מהלך לנורמליזציה עם סעודיה, עם אופק מדיני לפלסטינים. "
                              "כאן עובר קו השבר, בין ליברמן לבין הרשימה המשותפת."),
                ("arab", "ובחברה הערבית: תוכנית לאומית גדולה. שינוי היסטורי ביחסי יהודים וערבים.")],
        needs="הפער האידאולוגי כאן הוא הגדול מכולם, מליברמן ובנט ועד הרשימה המשותפת. "
              "מה שמחזיק את הממשלה הזאת הוא בעיקר מה שהיא לא: לא ממשלת נתניהו.",
    ),
    "eisenkot-haredim": dict(
        intro='התרחיש השלישי: החרדים עוברים צד. ש"ס ויהדות התורה מצטרפות לאיזנקוט, וליברמן יוצא. '
              'בלי ליברמן, צריך גם את רע"מ.',
        vote="62 בעד, 58 נגד. רוב, אבל דחוק.",
        stab="ציון היציבות: 35. שבירה. הבעיה הגדולה כאן היא הפערים האידאולוגיים: "
             "הדמוקרטים וביחד מול החרדים, בגיוס, בדת ומדינה ובכסף הקואליציוני.",
        pledges='ארבע הבטחות נשברות. ש"ס ויהדות התורה הבטיחו לא להיכנס לממשלה בלי חוק שמעגן את הפטור, '
                "ואיזנקוט ובנט הבטיחו לא להישען על מפלגה ערבית.",
        issues=[("draft", "וזה לב העניין. במודל, חוק הגיוס נוחת על יעדים מחייבים וסנקציות, ולכן החרדים מפרים את ההבטחה שלהם. "
                          "אם הוא היה נוחת על פטור, איזנקוט היה מפר את שלו. כל תוצאה בחוק הגיוס שוברת הבטחה של מישהו."),
                ("religion", "ובדת ומדינה, המטוטלת זזה לכיוון ההפוך: סטטוס קוו עם נטייה דתית.")],
        needs="כדי שזה יקרה, צריך נוסחת גיוס ששני הצדדים יכולים להציג כניצחון. "
              "וזה אולי הדבר הקשה ביותר בפוליטיקה הישראלית.",
    ),
    "utj-outside": dict(
        intro="התרחיש הרביעי מבוסס על הצעה שדווחה מיהדות התורה: לתמוך באיזנקוט מבחוץ, בלי לשבת בממשלה, "
              "תמורת התקדמות בחוק הפטור.",
        vote='גוש השינוי ורע"מ, 56, ועוד שבעה של יהדות התורה מבחוץ. 63 בעד, 57 נגד.',
        stab="אבל זה התרחיש עם ציון היציבות הנמוך מכולם: 24. שבירה מאוד. תוחלת חיים משוערת: חודשים עד שנה.",
        pledges='חמש הבטחות נשברות, וליברמן הוא החוליה החלשה. הוא צריך לשבת עם רע"מ, וגם להישען על החרדים. '
                "שני דברים שהבטיח במפורש לא לעשות. גם יהדות התורה מפרה, כי חוק הגיוס נוחת קרוב לסוף הפטור.",
        issues=[("draft", "בגיוס, הרוב בממשלה דוחף לשירות לכולם. התמיכה מבחוץ לא קונה ליהדות התורה הרבה.")],
        needs="כדי שזה יקרה, צריך פשרה בגיוס שלא נראית כמו פטור גורף, וליברמן שמוכן לבלוע הרבה.",
    ),
    "unity-right": dict(
        intro="מכאן עוברים לממשלות בראשות נתניהו. התרחיש החמישי הוא המודל שמקורבי נתניהו מקדמים, לפי הדיווחים: "
              "רוטציה עם איזנקוט. הליכוד, ישר!, בן גביר וסמוטריץ'. בלי החרדים.",
        vote="61 בעד, 59 נגד. בדיוק 61. כל חבר כנסת בקואליציה הוא לשון מאזניים.",
        stab="ציון היציבות: 34. שבירה. ורוטציות בישראל נוטות להתפרק לפני מועד ההחלפה. "
             "ממשלת נתניהו וגנץ ב-2020 התפרקה תוך שבעה חודשים.",
        pledges="חמש הבטחות נשברות. הבולטת שבהן: איזנקוט אמר, לא אשב תחת נתניהו, גם לא ברוטציה. "
                "וגם נתניהו מפר. הוא אמר על השותפים שלו, אנחנו יחד ונשארים יחד, ומשאיר את החרדים בחוץ.",
        issues=[("inquiry", "ועדת החקירה הממלכתית, הדרישה המרכזית של איזנקוט, נוחתת במודל על ועדה בהרכב פוליטי."),
                ("judiciary", "ובמערכת המשפט: שינויים חלקיים, למשל בהרכב הוועדה לבחירת שופטים.")],
        needs="ומעבר להבטחות, קשה לדמיין את בן גביר ואיזנקוט מסכימים על מדיניות ביטחון.",
    ),
    "unity-center": dict(
        intro="התרחיש השישי: אחדות במרכז. הליכוד, ישר!, ביחד וישראל ביתנו. בלי בן גביר, בלי סמוטריץ' ובלי החרדים.",
        vote="67 בעד, 53 נגד. רוב נוח.",
        stab="ציון היציבות: 46. זה הציון הגבוה ביותר מבין הממשלות שאפשר להקים בלי להחליף את נתניהו, "
             "והוא עדיין בתחום השביר.",
        pledges="והמחיר: שש הבטחות נשברות. איזנקוט, בנט וליברמן, שלושתם הבטיחו לא לשבת עם נתניהו. "
                "ונתניהו נפרד מכל הגוש שלו.",
        issues=[("inquiry", "הנקודה הרגישה היא ועדת החקירה. במודל היא נוחתת באמצע, על ועדה בהרכב משולב. "
                            "אף צד לא מקבל את מה שהציג כעניין עקרוני."),
                ("draft", "ובגיוס: יעדים מחייבים וסנקציות על מי שלא משרת.")],
        needs="כדי שזה יקרה, שלושה מנהיגים צריכים לחזור בהם מההבטחה המרכזית שלהם, ונתניהו צריך לוותר על השותפים הוותיקים שלו.",
    ),
    "unity-haredim": dict(
        intro='התרחיש השביעי: אחדות עם החרדים. הליכוד, ישר!, ש"ס ויהדות התורה.',
        vote="שוב 61 בדיוק, מול 59.",
        stab="ציון היציבות: 30. שבירה מאוד. חוק הגיוס הוא פצצה מתקתקת בלב הממשלה.",
        pledges="שבע הבטחות נשברות. איזנקוט מפר שתיים בבת אחת: הוא יושב תחת נתניהו, "
                "ועם מפלגות שמתנגדות לשירות לכולם. והחרדים לא מקבלים את חוק הפטור שלהם.",
        issues=[("draft", "בגיוס, המדיניות נוחתת בדיוק באמצע. פשרה ששני הצדדים יכעסו עליה."),
                ("religion", "ובדת ומדינה: סטטוס קוו עם נטייה דתית.")],
        needs="יש לזה תקדים: ממשלות נתניהו בין 2015 ל-2019 נפלו בסוף בדיוק על חוק הגיוס.",
    ),
    "likud-after": dict(
        intro="התרחיש השמיני היפותטי לגמרי: מה אם הליכוד מחליף את נתניהו? הליכוד, ישר!, ביחד וישראל ביתנו, "
              "הפעם בראשות איזנקוט.",
        vote="67 בעד, 53 נגד.",
        stab="ציון היציבות: 73. יציבה. יכולה לשרוד קדנציה מלאה. זה התרחיש היחיד בסרטון שמגיע לציון כזה.",
        pledges="ואף הבטחה מרכזית לא נשברת. כל ההתחייבויות של איזנקוט, בנט וליברמן היו נגד נתניהו האיש, לא נגד הליכוד.",
        issues=[("draft", "בגיוס: סוף לפטור."),
                ("inquiry", "וגם ועדת חקירה ממלכתית לשבעה באוקטובר.")],
        needs="אבל אין שום סימן שזה קורה. נתניהו מוביל את הליכוד, ואמר שהוא לא רואה בתוצאות הפסד. "
              "התרחיש כאן כי הוא מראה עד כמה הקיפאון הפוליטי קשור לאדם אחד.",
    ),
    "almost": dict(
        intro="והתרחיש האחרון: כמעט. גוש נתניהו, 53, ועוד ליברמן.",
        vote="60 בעד, 60 נגד. תיקו לא מספיק. הממשלה לא עוברת.",
        stab=None,
        pledges="ובשביל התיקו הזה נשברות שבע הבטחות. ליברמן לבדו צריך להפר את כל שלוש ההבטחות שלו: לא לשבת עם נתניהו, "
                "לא לשבת עם החרדים, וגיוס לכולם.",
        issues=[],
        needs="הדרך היחידה לסגור את הפער: חבר כנסת אחד שעורק ממפלגה אחרת. זה כבר קרה, עם עידית סילמן ב-2022. "
              "ולכן, בכנסת הזאת, כל חבר כנסת שווה הרבה.",
    ),
}


def check_narration():
    """הקריינות כתובה ביד; אם data.js השתנה והמספרים זזו, עדיף להיכשל מאשר להקריא מספר ישן."""
    for sc in SC:
        n = N[sc["id"]]
        t = sc["tally"]
        for v in (t["forS"], t["against"]):
            assert str(v) in n["vote"], f'{sc["id"]}: vote narration lacks {v}'
        if n["stab"]:
            assert str(sc["stability"]["score"]) in n["stab"], f'{sc["id"]}: stability is {sc["stability"]["score"]}'
        nb = len(sc["broken"])
        words = {0: "אף הבטחה", 3: "שלוש", 4: "ארבע", 5: "חמש", 6: "שש", 7: "שבע"}
        assert words.get(nb, "?") in n["pledges"], f'{sc["id"]}: {nb} broken pledges'


check_narration()


# ---------- סצנות ----------

class CoalitionScene(HebrewScene):
    pass


class Title(CoalitionScene):
    def construct(self):
        title = bold("מי יקים את הממשלה?", 84)
        sub = heb("תשעה תרחישים אחרי הבחירות לכנסת ה-26", 40, GREY_A)
        date = heb("הנתונים נכונים ל-" + DATA["updated"], 26, MUTED)
        VGroup(title, sub, date).arrange(DOWN, buff=0.45)
        hc = Hemicycle(center=1.2 * DOWN, radius=1.6)
        VGroup(title, sub, date).next_to(hc, UP, buff=0.6)
        self.say("הבחירות לכנסת העשרים ושש הסתיימו, ואף גוש לא קיבל רוב. "
                 "בסרטון הזה נעבור על תשעה תרחישים להקמת ממשלה: איך הקואליציה תיראה, כמה היא יציבה, "
                 "ואילו הבטחות בחירות יצטרכו להישבר בדרך.")
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(sub, shift=0.2 * UP), LaggedStartMap(GrowFromCenter, hc, lag_ratio=0.01, run_time=3))
        self.play(FadeIn(date))
        self.fade_all()


class Knesset(CoalitionScene):
    def construct(self):
        hc = Hemicycle(center=0.9 * DOWN, radius=3.0)
        head = bold("הכנסת ה-26", 44).to_edge(UP, buff=0.35)
        rows = VGroup()
        for pid in NEUTRAL[::-1]:
            p = P[pid]
            rows.add(VGroup(heb(f'{p["name"]} {p["seats"]}', 24, GREY_A),
                            Dot(radius=0.1, fill_color=p["color"])).arrange(RIGHT, buff=0.15))
        rows = VGroup(*rows[::-1])
        legend = VGroup(VGroup(*rows[:6]).arrange(LEFT, buff=0.5), VGroup(*rows[6:]).arrange(LEFT, buff=0.5)) \
            .arrange(DOWN, buff=0.3).to_edge(DOWN, buff=0.5)

        self.say("זו הכנסת החדשה: מאה ועשרים מושבים, נקודה לכל חבר כנסת. "
                 "הגדולה היא הליכוד של נתניהו, עם 27. אחריה ישר! של גדי איזנקוט, מפלגה חדשה עם 21, "
                 "וביחד, הריצה המשותפת של בנט ולפיד, עם 12.")
        self.play(FadeIn(head))
        start = 0
        for pid in NEUTRAL:
            k = P[pid]["seats"]
            self.play(LaggedStartMap(GrowFromCenter, VGroup(*hc[start:start + k]), lag_ratio=0.05),
                      FadeIn(rows[NEUTRAL.index(pid)]), run_time=0.7)
            start += k
        self.say("ועוד שמונה מפלגות, כל אחת בין 6 ל-11 מנדטים. ארבע רשימות קרובות של שישה או שבעה מנדטים בלבד.")
        self.wait_speech()

        blocs = [(0, 53, "#7f9cf5", "גוש נתניהו 53"), (53, 72, GREEN_B, "המפלגות הערביות 19"),
                 (72, 120, TEAL_B, "גוש השינוי 48")]
        arcs = [bloc_arc(hc, a, b, c, lab) for a, b, c, lab in blocs]
        self.say("עכשיו נקבץ אותן לגושים. גוש נתניהו: הליכוד, עוצמה יהודית, הציונות הדתית, ש\"ס ויהדות התורה. 53.")
        self.play(FadeOut(legend), ShowCreation(arcs[0][0]), FadeIn(arcs[0][1]))
        self.say("גוש השינוי: ישר!, ביחד, הדמוקרטים וישראל ביתנו. 48.")
        self.play(ShowCreation(arcs[2][0]), FadeIn(arcs[2][1]))
        self.say('והמפלגות הערביות, הרשימה המשותפת ורע"מ: 19.')
        self.play(ShowCreation(arcs[1][0]), FadeIn(arcs[1][1]))
        m61 = hc.marker_61()
        self.say("כדי להקים ממשלה רגילה צריך 61. אף גוש לא מגיע לשם לבד, ולכן כל ממשלה תדרוש ממישהו לחצות קווים.")
        self.play(ShowCreation(m61[0]), FadeIn(m61[1]))
        self.play(Indicate(m61[1], scale_factor=1.6))
        self.fade_all()


class Rules(CoalitionScene):
    def construct(self):
        head = bold("איך קמה ממשלה?", 48).to_edge(UP, buff=0.4)
        self.say("איך קמה ממשלה? היא צריכה לעבור הצבעת אמון בכנסת. "
                 "והנה פרט שחשוב לכל הסרטון: לא צריך 61. צריך רק יותר בעד מאשר נגד. מי שנמנע, לא נספר.")
        self.play(FadeIn(head))
        rule = bold("בעד  >  נגד", 64, WHITE).shift(1.4 * UP)
        note = heb("הנמנעים לא נספרים", 32, MUTED).next_to(rule, DOWN, buff=0.3)
        self.play(Write(rule))
        self.play(FadeIn(note, shift=0.2 * UP))
        self.wait_speech()

        roles = VGroup()
        for role, txt in [("in", "בקואליציה: יושבת בממשלה ומצביעה בעד"),
                          ("support", "תומכת מבחוץ: מצביעה בעד, בלי לשבת בממשלה"),
                          ("abstain", "נמנעת: לא מפילה, אבל גם לא עוזרת"),
                          ("opp", "אופוזיציה: מצביעה נגד")]:
            roles.add(VGroup(heb(txt, 30, GREY_A), legend_dot(role, TEAL_B, 0.16)).arrange(RIGHT, buff=0.3))
        roles.arrange(DOWN, buff=0.35, aligned_edge=RIGHT).next_to(note, DOWN, buff=0.7)
        self.say("לכן מפלגה יכולה לעזור לממשלה בשלוש דרכים: לשבת בתוכה, לתמוך בה מבחוץ, או רק להימנע. "
                 "כך נסמן אותן לאורך הסרטון.")
        self.play(LaggedStartMap(FadeIn, roles, shift=0.2 * LEFT, lag_ratio=0.5, run_time=3))
        self.wait_speech()
        self.play(FadeOut(VGroup(rule, note, roles)))

        self.say("אבל לא כל צירוף אפשרי. חלק מהמפלגות פוסלות זו את זו: בן גביר וסמוטריץ' לא ישבו עם המפלגות הערביות, "
                 "והדמוקרטים לא ישבו עם נתניהו. כל התרחישים בסרטון לא שוברים אף פסילה הדדית.")
        pairs = VGroup()
        for a, b in [("O", "R"), ("RZ", "J"), ("L", "D"), ("O", "B")]:
            pairs.add(VGroup(heb(P[a]["name"], 30, P[a]["color"]), heb("✕", 34, CRIT), heb(P[b]["name"], 30, P[b]["color"]))
                      .arrange(LEFT, buff=0.35))
        pairs.arrange(DOWN, buff=0.4).shift(0.3 * DOWN)
        self.play(LaggedStartMap(FadeIn, pairs, lag_ratio=0.4, run_time=2.5))
        self.wait_speech()
        self.play(FadeOut(pairs))

        g, v = gauge(60, "warn", radius=1.4)
        g.shift(0.9 * UP + 3.2 * LEFT)
        facts = VGroup(*(heb(t, 26, GREY_A) for t in [
            "גודל הרוב", "המרחק האידאולוגי בין השותפות", "כמה מפלגות יכולות להפיל את הממשלה לבד",
            "כמה הבטחות נשברו", "האם לאופוזיציה יש 61 כדי להחליף את הממשלה"])).arrange(DOWN, buff=0.22, aligned_edge=RIGHT)
        facts.next_to(g, RIGHT, buff=1.0).align_to(g, UP)
        self.say("לכל תרחיש נחשב ציון יציבות, מאפס עד מאה. הוא מחבר את גודל הרוב, המרחק האידאולוגי בין השותפות, "
                 "כמה מפלגות יכולות להפיל את הממשלה לבד, כמה הבטחות נשברו, והאם לאופוזיציה יש בכלל 61 "
                 "כדי להחליף את הממשלה בלי בחירות. זה מודל פשוט, לא תחזית.")
        self.add(g)
        self.play(v.animate.set_value(60), LaggedStartMap(FadeIn, facts, lag_ratio=0.4), run_time=4)
        self.wait_speech()
        self.play(FadeOut(VGroup(g, facts)))

        ax = Line(4 * RIGHT, 4 * LEFT).set_stroke(GREY_C, 4).shift(1.2 * DOWN)
        poles = VGroup(heb("פטור מעוגן בחוק", 22, MUTED).next_to(ax.get_start(), DOWN, buff=0.25),
                       heb("שירות לכולם", 22, MUTED).next_to(ax.get_end(), DOWN, buff=0.25))
        lab = bold("גיוס חרדים", 30).next_to(ax, UP, buff=1.2)
        dots = VGroup(Dot(ax.point_from_proportion(0.1), radius=0.16, fill_color=P["U"]["color"]),
                      Dot(ax.point_from_proportion(0.85), radius=0.24, fill_color=P["Y"]["color"]),
                      Dot(ax.point_from_proportion(0.75), radius=0.18, fill_color=P["B"]["color"]))
        for d in dots:
            d.shift(0.3 * UP)
        dia = Square(0.3).rotate(PI / 4).set_fill(WHITE, 1).set_stroke(BG, 2).move_to(ax.get_center())
        self.say("ולכל נושא, נראה לאן המדיניות צפויה לנחות. כל מפלגה יושבת על ציר לפי העמדה שלה, "
                 "והמעוין הלבן מראה איפה נוחתת הממשלה. ככל שמפלגה רחוקה מהמעוין, היא מוותרת יותר.")
        self.play(FadeIn(lab), ShowCreation(ax), FadeIn(poles))
        self.play(LaggedStartMap(GrowFromCenter, dots, lag_ratio=0.3))
        self.play(FadeIn(dia, scale=0.3))
        self.play(dia.animate.move_to(ax.point_from_proportion(0.68)), run_time=2)
        self.fade_all()


class ScenarioScene(CoalitionScene):
    IDX = 0

    def construct(self):
        sc = SC[self.IDX]
        n = N[sc["id"]]
        t = sc["tally"]

        # --- כותרת וחצי עיגול במסך מלא ---
        kicker = heb(f"תרחיש {self.IDX + 1} מתוך {len(SC)}", 24, MUTED)
        title = bold(sc["title"], 52)
        tag = heb(sc["tag"] + "  ·  ראש הממשלה: " + PM_NAME[sc["mode"]], 28, GREY_A)
        header = VGroup(kicker, title, tag).arrange(DOWN, buff=0.15).to_edge(UP, buff=0.3)
        hc = Hemicycle(center=0.8 * DOWN + 2.3 * RIGHT, radius=2.8)
        m61 = hc.marker_61()

        self.say(n["intro"])
        self.play(FadeIn(header, shift=0.2 * DOWN), FadeIn(hc), FadeIn(m61))
        self.play(*hc.arrange_anims(sc["order"]), run_time=2.5)

        groups = {}
        for pid, role in sc["order"]:
            groups.setdefault(role, []).append(pid)
        legend = VGroup()
        for role in ["in", "support", "abstain", "opp"]:
            if role not in groups:
                continue
            ids = groups[role]
            total = sum(P["L" if i == "Lx" else i]["seats"] for i in ids)
            col = VGroup(VGroup(bold(f"{ROLE_NAME[role]}  {total}", 28), legend_dot(role, GREY_A, 0.13))
                         .arrange(RIGHT, buff=0.2))
            if role != "opp":
                for pid in ids:
                    key = "L" if pid == "Lx" else pid
                    col.add(VGroup(heb(f'{name(pid, sc)} {P[key]["seats"]}', 24, GREY_A),
                                   legend_dot(role, P[key]["color"], 0.1)).arrange(RIGHT, buff=0.18))
            else:
                col.add(heb("  ".join(short(i, sc) for i in ids), 20, MUTED))
            legend.add(col.arrange(DOWN, buff=0.12, aligned_edge=RIGHT))
        legend.arrange(DOWN, buff=0.35, aligned_edge=RIGHT).move_to(3.6 * LEFT + 0.5 * DOWN)
        if legend.get_height() > 5.2:
            legend.set_height(5.2)
        self.play(LaggedStartMap(FadeIn, legend, shift=0.2 * LEFT, lag_ratio=0.3, run_time=2))

        # --- ספירת הקולות ---
        W = 11.0
        bar_y = 3.35 * DOWN
        segs, x = VGroup(), RIGHT_EDGE - 0.3
        parts = [("בעד", t["forS"], GOOD), ("נמנעים", t["abs"], GREY_C), ("נגד", t["against"], CRIT)]
        for lab, v, c in parts:
            if not v:
                continue
            w = W * v / 120
            r = Rectangle(w, 0.42).set_fill(c, 0.9).set_stroke(BG, 2)
            r.move_to(np.array([x - w / 2, bar_y[1], 0]))
            txt = bold(f"{lab} {v}", 24, WHITE if c != GREY_C else BG).move_to(r)
            if txt.get_width() > w - 0.1:
                txt.set_fill(GREY_A).next_to(r, UP, buff=0.1)
            segs.add(VGroup(r, txt))
            x -= w
        verdict = bold("הממשלה קמה" if t["pass"] else "לא עוברת", 34, GOOD if t["pass"] else CRIT)
        verdict.next_to(segs, LEFT, buff=0.35)
        self.say(n["vote"])
        self.play(FadeOut(legend), LaggedStart(*(GrowFromEdge(s[0], RIGHT) for s in segs), lag_ratio=0.6, run_time=2),
                  LaggedStart(*(FadeIn(s[1]) for s in segs), lag_ratio=0.6, run_time=2.5))
        self.play(FadeIn(verdict, scale=1.4))
        self.wait_speech()

        # --- עמודת סיכום בצד ימין ---
        side_title = bold(sc["title"], 34).to_corner(UR, buff=0.35)
        if side_title.get_width() > 4.6:
            side_title.set_width(4.6)
        side_tag = heb(PM_NAME[sc["mode"]] + " · " + sc["tag"], 20, MUTED).next_to(side_title, DOWN, buff=0.12) \
            .align_to(side_title, RIGHT)
        mini = VGroup(hc, m61)
        score_lines = VGroup(heb(f'בעד {t["forS"]} · נגד {t["against"]}' + (f' · נמנעים {t["abs"]}' if t["abs"] else ""),
                                 24, GREY_A))
        self.play(FadeOut(VGroup(header, segs, verdict)),
                  mini.animate.scale(0.62).next_to(side_tag, DOWN, buff=0.35).align_to(side_title, RIGHT),
                  FadeIn(VGroup(side_title, side_tag)))
        score_lines.next_to(mini, DOWN, buff=0.25).align_to(side_title, RIGHT)
        self.play(FadeIn(score_lines))
        divider = Line(3.7 * UP, 3.7 * DOWN).set_stroke(GREY_D, 1).move_to(2.0 * RIGHT)
        self.play(ShowCreation(divider))
        L_RIGHT, L_LEFT = 1.6, -6.8  # גבולות האזור השמאלי

        def place(panel):
            """מגדיל או מקטין את הפאנל כך שימלא את האזור השמאלי, עד פי 1.45."""
            panel.scale(min((L_RIGHT - L_LEFT) / panel.get_width(), 6.6 / panel.get_height(), 1.45))
            return panel.move_to(np.array([(L_RIGHT + L_LEFT) / 2, 0, 0])).align_to(np.array([L_RIGHT, 0, 0]), RIGHT)

        def add_score(text, color):
            line = heb(text, 24, color).next_to(score_lines, DOWN, buff=0.18).align_to(side_title, RIGHT)
            score_lines.add(line)
            self.play(FadeIn(line, shift=0.15 * UP))

        current = None

        # --- יציבות ---
        st = sc["stability"]
        if n["stab"]:
            g, v = gauge(st["score"], st["cls"], radius=1.45)
            lab = bold(st["label"], 40, CLS_COLOR[st["cls"]])
            life = heb("תוחלת חיים משוערת: " + st["life"], 24, GREY_A)
            head = bold("כמה יציבה ממשלה כזו?", 34)
            factors = VGroup()
            for fname, val, _ in st["factors"]:
                val = round(val) if abs(val - round(val)) < 0.01 else round(val, 1)
                w = min(3.0, abs(val) * 0.15) or 0.04
                bar = Rectangle(w, 0.22).set_fill(GOOD if val >= 0 else CRIT, 0.9).set_stroke(width=0)
                num = heb(("+" if val > 0 else "") + str(val), 22, GREY_A)
                row = VGroup(heb(fname, 22, GREY_A), bar, num).arrange(LEFT, buff=0.2)
                factors.add(row)
            factors.arrange(DOWN, buff=0.14, aligned_edge=RIGHT)
            gbox = VGroup(g, VGroup(lab, life).arrange(DOWN, buff=0.12).next_to(g[0], DOWN, buff=0.1))
            panel = VGroup(head, VGroup(factors, gbox).arrange(LEFT, buff=0.7, aligned_edge=UP)).arrange(DOWN, buff=0.45, aligned_edge=RIGHT)
            place(panel)
            self.say(n["stab"])
            self.play(FadeIn(head), FadeIn(g[0]))
            self.add(g[1], g[2])
            self.play(v.animate.set_value(st["score"]), run_time=2)
            self.play(FadeIn(lab, scale=1.3), FadeIn(life))
            self.play(LaggedStartMap(FadeIn, factors, shift=0.15 * LEFT, lag_ratio=0.25, run_time=2.5))
            self.wait_speech()
            g[1].clear_updaters()
            g[2].clear_updaters()
            current = panel
            add_score(f'יציבות {st["score"]} · {st["label"]}', CLS_COLOR[st["cls"]])

        # --- הבטחות שנשברות ---
        head = bold(f'הבטחות שנשברות: {len(sc["broken"])}', 34)
        rows = VGroup()
        for b in sc["broken"]:
            word, c = SEV[b["sev"]]
            pill_t = bold("הפרה " + word, 20, BG)
            pill = VGroup(RoundedRectangle(pill_t.get_width() + 0.3, 0.38, corner_radius=0.19).set_fill(c, 1).set_stroke(width=0), pill_t)
            pc = P[b["party"]]["color"] if b["party"] in P else GREY_A
            text = b["text"] if '"' in b["text"] else f'"{b["text"]}"'
            row = VGroup(pill, bold(b["partyName"], 24, pc), heb(text, 24, GREY_A)).arrange(LEFT, buff=0.25)
            rows.add(row)
        if not len(rows):
            rows.add(heb("אף הבטחה מרכזית לא נשברת בצירוף הזה", 28, GOOD))
        rows.arrange(DOWN, buff=0.28, aligned_edge=RIGHT)
        panel = place(VGroup(head, rows).arrange(DOWN, buff=0.45, aligned_edge=RIGHT))
        self.say(n["pledges"])
        self.play(*([FadeOut(current, shift=0.3 * UP)] if current else []), FadeIn(head))
        self.play(LaggedStartMap(FadeIn, rows, shift=0.2 * LEFT, lag_ratio=0.35, run_time=max(1.5, 0.6 * len(rows))))
        self.wait_speech()
        current = panel
        add_score(f'הבטחות שנשברות: {len(sc["broken"])}', CRIT if sc["broken"] else GOOD)

        # --- איפה נוחתת המדיניות ---
        outs = {o["id"]: o for o in sc["outcomes"]}
        for iid, said in n["issues"]:
            o = outs[iid]
            AX_R, AX_L = L_RIGHT - 0.3, L_LEFT + 0.3
            xp = lambda val: AX_R - (val + 2) / 4 * (AX_R - AX_L)  # noqa: E731
            head = bold(o["name"], 36)
            ax = Line(np.array([AX_R, 0, 0]), np.array([AX_L, 0, 0])).set_stroke(GREY_C, 4)
            poles = VGroup(heb(o["minus"], 20, MUTED).next_to(ax.get_start(), DOWN, buff=0.25).align_to(ax.get_start(), RIGHT),
                           heb(o["plus"], 20, MUTED).next_to(ax.get_end(), DOWN, buff=0.25).align_to(ax.get_end(), LEFT))
            stacks, dots = {}, VGroup()
            for pp in sorted(o["parties"], key=lambda q: -P["L" if q["id"] == "Lx" else q["id"]]["seats"]):
                key = "L" if pp["id"] == "Lx" else pp["id"]
                r = 0.07 + np.sqrt(P[key]["seats"]) * 0.03
                k = stacks.get(pp["pos"], 0)
                stacks[pp["pos"]] = k + 1
                d = Dot(radius=r, fill_color=P[key]["color"], fill_opacity=1 if pp["role"] == "in" else 0.5)
                d.move_to(np.array([xp(pp["pos"]), 0.35 + 0.42 * k, 0]))
                lab = heb(short(pp["id"], sc), 20, GREY_A).next_to(d, RIGHT, buff=0.08)
                dots.add(VGroup(d, lab))
            dia = Square(0.3).rotate(PI / 4).set_fill(WHITE, 1).set_stroke(BG, 2).move_to(ax.get_center())
            txt = para(o["text"], 24, GREY_A, n=62).next_to(poles, DOWN, buff=0.45).align_to(ax.get_start(), RIGHT)
            head.next_to(dots, UP, buff=0.35).align_to(ax.get_start(), RIGHT)
            panel = VGroup(head, dots, ax, poles, dia, txt)
            place(panel)
            self.say(said)
            self.play(FadeOut(current, shift=0.3 * UP), FadeIn(head), ShowCreation(ax), FadeIn(poles))
            self.play(LaggedStartMap(GrowFromCenter, dots, lag_ratio=0.15, run_time=1.2))
            self.play(FadeIn(dia, scale=0.3))
            self.play(dia.animate.move_to(np.array([ax.point_from_proportion((o["value"] + 2) / 4)[0], dia.get_y(), 0])), run_time=2)
            self.play(FadeIn(txt, shift=0.15 * UP))
            self.wait_speech()
            current = panel

        # --- מה צריך לקרות ---
        head = bold("מה צריך לקרות כדי שזה יקרה", 34)
        bullets = VGroup()
        for need in sc["needs"]:
            lines = para(need, 26, GREY_A, n=52, buff=0.1)
            mark = Dot(radius=0.07, fill_color=TEAL_B).next_to(lines[0], RIGHT, buff=0.2)
            bullets.add(VGroup(lines, mark))
        bullets.arrange(DOWN, buff=0.35, aligned_edge=RIGHT)
        panel = place(VGroup(head, bullets).arrange(DOWN, buff=0.45, aligned_edge=RIGHT))
        self.say(n["needs"])
        self.play(FadeOut(current, shift=0.3 * UP), FadeIn(head))
        self.play(LaggedStartMap(FadeIn, bullets, shift=0.2 * LEFT, lag_ratio=0.5, run_time=2.5))
        self.fade_all()


SCENARIO_SCENES = []
for _i, _sc in enumerate(SC):
    _name = "S%d_%s" % (_i + 1, "".join(w.capitalize() for w in _sc["id"].split("-")))
    _cls = type(_name, (ScenarioScene,), {"IDX": _i, "__module__": __name__})
    globals()[_name] = _cls
    SCENARIO_SCENES.append(_cls)


class Summary(CoalitionScene):
    def construct(self):
        head = bold("כל התרחישים, זה לצד זה", 44).to_edge(UP, buff=0.35)
        cols = VGroup(heb("יציבות", 22, MUTED), heb("הבטחות שנשברות", 22, MUTED))
        rows = VGroup()
        for i, sc in enumerate(SC):
            st, t = sc["stability"], sc["tally"]
            title = heb(f'{i + 1}. {sc["title"]}', 26, WHITE)
            if t["pass"]:
                bar = Rectangle(st["score"] * 0.05, 0.36).set_fill(CLS_COLOR[st["cls"]], 0.95).set_stroke(width=0)
                val = heb(f'{st["score"]} {st["label"]}', 22, CLS_COLOR[st["cls"]])
            else:
                bar = Rectangle(0.05, 0.36).set_fill(GREY_D, 1).set_stroke(width=0)
                val = heb("לא עוברת", 22, CRIT)
            nb = heb(str(len(sc["broken"])), 26, CRIT if sc["broken"] else GOOD)
            rows.add(VGroup(title, bar, val, nb))
        # עמודות קבועות: שם מימין, עמודה, ערך, ומספר ההבטחות בשמאל
        x_title, x_bar, x_nb = 6.6, 1.2, -5.6
        for k, r in enumerate(rows):
            y = 2.4 - k * 0.62
            r[0].move_to(np.array([0, y, 0])).align_to(np.array([x_title, 0, 0]), RIGHT)
            r[1].move_to(np.array([0, y, 0])).align_to(np.array([x_bar, 0, 0]), RIGHT)
            r[2].next_to(r[1], LEFT, buff=0.2)
            r[3].move_to(np.array([x_nb, y, 0]))
        cols[0].move_to(np.array([0, 3.0, 0])).align_to(np.array([x_bar, 0, 0]), RIGHT)
        cols[1].move_to(np.array([x_nb, 3.0, 0]))
        head.to_edge(UP, buff=0.2)
        self.say("כשמסדרים את כל התשעה זה לצד זה, התמונה ברורה. כל ממשלה שאפשר להקים היום דורשת לשבור הבטחות מרכזיות, "
                 "וכולן שבירות.")
        self.play(FadeIn(head), FadeIn(cols))
        self.play(LaggedStart(*(AnimationGroup(FadeIn(r[0]), GrowFromEdge(r[1], RIGHT), FadeIn(r[2]), FadeIn(r[3]))
                                for r in rows), lag_ratio=0.25, run_time=5))
        self.wait_speech()
        best = rows[[s["id"] for s in SC].index("likud-after")]
        self.say("התרחיש היחיד שמקבל ציון יציב, ובלי אף הבטחה שבורה, הוא ההיפותטי: הליכוד בלי נתניהו.")
        self.play(*(r.animate.set_opacity(0.3) for r in rows if r is not best))
        self.play(Indicate(best, scale_factor=1.05, color=GOOD))
        self.wait_speech()
        self.play(*(r.animate.set_opacity(1) for r in rows))
        self.say("ועוד דבר אחד: כל ממשלה חדשה תצטרך להעביר את תקציב 2027 תוך חודשים ספורים. "
                 "אם היא לא תצליח, הכנסת מתפזרת, והולכים שוב לבחירות.")
        self.wait_speech()
        self.fade_all()


class Credits(CoalitionScene):
    def construct(self):
        lines = VGroup(
            bold("רוצים לבנות קואליציה בעצמכם?", 44),
            heb("האפליקציה \"מי יקים את הממשלה?\" מאפשרת להרכיב כל צירוף ולראות את הניתוח המלא", 28, GREY_A),
            Text("is.gd/miyakim", font=FONT, font_size=44, fill_color=TEAL_B),
            heb("ציון היציבות הוא מודל פשוט, לא תחזית. הנתונים נכונים ל-" + DATA["updated"] + ".", 24, MUTED),
            heb("מקורות: ועדת הבחירות, ויקיפדיה, Times of Israel, JTA, ynet, ערוץ 7 ועוד", 22, MUTED),
        ).arrange(DOWN, buff=0.45)
        self.say("רוצים לבנות קואליציה בעצמכם? באפליקציה, מי יקים את הממשלה, אפשר להרכיב כל צירוף ולראות את הניתוח המלא. "
                 "ציון היציבות הוא מודל פשוט, לא תחזית, והנתונים נכונים לשלושים באוקטובר.")
        self.play(LaggedStartMap(FadeIn, lines, shift=0.2 * UP, lag_ratio=0.4, run_time=3))
        self.wait(3)
        self.fade_all(run_time=1.5)


SCENES = [Title, Knesset, Rules, *SCENARIO_SCENES, Summary, Credits]
