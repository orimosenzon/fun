"""הרחבות שדות ותורת גלואה: הסימטריות של המספרים, ולמה אין נוסחה למעלה חמישית.

בעקבות הפוסטים של גדי אלכסנדרוביץ' בבלוג "לא מדויק" (gadial.net):
תורת גלואה - מה הרעיון הבסיסי בה? (2009), למה (בגדול, מאוד בגדול) לא ניתן לפתור משוואות ממעלה חמישית (2009),
ומבוא לתורת גלואה (2018). השדות, המימד ומשפט המכפלה כבר הוצגו בסרטון על סרגל ומחוגה.
קריינות: Azure Hila. נוסה Phonikud (NIKUD למטה נשאר משם), אבל הקול שלו פחות ברור.
רינדור: ./render.sh --hd scenes/galois.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403
import common  # noqa: E402

common.use_azure()
# אורי שמע: "שָׁלוֹשׁ מאות" ו-"גְלוּאָה"
common.SPOKEN.update({"גלואה": "גָּלוּאָה", "שלוש מאות": "שְׁלוֹשׁ מאות", "שלוש הנקודות": "שְׁלוֹשׁ הנקודות"})
# נמצאו בקריאת audio/nikud_galois.log. מילים שתלויות בהקשר מנוקדות ביד בתוך הטקסט
common.NIKUD.update({
    "גלואה": "גָּלוּאָה", "אוטומורפיזם": "אוֹטוֹמוֹרְפִיזְם", "אומגה": "אוֹמֶגָה", "באומגה": "בְּאוֹמֶגָה",
    "למעלה": "לְמַעֲלָה", "ולמעלה": "וּלְמַעֲלָה",  # מעלה של משוואה, לא כיוון
    "השאלה": "הַשְּׁאֵלָה", "שש": "שֵׁשׁ", "ממימד": "מִמֵּימַד", "גדי": "גָּדִי", "פורייה": "פוּרְיֶה",
    "ממשי": "מַמָּשִׁי", "הממשי": "הַמַּמָּשִׁי", "ממשיים": "מַמָּשִׁיִּים", "הממשיים": "הַמַּמָּשִׁיִּים",
    "רציונליים": "רַצְיוֹנָלִיִּים", "הרציונליים": "הָרַצְיוֹנָלִיִּים", "לרציונליים": "לָרַצְיוֹנָלִיִּים",
    "והרציונליים": "וְהָרַצְיוֹנָלִיִּים", "רדיקלים": "רָדִיקָלִים", "ברדיקלים": "בְּרָדִיקָלִים",
    "הסימטריות": "הַסִּימֶטְרִיּוֹת", "סימטריות": "סִימֶטְרִיּוֹת", "סימטריה": "סִימֶטְרִיָּה",
    "פולינום": "פּוֹלִינוֹם", "לפולינום": "לְפּוֹלִינוֹם", "האיקוסאהדרון": "הָאִיקוֹסָהֶדְרוֹן",
})

Q_COL = TEAL        # הרציונליים
ROOT = [YELLOW, PINK, BLUE_B, GREEN_B, ORANGE]  # צבע לכל שורש
SYM = GOLD          # סימטריות
GRID = BLUE_E


def dot(p, color=YELLOW, r=0.07):
    return Dot(p, radius=r, fill_color=color)


def tex(s, size=40, color=WHITE):
    return Tex(s, font_size=size).set_color(color)


def roots_of_unity_plane(scale=1.6):
    plane = ComplexPlane((-2, 2, 1), (-2, 2, 1), background_line_style=dict(stroke_color=GRID, stroke_width=1))
    plane.set_height(4 * scale)
    return plane


class GaloisScene(HebrewScene):
    def caption(self, text, say=None, size=32, wait=0.3):
        super().caption(text, say=say, size=size, wait=wait)


# ---------- סצנות ----------

class Title(GaloisScene):
    def construct(self):
        rows = VGroup(
            VGroup(heb("מעלה 2", 30, GREY_A), tex(R"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}", 40)),
            VGroup(heb("מעלה 3", 30, GREY_A),
                   tex(R"x = \sqrt[3]{-\tfrac{q}{2} + \sqrt{\tfrac{q^2}{4} + \tfrac{p^3}{27}}}"
                       R" + \sqrt[3]{-\tfrac{q}{2} - \sqrt{\tfrac{q^2}{4} + \tfrac{p^3}{27}}}", 34)),
            VGroup(heb("מעלה 4", 30, GREY_A), heb("נוסחה באורך עמוד שלם", 32, GREY_B)),
            VGroup(heb("מעלה 5", 30, GREY_A), tex("?", 72, RED)),
        )
        for r in rows:
            r[0].move_to([5.2, 0, 0])
            r[1].move_to([-0.6, 0, 0])
        for i, r in enumerate(rows):
            r.shift((2.6 - 1.6 * i) * UP)
        self.say("למשוואה ריבועית יש נוסחה, שכל תלמיד בתיכון מכיר.")
        self.play(FadeIn(rows[0], shift=0.2 * UP))
        self.wait_speech()
        self.say("גם למשוואה ממעלה שלישית יש נוסחה, מהמאה השש עשרה, עם שורש ריבועי בתוך שורש שלישי. "
                 "ולמעלה רביעית יש נוסחה מסובכת עוד יותר, אבל יש.")
        self.play(FadeIn(rows[1], shift=0.2 * UP))
        self.play(FadeIn(rows[2], shift=0.2 * UP))
        self.wait_speech()
        self.say("ולמעלה חמישית? מתמטיקאים חיפשו נוסחה כזאת במשך כמעט שלוש מאות שנה, ולא מצאו. "
                 "והתשובה המלאה הגיעה מבחור צעיר בשם אֶוָרִיסְט גלואה, שהבין שהשאלה בכלל לא קשורה לנוסחאות, "
                 "אלא לסימטריות.")
        self.play(FadeIn(rows[3], shift=0.2 * UP))
        self.play(rows[3][1].animate.scale(1.4), rate_func=there_and_back, run_time=1.2)
        self.wait_speech()
        self.play(*map(FadeOut, rows))
        title = heb("תורת גלואה", 84).move_to(0.3 * UP)
        sub = heb("הסימטריות הנסתרות של המספרים", 36, GREY_A).next_to(title, DOWN, buff=0.35)
        self.say("זה סרטון על תורת גלואה: על הרחבות של שדות, על הסימטריות שמסתתרות בתוכן, "
                 "ועל הסיבה שאין נוסחה למשוואה ממעלה חמישית.")
        self.play(FadeIn(title, shift=0.3 * UP), FadeIn(sub))
        self.wait_speech()
        self.fade_all()


class Fields(GaloisScene):
    def construct(self):
        self.caption("שדה: אוסף מספרים שסגור לארבע פעולות החשבון",
                     say="נתחיל בתזכורת קצרה מהסרטון על סרגל ומחוגה. שדה הוא אוסף של מספרים שסגור לארבע פעולות החשבון: "
                         "מחברים, מחסרים, כופלים ומחלקים, ולא יוצאים ממנו.")
        line = NumberLine((-3, 3, 1), width=11, include_numbers=False).shift(1.6 * UP)
        fracs = [(-5, 2), (-1, 1), (-1, 3), (0, 1), (1, 2), (3, 4), (1, 1), (5, 3), (2, 1), (7, 3)]
        dots = VGroup(*(dot(line.n2p(a / b), Q_COL, 0.06) for a, b in fracs))
        qlabel = tex(R"\mathbb{Q}", 56, Q_COL).next_to(line, RIGHT, buff=0.3)
        self.play(ShowCreation(line), LaggedStartMap(FadeIn, dots, lag_ratio=0.1), FadeIn(qlabel))
        self.wait_speech()
        self.caption("הרציונליים הם שדה. השלמים לא: חצי יוצא מהם",
                     say="המספרים הרציונליים, השברים, הם שדה. המספרים השלמים, לעומת זאת, הם לא שדה: "
                         "אחת חלקי שתיים כבר איננו מספר שלם.")
        self.wait_speech()

        plane = NumberPlane((-3, 4, 1), (-2, 3, 1), background_line_style=dict(stroke_color=GRID, stroke_width=1))
        plane.set_height(3.8).move_to([-3.2, -1.0, 0])
        xl = tex("1", 32).next_to(plane.c2p(4, 0), RIGHT, buff=0.15)
        yl = tex(R"\sqrt{2}", 32).next_to(plane.c2p(0, 3), UP, buff=0.1)
        form = tex(R"\mathbb{Q}(\sqrt{2}) = \{ a + b\sqrt{2} \}", 46, Q_COL).move_to([3.3, -0.2, 0])
        dim = heb("מימד 2 מעל הרציונליים", 32, GREY_A).next_to(form, DOWN, buff=0.4)
        self.caption("מוסיפים √2 ומקבלים שדה גדול יותר",
                     say="ואפשר להרחיב שדה. מוסיפים לרציונליים את שורש שתיים, וסוגרים תחת ארבע הפעולות. "
                         "כל מספר בשדה החדש הוא מספר רציונלי, ועוד מספר רציונלי כפול שורש שתיים. "
                         "שני מספרים מתארים אותו, ולכן המימד של ההרחבה הוא שתיים.")
        self.play(FadeOut(dots), FadeOut(line), FadeOut(qlabel))
        self.play(ShowCreation(plane), FadeIn(xl), FadeIn(yl))
        pts = VGroup(*(dot(plane.c2p(a, b), Q_COL, 0.05) for a in range(-3, 5) for b in range(-2, 4)))
        self.play(Write(form), LaggedStartMap(FadeIn, pts, lag_ratio=0.02))
        self.play(FadeIn(dim))
        self.wait_speech()
        self.caption("השאלה של גלואה: אילו סימטריות יש להרחבה?",
                     say="עד כאן, זה מה שכבר ראינו. הרעיון החדש של גלואה היה לשאול שאלה אחרת לגמרי: "
                         "אילו סימטריות יש להרחבה כזאת?")
        self.wait(1)
        self.fade_all()


class Conjugate(GaloisScene):
    def construct(self):
        plane = ComplexPlane((-3, 3, 1), (-2.5, 2.5, 1), background_line_style=dict(stroke_color=GRID, stroke_width=1))
        plane.set_height(6.2).to_edge(LEFT, buff=0.6).shift(0.3 * UP)
        real = Line(plane.c2p(-3, 0), plane.c2p(3, 0)).set_stroke(Q_COL, 5)
        self.caption("סימטריה ראשונה: הצמוד המרוכב",
                     say="נתחיל מסימטריה שכולנו מכירים: הצמוד של מספר מרוכב. "
                         "לוקחים מספר מרוכב, ומשקפים אותו ביחס לציר הממשי.")
        self.play(ShowCreation(plane))
        z, w = complex(1.5, 0.5), complex(1, 1)
        spec = [(z, "z", YELLOW), (w, "w", PINK), (z * w, "zw", GREEN_B)]
        dots = VGroup(*(dot(plane.c2p(c.real, c.imag), col, 0.09) for c, _, col in spec))
        tags = VGroup(*(tex(t, 34, col).next_to(d, UR, buff=0.05) for d, (_, t, col) in zip(dots, spec)))
        self.play(FadeIn(dots[:2]), FadeIn(tags[:2]))
        self.wait_speech()

        rule = VGroup(tex(R"\overline{z + w} = \overline{z} + \overline{w}", 42),
                      tex(R"\overline{z \cdot w} = \overline{z} \cdot \overline{w}", 42)).arrange(DOWN, buff=0.4)
        rule.move_to([3.9, 1.3, 0])
        self.caption("השיקוף מכבד את החשבון",
                     say="והשיקוף הזה מיוחד: הוא מכבד את החשבון. נכפול זֶד בְּדַבֶּל-יוּ, ונקבל נקודה חדשה. "
                         "עכשיו נשקף את שלוש הנקודות.")
        self.play(TransformFromCopy(dots[:2], dots[2]), FadeIn(tags[2]))
        self.wait_speech()
        mirrored = VGroup(*(dot(plane.c2p(c.real, -c.imag), col, 0.09) for c, _, col in spec))
        mtags = VGroup(*(tex(R"\overline{" + t + "}", 34, col).next_to(d, DR, buff=0.05)
                         for d, (_, t, col) in zip(mirrored, spec)))
        self.play(*(TransformFromCopy(a, b, path_arc=-0.5) for a, b in zip(dots, mirrored)),
                  FadeIn(mtags), run_time=2)
        self.say("והנקודות המשוקפות עדיין מקיימות את אותו יחס בדיוק: המכפלה של הצמודים היא הצמוד של המכפלה. "
                 "לשקף ואז לכפול, זה כמו לכפול ואז לשקף.")
        self.play(Write(rule))
        self.wait_speech()
        self.caption("הממשיים לא זזים",
                     say="ועוד דבר: המספרים הממשיים, שעל הציר, נשארים במקומם. הסימטריה הזאת שומרת עליהם.")
        self.play(ShowCreation(real))
        self.play(real.animate.set_stroke(width=9), rate_func=there_and_back, run_time=1.2)
        self.wait_speech()
        name = VGroup(heb("סימטריה של שדה", 40, SYM), heb("פונקציה הפיכה ששומרת על חיבור ועל כפל", 28, GREY_A))
        name.arrange(DOWN, buff=0.25).move_to([3.9, -1.2, 0])
        self.caption("לסימטריה כזאת קוראים אוטומורפיזם",
                     say="לפונקציה הפיכה ששומרת על החיבור ועל הכפל קוראים אוטומורפיזם. "
                         "בשפה פשוטה, סימטריה של השדה.")
        self.play(FadeIn(name, shift=0.2 * UP))
        self.wait(1)
        self.fade_all()


class SqrtTwoFlip(GaloisScene):
    def construct(self):
        plane = NumberPlane((-3, 4, 1), (-3, 3, 1), background_line_style=dict(stroke_color=GRID, stroke_width=1))
        plane.set_height(5.2).move_to([-3.4, 0.3, 0])
        yl = tex(R"\sqrt{2}", 30).next_to(plane.c2p(0, 3), UP, buff=0.08)
        pts = VGroup(*(dot(plane.c2p(a, b), Q_COL if b == 0 else interpolate_color(BLUE, RED, (b + 2) / 4), 0.06)
                       for a in range(-3, 5) for b in range(-2, 3)))
        rule = tex(R"a + b\sqrt{2} \quad \longmapsto \quad a - b\sqrt{2}", 44, SYM).move_to([3.3, 2.5, 0])
        self.caption("גם לשדה של √2 יש שיקוף",
                     say="גם לִשְׂדֵה שורש שתיים יש סימטריה כזאת. במקום פלוס שורש שתיים, כותבים מינוס שורש שתיים. "
                         "במישור של המקדמים, זה שיקוף.")
        self.play(ShowCreation(plane), FadeIn(yl), LaggedStartMap(FadeIn, pts, lag_ratio=0.01))
        self.play(Write(rule))
        self.play(pts.animate.stretch(-1, 1, about_point=plane.c2p(0, 0)), run_time=2)
        self.wait_speech()

        check = VGroup(tex(R"(1+\sqrt{2})^2 = 3 + 2\sqrt{2}", 38),
                       tex(R"(1-\sqrt{2})^2 = 3 - 2\sqrt{2}", 38, SYM)).arrange(DOWN, buff=0.35)
        check.move_to([3.3, 0.9, 0])
        self.caption("והוא מכבד את הכפל",
                     say="והוא מכבד את החשבון. אחת ועוד שורש שתיים, בריבוע, הוא שלוש ועוד שתיים שורש שתיים. "
                         "ואחת פחות שורש שתיים, בריבוע, הוא בדיוק שלוש פחות שתיים שורש שתיים.")
        self.play(Write(check[0]))
        self.play(TransformFromCopy(check[0], check[1]))
        self.wait_speech()

        line = NumberLine((-1, 6, 1), width=6.4).move_to([3.3, -1.6, 0])
        line.add_numbers(font_size=24)
        pairs = [(1 + 2 ** 0.5, 1 - 2 ** 0.5, YELLOW), (3 + 2 * 2 ** 0.5, 3 - 2 * 2 ** 0.5, PINK)]
        self.caption("על ישר המספרים, הוא קופץ בפראות",
                     say="אבל על ישר המספרים, הסימטריה הזאת נראית מטורפת. "
                         "אחת ועוד שורש שתיים, שהוא בערך שתיים וארבע דצימליות, קופץ אל מתחת לאפס. "
                         "וחמש נקודה שמונה קופץ לאפס נקודה שבע עשרה. זו סימטריה שלא רואים בעין: היא קיימת רק בחשבון.")
        self.play(ShowCreation(line))
        for src, dst, col in pairs:
            a, b = dot(line.n2p(src), col, 0.08), dot(line.n2p(dst), col, 0.08)
            self.play(FadeIn(a))
            self.play(TransformFromCopy(a, b, path_arc=PI * 0.8), run_time=1.5)
        self.wait(1)
        self.fade_all()


class RootsToRoots(GaloisScene):
    def construct(self):
        chain = VGroup(
            tex(R"\varphi(1) = 1", 42),
            tex(R"\varphi(2) = \varphi(1+1) = 1 + 1 = 2", 42),
            tex(R"\varphi\left(\tfrac{1}{2}\right) = \tfrac{1}{2}", 42),
        ).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to([0, 1.6, 0])
        self.caption("סימטריה לא יכולה להזיז את הרציונליים",
                     say="למה דווקא שיקוף? בואו נראה מה סימטריה של שדה יכולה לעשות. "
                         "קודם כל, היא חייבת לשלוח את אחת לאחת, כי אחת היא היחיד שכפל בו לא משנה כלום. "
                         "ואז גם שתיים, שהוא אחת ועוד אחת, נשאר במקום. וכך כל מספר שלם, וכל שבר.")
        for c in chain:
            self.play(Write(c), run_time=1.2)
        self.wait_speech()
        self.play(FadeOut(chain))

        root = VGroup(tex(R"\theta^2 = 2", 44),
                      tex(R"\Longrightarrow \quad \varphi(\theta)^2 = \varphi(2) = 2", 44)).arrange(RIGHT, buff=0.4)
        root.move_to([0, 2.4, 0])
        self.caption("שורש של פולינום עובר לשורש של אותו פולינום",
                     say="ועכשיו הנקודה החשובה. שורש שתיים מקיים: בריבוע הוא שתיים. "
                         "נפעיל את הסימטריה על המשוואה הזאת. שתיים לא זז, ולכן גם התמונה של שורש שתיים, בריבוע, היא שתיים. "
                         "כלומר, הסימטריה שולחת שורש של פולינום אל שורש של אותו פולינום.")
        self.play(Write(root[0]))
        self.play(Write(root[1]))
        line = NumberLine((-2, 2, 1), width=8).shift(0.2 * DOWN)
        line.add_numbers(font_size=26)
        r1, r2 = (dot(line.n2p(s * 2 ** 0.5), c, 0.1) for s, c in ((1, ROOT[0]), (-1, ROOT[1])))
        t1 = tex(R"\sqrt{2}", 34, ROOT[0]).next_to(r1, UP, buff=0.2)
        t2 = tex(R"-\sqrt{2}", 34, ROOT[1]).next_to(r2, UP, buff=0.2)
        poly = tex(R"x^2 - 2 = 0", 40, GREY_A).next_to(line, DOWN, buff=0.6)
        self.play(ShowCreation(line), FadeIn(VGroup(r1, r2, t1, t2)), FadeIn(poly))
        self.wait_speech()
        self.caption("יש רק שתי אפשרויות: להשאיר, או להחליף",
                     say="לפולינום איקס בריבוע פחות שתיים יש רק שני שורשים. אז יש רק שתי סימטריות: "
                         "הזהות, שמשאירה הכל במקום, והשיקוף, שמחליף בין שני השורשים.")
        self.play(Swap(r1, r2), Swap(t1, t2), run_time=1.6)
        self.play(Swap(r1, r2), Swap(t1, t2), run_time=1.6)
        self.wait_speech()
        big = heb("סימטריה של הרחבה = ערבוב של השורשים", 40, SYM).shift(2.3 * DOWN)
        self.caption("זה הרעיון של גלואה",
                     say="וזה הרעיון המרכזי של גלואה: הסימטריות של שדה הן בעצם דרכים לערבב את השורשים של פולינום, "
                         "בלי שאף אחד ירגיש, כי כל המשוואות עם מקדמים רציונליים נשארות נכונות.")
        self.play(FadeIn(big, shift=0.2 * UP))
        self.wait(1)
        self.fade_all()


class TwoRoots(GaloisScene):
    def construct(self):
        field = tex(R"\mathbb{Q}(\sqrt{2}, \sqrt{3})", 50, Q_COL).to_edge(UP, buff=0.5)
        basis = tex(R"1,\ \sqrt{2},\ \sqrt{3},\ \sqrt{6}", 40, GREY_A).next_to(field, DOWN, buff=0.3)
        self.caption("שני שורשים: מימד 4",
                     say="עכשיו נוסיף לרציונליים שני שורשים: שורש שתיים ושורש שלוש. "
                         "כל מספר בשדה הזה בנוי מארבעה חלקים: אחת, שורש שתיים, שורש שלוש, ושורש שש. המימד הוא ארבע.")
        self.play(Write(field))
        self.play(FadeIn(basis))
        self.wait_speech()

        signs = [("+", "+"), ("-", "+"), ("+", "-"), ("-", "-")]
        fixed = [R"\mathbb{Q}(\sqrt{2},\sqrt{3})", R"\mathbb{Q}(\sqrt{3})", R"\mathbb{Q}(\sqrt{2})", R"\mathbb{Q}(\sqrt{6})"]
        cards = VGroup()
        for (s2, s3), fx in zip(signs, fixed):
            body = VGroup(
                tex(R"\sqrt{2} \mapsto " + ("" if s2 == "+" else "-") + R"\sqrt{2}", 32,
                    WHITE if s2 == "+" else SYM),
                tex(R"\sqrt{3} \mapsto " + ("" if s3 == "+" else "-") + R"\sqrt{3}", 32,
                    WHITE if s3 == "+" else SYM),
                tex(R"\sqrt{6} \mapsto " + ("-" if s2 != s3 else "") + R"\sqrt{6}", 32,
                    WHITE if s2 == s3 else SYM),
            ).arrange(DOWN, buff=0.22)
            box = SurroundingRectangle(body, buff=0.25).set_stroke(GREY_B, 2)
            cards.add(VGroup(box, body))
        cards.arrange(RIGHT, buff=0.35).shift(0.2 * DOWN)
        self.caption("ארבע סימטריות: כל שורש הולך לעצמו או למינוס עצמו",
                     say="כמה סימטריות יש? שורש שתיים יכול ללכת לעצמו, או למינוס עצמו. וגם שורש שָׁלוֹשׁ. "
                         "שתיים כפול שתיים, ארבע סימטריות. בדיוק כמו המימד.")
        self.play(LaggedStartMap(FadeIn, cards, shift=0.2 * UP, lag_ratio=0.3, run_time=2.5))
        self.wait_speech()
        self.caption("מינוס כפול מינוס: √6 נשאר במקום",
                     say="ושימו לב לסימטריה האחרונה, שהופכת את שני הסימנים. שורש שש הוא שורש שתיים כפול שורש שלוש, "
                         "ומינוס כפול מינוס הוא פלוס. אז שורש שש לא זז.")
        self.play(Indicate(cards[3][1][2], color=WHITE, scale_factor=1.3))
        self.wait_speech()
        labels = VGroup(*(VGroup(heb("שומרת על", 24, GREY_A), tex(fx, 30, Q_COL)).arrange(DOWN, buff=0.12)
                          .next_to(c, DOWN, buff=0.3) for c, fx in zip(cards, fixed)))
        self.caption("כל סימטריה שומרת על שדה ביניים אחר",
                     say="כל סימטריה משאירה במקום חלק מהשדה. זו שמשנה רק את שורש שתיים שומרת על השדה של שורש שָׁלוֹשׁ. "
                         "זו שמשנה רק את שורש שלוש שומרת על השדה של שורש שתיים. והאחרונה שומרת על השדה של שורש שש. "
                         "אלה בדיוק שלושת השדות שיושבים באמצע, בין הרציונליים לבין השדה כולו.")
        self.play(LaggedStartMap(FadeIn, labels, shift=0.2 * DOWN, lag_ratio=0.3, run_time=2.5))
        self.wait(1)
        self.fade_all()


class CubeRoot(GaloisScene):
    def construct(self):
        plane = ComplexPlane((-2, 2, 1), (-2, 2, 1), background_line_style=dict(stroke_color=GRID, stroke_width=1))
        plane.set_height(6.2).to_edge(LEFT, buff=1.0).shift(0.3 * UP)
        r = 2 ** (1 / 3)
        roots = [r * np.exp(2j * PI * k / 3) for k in range(3)]
        circle = Circle(radius=plane.c2p(r, 0)[0] - plane.c2p(0, 0)[0]).move_to(plane.c2p(0, 0))
        circle.set_stroke(GREY_C, 1.5)
        dots = VGroup(*(dot(plane.c2p(z.real, z.imag), ROOT[k], 0.11) for k, z in enumerate(roots)))
        names = [R"\sqrt[3]{2}", R"\omega\sqrt[3]{2}", R"\omega^2\sqrt[3]{2}"]
        dirs = [RIGHT, UL, DL]
        tags = VGroup(*(tex(n, 32, ROOT[k]).next_to(d, dirs[k], buff=0.12) for k, (n, d) in enumerate(zip(names, dots))))
        poly = tex(R"x^3 - 2 = 0", 44).move_to([3.6, 2.4, 0])
        self.caption("לפולינום x³ − 2 יש שלושה שורשים",
                     say="ועכשיו דוגמה שבה משהו משתבש. לפולינום איקס בשלישית פחות שתיים יש שלושה שורשים. "
                         "אחד ממשי, השורש השלישי של שתיים, ושניים מרוכבים, שיוצרים איתו משולש שווה צלעות.")
        self.play(ShowCreation(plane), Write(poly))
        self.play(ShowCreation(circle), LaggedStartMap(FadeIn, dots, lag_ratio=0.4), FadeIn(tags))
        self.wait_speech()
        real = Line(plane.c2p(-2, 0), plane.c2p(2, 0)).set_stroke(Q_COL, 6)
        field = tex(R"\mathbb{Q}(\sqrt[3]{2}) \subset \mathbb{R}", 44, Q_COL).move_to([3.6, 1.2, 0])
        self.caption("אבל השדה של ∛2 כולו ממשי",
                     say="אבל השדה שמתקבל מהוספת השורש השלישי של שתיים לרציונליים, כולו נמצא על הציר הממשי. "
                         "שני השורשים האחרים בכלל לא בתוכו.")
        self.play(ShowCreation(real), Write(field))
        self.play(dots[1:].animate.set_opacity(0.25), tags[1:].animate.set_opacity(0.25))
        self.wait_speech()
        verdict = VGroup(heb("מימד 3", 36), heb("סימטריה אחת בלבד: הזהות", 36, RED)).arrange(DOWN, buff=0.3)
        verdict.move_to([3.6, -0.6, 0])
        self.caption("אין לאן לשלוח את ∛2: חסרות סימטריות",
                     say="אז לאן הסימטריה יכולה לשלוח את השורש השלישי של שתיים? רק לשורש של אותו פולינום, "
                         "ובתוך השדה יש רק אחד כזה: הוא עצמו. המימד הוא שלוש, אבל יש רק סימטריה אחת, הזהות. "
                         "חסרות לנו סימטריות.")
        self.play(FadeIn(verdict, shift=0.2 * UP))
        self.wait(1)
        self.fade_all()


class SplittingField(GaloisScene):
    def construct(self):
        plane = ComplexPlane((-2, 2, 1), (-2, 2, 1), background_line_style=dict(stroke_color=GRID, stroke_width=1))
        plane.set_height(5.6).to_edge(LEFT, buff=0.8).shift(0.4 * UP)
        r = 2 ** (1 / 3)
        pts = [plane.c2p(*(lambda z: (z.real, z.imag))(r * np.exp(2j * PI * k / 3))) for k in range(3)]
        tri = Polygon(*pts).set_stroke(GREY_B, 2)
        dots = VGroup(*(dot(p, ROOT[k], 0.13) for k, p in enumerate(pts)))
        field = tex(R"\mathbb{Q}(\sqrt[3]{2}, \omega)", 48, Q_COL).move_to([3.6, 2.6, 0])
        dim = tex(R"[\,\mathbb{Q}(\sqrt[3]{2}, \omega) : \mathbb{Q}\,] = 3 \cdot 2 = 6", 34, GREY_A)
        dim.next_to(field, DOWN, buff=0.35)
        self.caption("הפתרון: להוסיף את כל השורשים",
                     say="הפתרון של גלואה: להוסיף לשדה את כל השורשים של הפולינום, ביחד. "
                         "מספיק להוסיף גם את אומגה, השורש השלישי של אחת. "
                         "השדה שמתקבל נקרא שדה הפיצול, והמימד שלו הוא שלוש כפול שתיים, כלומר שש.")
        self.play(ShowCreation(plane))
        self.play(ShowCreation(tri), LaggedStartMap(GrowFromCenter, dots, lag_ratio=0.3))
        self.play(Write(field))
        self.play(FadeIn(dim))
        self.wait_speech()

        center = plane.c2p(0, 0)
        group = VGroup(tri, dots)
        rot = heb("סיבוב: כפל באומגה", 30, SYM).move_to([3.6, 0.6, 0])
        self.caption("עכשיו יש סיבוב",
                     say="ועכשיו יש סימטריות חדשות. הראשונה מסובבת את המשולש: "
                         "כל שורש עובר לשורש הבא, כמו כפל באומגה.")
        self.play(FadeIn(rot))
        for _ in range(3):
            self.play(Rotate(group, TAU / 3, about_point=center), run_time=1.2)
        self.wait_speech()
        ref = heb("שיקוף: הצמוד המרוכב", 30, SYM).next_to(rot, DOWN, buff=0.3)
        self.caption("ויש שיקוף",
                     say="השנייה היא חברה ותיקה: הצמוד המרוכב, שמשקף את המשולש ומחליף בין שני השורשים המרוכבים.")
        self.play(FadeIn(ref))
        self.play(Rotate(group, PI, axis=RIGHT, about_point=center), run_time=1.5)
        self.play(Rotate(group, PI, axis=RIGHT, about_point=center), run_time=1.5)
        self.wait_speech()

        perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)]
        minis = VGroup()
        for p in perms:
            small = [np.array([np.cos(a), np.sin(a), 0]) * 0.42 for a in (0, TAU / 3, 2 * TAU / 3)]
            m = VGroup(Polygon(*small).set_stroke(GREY_B, 1.5),
                       *(dot(small[i], ROOT[p[i]], 0.08) for i in range(3)))
            minis.add(m)
        minis.arrange_in_grid(2, 3, buff=0.45).move_to([3.6, -1.4, 0])
        self.caption("שש סימטריות: כל הסידורים של שלושה שורשים",
                     say="כשמרכיבים סיבובים ושיקופים, מקבלים שש סימטריות. אלה כל שש הדרכים לסדר שלושה שורשים. "
                         "ושש, זה בדיוק המימד. כשזה קורה, ההרחבה נקראת הרחבת גלואה.")
        self.play(LaggedStartMap(FadeIn, minis, lag_ratio=0.2, run_time=2.5))
        self.wait(1)
        self.fade_all()


class Correspondence(GaloisScene):
    def construct(self):
        # שדות מימין, בגובה לפי המימד; החבורות משמאל באותו מבנה בדיוק
        fy = {"E": 2.5, "c": -0.1, "w": -1.1, "Q": -2.7}
        fields = {
            "E": tex(R"\mathbb{Q}(\sqrt[3]{2},\omega)", 32),
            "c0": tex(R"\mathbb{Q}(\sqrt[3]{2})", 30, ROOT[0]),
            "c1": tex(R"\mathbb{Q}(\omega\sqrt[3]{2})", 30, ROOT[1]),
            "c2": tex(R"\mathbb{Q}(\omega^2\sqrt[3]{2})", 30, ROOT[2]),
            "w": tex(R"\mathbb{Q}(\omega)", 30, SYM),
            "Q": tex(R"\mathbb{Q}", 36, Q_COL),
        }
        groups = {
            "E": heb("רק הזהות", 26), "c0": heb("שיקוף 1", 26, ROOT[0]), "c1": heb("שיקוף 2", 26, ROOT[1]),
            "c2": heb("שיקוף 3", 26, ROOT[2]), "w": heb("הסיבובים", 26, SYM), "Q": heb("כל השש", 28, Q_COL),
        }
        xs = {"E": 0, "c0": -1.9, "c1": -0.6, "c2": 0.7, "w": 2.0, "Q": 0}

        def place(mobs, cx):
            for k, m in mobs.items():
                m.move_to([cx + xs[k], fy.get(k, fy["c"] if k.startswith("c") else 0), 0])
            edges = VGroup(*(Line(mobs[a].get_bottom(), mobs[b].get_top(), buff=0.12).set_stroke(GREY_C, 2)
                             for a, b in [("E", "c0"), ("E", "c1"), ("E", "c2"), ("E", "w"),
                                          ("c0", "Q"), ("c1", "Q"), ("c2", "Q"), ("w", "Q")]))
            return edges

        fe = place(fields, 3.5)
        ge = place(groups, -3.5)
        fh = heb("שדות ביניים", 32, GREY_A).move_to([3.5, 3.4, 0])
        gh = heb("קבוצות של סימטריות", 32, GREY_A).move_to([-3.5, 3.4, 0])
        self.caption("כל השדות שבין הרציונליים לשדה הפיצול",
                     say="ועכשיו המשפט היסודי של תורת גלואה. מימין, כל השדות שיושבים בין הרציונליים לבין שדה הפיצול. "
                         "שלושה שדות ממימד שלוש, אחד לכל שורש, ושדה אחד ממימד שתיים, של אומגה.")
        self.play(FadeIn(fh), *(FadeIn(m) for m in fields.values()))
        self.play(ShowCreation(fe), run_time=1.5)
        self.wait_speech()
        self.caption("ומשמאל: קבוצות של סימטריות שסגורות להרכבה",
                     say="ומשמאל, הקבוצות של הסימטריות שסגורות להרכבה, מה שנקרא תת חבורות. "
                         "רק הזהות. כל אחד משלושת השיקופים, יחד עם הזהות. שלושת הסיבובים. וכל השש.")
        self.play(FadeIn(gh), *(FadeIn(m) for m in groups.values()))
        self.play(ShowCreation(ge), run_time=1.5)
        self.wait_speech()
        self.caption("אותו ציור בדיוק, הפוך",
                     say="ושני הציורים זהים. לכל שדה ביניים מתאימה קבוצת הסימטריות ששומרות עליו. "
                         "השיקוף שמחליף את שני השורשים המרוכבים שומר על השורש הממשי, ולכן על השדה שלו. "
                         "הסיבובים שומרים על אומגה. והרציונליים נשמרים על ידי כל השש.")
        for k in ["c0", "w", "Q", "E"]:
            link = CurvedArrow(groups[k].get_right() + 0.1 * RIGHT, fields[k].get_left() + 0.1 * LEFT, angle=-0.4)
            link.set_color(SYM)
            self.play(ShowCreation(link), Indicate(groups[k]), Indicate(fields[k]), run_time=1.3)
            self.play(FadeOut(link), run_time=0.4)
        self.wait_speech()
        self.caption("שדה גדול יותר ↔ פחות סימטריות",
                     say="וההתאמה הופכת את הסדר: ככל שהשדה גדול יותר, פחות סימטריות שומרות עליו. "
                         "את כל המבנה של ההרחבה אפשר לקרוא מהמבנה של קבוצת הסימטריות, שנקראת חבורת גלואה.")
        self.wait(1)
        self.fade_all()


class Radicals(GaloisScene):
    def construct(self):
        quad = tex(R"x = \frac{-b \pm \sqrt{D}}{2}", 48).move_to([0, 2.5, 0])
        self.caption("פתרון ברדיקלים: רק חשבון ושורשים",
                     say="איך כל זה קשור לנוסחאות? נוסחה לשורשים, כמו הנוסחה הריבועית, "
                         "משתמשת רק בארבע פעולות החשבון ובהוצאת שורשים. זה נקרא פתרון ברדיקלים.")
        self.play(Write(quad))
        self.wait_speech()
        pm = SurroundingRectangle(quad[R"\pm"], buff=0.08).set_stroke(SYM, 3)
        self.caption("הפלוס-מינוס הוא הסימטריה שמחליפה את שני השורשים",
                     say="ותסתכלו על הפלוס מינוס. הוא בדיוק הסימטריה שמחליפה בין שני השורשים. "
                         "הוצאת שורש ריבועי מוסיפה לַשָּׂדֶה סימטריה של שתיים.")
        self.play(ShowCreation(pm))
        self.wait_speech()
        self.play(FadeOut(pm), quad.animate.scale(0.7).to_corner(UR))

        tower = VGroup(tex(R"\mathbb{Q}", 40, Q_COL), tex(R"\mathbb{Q}(\sqrt{\Delta})", 36),
                       tex(R"\mathbb{Q}(\sqrt{\Delta}, \sqrt[3]{\cdots})", 36)).arrange(UP, buff=1.1)
        tower.move_to([3.0, -0.6, 0])
        steps_f = VGroup(*(tex(s, 34, SYM).next_to(Line(tower[i].get_top(), tower[i + 1].get_bottom()), RIGHT, buff=0.3)
                           for i, s in enumerate([R"\sqrt{\ }", R"\sqrt[3]{\ }"])))
        lines_f = VGroup(*(Line(tower[i].get_top(), tower[i + 1].get_bottom(), buff=0.12) for i in range(2)))
        chain = VGroup(heb("כל השש", 32, Q_COL), heb("שלושת הסיבובים", 30), heb("רק הזהות", 30)).arrange(UP, buff=1.1)
        chain.move_to([-3.0, -0.6, 0])
        lines_g = VGroup(*(Line(chain[i].get_top(), chain[i + 1].get_bottom(), buff=0.12) for i in range(2)))
        steps_g = VGroup(*(heb(s, 26, SYM).next_to(lines_g[i], LEFT, buff=0.3) for i, s in enumerate(["÷2", "÷3"])))
        self.caption("הנוסחה של קרדנו: שורש ריבועי, ואז שורש שלישי",
                     say="ובמעלה שלישית? הנוסחה של קַרְדָּנוֹ מוציאה קודם שורש ריבועי, ואחר כך שורש שלישי. "
                         "זה מגדל של שתי הרחבות.")
        self.play(FadeIn(tower[0]))
        for i in range(2):
            self.play(ShowCreation(lines_f[i]), FadeIn(steps_f[i]), FadeIn(tower[i + 1], shift=0.2 * UP))
        self.wait_speech()
        self.caption("ובצד של הסימטריות: מקלפים שכבה אחרי שכבה",
                     say="ובצד של הסימטריות, זה מתאים לקילוף של חבורת גלואה, שכבה אחרי שכבה. "
                         "משש סימטריות, לשלושת הסיבובים, ומשם לַזֶּהוּת. "
                         "בכל שלב, השכבה שמקלפים פשוטה: סיבוב מחזורי, שמתאים להוצאת שורש אחד.")
        self.play(FadeIn(chain[0]))
        for i in range(2):
            self.play(ShowCreation(lines_g[i]), FadeIn(steps_g[i]), FadeIn(chain[i + 1], shift=0.2 * UP))
        self.wait_speech()
        name = heb("חבורה פתירה: אפשר לקלף אותה כך עד הסוף", 36, SYM).to_edge(UP, buff=0.6)
        self.caption("משוואה פתירה ברדיקלים ⟺ חבורת גלואה פתירה",
                     say="לחבורה שאפשר לקלף ככה עד הסוף קוראים חבורה פתירה. וזה המשפט של גלואה: "
                         "למשוואה יש נוסחה ברדיקלים, אם ורק אם חבורת גלואה שלה פתירה.")
        self.play(FadeOut(quad), FadeIn(name, shift=0.2 * DOWN))
        self.wait(1)
        self.fade_all()


class Quintic(GaloisScene):
    def construct(self):
        roots = np.roots([1, 0, 0, 0, -1, -1])
        roots = sorted(roots, key=lambda z: np.angle(z))
        plane = ComplexPlane((-2, 2, 1), (-2, 2, 1), background_line_style=dict(stroke_color=GRID, stroke_width=1))
        plane.set_height(5.4).to_edge(LEFT, buff=0.9).shift(0.3 * UP)
        poly = tex(R"x^5 - x - 1 = 0", 46).move_to([3.6, 2.6, 0])
        dots = VGroup(*(dot(plane.c2p(z.real, z.imag), ROOT[k], 0.12) for k, z in enumerate(roots)))
        self.caption("משוואה ממעלה חמישית, עם חמישה שורשים",
                     say="והנה משוואה ממעלה חמישית: איקס בחמישית, פחות איקס, פחות אחת, שווה אפס. "
                         "יש לה חמישה שורשים: אחד ממשי, וארבעה מרוכבים.")
        self.play(ShowCreation(plane), Write(poly))
        self.play(LaggedStartMap(GrowFromCenter, dots, lag_ratio=0.25))
        self.wait_speech()
        count = tex(R"5! = 120", 46, SYM).move_to([3.6, 1.3, 0])
        self.caption("חבורת הגלואה שלה: כל 120 הסידורים של חמישה שורשים",
                     say="מתברר שהסימטריות של שדה הפיצול שלה מערבבות את חמשת השורשים בכל דרך אפשרית. "
                         "חמש עצרת, מאה ועשרים סידורים.")
        self.play(Write(count))
        rng = np.random.default_rng(5)
        for _ in range(4):
            i, j = rng.choice(5, 2, replace=False)
            a, b = dots[i].get_center(), dots[j].get_center()
            self.play(dots[i].animate.move_to(b).set_anim_args(path_arc=PI / 2),
                      dots[j].animate.move_to(a).set_anim_args(path_arc=PI / 2), run_time=0.9)
        self.wait_speech()

        blocks = VGroup(tex("120", 40, Q_COL), tex("60", 40), tex(R"\times", 56, RED)).arrange(DOWN, buff=0.9)
        blocks.move_to([3.6, -1.4, 0]).shift(0.4 * UP)
        lines = VGroup(*(Line(blocks[i].get_bottom(), blocks[i + 1].get_top(), buff=0.12) for i in range(2)))
        lines[1].set_stroke(RED)
        self.caption("את 60 אי אפשר לקלף",
                     say="ואת החבורה הזאת אי אפשר לקלף. השכבה הראשונה יורדת ממאה ועשרים לשישים, "
                         "אבל לחבורה של שישים אין שום שכבה פשוטה להוריד. היא קשורה לסימטריות של האיקוסאהדרון, "
                         "והיא חבורה פשוטה: אי אפשר לפרק אותה לחלקים קטנים יותר.")
        self.play(FadeIn(blocks[0]))
        self.play(ShowCreation(lines[0]), FadeIn(blocks[1], shift=0.2 * DOWN))
        self.play(ShowCreation(lines[1]), GrowFromCenter(blocks[2]))
        self.wait_speech()
        verdict = heb("אין נוסחה ברדיקלים", 44, RED).move_to([0, -3.0, 0])
        self.clear_caption()
        self.say("ולכן, את השורשים של המשוואה הזאת אי אפשר לכתוב בעזרת חשבון ושורשים. "
                 "לא שעוד לא מצאנו נוסחה: אין נוסחה, ולא תהיה. השורשים קיימים, אפשר לחשב אותם בכל דיוק, "
                 "אבל אין להם נוסחה.")
        self.play(FadeIn(verdict, shift=0.2 * UP))
        self.wait(1)
        self.fade_all()


class History(GaloisScene):
    def construct(self):
        events = [
            ("1824", "נילס אָבֶּל מוכיח שאין נוסחה כללית למעלה חמישית"),
            ("1829", "גלואה, בן שבע עשרה, שולח את עבודתו לאקדמיה"),
            ("1830", "פוּרְיֶה לוקח את כתב היד הביתה, ומת. הכתב אובד"),
            ("1831", "פּוּאָסוֹן דוחה את העבודה: לא מובנת"),
            ("1832", "דו קרב. גלואה נהרג בגיל עשרים"),
            ("1846", "לִיוּבִיל מפרסם את העבודה, וכולם מבינים"),
        ]
        rows = VGroup(*(VGroup(tex(y, 34, SYM), heb(NIQQUD.sub("", t), 30)).arrange(LEFT, buff=0.5) for y, t in events))
        rows.arrange(DOWN, buff=0.38, aligned_edge=RIGHT).move_to([0.6, 0.4, 0])
        for row, sentence in zip(rows, [
            "באלף שמונה מאות עשרים וארבע, נִילְס אָבֶּל, צעיר נורווגי, הוכיח שאין נוסחה כללית למעלה חמישית.",
            "גלואה, צעיר צרפתי, הלך רחוק יותר: הוא הבין בדיוק אילו משוואות פתירות ולמה. "
            "בגיל שבע עשרה הוא שלח את העבודה שלו לאקדמיה למדעים בפריז.",
            "פוּרְיֶה, אותו פורייה מהסרטונים הקודמים, לקח את כתב היד הביתה כדי לקרוא אותו, ומת. כתב היד אבד.",
            "שנה אחר כך, פּוּאָסוֹן דחה גרסה חדשה של העבודה, כי לא הצליח להבין אותה.",
            "ובאלף שמונה מאות שלושים ושתיים, גלואה נהרג בדו קרב, בגיל עשרים. "
            "בלילה שלפני, הוא כתב מכתב לחבר שלו שֶׁבַלְיֶה, עם עיקרי התורה שלו.",
            "רק ארבע עשרה שנה אחרי מותו, לִיוּבִיל פרסם את העבודה, והעולם הבין מה הוא איבד.",
        ]):
            self.say(sentence)
            self.play(FadeIn(row, shift=0.2 * LEFT))
            self.wait_speech()
        self.wait(1)
        self.fade_all()


class Summary(GaloisScene):
    def construct(self):
        items = VGroup(*(heb(t, 32) for t in [
            "שדה: אוסף מספרים שסגור לחשבון. מרחיבים אותו בעזרת שורשים",
            "סימטריה של הרחבה: מערבבת את השורשים ושומרת על כל החשבון",
            "בהרחבת גלואה, מספר הסימטריות שווה למימד",
            "שדות הביניים מתאימים בדיוק לתת־חבורות",
            "נוסחה ברדיקלים קיימת אם ורק אם החבורה פתירה",
        ])).arrange(DOWN, buff=0.42, aligned_edge=RIGHT).move_to(0.3 * UP)
        self.say("לסיכום. שדה הוא אוסף מספרים שסגור לחשבון, ואפשר להרחיב אותו בעזרת שורשים. "
                 "לכל הרחבה יש סימטריות, שמערבבות את השורשים בלי לשבור שום משוואה. "
                 "בהרחבת גלואה, יש בדיוק כמה סימטריות כמו המימד, ושדות הביניים מתאימים בדיוק לתת החבורות. "
                 "ולכן, השאלה אם יש נוסחה לפתרון משוואה, היא בעצם שאלה על הסימטריות שלה.")
        self.play(LaggedStartMap(FadeIn, items, shift=0.2 * LEFT, lag_ratio=0.8, run_time=12))
        self.wait_speech()
        self.fade_all()


class Credits(GaloisScene):
    def construct(self):
        thanks = heb("תודה", 60, Q_COL)
        line1 = heb("הסרטון מבוסס על הפוסטים של", 34)
        line2 = heb("גדי אלכסנדרוביץ' בבלוג \"לא מדויק\"", 44, YELLOW)
        posts = VGroup(*(heb(t, 26, GREY_A) for t in [
            "2009: תורת גלואה - מה הרעיון הבסיסי בה?",
            "2009: למה לא ניתן לפתור משוואות ממעלה חמישית ומעלה",
            "2018: מבוא לתורת גלואה",
        ])).arrange(DOWN, buff=0.18)
        site = Text("gadial.net", font=FONT, font_size=34, fill_color=GREY_B)
        voice = heb("קריינות: Azure Hila", 24, GREY_B)
        VGroup(thanks, line1, line2, posts, site, voice).arrange(DOWN, buff=0.34)
        self.say("הסרטון הזה מבוסס על פוסטים של גדי אלכסנדרוביץ' בבלוג שלו, לא מדויק: "
                 "תורת גלואה, מה הרעיון הבסיסי בה. למה לא ניתן לפתור משוואות ממעלה חמישית. ומבוא לתורת גלואה. "
                 "תודה, גדי.")
        self.play(Write(thanks))
        self.play(LaggedStartMap(FadeIn, VGroup(line1, line2, posts, site, voice), shift=0.2 * UP, lag_ratio=0.4,
                                 run_time=3))
        self.wait(3)
        self.fade_all(run_time=1.5)


SCENES = [Title, Fields, Conjugate, SqrtTwoFlip, RootsToRoots, TwoRoots, CubeRoot, SplittingField, Correspondence,
          Radicals, Quintic, History, Summary, Credits]
