"""חוק החילוף בכפל: למה 3 × 5 = 5 × 3, כשמדובר בשני סיפורים שונים לגמרי.

מיועד לתלמידי כיתה ב', עם קטעים למבוגרים (הוכחה מול בדיקה, סיבובים, קווטרניונים, פאנו).
קריינות: Azure, הקול he-IL-Hila:DragonLatestNeural (ראו use_azure ב-common.py).
רינדור: ./render.sh --hd scenes/commutative.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403
import common  # noqa: E402

common.use_azure()
common.SPOKEN.update({"פוליה": "פּוֹלְיָה", "המילטון": "הֶמִילְטוֹן", "קווטרניונים": "קְוָטֶרְנְיוֹנִים",
                      "פאנו": "פֵּאָנוֹ"})

A_COLOR = BLUE_C      # מספר הקבוצות / השורות
B_COLOR = ORANGE      # כמה בכל קבוצה / בכל שורה
APPLE = RED_D
BAG = "#c8a165"
ROUNDS = [RED_C, ORANGE, YELLOW_D, GREEN_C, BLUE_C]


class MulScene(HebrewScene):
    def caption(self, text, say=None, size=34, wait=0.3):
        super().caption(text, say=say, size=size, wait=wait)


# ---------- ציורים ----------

def apple(r=0.15, color=APPLE):
    body = Circle(radius=r).set_fill(color, 1).set_stroke(interpolate_color(color, BLACK, 0.4), 1.5)
    stem = Line(ORIGIN, 0.1 * UP).set_stroke("#8b5a2b", 3).move_to(body.get_top() + 0.04 * UP)
    leaf = Ellipse(width=0.12, height=0.06).set_fill(GREEN_C, 1).set_stroke(width=0)
    leaf.next_to(stem, RIGHT, buff=0).shift(0.03 * UP)
    return VGroup(body, stem, leaf)


def dot(color=TEAL, r=0.13):
    return Circle(radius=r).set_fill(color, 1).set_stroke(interpolate_color(color, WHITE, 0.4), 1.5)


def grid(rows, cols, make=apple, gap=0.42):
    """שורות של חפצים, כ-VGroup של שורות."""
    g = VGroup(*(VGroup(*(make().move_to([c * gap, -r * gap, 0]) for c in range(cols))) for r in range(rows)))
    return g.center()


def bags(n, per, gap=0.42, vgap=0.62):
    """n שקיות זו מתחת לזו, ובכל אחת per תפוחים בשורה."""
    out = VGroup()
    for i in range(n):
        row = VGroup(*(apple().move_to([c * gap, 0, 0]) for c in range(per)))
        frame = RoundedRectangle(width=row.get_width() + 0.3, height=0.52, corner_radius=0.14)
        frame.set_stroke(BAG, 2.5).set_fill(BAG, 0.12).move_to(row)
        out.add(VGroup(frame, row).shift(-i * vgap * UP))
    return out.center()


def bag_around(row):
    frame = RoundedRectangle(width=row.get_width() + 0.3, height=row.get_height() + 0.22, corner_radius=0.14)
    return frame.set_stroke(BAG, 2.5).set_fill(BAG, 0.12).move_to(row)


def all_apples(bag_group):
    return [a for b in bag_group for a in b[1]]


def kid(color):
    head = Circle(radius=0.28).set_fill("#f1c27d", 1).set_stroke(width=0)
    eyes = VGroup(*(Dot(radius=0.035, fill_color=BLACK).move_to(head.get_center() + [dx, 0.05, 0])
                    for dx in (-0.1, 0.1)))
    smile = Arc(start_angle=-PI * 0.8, angle=PI * 0.6, radius=0.13).set_stroke(BLACK, 2)
    smile.move_to(head.get_center() + 0.09 * DOWN)
    body = RoundedRectangle(width=0.8, height=0.75, corner_radius=0.3).set_fill(color, 1).set_stroke(width=0)
    body.next_to(head, DOWN, buff=0.04)
    return VGroup(body, head, eyes, smile)


def star(r):
    pts = [r * (1 if k % 2 == 0 else 0.45) * np.array([np.cos(PI / 2 + k * PI / 5), np.sin(PI / 2 + k * PI / 5), 0])
           for k in range(10)]
    return Polygon(*pts)


def candy(color=PINK, r=0.13):
    return Circle(radius=r).set_fill(color, 1).set_stroke(WHITE, 1.5)


SHIRT = [(-0.2, 0.5), (0.2, 0.5), (0.5, 0.4), (0.75, 0.1), (0.45, -0.02), (0.36, 0.12), (0.36, -0.5),
         (-0.36, -0.5), (-0.36, 0.12), (-0.45, -0.02), (-0.75, 0.1), (-0.5, 0.4)]
PANTS = [(-0.36, 0.5), (0.36, 0.5), (0.42, -0.5), (0.1, -0.5), (0, 0.05), (-0.1, -0.5), (-0.42, -0.5)]


def shirt(color, s=0.5):
    return Polygon(*[[x, y, 0] for x, y in SHIRT]).set_fill(color, 1).set_stroke(WHITE, 1).scale(s)


def pants(color, s=0.5):
    return Polygon(*[[x, y, 0] for x, y in PANTS]).set_fill(color, 1).set_stroke(WHITE, 1).scale(s)


def rotate_in_place(mobs, center, angle=PI / 2):
    """כל חפץ זז בקשת למקומו החדש אבל לא מסתובב, כדי שהעוקצים יישארו למעלה."""
    rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return [m.animate(path_arc=angle).move_to(center + rot @ (m.get_center() - center)) for m in mobs]


def count_up(scene, mobs, label, run_time=3):
    """סופר את החפצים אחד אחד: כל אחד מהבהב והמונה עולה."""
    label.set_value(0)
    scene.play(LaggedStart(*(Indicate(m, scale_factor=1.35, color=YELLOW) for m in mobs), lag_ratio=1 / len(mobs)),
               ChangeDecimalToValue(label, len(mobs), rate_func=linear), run_time=run_time)


# ---------- סצנות ----------

class Title(MulScene):
    def construct(self):
        eq = Tex(R"3 \times 5 = 5 \times 3", font_size=120, t2c={"3": A_COLOR, "5": B_COLOR}).shift(0.9 * UP)
        self.say("שלוש כפול חמש שווה חמש כפול שלוש. את זה יודע כל ילד.")
        self.play(Write(eq), run_time=2)
        self.wait_speech()
        q = Tex("?", font_size=150).set_color(YELLOW).next_to(eq, UP, buff=0.1)
        self.say("אבל למה זה נכון? אם חושבים על זה רגע, זה בכלל לא מובן מאליו.")
        self.play(FadeIn(q, scale=2), run_time=1)
        self.play(WiggleOutThenIn(q), run_time=1.5)
        title = heb("חוק החילוף בכפל", 84).move_to(1.6 * DOWN)
        sub = heb("למה מותר להחליף את הסדר?", 38, GREY_A).next_to(title, DOWN, buff=0.35)
        self.say("בסרטון הזה נבין למה מותר להחליף את הסדר בכפל, ונראה כמה דרכים לראות שזה תמיד נכון.")
        self.play(FadeIn(title, shift=0.3 * UP))
        self.play(FadeIn(sub))
        self.wait(1)
        self.fade_all()


class TwoStories(MulScene):
    def construct(self):
        left = bags(3, 5).scale(1.45).move_to(3.4 * LEFT + 0.6 * UP)
        right = bags(5, 3).scale(1.45).move_to(3.4 * RIGHT + 0.6 * UP)
        lt = Tex(R"3 \times 5", font_size=60, t2c={"3": A_COLOR, "5": B_COLOR}).move_to(3.4 * LEFT + 3.4 * UP)
        rt = Tex(R"5 \times 3", font_size=60, t2c={"5": A_COLOR, "3": B_COLOR}).move_to(3.4 * RIGHT + 3.4 * UP)

        self.caption("שלוש פעמים חמש: שלוש שקיות, בכל אחת חמישה תפוחים",
                     say="מה זה בכלל כפל? שלוש כפול חמש זה שלוש פעמים חמש. "
                         "למשל, שלוש שקיות, ובכל שקית חמישה תפוחים.")
        self.play(Write(lt))
        self.play(LaggedStartMap(FadeIn, left, shift=0.3 * DOWN, lag_ratio=0.3), run_time=2)
        ls = Tex("5+5+5", font_size=54).set_color(B_COLOR).move_to([left.get_x() - 0.6, -2.35, 0])
        self.caption("כדי לדעת כמה תפוחים יש, מחברים",
                     say="כדי לדעת כמה תפוחים יש, מחברים: חמש, ועוד חמש, ועוד חמש.")
        self.play(Write(ls))
        self.wait_speech()

        self.caption("חמש פעמים שלוש: חמש שקיות, בכל אחת שלושה תפוחים",
                     say="ומה זה חמש כפול שלוש? זה חמש פעמים שלוש: "
                         "חמש שקיות, ובכל שקית שלושה תפוחים. מחברים שלוש, חמש פעמים.")
        self.play(Write(rt))
        self.play(LaggedStartMap(FadeIn, right, shift=0.3 * DOWN, lag_ratio=0.2), run_time=2)
        rs = Tex("3+3+3+3+3", font_size=54).set_color(B_COLOR).move_to([right.get_x() - 0.6, -2.35, 0])
        self.play(Write(rs))
        self.wait_speech()

        self.caption("שני סיפורים שונים",
                     say="שימו לב: אלה שני סיפורים שונים. בצד אחד יש מעט שקיות מלאות, "
                         "ובצד השני הרבה שקיות עם מעט תפוחים. גם החשבון שונה: "
                         "בצד אחד מחברים שלושה מספרים, ובצד השני חמישה.")
        self.play(LaggedStart(*(Indicate(b[0], color=YELLOW) for b in left), lag_ratio=0.3), run_time=2)
        self.play(LaggedStart(*(Indicate(b[0], color=YELLOW) for b in right), lag_ratio=0.2), run_time=2)
        self.wait_speech()

        lc = Integer(0, font_size=60, color=YELLOW).next_to(ls, RIGHT, buff=0.3)
        rc = Integer(0, font_size=60, color=YELLOW).next_to(rs, RIGHT, buff=0.3)
        leq = Tex("=", font_size=54).next_to(ls, RIGHT, buff=0.2)
        req = Tex("=", font_size=54).next_to(rs, RIGHT, buff=0.2)
        lc.next_to(leq, RIGHT, buff=0.2)
        rc.next_to(req, RIGHT, buff=0.2)
        self.caption("עכשיו נספור", say="עכשיו נספור את התפוחים. בצד שמאל:")
        self.add(leq, lc)
        count_up(self, all_apples(left), lc)
        self.say("חמש עשרה. ובצד ימין:")
        self.wait_speech()
        self.add(req, rc)
        count_up(self, all_apples(right), rc)
        self.caption("שוב חמש עשרה! צירוף מקרים?",
                     say="שוב חמש עשרה! שני סיפורים שונים, שני חשבונות שונים, ואותה תוצאה. צירוף מקרים?")
        self.play(Indicate(lc, scale_factor=1.4), Indicate(rc, scale_factor=1.4), run_time=1.5)
        self.fade_all()


class Addition(MulScene):
    def construct(self):
        a = VGroup(*(dot(A_COLOR, 0.2) for _ in range(3))).arrange(RIGHT, buff=0.2)
        b = VGroup(*(dot(B_COLOR, 0.2) for _ in range(5))).arrange(RIGHT, buff=0.2)
        row = VGroup(a, b).arrange(RIGHT, buff=0.7).shift(0.6 * UP)
        t1 = Tex("3 + 5", font_size=70, t2c={"3": A_COLOR, "5": B_COLOR}).next_to(row, UP, buff=0.6)
        self.caption("בחיבור קל לראות שהסדר לא משנה",
                     say="קודם, בואו נראה מקרה קל יותר: חיבור. שלוש ועוד חמש: "
                         "שמים שלושה כדורים כחולים, ולידם חמישה כתומים.")
        self.play(Write(t1), FadeIn(a, lag_ratio=0.2), FadeIn(b, lag_ratio=0.2), run_time=2)
        self.wait_speech()
        t2 = Tex("5 + 3", font_size=70, t2c={"3": A_COLOR, "5": B_COLOR}).move_to(t1)
        self.caption("הולכים לצד השני של השולחן",
                     say="עכשיו הולכים לצד השני של השולחן. משם רואים קודם את החמישה הכתומים, "
                         "ואחריהם את השלושה הכחולים. חמש ועוד שלוש. אבל אלה אותם כדורים בדיוק!")
        self.play(Rotate(row, PI), run_time=2.5)
        self.play(TransformMatchingTex(t1, t2), run_time=1.2)
        self.wait_speech()
        self.caption("בחיבור, שני המספרים ממלאים אותו תפקיד",
                     say="בחיבור, שני המספרים ממלאים אותו תפקיד: שניהם ערימה של כדורים. "
                         "לכן קל לראות שהסדר לא משנה. בכפל המצב אחר לגמרי.")
        self.fade_all()


class Roles(MulScene):
    def construct(self):
        def bill():
            r = RoundedRectangle(width=2.4, height=1.15, corner_radius=0.1).set_fill("#2e7d4f", 1)
            r.set_stroke("#a5d6a7", 3)
            return VGroup(r, Text("100", font=FONT, font_size=44, fill_color="#e8f5e9").move_to(r))

        bills = VGroup(bill(), bill()).arrange(DOWN, buff=0.3).move_to(3.6 * LEFT + 0.4 * UP)
        coins = VGroup(*(Circle(radius=0.12).set_fill(GOLD_D, 1).set_stroke(GOLD_A, 1.5) for _ in range(100)))
        coins.arrange_in_grid(10, 10, buff=0.07).move_to(3.6 * RIGHT + 0.4 * UP)
        lt = Tex(R"2 \times 100", font_size=56).next_to(bills, UP, buff=0.5)
        rt = Tex(R"100 \times 2", font_size=56).move_to(lt).set_x(coins.get_x())
        self.caption("פעמיים מאה: שני שטרות של מאה שקל",
                     say="כדי להרגיש עד כמה הסיפורים שונים, חשבו על כסף. "
                         "פעמיים מאה: שני שטרות של מאה שקל.")
        self.play(Write(lt), LaggedStartMap(FadeIn, bills, shift=0.3 * DOWN, lag_ratio=0.4))
        self.wait_speech()
        self.caption("מאה פעמים שתיים: מאה מטבעות של שני שקלים",
                     say="ומאה פעמים שתיים: מאה מטבעות של שני שקלים.")
        self.play(Write(rt), LaggedStartMap(FadeIn, coins, scale=0.3, lag_ratio=0.02), run_time=3)
        self.wait_speech()
        self.caption("אותו סכום, אבל נסו להכניס מאה מטבעות לארנק",
                     say="בשני המקרים יש בדיוק מאתיים שקל. אבל נסו להכניס מאה מטבעות לארנק! "
                         "זה לא אותו דבר.")
        self.play(WiggleOutThenIn(coins), run_time=2)
        self.wait_speech()
        self.play(*map(FadeOut, [bills, coins, lt, rt]))

        eq = Tex(R"3 \times 5", font_size=130, t2c={"3": A_COLOR, "5": B_COLOR}).shift(0.8 * UP)
        b1 = Brace(eq[0], DOWN, buff=0.2).set_color(A_COLOR)
        b2 = Brace(eq[2], DOWN, buff=0.2).set_color(B_COLOR)
        l1 = heb("כמה קבוצות", 32, A_COLOR).next_to(b1, DOWN).shift(0.4 * LEFT)
        l2 = heb("כמה בכל קבוצה", 32, B_COLOR).next_to(b2, DOWN).shift(0.4 * RIGHT)
        self.caption("בכפל, לכל מספר יש תפקיד משלו",
                     say="בכפל, לכל אחד מהמספרים יש תפקיד אחר. המספר הראשון אומר כמה קבוצות יש, "
                         "והשני אומר כמה דברים יש בכל קבוצה.")
        self.play(Write(eq))
        self.play(GrowFromCenter(b1), FadeIn(l1))
        self.play(GrowFromCenter(b2), FadeIn(l2))
        self.wait_speech()
        self.caption("כשמחליפים את הסדר, השקיות הופכות לתפוחים",
                     say="אז כשמחליפים את הסדר, מחליפים גם את התפקידים: "
                         "מספר השקיות הופך למספר התפוחים בכל שקית, ולהפך. למה שהתוצאה תישאר אותו דבר?")
        self.play(Swap(eq[0], eq[2]), run_time=1.5)
        self.wait_speech()
        self.fade_all()


class OrderMatters(MulScene):
    def construct(self):
        socks = VGroup(heb("קודם גרביים, אחר כך נעליים", 38, GREEN_B),
                       heb("קודם נעליים, אחר כך גרביים", 38, RED_B)).arrange(DOWN, buff=0.4).shift(2.2 * UP)
        self.caption("בדרך כלל, לסדר יש משמעות",
                     say="וזה באמת מפתיע, כי בדרך כלל הסדר חשוב מאוד. "
                         "קודם גרביים ואחר כך נעליים, זה הגיוני. קודם נעליים ואחר כך גרביים? זה כבר מצחיק.")
        self.play(FadeIn(socks[0], shift=0.2 * UP))
        self.play(FadeIn(socks[1], shift=0.2 * UP))
        self.wait_speech()

        pairs = [(R"5 - 3 = 2", R"3 - 5 = \,?"), (R"6 : 2 = 3", R"2 : 6 = \tfrac13"), (R"2^3 = 8", R"3^2 = 9")]
        rows = VGroup(*(VGroup(Tex(x, font_size=56).set_color(GREEN_B), Tex(y, font_size=56).set_color(RED_B))
                        .arrange(RIGHT, buff=1.2) for x, y in pairs)).arrange(DOWN, buff=0.5).shift(0.6 * DOWN)
        self.caption("גם בחשבון: חיסור וחילוק",
                     say="גם בחשבון. אם יש לי חמש עוגיות ואני אוכלת שלוש, נשארות שתיים. "
                         "אבל אם יש לי שלוש, אני לא יכולה לאכול חמש. "
                         "וגם לחלק שש עוגיות לשני ילדים זה לא כמו לחלק שתי עוגיות לשישה ילדים.")
        self.play(FadeIn(rows[0], shift=0.2 * UP))
        self.play(FadeIn(rows[1], shift=0.2 * UP))
        self.wait_speech()
        self.caption("ובחזקה, שהיא כפל חוזר, כמעט אבל לא",
                     say="ולמבוגרים שבינינו: חזקה היא כפל חוזר, כמו שכפל הוא חיבור חוזר. "
                         "שתיים בחזקת שלוש זה שתיים כפול שתיים כפול שתיים, שמונה. "
                         "שלוש בחזקת שתיים זה שלוש כפול שלוש, תשע. כמעט, אבל לא שווה.")
        self.play(FadeIn(rows[2], shift=0.2 * UP))
        self.play(Indicate(rows[2], scale_factor=1.15))
        self.wait_speech()
        self.caption("אז חוק החילוף בכפל צריך הסבר",
                     say="אז העובדה שבכפל מותר להחליף את הסדר היא לא מובנת מאליה. היא צריכה הסבר.")
        self.fade_all()


class Checking(MulScene):
    def construct(self):
        l1 = Tex(R"7 \times 8 = 8+8+8+8+8+8+8 = 56", font_size=48)
        l2 = Tex(R"8 \times 7 = 7+7+7+7+7+7+7+7 = 56", font_size=48)
        l3 = Tex(R"12 \times 17 = 204 = 17 \times 12", font_size=48)
        l4 = Tex(R"123 \times 45 = 5535 = 45 \times 123", font_size=48)
        lines = VGroup(l1, l2, l3, l4).arrange(DOWN, buff=0.45).shift(0.7 * UP)
        self.caption("אפשר לבדוק עוד דוגמאות",
                     say="אפשר לנסות עוד דוגמאות. שבע כפול שמונה: מחברים שמונה, שבע פעמים, "
                         "ויוצא חמישים ושש. שמונה כפול שבע: מחברים שבע, שמונה פעמים. שוב חמישים ושש.")
        self.play(Write(l1), run_time=2)
        self.play(Write(l2), run_time=2)
        self.wait_speech()
        self.caption("תמיד יוצא שווה",
                     say="גם עם מספרים גדולים יותר, תמיד יוצא שווה.")
        self.play(FadeIn(l3, shift=0.2 * UP))
        self.play(FadeIn(l4, shift=0.2 * UP))
        self.wait_speech()
        self.caption("אבל יש אינסוף מספרים",
                     say="אבל יש אינסוף מספרים. גם אם נבדוק מיליון דוגמאות, "
                         "איך נדע שהדוגמה הבאה לא תפתיע אותנו?")
        self.play(FadeOut(lines))
        qs = VGroup(*(Tex(R"? \times ? = ? \times ?", font_size=34).set_color(GREY_B) for _ in range(9)))
        for i, q in enumerate(qs):
            q.move_to([(-5 + 5 * (i % 3)), 2.6 - 1.5 * (i // 3), 0])
        self.play(LaggedStartMap(FadeIn, qs, scale=0.5, lag_ratio=0.15), run_time=2)
        self.wait_speech()
        self.caption("זה כבר קרה: השערה שנפלה אחרי מאות מיליונים",
                     say="ולמבוגרים: במתמטיקה זה באמת קרה. השערה מפורסמת של המתמטיקאי ג'ורג' פוליה "
                         "נכונה לכל המספרים עד תשע מאות ושישה מיליון בערך, ושם, פתאום, היא נכשלת.")
        big = Integer(906150257, font_size=72, color=RED_B).shift(0.5 * UP)
        self.play(FadeOut(qs), FadeIn(big, scale=0.5))
        self.wait_speech()
        self.caption("אנחנו צריכים סיבה, לא רק דוגמאות",
                     say="אז דוגמאות לא מספיקות. אנחנו צריכים סיבה: הסבר שיראה למה זה חייב להיות נכון, תמיד.")
        self.fade_all()


class Array(MulScene):
    def construct(self):
        b = bags(3, 5, gap=0.55, vgap=0.7).scale(1.3).shift(0.5 * UP)
        t = Tex(R"3 \times 5", font_size=64, t2c={"3": A_COLOR, "5": B_COLOR}).to_corner(UL, buff=0.7)
        self.caption("ההסבר הראשון: נסדר את התפוחים",
                     say="הנה ההסבר הראשון. ניקח את שלוש השקיות שלנו, עם חמישה תפוחים בכל אחת.")
        self.play(Write(t), LaggedStartMap(FadeIn, b, shift=0.3 * DOWN, lag_ratio=0.3))
        self.wait_speech()
        apples = VGroup(*all_apples(b))
        self.caption("שלוש שורות, בכל שורה חמישה",
                     say="נוציא את התפוחים מהשקיות ונשאיר אותם מסודרים בשורות: "
                         "שלוש שורות, ובכל שורה חמישה תפוחים.")
        self.play(*(FadeOut(x[0]) for x in b), run_time=1.2)
        self.play(apples.animate.arrange_in_grid(3, 5, buff=0.55).move_to(0.5 * UP))
        self.wait_speech()
        self.caption("ועכשיו נסובב",
                     say="ועכשיו, נסובב את כל הסידור ברבע סיבוב. צפו טוב.")
        self.wait(0.8)
        self.play(*rotate_in_place(apples, apples.get_center()), run_time=3)
        self.wait_speech()
        rows = [VGroup(*sorted(apples, key=lambda a: -a.get_y())[3 * i:3 * i + 3]) for i in range(5)]
        frames = VGroup(*(bag_around(r) for r in rows))
        t2 = Tex(R"5 \times 3", font_size=64, t2c={"5": A_COLOR, "3": B_COLOR}).move_to(t)
        self.caption("עכשיו יש חמש שורות, בכל שורה שלושה",
                     say="מה קיבלנו? חמש שורות, ובכל שורה שלושה תפוחים. אם נשים כל שורה בשקית, "
                         "יש לנו חמש שקיות של שלושה. זה בדיוק חמש כפול שלוש!")
        self.play(LaggedStartMap(FadeIn, frames, lag_ratio=0.2), TransformMatchingTex(t, t2), run_time=2)
        self.wait_speech()
        self.caption("לא הוספנו ולא הורדנו אף תפוח",
                     say="ולא הוספנו אף תפוח, ולא הורדנו אף תפוח. רק סובבנו. "
                         "לכן שלוש כפול חמש חייב להיות שווה לחמש כפול שלוש.")
        eq = Tex(R"3 \times 5 = 5 \times 3", font_size=64, t2c={"3": A_COLOR, "5": B_COLOR}).to_corner(UL, buff=0.7)
        self.play(TransformMatchingTex(t2, eq))
        self.wait_speech()
        self.caption("אפשר גם בלי לסובב: מסתכלים מהצד",
                     say="ואפשר אפילו בלי לסובב שום דבר: פשוט לעמוד בצד אחר של השולחן ולהסתכל.")
        self.fade_all()


class RowsColumns(MulScene):
    def construct(self):
        g = grid(4, 6, lambda: dot(TEAL, 0.17), gap=0.62).shift(0.5 * UP)
        self.caption("אותו סידור, שתי דרכים לספור",
                     say="הנה עוד דרך לראות את אותו רעיון, בלי לזוז בכלל. "
                         "סידור של ארבע שורות ושש עמודות.")
        self.play(LaggedStartMap(FadeIn, VGroup(*g.family_members_with_points()), scale=0.4, lag_ratio=0.02),
                  run_time=1.5)
        self.wait_speech()
        self.caption("לפי השורות: ארבע שורות של שש",
                     say="נספור לפי השורות: שש, שתים עשרה, שמונה עשרה, עשרים וארבע. "
                         "ארבע שורות של שש: ארבע כפול שש.")
        boxes = VGroup()
        labels = VGroup()
        for i, row in enumerate(g):
            box = SurroundingRectangle(row, buff=0.1).set_stroke(A_COLOR, 3)
            lab = Integer(6 * (i + 1), font_size=40, color=A_COLOR).next_to(box, RIGHT, buff=0.3)
            self.play(ShowCreation(box), FadeIn(lab), run_time=0.8)
            boxes.add(box), labels.add(lab)
        t1 = Tex(R"4 \times 6 = 24", font_size=56).set_color(A_COLOR).to_corner(UL, buff=0.6)
        self.play(Write(t1))
        self.wait_speech()
        self.play(FadeOut(boxes), FadeOut(labels))
        self.caption("לפי העמודות: שש עמודות של ארבע",
                     say="ועכשיו לפי העמודות: ארבע, שמונה, שתים עשרה, שש עשרה, עשרים, עשרים וארבע. "
                         "שש עמודות של ארבע: שש כפול ארבע.")
        for j in range(6):
            col = VGroup(*(row[j] for row in g))
            box = SurroundingRectangle(col, buff=0.1).set_stroke(B_COLOR, 3)
            lab = Integer(4 * (j + 1), font_size=36, color=B_COLOR).next_to(box, DOWN, buff=0.2)
            self.play(ShowCreation(box), FadeIn(lab), run_time=0.7)
            boxes.add(box), labels.add(lab)
        t2 = Tex(R"6 \times 4 = 24", font_size=56).set_color(B_COLOR).to_corner(UR, buff=0.6)
        self.play(Write(t2))
        self.wait_speech()
        self.caption("סופרים את אותן נקודות, פעם כך ופעם כך",
                     say="ספרנו את אותן נקודות בדיוק, פעם לפי השורות ופעם לפי העמודות. "
                         "מספר הנקודות לא יכול להשתנות רק בגלל שספרנו אותן בסדר אחר.")
        self.play(Indicate(t1), Indicate(t2))
        self.fade_all()


class AnyNumbers(MulScene):
    def construct(self):
        self.caption("זה עובד עם כל שני מספרים",
                     say="והנה הדבר החשוב: ההסבר הזה לא השתמש בשום דבר מיוחד במספרים שלוש וחמש. "
                         "הוא עובד עם כל שני מספרים.")
        prev = None
        for r, c in [(2, 7), (6, 3), (5, 9), (8, 12)]:
            gap = min(0.5, 5.2 / max(r, c))
            g = grid(r, c, lambda: dot(TEAL, gap * 0.32), gap=gap).shift(0.4 * UP)
            eq = Tex(Rf"{r} \times {c} = {c} \times {r}", font_size=56).to_corner(UL, buff=0.6)
            if prev is None:
                self.play(FadeIn(g, lag_ratio=0.02), Write(eq))
            else:
                self.play(FadeOut(prev[0]), FadeIn(g, lag_ratio=0.02), ReplacementTransform(prev[1], eq))
            self.play(Rotate(g, PI / 2), run_time=1.5)
            prev = (g, eq)
        self.wait_speech()
        self.play(FadeOut(prev[0]), FadeOut(prev[1]))

        rect = Rectangle(width=4, height=2.2).set_stroke(TEAL, 3).set_fill(TEAL, 0.15).shift(0.2 * UP)
        inner = VGroup(*(dot(TEAL, 0.09).move_to(rect.get_corner(UL) + [0.35 + 0.5 * i, -0.3 - 0.5 * j, 0])
                         for i in range(3) for j in range(2)))
        dots_h = Tex(R"\cdots", font_size=50).move_to(rect.get_center() + 0.7 * RIGHT + 0.55 * UP)
        dots_v = Tex(R"\vdots", font_size=50).move_to(rect.get_center() + 1.35 * LEFT + 0.3 * DOWN)
        stuff = VGroup(rect, inner, dots_h, dots_v)
        ba = Brace(rect, LEFT).set_color(A_COLOR)
        bb = Brace(rect, UP).set_color(B_COLOR)
        la = Tex("a", font_size=56).set_color(A_COLOR).next_to(ba, LEFT)
        lb = Tex("b", font_size=56).set_color(B_COLOR).next_to(bb, UP)
        eq = Tex(R"a \times b = b \times a", font_size=64, t2c={"a": A_COLOR, "b": B_COLOR}).shift(2.6 * DOWN)
        self.caption("כל מכפלה היא מלבן, וכל מלבן אפשר לסובב",
                     say="כל מכפלה של שני מספרים אפשר לסדר כמלבן: אַ שורות, ובכל שורה בֵּ נקודות. "
                         "וכל מלבן אפשר לסובב, ואז יש בֵּ שורות, ובכל שורה אַ נקודות.")
        self.play(FadeIn(stuff), GrowFromCenter(ba), GrowFromCenter(bb), FadeIn(la), FadeIn(lb))
        self.wait(0.5)
        self.play(*map(FadeOut, [ba, bb, la, lb]))
        self.play(Rotate(stuff, PI / 2), run_time=2)
        ba2 = Brace(stuff, LEFT).set_color(B_COLOR)
        bb2 = Brace(stuff, UP).set_color(A_COLOR)
        lb2 = Tex("b", font_size=56).set_color(B_COLOR).next_to(ba2, LEFT)
        la2 = Tex("a", font_size=56).set_color(A_COLOR).next_to(bb2, UP)
        self.play(GrowFromCenter(ba2), GrowFromCenter(bb2), FadeIn(la2), FadeIn(lb2))
        self.play(Write(eq))
        self.wait_speech()
        self.caption("זו כבר לא בדיקה. זו הוכחה",
                     say="וזה ההבדל הגדול. דוגמאות בודקות מקרה אחד בכל פעם. "
                         "ההסבר הזה מכסה את כל אינסוף המקרים בבת אחת. במתמטיקה, לזה קוראים הוכחה.")
        self.play(Indicate(eq, scale_factor=1.15))
        self.fade_all()


class Chocolate(MulScene):
    def construct(self):
        def bar(r, c, s=0.62):
            sq = VGroup(*(Square(s).set_fill("#5b3a1e", 1).set_stroke("#9c6b3c", 3)
                          .move_to([j * s, -i * s, 0]) for i in range(r) for j in range(c)))
            return sq.center()

        choc = bar(3, 5).shift(0.6 * UP)
        self.caption("גם שוקולד עובד ככה",
                     say="אותו דבר עם טבלת שוקולד. שלוש שורות של חמש קוביות. "
                         "מסובבים את הטבלה, ויש חמש שורות של שלוש קוביות. ואף קוביה לא נאכלה בדרך.")
        self.play(FadeIn(choc, lag_ratio=0.05), run_time=1.5)
        self.play(Rotate(choc, PI / 2), run_time=2)
        self.wait_speech()
        self.play(FadeOut(choc))

        unit = 1.0
        plane = NumberPlane((-1, 6), (-1, 4), width=7 * unit, height=5 * unit,
                            background_line_style=dict(stroke_color=GREY_D, stroke_width=1))
        plane.shift(0.4 * UP)
        rect = Rectangle(width=2.5 * unit, height=1.5 * unit).set_fill(GOLD_E, 0.6).set_stroke(GOLD, 3)
        rect.move_to(plane.c2p(0, 0), DL)
        w = Tex("2.5", font_size=40).next_to(rect, DOWN, buff=0.15)
        h = Tex("1.5", font_size=40).next_to(rect, LEFT, buff=0.15)
        self.caption("ומה עם מספרים שאינם שלמים?",
                     say="ולמבוגרים: מה עם מספרים שאינם שלמים, כמו שתיים וחצי כפול אחת וחצי? "
                         "אי אפשר לסדר שתיים וחצי שורות של תפוחים. אבל אפשר לצייר מלבן.")
        self.play(FadeIn(plane), run_time=1)
        self.play(DrawBorderThenFill(rect), FadeIn(w), FadeIn(h))
        self.wait_speech()
        self.caption("המכפלה היא שטח המלבן, והשטח לא משתנה כשמסובבים",
                     say="המכפלה היא השטח של המלבן. כשמסובבים את המלבן, הצלעות מתחלפות, "
                         "אבל השטח שלו לא משתנה. אותו רעיון בדיוק, רק עם שטח במקום ספירה.")
        self.play(FadeOut(w), FadeOut(h))
        self.play(Rotate(rect, PI / 2, about_point=rect.get_corner(DR)), run_time=2)
        self.play(rect.animate.move_to(plane.c2p(0, 0), DL))
        w2 = Tex("1.5", font_size=40).next_to(rect, DOWN, buff=0.15)
        h2 = Tex("2.5", font_size=40).next_to(rect, LEFT, buff=0.15)
        self.play(FadeIn(w2), FadeIn(h2))
        eq = Tex(R"2.5 \times 1.5 = 1.5 \times 2.5 = 3.75", font_size=48).to_corner(UR, buff=0.5)
        self.play(Write(eq))
        self.wait_speech()
        self.fade_all()


class Dealing(MulScene):
    def construct(self):
        xs = [-3.6, 0, 3.6]
        kids = VGroup(*(kid(c).move_to([x, -2.35, 0]) for x, c in zip(xs, [PURPLE_B, TEAL_D, MAROON_C])))
        pile = VGroup(*(candy(GREY_A) for _ in range(15)))
        pile.arrange_in_grid(3, 5, buff=0.08).move_to(2.9 * UP)
        self.caption("סבתא מחלקת סוכריות לשלושה נכדים",
                     say="הסבר נוסף, הפעם בלי לסובב שום דבר. סבתא רוצה לתת לשלושה נכדים "
                         "חמש סוכריות לכל אחד. כמה סוכריות היא צריכה? שלוש פעמים חמש.")
        self.play(LaggedStartMap(FadeIn, kids, shift=0.3 * UP, lag_ratio=0.3), FadeIn(pile, lag_ratio=0.05))
        self.wait_speech()
        self.caption("היא מחלקת כמו קלפים: סוכריה לכל נכד, וחוזר חלילה",
                     say="איך היא מחלקת? כמו שמחלקים קלפים: סוכריה לנכד הראשון, לשני, לשלישי. "
                         "ואז עוד סיבוב, ועוד סיבוב.")
        counter = VGroup(heb("סיבוב", 34), Integer(0, font_size=40)).arrange(LEFT, buff=0.2).to_corner(UR, buff=0.5)
        self.play(FadeIn(counter))
        k = 14
        placed = [[None] * 5 for _ in range(3)]
        for r in range(5):
            anims = []
            for i in range(3):
                cnd = pile[k]
                k -= 1
                placed[i][r] = cnd
                target = [xs[i], -1.05 + r * 0.48, 0]
                anims.append(cnd.animate.move_to(target).set_fill(ROUNDS[r]))
            self.play(counter[1].animate.set_value(r + 1), LaggedStart(*anims, lag_ratio=0.35),
                      run_time=1.6 if r < 2 else 1.0)
        self.wait_speech()
        self.caption("בכל סיבוב יצאו שלוש סוכריות, והיו חמישה סיבובים",
                     say="בכל סיבוב יצאו שלוש סוכריות, אחת לכל נכד. וכמה סיבובים היו? חמישה, "
                         "כי כל נכד צריך חמש. אז סבתא חילקה חמש פעמים שלוש.")
        rows = VGroup(*(SurroundingRectangle(VGroup(*(placed[i][r] for i in range(3))), buff=0.08)
                        .set_stroke(ROUNDS[r], 2.5) for r in range(5)))
        self.play(LaggedStartMap(ShowCreation, rows, lag_ratio=0.3), run_time=2)
        self.wait_speech()
        cols = VGroup(*(SurroundingRectangle(VGroup(*placed[i]), buff=0.1).set_stroke(WHITE, 3) for i in range(3)))
        t1 = Tex(R"3 \times 5", font_size=54).move_to(4.9 * LEFT + 2.8 * UP)
        t2 = Tex(R"5 \times 3", font_size=54).next_to(t1, DOWN, buff=0.3)
        self.caption("לפי נכדים: שלוש פעמים חמש. לפי סיבובים: חמש פעמים שלוש",
                     say="לפי הנכדים: שלושה נכדים, לכל אחד חמש. לפי הסיבובים: חמישה סיבובים, בכל אחד שלוש. "
                         "אותן סוכריות בדיוק! ושימו לב, הסוכריות הסתדרו בעצמן במלבן.")
        self.play(FadeOut(rows), ShowCreation(cols), Write(t1))
        self.play(FadeOut(cols), FadeIn(rows), Write(t2))
        self.wait_speech()
        self.fade_all()


class Outfits(MulScene):
    def construct(self):
        sc = [RED_C, YELLOW_D, PURPLE_B]
        pc = [BLUE_D, GREY_B, TEAL_D, MAROON_C, GOLD_E]
        x0, y0, dx, dy = -2.6, 1.0, 1.3, 1.25
        shirts = VGroup(*(shirt(c, 0.6).move_to([x0 - 1.5, y0 - i * dy, 0]) for i, c in enumerate(sc)))
        pantss = VGroup(*(pants(c, 0.6).move_to([x0 + j * dx, y0 + 1.35, 0]) for j, c in enumerate(pc)))
        self.caption("שלוש חולצות וחמישה מכנסיים",
                     say="ועוד הסבר, מעולם הבגדים. בארון יש שלוש חולצות וחמישה מכנסיים. "
                         "כמה תלבושות שונות אפשר להרכיב?")
        self.play(LaggedStartMap(FadeIn, shirts, shift=0.2 * RIGHT, lag_ratio=0.3),
                  LaggedStartMap(FadeIn, pantss, shift=0.2 * DOWN, lag_ratio=0.2), run_time=2)
        self.wait_speech()
        cells = [[None] * 5 for _ in range(3)]
        self.caption("כל תלבושת היא זוג: חולצה ומכנסיים",
                     say="כל תלבושת היא זוג: חולצה אחת ומכנסיים אחד. נסדר את כל הזוגות בטבלה.")
        anims = []
        for i in range(3):
            for j in range(5):
                o = VGroup(shirt(sc[i], 0.32), pants(pc[j], 0.32)).arrange(DOWN, buff=0.02)
                o.move_to([x0 + j * dx, y0 - i * dy, 0])
                cells[i][j] = o
                anims.append(FadeIn(o, scale=0.5))
        self.play(LaggedStart(*anims, lag_ratio=0.12), run_time=3)
        self.wait_speech()
        self.caption("לפי החולצות: לכל חולצה חמישה מכנסיים",
                     say="נספור לפי החולצות: לכל חולצה יש חמישה מכנסיים. שלוש חולצות, כל אחת עם חמישה: "
                         "שלוש כפול חמש.")
        boxes = VGroup(*(SurroundingRectangle(VGroup(shirts[i], *cells[i]), buff=0.08).set_stroke(A_COLOR, 3)
                         for i in range(3)))
        self.play(LaggedStartMap(ShowCreation, boxes, lag_ratio=0.4), run_time=2)
        self.wait_speech()
        self.play(FadeOut(boxes))
        self.caption("לפי המכנסיים: לכל מכנסיים שלוש חולצות",
                     say="ועכשיו לפי המכנסיים: לכל מכנסיים יש שלוש חולצות. חמישה מכנסיים, "
                         "כל אחד עם שלוש: חמש כפול שלוש.")
        boxes = VGroup(*(SurroundingRectangle(VGroup(pantss[j], *(cells[i][j] for i in range(3))), buff=0.08)
                         .set_stroke(B_COLOR, 3) for j in range(5)))
        self.play(LaggedStartMap(ShowCreation, boxes, lag_ratio=0.3), run_time=2)
        self.wait_speech()
        self.caption("אותם זוגות, נספרים פעמיים",
                     say="אלה אותן חמש עשרה תלבושות. ולמבוגרים: כך מתמטיקאים מגדירים כפל בעומק. "
                         "אַ כפול בֵּ הוא מספר הזוגות שאפשר להרכיב מקבוצה של אַ דברים וקבוצה של בֵּ דברים. "
                         "וזוג חולצה ומכנסיים הוא אותו זוג, לא משנה מאיזה צד מתחילים.")
        self.play(FadeOut(boxes))
        self.play(LaggedStart(*(Indicate(cells[i][j]) for i in range(3) for j in range(5)), lag_ratio=0.08),
                  run_time=2.5)
        self.wait_speech()
        self.fade_all()


class TimesTable(MulScene):
    def construct(self):
        n, s = 10, 0.5
        origin = np.array([-2.3, 2.75, 0])
        cells = {}
        group = VGroup()
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                t = Text(str(i * j), font=FONT, font_size=20, fill_color=GREY_A)
                t.move_to(origin + [(j - 1) * s, -(i - 1) * s, 0])
                cells[i, j] = t
                group.add(t)
        heads = VGroup(*(Text(str(k), font=FONT, font_size=22, fill_color=YELLOW)
                         .move_to(origin + [(k - 1) * s, s, 0]) for k in range(1, n + 1)),
                       *(Text(str(k), font=FONT, font_size=22, fill_color=YELLOW)
                         .move_to(origin + [-s, -(k - 1) * s, 0]) for k in range(1, n + 1)))
        table = VGroup(group, heads).shift(0.1 * DOWN)
        self.caption("ולמה זה שימושי? לוח הכפל",
                     say="ולמה זה שימושי? הנה לוח הכפל, מאה תרגילים.")
        self.play(FadeIn(heads), FadeIn(group, lag_ratio=0.01), run_time=2)
        self.wait_speech()
        c1 = SurroundingRectangle(cells[3, 7], buff=0.06).set_stroke(A_COLOR, 3)
        c2 = SurroundingRectangle(cells[7, 3], buff=0.06).set_stroke(B_COLOR, 3)
        self.caption("שלוש כפול שבע, ושבע כפול שלוש",
                     say="שלוש כפול שבע הוא עשרים ואחת. ושבע כפול שלוש? גם עשרים ואחת, "
                         "בדיוק בצד השני של האלכסון.")
        self.play(ShowCreation(c1))
        self.play(ShowCreation(c2))
        diag = VGroup(*(SurroundingRectangle(cells[k, k], buff=0.04).set_stroke(GREEN_B, 2) for k in range(1, n + 1)))
        self.play(LaggedStartMap(ShowCreation, diag, lag_ratio=0.1), run_time=1.5)
        self.wait_speech()
        lower = VGroup(*(cells[i, j] for i in range(1, n + 1) for j in range(1, n + 1) if i > j))
        self.caption("הלוח הוא תמונת מראה של עצמו: צריך לזכור רק חצי",
                     say="כל הלוח הוא תמונת מראה של עצמו סביב האלכסון. בזכות חוק החילוף, "
                         "צריך ללמוד רק את החצי העליון. במקום מאה תרגילים, חמישים וחמישה.")
        self.play(lower.animate.set_opacity(0.12), FadeOut(c1), FadeOut(c2), run_time=1.5)
        self.wait_speech()
        self.play(lower.animate.set_opacity(1), FadeOut(diag))
        self.play(FadeOut(table))
        trick = VGroup(Tex(R"9 \times 2", font_size=72), Tex(R"\to", font_size=72),
                       Tex(R"2 \times 9 = 9 + 9 = 18", font_size=72)).arrange(RIGHT, buff=0.4).shift(0.5 * UP)
        self.caption("וגם טריק: לבחור את הסדר הנוח",
                     say="וגם טריק לחשבון בעל פה: תשע פעמים שתיים זה לחבר שתיים תשע פעמים. "
                         "אבל אפשר להפוך לשתיים פעמים תשע: תשע ועוד תשע, שמונה עשרה. הרבה יותר קל.")
        self.play(Write(trick[0]))
        self.play(FadeIn(trick[1]), Write(trick[2]))
        self.wait_speech()
        self.fade_all()


class NotAlways(MulScene):
    def construct(self):
        def letter():
            return Text("F", font="DejaVu Sans", font_size=150, fill_color=YELLOW, weight=BOLD)

        panels = VGroup(*(RoundedRectangle(width=5.6, height=4.6, corner_radius=0.2).set_stroke(GREY_B, 2)
                          for _ in range(2))).arrange(RIGHT, buff=0.8).shift(0.4 * UP)
        f1, f2 = letter().move_to(panels[0]), letter().move_to(panels[1])
        h1 = heb("קודם סיבוב, אחר כך היפוך", 30).next_to(panels[0], UP, buff=0.2)
        h2 = heb("קודם היפוך, אחר כך סיבוב", 30).next_to(panels[1], UP, buff=0.2)
        self.caption("האם כל כפל הוא חילופי?",
                     say="ולמבוגרים, שאלה לסיום: האם כל פעולה שקוראים לה כפל מקיימת חוק חילוף? "
                         "מתמטיקאים קוראים כפל גם לפעולה של לעשות תנועה אחת ואחריה עוד תנועה. "
                         "קחו את האות אֶף, ושתי תנועות: סיבוב ברבע, והיפוך כמו במראה.")
        self.play(FadeIn(panels), FadeIn(f1), FadeIn(f2), FadeIn(h1), FadeIn(h2))
        self.wait_speech()
        self.caption("משמאל סיבוב ואז היפוך. מימין היפוך ואז סיבוב",
                     say="בצד שמאל: קודם מסובבים, אחר כך הופכים. בצד ימין: קודם הופכים, אחר כך מסובבים.")
        self.play(Rotate(f1, PI / 2), run_time=1.5)
        self.play(f1.animate.flip(UP), run_time=1.5)
        self.play(f2.animate.flip(UP), run_time=1.5)
        self.play(Rotate(f2, PI / 2), run_time=1.5)
        self.wait_speech()
        neq = Tex(R"\neq", font_size=90).set_color(RED_B).move_to(panels)
        self.caption("התוצאה שונה! כאן הסדר משנה",
                     say="והתוצאות שונות! אחת מצביעה למעלה והשנייה למטה. "
                         "בכפל של תנועות, הסדר משנה. אז חוק החילוף לא בא בחינם.")
        self.play(FadeIn(neq, scale=2))
        self.wait_speech()
        self.play(*map(FadeOut, [panels, f1, f2, h1, h2, neq]))

        q1 = Tex(R"i^2 = j^2 = k^2 = ijk = -1", font_size=60).shift(1.5 * UP)
        q2 = VGroup(Tex(R"ij = k", font_size=72).set_color(GREEN_B), Tex(R"ji = -k", font_size=72).set_color(RED_B))
        q2.arrange(RIGHT, buff=1.5).shift(0.1 * DOWN)
        yr = heb("דבלין, 1843", 34, GREY_A).shift(1.4 * DOWN)
        self.caption("המילטון והקווטרניונים",
                     say="ב-1843 המתמטיקאי האירי ויליאם המילטון גילה מספרים חדשים, הקווטרניונים. "
                         "בהם, איי כפול ג'יי שווה קיי, אבל ג'יי כפול איי שווה מינוס קיי. "
                         "הוא כל כך התרגש, שחרט את הנוסחה על אבני גשר בדבלין.")
        self.play(Write(q1), run_time=2)
        self.play(FadeIn(q2, shift=0.2 * UP), FadeIn(yr))
        self.wait_speech()
        self.caption("והם מסובבים היום חלליות ודמויות במשחקי מחשב",
                     say="ולא סתם: היום משתמשים בקווטרניונים כדי לסובב חלליות, רובוטים ודמויות במשחקי מחשב. "
                         "כי סיבובים, כמו שראינו, לא מתחלפים.")
        self.wait_speech()
        self.fade_all()


class Counting(MulScene):
    def construct(self):
        rng = np.random.default_rng(3)
        pts = []
        while len(pts) < 12:
            p = np.array([rng.uniform(-4.5, 4.5), rng.uniform(-1.6, 2.6), 0])
            if all(np.linalg.norm(p - q) > 1.0 for q in pts):
                pts.append(p)
        stars = VGroup(*(star(0.25).set_fill(YELLOW, 1).set_stroke(width=0).move_to(p) for p in pts))
        self.caption("אז מה הסוד?",
                     say="אז מה הסוד שעומד מאחורי כל ההסברים? הנה שנים עשר כוכבים.")
        self.play(LaggedStartMap(FadeIn, stars, scale=0.3, lag_ratio=0.1))
        self.wait_speech()
        self.caption("סופרים בסדר אחד, ובסדר אחר",
                     say="אפשר לספור אותם משמאל לימין, או מלמעלה למטה, או בכל סדר שרוצים. "
                         "תמיד יוצא שנים עשר.")
        for order in (np.argsort([p[0] for p in pts]), np.argsort([-p[1] for p in pts]), rng.permutation(12)):
            nums = VGroup()
            for k, idx in enumerate(order):
                nums.add(Integer(k + 1, font_size=28, color=WHITE).next_to(stars[idx], UP, buff=0.08))
            self.play(LaggedStartMap(FadeIn, nums, lag_ratio=0.15), run_time=2)
            self.play(FadeOut(nums), run_time=0.5)
        self.wait_speech()
        self.caption("מספר הדברים לא תלוי בסדר שבו סופרים אותם",
                     say="זה הסוד: מספר הדברים לא תלוי בסדר שבו סופרים אותם. "
                         "סיבוב המלבן, השורות והעמודות, הסוכריות והבגדים, "
                         "כולם סופרים את אותם דברים פעמיים, בשני סדרים שונים.")
        self.wait_speech()
        self.play(FadeOut(stars))

        peano = VGroup(Tex(R"a \cdot 0 = 0", font_size=56),
                       Tex(R"a \cdot (b+1) = a \cdot b + a", font_size=56)).arrange(DOWN, buff=0.4).shift(0.8 * UP)
        self.caption("ובספרי המתמטיקה: אקסיומות פאנו",
                     say="ולמבוגרים: בספרי מתמטיקה מגדירים כפל בעזרת אקסיומות פאנו, בשתי שורות כאלה. "
                         "בהגדרה הזו אַ ובֵּ ממלאים תפקידים שונים לגמרי, ממש כמו השקיות והתפוחים, "
                         "וכדי להוכיח שמותר להחליף ביניהם צריך אינדוקציה כפולה. "
                         "אבל הרעיון שבלב ההוכחה הוא אותו רעיון: אותה כמות, שנספרת בשתי דרכים.")
        self.play(Write(peano), run_time=2)
        self.wait_speech()
        self.fade_all()


class Summary(MulScene):
    def construct(self):
        def panel(mob, text):
            box = RoundedRectangle(width=5.4, height=2.7, corner_radius=0.2).set_stroke(GREY_B, 2)
            mob.move_to(box.get_center() + 0.25 * UP)
            lab = heb(text, 28).next_to(box.get_bottom(), UP, buff=0.18)
            return VGroup(box, mob, lab)

        arr = grid(3, 5, lambda: dot(TEAL, 0.11), gap=0.32)
        arr2 = grid(5, 3, lambda: dot(TEAL, 0.11), gap=0.32).scale(0.9)
        p1 = panel(VGroup(arr, Tex(R"\to", font_size=40), arr2).arrange(RIGHT, buff=0.3), "מסובבים את המלבן")
        g = grid(3, 5, lambda: dot(GREY_B, 0.11), gap=0.32)
        rows_box = VGroup(*(SurroundingRectangle(r, buff=0.04).set_stroke(A_COLOR, 2) for r in g))
        cols_box = VGroup(*(SurroundingRectangle(VGroup(*(r[j] for r in g)), buff=0.04).set_stroke(B_COLOR, 2)
                            for j in range(5)))
        p2 = panel(VGroup(g, rows_box, cols_box), "סופרים שורות או עמודות")
        cands = VGroup(*(VGroup(*(candy(ROUNDS[r], 0.1) for r in range(5))).arrange(UP, buff=0.05) for _ in range(3)))
        cands.arrange(RIGHT, buff=0.5)
        p3 = panel(cands, "מחלקים בסיבובים")
        outfit = VGroup(*(VGroup(shirt(c, 0.25), pants(d, 0.25)).arrange(DOWN, buff=0.02)
                          for c, d in [(RED_C, BLUE_D), (YELLOW_D, GREY_B), (PURPLE_B, TEAL_D)])).arrange(RIGHT, buff=0.3)
        p4 = panel(outfit, "סופרים זוגות")
        panels = VGroup(p1, p2, p3, p4).arrange_in_grid(2, 2, buff=0.35).shift(0.25 * UP)
        self.caption("ארבע דרכים לראות את אותו דבר",
                     say="אז בואו נסכם. שלוש כפול חמש וחמש כפול שלוש הם שני סיפורים שונים, "
                         "ובכל זאת הם תמיד שווים. ראינו ארבע דרכים להבין למה: "
                         "לסובב את המלבן, לספור לפי שורות או לפי עמודות, לחלק בסיבובים, ולספור זוגות.")
        self.play(LaggedStartMap(FadeIn, panels, shift=0.2 * UP, lag_ratio=0.5), run_time=4)
        self.wait_speech()
        self.caption("בפעם הבאה שתגידו שלוש כפול חמש, תדעו למה",
                     say="ובכולן, הסוד הוא אחד: אותם דברים, שנספרים בשני סדרים שונים. "
                         "אז בפעם הבאה שתגידו שלוש כפול חמש שווה חמש כפול שלוש, תדעו גם למה.")
        self.wait_speech()
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None
        end = Tex(R"a \times b = b \times a", font_size=110, t2c={"a": A_COLOR, "b": B_COLOR})
        thanks = heb("תודה שצפיתם", 44, GREY_A).next_to(end, DOWN, buff=0.7)
        self.say("תודה שצפיתם.")
        self.play(Write(end), run_time=2)
        self.play(FadeIn(thanks))
        self.wait(2.5)
        self.fade_all(run_time=1.5)


SCENES = [Title, TwoStories, Addition, Roles, OrderMatters, Checking, Array, RowsColumns, AnyNumbers, Chocolate,
          Dealing, Outfits, TimesTable, NotAlways, Counting, Summary]
