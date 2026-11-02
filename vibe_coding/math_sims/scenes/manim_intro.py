"""מבוא ל-ManimGL: מי זה גרנט סנדרסון, איך נולדה הספרייה, ומושגי היסוד שלה.

Scene, Mobject, Animation ו-Updater, עם קוד אמיתי על המסך והתוצאה שלו לידו.
קריינות: Azure, הקול he-IL-Hila:DragonLatestNeural (ראו use_azure ב-common.py).
רינדור: ./render.sh --hd scenes/manim_intro.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403
import common  # noqa: E402

common.use_azure()

MONO = "DejaVu Sans Mono"
CODE_BG = "#1c1f27"
KEYWORDS = {"class": PURPLE_B, "def": PURPLE_B, "from": PURPLE_B, "import": PURPLE_B, "self": ORANGE,
            "lambda": PURPLE_B, "return": PURPLE_B}
BLUE_EYE, BROWN_EYE = "#3b8fd9", "#8b5a2b"


class IntroScene(HebrewScene):
    def caption(self, text, say=None, size=32, wait=0.3):
        super().caption(text, say=say, size=size, wait=wait)


# ---------- עזרים ----------

def code_block(lines, size=24, width=None):
    """שורות קוד בגופן קבוע, שורה כ-Text נפרד (כדי שאפשר יהיה להדגיש שורה), עם ההזחה המקורית."""
    cw = Text("M" * 20, font=MONO, font_size=size).get_width() / 20
    lh = size / 48 * 0.72
    rows = VGroup()
    for i, line in enumerate(lines):
        body = line.lstrip()
        t = Text(body or ".", font=MONO, font_size=size, t2c=KEYWORDS)
        if not body:
            t.set_opacity(0)
        t.move_to([0, -i * lh, 0], LEFT).shift((len(line) - len(body)) * cw * RIGHT)
        rows.add(t)
    w = width or rows.get_width() + 0.6
    box = Rectangle(width=w, height=rows.get_height() + 0.5).set_fill(CODE_BG, 1).set_stroke(GREY_D, 2)
    box.move_to(rows.get_center()).align_to(rows, LEFT).shift(0.3 * LEFT)
    return VGroup(box, rows)


def highlight(block, i, color=YELLOW):
    box, rows = block
    r = rows[i]
    return Rectangle(width=box.get_width() - 0.1, height=r.get_height() + 0.12).set_fill(color, 0.15) \
        .set_stroke(width=0).move_to(r).set_x(box.get_x())


def screen(width=6.2):
    """מסגרת שמייצגת את מסך הסרטון (יחס 16:9) כדי להראות את התוצאה של הקוד."""
    return Rectangle(width=width, height=width * 9 / 16).set_stroke(GREY_B, 2).set_fill(BLACK, 0.35)


def eye(r=1.4):
    """הלוגו: עין עם הטרוכרומיה סקטוריאלית, שלושה רבעים כחולים ורבע חום."""
    blue = AnnularSector(inner_radius=r * 0.33, outer_radius=r, angle=1.5 * PI, start_angle=PI / 2)
    blue.set_fill(BLUE_EYE, 1).set_stroke(width=0)
    brown = AnnularSector(inner_radius=r * 0.33, outer_radius=r, angle=0.5 * PI, start_angle=0)
    brown.set_fill(BROWN_EYE, 1).set_stroke(width=0)
    pupil = Circle(radius=r * 0.33).set_fill(BLACK, 1).set_stroke(width=0)
    shine = Circle(radius=r * 0.08).set_fill(WHITE, 0.9).set_stroke(width=0).move_to(r * 0.15 * (UP + LEFT))
    return VGroup(blue, brown, pupil, shine)


def pi_creature(color=BLUE_EYE, h=2.0):
    body = Tex(R"\pi", font_size=int(h * 140)).set_color(color)
    eyes = VGroup(*(VGroup(Circle(radius=h * 0.09).set_fill(WHITE, 1).set_stroke(BLACK, 1),
                           Dot(radius=h * 0.04, fill_color=BLACK).shift(h * 0.02 * RIGHT)) for _ in range(2)))
    eyes.arrange(RIGHT, buff=h * 0.04).move_to(body.get_top() + h * 0.02 * DOWN)
    return VGroup(body, eyes)


# ---------- סצנות ----------

class Title(IntroScene):
    def construct(self):
        eq = Tex(R"e^{i\pi} + 1 = 0", font_size=90).shift(1.6 * UP)
        sq = Square(1.6).set_fill(BLUE, 0.6).set_stroke(BLUE_A, 3).shift(3.5 * LEFT + 1.0 * DOWN)
        ax = Axes((-3, 3), (-1.5, 1.5), width=5, height=2.4).shift(3 * RIGHT + 1.0 * DOWN)
        graph = ax.get_graph(np.sin, color=YELLOW)
        self.say("כל מה שראיתם בסרטונים שלנו, הנוסחאות שנכתבות, הצורות שמשתנות והגרפים שמצטיירים, "
                 "נוצר בספרייה אחת של פייתון.")
        self.play(Write(eq), run_time=2)
        self.play(DrawBorderThenFill(sq), ShowCreation(ax))
        self.play(Transform(sq, Circle(radius=0.9).set_fill(GREEN, 0.6).set_stroke(GREEN_A, 3).move_to(sq)),
                  ShowCreation(graph), run_time=2)
        self.wait_speech()
        self.play(*map(FadeOut, [eq, sq, ax, graph]))
        title = Text("ManimGL", font=FONT, font_size=110, weight=BOLD).shift(0.6 * UP)
        title.set_color_by_gradient(BLUE_EYE, TEAL, BROWN_EYE)
        sub = heb("הספרייה שמאחורי הסרטונים של 3Blue1Brown", 38, GREY_A).next_to(title, DOWN, buff=0.5)
        self.say("קוראים לה מָאנִים ג'י אֶל. היא נכתבה במקור כדי ליצור את הסרטונים של ערוץ היוטיוב ת'רי בלו וָואן בראון. "
                 "בסרטון הזה נכיר את מי שכתב אותה, את הסיפור שלה, ואת הרעיונות שעליהם היא בנויה.")
        self.play(Write(title), run_time=2)
        self.play(FadeIn(sub, shift=0.2 * UP))
        self.wait(1)
        self.fade_all()


class Grant(IntroScene):
    def construct(self):
        e = eye(1.5).shift(3.6 * LEFT + 0.6 * UP)
        name = heb("גרנט סנדרסון", 54).shift(2.6 * RIGHT + 2.4 * UP)
        facts = VGroup(*(heb(t, 30, GREY_A) for t in [
            "תואר במתמטיקה, סטנפורד, 2015",
            "יצירת תוכן ב-Khan Academy",
            "פותח את הערוץ 3Blue1Brown ב-2015",
        ])).arrange(DOWN, buff=0.3, aligned_edge=RIGHT).next_to(name, DOWN, buff=0.5).align_to(name, RIGHT)
        self.caption("גרנט סנדרסון",
                     say="קודם כל, האיש. גרנט סנדרסון למד מתמטיקה באוניברסיטת סטנפורד, וסיים ב-2015. "
                         "אחר כך עבד קצת בקאן אקדמי, ובמקביל פתח ערוץ יוטיוב משלו.")
        self.play(Write(name))
        self.play(LaggedStartMap(FadeIn, facts, shift=0.2 * LEFT, lag_ratio=0.4), run_time=2.5)
        self.wait_speech()
        self.caption("שם הערוץ בא מהעין הימנית שלו",
                     say="ולמה ת'רי בלו וָואן בראון, שלושה כחול ואחד חום? בגלל העין הימנית שלו. "
                         "יש לו הטרוכרומיה: שלושה רבעים מהקשתית כחולים, ורבע אחד חום. "
                         "הלוגו של הערוץ הוא ציור של העין הזו.")
        self.play(FadeIn(e[0], scale=0.6), FadeIn(e[2:]), run_time=1.2)
        self.play(FadeIn(e[1], scale=1.3), run_time=1)
        q = VGroup(heb("3 כחול", 30, BLUE_EYE), heb("1 חום", 30, "#c08a5a")).arrange(DOWN, buff=0.2)
        q.next_to(e, DOWN, buff=0.4)
        self.play(FadeIn(q))
        self.wait_speech()

        pc = pi_creature(h=1.6).move_to(e)
        series = VGroup(*(heb(t, 28) for t in [
            "המהות של האלגברה הלינארית",
            "המהות של החשבון האינפיניטסימלי",
            "רשתות נוירונים",
        ])).arrange(DOWN, buff=0.28, aligned_edge=RIGHT).move_to(facts, UR)
        self.caption("סדרות שמיליוני אנשים ראו",
                     say="הערוץ התפרסם בזכות סדרות כמו המהות של האלגברה הלינארית, המהות של החשבון האינפיניטסימלי, "
                         "ורשתות נוירונים. מיליוני אנשים מנויים עליו. "
                         "והדמויות הקטנות שמופיעות בסרטונים שלו הן יצורי פאי, האות פאי עם עיניים.")
        self.play(FadeOut(facts), FadeOut(q))
        self.play(LaggedStartMap(FadeIn, series, shift=0.2 * LEFT, lag_ratio=0.4))
        self.play(ReplacementTransform(e, pc), run_time=1.5)
        self.play(pc[1].animate.stretch(0.1, 1), run_time=0.15)
        self.play(pc[1].animate.stretch(10, 1), run_time=0.15)
        self.wait_speech()
        self.caption("הרעיון: לראות את המתמטיקה",
                     say="הרעיון שמאחורי כל הסרטונים שלו פשוט: מתמטיקה צריך לראות, לא רק לקרוא. "
                         "ובשביל זה הוא היה צריך כלי שיודע להזיז נוסחאות וצורות בדיוק כמו שהוא רוצה.")
        self.play(WiggleOutThenIn(pc))
        self.wait_speech()
        self.fade_all()


class History(IntroScene):
    def construct(self):
        line = Line(6 * LEFT, 6 * RIGHT).set_stroke(GREY_B, 3).shift(1.6 * UP)
        years = [2015, 2020, 2021, 2026]
        xs = [-4.5, -0.6, 2.4, 5.2]
        ticks = VGroup(*(VGroup(Dot(line.get_left() * 0 + [x, 1.6, 0], radius=0.09),
                                Text(str(y), font=FONT, font_size=30).move_to([x, 2.15, 0])) for x, y in zip(xs, years)))
        self.caption("2015: כלי פרטי של אדם אחד",
                     say="והספרייה עצמה. גרנט התחיל לכתוב אותה ב-2015, בשביל הסרטונים הראשונים שלו. "
                         "השם מָאנִים בא מ-Math Animation, אנימציה מתמטית.")
        self.play(ShowCreation(line), FadeIn(ticks[0]))
        word = Text("Math ANIMation", font=FONT, font_size=44).shift(0.6 * UP)
        word[0].set_color(BLUE), word[4:7].set_color(BLUE)
        manim = Text("manim", font=MONO, font_size=60).set_color(BLUE).next_to(word, DOWN, buff=0.7)
        self.play(Write(word))
        self.play(TransformFromCopy(VGroup(word[0], word[4:7]), manim))
        self.wait_speech()
        self.caption("קוד פתוח, אבל בלי הבטחות",
                     say="זה היה כלי פרטי: גרנט שינה אותו כל פעם שסרטון חדש היה צריך משהו חדש, בלי תיעוד ובלי לדאוג למשתמשים אחרים. "
                         "הוא פרסם את הקוד בגיטהאב, ואנשים רבים התאהבו בסגנון ורצו להשתמש בו בעצמם.")
        stars = VGroup(*(Dot(radius=0.05, fill_color=YELLOW).move_to(
            manim.get_center() + np.array([1.5 * np.cos(t), 0.55 * np.sin(t), 0])) for t in np.linspace(0, TAU, 24)))
        self.play(LaggedStartMap(FadeIn, stars, scale=3, lag_ratio=0.05), run_time=2)
        self.wait_speech()
        self.play(FadeOut(stars), FadeOut(word))

        base = manim.copy()
        ce = VGroup(Text("Manim Community", font=FONT, font_size=34).set_color(GREEN_B),
                    Text("pip install manim", font=MONO, font_size=24).set_color(GREY_A)).arrange(DOWN, buff=0.15)
        gl = VGroup(Text("ManimGL", font=FONT, font_size=34).set_color(BLUE_EYE),
                    Text("pip install manimgl", font=MONO, font_size=24).set_color(GREY_A)).arrange(DOWN, buff=0.15)
        ce.move_to([-3.2, -2.0, 0])
        gl.move_to([3.2, -2.0, 0])
        a1 = Arrow(manim.get_bottom(), ce.get_top(), buff=0.2).set_color(GREEN_B)
        a2 = Arrow(manim.get_bottom(), gl.get_top(), buff=0.2).set_color(BLUE_EYE)
        self.caption("2020: הקהילה מפצלת את הספרייה",
                     say="ב-2020 קבוצת מפתחים החליטה לקחת את הקוד ולפצל אותו לגרסה של הקהילה. "
                         "המטרה: גרסה יציבה, בדוקה, עם תיעוד מסודר, שקל להתחיל איתה.")
        self.play(FadeIn(ticks[1]), base.animate.shift(0.3 * UP))
        self.play(GrowArrow(a1), FadeIn(ce, shift=0.2 * DOWN))
        self.wait_speech()
        self.caption("והגרסה של גרנט עוברת ל-OpenGL",
                     say="גרנט המשיך לפתח את הגרסה שלו, והעביר אותה לרנדור בכרטיס הגרפי, באוֹפֶּן ג'י אֶל. "
                         "מכאן השם מָאנִים ג'י אֶל. היא מהירה, ויודעת גם לפתוח חלון אינטראקטיבי.")
        self.play(FadeIn(ticks[2]), GrowArrow(a2), FadeIn(gl, shift=0.2 * DOWN))
        self.wait_speech()
        self.caption("היום: שתי ספריות קרובות, עם שמות דומים",
                     say="היום יש שתי ספריות קרובות, עם הרבה קוד משותף אבל לא זהות. "
                         "צריך לשים לב איזו מהן מתקינים ולאיזו מתאים המדריך שקוראים. "
                         "אנחנו משתמשים במָאנִים ג'י אֶל, הגרסה שגרנט עצמו מייצר בה את הסרטונים שלו.")
        self.play(FadeIn(ticks[3]), Indicate(gl, scale_factor=1.15), run_time=1.5)
        self.wait_speech()
        self.fade_all()


class BigIdea(IntroScene):
    def construct(self):
        stage = RoundedRectangle(width=7.5, height=3.6, corner_radius=0.15).set_stroke(GREY_B, 3).shift(0.5 * UP)
        curtain_l = Rectangle(width=0.5, height=3.6).set_fill(MAROON_D, 1).set_stroke(width=0).next_to(stage, LEFT, buff=0)
        curtain_r = curtain_l.copy().next_to(stage, RIGHT, buff=0)
        actors = VGroup(Circle(radius=0.55).set_fill(BLUE, 0.7).set_stroke(BLUE_A, 3),
                        Square(1.0).set_fill(GREEN, 0.7).set_stroke(GREEN_A, 3),
                        Tex(R"x^2", font_size=80)).arrange(RIGHT, buff=1.0).move_to(stage)
        lab1 = VGroup(Text("Scene", font=MONO, font_size=34).set_color(YELLOW), heb("הבמה", 30))
        lab2 = VGroup(Text("Mobject", font=MONO, font_size=34).set_color(YELLOW), heb("השחקנים", 30))
        lab3 = VGroup(Text("Animation", font=MONO, font_size=34).set_color(YELLOW), heb("מה שהם עושים", 30))
        for l in (lab1, lab2, lab3):
            l.arrange(DOWN, buff=0.15)
        labs = VGroup(lab3, lab2, lab1).arrange(RIGHT, buff=1.6).shift(2.3 * DOWN)
        self.caption("שלושה מושגים, כמו בתיאטרון",
                     say="לפני שנראה קוד, הנה התמונה הגדולה. מָאנִים בנויה כמו תיאטרון, ויש בה שלושה מושגי יסוד.")
        self.play(ShowCreation(stage), FadeIn(curtain_l), FadeIn(curtain_r))
        self.wait_speech()
        self.caption("סִין: הבמה", say="הראשון הוא הסִין, הבמה. כל סרטון, או חלק של סרטון, הוא מחלקה של פייתון שיורשת מ-Scene.")
        self.play(FadeIn(lab1, shift=0.2 * UP), Indicate(stage, color=YELLOW))
        self.wait_speech()
        self.caption("מוֹבְּגֶ'קְט: השחקנים",
                     say="השני הוא המוֹבְּגֶ'קְט, קיצור של אובייקט מתמטי. אלה השחקנים: עיגול, ריבוע, נוסחה, גרף. "
                         "כל דבר שרואים על המסך הוא מוֹבְּגֶ'קְט.")
        self.play(FadeIn(lab2, shift=0.2 * UP), LaggedStartMap(FadeIn, actors, scale=0.5, lag_ratio=0.3))
        self.wait_speech()
        self.caption("אנימציה: מה שהשחקנים עושים",
                     say="והשלישי הוא האנימציה: מה שהשחקנים עושים על הבמה. "
                         "נכנסים, זזים, משנים צורה, נעלמים.")
        self.play(FadeIn(lab3, shift=0.2 * UP))
        self.play(actors[0].animate.shift(0.6 * UP), Rotate(actors[1], PI / 2),
                  actors[2].animate.scale(1.3).set_color(ORANGE), run_time=1.5)
        self.play(Transform(actors[1], Circle(radius=0.55).set_fill(GREEN, 0.7).set_stroke(GREEN_A, 3).move_to(actors[1])),
                  actors[0].animate.shift(0.6 * DOWN), run_time=1.5)
        self.wait_speech()
        self.caption("כל השאר הוא פרטים",
                     say="כמעט כל מה שיש בספרייה הוא סוג של אחד משלושת אלה. מי שמבין אותם, מבין איך הכל בנוי.")
        self.wait_speech()
        self.fade_all()


class Hello(IntroScene):
    def construct(self):
        src = ["from manimlib import *",
               "",
               "class Hello(Scene):",
               "    def construct(self):",
               "        circle = Circle(radius=1.5)",
               "        circle.set_fill(BLUE, 0.5)",
               "        self.play(ShowCreation(circle))",
               "        self.wait()"]
        code = code_block(src, 34).to_edge(LEFT, buff=0.3).shift(0.95 * UP)
        scr = screen(4.6).to_edge(RIGHT, buff=0.3).shift(1.6 * UP)
        k = scr.get_width() / FRAME_WIDTH
        circ = Circle(radius=1.5 * k).set_stroke(RED, 4 * k + 1).move_to(scr)
        self.caption("התוכנית הראשונה", say="הנה תוכנית שלמה במָאנִים ג'י אֶל. שמונה שורות.")
        self.play(FadeIn(code[0]), LaggedStartMap(FadeIn, code[1], lag_ratio=0.1), FadeIn(scr))
        self.wait_speech()
        hl = highlight(code, 0)
        self.caption("מייבאים את הספרייה", say="השורה הראשונה מייבאת את כל הספרייה. השם הפנימי שלה הוא מָאנִים-לִיבּ.")
        self.play(FadeIn(hl))
        self.wait_speech()
        self.caption("סצנה היא מחלקה, והמתודה construct היא התסריט",
                     say="אחר כך מגדירים סצנה: מחלקה שיורשת מ-Scene. בתוכה יש מתודה אחת חשובה, construct. "
                         "זה התסריט: מה שכתוב בה קורה בסרטון, לפי הסדר.")
        self.play(Transform(hl, highlight(code, 2)))
        self.play(Transform(hl, highlight(code, 3)))
        self.wait_speech()
        self.caption("יוצרים מובג'קט", say="עכשיו יוצרים מוֹבְּגֶ'קְט: עיגול ברדיוס אחד וחצי. ונותנים לו מילוי כחול, בשקיפות של חצי.")
        self.play(Transform(hl, highlight(code, 4)))
        self.wait(1.5)
        self.play(Transform(hl, highlight(code, 5)))
        self.wait_speech()
        self.caption("ואז: הצג אנימציה",
                     say="ואז השורה החשובה: סֶלְף פְּלֵיי, הצג אנימציה. האנימציה כאן היא ציור של העיגול, קו אחרי קו. "
                         "ובסוף, המתנה של שנייה.")
        self.play(Transform(hl, highlight(code, 6)))
        circ.set_fill(BLUE, 0.5)
        self.play(ShowCreation(circ), run_time=1.5)
        self.play(Transform(hl, highlight(code, 7)))
        self.wait_speech()
        self.play(FadeOut(hl))

        cmd1 = Text("manimgl hello.py Hello", font=MONO, font_size=36).set_color(GREEN_B)
        cmd2 = Text("manimgl hello.py Hello -w", font=MONO, font_size=36).set_color(GREEN_B)
        cmds = VGroup(cmd1, cmd2).arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        term = SurroundingRectangle(cmds, buff=0.3).set_fill(BLACK, 1).set_stroke(GREY_D, 2)
        prompt = VGroup(term, cmds).move_to(0.8 * UP + 1.0 * LEFT)
        n1 = heb("פותח חלון", 30, GREY_A).next_to(term, RIGHT, buff=0.3).match_y(cmd1)
        n2 = heb("כותב קובץ וידאו", 30, GREY_A).next_to(term, RIGHT, buff=0.3).match_y(cmd2)
        self.caption("מריצים מהטרמינל",
                     say="מריצים מהטרמינל: מָאנִים ג'י אֶל, שם הקובץ, ושם הסצנה. "
                         "בלי דגלים נפתח חלון, ואפשר לראות את הסצנה ואפילו לשחק איתה. "
                         "עם מינוס w, הספרייה כותבת קובץ וידאו.")
        self.play(FadeOut(code), FadeOut(scr), FadeOut(circ))
        self.play(FadeIn(term), Write(cmd1), FadeIn(n1))
        self.play(Write(cmd2), FadeIn(n2))
        self.wait_speech()
        self.fade_all()


class Coordinates(IntroScene):
    def construct(self):
        plane = NumberPlane((-7, 7), (-4, 4), background_line_style=dict(stroke_color=BLUE_E, stroke_width=1))
        plane.axes.set_stroke(GREY_B, 2)
        self.caption("המסך הוא מישור", say="איפה דברים נמצאים? המסך הוא מישור עם צירים. הראשית באמצע.")
        self.play(ShowCreation(plane, lag_ratio=0.05), run_time=2)
        o = Dot(ORIGIN, color=YELLOW)
        self.play(FadeIn(o, scale=3))
        w = Arrow(FRAME_WIDTH / 2 * LEFT, FRAME_WIDTH / 2 * RIGHT, buff=0).shift(3.1 * UP).set_color(YELLOW)
        h = Arrow(FRAME_HEIGHT / 2 * DOWN, FRAME_HEIGHT / 2 * UP, buff=0).shift(6.4 * LEFT).set_color(YELLOW)
        wl = Tex(R"\approx 14.2", font_size=40).next_to(w, DOWN, buff=0.1).shift(2 * RIGHT)
        hl = Tex("8", font_size=40).next_to(h, RIGHT, buff=0.15).shift(1.5 * UP)
        self.wait_speech()
        self.caption("גובה 8 יחידות, רוחב בערך 14",
                     say="גובה המסך הוא שמונה יחידות, והרוחב בערך ארבע עשרה. "
                         "כך שעיגול ברדיוס אחד וחצי ממלא בערך שליש מהגובה.")
        self.play(GrowFromCenter(w), GrowFromCenter(h), FadeIn(wl), FadeIn(hl))
        self.wait_speech()
        self.play(*map(FadeOut, [w, h, wl, hl]))

        arrows = VGroup(*(Arrow(ORIGIN, d, buff=0).set_color(c) for d, c in [(UP, GREEN), (RIGHT, RED), (DOWN, GREEN), (LEFT, RED)]))
        names = VGroup(*(Text(n, font=MONO, font_size=28).set_color(c).next_to(a.get_end(), d, buff=0.12)
                         for n, a, d, c in zip(["UP", "RIGHT", "DOWN", "LEFT"], arrows, [UP, RIGHT, DOWN, LEFT],
                                                [GREEN, RED, GREEN, RED])))
        self.caption("כיוונים הם וקטורים עם שמות",
                     say="לכיוונים יש שמות: אַפּ, רַייט, דָאוּן ולֶפְט. כל אחד מהם הוא וקטור באורך אחת, "
                         "ואפשר לחבר אותם ולכפול אותם במספרים, כמו וקטורים רגילים.")
        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.2), LaggedStartMap(FadeIn, names, lag_ratio=0.2))
        self.wait_speech()
        sq = Square(1).set_fill(TEAL, 0.7).set_stroke(TEAL_A, 3)
        cmd = Text("square.move_to(3*RIGHT + 2*UP)", font=MONO, font_size=38)
        cmd_bg = SurroundingRectangle(cmd, buff=0.15).set_fill(CODE_BG, 1).set_stroke(GREY_D, 1)
        cmdg = VGroup(cmd_bg, cmd).to_corner(UL, buff=0.3)
        self.caption("שולחים מובג'קט לנקודה", say="למשל, שלוש פעמים ימינה ועוד פעמיים למעלה, זו הנקודה שלוש, שתיים. "
                                                  "מוֹבְּגֶ'קְט שולחים לשם עם move_to.")
        self.play(FadeOut(arrows), FadeOut(names), FadeIn(sq), FadeIn(cmdg))
        self.play(sq.animate.move_to(3 * RIGHT + 2 * UP), run_time=1.5)
        self.wait_speech()

        c = Circle(radius=0.5).set_fill(ORANGE, 0.7).set_stroke(ORANGE, 3)
        cmd2 = Text("circle.next_to(square, DOWN)", font=MONO, font_size=38)
        cmd2g = VGroup(SurroundingRectangle(cmd2, buff=0.15).set_fill(CODE_BG, 1).set_stroke(GREY_D, 1), cmd2)
        cmd2g.move_to(cmdg, LEFT)
        self.caption("או ממקמים ביחס לאחרים",
                     say="אבל לרוב לא חושבים במספרים. ממקמים דברים ביחס לדברים אחרים: "
                         "נקסט טוּ, ליד. מתחת לריבוע.")
        self.play(FadeOut(cmdg), FadeIn(cmd2g), FadeIn(c))
        self.play(c.animate.next_to(sq, DOWN), run_time=1.5)
        self.wait_speech()
        row = VGroup(*(Circle(radius=0.35).set_fill(col, 0.8).set_stroke(WHITE, 1)
                       for col in [RED, ORANGE, YELLOW, GREEN, BLUE])).move_to(2.5 * LEFT + 1.5 * DOWN)
        for i, m in enumerate(row):
            m.shift([0.4 * np.sin(3 * i), 0.6 * np.cos(2 * i), 0])
        cmd3 = Text("group.arrange(RIGHT, buff=0.3)", font=MONO, font_size=38)
        cmd3g = VGroup(SurroundingRectangle(cmd3, buff=0.15).set_fill(CODE_BG, 1).set_stroke(GREY_D, 1), cmd3)
        cmd3g.move_to(cmdg, LEFT)
        self.caption("או מסדרים קבוצה בשורה",
                     say="או מסדרים כמה דברים בשורה, עם מרווח קבוע, בפקודה אחת: אָרֵינְג'. "
                         "יש עוד כמה כאלה: to_edge, לקצה המסך, align_to, ליישור. ומהן בונים כל פריסה.")
        self.play(FadeOut(cmd2g), FadeIn(cmd3g), LaggedStartMap(FadeIn, row, scale=0.5, lag_ratio=0.1))
        self.play(row.animate.arrange(RIGHT, buff=0.3).move_to(2.5 * LEFT + 1.5 * DOWN), run_time=1.5)
        self.wait_speech()
        self.fade_all()


class Points(IntroScene):
    def construct(self):
        circ = Circle(radius=2.2).set_stroke(BLUE, 5).shift(2.6 * LEFT + 0.4 * UP)
        self.caption("מה זה עיגול, בעיני הספרייה?",
                     say="עכשיו נציץ מתחת למכסה. מה זה בעצם עיגול, בעיני הספרייה?")
        self.play(ShowCreation(circ))
        self.wait_speech()
        pts = circ.get_points()
        anchors = VGroup(*(Dot(p, radius=0.07, fill_color=YELLOW) for p in pts[::2]))
        handles = VGroup(*(Dot(p, radius=0.05, fill_color=RED) for p in pts[1::2]))
        links = VGroup(*(Line(pts[i], pts[i + 1]).set_stroke(GREY_B, 1.5) for i in range(len(pts) - 1)))
        self.caption("רצף של עקומות בזייה",
                     say="התשובה: רשימה של נקודות. העיגול בנוי מעקומות קטנות שנקראות עקומות בֶּזְיֶה. "
                         "לכל עקומה יש נקודת התחלה וסוף, בצהוב, ונקודת שליטה באמצע, באדום, שמושכת אליה את הקו.")
        self.play(circ.animate.set_stroke(opacity=0.4), FadeIn(links), FadeIn(anchors, lag_ratio=0.1), FadeIn(handles, lag_ratio=0.1))
        self.wait_speech()
        n = Integer(len(pts), font_size=40).set_color(YELLOW)
        lab = VGroup(Text("circle.get_points()", font=MONO, font_size=34), Tex(R"\to", font_size=40), n)
        lab.arrange(RIGHT, buff=0.25).shift(3.4 * RIGHT + 2.5 * UP)
        self.play(FadeIn(lab))
        pi = Tex(R"\pi", font_size=300).set_fill(opacity=0).set_stroke(WHITE, 3).shift(3.4 * RIGHT + 0.0 * DOWN)
        self.caption("גם אותיות ונוסחאות הן עקומות",
                     say="וגם אותיות ונוסחאות. מָאנִים מריצה לָטֶך, הופכת את התוצאה לציור וקטורי, "
                         "וכל אות הופכת לעוד כמה עקומות בֶּזְיֶה. בגלל זה אפשר להזיז, לצבוע ולעוות כל אות בנפרד.")
        self.play(Write(pi), run_time=2)
        pp = pi[0].get_points()
        pdots = VGroup(*(Dot(p, radius=0.04, fill_color=YELLOW) for p in pp[::2]))
        self.play(FadeIn(pdots, lag_ratio=0.05), run_time=1.5)
        self.wait_speech()
        self.caption("הכל נקודות. וזה מה שמאפשר לצורה להפוך לצורה",
                     say="אז הכל, בסופו של דבר, נקודות. וזה הסוד של האנימציות: כדי להפוך עיגול לאות פאי, "
                         "מספיק להזיז כל נקודה של העיגול אל נקודה של הפאי.")
        self.play(FadeOut(anchors), FadeOut(handles), FadeOut(links), FadeOut(pdots), FadeOut(lab))
        circ.set_stroke(opacity=1)
        self.play(Transform(circ, pi.copy().set_stroke(BLUE, 5).move_to(circ)), run_time=2.5)
        self.wait_speech()
        self.fade_all()


class Groups(IntroScene):
    def construct(self):
        tex = Tex(R"E = mc^2", font_size=110).shift(1.5 * UP)
        self.caption("מובג'קטים בנויים ממובג'קטים",
                     say="עוד רעיון יסוד: מוֹבְּגֶ'קְטִים בנויים ממוֹבְּגֶ'קְטִים קטנים יותר. "
                         "קחו את הנוסחה E שווה m c בריבוע. היא מוֹבְּגֶ'קְט אחד, אבל בתוכה יש מוֹבְּגֶ'קְט לכל סימן.")
        self.play(Write(tex))
        self.wait_speech()
        root = Text("Tex", font=MONO, font_size=40).set_color(YELLOW).shift(0.3 * DOWN)
        kids = VGroup(*(Text(ch, font=MONO, font_size=40) for ch in ["E", "=", "m", "c", "2"])).arrange(RIGHT, buff=0.9)
        kids.shift(1.8 * DOWN)
        edges = VGroup(*(Line(root.get_bottom(), k.get_top(), buff=0.12).set_stroke(GREY_B, 2) for k in kids))
        self.caption("עץ: הורה עם ילדים", say="זה עץ: מוֹבְּגֶ'קְט הורה, וחמישה ילדים. "
                                              "לילדים ניגשים כמו לרשימה בפייתון, עם סוגריים מרובעים.")
        self.play(FadeIn(root), LaggedStartMap(ShowCreation, edges, lag_ratio=0.15),
                  LaggedStartMap(FadeIn, kids, lag_ratio=0.15))
        self.wait_speech()
        idx = Text("tex[2].set_color(RED)", font=MONO, font_size=40).to_corner(UL, buff=0.4)
        self.caption("פונים לילד אחד, וצובעים רק אותו", say="למשל, tex בסוגריים שתיים: הילד השלישי, האות m. צובעים רק אותה באדום.")
        self.play(FadeIn(idx))
        self.play(tex[2].animate.set_color(RED), kids[2].animate.set_color(RED), run_time=1.2)
        self.wait_speech()
        idx2 = Text("tex.set_color(BLUE)", font=MONO, font_size=40).move_to(idx, LEFT)
        self.caption("פקודה להורה עוברת לכל הילדים",
                     say="ופקודה שנותנים להורה עוברת לכל הילדים. צובעים את כל הנוסחה בכחול, מזיזים אותה, מסובבים אותה: "
                         "הכל קורה לכל העץ יחד.")
        self.play(FadeOut(idx), FadeIn(idx2))
        self.play(tex.animate.set_color(BLUE), kids.animate.set_color(BLUE), run_time=1.2)
        self.play(tex.animate.rotate(0.15).shift(0.3 * RIGHT), run_time=1)
        self.wait_speech()
        self.play(*map(FadeOut, [root, kids, edges, idx2, tex]))

        shapes = VGroup(Circle(radius=0.6).set_fill(RED, 0.7), Square(1.1).set_fill(GREEN, 0.7),
                        Triangle().scale(0.7).set_fill(BLUE, 0.7)).arrange(RIGHT, buff=0.8).shift(0.6 * UP)
        vg = Text("group = VGroup(circle, square, triangle)", font=MONO, font_size=38).to_corner(UL, buff=0.4)
        self.caption("וי גרופ: קבוצה שבונים בעצמנו",
                     say="ואפשר לבנות עצים כאלה בעצמנו, עם וִי גְרוּפּ: מחברים כמה מוֹבְּגֶ'קְטִים לקבוצה אחת, "
                         "ומזיזים, מסדרים או מעלימים אותם יחד.")
        self.play(FadeIn(vg), LaggedStartMap(FadeIn, shapes, scale=0.5, lag_ratio=0.2))
        box = SurroundingRectangle(shapes, buff=0.25).set_stroke(YELLOW, 2)
        self.play(ShowCreation(box))
        self.play(VGroup(shapes, box).animate.scale(0.7).shift(1.5 * DOWN), run_time=1.5)
        self.wait_speech()
        self.fade_all()


class Animations(IntroScene):
    def construct(self):
        self.caption("מה זה אנימציה?", say="ועכשיו לכוכבת של הספרייה: האנימציה. מה זה בעצם?")
        self.wait_speech()
        sq = Square(1.6).set_fill(BLUE, 0.6).set_stroke(BLUE_A, 4)
        circ = Circle(radius=0.9).set_fill(GREEN, 0.6).set_stroke(GREEN_A, 4)
        sq.align_data_and_family(circ)  # אותו מספר נקודות, כדי שאפשר יהיה לערבב
        xs = np.linspace(-5, 5, 5)
        frames = VGroup()
        for x, a in zip(xs, np.linspace(0, 1, 5)):
            m = sq.copy()
            m.interpolate(sq, circ, a)
            m.move_to([x, 1.2, 0])
            frames.add(m)
        alphas = VGroup(*(Tex(Rf"\alpha = {a:g}", font_size=34).next_to(f, DOWN, buff=0.3)
                          for f, a in zip(frames, [0, 0.25, 0.5, 0.75, 1])))
        self.caption("אנימציה היא פונקציה של אלפא, מ-0 עד 1",
                     say="אנימציה היא פונקציה של מספר אחד, אַלְפָא, שעולה מאפס לאחת. "
                         "באלפא אפס רואים את ההתחלה, באלפא אחת את הסוף, וביניהם, תערובת. "
                         "כאן, ריבוע שהופך לעיגול, בחמישה רגעים.")
        self.play(LaggedStartMap(FadeIn, frames, lag_ratio=0.3), LaggedStartMap(FadeIn, alphas, lag_ratio=0.3), run_time=3)
        self.wait_speech()
        self.caption("בכל פריים, הספרייה מחשבת את אלפא ומציירת",
                     say="כשמריצים self.play, הספרייה מחשבת שלושים או שישים פריימים בשנייה. "
                         "בכל פריים היא מחשבת כמה זמן עבר, ממירה את זה לאלפא, ומציירת את התערובת.")
        live = sq.copy().move_to(1.6 * DOWN)
        t = ValueTracker(0)
        live.add_updater(lambda m: m.become(sq.copy().interpolate(sq, circ, t.get_value()).move_to(1.6 * DOWN)))
        mark = always_redraw(lambda: Triangle().scale(0.12).set_fill(YELLOW, 1).set_stroke(width=0).rotate(PI)
                             .move_to([interpolate(-5, 5, t.get_value()), 0.05, 0]))
        self.add(live, mark)
        self.play(t.animate.set_value(1), run_time=3)
        self.play(t.animate.set_value(0), run_time=1)
        live.clear_updaters()
        self.wait_speech()
        self.play(*map(FadeOut, [frames, alphas, live, mark]))

        ax = Axes((0, 1, 0.5), (0, 1, 0.5), width=3.4, height=2.6)
        lin = ax.get_graph(lambda x: x, color=GREY_B)
        smo = ax.get_graph(smooth, color=YELLOW)
        g = VGroup(ax, lin, smo).shift(3 * LEFT + 0.4 * UP)
        l1 = Text("linear", font=MONO, font_size=24).set_color(GREY_B).next_to(ax, DOWN).shift(0.8 * LEFT)
        l2 = Text("smooth", font=MONO, font_size=24).set_color(YELLOW).next_to(ax, DOWN).shift(0.8 * RIGHT)
        d1 = Dot(color=GREY_B).move_to(1 * RIGHT + 1.3 * UP)
        d2 = Dot(color=YELLOW).move_to(1 * RIGHT + 0.3 * DOWN)
        self.caption("פונקציית קצב: איך הזמן הופך לאלפא",
                     say="ואיך זמן הופך לאלפא? דרך פונקציית קצב. בקצב לינארי, התנועה אחידה. "
                         "בקצב החלק, ברירת המחדל, התנועה מאיצה בהתחלה ומאטה בסוף. "
                         "זה אחד הסודות של המראה הרגוע של הסרטונים האלה.")
        self.play(ShowCreation(ax), ShowCreation(lin), ShowCreation(smo), FadeIn(l1), FadeIn(l2))
        self.add(d1, d2)
        self.play(d1.animate.shift(4 * RIGHT), rate_func=linear, run_time=2.5)
        self.play(d2.animate.shift(4 * RIGHT), rate_func=smooth, run_time=2.5)
        self.wait_speech()
        self.play(*map(FadeOut, [g, l1, l2, d1, d2]))

        src = ["self.play(FadeIn(square))",
               "self.play(Transform(square, circle))",
               "self.play(Write(formula), run_time=3)",
               "self.play(FadeOut(circle), FadeOut(formula))"]
        code = code_block(src, 36).move_to(0.6 * UP)
        self.caption("יש עשרות אנימציות מוכנות",
                     say="יש עשרות אנימציות מוכנות: להופיע, להפוך צורה, לכתוב נוסחה, להיעלם. "
                         "כמה אנימציות באותו self.play רצות יחד, ו-run_time קובע כמה שניות זה ייקח.")
        self.play(FadeIn(code[0]), LaggedStartMap(FadeIn, code[1], lag_ratio=0.3), run_time=2)
        self.wait_speech()
        self.fade_all()


class AnimateSyntax(IntroScene):
    def construct(self):
        src = ["self.play(",
               "    square.animate",
               "        .shift(3*RIGHT)",
               "        .rotate(PI/4)",
               "        .set_color(RED)",
               ")"]
        code = code_block(src, 40).to_edge(LEFT, buff=0.5).shift(0.6 * UP)
        sq = Square(1.4).set_fill(BLUE, 0.7).set_stroke(BLUE_A, 3).move_to(1.5 * RIGHT + 0.6 * UP)
        self.caption("הקיצור animate", say="ויש קיצור שימושי מאוד: אַנִימֵייט. "
                                           "כותבים מה רוצים לעשות למוֹבְּגֶ'קְט, ושמים לפני זה את המילה animate.")
        self.play(FadeIn(code), FadeIn(sq))
        self.wait_speech()
        ghost = sq.copy().shift(3 * RIGHT).rotate(PI / 4).set_color(RED).set_fill(opacity=0.2).set_stroke(opacity=0.6)
        ghost_l = heb("מצב היעד", 28, GREY_A).next_to(ghost, DOWN, buff=0.5)
        self.caption("הספרייה מחשבת את מצב היעד, ומחליקה אליו",
                     say="מה שקורה מאחורי הקלעים: הספרייה עושה את הפעולות על עותק, ומקבלת את מצב היעד. "
                         "ואז היא מחליקה מהמצב הנוכחי אל היעד, בדיוק כמו טרנספורם.")
        self.play(FadeIn(ghost), FadeIn(ghost_l))
        self.wait(0.5)
        self.play(sq.animate.shift(3 * RIGHT).rotate(PI / 4).set_color(RED), run_time=2.5)
        self.wait_speech()
        self.caption("בלי animate, השינוי קורה מיד",
                     say="ובלי המילה animate, אותן פקודות פשוט משנות את המוֹבְּגֶ'קְט מיד, בלי תנועה. "
                         "כך מכינים את הבמה לפני שמתחילים לשחק.")
        self.play(FadeOut(ghost), FadeOut(ghost_l))
        sq.shift(3 * LEFT).rotate(-PI / 4).set_color(BLUE)
        self.wait(0.8)
        sq.shift(3 * RIGHT).rotate(PI / 4).set_color(RED)
        self.wait_speech()
        self.fade_all()


class Updaters(IntroScene):
    def construct(self):
        ax = Axes((-1, 7), (-1.5, 1.5), width=7.5, height=3.0).shift(2.6 * RIGHT + 1.6 * UP)
        graph = ax.get_graph(np.sin, x_range=(0, 6.5), color=BLUE)
        x = ValueTracker(0.5)
        dot = always_redraw(lambda: Dot(ax.i2gp(x.get_value(), graph), color=YELLOW, radius=0.1))
        vline = always_redraw(lambda: ax.get_v_line(dot.get_center()).set_stroke(YELLOW, 2))
        num = DecimalNumber(0, num_decimal_places=2, font_size=36).set_color(YELLOW)
        num.add_updater(lambda m: m.set_value(np.sin(x.get_value())).next_to(dot, UP, buff=0.2))
        src = ["x = ValueTracker(0.5)",
               "dot = always_redraw(",
               "    lambda: Dot(ax.i2gp(",
               "        x.get_value(), graph)))",
               "",
               "self.play(x.animate.set_value(6))"]
        code = code_block(src, 30).to_corner(DL, buff=0.3).shift(0.9 * UP)
        self.caption("הרעיון הרביעי: דברים שתלויים בדברים",
                     say="ויש עוד רעיון יסוד אחד, רביעי. לפעמים מוֹבְּגֶ'קְט צריך לעקוב אחרי משהו אחר. "
                         "נקודה שרצה על גרף, קו שמחובר אליה, ומספר שמראה את הגובה שלה.")
        self.play(ShowCreation(ax), ShowCreation(graph))
        self.add(dot, vline, num)
        self.play(x.animate.set_value(3.5), run_time=3)
        self.wait_speech()
        self.caption("ואליו טרקר: מספר שאפשר להזיז",
                     say="בשביל זה יש וָולְיוּ טְרֶקֶר: מוֹבְּגֶ'קְט בלתי נראה שמחזיק מספר אחד. "
                         "ויש אַפְּדֵייטֶר: פונקציה שרצה בכל פריים ומעדכנת את המוֹבְּגֶ'קְט. "
                         "always_redraw בונה את הנקודה מחדש בכל פריים, לפי הערך הנוכחי של המספר.")
        self.play(FadeIn(code))
        self.play(FadeIn(highlight(code, 0)))
        self.wait(2)
        self.play(FadeIn(highlight(code, 1)), FadeIn(highlight(code, 2)), FadeIn(highlight(code, 3)))
        self.wait_speech()
        self.caption("מזיזים רק את המספר, וכל השאר נגרר אחריו",
                     say="ועכשיו מזיזים רק את המספר. הנקודה, הקו והמספר שעל המסך מתעדכנים לבד. "
                         "זה כמו גיליון אלקטרוני: משנים תא אחד, וכל מה שתלוי בו משתנה.")
        self.play(FadeIn(highlight(code, 5)))
        self.play(x.animate.set_value(6.2), run_time=3)
        self.play(x.animate.set_value(1.2), run_time=3)
        self.wait_speech()
        self.fade_all()


class BehindThisVideo(IntroScene):
    def construct(self):
        steps = ["script.py", "construct()", "OpenGL", "ffmpeg", "video.mp4"]
        notes = ["תסריט", "הסצנה רצה", "פריים אחר פריים", "מחבר לסרטון", "התוצאה"]
        boxes = VGroup()
        for s, n in zip(steps, notes):
            t = Text(s, font=MONO, font_size=24)
            b = RoundedRectangle(width=2.2, height=0.9, corner_radius=0.12).set_stroke(BLUE_B, 2).set_fill(BLUE_E, 0.4)
            t.move_to(b)
            lab = heb(n, 22, GREY_A).next_to(b, DOWN, buff=0.15)
            boxes.add(VGroup(b, t, lab))
        boxes.arrange(RIGHT, buff=0.45).shift(1.8 * UP)
        arrows = VGroup(*(Arrow(boxes[i][0].get_right(), boxes[i + 1][0].get_left(), buff=0.05).set_color(GREY_B)
                          for i in range(4)))
        self.caption("מה קורה כשמרנדרים", say="אז מה קורה כשמרנדרים סרטון? קובץ פייתון, הסצנה רצה, "
                                              "הכרטיס הגרפי מצייר כל פריים, ותוכנה בשם אֶף אֶף אֶמְפֶּג מחברת את הפריימים לקובץ וידאו.")
        for i, b in enumerate(boxes):
            anims = [FadeIn(b, shift=0.2 * RIGHT)] + ([GrowArrow(arrows[i - 1])] if i else [])
            self.play(*anims, run_time=0.8)
        self.wait_speech()

        src = ["def say(self, text):",
               "    path, duration = tts(text)",
               "    self.add_sound(path)",
               "    self.speech_end = self.time + duration"]
        code = code_block(src, 34).move_to(1.0 * DOWN)
        self.caption("והסרטון הזה? קצת קוד משלנו מעל הספרייה",
                     say="והסרטון הזה, כמו כל הסרטונים שלנו, בנוי בדיוק כך, עם שכבה קטנה של קוד משלנו מעל הספרייה. "
                         "פונקציה שהופכת טקסט עברי לקריינות, מוסיפה את הצליל לסצנה, וזוכרת מתי הקריינות נגמרת, "
                         "כדי שהאנימציה הבאה תחכה לה.")
        self.play(FadeIn(code[0]), LaggedStartMap(FadeIn, code[1], lag_ratio=0.2), run_time=1.5)
        self.play(FadeIn(highlight(code, 2)))
        self.wait_speech()
        self.caption("כל סצנה היא קובץ וידאו, ובסוף מחברים",
                     say="כל סצנה נשמרת כקובץ וידאו נפרד, ובסוף סקריפט קטן מחבר את כולן לסרטון אחד. "
                         "ואם משנים משהו בסצנה אחת, מרנדרים מחדש רק אותה.")
        self.wait_speech()
        self.fade_all()


class Summary(IntroScene):
    def construct(self):
        items = [("Scene", "הבמה: מחלקה עם construct, התסריט"),
                 ("Mobject", "כל מה שעל המסך: נקודות, עקומות, עצים של ילדים"),
                 ("Animation", "פונקציה של אלפא מ-0 עד 1, דרך self.play"),
                 ("Updater", "מוֹבג'קט שמתעדכן בכל פריים, למשל לפי ValueTracker")]
        rows = VGroup()
        for name, desc in items:
            n = Text(name, font=MONO, font_size=34).set_color(YELLOW)
            d = heb(desc, 28)
            rows.add(VGroup(n, d))
        for r in rows:
            r[0].move_to([-4.6, 0, 0], LEFT)
            r[1].move_to([5.6, 0, 0], RIGHT)
        rows.arrange(DOWN, buff=0.55).shift(0.6 * UP)
        for r in rows:
            r[0].set_x(-4.6, LEFT)
            r[1].set_x(5.6, RIGHT)
        self.caption("ארבעה מושגים, וכל השאר בנוי מהם",
                     say="נסכם. ארבעה מושגים. סִין: הבמה, עם התסריט במתודה construct. "
                         "מוֹבְּגֶ'קְט: כל מה שרואים, שהוא בסופו של דבר נקודות ועקומות, מסודרות בעץ. "
                         "אנימציה: מעבר ממצב למצב לפי אלפא, דרך self.play. "
                         "ואַפְּדֵייטֶר: מוֹבְּגֶ'קְט שמתעדכן לבד בכל פריים.")
        self.play(LaggedStartMap(FadeIn, rows, shift=0.2 * UP, lag_ratio=0.6), run_time=6)
        self.wait_speech()
        tips = VGroup(*(heb(t, 28, GREY_A) for t in [
            "להתחיל מסצנה קטנה, ולרנדר באיכות נמוכה עם -l",
            "ללמוד מהקוד של הסרטונים של גרנט: github.com/3b1b/videos",
            "לבדוק תמיד לאיזו גרסה מתאים מדריך: manimgl או manim",
        ])).arrange(DOWN, buff=0.3).shift(0.4 * UP)
        self.caption("כמה עצות לדרך",
                     say="וכמה עצות לדרך. להתחיל מסצנה קטנה, ולרנדר באיכות נמוכה, כי זה הרבה יותר מהר. "
                         "ללמוד מהקוד של הסרטונים של גרנט, שפתוח בגיטהאב. "
                         "ולבדוק תמיד לאיזו מהגרסאות מתאים המדריך שאתם קוראים.")
        self.play(FadeOut(rows))
        self.play(LaggedStartMap(FadeIn, tips, shift=0.2 * UP, lag_ratio=0.5), run_time=3)
        self.wait_speech()
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None
        e = eye(1.0).shift(1.0 * UP)
        thanks = heb("תודה לגרנט סנדרסון", 44).next_to(e, DOWN, buff=0.6)
        site = Text("3blue1brown.com", font=FONT, font_size=30).set_color(GREY_B).next_to(thanks, DOWN, buff=0.3)
        self.say("ותודה לגרנט סנדרסון, שכתב את הכלי ונתן אותו לכולם.")
        self.play(FadeIn(e, scale=0.5))
        self.play(FadeIn(thanks), FadeIn(site))
        self.wait(2.5)
        self.fade_all(run_time=1.5)


SCENES = [Title, Grant, History, BigIdea, Hello, Coordinates, Points, Groups, Animations, AnimateSyntax, Updaters,
          BehindThisVideo, Summary]
