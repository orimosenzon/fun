"""פונקציית זטה של רימן והמספרים הראשוניים: סרטון ב-ManimGL עם קריינות בעברית.

הנתונים המספריים מחושבים מראש ב-compute/zeta_data.py.
הקריינות נוצרת ב-edge-tts (קול he-IL-AvriNeural) ונשמרת במטמון ב-audio/.
רינדור כל הסצנות וחיבורן לסרטון אחד: ./render.sh
"""
import asyncio
import hashlib
import re
from pathlib import Path

import edge_tts
import numpy as np
from pydub import AudioSegment

from manimlib import *

ROOT = Path(__file__).resolve().parent.parent
DATA = np.load(ROOT / "data" / "zeta_data.npz")
AUDIO_DIR = ROOT / "audio"
VOICE = "he-IL-AvriNeural"
SPEECH_GAP = 0.35  # שקט בין משפט למשפט
FONT = "Rubik"
BG = "#14161c"
PRIME_COLOR = YELLOW
ZERO_COLOR = RED_B
CURVE_COLOR = TEAL


LTR_START = r"[A-Za-z0-9π½²ˢ]"
LTR_END = r"[A-Za-z0-9π½²ˢ)]"
# רצף לטיני/מתמטי בתוך משפט עברי: מתחיל ונגמר בתו "חזק", כך שפיסוק וסוגריים עבריים נשארים בחוץ
LTR_RUN = re.compile(rf"{LTR_START}(?:[^֐-׿]*{LTR_END})?")


def heb(text, size=36, color=WHITE, **kwargs):
    """טקסט עברי. ביטויים כמו π(x) או 12 = 2·2·3 נעטפים ב-LRI/PDI כדי שלא יתהפכו."""
    text = LTR_RUN.sub(lambda m: "⁦" + m.group(0) + "⁩", text)
    return Text(text, font=FONT, font_size=size, fill_color=color, **kwargs)


def tts(text):
    """קובץ הקראה של הטקסט (במטמון לפי הקול והטקסט) ואורכו בשניות."""
    key = hashlib.sha1(f"{VOICE}|{text}".encode()).hexdigest()[:16]
    path = AUDIO_DIR / f"{key}.mp3"
    if not path.exists():
        AUDIO_DIR.mkdir(exist_ok=True)
        asyncio.run(edge_tts.Communicate(text, VOICE).save(str(path)))
    return path, AudioSegment.from_file(path).duration_seconds


class HebrewScene(Scene):
    default_camera_config = dict(background_color=BG)

    def setup(self):
        super().setup()
        self.cap = None
        self.speech_end = 0

    def say(self, text):
        """מתחיל להקריא מהרגע הזה. האנימציות שאחריו רצות במקביל להקראה."""
        self.wait_speech()
        path, duration = tts(text)
        self.add_sound(str(path))
        self.speech_end = self.time + duration + SPEECH_GAP

    def wait_speech(self):
        if self.speech_end > self.time:
            self.wait(self.speech_end - self.time)

    def caption(self, text, say=None, size=34, wait=0.5):
        """כיתובית קצרה על המסך, והקראה של say (או של הכיתובית עצמה)."""
        self.say(say or text)
        new = heb(text, size).to_edge(DOWN, buff=0.4)
        if self.cap is None:
            self.play(FadeIn(new, shift=0.2 * UP))
        else:
            self.play(FadeOut(self.cap, shift=0.2 * UP), FadeIn(new, shift=0.2 * UP))
        self.cap = new
        self.wait(wait)

    def clear_caption(self):
        if self.cap is not None:
            self.play(FadeOut(self.cap))
            self.cap = None

    def fade_all(self, run_time=1):
        self.wait_speech()
        self.play(*map(FadeOut, self.mobjects), run_time=run_time)
        self.cap = None


def polyline(points, **style):
    return VMobject(**style).set_points_as_corners(points)


def staircase(axes, xs, ys, **style):
    """π(x) כמדרגות אמיתיות: קטעים אופקיים וקפיצות אנכיות."""
    pts = [axes.c2p(xs[0], ys[0])]
    for x, y, y_prev in zip(xs[1:], ys[1:], ys[:-1]):
        if y != y_prev:
            pts.append(axes.c2p(x, y_prev))
        pts.append(axes.c2p(x, y))
    return polyline(pts, **style)


def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def term(n, size=36):
    """האיבר 1/nˢ בטור של זטה."""
    return Tex("1" if n == 1 else Rf"\frac{{1}}{{{n}^s}}", font_size=size)


class Title(HebrewScene):
    def construct(self):
        title = heb("פונקציית זטה של רימן", 72)
        sub = heb("והסוד שמסתתר במספרים הראשוניים", 40, GREY_A).next_to(title, DOWN, buff=0.5)
        zeta = Tex(R"\zeta(s)", font_size=120).set_color(CURVE_COLOR).next_to(title, UP, buff=0.7)
        self.say("פונקציית זטה של רימן, והסוד שמסתתר במספרים הראשוניים.")
        self.play(Write(zeta), run_time=1.5)
        self.play(FadeIn(title, shift=0.3 * UP))
        self.play(FadeIn(sub))
        self.wait(1)
        self.fade_all()


class PrimeGrid(HebrewScene):
    def construct(self):
        cells = VGroup()
        for n in range(1, 101):
            square = Square(0.6).set_stroke(GREY_D, 1)
            num = Integer(n, font_size=24).move_to(square)
            cells.add(VGroup(square, num))
        cells.arrange_in_grid(10, 10, buff=0).to_edge(UP, buff=0.4)

        self.play(LaggedStartMap(FadeIn, cells, lag_ratio=0.01, run_time=2))
        self.caption("מספר ראשוני מתחלק רק ב-1 ובעצמו",
                     say="מספר ראשוני הוא מספר שמתחלק רק באחת ובעצמו. "
                         "שתיים, שלוש, חמש, שבע, אחת עשרה, וכן הלאה.")

        primes = [cell for n, cell in enumerate(cells, 1) if is_prime(n)]
        others = [cell for n, cell in enumerate(cells, 1) if not is_prime(n)]
        self.play(
            LaggedStart(*(
                cell[0].animate.set_fill(PRIME_COLOR, 0.35).set_stroke(PRIME_COLOR, 2)
                for cell in primes
            ), lag_ratio=0.08, run_time=4),
            *(cell[1].animate.set_opacity(0.35) for cell in others),
        )
        self.caption("הם נראים מפוזרים בלי שום חוק",
                     say="כשצובעים אותם על הלוח, הם נראים מפוזרים בלי שום חוק.")
        self.caption("25 ראשוניים עד 100, ואחריהם מרווחים שגדלים ומתכווצים",
                     say="יש עשרים וחמישה ראשוניים עד מאה, "
                         "והמרווחים ביניהם גדלים ומתכווצים בלי סדר שנראה לעין.")
        self.caption("השאלה של רימן: כמה ראשוניים יש עד x?",
                     say="והשאלה של רימן הייתה: כמה ראשוניים יש עד מספר נתון, איקס?")
        self.fade_all()


class PrimeCounting(HebrewScene):
    def construct(self):
        xs, pi = DATA["x_grid"], DATA["pi_exact"]
        axes = Axes((0, 100, 10), (0, 28, 5), width=11, height=5.2).to_edge(UP, buff=0.6)
        axes.add_coordinate_labels(font_size=20)
        label = Tex(R"\pi(x)", font_size=40).set_color(PRIME_COLOR)
        label.next_to(axes.c2p(0, 28), RIGHT)
        self.play(ShowCreation(axes), Write(label))

        stairs = staircase(axes, xs, pi, stroke_color=PRIME_COLOR, stroke_width=3)
        self.caption("פונקציית הספירה π(x): קופצת ב-1 בכל ראשוני",
                     say="נסמן בפאי של איקס את מספר הראשוניים עד איקס. "
                         "הגרף שלה הוא מדרגות: הוא קופץ באחת בכל פעם שפוגשים ראשוני.")
        self.play(ShowCreation(stairs, rate_func=linear), run_time=5)
        self.wait()

        gauss = axes.get_graph(lambda x: x / np.log(x), x_range=(2.2, 100), color=BLUE)
        gauss_label = Tex(R"\frac{x}{\ln x}", font_size=34).set_color(BLUE)
        gauss_label.next_to(axes.c2p(100, 100 / np.log(100)), RIGHT, buff=0.1)
        self.caption("גאוס, בגיל 15: הצפיפות שלהם בערך 1 חלקי ln x",
                     say="גאוס, כשהיה בן חמש עשרה, שם לב שהצפיפות של הראשוניים ליד איקס "
                         "היא בערך אחת חלקי הלוגריתם של איקס.")
        self.play(ShowCreation(gauss), Write(gauss_label), run_time=2)
        self.wait()

        smooth = polyline(axes.c2p(xs, DATA["R_main"]), stroke_color=GREEN, stroke_width=3)
        smooth_label = Tex(R"R(x)", font_size=34).set_color(GREEN)
        smooth_label.next_to(smooth.get_end(), RIGHT, buff=0.1).shift(0.3 * UP)
        self.caption("הקירוב של רימן, R(x), תופס את המגמה מצוין...",
                     say="הקירוב של רימן, אר של איקס, תופס את המגמה הכללית מצוין.")
        self.play(ShowCreation(smooth), Write(smooth_label), run_time=2)
        self.wait()
        self.caption("...אבל לא את המדרגות. איפה מסתתר המידע עליהן?",
                     say="אבל לא את המדרגות עצמן. איפה מסתתר המידע עליהן?")
        self.fade_all()


class ZetaDefinition(HebrewScene):
    def construct(self):
        definition = Tex(R"\zeta(s) = 1 + \frac{1}{2^s} + \frac{1}{3^s} + \frac{1}{4^s} + \cdots",
                         font_size=60)
        definition.to_edge(UP, buff=1)
        self.play(Write(definition), run_time=2)
        self.caption("הכירו את פונקציית זטה: סכום של אחד חלקי n בחזקת s",
                     say="הכירו את פונקציית זטה. מחברים אחת, ועוד אחת חלקי שתיים בחזקת אס, "
                         "ועוד אחת חלקי שלוש בחזקת אס, וכן הלאה, על כל המספרים הטבעיים.")

        two = Tex(R"\zeta(2) = 1 + \frac{1}{4} + \frac{1}{9} + \frac{1}{16} + \cdots", font_size=48)
        two.next_to(definition, DOWN, buff=0.8)
        self.play(FadeIn(two, shift=DOWN))

        n_tracker = ValueTracker(1)
        partial = DecimalNumber(1, num_decimal_places=5, font_size=48)
        count = Integer(1, font_size=36, group_with_commas=False).set_color(GREY_B)
        row = VGroup(heb("סכום חלקי אחרי", 32), count, heb("איברים:", 32), partial)

        def update_row(row):
            n = int(n_tracker.get_value())
            count.set_value(n)
            partial.set_value(sum(1 / k ** 2 for k in range(1, n + 1)))
            row.arrange(LEFT, buff=0.25).next_to(two, DOWN, buff=0.7)

        row.add_updater(update_row)
        self.add(row)
        self.say("למשל, כשאס שווה שתיים, מחברים את ההופכיים של כל הריבועים. "
                 "מה קורה לסכום כשמוסיפים עוד ועוד איברים?")
        self.play(n_tracker.animate.set_value(2000), run_time=5, rate_func=lambda t: t ** 3)

        target = Tex(R"\frac{\pi^2}{6} = 1.64493\ldots", font_size=48).set_color(CURVE_COLOR)
        target.next_to(row, DOWN, buff=0.5)
        self.caption("אוילר, 1734: הסכום מתכנס בדיוק ל-π²/6",
                     say="בשנת אלף שבע מאות שלושים וארבע הוכיח אוילר "
                         "שהסכום מתכנס בדיוק לפאי בריבוע חלקי שש.")
        self.play(Write(target))
        self.wait(1)
        self.caption("אבל מה הקשר למספרים הראשוניים?",
                     say="פאי, בבעיה שאין בה שום עיגול. אבל מה הקשר למספרים הראשוניים?")
        self.fade_all()


class EulerProduct(HebrewScene):
    def construct(self):
        eq = Tex(
            R"\sum_{n=1}^{\infty} \frac{1}{n^s} = \prod_{p \text{ prime}} \frac{1}{1 - p^{-s}}",
            font_size=60,
        )
        eq[R"\prod_{p \text{ prime}} \frac{1}{1 - p^{-s}}"].set_color(PRIME_COLOR)
        eq.to_edge(UP, buff=0.8)
        self.play(Write(eq), run_time=2)
        self.caption("המכפלה של אוילר: הסכום על כל המספרים = מכפלה על הראשוניים בלבד", size=30,
                     say="התשובה היא אחת הנוסחאות היפות במתמטיקה: המכפלה של אוילר. "
                         "משמאל, סכום על כל המספרים הטבעיים. "
                         "מימין, מכפלה שבה מופיעים רק המספרים הראשוניים.")
        self.play(FlashAround(eq[R"\sum_{n=1}^{\infty} \frac{1}{n^s}"], color=WHITE), run_time=1.5)
        self.play(FlashAround(eq[R"\prod_{p \text{ prime}} \frac{1}{1 - p^{-s}}"], color=PRIME_COLOR),
                  run_time=1.5)

        steps = VGroup(
            heb("1. הטור ההנדסי: נוסחת הסכום והגבול שלה", 34),
            heb("2. פתיחת סוגריים: בוחרים איבר מכל סוגר", 34),
            heb("3. אינסוף סוגריים, והפירוק היחיד לראשוניים", 34),
            heb("4. בדיקה במספרים, ומתנה בדרך", 34),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.4).next_to(eq, DOWN, buff=0.9)
        self.caption("איך ייתכן שהם שווים?", size=30,
                     say="איך ייתכן שהם שווים? הראשוניים הם רק חלק קטן מהמספרים, "
                         "ובכל זאת שני הצדדים זהים. נבין את זה צעד אחר צעד.")
        self.play(LaggedStartMap(FadeIn, steps, shift=0.2 * LEFT, lag_ratio=0.4, run_time=3))
        self.fade_all()


class GeometricSeries(HebrewScene):
    def construct(self):
        header = heb("צעד 1: הטור ההנדסי", 40, CURVE_COLOR).to_edge(UP, buff=0.4)
        self.play(FadeIn(header))

        seq = Tex(R"1,\; q,\; q^2,\; q^3,\; q^4,\; \ldots", font_size=52).next_to(header, DOWN, buff=0.6)
        self.caption("סדרה הנדסית: כל איבר הוא הקודם כפול q",
                     say="צעד ראשון: נזכר בסדרה הנדסית. זו סדרה שבה כל איבר הוא הקודם, "
                         "כפול מספר קבוע שנקרא המנה ומסומן באות קיו. "
                         "נתחיל מאחת: אחת, קיו, קיו בריבוע, קיו בשלישית, וכן הלאה.")
        self.play(Write(seq), run_time=2)

        partial = Tex(R"S_n = 1 + q + q^2 + \cdots + q^{n-1}", font_size=48)
        formula = Tex(R"S_n = a_1 \cdot \frac{q^n - 1}{q - 1} = \frac{1 - q^n}{1 - q}", font_size=48)
        VGroup(partial, formula).arrange(DOWN, buff=0.5).next_to(header, DOWN, buff=0.6)
        self.caption("סכום n האיברים הראשונים: הנוסחה מבית הספר",
                     say="לסכום של אן האיברים הראשונים יש נוסחה מבית הספר: האיבר הראשון, "
                         "כפול קיו בחזקת אן פחות אחת, חלקי קיו פחות אחת. "
                         "כאן האיבר הראשון הוא אחת, ואם הופכים סימן במונה ובמכנה, מקבלים: "
                         "אחת פחות קיו בחזקת אן, חלקי אחת פחות קיו.")
        self.play(FadeTransform(seq, partial))
        self.play(Write(formula), run_time=2)
        self.wait_speech()
        top = VGroup(partial, formula)
        self.play(top.animate.scale(0.8).next_to(header, DOWN, buff=0.3))

        # הגבול, בדוגמה q = ½: על הישר רואים את Sₙ מתקרב ל-2 בזמן ש-qⁿ מתאפס
        line = NumberLine((0, 2, 0.5), width=11).move_to(0.5 * DOWN)
        line.add_numbers([0, 1, 2], font_size=28)
        colors = [BLUE, TEAL, GREEN, YELLOW, GOLD, ORANGE, RED, PINK]
        pieces, start = VGroup(), 0.0
        for i, color in enumerate(colors):
            length = 0.5 ** i
            pieces.add(Rectangle(
                width=line.get_unit_size() * length, height=0.35,
                fill_color=color, fill_opacity=0.85, stroke_width=1, stroke_color=BG,
            ).move_to(line.n2p(start + length / 2) + 0.35 * UP))
            start += length
        q_n = DecimalNumber(1, num_decimal_places=4, font_size=40)
        s_n = DecimalNumber(0, num_decimal_places=4, font_size=40)
        readout = VGroup(
            VGroup(Tex(R"q = \tfrac12", font_size=40)),
            VGroup(Tex(R"q^n =", font_size=40), q_n).arrange(RIGHT, buff=0.2),
            VGroup(Tex(R"S_n =", font_size=40), s_n).arrange(RIGHT, buff=0.2),
        ).arrange(RIGHT, buff=1.2).next_to(line, DOWN, buff=0.7)
        self.caption("למשל q = ½: החזקות של q הולכות ומתאפסות",
                     say="ועכשיו הגבול. מה קורה כשאן גדל? אם קיו בין אפס לאחת, "
                         "החזקות שלו הולכות וקטנות ושואפות לאפס. "
                         "למשל, כשקיו שווה חצי: חצי, רבע, שמינית. "
                         "ובכל צעד הסכום מתקדם חצי מהדרך שנשארה עד שתיים.")
        self.play(ShowCreation(line), FadeIn(readout))
        for i, piece in enumerate(pieces):
            n = i + 1
            self.play(GrowFromEdge(piece, LEFT),
                      ChangeDecimalToValue(q_n, 0.5 ** n), ChangeDecimalToValue(s_n, 2 - 2 * 0.5 ** n),
                      run_time=max(1.2 - 0.15 * i, 0.4))
        self.wait_speech()
        self.play(FadeOut(VGroup(line, pieces, readout)))

        limit = Tex(R"\lim_{n \to \infty} S_n = \frac{1 - 0}{1 - q} = \frac{1}{1 - q}", font_size=52)
        series = Tex(R"1 + q + q^2 + q^3 + \cdots = \frac{1}{1 - q}", font_size=56).set_color(CURVE_COLOR)
        VGroup(limit, series).arrange(DOWN, buff=0.6).move_to(0.6 * DOWN)
        self.caption("המונה שואף ל-1, ולכן הטור כולו שווה 1 חלקי 1−q",
                     say="ולכן, כשאן שואף לאינסוף, קיו בחזקת אן נעלם, המונה שואף לאחת, "
                         "וסכום הטור האינסופי כולו הוא אחת חלקי אחת פחות קיו. "
                         "כשקיו שווה חצי, זה בדיוק שתיים, כמו שראינו על הישר.")
        self.play(Write(limit), run_time=2)
        self.play(Write(series))
        self.wait_speech()
        self.play(FadeOut(VGroup(top, limit)), series.animate.next_to(header, DOWN, buff=0.4))

        sub = Tex(R"q = \frac{1}{p^s}", font_size=52).set_color(PRIME_COLOR).move_to(1.0 * UP)
        factor = Tex(
            R"\frac{1}{1 - p^{-s}} = 1 + \frac{1}{p^s} + \frac{1}{(p^2)^s} + \frac{1}{(p^3)^s} + \cdots",
            font_size=50,
        ).next_to(sub, DOWN, buff=0.4)
        two = Tex(R"\frac{1}{1 - 2^{-s}} = 1 + \frac{1}{2^s} + \frac{1}{4^s} + \frac{1}{8^s} + \frac{1}{16^s} + \cdots",
                  font_size=46).set_color(PRIME_COLOR).next_to(factor, DOWN, buff=0.4)
        self.caption("מציבים q = 1/pˢ: כל גורם במכפלה הוא סכום החזקות של p",
                     say="עכשיו נציב קיו שווה אחת חלקי פי בחזקת אס, כשפי ראשוני. "
                         "פי הוא לפחות שתיים, אז קיו קטן מאחת והנוסחה תקפה. "
                         "יוצא שכל גורם במכפלה של אוילר הוא בעצמו טור אינסופי: "
                         "אחת, ועוד אחת חלקי פי, ועוד אחת חלקי פי בריבוע, וכן הלאה.")
        self.play(Write(sub))
        self.play(Write(factor), run_time=2)
        self.caption("הגורם של 2 מכיל את כל החזקות של 2",
                     say="למשל, הגורם של שתיים מכיל את כל החזקות של שתיים: "
                         "אחת, שתיים, ארבע, שמונה, שש עשרה, וכן הלאה.")
        self.play(FadeIn(two, shift=DOWN))
        self.wait(1)
        self.fade_all()


class OpenBrackets(HebrewScene):
    def construct(self):
        header = heb("צעד 2: פותחים סוגריים", 40, CURVE_COLOR).to_edge(UP, buff=0.3)
        self.play(FadeIn(header))

        prod = Tex("(1 + x)(1 + y)", font_size=60)
        result = Tex("= 1 + y + x + xy", font_size=60)
        VGroup(prod, result).arrange(RIGHT, buff=0.3).next_to(header, DOWN, buff=0.6)
        self.caption("כמה זה, כשפותחים את הסוגריים?",
                     say="צעד שני: נזכר איך פותחים סוגריים. "
                         "כמה זה אחת ועוד איקס, כפול אחת ועוד וואי?")
        self.play(Write(prod))

        # טבלת הבחירות: שורה לכל איבר מהסוגר הראשון, עמודה לכל איבר מהשני
        firsts = [prod["1"][0], prod["x"][0]]
        seconds = [prod["1"][1], prod["y"][0]]
        names = [["1", "y"], ["x", "xy"]]
        size, center = 1.2, np.array([0, -0.6, 0])
        boxes, cells = VGroup(), {}
        for i in range(2):
            for j in range(2):
                pos = center + size * np.array([j - 0.5, 0.5 - i, 0])
                boxes.add(Square(size).set_stroke(GREY_C, 1).move_to(pos))
                cells[i, j] = Tex(names[i][j], font_size=50).move_to(pos)
        heads = VGroup(
            *(Tex(t, font_size=40).set_color(BLUE).move_to(center + size * np.array([-1.3, 0.5 - i, 0]))
              for i, t in enumerate(["1", "x"])),
            *(Tex(t, font_size=40).set_color(GREEN).move_to(center + size * np.array([j - 0.5, 1.3, 0]))
              for j, t in enumerate(["1", "y"])),
        )
        self.caption("בוחרים איבר אחד מכל סוגר, ומכפילים",
                     say="כל איבר בתוצאה נוצר מבחירה: בוחרים איבר אחד מהסוגריים הראשונים, "
                         "איבר אחד מהשניים, ומכפילים. "
                         "שתי אפשרויות כפול שתי אפשרויות, ולכן יש ארבעה איברים.")
        self.play(ShowCreation(boxes), FadeIn(heads))
        for i in range(2):
            for j in range(2):
                rects = VGroup(SurroundingRectangle(firsts[i], buff=0.06, stroke_color=BLUE),
                               SurroundingRectangle(seconds[j], buff=0.06, stroke_color=GREEN))
                self.play(ShowCreation(rects), TransformFromCopy(VGroup(firsts[i], seconds[j]), cells[i, j]),
                          run_time=0.9)
                self.play(FadeOut(rects), run_time=0.3)
        self.play(Write(result))
        self.wait_speech()
        self.play(FadeOut(VGroup(boxes, heads, *cells.values(), prod, result)))

        square = VGroup(
            Tex(R"(1 + x)^2 = (1 + x)(1 + x)", font_size=52),
            Tex(R"= 1 + x + x + x^2 = 1 + 2x + x^2", font_size=52),
        ).arrange(DOWN, buff=0.5).next_to(header, DOWN, buff=0.9)
        square[1]["2x"].set_color(PRIME_COLOR)
        self.caption("שתי בחירות שונות נותנות x, ולכן המקדם 2",
                     say="ומה עם אחת ועוד איקס, בריבוע? כאן שתי בחירות שונות נותנות את אותו איבר, איקס: "
                         "איקס מהסוגריים הראשונים, או איקס מהשניים. לכן איקס מופיע פעמיים, והמקדם שלו שתיים. "
                         "שימו לב לזה, זה יהיה חשוב בהמשך.")
        self.play(Write(square[0]))
        self.play(Write(square[1]), run_time=2)
        self.play(FlashAround(square[1]["2x"], color=PRIME_COLOR))
        self.wait_speech()
        self.play(FadeOut(square))

        # שני סוגריים אינסופיים: טבלת כפל של חזקות 2 וחזקות 3
        brackets = Tex(R"\left(1 + \frac{1}{2^s} + \frac{1}{4^s} + \cdots\right)"
                       R"\left(1 + \frac{1}{3^s} + \frac{1}{9^s} + \frac{1}{27^s} + \cdots\right)", font_size=40)
        brackets.next_to(header, DOWN, buff=0.4)
        self.caption("ומה אם בכל סוגר יש אינסוף איברים?",
                     say="ומה אם בכל סוגר יש אינסוף איברים? ניקח את החזקות של שתיים ואת החזקות של שלוש. "
                         "הרעיון לא משתנה: כל איבר בתוצאה הוא בחירה של חזקה של שתיים, "
                         "וחזקה של שלוש.")
        self.play(Write(brackets), run_time=2)

        w, h, x0, y0 = 1.3, 0.9, -4.6, 1.0
        twos, threes = [1, 2, 4], [1, 3, 9, 27]
        row_heads = VGroup(*(term(a, 32).set_color(BLUE).move_to([x0, y0 - h * (i + 1), 0])
                             for i, a in enumerate(twos)))
        col_heads = VGroup(*(term(b, 32).set_color(GREEN).move_to([x0 + w * (j + 1), y0, 0])
                             for j, b in enumerate(threes)))
        table = VGroup(*(term(a * b, 32).move_to([x0 + w * (j + 1), y0 - h * (i + 1), 0])
                         for i, a in enumerate(twos) for j, b in enumerate(threes)))
        dots = VGroup(Tex(R"\vdots", font_size=32).move_to([x0 + w * 2.5, y0 - h * 4, 0]),
                      Tex(R"\cdots", font_size=32).move_to([x0 + w * 5, y0 - h * 2, 0]))
        rule = VGroup(Line([x0 + w / 2, y0 + h / 2, 0], [x0 + w / 2, y0 - h * 4.3, 0]),
                      Line([x0 - w / 2, y0 - h / 2, 0], [x0 + w * 5.4, y0 - h / 2, 0])).set_stroke(GREY_C, 1)
        self.play(ShowCreation(rule), FadeIn(row_heads), FadeIn(col_heads))
        self.caption("טבלת כפל: כל המספרים שבנויים רק מ-2 ומ-3",
                     say="התוצאה היא טבלת כפל אינסופית. בכל משבצת יש אחת חלקי מספר שבנוי רק משתיים ומשלוש: "
                         "שש, שתים עשרה, שמונה עשרה, שלושים ושש.")
        self.play(LaggedStartMap(FadeIn, table, lag_ratio=0.12, run_time=3), FadeIn(dots))
        self.wait_speech()

        compact = Tex(R"= \frac{1}{1 - 2^{-s}} \cdot \frac{1}{1 - 3^{-s}}", font_size=40).set_color(PRIME_COLOR)
        compact.next_to(dots[1], RIGHT, buff=0.4).set_y(y0 - h * 2)
        self.caption("ובכתיב המקוצר: המכפלה של שני גורמי אוילר",
                     say="ובכתיב המקוצר מהצעד הקודם, כל הטבלה הזאת היא פשוט המכפלה של שני הגורמים של אוילר, "
                         "של שתיים ושל שלוש.")
        self.play(Write(compact))
        self.wait(1)
        self.fade_all()


class InfiniteProduct(HebrewScene):
    def construct(self):
        header = heb("צעד 3: אינסוף סוגריים", 40, CURVE_COLOR).to_edge(UP, buff=0.3)
        product = Tex(R"\left(1 + \tfrac{1}{2^s} + \tfrac{1}{4^s} + \cdots\right)"
                      R"\left(1 + \tfrac{1}{3^s} + \tfrac{1}{9^s} + \cdots\right)"
                      R"\left(1 + \tfrac{1}{5^s} + \cdots\right)\left(1 + \tfrac{1}{7^s} + \cdots\right)\cdots",
                      font_size=36).next_to(header, DOWN, buff=0.45)
        self.play(FadeIn(header))
        self.caption("סוגריים אחד לכל ראשוני",
                     say="עכשיו אוילר לוקח את הרעיון עד הסוף: סוגריים אחד לכל ראשוני, אינסוף סוגריים. "
                         "איבר בתוצאה הוא שוב בחירה: חזקה אחת מכל סוגר, ומכפילים.")
        self.play(Write(product), run_time=2.5)

        primes = [2, 3, 5, 7, 11, 13]
        heads = VGroup(*(Tex(str(p), font_size=44).set_color(PRIME_COLOR) for p in primes),
                       Tex(R"\cdots", font_size=44))
        heads.arrange(RIGHT, buff=0.8).move_to(1.1 * UP)
        exps = VGroup(*(Integer(0, font_size=44).move_to(head.get_center() + 0.8 * DOWN) for head in heads[:-1]))
        exp_dots = Tex(R"\cdots", font_size=44).move_to(heads[-1].get_center() + 0.8 * DOWN)
        tags = VGroup(heb("ראשוני", 28, GREY_B).next_to(heads, RIGHT, buff=0.6),
                      heb("חזקה", 28, GREY_B).next_to(VGroup(exps, exp_dots), RIGHT, buff=0.6))
        tags[1].align_to(tags[0], LEFT)

        def choice_tex(ks, n):
            powers = R" \cdot ".join(f"{p}^{{{k}}}" for p, k in zip(primes, ks))
            value = Rf"\frac{{1}}{{{n}^s}}" if n > 1 else "1"
            return Tex(Rf"\frac{{1}}{{\left({powers} \cdots\right)^s}} = {{{value}}}", font_size=42).move_to(1.0 * DOWN)

        self.caption("בחירה = רשימת חזקות, אחת לכל ראשוני",
                     say="אפשר לתאר כל בחירה כרשימה של חזקות, אחת לכל ראשוני. "
                         "למשל: שתיים בחזקת שתיים, שלוש בחזקת אחת, ואפס בכל השאר. "
                         "זה אחת חלקי שתים עשרה בחזקת אס.")
        self.play(FadeIn(heads), FadeIn(exps), FadeIn(exp_dots), FadeIn(tags))
        choices = [
            ((2, 1, 0, 0, 0, 0), 12, None),
            ((0, 0, 0, 1, 0, 0), 7, "שבע בחזקת אחת, ואפס בכל השאר, נותן שבע."),
            ((1, 2, 1, 0, 0, 0), 90, "שתיים, שלוש בריבוע וחמש נותנים תשעים."),
            ((0, 0, 0, 0, 0, 0), 1, "ואפס בכל המקומות נותן את האחת."),
        ]
        shown = None
        for ks, n, text in choices:
            if text:
                self.say(text)
            new = choice_tex(ks, n)
            # צבע אחרי השינוי: animate על מספר באמצע ChangeDecimalToValue קורס
            self.play(*(ChangeDecimalToValue(e, k) for e, k in zip(exps, ks)),
                      FadeTransform(shown, new) if shown else FadeIn(new))
            for e, k in zip(exps, ks):
                e.set_color(CURVE_COLOR if k else WHITE)
            shown = new
            self.wait(1)
        self.wait_speech()

        vanish = Tex(R"\frac{1}{2^s} \cdot \frac{1}{3^s} \cdot \frac{1}{5^s} \cdot \frac{1}{7^s} \cdots"
                     R"\;\longrightarrow\; 0", font_size=46).move_to(1.0 * DOWN)
        vanish["0"].set_color(RED)
        self.caption("חזקה חיובית באינסוף סוגריים? המכפלה שואפת ל-0", size=30,
                     say="ומה אם בוחרים חזקה חיובית מאינסוף סוגריים? "
                         "אז כופלים אינסוף מספרים שכולם קטנים מחצי, והמכפלה שואפת לאפס. איברים כאלה נעלמים. "
                         "שורדות רק בחירות שבהן, החל ממקום מסוים, כל החזקות הן אפס.")
        self.play(*(ChangeDecimalToValue(e, 1) for e in exps),
                  exp_dots.animate.set_color(RED), FadeTransform(shown, vanish))
        exps.set_color(RED)
        self.wait_speech()
        self.play(FadeOut(VGroup(heads, exps, exp_dots, tags, vanish)))

        cells = VGroup()
        for n in range(1, 31):
            square = Square(0.62).set_stroke(GREY_D, 1)
            cells.add(VGroup(square, Integer(n, font_size=24).move_to(square)))
        cells.arrange_in_grid(2, 15, buff=0).move_to(0.6 * UP)
        self.caption("השורדים: מכפלות סופיות של ראשוניים, כלומר כל המספרים הטבעיים", size=30,
                     say="ומה הן מכפלות סופיות של ראשוניים? אלה בדיוק כל המספרים הטבעיים. "
                         "כל אחד מהם מופיע כאחת חלקי המספר בחזקת אס.")
        self.play(FadeIn(cells))
        self.play(LaggedStart(*(c[0].animate.set_fill(PRIME_COLOR, 0.4).set_stroke(PRIME_COLOR, 2)
                                for c in cells), lag_ratio=0.05, run_time=2))
        self.wait_speech()

        unique = Tex(R"12 = 2^2 \cdot 3^1 \cdot 5^0 \cdot 7^0 \cdots", font_size=46).set_color(CURVE_COLOR)
        unique.next_to(cells, DOWN, buff=0.7)
        self.caption("וכל מספר מופיע פעם אחת בדיוק: הפירוק לראשוניים יחיד", size=30,
                     say="ועכשיו נזכור את המקדם שתיים מקודם. אם מספר היה מתקבל משתי בחירות שונות, "
                         "הוא היה מופיע פעמיים. אבל לכל מספר יש פירוק יחיד לגורמים ראשוניים, "
                         "וזה המשפט היסודי של האריתמטיקה. לשתים עשרה יש רק רשימת חזקות אחת: "
                         "שתיים, אחת, ואפס בכל השאר. לכן כל מספר מופיע בדיוק פעם אחת.")
        self.play(Write(unique))
        self.play(cells[11][0].animate.set_fill(CURVE_COLOR, 0.7), Flash(cells[11], color=CURVE_COLOR))
        self.wait_speech()
        self.play(FadeOut(VGroup(cells, unique, product)))

        final = Tex(R"\prod_p \frac{1}{1 - p^{-s}} = \sum_{n=1}^{\infty} \frac{1}{n^s}", font_size=68)
        final[R"\prod_p \frac{1}{1 - p^{-s}}"].set_color(PRIME_COLOR)
        final.move_to(0.4 * UP)
        self.caption("זה הלב של העניין: המשפט היסודי של האריתמטיקה, בשפה של אנליזה", size=30,
                     say="וזה הלב של העניין. המכפלה של אוילר היא המשפט היסודי של האריתמטיקה, "
                         "כתוב בשפה של אנליזה.")
        self.play(Write(final), run_time=2)
        self.caption("בדיוק מתמטי: עובדים עם s > 1, ששם שני האגפים סופיים", size=30,
                     say="והערה למדקדקים: כדי שהמשחק עם אינסוף סוגריים יהיה מדויק, עובדים עם אס גדול מאחת. "
                         "שם שני האגפים הם מספרים סופיים, ואפשר באמת להשוות ביניהם.")
        self.wait(1)
        self.fade_all()


class EulerCheck(HebrewScene):
    def construct(self):
        header = heb("צעד 4: בדיקה במספרים", 40, CURVE_COLOR).to_edge(UP, buff=0.3)
        self.play(FadeIn(header))

        at_two = Tex(R"\prod_p \frac{1}{1 - \frac{1}{p^2}} = \frac{4}{3}\cdot\frac{9}{8}\cdot\frac{25}{24}"
                     R"\cdot\frac{49}{48}\cdot\frac{121}{120}\cdots", font_size=48)
        at_two.next_to(header, DOWN, buff=0.6)
        self.caption("ב-s = 2, מכפלה על הראשוניים בלבד",
                     say="בואו נבדוק במספרים. כשאס שווה שתיים, כל גורם הוא פי בריבוע "
                         "חלקי פי בריבוע פחות אחת: ארבע שלישים, תשע שמיניות, עשרים וחמש חלקי עשרים וארבע.")
        self.play(Write(at_two), run_time=2)

        sieve = np.ones(20000, dtype=bool)
        sieve[:2] = False
        for d in range(2, 142):
            sieve[d * d::d] = False
        primes = np.flatnonzero(sieve)[:2000]
        products = np.cumprod(1 / (1 - primes.astype(float) ** -2))

        k_tracker = ValueTracker(1)
        value = DecimalNumber(products[0], num_decimal_places=5, font_size=48)
        count = Integer(1, font_size=36, group_with_commas=False).set_color(GREY_B)
        row = VGroup(heb("מכפלה על", 32), count, heb("ראשוניים:", 32), value)

        def update_row(row):
            k = int(k_tracker.get_value())
            count.set_value(k)
            value.set_value(products[k - 1])
            row.arrange(LEFT, buff=0.25).next_to(at_two, DOWN, buff=0.8)

        row.add_updater(update_row)
        self.add(row)
        target = Tex(R"\frac{\pi^2}{6} = 1.64493\ldots", font_size=48).set_color(CURVE_COLOR)
        target.next_to(row, DOWN, buff=0.6)
        self.say("וככל שמכפילים ביותר ראשוניים, המכפלה מתקרבת לאותו מספר בדיוק: "
                 "פאי בריבוע חלקי שש. אותו מספר, משני כיוונים שונים לגמרי.")
        self.play(k_tracker.animate.set_value(len(primes)), run_time=5, rate_func=lambda t: t ** 3)
        self.play(Write(target))
        self.wait_speech()
        row.clear_updaters()
        self.play(*map(FadeOut, [at_two, row, target]))

        harmonic = Tex(R"1 + \frac12 + \frac13 + \frac14 + \frac15 + \cdots = \infty", font_size=50)
        product = Tex(R"\prod_p \frac{1}{1 - \frac1p} = \frac{2}{1}\cdot\frac{3}{2}\cdot\frac{5}{4}\cdot\frac{7}{6}\cdots",
                      font_size=50).set_color(PRIME_COLOR)
        VGroup(harmonic, product).arrange(DOWN, buff=0.7).next_to(header, DOWN, buff=0.8)
        self.caption("מתנה בדרך: יש אינסוף ראשוניים",
                     say="ומכאן מתנה קטנה: ההוכחה של אוילר שיש אינסוף ראשוניים. "
                         "כשאס שווה אחת, הסכום הוא הטור ההרמוני, והוא גדל לאינסוף.")
        self.play(Write(harmonic), run_time=2)
        self.play(Write(product), run_time=2)
        self.caption("אילו היו מעט ראשוניים, המכפלה הייתה סופית",
                     say="אילו היה רק מספר סופי של ראשוניים, המכפלה הזאת הייתה מספר רגיל וסופי. "
                         "אבל היא שווה לסכום אינסופי. לכן הראשוניים לא נגמרים לעולם.")
        self.play(FlashAround(product, color=PRIME_COLOR), run_time=1.5)
        self.caption("זטה יודעת הכול על הראשוניים. רימן הבין איך לחלץ את זה", size=30,
                     say="המכפלה הזאת היא הגשר. זטה יודעת הכול על הראשוניים, "
                         "ורימן הבין איך לחלץ ממנה את המידע הזה.")
        self.wait(1)
        self.fade_all()


class ComplexZeros(HebrewScene):
    def construct(self):
        title = heb("רימן, 1859: נציב ב-s מספרים מרוכבים", 40).to_edge(UP, buff=0.3)
        self.say("רימן, בשנת אלף שמונה מאות חמישים ותשע, עשה צעד נועז: הוא הציב בזטה מספרים מרוכבים.")
        self.play(FadeIn(title))

        axes = Axes((-8, 3, 1), (-35, 35, 10), width=9, height=6.2)
        axes.next_to(title, DOWN, buff=0.3)
        axes.add_coordinate_labels(x_values=[-6, -4, -2, 1], y_values=[-30, -20, -10, 10, 20, 30],
                                   font_size=18)
        re_label = Tex(R"\mathrm{Re}(s)", font_size=28).next_to(axes.x_axis.get_end(), DOWN)
        im_label = Tex(R"\mathrm{Im}(s)", font_size=28).next_to(axes.y_axis.get_end(), RIGHT)
        self.play(ShowCreation(axes), Write(re_label), Write(im_label))

        strip = Rectangle(
            width=axes.x_axis.get_unit_size(), height=axes.y_axis.get_length(),
            fill_color=BLUE, fill_opacity=0.18, stroke_width=0,
        ).move_to(axes.c2p(0.5, 0))
        crit = DashedLine(axes.c2p(0.5, -35), axes.c2p(0.5, 35), stroke_color=CURVE_COLOR)
        pole = Tex(R"\times", font_size=40).set_color(GREY_B).move_to(axes.c2p(1, 0))

        self.caption("ב-s=1 זטה מתפוצצת לאינסוף (הטור ההרמוני)",
                     say="בנקודה אס שווה אחת זטה מתפוצצת לאינסוף, כי שם היא הטור ההרמוני.")
        self.play(Write(pole))
        self.wait()

        trivial = VGroup(*(Dot(axes.c2p(x, 0), radius=0.07, fill_color=GREY_A) for x in (-2, -4, -6, -8)))
        self.caption("יש לה אפסים 'טריוויאליים' בכל המספרים הזוגיים השליליים",
                     say="יש לה אפסים שנקראים טריוויאליים, בכל המספרים הזוגיים השליליים: "
                         "מינוס שתיים, מינוס ארבע, מינוס שש.")
        self.play(LaggedStartMap(GrowFromCenter, trivial, lag_ratio=0.2))
        self.wait()

        self.caption("וכל השאר חיים ברצועה הקריטית, בין 0 ל-1",
                     say="וכל שאר האפסים חיים ברצועה צרה בין אפס לאחת, שנקראת הרצועה הקריטית.")
        self.play(FadeIn(strip))
        self.wait()

        gammas = [g for g in DATA["zeros"] if g < 35]
        zeros = VGroup(*(
            Dot(axes.c2p(0.5, sign * g), radius=0.07, fill_color=ZERO_COLOR)
            for g in gammas for sign in (1, -1)
        ))
        first = Tex(R"\tfrac12 + 14.1347\ldots i", font_size=28).set_color(ZERO_COLOR)
        first.next_to(axes.c2p(0.5, gammas[0]), RIGHT, buff=0.3)
        self.play(LaggedStartMap(GrowFromCenter, zeros, lag_ratio=0.1), Write(first), run_time=2)
        self.caption("והפלא: כל האפסים שנמצאו יושבים בדיוק על הקו Re = ½",
                     say="והפלא: כל האפסים שנמצאו אי פעם יושבים בדיוק על הקו שבו החלק הממשי הוא חצי.")
        self.play(ShowCreation(crit), run_time=1.5)
        self.wait(1)
        self.fade_all()


class CriticalLineSpiral(HebrewScene):
    def construct(self):
        ts, zs, gammas = DATA["crit_t"], DATA["crit_z"], DATA["zeros"]
        t_max = ts[-1]

        line_axes = Axes((0, 1, 0.5), (0, t_max, 10), width=2, height=5.8)
        line_axes.to_edge(LEFT, buff=1.2).shift(0.2 * UP)
        line_axes.add_coordinate_labels(x_values=[], y_values=range(10, 51, 10), font_size=18)
        crit = Line(line_axes.c2p(0.5, 0), line_axes.c2p(0.5, t_max), stroke_color=CURVE_COLOR)
        s_label = Tex(R"s = \tfrac12 + it", font_size=32).next_to(line_axes, UP, buff=0.15)

        plane = ComplexPlane((-2, 4), (-3, 3))
        plane.set_height(5.8).to_edge(RIGHT, buff=0.9).shift(0.2 * UP)
        plane.add_coordinate_labels(font_size=18)
        z_label = Tex(R"\zeta(s)", font_size=36).set_color(CURVE_COLOR)
        z_label.next_to(plane, UP, buff=0.15)
        arrow = Arrow(line_axes.get_right(), plane.get_left(), buff=0.4, stroke_color=GREY_B)

        self.play(ShowCreation(line_axes), ShowCreation(crit), Write(s_label))
        self.play(ShowCreation(plane), GrowArrow(arrow), Write(z_label))
        self.caption("נטייל לאורך הקו הקריטי ונראה לאן זטה שולחת כל נקודה", size=30,
                     say="בואו נטייל לאורך הקו הקריטי, ונראה לאן זטה שולחת כל נקודה עליו.")

        t_tracker = ValueTracker(0)
        s_dot = Dot(fill_color=WHITE).add_updater(
            lambda d: d.move_to(line_axes.c2p(0.5, t_tracker.get_value())))
        z_dot = Dot(fill_color=WHITE).add_updater(lambda d: d.move_to(
            plane.n2p(complex(np.interp(t_tracker.get_value(), ts, zs.real),
                              np.interp(t_tracker.get_value(), ts, zs.imag)))))
        self.add(s_dot, z_dot)

        marks = [0, *(g for g in gammas if g < t_max), t_max]
        for i, (a, b) in enumerate(zip(marks[:-1], marks[1:])):
            mask = (ts >= a) & (ts <= b)
            segment = polyline(plane.c2p(zs[mask].real, zs[mask].imag),stroke_color=CURVE_COLOR, stroke_width=2.5)
            duration = max(4.0 - 0.6 * i, 1.0) if i < 3 else 1.0
            self.play(ShowCreation(segment, rate_func=linear),
                      t_tracker.animate.set_value(b).set_anim_args(rate_func=linear),
                      run_time=duration)
            if b == t_max:
                break
            marker = Dot(line_axes.c2p(0.5, b), radius=0.06, fill_color=ZERO_COLOR)
            self.play(Flash(plane.n2p(0), color=ZERO_COLOR, flash_radius=0.3),
                      FadeIn(marker, scale=3), run_time=0.5)
            if i == 0:
                self.caption("הנה! זטה פוגעת בדיוק ב-0: האפס הראשון, t ≈ 14.13", size=30,
                             say="הנה! זטה פוגעת בדיוק באפס. "
                                 "זה האפס הראשון, בגובה ארבע עשרה נקודה אחת שלוש.")
                self.caption("כל מעבר דרך הראשית הוא עוד אפס", size=30, wait=0.3,
                             say="ובכל פעם שהמסלול עובר דרך הראשית, זה עוד אפס.")

        self.caption("עשרה אפסים עד 50. ידועים יותר מ-10 טריליון, וכולם על הקו", size=30,
                     say="עשרה אפסים עד גובה חמישים. "
                         "היום ידועים יותר מעשרה טריליון אפסים, וכולם על הקו.")
        self.wait(1)
        self.fade_all()


class ExplicitFormula(HebrewScene):
    def construct(self):
        xs, pi = DATA["x_grid"], DATA["pi_exact"]
        formula = Tex(R"\pi(x) \approx R(x) - \sum_{\rho} R(x^{\rho})", font_size=46)
        formula[R"\sum_{\rho} R(x^{\rho})"].set_color(ZERO_COLOR)
        formula.to_edge(UP, buff=0.3)
        self.play(Write(formula))
        self.caption("הנוסחה המפורשת של רימן: כל אפס ρ מוסיף גל לקירוב", size=30,
                     say="וכאן מגיע הקסם. הנוסחה המפורשת של רימן אומרת שכל אפס, רו, "
                         "מוסיף גל אחד לקירוב החלק.")

        axes = Axes((0, 100, 10), (0, 28, 5), width=11, height=4.8).next_to(formula, DOWN, buff=0.3)
        axes.add_coordinate_labels(font_size=20)
        stairs = staircase(axes, xs, pi, stroke_color=PRIME_COLOR, stroke_width=2, stroke_opacity=0.6)
        self.play(ShowCreation(axes), ShowCreation(stairs), run_time=1.5)

        def curve(ys):
            return polyline(axes.c2p(xs, ys), stroke_color=CURVE_COLOR, stroke_width=3)

        approx = curve(DATA["R_main"])
        label = heb("אפסים:", 34).next_to(axes.c2p(5, 26), RIGHT)
        k_tracker = ValueTracker(0)
        count = Integer(0, font_size=40).set_color(ZERO_COLOR)
        count.add_updater(lambda m: m.set_value(round(k_tracker.get_value())).next_to(label, LEFT, buff=0.25))
        counter = VGroup(label, count)
        self.play(ShowCreation(approx), FadeIn(counter))
        self.wait()

        # הגל של האפס הראשון, מוצג לבד על רקע הגרף המעומעם
        plot = VGroup(axes, stairs, approx, counter)
        wave = DATA["approx_1"] - DATA["R_main"]
        wave_axes = Axes((0, 100, 10), (-1, 1, 1), width=11, height=2.6).move_to(axes)
        first_wave = polyline(wave_axes.c2p(xs, wave), stroke_color=ZERO_COLOR, stroke_width=3)
        wave_label = Tex(R"\rho_1 = \tfrac12 + 14.13\,i", font_size=34).set_color(ZERO_COLOR)
        wave_label.next_to(wave_axes, UP, buff=0.1).to_edge(RIGHT, buff=1.2)
        self.caption("זה הגל שתורם האפס הראשון:", size=30, wait=0.3,
                     say="זה הגל שתורם האפס הראשון. הוא מתנודד, "
                         "ומהירות התנודה נקבעת לפי הגובה של האפס על הקו.")
        plot.save_state()
        # רק עמעום קווים: set_opacity על polyline היה צובע גם את המילוי מתחת לעקומה
        self.play(VGroup(axes, stairs, approx).animate.set_stroke(opacity=0.15),
                  counter.animate.set_opacity(0.15))
        self.play(ShowCreation(first_wave), Write(wave_label), run_time=2)
        self.wait_speech()
        self.play(FadeOut(first_wave), FadeOut(wave_label), Restore(plot))

        for k in (1, 3, 10, 30, 100):
            self.play(Transform(approx, curve(DATA[f"approx_{k}"])), k_tracker.animate.set_value(k),
                      run_time=1.6)
            if k == 3:
                self.caption("עוד אפסים, עוד גלים, והמדרגות מתחילות להופיע", size=30, wait=0.3,
                             say="מוסיפים עוד אפסים, עוד גלים, והמדרגות מתחילות להופיע.")
            else:
                self.wait(0.6)

        self.caption("עם 100 אפסים, המוזיקה של האפסים משחזרת את הראשוניים", size=30,
                     say="עם מאה אפסים, המוזיקה של האפסים משחזרת את הראשוניים כמעט בדיוק.")
        self.wait(1)
        self.fade_all()


class Hypothesis(HebrewScene):
    def construct(self):
        title = heb("השערת רימן", 64).set_color(CURVE_COLOR).to_edge(UP, buff=0.8)
        statement = heb("כל האפסים הלא טריוויאליים של זטה יושבים על הקו", 38)
        line = Tex(R"\mathrm{Re}(s) = \tfrac{1}{2}", font_size=56)
        VGroup(statement, line).arrange(DOWN, buff=0.4).next_to(title, DOWN, buff=0.8)
        self.say("השערת רימן: כל האפסים הלא טריוויאליים של זטה "
                 "יושבים על הקו שבו החלק הממשי הוא חצי.")
        self.play(Write(title))
        self.play(FadeIn(statement), Write(line))
        self.wait(1.5)

        meaning = heb("אם היא נכונה, הראשוניים סוטים מהממוצע הכי מעט שאפשר:", 32, GREY_A)
        bound = Tex(R"\left|\pi(x) - \mathrm{Li}(x)\right| \le \tfrac{1}{8\pi}\sqrt{x}\,\ln x", font_size=48)
        bound.set_color(PRIME_COLOR)
        VGroup(meaning, bound).arrange(DOWN, buff=0.35).next_to(line, DOWN, buff=0.8)
        self.say("אם היא נכונה, הראשוניים סוטים מהממוצע הכי מעט שאפשר, "
                 "בערך כמו השורש של איקס.")
        self.play(FadeIn(meaning))
        self.play(Write(bound))
        self.wait(1)

        footer = heb("פתוחה מאז 1859. פרס המילניום: מיליון דולר", 34, GREY_B).to_edge(DOWN, buff=0.6)
        self.say("ההשערה פתוחה מאז אלף שמונה מאות חמישים ותשע, "
                 "ומי שיוכיח אותה יקבל פרס של מיליון דולר.")
        self.play(FadeIn(footer, shift=UP))
        self.wait(2)
        self.fade_all(run_time=1.5)


class Credits(HebrewScene):
    def construct(self):
        thanks = heb("תודה", 60, CURVE_COLOR)
        line1 = heb("ההסבר על מכפלת אוילר נכתב בהשראת", 36)
        line2 = heb("גדי אלכסנדרוביץ' והבלוג שלו \"לא מדויק\"", 44, PRIME_COLOR)
        post = heb("הפוסט \"הוכחת אוילר לקיום אינסוף ראשוניים\"", 30, GREY_A)
        site = Text("gadial.net", font=FONT, font_size=34, fill_color=GREY_B)
        VGroup(thanks, line1, line2, post, site).arrange(DOWN, buff=0.45)
        self.say("ההסבר על מכפלת אוילר נכתב בהשראת הפוסט של גדי אלכסנדרוביץ' "
                 "על ההוכחה של אוילר לקיום אינסוף ראשוניים, בבלוג שלו, לא מדויק. תודה, גדי.")
        self.play(Write(thanks))
        self.play(LaggedStartMap(FadeIn, VGroup(line1, line2, post, site), shift=0.2 * UP, lag_ratio=0.4,
                                 run_time=2.5))
        self.wait(2)
        self.fade_all(run_time=1.5)


SCENES = [Title, PrimeGrid, PrimeCounting, ZetaDefinition,
          EulerProduct, GeometricSeries, OpenBrackets, InfiniteProduct, EulerCheck,
          ComplexZeros, CriticalLineSpiral, ExplicitFormula, Hypothesis, Credits]
