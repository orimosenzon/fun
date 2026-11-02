"""התמרת פורייה: מהזמן אל התדר, ההתמרה הבדידה, קונבולוציה והתמרת פורייה המהירה.

ההמשך של fourier.py, בעקבות הפוסטים של גדי אלכסנדרוביץ' בבלוג "לא מדויק" (gadial.net):
התמרת פורייה - מה זה בכלל? (2014), התמרת פורייה הבדידה - מה זה בכלל? (2014),
והתמרת פורייה המהירה (2014).
רינדור: ./render.sh --hd scenes/fourier_transform.py
"""
import colorsys
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# הקריינות (Azure, מילון ההגייה) והעזרים הגרפיים מגיעים מהסרטון על טורי פורייה
from fourier import (AXIS, F_COLOR, NEG, POS, SUM_COLOR, WAVES, FourierScene, bars, curve,  # noqa: E402
                     direction, signed_area, square_coeffs)
from common import *  # noqa: E402,F403
import common  # noqa: E402

common.SPOKEN.update({"קולי וטוקי": "קוּלִי וטוּקִי", "דיראק": "דִירָאק", "ונדרמונדה": "וַנְדֶרְמוֹנְדֶה",
                      "נייקוויסט": "נַייקוִויסְט", "פואסון": "פּוּאָסוֹן"})

SPEC = ORANGE  # צבע ההתמרה


# ---------- פונקציות ----------

def rect(t):
    return np.where(np.abs(t) < 0.5, 1.0, 0.0)


def sinc(w):
    return np.sinc(w)  # sin(πw) / (πw)


def ft_numeric(f, ws, t0=-12, t1=12, n=24000):
    """f̂(ω) = ∫ f(t) e^{-2πiωt} dt, בסכום רימן. מחזיר מערך מרוכב."""
    ts = np.linspace(t0, t1, n, endpoint=False)
    dt = (t1 - t0) / n
    ft = f(ts)
    return np.array([np.sum(ft * np.exp(-2j * PI * w * ts)) * dt for w in np.atleast_1d(ws)])


def phase_color(z):
    """צבע לפי הזווית של מספר מרוכב (גלגל צבעים), ובהירות לפי הגודל."""
    h = (np.angle(z) / TAU) % 1
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def tone(freqs, duration, name, amps=None, rate=48000):
    """אקורד אמיתי לשמיעה: wav במטמון של הקריינות."""
    path = common.AUDIO_DIR / f"tone_{name}.wav"
    if not path.exists():
        t = np.arange(int(duration * rate)) / rate
        amps = amps or [1] * len(freqs)
        y = sum(a * np.sin(TAU * f * t) for f, a in zip(freqs, amps))
        fade = np.minimum(1, np.minimum(t / 0.05, (duration - t) / 0.3))
        y = 0.25 * y / max(len(freqs), 1) * fade
        common.AUDIO_DIR.mkdir(exist_ok=True)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
            w.writeframes((y * 32767).astype(np.int16).tobytes())
    return str(path)


CHORD = [261.63, 327.03, 392.44]  # דו-מי-סול, ביחס 4:5:6 בדיוק כמו בגרפים


# ---------- ציור ----------

def axes(x_range, y_range, width, height, x_labels=None, y_labels=None, size=22):
    ax = Axes(x_range, y_range, width=width, height=height,
              axis_config=dict(stroke_color=AXIS, stroke_width=2))
    if x_labels:
        ax.add(VGroup(*(Tex(t, font_size=size).set_color(GREY_A).next_to(ax.c2p(x, 0), DOWN, buff=0.12)
                        for x, t in x_labels)))
    if y_labels:
        ax.add(VGroup(*(Tex(t, font_size=size).set_color(GREY_A).next_to(ax.c2p(x_range[0], y), LEFT, buff=0.12)
                        for y, t in y_labels)))
    return ax


def int_labels(values):
    return [(v, str(v)) for v in values]


def stems(ax, xs, ys, color=SPEC, width=3, radius=0.05):
    group = VGroup()
    for x, y in zip(xs, ys):
        group.add(VGroup(Line(ax.c2p(x, 0), ax.c2p(x, y)).set_stroke(color, width),
                         Dot(ax.c2p(x, y), radius=radius, fill_color=color)))
    return group


def spike(ax, x, h, color=SPEC):
    return Arrow(ax.c2p(x, 0), ax.c2p(x, h), buff=0, thickness=4).set_color(color)


class TransformScene(FourierScene):
    pass


# ---------- סצנות ----------

class Title(TransformScene):
    def construct(self):
        top = axes((0, 2, 0.5), (-3, 3, 1), 11, 2.6).shift(1.8 * UP)
        sig = lambda t: np.sin(TAU * 4 * t) + np.sin(TAU * 5 * t) + np.sin(TAU * 6 * t)
        wave_mob = curve(top, sig, 0, 2, SUM_COLOR, 3, step=0.002)
        low = axes((0, 8, 1), (0, 1.3, 1), 7, 1.8, x_labels=int_labels(range(1, 8))).shift(1.8 * DOWN)
        self.say("שומעים? זה אקורד: שלושה צלילים שמתנגנים ביחד. "
                 "האוזן שומעת גל אחד מסובך, אבל היא יודעת להפריד אותו לשלושה צלילים.")
        self.play(FadeIn(top), ShowCreation(wave_mob), run_time=2)
        self.add_sound(tone(CHORD, 3.5, "chord"))
        self.wait(3.5)
        self.wait_speech()
        parts = VGroup(*(curve(top, lambda t, f=f: np.sin(TAU * f * t), 0, 2, WAVES[i], 2.5, step=0.002)
                         for i, f in enumerate((4, 5, 6))))
        self.say("והתמרת פורייה עושה בדיוק את זה, במתמטיקה: היא לוקחת גל, ומגלה אילו תדרים יש בו ובאיזו עוצמה.")
        self.play(FadeIn(low), *(TransformFromCopy(wave_mob, p) for p in parts), run_time=2)
        spikes = VGroup(*(spike(low, f, 1, WAVES[i]) for i, f in enumerate((4, 5, 6))))
        self.play(*(ReplacementTransform(p, s) for p, s in zip(parts, spikes)), run_time=2)
        self.wait_speech()
        title = heb("התמרת פורייה", 80).move_to(0.1 * UP)
        sub = heb("מהזמן אל התדר, ובחזרה", 34, GREY_A).next_to(title, DOWN, buff=0.3)
        self.say("בסרטון הקודם ראינו איך טור פורייה מפרק פונקציה מחזורית לגלים. "
                 "היום נלך צעד אחד הלאה: התמרת פורייה, הגרסה הבדידה שלה, והאלגוריתם המהיר שמריץ אותה בכל טלפון.")
        self.play(FadeIn(title, shift=0.3 * UP), FadeIn(sub))
        self.wait_speech()
        self.fade_all()


class Recap(TransformScene):
    def construct(self):
        formulas = VGroup(
            Tex(R"f(x) = \sum_{n=-\infty}^{\infty} c_n\, e^{inx}", font_size=44),
            Tex(R"c_n = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)\, e^{-inx}\, dx", font_size=44),
        ).arrange(RIGHT, buff=1.0).to_edge(UP, buff=0.5)
        self.caption("תזכורת: טור פורייה של פונקציה מחזורית",
                     say="תזכורת קצרה. פונקציה מחזורית אפשר לכתוב כסכום של גלים, אֵה בחזקת איי אן איקס, "
                         "ולכל מספר שלם אן יש מקדם, צֵה אֶן. גדי קורא לגלים האלה הרמוניות.")
        self.play(Write(formulas[0]), run_time=1.5)
        self.play(Write(formulas[1]), run_time=1.5)
        ax_f = axes((-3 * PI, 3 * PI, PI), (-1.5, 1.5, 1), 6, 2.2).move_to(3.4 * LEFT + 0.3 * DOWN)
        sq = curve(ax_f, lambda x: np.where(np.mod(x + PI, TAU) - PI >= 0, 1.0, -1.0), -3 * PI, 3 * PI,
                   F_COLOR, 3.5, step=0.02, jumps=[k * PI for k in range(-3, 4)])
        ax_c = axes((-8, 8, 1), (0, 0.8, 1), 6, 2.2, x_labels=[(n, str(n)) for n in (-7, -5, -3, 3, 5, 7)] + [(0, "0")])
        ax_c.move_to(3.4 * RIGHT + 0.3 * DOWN)
        ns = np.arange(-7, 8)
        cs = np.where(ns % 2 == 1, 2 / (PI * np.maximum(np.abs(ns), 1)), 0)
        st = stems(ax_c, ns, cs)
        self.play(FadeIn(ax_f), ShowCreation(sq))
        self.play(FadeIn(ax_c), LaggedStart(*(GrowFromEdge(s, DOWN) for s in st), lag_ratio=0.05))
        self.wait_speech()
        self.caption("פונקציה מחזורית: מקדמים רק במספרים השלמים",
                     say="שימו לב לצורה של התשובה: הפונקציה מחזורית, והמקדמים יושבים רק על המספרים השלמים, "
                         "בנפרד, כמו שיניים של מסרק. אבל מה עושים עם פונקציה שאינה מחזורית בכלל? "
                         "למשל, הבזק אחד, שקורה פעם אחת ולא חוזר.")
        self.play(Indicate(st, color=YELLOW))
        self.wait_speech()
        self.fade_all()


class Bridge(TransformScene):
    def construct(self):
        top = axes((-9, 9, 1), (0, 1.3, 1), 12, 1.6).shift(2.1 * UP)
        low = axes((-4, 4, 1), (-0.3, 1.1, 1), 11, 3.0, x_labels=int_labels(range(-3, 4))).shift(1.0 * DOWN)
        L = ValueTracker(2)

        def train():
            period = L.get_value()
            return curve(top, lambda t: rect(((t + period / 2) % period) - period / 2), -9, 9, F_COLOR, 3,
                         step=0.005, jumps=[k * period + s for k in range(-20, 21) for s in (-0.5, 0.5)])

        def spectrum():
            period = L.get_value()
            ns = np.arange(-int(4 * period), int(4 * period) + 1)
            return stems(low, ns / period, sinc(ns / period), SPEC, 2.5 if period < 6 else 1.6,
                         0.05 if period < 6 else 0.03)

        pulses = always_redraw(train)
        spec = always_redraw(spectrum)
        label = VGroup(heb("המחזור:", 30, GREY_A), DecimalNumber(2, num_decimal_places=0, font_size=34))
        label.arrange(LEFT, buff=0.2).to_corner(UR, buff=0.3)
        label[1].add_updater(lambda m: m.set_value(L.get_value()).next_to(label[0], LEFT, buff=0.2))
        self.caption("מגדילים את המחזור, והמקדמים מתקרבים זה לזה",
                     say="הנה טריק שעוזר להבין. נתחיל מפולסים שחוזרים על עצמם, ונחשב את מקדמי פורייה שלהם. "
                         "למטה, כל מקדם מצויר במקום של התדר שלו.")
        self.play(FadeIn(top), FadeIn(low), FadeIn(label))
        self.add(pulses, spec)
        self.wait_speech()
        self.say("עכשיו נרחיק את הפולסים זה מזה. המחזור גדל, והתדרים, אן חלקי המחזור, נדחסים זה לזה. "
                 "השיניים של המסרק מתקרבות.")
        for v in (4, 8, 16):
            self.play(L.animate.set_value(v), run_time=2.5)
            self.wait(0.5)
        self.wait_speech()
        envelope = curve(low, sinc, -4, 4, YELLOW, 3, step=0.01)
        self.caption("בגבול: פולס אחד, וההתמרה היא עקומה רציפה",
                     say="כשהמחזור שואף לאינסוף, נשאר פולס אחד בודד, והמקדמים כבר לא יושבים במקומות נפרדים. "
                         "הם ממלאים עקומה רציפה. כמו שגדי כותב: התמרת פורייה היא כמו טור פורייה, "
                         "רק לפונקציות על כל הישר, ועם מספר לא בן מניה של מקדמים.")
        self.play(ShowCreation(envelope), run_time=2)
        self.wait_speech()
        for m in (pulses, spec, label[1]):
            m.clear_updaters()
        self.fade_all()


class IntegralSum(TransformScene):
    def construct(self):
        ax = axes((0, 4, 1), (0, 2, 1), 6, 3).move_to(3.4 * LEFT + 0.6 * UP)
        f = lambda x: 1 + 0.6 * np.sin(1.5 * x)
        graph = curve(ax, f, 0, 4, F_COLOR, 3.5)
        S = Tex(R"\sum", font_size=120).move_to(3.4 * RIGHT + 1.4 * UP)
        I = Tex(R"\int", font_size=120).move_to(S)

        def riemann(n):
            dx = 4 / n
            return VGroup(*(Rectangle(width=ax.c2p(dx, 0)[0] - ax.c2p(0, 0)[0], height=ax.c2p(0, f((k + 0.5) * dx))[1] - ax.c2p(0, 0)[1])
                            .move_to(ax.c2p((k + 0.5) * dx, f((k + 0.5) * dx) / 2)).set_stroke(BG, 1 if n < 60 else 0)
                            .set_fill(BLUE, 0.6) for k in range(n)))

        rects = riemann(6)
        self.caption("סכום של אינסוף איברים: אינטגרל",
                     say="מה זה סכום עם מספר לא בן מניה של איברים? בעצם, כולנו מכירים: זה אינטגרל. "
                         "סוכמים אינסוף פסים דקים, ולכל אחד יש משקל זעיר.")
        self.play(FadeIn(ax), ShowCreation(graph), FadeIn(rects), Write(S))
        for n in (12, 30, 120):
            self.play(Transform(rects, riemann(n)), run_time=1.2)
        self.play(TransformMatchingShapes(S, I))
        self.wait_speech()
        self.say("גדי מזכיר שסימן האינטגרל הוא בסך הכל אֶס מוארך, מהמילה סאם, סכום. או לפחות זה הצ'יזבט.")
        self.play(I.animate.stretch(1.3, 1))
        self.wait_speech()
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None

        head = VGroup(heb("טור פורייה", 34, SUM_COLOR), heb("התמרת פורייה", 34, SPEC)).arrange(RIGHT, buff=3.2)
        head.to_edge(UP, buff=0.5)
        analysis = VGroup(
            Tex(R"c_n = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x)\, e^{-inx}\, dx", font_size=36),
            Tex(R"\hat f(\omega) = \int_{-\infty}^{\infty} f(t)\, e^{-2\pi i \omega t}\, dt", font_size=36),
        )
        synthesis = VGroup(
            Tex(R"f(x) = \sum_{n=-\infty}^{\infty} c_n\, e^{inx}", font_size=36),
            Tex(R"f(t) = \int_{-\infty}^{\infty} \hat f(\omega)\, e^{2\pi i \omega t}\, d\omega", font_size=36),
        )
        for row in (analysis, synthesis):
            for m, h in zip(row, head):
                m.match_x(h)
        analysis.shift(1.3 * UP)
        synthesis.shift(0.6 * DOWN)
        tags = VGroup(heb("אנליזה: פירוק", 28, GREY_A).next_to(analysis, UP, buff=0.25),
                      heb("סינתזה: בנייה מחדש", 28, GREY_A).next_to(synthesis, UP, buff=0.25))
        tags[0].match_x(ORIGIN)
        tags[1].match_x(ORIGIN)
        self.caption("שתי משוואות: אנליזה וסינתזה",
                     say="אז הנה ההגדרה. משמאל לימין, כמו שגדי מסדר אותן: בטור פורייה, משוואת האנליזה מפרקת את הפונקציה למקדמים, "
                         "ומשוואת הסינתזה בונה אותה מחדש. בהתמרה, בדיוק אותו דבר: "
                         "במקום מקדם לכל מספר שלם, יש ערך לכל תדר אומגה, ובמקום סכום, אינטגרל.")
        self.play(FadeIn(head))
        self.play(FadeIn(tags[0]), Write(analysis[0]), Write(analysis[1]), run_time=2)
        self.play(FadeIn(tags[1]), Write(synthesis[0]), Write(synthesis[1]), run_time=2)
        self.wait_speech()
        self.caption("t הוא זמן, ω הוא תדר",
                     say="ושימו לב לשמות. אֶף מקבלת את טי, זמן, וההתמרה מקבלת את אומגה, תדר. "
                         "ההתמרה מעבירה אותנו, כמו שאוהבים לומר, ממרחב הזמן אל מרחב התדר.")
        self.play(Indicate(analysis[1][R"\omega"], color=YELLOW), Indicate(analysis[1]["t"], color=YELLOW))
        self.wait_speech()
        self.fade_all()


class Detector(TransformScene):
    def construct(self):
        env = lambda t: np.exp(-PI * (t / 3.5) ** 2)
        f = lambda t: (np.cos(TAU * 1.0 * t) + 0.6 * np.cos(TAU * 2.5 * t)) * env(t)
        top = axes((-6, 6, 1), (-1.7, 1.7, 1), 7, 1.6).move_to(3.2 * LEFT + 2.5 * UP)
        mid = axes((-6, 6, 1), (-1.7, 1.7, 1), 7, 1.6).move_to(3.2 * LEFT + 0.4 * UP)
        spec = axes((0, 3.5, 0.5), (-0.4, 2.0, 1), 5.4, 3.0,
                    x_labels=[(0.5 * k, f"{0.5 * k:g}") for k in range(1, 8)]).move_to(3.9 * RIGHT + 0.9 * UP)
        f_graph = curve(top, f, -6, 6, F_COLOR, 3)
        w = ValueTracker(0.3)
        probe = always_redraw(lambda: curve(top, lambda t: np.cos(TAU * w.get_value() * t), -6, 6, BLUE, 1.8).set_stroke(opacity=0.7))
        prod = lambda t: f(t) * np.cos(TAU * w.get_value() * t)
        area = always_redraw(lambda: signed_area(mid, prod, -6, 6, step=0.01))
        pcurve = always_redraw(lambda: curve(mid, prod, -6, 6, WHITE, 2))
        ws = np.linspace(0, 3.5, 351)
        values = ft_numeric(f, ws).real

        def traced():
            k = int(np.searchsorted(ws, w.get_value()))
            k = max(k, 2)
            return VMobject().set_points_as_corners(spec.c2p(ws[:k], values[:k])).set_stroke(SPEC, 4)

        trace = always_redraw(traced)
        point = always_redraw(lambda: Dot(spec.c2p(w.get_value(), np.interp(w.get_value(), ws, values)), fill_color=SPEC))
        w_lab = VGroup(Tex(R"\omega =", font_size=34), DecimalNumber(0.3, num_decimal_places=2, font_size=34))
        w_lab.arrange(RIGHT).next_to(spec, UP, buff=0.3)
        w_lab[1].add_updater(lambda m: m.set_value(w.get_value()).next_to(w_lab[0], RIGHT))
        omega_axis = Tex(R"\omega", font_size=30).next_to(spec.x_axis, RIGHT, buff=0.1)
        self.caption("גלאי: כופלים בגל בתדר ω ומודדים את השטח",
                     say="איך הנוסחה מוצאת תדרים? בדיוק כמו הגלאים מהסרטון הקודם. הנה אות עם שני רכיבים. "
                         "כופלים אותו בגל בודק בתדר אומגה, ומודדים את השטח מתחת למכפלה. "
                         "אחר כך מזיזים את אומגה, ומציירים את השטח כפונקציה שלו.")
        self.play(FadeIn(top), ShowCreation(f_graph), FadeIn(mid), FadeIn(spec), FadeIn(omega_axis))
        self.add(probe, area, pcurve, trace, point)
        self.play(FadeIn(w_lab))
        self.wait_speech()
        self.say("כשהגל הבודק לא מתאים לאף רכיב, הירוק והאדום מתקזזים, וההתמרה קרובה לאפס.")
        self.play(w.animate.set_value(0.65), run_time=3, rate_func=linear)
        self.wait_speech()
        self.say("אבל בתדר אחת, הגל הבודק עולה ויורד בדיוק יחד עם האות, והמכפלה כמעט כולה חיובית. נוצרת פסגה.")
        self.play(w.animate.set_value(1.0), run_time=2.5, rate_func=linear)
        self.wait_speech()
        self.say("ממשיכים לסרוק. שוב קיזוז, ובתדר שתיים וחצי פסגה שנייה, נמוכה יותר, כי הרכיב הזה חלש יותר.")
        self.play(w.animate.set_value(2.5), run_time=5, rate_func=linear)
        self.wait_speech()
        self.play(w.animate.set_value(3.5), run_time=2, rate_func=linear)
        self.caption("הקוסינוס מוצא את הגלים שבפאזה; הסינוס את השאר",
                     say="ולמה באמת יש בנוסחה אקספוננט מרוכב? כי גל יכול להיות מוזז. "
                         "הקוסינוס הבודק מוצא רכיבים שמתחילים בפסגה, והסינוס את אלה שמתחילים באפס. "
                         "אֵה בחזקת מינוס שתיים פאי איי אומגה טי עושה את שניהם בבת אחת: "
                         "החלק הממשי הוא קוסינוס, והחלק המדומה הוא סינוס.")
        self.wait_speech()
        for m in (probe, area, pcurve, trace, point, w_lab[1]):
            m.clear_updaters()
        self.fade_all()


class Equalizer(TransformScene):
    def construct(self):
        amps = [ValueTracker(1), ValueTracker(1), ValueTracker(1)]
        top = axes((0, 2, 0.5), (-3, 3, 1), 11, 2.6).shift(1.7 * UP)
        low = axes((0, 8, 1), (0, 1.3, 1), 7, 2.0, x_labels=int_labels(range(1, 8))).shift(1.5 * DOWN)
        sig = lambda t: sum(a.get_value() * np.sin(TAU * f * t) for a, f in zip(amps, (4, 5, 6)))
        wave_mob = always_redraw(lambda: curve(top, sig, 0, 2, SUM_COLOR, 3, step=0.002))
        spikes = always_redraw(lambda: VGroup(*(spike(low, f, max(a.get_value(), 0.02), WAVES[i])
                                                for i, (a, f) in enumerate(zip(amps, (4, 5, 6))))))
        t_lab = heb("זמן", 28, GREY_A).next_to(top, LEFT, buff=0.2)
        w_lab = heb("תדר", 28, GREY_A).next_to(low, LEFT, buff=0.2)
        self.caption("אקוולייזר: משנים את ההתמרה, וחוזרים",
                     say="גדי מציע אינטואיציה מוכרת: אקוולייזר. כלים שונים מנגנים בתדרים שונים. "
                         "האקוולייזר מגביר ומחליש תדרים, ואפשר לחשוב עליו כך: מבצעים התמרת פורייה, "
                         "משנים חלק מהערכים, ומתמירים בחזרה.")
        self.play(FadeIn(top), FadeIn(low), FadeIn(t_lab), FadeIn(w_lab))
        self.add(wave_mob, spikes)
        self.wait_speech()
        self.say("נשמע שוב את האקורד, ועכשיו נוריד לגמרי את הצליל האמצעי.")
        self.wait_speech()
        self.add_sound(tone(CHORD, 2.5, "chord_short"))
        self.wait(2.7)
        self.play(amps[1].animate.set_value(0), run_time=2)
        self.add_sound(tone([CHORD[0], CHORD[2]], 2.5, "chord_no_mid"))
        self.wait(2.7)
        self.say("הגל השתנה לגמרי, אבל במרחב התדר השינוי היה פשוט: מחקנו קו אחד. "
                 "ועכשיו נחזיר אותו, ונחזק את הצליל הנמוך.")
        self.play(amps[1].animate.set_value(1), amps[0].animate.set_value(1.25), amps[2].animate.set_value(0.4), run_time=3)
        self.wait_speech()
        self.caption("ספקטרוסקופיה: התדרים הם חתימה של החומר",
                     say="והרעיון לא מוגבל לצלילים. בספקטרוסקופיה מפרקים את האור שחומר פולט לתדרים, "
                         "וכל חומר פולט תדרים משלו, כמו חתימה. כך יודעים ממה עשויים כוכבים רחוקים.")
        self.wait_speech()
        for m in (wave_mob, spikes):
            m.clear_updaters()
        self.fade_all()


class Beats(TransformScene):
    def construct(self):
        top = axes((0, 4, 1), (-1.3, 1.3, 1), 11, 1.3).shift(2.6 * UP)
        mid = axes((0, 4, 1), (-1.3, 1.3, 1), 11, 1.3).shift(1.1 * UP)
        bot = axes((0, 4, 1), (-2.3, 2.3, 1), 11, 2.0).shift(1.1 * DOWN)
        a = lambda t: np.sin(TAU * 5 * t)
        b = lambda t: np.sin(TAU * 5.5 * t)
        ga = curve(top, a, 0, 4, BLUE, 2.5, step=0.002)
        gb = curve(mid, b, 0, 4, GREEN, 2.5, step=0.002)
        gs = curve(bot, lambda t: a(t) + b(t), 0, 4, SUM_COLOR, 3, step=0.002)
        self.caption("איפה מתחבא המידע על התדרים?",
                     say="גדי מודה שבהתחלה זה נראה לו מוזר: איך כל המידע על התדרים מתחבא בתוך הפונקציה? "
                         "התשובה היא התאבכות. כל גל עולה ויורד בקצב משלו, ובכל רגע הערך של הפונקציה הוא הערבוב שלהם.")
        self.play(FadeIn(top), ShowCreation(ga), FadeIn(mid), ShowCreation(gb))
        self.play(FadeIn(bot), TransformFromCopy(VGroup(ga, gb), gs), run_time=2)
        self.wait_speech()
        boxes = VGroup(*(Rectangle(width=bot.c2p(0.35, 0)[0] - bot.c2p(0, 0)[0], height=4.6)
                         .move_to(top.c2p(x, 0) + (top.c2p(0, 0)[1] - (top.c2p(0, 0)[1] + bot.c2p(0, 0)[1]) / 2) * DOWN)
                         .set_stroke(c, 2).set_fill(c, 0.08) for x, c in ((0.0 + 0.17, GREEN), (1.0, RED), (2.0 + 0.0, GREEN))))
        self.caption("כשהגלים מסונכרנים הם מתחזקים, וכשהם הפוכים, מתבטלים",
                     say="כשהשניים בשלב, הם מחזקים זה את זה. חצי שנייה אחר כך, אחד בפסגה והשני בשפל, והם מבטלים זה את זה. "
                         "לזה קוראים פעימות. כל רגע מלמד אותנו משהו על הערבוב, ומכל הרגעים ביחד אפשר לשחזר את כל התדרים.")
        self.play(LaggedStart(*(FadeIn(bx) for bx in boxes), lag_ratio=0.5))
        self.wait_speech()
        self.fade_all()


class Delta(TransformScene):
    def construct(self):
        top = axes((-8, 8, 2), (-1.3, 1.3, 1), 11, 1.6).shift(2.3 * UP)
        low = axes((0, 2, 0.5), (-1, 6, 1), 8, 3.3,
                   x_labels=[(0.5, "0.5"), (1, "1"), (1.5, "1.5")]).shift(1.0 * DOWN)
        T = ValueTracker(2)
        window = lambda t: np.where(np.abs(t) < T.get_value() / 2, 1.0, 0.0)
        sig = always_redraw(lambda: curve(top, lambda t: np.cos(TAU * t) * window(t), -8, 8, F_COLOR, 2.5, step=0.004))
        ws = np.linspace(0, 2, 801)

        def spec():
            tt = T.get_value()
            vals = tt / 2 * (sinc((ws - 1) * tt) + sinc((ws + 1) * tt))
            return VMobject().set_points_as_corners(low.c2p(ws, np.clip(vals, -1, 6.5))).set_stroke(SPEC, 3.5)

        sp = always_redraw(spec)
        self.caption("גל טהור, שנמשך לנצח: האינטגרל מתבדר",
                     say="ומה ההתמרה של גל טהור, עם תדר אחד בדיוק? אם מציבים בנוסחה, בתדר אחת מקבלים "
                         "אינטגרל של אחת על כל הישר, והוא מתבדר. משהו בהגדרות הרגילות לא עובד.")
        self.play(FadeIn(top), FadeIn(low))
        self.add(sig, sp)
        self.wait_speech()
        self.caption("חותכים חלון: ככל שהוא ארוך, הפסגה גבוהה וצרה",
                     say="נבין מה קורה בעזרת חלון: ניקח את הגל רק לזמן מוגבל. ככל שהחלון ארוך יותר, "
                         "הפסגה בתדר אחת גבוהה יותר וצרה יותר. אבל השטח שמתחתיה נשאר קבוע.")
        for v in (4, 8, 16):
            self.play(T.animate.set_value(v), run_time=2.5)
            self.wait(0.4)
        self.wait_speech()
        delta = Tex(R"\delta(\omega - 1)", font_size=40).set_color(SPEC).next_to(low.c2p(1, 5.5), RIGHT, buff=0.2)
        self.caption("בגבול: הדלתא של דיראק",
                     say="בגבול מקבלים פונקציה שהיא אפס בכל מקום חוץ מנקודה אחת, ובכל זאת השטח שלה אחת: הדלתא של דיראק. "
                         "פונקציה רגילה כזו לא קיימת, וצריך להכליל את המושג, למשל למידות או להתפלגויות.")
        self.play(Write(delta))
        self.wait_speech()
        self.say("אבל גדי מרגיע: יש פורמליזם. הוא עובד. כולם שמחים. המתמטיקה ניצלה שוב.")
        self.wait_speech()
        for m in (sig, sp):
            m.clear_updaters()
        self.fade_all()


class Uncertainty(TransformScene):
    def construct(self):
        a = ValueTracker(1)
        left = axes((-3, 3, 1), (0, 2.2, 1), 5.8, 3.2, x_labels=int_labels((-2, -1, 1, 2))).move_to(3.5 * LEFT + 0.6 * UP)
        right = axes((-3, 3, 1), (0, 2.2, 1), 5.8, 3.2, x_labels=int_labels((-2, -1, 1, 2))).move_to(3.5 * RIGHT + 0.6 * UP)
        g = always_redraw(lambda: curve(left, lambda t: np.exp(-PI * (a.get_value() * t) ** 2), -3, 3, F_COLOR, 4, step=0.01))
        G = always_redraw(lambda: curve(right, lambda w: np.exp(-PI * (w / a.get_value()) ** 2) / a.get_value(), -3, 3,
                                        SPEC, 4, step=0.01))
        labels = VGroup(heb("זמן", 30, F_COLOR).next_to(left, UP, buff=0.2), heb("תדר", 30, SPEC).next_to(right, UP, buff=0.2))
        self.caption("הגאוסיאן: ההתמרה שלו היא שוב גאוסיאן",
                     say="עוד תכונה יפה. פעמון גאוס, אֵה בחזקת מינוס פאי טי בריבוע, הוא ההתמרה של עצמו.")
        self.play(FadeIn(left), FadeIn(right), FadeIn(labels))
        self.add(g, G)
        self.wait_speech()
        self.caption("מכווצים בזמן, ומתרחבים בתדר",
                     say="ועכשיו נכווץ אותו בזמן. ההתמרה מתרחבת. ונמתח אותו בזמן, וההתמרה מתכווצת. "
                         "אי אפשר להיות צר בשני העולמות. צליל קצר, כמו נקישה, מכיל המון תדרים, "
                         "וצליל ארוך ונקי הוא כמעט תדר אחד. זה הגרעין המתמטי של עקרון אי הוודאות של הייזנברג.")
        self.play(a.animate.set_value(2.2), run_time=3)
        self.play(a.animate.set_value(0.5), run_time=3)
        self.play(a.animate.set_value(1), run_time=2)
        self.wait_speech()
        for m in (g, G):
            m.clear_updaters()
        self.fade_all()


class Diagram(TransformScene):
    def construct(self):
        sigma = 1.1
        f = lambda t: np.exp(-t ** 2 / (2 * sigma ** 2))
        F = lambda w: sigma * np.sqrt(TAU) * np.exp(-2 * PI ** 2 * sigma ** 2 * w ** 2)
        g = lambda x: sum(f(x + TAU * k) for k in range(-6, 7))
        pos = {"f": np.array([-3.6, 1.7, 0]), "F": np.array([3.6, 1.7, 0]),
               "g": np.array([-3.6, -1.5, 0]), "G": np.array([3.6, -1.5, 0])}
        ax_f = axes((-3 * PI, 3 * PI, PI), (0, 1.6, 1), 5.4, 1.5).move_to(pos["f"])
        ax_g = axes((-3 * PI, 3 * PI, PI), (0, 1.6, 1), 5.4, 1.5).move_to(pos["g"])
        ax_F = axes((-0.8, 0.8, 0.2), (0, 3.0, 1), 5.4, 1.5).move_to(pos["F"])
        ax_G = axes((-5, 5, 1), (0, 0.5, 1), 5.4, 1.5).move_to(pos["G"])
        names = VGroup(Tex("f", font_size=40).next_to(ax_f, UP, buff=0.1), Tex(R"\hat f", font_size=40).next_to(ax_F, UP, buff=0.1),
                       Tex("g", font_size=40).next_to(ax_g, DOWN, buff=0.1), Tex(R"\hat g", font_size=40).next_to(ax_G, DOWN, buff=0.1))
        arrows = VGroup(
            Arrow(ax_f.get_right(), ax_F.get_left()).set_color(GREY_A),
            Arrow(ax_g.get_right(), ax_G.get_left()).set_color(GREY_A),
            Arrow(ax_f.get_bottom(), ax_g.get_top() + 0.1 * UP).set_color(TEAL),
            Arrow(ax_F.get_bottom(), ax_G.get_top() + 0.1 * UP).set_color(TEAL),
        )
        arrow_tags = VGroup(heb("התמרה", 24, GREY_A).next_to(arrows[0], UP, buff=0.05),
                            heb("טור", 24, GREY_A).next_to(arrows[1], UP, buff=0.05),
                            heb("סכימה", 26, TEAL).next_to(arrows[2], LEFT, buff=0.1),
                            heb("דגימה", 26, TEAL).next_to(arrows[3], RIGHT, buff=0.1))
        self.caption("שתי ההתמרות, ומה מחבר ביניהן",
                     say="ועכשיו לרעיון שגדי מסיים בו את הפוסט. יש לנו שני עולמות: פונקציה על כל הישר וההתמרה שלה, "
                         "ופונקציה מחזורית וטור הפורייה שלה. האם אפשר לעבור ביניהם, כך שלא משנה באיזה מסלול הולכים?")
        self.play(FadeIn(VGroup(ax_f, ax_F, ax_g, ax_G, names)))
        self.play(GrowArrow(arrows[0]), GrowArrow(arrows[1]), FadeIn(arrow_tags[:2]))
        self.wait_speech()

        fg = curve(ax_f, f, -3 * PI, 3 * PI, F_COLOR, 3)
        self.play(ShowCreation(fg))
        copies = VGroup(*(curve(ax_g, lambda x, k=k: f(x + TAU * k), -3 * PI, 3 * PI, F_COLOR, 2).set_stroke(opacity=0.5)
                          for k in range(-3, 4)))
        gg = curve(ax_g, g, -3 * PI, 3 * PI, SUM_COLOR, 3.5)
        self.caption("סכימה: מחברים את כל העותקים המוזזים ב-2π",
                     say="מהעולם הרציף לעולם המחזורי עוברים בסכימה: לוקחים את הפונקציה, מזיזים אותה בכל הכפולות של שתיים פאי, "
                         "ומחברים את כל העותקים. מה שיוצא חייב להיות מחזורי.")
        self.play(GrowArrow(arrows[2]), FadeIn(arrow_tags[2]))
        self.play(LaggedStart(*(TransformFromCopy(fg, c) for c in copies), lag_ratio=0.15, run_time=2.5))
        self.play(ReplacementTransform(copies, gg), run_time=1.5)
        self.wait_speech()

        Fg = curve(ax_F, F, -0.8, 0.8, SPEC, 3.5)
        self.play(ShowCreation(Fg))
        ns = np.arange(-5, 6)
        samples = stems(ax_G, ns, F(ns / TAU) / TAU, YELLOW, 3, 0.06)
        coeffs = [np.sum(g(xs) * np.cos(n * xs)) * (TAU / 4000) / TAU
                  for xs in [np.linspace(-PI, PI, 4000, endpoint=False)] for n in ns]
        bars_g = VGroup(*(Rectangle(width=0.25, height=max(c, 0.001) * (ax_G.c2p(0, 1)[1] - ax_G.c2p(0, 0)[1]))
                          .move_to(ax_G.c2p(n, c / 2)).set_stroke(width=0).set_fill(SUM_COLOR, 0.55) for n, c in zip(ns, coeffs)))
        self.caption("דגימה: לוקחים את ההתמרה רק בנקודות n חלקי 2π",
                     say="ומהצד השני: מההתמרה הרציפה עוברים לטור בדגימה. לוקחים את הערכים שלה רק בנקודות בודדות, "
                         "אן חלקי שתיים פאי, ומחלקים בשתיים פאי.")
        self.play(GrowArrow(arrows[3]), FadeIn(arrow_tags[3]))
        self.play(LaggedStart(*(GrowFromEdge(s, DOWN) for s in samples), lag_ratio=0.08, run_time=2))
        self.wait_speech()
        self.caption("שני המסלולים נותנים בדיוק אותם מספרים",
                     say="ועכשיו המבחן. נחשב ישירות את מקדמי הפורייה של הפונקציה המחזורית, ונצייר אותם כעמודות. "
                         "והם נוחתים בדיוק על הנקודות שדגמנו. קסם! קיבלנו, כמו שגדי אומר, דיאגרמה קומוטטיבית. "
                         "לשוויון הזה קוראים נוסחת הסכימה של פואסון.")
        self.play(LaggedStart(*(GrowFromEdge(b_, DOWN) for b_ in bars_g), lag_ratio=0.08, run_time=2))
        self.play(*(Flash(s[1].get_center(), color=YELLOW, flash_radius=0.2) for s in samples[3:8]))
        self.wait_speech()
        self.fade_all()


class Discrete(TransformScene):
    def construct(self):
        N = 8
        ax = axes((0, 16, 1), (-1.4, 1.4, 1), 11, 2.2, x_labels=int_labels(range(0, 16, 2))).shift(1.9 * UP)
        vals = 0.8 * np.cos(TAU * np.arange(16) / 8) + 0.4 * np.sin(TAU * 3 * np.arange(16) / 8)
        st = stems(ax, np.arange(16), vals, F_COLOR, 3, 0.07)
        divider = DashedLine(ax.c2p(7.5, -1.3), ax.c2p(7.5, 1.3)).set_stroke(GREY_B, 2)
        self.caption("במחשב: רשימה סופית של מספרים",
                     say="עד עכשיו דיברנו על פונקציות על הישר הממשי. אבל במחשב שומרים רשימה של מספרים: "
                         "דגימות של צליל, פיקסלים של תמונה. גדי עובר לפונקציות על המספרים השלמים, "
                         "והמעניינות ביותר הן המחזוריות. אז כל הפונקציה היא רשימה סופית של אן מספרים, שחוזרת על עצמה.")
        self.play(FadeIn(ax), LaggedStart(*(GrowFromEdge(s, DOWN) for s in st), lag_ratio=0.05))
        self.play(ShowCreation(divider))
        self.wait_speech()

        circle_center = np.array([-3.5, -1.5, 0])
        r = 1.3
        circle = Circle(radius=r).move_to(circle_center).set_stroke(GREY_B, 1.5)
        roots = VGroup(*(Dot(circle_center + r * direction(TAU * k / N), radius=0.07, fill_color=YELLOW) for k in range(N)))
        labels = VGroup(*(Tex(rf"\omega^{k}" if k > 1 else (R"\omega" if k == 1 else "1"), font_size=24)
                          .move_to(circle_center + (r + 0.35) * direction(TAU * k / N)) for k in range(N)))
        formula = Tex(R"\omega = e^{2\pi i / N}, \qquad \omega^N = 1", font_size=38).move_to(3.4 * RIGHT + 0.0 * DOWN)
        self.caption("אבני הבניין: שורשי היחידה",
                     say="ומה אבני הבניין? גלים שחוזרים על עצמם אחרי אן צעדים בדיוק. "
                         "אלה החזקות של אומגה, אֵה בחזקת שתיים פאי איי חלקי אן: נקודות שמחלקות את מעגל היחידה לאן חלקים שווים. "
                         "קוראים להן שורשי היחידה, כי כל אחת מהן בחזקת אן שווה אחת.")
        self.play(ShowCreation(circle), LaggedStart(*(GrowFromCenter(d) for d in roots), lag_ratio=0.1), FadeIn(labels))
        self.play(Write(formula))
        self.wait_speech()

        k_tr = ValueTracker(1)
        step = ValueTracker(0)
        walker = always_redraw(lambda: Arrow(circle_center, circle_center + r * direction(TAU * k_tr.get_value() * step.get_value() / N),
                                             buff=0).set_color(SPEC))
        k_lab = VGroup(Tex("k =", font_size=36), Integer(1, font_size=36)).arrange(RIGHT).next_to(formula, DOWN, buff=0.6)
        self.caption("הגל ה-k: מתקדמים k צעדים על המעגל בכל פעם",
                     say="הגל מספר קֵי הולך על הנקודות האלה בצעדים של קֵי: קֵי שווה אחת מתקדם לאט, "
                         "קֵי שווה שלוש מדלג שלוש נקודות בכל צעד. אחרי אן צעדים כולם חוזרים להתחלה.")
        self.add(walker)
        self.play(FadeIn(k_lab))
        for k in (1, 3):
            k_tr.set_value(k)
            k_lab[1].set_value(k)
            step.set_value(0)
            self.play(step.animate.set_value(N), run_time=4, rate_func=lambda t: np.floor(t * N + 0.0001) / N)
            self.wait(0.4)
        self.wait_speech()
        walker.clear_updaters()
        self.fade_all()


class RootsSum(TransformScene):
    def construct(self):
        N = 8
        self.caption("למה הגלים ניצבים: סכום החזקות של שורש יחידה",
                     say="כמו בטורים, כל העניין עובד כי הגלים האלה ניצבים זה לזה. וכאן ההוכחה ממש יפה. "
                         "המכפלה הפנימית של שני גלים שונים היא סכום של כל החזקות של שורש יחידה אחד. בואו נחבר אותן, חץ אחרי חץ.")
        origin = np.array([-2.5, -0.6, 0])
        for k, place in ((1, np.array([-4.5, 0.2, 0])), (2, np.array([-0.4, 0.2, 0])), (3, np.array([3.8, 0.2, 0]))):
            p = place.copy()
            chain = VGroup()
            for j in range(N):
                q = p + 0.9 * direction(TAU * k * j / N)
                chain.add(Arrow(p, q, buff=0, thickness=3).set_color(interpolate_color(BLUE, GREEN, j / N)))
                p = q
            lab = Tex(rf"\sum_{{j=0}}^{{7}} \omega^{{{k}j}} = 0", font_size=34).next_to(chain, DOWN, buff=0.35)
            self.play(LaggedStart(*(GrowArrow(a) for a in chain), lag_ratio=0.3, run_time=3))
            self.play(Write(lab), Flash(place, color=YELLOW, flash_radius=0.3))
        self.wait_speech()
        self.caption("הסכום חוזר לנקודת ההתחלה: אפס",
                     say="בכל פעם החצים מסתובבים וחוזרים בדיוק לנקודת ההתחלה, אז הסכום הוא אפס. "
                         "גדי מראה את זה עם הנוסחה לסכום של סדרה הנדסית: במונה יש אומגה בחזקת אן פחות אחת, וזה אפס. "
                         "ורק כשהגלים זהים, כל החצים מצביעים לאותו כיוון, והסכום הוא אן.")
        geo = Tex(R"\sum_{j=0}^{N-1} q^j = \frac{q^N - 1}{q - 1} = 0 \quad (q^N = 1)", font_size=38).to_edge(UP, buff=0.4)
        self.play(Write(geo))
        self.wait_speech()
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None

        cell = 0.42
        W = VGroup()
        for n in range(N):
            for k in range(N):
                z = np.exp(2j * PI * n * k / N)
                sq = Square(cell).set_stroke(BG, 1).set_fill(phase_color(z), 1)
                W.add(sq.move_to(np.array([k * cell, -n * cell, 0])))
        W.move_to(3.6 * LEFT + 0.4 * UP)
        w_lab = Tex(R"W_{nk} = \omega^{nk}", font_size=36).next_to(W, UP, buff=0.3)
        wheel = VGroup(*(Dot(np.array([0, 0, 0]) + 0.35 * direction(TAU * j / 24), radius=0.06,
                             fill_color=phase_color(np.exp(1j * TAU * j / 24))) for j in range(24))).next_to(W, LEFT, buff=0.4)
        WW = VGroup()
        for n in range(N):
            for k in range(N):
                sq = Square(cell).set_stroke(GREY_D, 1).set_fill(YELLOW if n == k else BG, 0.85 if n == k else 1)
                WW.add(VGroup(sq, Tex("8" if n == k else "0", font_size=20).set_color(BLACK if n == k else GREY_B))
                       .move_to(np.array([k * cell, -n * cell, 0])))
        WW.move_to(3.0 * RIGHT + 0.4 * UP)
        ww_lab = Tex(R"W\, W^{*} = N\, I", font_size=36).next_to(WW, UP, buff=0.3)
        inv = Tex(R"b_k = \frac{1}{N} \sum_{n=0}^{N-1} a_n\, \omega^{-kn}", font_size=40).move_to(1.9 * DOWN)
        self.caption("מטריצת פורייה: כל השורשים, בכל החזקות",
                     say="את כל ההתמרה אפשר לכתוב כמטריצה, מטריצת ונדרמונדה של שורשי היחידה. "
                         "כאן כל משבצת צבועה לפי הזווית של המספר המרוכב שבה.")
        self.play(FadeIn(wheel), LaggedStart(*(FadeIn(c) for c in W), lag_ratio=0.005, run_time=2), Write(w_lab))
        self.wait_speech()
        self.caption("ההופכית כמעט בחינם: הצמודה חלקי N",
                     say="ובזכות הסכומים שראינו, כשכופלים את המטריצה בצמודה שלה מקבלים אן באלכסון ואפסים בכל השאר. "
                         "כלומר ההופכית שלה היא פשוט הצמודה חלקי אן. וזו בדיוק משוואת האנליזה של ההתמרה הבדידה.")
        self.play(LaggedStart(*(FadeIn(c) for c in WW), lag_ratio=0.005, run_time=2), Write(ww_lab))
        self.play(Write(inv))
        self.wait_speech()
        self.fade_all()


class DFTExample(TransformScene):
    def construct(self):
        N = 16
        n = np.arange(N)
        a = 0.6 + np.cos(TAU * 2 * n / N) + 0.7 * np.sin(TAU * 5 * n / N)
        b = np.fft.fft(a) / N
        top = axes((-0.5, 16, 1), (-2.2, 2.6, 1), 11, 2.4, x_labels=int_labels(range(0, 16, 3))).shift(1.9 * UP)
        low = axes((-0.5, 16, 1), (0, 0.7, 1), 11, 2.4, x_labels=int_labels(range(0, 16))).shift(1.3 * DOWN)
        st = stems(top, n, a, F_COLOR, 3, 0.07)
        self.caption("שש עשרה דגימות של אות",
                     say="נראה דוגמה. שש עשרה דגימות של אות. במבט ראשון זה נראה כמו רעש.")
        self.play(FadeIn(top), LaggedStart(*(GrowFromEdge(s, DOWN) for s in st), lag_ratio=0.05))
        self.wait_speech()
        mags = np.abs(b)
        bs = VGroup(*(Rectangle(width=0.35, height=max(m_ * (low.c2p(0, 1)[1] - low.c2p(0, 0)[1]), 0.005))
                      .move_to(low.c2p(k, m_ / 2)).set_stroke(width=0).set_fill(phase_color(b[k]) if m_ > 0.01 else GREY_D, 0.95)
                      for k, m_ in enumerate(mags)))
        self.caption("ההתמרה: ממוצע, תדר 2, תדר 5, ותמונות ראי",
                     say="ואחרי ההתמרה הכל ברור. עמודה באפס: הממוצע. עמודות בתדר שתיים ובתדר חמש: שני הגלים. "
                         "ולמה יש גם עמודות בארבע עשרה ובאחת עשרה? כי האות ממשי, ותדר מינוס שתיים, "
                         "שהוא תדר ארבע עשרה במעגל של שש עשרה, חייב לבוא יחד עם תדר שתיים. הצבע מראה את הפאזה.")
        self.play(FadeIn(low), LaggedStart(*(GrowFromEdge(r, DOWN) for r in bs), lag_ratio=0.05, run_time=2))
        self.wait_speech()
        recon = VGroup()
        for k in (0, 2, 14, 5, 11):
            recon.add(bs[k].copy())
        self.caption("וסינתזה: בונים בחזרה מחמש עמודות",
                     say="ובכיוון ההפוך, משוואת הסינתזה: מחברים את הגלים של חמש העמודות האלה, ומקבלים בחזרה בדיוק את שש עשרה הדגימות.")
        rebuilt = curve(top, lambda x: 0.6 + np.cos(TAU * 2 * x / N) + 0.7 * np.sin(TAU * 5 * x / N), -0.3, 15.3, SUM_COLOR, 2.5, step=0.01)
        self.play(*(Indicate(bs[k], color=YELLOW) for k in (0, 2, 14, 5, 11)))
        self.play(ShowCreation(rebuilt), run_time=2.5)
        self.wait_speech()
        self.fade_all()


class Convolution(TransformScene):
    def construct(self):
        prod = Tex(R"(1 + 2x + 3x^2)(4 + 5x + 6x^2) = 4 + 13x + 28x^2 + 27x^3 + 18x^4", font_size=40).to_edge(UP, buff=0.5)
        rule = Tex(R"c_n = \sum_{k} a_k\, b_{n-k}", font_size=44).next_to(prod, DOWN, buff=0.4)
        self.caption("כפל פולינומים: בוחרים איבר מכל סוגריים",
                     say="עכשיו לשימוש הראשון, ולמושג חשוב: קונבולוציה. גדי מציג אותו דרך כפל פולינומים. "
                         "כשפותחים סוגריים, המקדם של איקס בריבוע מגיע מכל הזוגות שהחזקות שלהם מסתכמות לשתיים: "
                         "אחת כפול שש, שתיים כפול חמש, ושלוש כפול ארבע. ביחד, עשרים ושמונה.")
        self.play(Write(prod), run_time=2.5)
        self.play(Write(rule))
        self.wait_speech()

        a = [1, 2, 3]
        b = [4, 5, 6]
        size = 0.75
        y_a, y_b = -0.5, 0.5

        def box(v, color, x, y):
            return VGroup(Square(size).set_stroke(color, 2).set_fill(color, 0.15), Tex(str(v), font_size=36)).move_to([x, y, 0])

        row_a = VGroup(*(box(v, BLUE, (i - 1) * size, y_a) for i, v in enumerate(a)))
        shift_t = ValueTracker(0)

        def row_b():
            s = shift_t.get_value()
            return VGroup(*(box(v, GREEN, (-i - 1 + s) * size, y_b) for i, v in enumerate(b)))

        top_row = always_redraw(row_b)
        result = VGroup()
        c_vals = [4, 13, 28, 27, 18]
        self.caption("הציור של גדי: הופכים את b ומחליקים",
                     say="והנה הציור שגדי אוהב. כותבים את המקדמים של הפולינום השני בסדר הפוך מעל הראשון, "
                         "כופלים כל זוג שעומד זה מעל זה, ומחברים. ואז מזיזים צעד אחד ימינה, ושוב. כל צעד נותן מקדם אחד.")
        self.play(FadeIn(row_a), FadeIn(top_row))
        for n_ in range(5):
            self.play(shift_t.animate.set_value(n_), run_time=0.8)
            pairs = [(k, n_ - k) for k in range(3) if 0 <= n_ - k < 3]
            hl = VGroup(*(SurroundingRectangle(VGroup(row_a[k]), buff=0.04).set_stroke(YELLOW, 3) for k, _ in pairs))
            terms = " + ".join(f"{a[k]} \\cdot {b[j]}" for k, j in pairs)
            line = Tex(f"c_{n_} = {terms} = {c_vals[n_]}", font_size=30).move_to(np.array([0, -1.6 - 0.0, 0]))
            result.add(line)
            self.play(FadeIn(hl), FadeIn(line), run_time=0.6)
            self.wait(0.9)
            self.play(FadeOut(hl), FadeOut(line), run_time=0.3)
        self.wait_speech()
        top_row.clear_updaters()
        numbers = Tex(R"321 \times 654 = 4 + 13\cdot 10 + 28 \cdot 10^2 + 27 \cdot 10^3 + 18 \cdot 10^4 = 209934",
                      font_size=36).move_to(2.2 * DOWN)
        self.caption("וגם כפל מספרים: x = 10, ועוד נשא",
                     say="ושימו לב: מספר הוא פולינום שבו איקס שווה עשר. אז גם כפל ארוך של מספרים, כמו שלמדנו בבית הספר, "
                         "הוא קונבולוציה של הספרות, ועוד העברה של נשא.")
        self.play(Write(numbers), run_time=2.5)
        self.wait_speech()
        self.fade_all()


class ConvTheorem(TransformScene):
    def construct(self):
        thm = Tex(R"\mathcal{F}[f * g] = \mathcal{F}[f] \cdot \mathcal{F}[g]", font_size=52).to_edge(UP, buff=0.4)
        self.caption("משפט הקונבולוציה: קונבולוציה הופכת לכפל",
                     say="והנה הקשר להתמרת פורייה: ההתמרה של קונבולוציה היא המכפלה של ההתמרות. "
                         "פעולה מסובכת במרחב הזמן הופכת לכפל פשוט, נקודה בנקודה, במרחב התדר.")
        self.play(Write(thm), run_time=2)
        self.wait_speech()
        left = axes((-2, 2, 1), (0, 1.3, 1), 5.6, 2.2, x_labels=int_labels((-1, 1))).move_to(3.5 * LEFT + 0.5 * UP)
        right = axes((-4, 4, 1), (-0.3, 1.1, 1), 5.6, 2.2, x_labels=int_labels((-3, -2, -1, 1, 2, 3))).move_to(3.5 * RIGHT + 0.5 * UP)
        s = ValueTracker(-1.5)
        fixed = curve(left, rect, -2, 2, BLUE, 3, step=0.005, jumps=(-0.5, 0.5))
        moving = always_redraw(lambda: curve(left, lambda t: rect(t - s.get_value()), -2, 2, GREEN, 3, step=0.005,
                                             jumps=(s.get_value() - 0.5, s.get_value() + 0.5)))
        overlap = always_redraw(lambda: Rectangle(
            width=max(1 - abs(s.get_value()), 0.001) * (left.c2p(1, 0)[0] - left.c2p(0, 0)[0]),
            height=left.c2p(0, 1)[1] - left.c2p(0, 0)[1]).move_to(left.c2p(s.get_value() / 2, 0.5))
            .set_stroke(width=0).set_fill(YELLOW, 0.4 if abs(s.get_value()) < 1 else 0))
        tri_trace = always_redraw(lambda: curve(left, lambda t: np.maximum(1 - np.abs(t), 0), -1.5, max(s.get_value(), -1.49),
                                                F_COLOR, 4, step=0.005))
        self.caption("פולס כפול פולס: השטח המשותף בונה משולש",
                     say="דוגמה. הקונבולוציה של פולס עם עצמו: מחליקים עותק אחד על השני, ובכל מקום מודדים את השטח המשותף. "
                         "יוצא משולש.")
        self.play(FadeIn(left), ShowCreation(fixed))
        self.add(overlap, moving, tri_trace)
        self.play(s.animate.set_value(1.5), run_time=6, rate_func=linear)
        self.wait_speech()
        s1 = curve(right, sinc, -4, 4, BLUE, 2.5, step=0.01).set_stroke(opacity=0.7)
        s2 = curve(right, lambda w: sinc(w) ** 2, -4, 4, F_COLOR, 4, step=0.01)
        lab = VGroup(Tex(R"\hat{\text{rect}} = \text{sinc}", font_size=30).set_color(BLUE),
                     Tex(R"\hat{\text{tri}} = \text{sinc}^2", font_size=30).set_color(F_COLOR)).arrange(DOWN, aligned_edge=LEFT)
        lab.next_to(right, DOWN, buff=0.3)
        self.caption("ובתדר: ההתמרה של המשולש היא הריבוע של ההתמרה של הפולס",
                     say="ובמרחב התדר: ההתמרה של הפולס היא העקומה הכחולה שראינו בהתחלה, וההתמרה של המשולש היא פשוט הריבוע שלה.")
        self.play(FadeIn(right), ShowCreation(s1))
        self.play(TransformFromCopy(s1, s2), FadeIn(lab), run_time=2)
        self.wait_speech()
        self.caption("כמו מעבר לבסיס נוח יותר באלגברה לינארית",
                     say="גדי משווה את זה לאלגברה לינארית: כשחישוב מסובך בבסיס אחד, עוברים לבסיס נוח, מחשבים שם וחוזרים. "
                         "במובן מסוים זו כל המהות של התמרת פורייה: מעבר לבסיס נוח יותר.")
        self.wait_speech()
        for m in (moving, overlap, tri_trace):
            m.clear_updaters()
        self.fade_all()


class FastMultiply(TransformScene):
    def construct(self):
        def node(text, color):
            t = heb(text, 28)
            return VGroup(RoundedRectangle(width=t.get_width() + 0.6, height=1.0, corner_radius=0.15)
                          .set_stroke(color, 2.5).set_fill(color, 0.12), t)

        a = node("מקדמים של A", BLUE).move_to([4.6, 1.8, 0])
        b = node("מקדמים של B", GREEN).move_to([4.6, -0.2, 0])
        fa = node("ערכים של A בשורשי היחידה", BLUE).move_to([0, 1.8, 0])
        fb = node("ערכים של B בשורשי היחידה", GREEN).move_to([0, -0.2, 0])
        mul = node("כפל נקודה בנקודה", YELLOW).move_to([-4.3, 0.8, 0])
        out = node("מקדמים של A כפול B", F_COLOR).move_to([-4.3, -1.9, 0])
        arr = VGroup(Arrow(a.get_left(), fa.get_right()), Arrow(b.get_left(), fb.get_right()),
                     Arrow(fa.get_left(), mul.get_right() + 0.2 * UP), Arrow(fb.get_left(), mul.get_right() + 0.2 * DOWN),
                     Arrow(mul.get_bottom(), out.get_top()))
        tags = VGroup(Text("FFT", font=FONT, font_size=24).next_to(arr[0], UP, buff=0.05),
                      Text("FFT", font=FONT, font_size=24).next_to(arr[1], UP, buff=0.05),
                      heb("התמרה הפוכה", 24).next_to(arr[4], RIGHT, buff=0.1))
        self.caption("כפל מהיר של פולינומים, בשלושה צעדים",
                     say="ומכאן אלגוריתם לכפל פולינומים, מהיר יותר מפתיחת סוגריים. מתמירים את שתי רשימות המקדמים. "
                         "התמרה כזו היא בעצם הצבה של שורשי היחידה בפולינום. "
                         "כופלים את הערכים נקודה בנקודה, וזה כבר קל, ומתמירים בחזרה.")
        self.play(FadeIn(a), FadeIn(b))
        self.play(GrowArrow(arr[0]), GrowArrow(arr[1]), FadeIn(tags[:2]), FadeIn(fa), FadeIn(fb))
        self.play(GrowArrow(arr[2]), GrowArrow(arr[3]), FadeIn(mul))
        self.play(GrowArrow(arr[4]), FadeIn(tags[2]), FadeIn(out))
        self.wait_speech()
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None

        size = 0.7

        def row(vals, color, y):
            return VGroup(*(VGroup(Square(size).set_stroke(color, 2).set_fill(color, 0.15 if v else 0.03),
                                   Tex(str(v), font_size=30)).move_to([(i - 2.5) * size, y, 0]) for i, v in enumerate(vals)))

        wrong = row([4 + 27, 13 + 18, 28], RED, 0.4)
        cyc = Tex(R"(4,\ 13,\ 28,\ 27,\ 18) \;\to\; (4 + 27,\ 13 + 18,\ 28)", font_size=36).move_to(1.8 * UP)
        padded = row([4, 13, 28, 27, 18, 0], GREEN, -1.4)
        self.caption("זהירות: ההתמרה הבדידה מחזורית",
                     say="אבל יש מלכודת, וגדי מודה שבפוסט הקודם הוא שיקר קצת. ההתמרה הבדידה חושבת שהרשימה מחזורית. "
                         "אם מכפילים שתי רשימות באורך שלוש בהתמרה באורך שלוש, המקדמים הגבוהים מתגלגלים חזרה להתחלה ומתערבבים.")
        self.play(Write(cyc))
        self.play(FadeIn(wrong))
        self.wait_speech()
        self.caption("הפתרון: לרפד באפסים",
                     say="הפתרון: מרפדים את הרשימות באפסים, כך שיהיה מקום לכל התוצאה, ואז מתעלמים מהאפסים המיותרים. "
                         "זה מרגיש מלוכלך נורא, אומר גדי, אבל זה למעשה לא פתרון כזה גרוע.")
        self.play(FadeIn(padded))
        self.wait_speech()
        self.fade_all()


class FFTIdea(TransformScene):
    def construct(self):
        cost = axes((0, 1000, 200), (0, 1.1e6, 2.5e5), 6, 3.2).move_to(3.6 * LEFT + 0.6 * UP)
        nsq = curve(cost, lambda n: n ** 2, 1, 1000, RED, 3.5, step=2)
        nlog = curve(cost, lambda n: n * np.log2(np.maximum(n, 1)), 1, 1000, GREEN, 3.5, step=2)
        tags = VGroup(Tex("n^2", font_size=34).set_color(RED).next_to(cost.c2p(1000, 1e6), UP, buff=0.1),
                      Tex(R"n \log n", font_size=34).set_color(GREEN).next_to(cost.c2p(1000, 1e4), UP, buff=0.15))
        nums = VGroup(heb("עבור אלף מספרים:", 30),
                      heb("ישירות: מיליון פעולות", 30, RED),
                      heb("FFT: בערך עשרת אלפים", 30, GREEN)).arrange(DOWN, aligned_edge=RIGHT, buff=0.35).move_to(3.6 * RIGHT + 0.8 * UP)
        self.caption("חישוב ישיר: n בריבוע פעולות",
                     say="כדי שכל זה יהיה מהיר באמת, צריך לחשב את ההתמרה עצמה מהר. חישוב ישיר לוקח אן בריבוע פעולות: "
                         "אן מקדמים, ולכל אחד סכום של אן איברים. בשביל אלף מספרים, מיליון פעולות. "
                         "התמרת פורייה המהירה עושה את זה באן כפול לוג אן, בערך עשרת אלפים.")
        self.play(FadeIn(cost), ShowCreation(nsq), FadeIn(tags[0]))
        self.play(ShowCreation(nlog), FadeIn(tags[1]))
        self.play(LaggedStart(*(FadeIn(m) for m in nums), lag_ratio=0.4))
        self.wait_speech()
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None

        N = 8
        c = np.array([-3.4, 0.2, 0])
        r = 1.6
        circle = Circle(radius=r).move_to(c).set_stroke(GREY_B, 1.5)
        dots = VGroup(*(Dot(c + r * direction(TAU * k / N), radius=0.09, fill_color=interpolate_color(BLUE, PINK, k / N))
                        for k in range(N)))
        self.caption("הסוד: הריבועים של 8 שורשים הם 4 שורשים, כל אחד פעמיים",
                     say="והקסם מגיע מתכונה מיוחדת של שורשי היחידה. אם מעלים בריבוע את שמונת שורשי היחידה מסדר שמונה, "
                         "הזוויות מוכפלות, ומקבלים רק ארבעה שורשים, של סדר ארבע, וכל אחד מהם פעמיים.")
        self.play(ShowCreation(circle), LaggedStart(*(GrowFromCenter(d) for d in dots), lag_ratio=0.1))
        self.wait(0.5)
        targets = [c + r * direction(2 * TAU * k / N) + 0.13 * (1 if k >= N // 2 else -1) * direction(2 * TAU * k / N + PI / 2)
                   for k in range(N)]
        self.play(*(d.animate.move_to(t) for d, t in zip(dots, targets)), run_time=3)
        self.wait_speech()

        split = VGroup(
            Tex(R"A(x) = a_0 + a_1 x + a_2 x^2 + \cdots + a_7 x^7", font_size=34),
            Tex(R"A(x) = A_0(x^2) + x\, A_1(x^2)", font_size=40),
            Tex(R"A_0(y) = a_0 + a_2 y + a_4 y^2 + a_6 y^3", font_size=30).set_color(BLUE),
            Tex(R"A_1(y) = a_1 + a_3 y + a_5 y^2 + a_7 y^3", font_size=30).set_color(PINK),
        ).arrange(DOWN, buff=0.35).move_to(3.2 * RIGHT + 0.4 * UP)
        self.caption("הפרד ומשול: זוגיים ואי-זוגיים",
                     say="ולכן מפרקים את הפולינום לשני חצאים: המקדמים במקומות הזוגיים והמקדמים במקומות האי-זוגיים. "
                         "שני הפולינומים האלה צריכים להיות מוצבים רק בריבועים של השורשים, כלומר בארבעה שורשים במקום שמונה. "
                         "שתי בעיות בחצי הגודל, וחצי מהערכים מתקבלים בחינם.")
        for m in split:
            self.play(Write(m), run_time=1.2)
        self.wait_speech()
        self.fade_all()


class Butterfly(TransformScene):
    def construct(self):
        N = 8
        cols = 4
        xs = np.linspace(-5.2, 5.2, cols)
        ys = np.linspace(2.4, -2.2, N)

        def bitrev(k):
            return int(f"{k:03b}"[::-1], 2)

        nodes = VGroup()
        for ci, x in enumerate(xs):
            for ri, y in enumerate(ys):
                nodes.add(Dot([x, y, 0], radius=0.08, fill_color=GREY_A))
        in_labels = VGroup(*(Tex(f"a_{bitrev(r)}", font_size=26).next_to(nodes[r], LEFT, buff=0.15) for r in range(N)))
        out_labels = VGroup(*(Tex(f"b_{r}", font_size=26).next_to(nodes[(cols - 1) * N + r], RIGHT, buff=0.15) for r in range(N)))
        lines = VGroup()
        for stage in range(3):
            span = 2 ** stage
            layer = VGroup()
            for r in range(N):
                partner = r ^ span
                color = [BLUE, GREEN, ORANGE][stage]
                layer.add(Line(nodes[stage * N + r].get_center(), nodes[(stage + 1) * N + r].get_center()).set_stroke(color, 2))
                layer.add(Line(nodes[stage * N + r].get_center(), nodes[(stage + 1) * N + partner].get_center()).set_stroke(color, 2))
            lines.add(layer)
        self.caption("הרקורסיה: שמונה, ארבע, שתיים, אחת",
                     say="ממשיכים לחצות באופן רקורסיבי: שמונה הופך לשתי בעיות של ארבע, ארבע לשתיים, ושתיים לאחת, "
                         "וההתמרה של רשימה באורך אחת היא הרשימה עצמה. לכן גדי מניח שאורך הרשימה הוא חזקה של שתיים.")
        self.play(FadeIn(nodes), FadeIn(in_labels))
        self.wait_speech()
        self.caption("בכל שלב: n פעולות. מספר השלבים: log n",
                     say="ואז ממזגים בחזרה, שלב אחרי שלב. זה נראה כמו כנפי פרפר, ולכן קוראים לזה דיאגרמת פרפר. "
                         "בכל שלב יש בערך אן פעולות, ויש לוג אן שלבים. בדיוק כמו מיון מיזוג.")
        for layer in lines:
            self.play(LaggedStart(*(ShowCreation(l) for l in layer), lag_ratio=0.03, run_time=1.8))
        self.play(FadeIn(out_labels))
        self.wait_speech()
        source = ["def fft(a):",
                  "    n = len(a)",
                  "    if n == 1:",
                  "        return a",
                  "    omega = exp(2j * pi / n)",
                  "    b0 = fft(a[0::2])",
                  "    b1 = fft(a[1::2])",
                  "    return [b0[k % (n//2)] + omega**k * b1[k % (n//2)]",
                  "            for k in range(n)]"]
        # טקסט אחד רב-שורתי, כדי שההזחה תישמר
        code = Text("\n".join(source), font="DejaVu Sans Mono", font_size=26, t2c={"def": YELLOW, "return": YELLOW})
        box = SurroundingRectangle(code, buff=0.35).set_stroke(GREY_D, 2).set_fill("#1c1f27", 1)
        code = VGroup(box, code)
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None
        code.move_to(0.3 * UP)
        self.caption("כל האלגוריתם, בכמה שורות",
                     say="וכך זה נראה בקוד, בגרסה שגדי כותב בפוסט, כמעט מילה במילה: שתי קריאות רקורסיביות, ומיזוג אחד.")
        self.play(FadeIn(code, shift=0.2 * UP), run_time=1.5)
        self.wait_speech()
        self.fade_all()


class Aliasing(TransformScene):
    def construct(self):
        ax = axes((0, 2, 0.25), (-1.4, 1.4, 1), 12, 2.6, x_labels=[(0.5, "0.5"), (1, "1"), (1.5, "1.5"), (2, "2")]).shift(1.7 * UP)
        fast = curve(ax, lambda t: np.cos(TAU * 7 * t), 0, 2, BLUE, 3, step=0.002)
        ts = np.arange(0, 2.001, 1 / 8)
        samples = VGroup(*(Dot(ax.c2p(t, np.cos(TAU * 7 * t)), radius=0.08, fill_color=YELLOW) for t in ts))
        slow = curve(ax, lambda t: np.cos(TAU * 1 * t), 0, 2, RED, 3.5, step=0.004)
        self.caption("דוגמים גל מהיר לאט מדי, ורואים גל אחר",
                     say="ועכשיו, מה קורה כשדוגמים? זה המעבר מהעולם הרציף לעולם הבדיד. "
                         "הנה גל בתדר שבע, ונדגום אותו שמונה פעמים בשנייה.")
        self.play(FadeIn(ax), ShowCreation(fast), run_time=2)
        self.play(LaggedStart(*(GrowFromCenter(d) for d in samples), lag_ratio=0.05))
        self.wait_speech()
        self.say("עכשיו תסתכלו רק על הנקודות הצהובות. הן נמצאות בדיוק על גל אחר לגמרי, איטי, בתדר אחת. "
                 "מתוך הדגימות אי אפשר להבדיל בין שני הגלים. לתופעה קוראים התחזות, באנגלית אֵיילְיַאסינג.")
        self.play(fast.animate.set_stroke(opacity=0.25), ShowCreation(slow), run_time=2)
        self.wait_speech()

        spec = axes((-12, 12, 1), (0, 1.3, 1), 12, 1.8, x_labels=int_labels((-8, -7, -1, 1, 7, 8))).shift(1.6 * DOWN)
        band = Rectangle(width=spec.c2p(4, 0)[0] - spec.c2p(-4, 0)[0], height=2.0).move_to(spec.c2p(0, 0.6))
        band.set_stroke(GREY_A, 1.5).set_fill(WHITE, 0.05)
        orig = VGroup(spike(spec, -7, 1, BLUE), spike(spec, 7, 1, BLUE))
        copies = VGroup(*(spike(spec, x + 8 * k, 0.8, RED).set_opacity(0.8) for k in (-1, 1) for x in (-7, 7)
                          if abs(x + 8 * k) <= 12))
        self.caption("בתדר: דגימה = סכימה של עותקים מוזזים",
                     say="ובמרחב התדר רואים למה. זוכרים את הדיאגרמה? דגימה בזמן היא סכימה בתדר: "
                         "ההתמרה של הדגימות היא סכום של עותקים של ההתמרה המקורית, מוזזים בכפולות של קצב הדגימה, שמונה. "
                         "העותק של מינוס שבע, מוזז בשמונה, נוחת בדיוק על אחת.")
        self.play(FadeIn(spec), FadeIn(band), GrowArrow(orig[0]), GrowArrow(orig[1]))
        self.play(LaggedStart(*(GrowArrow(c) for c in copies), lag_ratio=0.3, run_time=2))
        self.wait_speech()
        self.caption("נייקוויסט: לדגום פי שניים מהתדר הגבוה ביותר",
                     say="ולכן הכלל: צריך לדגום יותר מפעמיים מהתדר הגבוה ביותר באות. "
                         "האוזן שומעת עד בערך עשרים אלף הרץ, ולכן דיסק דוגם ארבעים וארבעה אלף ומאה פעמים בשנייה.")
        self.wait_speech()
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None

        d = np.array([1.6, 1.0, 0])
        corners = {"f": np.array([-2.4, 2.2, 0]), "F": np.array([1.6, 2.2, 0]),
                   "g": np.array([-2.4, -0.6, 0]), "G": np.array([1.6, -0.6, 0])}
        front = {k: Tex(v, font_size=40).move_to(corners[k]) for k, v in
                 (("f", "f"), ("F", R"\hat f"), ("g", "g"), ("G", R"\hat g"))}
        back = {k: Tex(v, font_size=40).set_color(TEAL).move_to(corners[k] + d) for k, v in
                (("f", "u"), ("F", R"\hat u"), ("g", "v"), ("G", R"\hat v"))}
        edges = VGroup()
        for a_, b_ in (("f", "F"), ("g", "G"), ("f", "g"), ("F", "G")):
            edges.add(Line(front[a_].get_center(), front[b_].get_center(), buff=0.35).set_stroke(GREY_A, 2))
            edges.add(Line(back[a_].get_center(), back[b_].get_center(), buff=0.35).set_stroke(TEAL, 2))
        for k in corners:
            edges.add(DashedLine(front[k].get_center(), back[k].get_center(), buff=0.35).set_stroke(YELLOW, 2))
        legend = VGroup(heb("למעלה: הישר הממשי", 26), heb("מאחור: המספרים השלמים", 26, TEAL),
                        heb("מצהוב: דגימה בזמן, סכימה בתדר", 26, YELLOW)).arrange(DOWN, aligned_edge=RIGHT, buff=0.2)
        legend.to_corner(DR, buff=0.5).shift(0.6 * UP)
        self.caption("הקובייה הקומוטטיבית של גדי",
                     say="וגדי סוגר את התמונה: לפונקציות על המספרים השלמים יש בדיוק אותה דיאגרמה, "
                         "ודגימה מחברת אותה לדיאגרמה הרציפה. ביחד יוצאת קובייה קומוטטיבית: "
                         "ארבעה סוגים של פורייה, וכל המסלולים ביניהם מסכימים. גדי כותב: נסו לצייר אותה. אז ציירנו.")
        self.play(*(FadeIn(m) for m in front.values()), ShowCreation(edges[0:8:2]))
        self.play(*(FadeIn(m) for m in back.values()), ShowCreation(edges[1:8:2]))
        self.play(LaggedStart(*(ShowCreation(e) for e in edges[8:]), lag_ratio=0.2), FadeIn(legend))
        self.wait_speech()
        self.fade_all()


class Spectrogram(TransformScene):
    def construct(self):
        rate = 48000
        notes = [261.63, 329.63, 392.0, 523.25, 392.0, 329.63, 261.63]
        dur = 0.5
        t = np.arange(int(rate * dur)) / rate
        env = np.minimum(1, t / 0.02) * np.exp(-2.5 * t)
        y = np.concatenate([env * sum(a * np.sin(TAU * k * f * t) for k, a in ((1, 1), (2, 0.5), (3, 0.3)))
                            for f in notes])
        path = common.AUDIO_DIR / "tone_melody.wav"
        if not path.exists():
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
                w.writeframes((0.22 * y / np.abs(y).max() * 32767).astype(np.int16).tobytes())
        png = common.AUDIO_DIR / "spectrogram.png"
        if not png.exists():
            from PIL import Image
            win, hop = 2048, 240
            frames = [y[i:i + win] * np.hanning(win) for i in range(0, len(y) - win, hop)]
            mag = np.abs(np.fft.rfft(frames, axis=1))[:, : int(1700 * win / rate)]
            img = np.clip((np.log10(mag + 1e-3) + 1.0) / 3.2, 0, 1).T[::-1]
            stops = np.array([[20, 22, 28], [70, 30, 120], [220, 70, 60], [250, 170, 40], [255, 245, 170]], float)
            idx = img * (len(stops) - 1)
            lo = np.floor(idx).astype(int).clip(0, len(stops) - 2)
            fr = (idx - lo)[..., None]
            rgb = stops[lo] * (1 - fr) + stops[lo + 1] * fr
            Image.fromarray(rgb.astype(np.uint8)).resize((1200, 520)).save(png)
        total = dur * len(notes)
        image = ImageMobject(str(png)).set_width(11).move_to(0.5 * UP)
        frame = SurroundingRectangle(image, buff=0).set_stroke(GREY_B, 2)
        t_lab = heb("זמן", 28, GREY_A).next_to(frame, DOWN, buff=0.15)
        f_lab = heb("תדר", 28, GREY_A).next_to(frame, LEFT, buff=0.15)
        mask = Rectangle(width=image.get_width() + 0.02, height=image.get_height() + 0.02).set_stroke(width=0).set_fill(BG, 1)
        mask.move_to(image)
        self.caption("ספקטרוגרמה: התמרה של חלונות קצרים, זה אחרי זה",
                     say="ואיך רואים מנגינה, שבה התדרים משתנים בזמן? מתמירים חלונות קצרים, אחד אחרי השני, "
                         "ומציירים כל התמרה כעמודה של צבע. זו ספקטרוגרמה, מפה של זמן ותדר. הקשיבו.")
        self.play(FadeIn(frame), FadeIn(t_lab), FadeIn(f_lab))
        self.add(image, mask)
        self.wait_speech()
        self.add_sound(str(path))
        self.play(mask.animate.stretch(0.001, 0, about_edge=RIGHT), run_time=total, rate_func=linear)
        self.remove(mask)
        self.wait(0.5)
        self.caption("כל צליל: תדר יסוד, ומעליו ההרמוניות שלו",
                     say="כל תו הוא פס בהיר, והמנגינה עולה ויורדת. ומעל כל פס יש פסים חלשים יותר, בכפולות של התדר: "
                         "ההרמוניות. הן שנותנות לכל כלי את הצבע שלו. "
                         "וזוכרים את עקרון אי הוודאות? חלון קצר נותן דיוק בזמן ומטשטש את התדר, וחלון ארוך עושה את ההפך.")
        self.wait_speech()
        self.fade_all()


class Denoise(TransformScene):
    def construct(self):
        N = 256
        n = np.arange(N)
        clean = np.sin(TAU * 5 * n / N) + 0.6 * np.sin(TAU * 12 * n / N)
        noisy = clean + np.random.default_rng(7).normal(0, 0.7, N)
        top = axes((0, N, 32), (-3.2, 3.2, 1), 11.5, 2.4).shift(1.9 * UP)
        noisy_mob = VMobject().set_points_as_corners(top.c2p(n, noisy)).set_stroke(GREY_A, 1.8)
        clean_mob = VMobject().set_points_as_corners(top.c2p(n, clean)).set_stroke(SUM_COLOR, 4)
        self.caption("אות עם רעש",
                     say="שימוש מעשי אחרון: ניקוי רעש. הנה אות, מאתיים חמישים ושש דגימות, עם הרבה רעש. "
                         "קשה לראות מה מתחבא בו.")
        self.play(FadeIn(top), ShowCreation(noisy_mob), run_time=2)
        self.wait_speech()
        B = np.abs(np.fft.fft(noisy))[:64] / N * 2
        low = axes((0, 64, 8), (0, 1.2, 0.5), 11.5, 2.4, x_labels=int_labels((5, 12, 32, 48))).shift(1.4 * DOWN)
        unit = low.c2p(1, 0)[0] - low.c2p(0, 0)[0]
        bars_ = VGroup(*(Rectangle(width=unit * 0.7, height=max(v * (low.c2p(0, 1)[1] - low.c2p(0, 0)[1]), 0.003))
                         .move_to(low.c2p(k + 0.5, v / 2)).set_stroke(width=0).set_fill(SPEC, 0.9) for k, v in enumerate(B)))
        self.caption("בתדר: שתי פסגות מעל שטיח של רעש",
                     say="במרחב התדר התמונה שונה לגמרי. הרעש מתפזר על כל התדרים, שטיח נמוך ואחיד, "
                         "והאות עצמו מתרכז בשתי פסגות גבוהות, בתדר חמש ובתדר שתים עשרה.")
        self.play(FadeIn(low), LaggedStart(*(GrowFromEdge(b_, DOWN) for b_ in bars_), lag_ratio=0.01, run_time=2))
        self.wait_speech()
        thr = DashedLine(low.c2p(0, 0.35), low.c2p(64, 0.35)).set_stroke(RED, 2)
        keep = [k for k, v in enumerate(B) if v > 0.35]
        self.caption("מוחקים את כל מה שמתחת לסף, ומתמירים בחזרה",
                     say="אז מוחקים את כל התדרים שמתחת לסף, ומשאירים רק את הפסגות. ומתמירים בחזרה.")
        self.play(ShowCreation(thr))
        self.play(*(b_.animate.set_fill(opacity=0.12) for k, b_ in enumerate(bars_) if k not in keep), run_time=1.5)
        self.play(noisy_mob.animate.set_stroke(opacity=0.25), ShowCreation(clean_mob), run_time=2.5)
        self.wait_speech()
        self.say("והאות הנקי חוזר. בערך כך עובדים מסנני רעש בהקלטות, וזה גם הרעיון בבסיס דחיסה: "
                 "שומרים רק את התדרים שחשובים.")
        self.wait_speech()
        self.fade_all()


class History(TransformScene):
    def construct(self):
        line = Line([-6.4, 0.8, 0], [6.4, 0.8, 0]).set_stroke(GREY_B, 3)
        events = [
            (-5.4, "1805", "גאוס: מסלולי אסטרואידים", UP),
            (-2.7, "1822", "פורייה: תורת החום", DOWN),
            (0.0, "1930", "דיראק: פונקציית הדלתא", UP),
            (2.7, "1965", "קולי וטוקי: FFT", DOWN),
            (5.4, "1994", "שור: פורייה קוונטי", UP),
        ]
        texts = [
            "והנה הפתעה: את הרעיון של ההתמרה המהירה גילה כבר גאוס, ב-1805, כשחישב מסלולים של אסטרואידים. "
            "זה עוד לפני שפורייה פרסם משהו. אבל גאוס לא פרסם את זה בחייו.",
            "פורייה פרסם את הספר שלו על תורת החום ב-1822.",
            "דיראק הכניס את פונקציית הדלתא לפיזיקה בשנות העשרים והשלושים, הרבה לפני שהיה לה פורמליזם.",
            "ב-1965 גילו קולי וטוקי את ההתמרה המהירה מחדש, ומאז היא באחד המקומות הראשונים ברשימות של האלגוריתמים החשובים במאה העשרים.",
            "וב-1994 פיטר שור השתמש בהתמרת פורייה קוונטית כדי לפרק מספרים לגורמים. גם על זה גדי כתב.",
        ]
        self.play(ShowCreation(line))
        for (x, year, what, side), text in zip(events, texts):
            self.say(text)
            p = np.array([x, 0.8, 0])
            tick = Dot(p, radius=0.1, fill_color=YELLOW)
            label = VGroup(heb(year, 30, YELLOW), heb(what, 25)).arrange(DOWN, buff=0.12).next_to(p, side, buff=0.35)
            self.play(GrowFromCenter(tick), FadeIn(label, shift=0.2 * side))
            self.wait_speech()
        self.play(*map(FadeOut, self.mobjects))
        uses = VGroup(*(heb(t, 32) for t in [
            "מוזיקה ותמונות: MP3 ו-JPEG",
            "תקשורת: Wi-Fi ו-4G מעבירים מידע על תדרים רבים במקביל",
            "רפואה: MRI מודד את ההתמרה ומשחזר תמונה",
            "אסטרונומיה וכימיה: ספקטרוסקופיה",
            "חשבון: כפל של מספרים עם מיליוני ספרות",
        ])).arrange(DOWN, aligned_edge=RIGHT, buff=0.4).move_to(0.4 * UP)
        self.say("והיום ההתמרה המהירה רצה בכל מקום: בדחיסה של מוזיקה ותמונות, בכל חיבור אלחוטי, "
                 "במכשירי אם אר איי, בספקטרוסקופיה, ואפילו בכפל של מספרים עם מיליוני ספרות. "
                 "ובכל אחד מהמקרים האלה, זה אותו רעיון: לעבור לבסיס נוח יותר.")
        self.play(LaggedStart(*(FadeIn(u, shift=0.2 * LEFT) for u in uses), lag_ratio=0.5, run_time=5))
        self.wait_speech()
        self.fade_all()


class Credits(TransformScene):
    def construct(self):
        thanks = heb("תודה", 60, SUM_COLOR)
        line1 = heb("הסרטון מבוסס על הפוסטים של", 34)
        line2 = heb("גדי אלכסנדרוביץ' בבלוג \"לא מדויק\"", 44, YELLOW)
        posts = VGroup(*(heb(t, 26, GREY_A) for t in [
            "2014: התמרת פורייה - מה זה בכלל?",
            "2014: התמרת פורייה הבדידה - מה זה בכלל?",
            "2014: התמרת פורייה המהירה",
        ])).arrange(DOWN, buff=0.18)
        site = Text("gadial.net", font=FONT, font_size=34, fill_color=GREY_B)
        VGroup(thanks, line1, line2, posts, site).arrange(DOWN, buff=0.38)
        self.say("הסרטון הזה מבוסס על שלושה פוסטים של גדי אלכסנדרוביץ' בבלוג שלו, לא מדויק: "
                 "התמרת פורייה, מה זה בכלל. התמרת פורייה הבדידה, מה זה בכלל. והתמרת פורייה המהירה. תודה, גדי.")
        self.play(Write(thanks))
        self.play(LaggedStartMap(FadeIn, VGroup(line1, line2, posts, site), shift=0.2 * UP, lag_ratio=0.4, run_time=3))
        self.wait(3)
        self.fade_all(run_time=1.5)


SCENES = [Title, Recap, Bridge, IntegralSum, Detector, Equalizer, Beats, Delta, Uncertainty, Spectrogram, Diagram,
          Discrete, RootsSum, Aliasing, DFTExample, Denoise, Convolution, ConvTheorem, FastMultiply, FFTIdea, Butterfly,
          History, Credits]
