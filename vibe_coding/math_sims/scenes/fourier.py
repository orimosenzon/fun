"""טורי פורייה: איך סכום של סינוסים וקוסינוסים מקרב כמעט כל פונקציה.

בעקבות הפוסטים של גדי אלכסנדרוביץ' בבלוג "לא מדויק" (gadial.net):
טורי פורייה - מה זה בכלל? (2014), בסיסים אורתונורמליים במרחבי הילברט (2014),
התמרת פורייה - מה זה בכלל? (2014), יום פאי שמח! (2010),
וכמה מעלות טובות לרדיאן עלינו (2008).
קריינות: Azure, הקול he-IL-Hila:DragonLatestNeural (ראו use_azure ב-common.py).
רינדור: ./render.sh --hd scenes/fourier.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403
import common  # noqa: E402

common.use_azure()
# Hila הוגה "פורייה" כמו המקום פּוֹרִיָּה, ושמות אחרים בשגיאה. הניקוד נבדק בתמלול חוזר
common.SPOKEN.update({"פורייה": "פוריֶה", "דיריכלה": "דיריכלֶה", "לגראנז'": "לָגְרַאנְז'", "גיבס": "גִיבְּס",
                      "ליפוט פייר": "לִיפּוֹט פֶה-יֵר", "פייר": "פֶה-יֵר", "בסל": "בֶּסֶל", "אוילר": "אוֹילֶר",
                      "צזארו": "צ'זארו", "ג'וזיה": "ג'וֹזַיָּה"})

F_COLOR = YELLOW      # הפונקציה שמקרבים
SUM_COLOR = TEAL      # הסכום החלקי
POS, NEG = GREEN, RED
AXIS = GREY_B
WAVES = [BLUE, GREEN, ORANGE, PURPLE_B, PINK, GOLD, MAROON_B, TEAL_B]


# ---------- פונקציות ומקדמים ----------

def direction(angle):
    return np.array([np.cos(angle), np.sin(angle), 0])


def wrap(x):
    """מעביר את x לקטע [-π, π), כלומר ההמשכה המחזורית."""
    return np.mod(np.asarray(x, dtype=float) + PI, TAU) - PI


def square(x):
    return np.where(wrap(x) >= 0, 1.0, -1.0)


def saw(x):
    return wrap(x)


def tri(x):
    return np.abs(wrap(x))


def skyline(x):
    """פונקציה "מכוערת" במכוון: מדרגות, שיפוע ופינות."""
    x = wrap(x)
    return np.select(
        [x < -2.2, x < -1.2, x < -0.3, x < 0.9, x < 2.0],
        [0.3, 1.2, 0.5 + 0.6 * (x + 1.2), -0.4, 1.4 - 0.8 * (x - 0.9) ** 2],
        0.2)


def coeffs_numeric(f, K, samples=20000):
    """מקדמי פורייה a_0..a_K, b_0..b_K באינטגרציה נומרית (עם המכפלה הפנימית של גדי, חלקי π)."""
    xs = np.linspace(-PI, PI, samples, endpoint=False)
    dx = TAU / samples
    fx = f(xs)
    a = np.array([np.sum(fx * np.cos(k * xs)) * dx / PI for k in range(K + 1)])
    b = np.array([np.sum(fx * np.sin(k * xs)) * dx / PI for k in range(K + 1)])
    return a, b


def square_coeffs(K):
    k = np.arange(K + 1)
    b = np.where(k % 2 == 1, 4 / (PI * np.maximum(k, 1)), 0.0)
    return np.zeros(K + 1), b


def saw_coeffs(K):
    k = np.arange(K + 1)
    b = np.where(k > 0, 2 * (-1.0) ** (k + 1) / np.maximum(k, 1), 0.0)
    return np.zeros(K + 1), b


def tri_coeffs(K):
    k = np.arange(K + 1)
    a = np.where(k % 2 == 1, -4 / (PI * np.maximum(k, 1) ** 2), 0.0)
    a[0] = PI
    return a, np.zeros(K + 1)


def partial(x, a, b, n):
    """הסכום החלקי עד ההרמוניה n. n יכול להיות שבר: האיבר האחרון נכנס בהדרגה, כדי שהמעבר יהיה רציף."""
    x = np.asarray(x, dtype=float)
    total = np.full_like(x, a[0] / 2)
    full = int(np.floor(n))
    for k in range(1, min(full, len(a) - 1) + 1):
        total += a[k] * np.cos(k * x) + b[k] * np.sin(k * x)
    frac = n - full
    if frac > 1e-6 and full + 1 < len(a):
        k = full + 1
        total += frac * (a[k] * np.cos(k * x) + b[k] * np.sin(k * x))
    return total


def fejer(x, a, b, m):
    """ממוצע צזארו של הסכומים החלקיים S_0..S_m (סכום פייר)."""
    x = np.asarray(x, dtype=float)
    total = np.full_like(x, a[0] / 2)
    for k in range(1, m + 1):
        total += (1 - k / (m + 1)) * (a[k] * np.cos(k * x) + b[k] * np.sin(k * x))
    return total


# ---------- ציור ----------

def pi_axes(x_range=(-PI, PI), y_range=(-1.5, 1.5, 0.5), width=10, height=4, labels=True,
            half=False, label_size=28):
    """צירים עם שנתות בכפולות של π/2 ותוויות בכפולות של π (או של π/2 כש-half)."""
    ax = Axes((x_range[0], x_range[1], PI / 2), y_range, width=width, height=height,
              axis_config=dict(stroke_color=AXIS, stroke_width=2))
    if labels:
        names = {1: R"\pi", -1: R"-\pi", 2: R"2\pi", -2: R"-2\pi", 3: R"3\pi", -3: R"-3\pi",
                 0.5: R"\tfrac{\pi}{2}", -0.5: R"-\tfrac{\pi}{2}"}
        group = VGroup()
        k0, k1 = int(round(x_range[0] / (PI / 2))), int(round(x_range[1] / (PI / 2)))
        for k in range(k0, k1 + 1):
            key = k / 2 if k % 2 else k // 2
            if k == 0 or (k % 2 and not half) or key not in names:
                continue
            group.add(Tex(names[key], font_size=label_size).set_color(GREY_A)
                      .next_to(ax.c2p(k * PI / 2, 0), DOWN, buff=0.15))
        ax.add(group)
        ax.x_labels = group
    return ax


def curve(ax, f, x0=-PI, x1=PI, color=SUM_COLOR, width=3, step=0.004, jumps=()):
    """גרף של f בין x0 ל-x1. בנקודות jumps הגרף נקטע (בלי קו אנכי)."""
    cuts = [x0, *[j for j in sorted(jumps) if x0 < j < x1], x1]
    mob = VMobject()
    for i, (s, e) in enumerate(zip(cuts[:-1], cuts[1:])):
        eps = 1e-6 if jumps else 0  # גם בקצוות: קפיצה בדיוק בקצה לא תצייר קו אנכי
        xs = np.linspace(s + eps, e - eps, max(int((e - s) / step), 2))
        pts = ax.c2p(xs, f(xs))
        if i == 0:
            mob.set_points_as_corners(pts)
        else:
            mob.start_new_path(pts[0])
            mob.add_points_as_corners(pts[1:])
    return mob.set_stroke(color, width)


def square_graph(ax, x0=-PI, x1=PI, color=F_COLOR, width=4):
    jumps = [k * PI for k in range(-4, 5)]
    return curve(ax, square, x0, x1, color, width, step=0.05, jumps=jumps)


def signed_area(ax, f, x0=-PI, x1=PI, step=0.01, opacity=0.55):
    """השטח בין הגרף לציר: ירוק מעליו, אדום מתחתיו."""
    xs = np.linspace(x0, x1, int((x1 - x0) / step) + 1)
    ys = f(xs)
    group = VGroup()
    for clip, color in ((np.maximum, POS), (np.minimum, NEG)):
        top = ax.c2p(xs, clip(ys, 0))
        base = ax.c2p(xs[::-1], 0 * xs)
        group.add(Polygon(*top, *base).set_stroke(width=0).set_fill(color, opacity))
    return group


def diff_area(ax, f, g, x0=-PI, x1=PI, step=0.01, color=RED, opacity=0.3):
    """השטח שבין שני גרפים: השגיאה של הקירוב."""
    xs = np.linspace(x0, x1, int((x1 - x0) / step) + 1)
    pts = [*ax.c2p(xs, f(xs)), *ax.c2p(xs[::-1], g(xs[::-1]))]
    return Polygon(*pts).set_stroke(width=0).set_fill(color, opacity)


def integral(f, x0=-PI, x1=PI, n=20000):
    xs = np.linspace(x0, x1, n, endpoint=False) + (x1 - x0) / (2 * n)
    return np.sum(f(xs)) * (x1 - x0) / n


def bars(values, width=5.5, height=2.2, color=SUM_COLOR, labels=None, max_value=None, label_size=22):
    """תרשים עמודות של מקדמים: עמודה לכל מקדם, למעלה או למטה לפי הסימן."""
    n = len(values)
    max_value = max_value or max(abs(v) for v in values) or 1
    slot = width / n
    group = VGroup()
    base = Line(ORIGIN, width * RIGHT).set_stroke(AXIS, 2)
    group.add(base)
    rects = VGroup()
    for i, v in enumerate(values):
        h = max(abs(v) / max_value * height, 0.001)
        r = Rectangle(width=slot * 0.62, height=h).set_stroke(width=0).set_fill(color, 0.9)
        r.move_to((i + 0.5) * slot * RIGHT + (h / 2 if v >= 0 else -h / 2) * UP)
        rects.add(r)
    group.add(rects)
    if labels:
        group.add(VGroup(*(Tex(str(l), font_size=label_size).set_color(GREY_A)
                           .move_to((i + 0.5) * slot * RIGHT + 0.25 * DOWN) for i, l in enumerate(labels))))
    group.rects = rects
    return group


class FourierScene(HebrewScene):
    def caption(self, text, say=None, size=32, wait=0.3):
        super().caption(text, say=say, size=size, wait=wait)


# ---------- סצנות ----------

class Title(FourierScene):
    def construct(self):
        ax = pi_axes((-2 * PI, 2 * PI), (-1.6, 1.6, 0.5), width=12, height=3.6).shift(0.9 * UP)
        target = square_graph(ax, -2 * PI, 2 * PI).set_stroke(opacity=0.5)
        a, b = square_coeffs(80)
        n = ValueTracker(1)
        approx = always_redraw(lambda: curve(ax, lambda x: partial(x, a, b, n.get_value()), -2 * PI, 2 * PI,
                                             step=0.006, width=4))
        self.say("איך אפשר לבנות פונקציה עם קפיצות ופינות חדות מתוך גלים חלקים ועגולים לגמרי?")
        self.play(FadeIn(ax), ShowCreation(target), run_time=1.5)
        self.play(ShowCreation(approx), run_time=1.5)
        self.play(n.animate.set_value(9), run_time=4, rate_func=linear)
        self.play(n.animate.set_value(61), run_time=4, rate_func=smooth)
        approx.clear_updaters()
        title = heb("טורי פורייה", 84).move_to(2.0 * DOWN)
        sub = heb("איך סינוסים וקוסינוסים בונים כמעט כל פונקציה", 36, GREY_A).next_to(title, DOWN, buff=0.35)
        self.say("זה הרעיון של ז'וזף פורייה: כמעט כל פונקציה היא סכום של סינוסים וקוסינוסים. "
                 "זה אחד הרעיונות הפוריים והחשובים ביותר במתמטיקה, ובסרטון הזה נבין איך הוא עובד.")
        self.play(FadeIn(title, shift=0.3 * UP))
        self.play(FadeIn(sub))
        self.wait(1)
        self.fade_all()


class Decompose(FourierScene):
    def construct(self):
        primes = Tex(R"360 = 2 \cdot 2 \cdot 2 \cdot 3 \cdot 3 \cdot 5", font_size=60).move_to(2.2 * UP)
        self.caption("מתמטיקאים אוהבים לפרק דברים לגורמים",
                     say="מתמטיקאים אוהבים לפרק דברים לגורמים. כל מספר טבעי קל יותר להבין "
                         "כשמפרקים אותו למכפלה של ראשוניים.")
        self.play(Write(primes), run_time=2)
        self.wait_speech()

        ax = pi_axes((-2 * PI, 2 * PI), (-2, 2, 1), width=11, height=3.4).shift(0.6 * DOWN)
        sine = curve(ax, np.sin, -2 * PI, 2 * PI, F_COLOR, 4)
        taylor = Tex(R"\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots", font_size=46)
        taylor.move_to(2.2 * UP)
        self.caption("טור טיילור: סכום אינסופי של חזקות",
                     say="גם פונקציות אפשר לפרק. אחת הדרכים האהובות היא טור טיילור: "
                         "כותבים את הפונקציה כסכום אינסופי של חזקות פשוטות של איקס, "
                         "ולכל חזקה יש מקדם שאומר כמה היא משתתפת בסכום.")
        self.play(FadeOut(primes), FadeIn(ax), ShowCreation(sine))
        self.play(Write(taylor), run_time=2)

        def taylor_sum(m):
            def g(x):
                total, term = np.zeros_like(x), x.copy()
                for k in range(m):
                    total += term
                    term = -term * x * x / ((2 * k + 2) * (2 * k + 3))
                return np.clip(total, -2.15, 2.15)
            return g

        approx = curve(ax, taylor_sum(1), -2 * PI, 2 * PI, SUM_COLOR, 3)
        self.play(ShowCreation(approx))
        for m in range(2, 9):
            self.play(Transform(approx, curve(ax, taylor_sum(m), -2 * PI, 2 * PI, SUM_COLOR, 3)), run_time=0.8)
        self.wait_speech()
        self.say("ייצוג כזה הוא כוח. למשל, כדי לגזור את הפונקציה, מספיק לגזור כל חזקה בנפרד, וזה קל.")
        self.wait_speech()

        pieces = VGroup(*(curve(ax, lambda x, k=k: np.sin(k * x) / k * 1.4 if k % 2 else np.cos(k * x) * 0.9,
                                -2 * PI, 2 * PI, WAVES[k % len(WAVES)], 2.5) for k in (1, 2, 3)))
        fourier = Tex(R"f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left( a_n \cos nx + b_n \sin nx \right)",
                      font_size=46).move_to(taylor)
        self.caption("פורייה: לפרק לגלים, סינוסים וקוסינוסים",
                     say="פורייה הציע לפרק פונקציות לאבני בניין אחרות. לא חזקות של איקס, אלא גלים: "
                         "סינוסים וקוסינוסים. הם פחות פשוטים מחזקות, אבל במובן מסוים מעניינים יותר, "
                         "כי הם מחזוריים, חוזרים על עצמם שוב ושוב.")
        self.play(FadeOut(approx), FadeOut(sine), FadeTransform(taylor, fourier))
        self.play(LaggedStart(*(ShowCreation(p) for p in pieces), lag_ratio=0.4, run_time=3))
        self.wait(1)
        self.fade_all()


class SineCosine(FourierScene):
    def construct(self):
        r = 1.3
        center = np.array([-4.6, 0.6, 0])
        circle = Circle(radius=r).move_to(center).set_stroke(GREY_B, 2)
        ax = Axes((0, TAU + 0.3, PI / 2), (-1.4, 1.4, 1), width=7.2, height=2 * r * 1.4 / 1,
                  axis_config=dict(stroke_color=AXIS, stroke_width=2))
        ax.shift(center + r * 0 * UP - ax.c2p(0, 0) + 2.4 * RIGHT)
        xl = VGroup(*(Tex(t, font_size=26).set_color(GREY_A).next_to(ax.c2p(k * PI, 0), DOWN, buff=0.12)
                      for k, t in ((1, R"\pi"), (2, R"2\pi"))))
        theta = ValueTracker(0)

        def tip():
            return center + r * np.array([np.cos(theta.get_value()), np.sin(theta.get_value()), 0])

        radius = always_redraw(lambda: Line(center, tip()).set_stroke(WHITE, 2.5))
        point = always_redraw(lambda: Dot(tip(), radius=0.08, fill_color=YELLOW))
        height = always_redraw(lambda: Line(tip(), [tip()[0], center[1], 0]).set_stroke(BLUE, 4))
        link = always_redraw(lambda: DashedLine(tip(), ax.c2p(theta.get_value(), np.sin(theta.get_value())))
                             .set_stroke(GREY_B, 1.5))
        trace = always_redraw(lambda: curve(ax, np.sin, 0, max(theta.get_value(), 0.01), BLUE, 4, step=0.02))

        self.caption("סינוס: הגובה של נקודה שמסתובבת על מעגל",
                     say="נתחיל מהכוכבים של הסיפור, סינוס וקוסינוס. נקודה מסתובבת על מעגל שהרדיוס שלו אחת. "
                         "הגובה שלה, כפונקציה של הזווית, הוא הסינוס.")
        self.play(ShowCreation(circle), FadeIn(ax), FadeIn(xl))
        self.add(trace, radius, height, link, point)
        self.play(theta.animate.set_value(TAU), run_time=6, rate_func=linear)
        self.wait_speech()
        for m in (trace, link, height):
            m.clear_updaters()
        sin_graph = trace

        cos_trace = always_redraw(lambda: curve(ax, np.cos, 0, max(theta.get_value() - TAU, 0.01), ORANGE, 4,
                                                step=0.02))
        width_line = always_redraw(lambda: Line(center, [tip()[0], center[1], 0]).set_stroke(ORANGE, 4))
        labels = VGroup(Tex(R"\sin x", font_size=34).set_color(BLUE), Tex(R"\cos x", font_size=34).set_color(ORANGE))
        labels.arrange(DOWN, aligned_edge=LEFT).next_to(ax, UR, buff=-0.2).shift(0.5 * UP)
        self.caption("קוסינוס: המרחק האופקי של אותה נקודה",
                     say="והמרחק האופקי שלה הוא הקוסינוס. אחרי סיבוב שלם, שתיים פאי, הכל חוזר על עצמו: "
                         "שתי הפונקציות מחזוריות, עם מחזור של שתיים פאי.")
        self.play(FadeOut(link), FadeIn(labels[0]))
        self.add(cos_trace, width_line)
        self.play(theta.animate.set_value(2 * TAU), FadeIn(labels[1]), run_time=6, rate_func=linear)
        cos_trace.clear_updaters()
        self.wait_speech()

        shift_arrow = Arrow(ax.c2p(PI / 2, 1.2), ax.c2p(0, 1.2), buff=0).set_color(GREY_A)
        head = Tex(R"\cos x = \sin\left(x + \tfrac{\pi}{2}\right)", font_size=36).next_to(ax, UP, buff=0.35)
        self.caption("קוסינוס הוא סינוס שקיבל יתרון של π/2 במירוץ",
                     say="ושימו לב: הקוסינוס הוא בסך הכל סינוס שנתנו לו יתרון של פאי חלקי שתיים במירוץ.")
        self.play(GrowArrow(shift_arrow), Write(head))
        self.play(sin_graph.copy().animate.shift(ax.c2p(0, 0) - ax.c2p(PI / 2, 0)).set_stroke(ORANGE, 2, 0.7),
                  run_time=2)
        self.wait_speech()
        self.fade_all()

        # קפיץ: תנועה הרמונית
        anchor = np.array([-4.5, 3.0, 0])
        y = ValueTracker(0)

        def mass_pos():
            return anchor + (2.9 - 1.2 * np.sin(y.get_value())) * DOWN

        def spring():
            top, bottom = anchor, mass_pos() + 0.35 * UP
            pts = [top]
            for i in range(1, 16):
                t = i / 16
                p = interpolate(top, bottom, t)
                pts.append(p + (0.25 if i % 2 else -0.25) * RIGHT)
            pts.append(bottom)
            return VMobject().set_points_as_corners(pts).set_stroke(GREY_A, 2.5)

        ceiling = Line(anchor + 0.8 * LEFT, anchor + 0.8 * RIGHT).set_stroke(GREY_B, 4)
        spring_mob = always_redraw(spring)
        mass = always_redraw(lambda: Square(0.7).move_to(mass_pos()).set_fill(BLUE_D, 1).set_stroke(BLUE_B, 2))
        ax2 = Axes((0, 4 * PI, PI), (-1.4, 1.4, 1), width=8, height=2.4,
                   axis_config=dict(stroke_color=AXIS, stroke_width=2)).move_to([1.8, anchor[1] - 2.9, 0])
        trace2 = always_redraw(lambda: curve(ax2, np.sin, 0, max(y.get_value(), 0.01), BLUE, 4, step=0.03))
        dot2 = always_redraw(lambda: Dot(ax2.c2p(y.get_value(), np.sin(y.get_value())), fill_color=YELLOW))
        ode = Tex(R"f''(x) = -f(x)", font_size=48).move_to([1.8, 2.6, 0])
        self.caption("וגם: התנועה הטבעית של קפיץ",
                     say="יש עוד סיבה שהגלים האלה חשובים, וגדי כתב עליה פוסט שלם. "
                         "תלו משקולת על קפיץ, משכו ושחררו. המשקולת תתנודד למעלה ולמטה, והגובה שלה הוא בדיוק סינוס.")
        self.play(ShowCreation(ceiling), FadeIn(spring_mob), FadeIn(mass), FadeIn(ax2))
        self.add(trace2, dot2)
        self.play(y.animate.set_value(4 * PI), run_time=8, rate_func=linear)
        self.say("הסיבה היא שהכוח של הקפיץ תמיד מושך חזרה, ובעוצמה שגדלה עם ההתרחקות. "
                 "במילים של חשבון דיפרנציאלי: הנגזרת השנייה היא מינוס הפונקציה עצמה. "
                 "וסינוס וקוסינוס הם הפתרונות הבסיסיים של המשוואה הזאת.")
        self.play(Write(ode))
        self.wait_speech()
        for m in (spring_mob, mass, trace2, dot2):
            m.clear_updaters()
        self.fade_all()


class Harmonics(FourierScene):
    def construct(self):
        rows = VGroup()
        for i, k in enumerate((1, 2, 3)):
            ax = pi_axes(width=5.0, height=1.3, y_range=(-1.2, 1.2, 1), labels=(i == 2), label_size=24)
            g = curve(ax, lambda x, k=k: np.sin(k * x), color=WAVES[i], width=3.5)
            lab = Tex(rf"\sin {k if k > 1 else ''}x", font_size=34).set_color(WAVES[i]).next_to(ax, LEFT, buff=0.3)
            rows.add(VGroup(ax, g, lab))
        rows.arrange(DOWN, buff=0.35).move_to(3.3 * LEFT + 0.5 * UP)
        crows = VGroup()
        for i, k in enumerate((1, 2, 3)):
            ax = pi_axes(width=5.0, height=1.3, y_range=(-1.2, 1.2, 1), labels=(i == 2), label_size=24)
            g = curve(ax, lambda x, k=k: np.cos(k * x), color=WAVES[i + 3], width=3.5)
            lab = Tex(rf"\cos {k if k > 1 else ''}x", font_size=34).set_color(WAVES[i + 3]).next_to(ax, RIGHT, buff=0.3)
            crows.add(VGroup(ax, g, lab))
        crows.arrange(DOWN, buff=0.35).move_to(3.3 * RIGHT + 0.5 * UP)

        self.caption("לא רק sin x: גם sin 2x, sin 3x וכן הלאה",
                     say="אבן הבניין שלנו היא לא רק סינוס איקס, אלא כל המשפחה: סינוס של שתיים איקס, "
                         "סינוס של שלוש איקס, וכן הלאה. סינוס של אן איקס עושה אן גלים שלמים בקטע מינוס פאי עד פאי.")
        for row in rows:
            self.play(FadeIn(row[0]), ShowCreation(row[1]), FadeIn(row[2]), run_time=1.2)
        self.wait_speech()
        self.caption("וגם קוסינוסים: cos x, cos 2x, cos 3x",
                     say="ואותו דבר עם קוסינוסים. לכל הגלים האלה יש מחזור של שתיים פאי, "
                         "גם אם חלקם חוזרים על עצמם עוד יותר מהר.")
        for row in crows:
            self.play(FadeIn(row[0]), ShowCreation(row[1]), FadeIn(row[2]), run_time=1.2)
        self.wait_speech()
        self.play(FadeOut(rows), FadeOut(crows))

        ax = pi_axes(width=9, height=4.2, y_range=(-2.2, 2.2, 1)).move_to(1.6 * LEFT + 0.3 * UP)
        c1, c2, c3 = ValueTracker(1), ValueTracker(0), ValueTracker(0)

        def mix(x):
            return c1.get_value() * np.sin(x) + c2.get_value() * np.sin(2 * x) + c3.get_value() * np.cos(3 * x)

        graph = always_redraw(lambda: curve(ax, mix, color=SUM_COLOR, width=4))
        formula = Tex(R"f(x) = a\,\sin x + b\,\sin 2x + c\,\cos 3x", font_size=40).to_edge(UP, buff=0.25)
        knobs = VGroup()
        for i, (tracker, name) in enumerate(((c1, "a"), (c2, "b"), (c3, "c"))):
            track = Line(1.4 * DOWN, 1.4 * UP).set_stroke(GREY_D, 6)
            knob = always_redraw(lambda t=tracker, tr=track, c=WAVES[i]: Dot(
                tr.get_center() + t.get_value() * UP, radius=0.13, fill_color=c))
            label = Tex(name, font_size=34).next_to(track, DOWN, buff=0.15)
            knobs.add(VGroup(track, label, knob))
        knobs.arrange(RIGHT, buff=0.7).move_to(5.0 * RIGHT + 0.4 * UP)
        for k in knobs:
            k[2].resume_updating()
        self.caption("המקדמים הם כפתורים: כמה מכל גל",
                     say="עכשיו נערבב. ניקח כמה גלים, ולכל אחד נקבע מקדם: כמה ממנו נכנס לסכום. "
                         "המקדמים הם כמו כפתורים בעמדת מיקסר. מזיזים אותם, והצורה משתנה.")
        self.play(FadeIn(ax), Write(formula), FadeIn(knobs), ShowCreation(graph))
        self.play(c2.animate.set_value(0.8), run_time=2)
        self.play(c3.animate.set_value(-0.6), run_time=2)
        self.play(c1.animate.set_value(-0.5), c2.animate.set_value(-0.4), run_time=2.5)
        self.play(c1.animate.set_value(1.3), c3.animate.set_value(0.9), run_time=2.5)
        self.wait_speech()
        graph.clear_updaters()
        self.fade_all()


class TheQuestion(FourierScene):
    def construct(self):
        ax = pi_axes((-3 * PI, 3 * PI), (-0.8, 1.8, 1), width=13, height=2.6).shift(1.2 * UP)
        center = curve(ax, skyline, -PI, PI, F_COLOR, 4, step=0.01, jumps=(-2.2, -1.2, 0.9, 2.0))
        copies = VGroup(*(center.copy().shift(ax.c2p(k * TAU, 0) - ax.c2p(0, 0)).set_stroke(F_COLOR, 3, 0.55)
                          for k in (-1, 1)))
        box = Rectangle(width=ax.c2p(PI, 0)[0] - ax.c2p(-PI, 0)[0], height=3.0).move_to(ax.c2p(0, 0.5))
        box.set_stroke(GREY_A, 1.5).set_fill(WHITE, 0.04)
        self.caption("אילו פונקציות מחזוריות אפשר לכתוב כך?",
                     say="השאלה הגדולה היא אילו פונקציות מחזוריות אפשר לכתוב כצירוף של סינוסים וקוסינוסים. "
                         "פונקציה מחזורית עם מחזור שתיים פאי נקבעת לגמרי על ידי מה שקורה בקטע אחד באורך שתיים פאי, "
                         "והשאר הם עותקים.")
        self.play(FadeIn(ax), ShowCreation(center), run_time=2)
        self.play(FadeIn(box))
        self.play(*(TransformFromCopy(center, c) for c in copies), run_time=2)
        self.wait_speech()
        self.say("אז מעכשיו נדבר על פונקציות על הקטע ממינוס פאי עד פאי. והתשובה לשאלה, "
                 "וזו תשובה די מפתיעה, היא: הרבה. אפילו פונקציה עם מדרגות ופינות כמו זו.")
        self.play(FadeOut(copies), FadeOut(box))
        a, b = coeffs_numeric(skyline, 120)
        n = ValueTracker(1)
        approx = always_redraw(lambda: curve(ax, lambda x: partial(x, a, b, n.get_value()), -PI, PI, SUM_COLOR, 3))
        self.add(approx)
        self.play(n.animate.set_value(60), run_time=7, rate_func=lambda t: t ** 2)
        approx.clear_updaters()
        self.wait_speech()

        formula = Tex(R"f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} a_n \cos nx + \sum_{n=1}^{\infty} b_n \sin nx",
                      font_size=48).move_to(1.4 * DOWN)
        self.caption("השאלה הבוערת: איך מוצאים את המקדמים?",
                     say="ככה נראה טור פורייה. מספר קבוע, ואחריו מקדם לכל קוסינוס ומקדם לכל סינוס. "
                         "הקבוע כתוב כחצי אַ אפס, ותכף נבין למה. והשאלה הבוערת: איך מוצאים את המקדמים?")
        self.play(Write(formula), run_time=2.5)
        self.play(*(Indicate(formula[s], color=YELLOW) for s in ("a_n", "b_n")))
        self.wait_speech()
        self.fade_all()


class Vectors(FourierScene):
    def construct(self):
        plane = NumberPlane((-1, 5), (-1, 4), width=6, height=5,
                            background_line_style=dict(stroke_color=GREY_D, stroke_width=1)).move_to(3.3 * LEFT + 0.5 * UP)
        o = plane.c2p(0, 0)
        v = np.array([3.0, 2.0])
        arrow = Arrow(o, plane.c2p(*v), buff=0).set_color(F_COLOR)
        v_lab = Tex(R"\vec v", font_size=40).set_color(F_COLOR).next_to(plane.c2p(*v), UR, buff=0.1)
        self.caption("וקטור והקואורדינטות שלו",
                     say="כדי להבין איך מוצאים את המקדמים, נעשה עיקוף קטן דרך וקטורים. "
                         "הנה וקטור במישור. הקואורדינטות שלו אומרות כמה ממנו הולך בכל כיוון.")
        self.play(FadeIn(plane), GrowArrow(arrow), FadeIn(v_lab))

        def basis_arrows(angle):
            e1, e2 = direction(angle), direction(angle + PI / 2)
            unit = plane.c2p(1, 0)[0] - o[0]
            return VGroup(Arrow(o, o + unit * e1, buff=0).set_color(BLUE),
                          Arrow(o, o + unit * e2, buff=0).set_color(GREEN)), e1, e2

        (basis, e1, e2) = basis_arrows(0)
        self.play(GrowArrow(basis[0]), GrowArrow(basis[1]))
        self.wait_speech()

        angle = ValueTracker(0)

        def projections():
            a = angle.get_value()
            e1, e2 = direction(a)[:2], direction(a + PI / 2)[:2]
            p1, p2 = np.dot(v, e1) * e1, np.dot(v, e2) * e2
            return VGroup(
                DashedLine(plane.c2p(*v), plane.c2p(*p1)).set_stroke(BLUE_B, 2),
                DashedLine(plane.c2p(*v), plane.c2p(*p2)).set_stroke(GREEN_B, 2),
                Line(o, plane.c2p(*p1)).set_stroke(BLUE, 7, 0.6),
                Line(o, plane.c2p(*p2)).set_stroke(GREEN, 7, 0.6))

        proj = always_redraw(projections)
        moving_basis = always_redraw(lambda: basis_arrows(angle.get_value())[0])
        self.remove(basis)
        self.add(moving_basis)
        coords = Tex(R"\vec v = \langle \vec v, e_1 \rangle\, e_1 + \langle \vec v, e_2 \rangle\, e_2",
                     font_size=40).move_to(3.4 * RIGHT + 1.8 * UP)
        coords[R"e_1"].set_color(BLUE)
        coords[R"e_2"].set_color(GREEN)
        dotdef = Tex(R"\langle \vec u, \vec w \rangle = u_1 w_1 + u_2 w_2", font_size=36).next_to(coords, DOWN, buff=0.5)
        self.caption("הקואורדינטה = מכפלה פנימית עם וקטור הבסיס",
                     say="והנה הטריק: אם הבסיס אורתונורמלי, כלומר שני חצים באורך אחת שניצבים זה לזה, "
                         "אז כל קואורדינטה היא פשוט המכפלה הפנימית של הוקטור עם החץ המתאים. "
                         "גיאומטרית, זה הצל של הוקטור על הכיוון הזה.")
        self.add(proj)
        self.play(Write(coords))
        self.play(FadeIn(dotdef))
        self.wait_speech()
        self.say("וזה עובד בכל בסיס אורתונורמלי. מסובבים את החצים, והמכפלה הפנימית עדיין נותנת את המקדמים הנכונים.")
        self.play(angle.animate.set_value(0.55), run_time=3)
        self.play(angle.animate.set_value(-0.35), run_time=3)
        self.wait_speech()
        for m in (proj, moving_basis):
            m.clear_updaters()
        self.fade_all()

        # פונקציה כוקטור ארוך
        ax = pi_axes(width=10, height=3.0, y_range=(-1.6, 1.6, 1)).shift(1.0 * UP)
        f = lambda x: np.sin(x) + 0.5 * np.cos(2 * x) - 0.3 * np.sin(3 * x)
        graph = curve(ax, f, color=F_COLOR, width=3).set_stroke(opacity=0.35)
        self.caption("פונקציה היא וקטור עם אינסוף קואורדינטות",
                     say="עכשיו המעבר הגדול. פונקציה היא גם סוג של וקטור. דגמו אותה בשמונה נקודות, "
                         "וקיבלתם וקטור עם שמונה קואורדינטות.")
        self.play(FadeIn(ax), ShowCreation(graph))
        stems = None
        for n_pts in (8, 24, 90):
            xs = np.linspace(-PI, PI, n_pts + 2)[1:-1]
            new = VGroup(*(VGroup(Line(ax.c2p(x, 0), ax.c2p(x, f(x))).set_stroke(F_COLOR, 3 if n_pts < 50 else 1.5),
                                  Dot(ax.c2p(x, f(x)), radius=0.06 if n_pts < 50 else 0.03, fill_color=F_COLOR))
                           for x in xs))
            if stems is None:
                self.play(LaggedStart(*(FadeIn(m) for m in new), lag_ratio=0.1), run_time=2)
            else:
                self.play(FadeOut(stems), FadeIn(new), run_time=1.5)
            stems = new
            if n_pts == 8:
                self.wait_speech()
                self.say("עשרים וארבע נקודות, תשעים נקודות. ובגבול, אינסוף קואורדינטות, אחת לכל איקס.")
        self.play(FadeOut(stems), graph.animate.set_stroke(opacity=1, width=4))
        self.wait_speech()

        steps = VGroup(
            Tex(R"\langle \vec u, \vec w \rangle = \sum_i u_i\, w_i", font_size=42),
            Tex(R"\longrightarrow", font_size=42),
            Tex(R"\int_{-\pi}^{\pi} f(x)\, g(x)\, dx", font_size=42),
        ).arrange(RIGHT, buff=0.5).move_to(1.6 * DOWN)
        inner = Tex(R"\langle f, g \rangle = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x)\, g(x)\, dx", font_size=46)
        inner.move_to(steps)
        self.caption("סכום של מכפלות הופך לאינטגרל",
                     say="והמכפלה הפנימית? בוקטורים כופלים קואורדינטה בקואורדינטה ומחברים. "
                         "בפונקציות כופלים ערך בערך, ובמקום סכום מקבלים אינטגרל.")
        self.play(Write(steps[0]))
        self.play(GrowFromCenter(steps[1]), Write(steps[2]))
        self.wait_speech()
        self.caption("המכפלה הפנימית של פונקציות",
                     say="גדי מוסיף לזה גורם של אחד חלקי פאי, ותכף נבין למה. "
                         "זו המכפלה הפנימית שתלווה אותנו כל הסרטון.")
        self.play(FadeOut(steps), FadeIn(inner, scale=1.1))
        self.play(Indicate(inner[R"\frac{1}{\pi}"], color=YELLOW))
        self.wait_speech()
        self.fade_all()


class Orthogonal(FourierScene):
    def product_panel(self, f, g, f_tex, g_tex, f_color=BLUE, g_color=GREEN):
        top = pi_axes(width=6.2, height=1.6, y_range=(-1.2, 1.2, 1), labels=False).move_to(3.4 * LEFT + 1.9 * UP)
        bottom = pi_axes(width=6.2, height=1.9, y_range=(-1.2, 1.2, 1)).move_to(3.4 * LEFT + 0.6 * DOWN)
        gf = curve(top, f, color=f_color, width=3)
        gg = curve(top, g, color=g_color, width=3)
        labels = VGroup(Tex(f_tex, font_size=32).set_color(f_color), Tex(g_tex, font_size=32).set_color(g_color))
        labels.arrange(DOWN, aligned_edge=LEFT, buff=0.1).next_to(top, RIGHT, buff=0.2)
        prod = curve(bottom, lambda x: f(x) * g(x), color=WHITE, width=3)
        area = signed_area(bottom, lambda x: f(x) * g(x))
        return VGroup(top, gf, gg, labels), VGroup(bottom, area, prod)

    def construct(self):
        f, g = (lambda x: np.sin(2 * x)), (lambda x: np.sin(3 * x))
        waves, prod = self.product_panel(f, g, R"\sin 2x", R"\sin 3x")
        value = Tex(R"\langle \sin 2x, \sin 3x \rangle = 0", font_size=40).move_to(3.4 * RIGHT + 1.6 * UP)
        self.caption("כופלים את שתי הפונקציות, ומסתכלים על השטח",
                     say="בואו נבדוק מה המכפלה הפנימית של סינוס שתיים איקס וסינוס שלוש איקס. "
                         "כופלים את שתי הפונקציות נקודה בנקודה, ומסתכלים על השטח שמתחת לגרף של המכפלה.")
        self.play(FadeIn(waves[0]), ShowCreation(waves[1]), ShowCreation(waves[2]), FadeIn(waves[3]))
        self.play(FadeIn(prod[0]), ShowCreation(prod[2]), run_time=1.5)
        self.play(FadeIn(prod[1]))
        self.wait_speech()
        self.caption("הירוק והאדום מתקזזים בדיוק: האינטגרל הוא אפס",
                     say="השטח הירוק נמצא מעל הציר, האדום מתחתיו, והם מקזזים זה את זה בדיוק. "
                         "האינטגרל יוצא אפס. כלומר, שתי הפונקציות האלה ניצבות זו לזו, כמו שני חצים בזווית ישרה.")
        self.play(Write(value))
        self.wait_speech()

        identity = Tex(R"\sin 2x \sin 3x = \frac{\cos x - \cos 5x}{2}", font_size=38).next_to(value, DOWN, buff=0.6)
        mini = pi_axes(width=5.4, height=1.6, y_range=(-1.2, 1.2, 1), labels=False).next_to(identity, DOWN, buff=0.4)
        c1 = curve(mini, lambda x: np.cos(x) / 2, color=ORANGE, width=2.5)
        c5 = curve(mini, lambda x: -np.cos(5 * x) / 2, color=PURPLE_B, width=2.5)
        self.caption("זהות טריגונומטרית: מכפלה הופכת לסכום של גלים",
                     say="למה זה קורה? יש זהות טריגונומטרית שהופכת מכפלה של שני סינוסים לסכום של שני קוסינוסים רגילים. "
                         "וכל קוסינוס כזה עושה מספר שלם של גלים בקטע, ולכן השטח שלו מתאפס.")
        self.play(Write(identity))
        self.play(FadeIn(mini), ShowCreation(c1), ShowCreation(c5))
        self.wait_speech()
        self.play(FadeOut(VGroup(waves, prod, value, identity, mini, c1, c5)))

        f2 = lambda x: np.sin(3 * x)
        waves, prod = self.product_panel(f2, f2, R"\sin 3x", R"\sin 3x", BLUE, BLUE)
        value = Tex(R"\int_{-\pi}^{\pi} \sin^2 3x\, dx = \pi", font_size=40).move_to(3.4 * RIGHT + 1.6 * UP)
        norm = Tex(R"\langle \sin 3x, \sin 3x \rangle = \frac{1}{\pi}\cdot \pi = 1", font_size=40)
        norm.next_to(value, DOWN, buff=0.5)
        self.caption("פונקציה כפול עצמה: הכל חיובי",
                     say="ומה קורה כשכופלים את סינוס שלוש איקס בעצמו? עכשיו הכל חיובי, ושום דבר לא מתקזז. "
                         "השטח יוצא בדיוק פאי.")
        self.play(FadeIn(waves[0]), ShowCreation(waves[1]), FadeIn(waves[3][0]))
        self.play(FadeIn(prod[0]), ShowCreation(prod[2]), FadeIn(prod[1]), run_time=1.5)
        self.play(Write(value))
        self.wait_speech()
        self.caption("בזכות החלוקה ב-π, האורך של כל גל הוא 1",
                     say="ובגלל זה גדי מחלק בפאי: עם החלוקה, האורך של כל גל כזה יוצא בדיוק אחת.")
        self.play(Write(norm))
        self.wait_speech()
        self.play(FadeOut(VGroup(waves, prod, value, norm)))

        names = [R"1", R"\cos x", R"\sin x", R"\cos 2x", R"\sin 2x", R"\cos 3x", R"\sin 3x"]
        funcs = [lambda x: 1 + 0 * x, np.cos, np.sin, lambda x: np.cos(2 * x), lambda x: np.sin(2 * x),
                 lambda x: np.cos(3 * x), lambda x: np.sin(3 * x)]
        cell = 0.78
        grid = VGroup()
        for i, fi in enumerate(funcs):
            for j, fj in enumerate(funcs):
                val = integral(lambda x: fi(x) * fj(x)) / PI
                sq = Square(cell).set_stroke(GREY_D, 1)
                sq.set_fill(YELLOW if i == j and i else (ORANGE if i == j else BG), 0.75 if i == j else 1)
                txt = Tex(f"{val:.0f}" if abs(val) > 0.5 else "0", font_size=28)
                txt.set_color(BLACK if i == j else GREY_B)
                grid.add(VGroup(sq, txt).move_to(np.array([j * cell, -i * cell, 0])))
        grid.move_to(0.9 * RIGHT + 0.4 * UP)
        row_labels = VGroup(*(Tex(n, font_size=26).next_to(grid[i * 7], LEFT, buff=0.2) for i, n in enumerate(names)))
        col_labels = VGroup(*(Tex(n, font_size=24).next_to(grid[j], UP, buff=0.2) for j, n in enumerate(names)))
        self.caption("כל הגלים ניצבים זה לזה, ואורך כל אחד 1",
                     say="אם נחשב את כל המכפלות הפנימיות בין הגלים, נקבל טבלה שכולה אפסים, חוץ מאחדות על האלכסון. "
                         "בדיוק כמו חצים ניצבים באורך אחת: הגלים הם בסיס אורתונורמלי.")
        self.play(FadeIn(row_labels), FadeIn(col_labels))
        self.play(LaggedStart(*(FadeIn(c) for c in grid), lag_ratio=0.02, run_time=3))
        self.wait_speech()
        self.caption("חריג אחד: הפונקציה הקבועה 1, שהאורך שלה בריבוע הוא 2",
                     say="כמעט. יש חריג אחד: הפונקציה הקבועה אחת, שהיא בעצם קוסינוס של אפס איקס. "
                         "המכפלה שלה עם עצמה יוצאת שתיים ולא אחת. נזכור את זה, זה יחזור עוד מעט.")
        self.play(Indicate(grid[0], color=ORANGE, scale_factor=1.3), run_time=1.5)
        self.wait_speech()
        self.fade_all()


class Detector(FourierScene):
    def construct(self):
        secret = {"cos x": 0.0, "sin x": 1.5, "cos 2x": -0.5, "sin 2x": 0.0, "cos 3x": 0.0, "sin 3x": 0.8}

        def f(x):
            return 1.5 * np.sin(x) - 0.5 * np.cos(2 * x) + 0.8 * np.sin(3 * x)

        ax = pi_axes(width=6.4, height=2.0, y_range=(-2.5, 2.5, 1), labels=False).move_to(3.3 * LEFT + 2.0 * UP)
        graph = curve(ax, f, color=F_COLOR, width=3.5)
        f_lab = Tex("f(x) = ?", font_size=36).set_color(F_COLOR).next_to(ax, UP, buff=0.1)
        self.caption("פונקציה מסתורית: אילו גלים יש בה?",
                     say="עכשיו אפשר למצוא מקדמים. הנה פונקציה מסתורית, ואנחנו רוצים לדעת מאילו גלים היא בנויה. "
                         "בדיוק כמו בוקטורים, המקדם של כל גל הוא המכפלה הפנימית של הפונקציה עם הגל.")
        self.play(FadeIn(ax), ShowCreation(graph), FadeIn(f_lab))
        self.wait_speech()

        detectors = [("cos x", np.cos), ("sin x", np.sin), ("cos 2x", lambda x: np.cos(2 * x)),
                     ("sin 2x", lambda x: np.sin(2 * x)), ("cos 3x", lambda x: np.cos(3 * x)),
                     ("sin 3x", lambda x: np.sin(3 * x))]
        chart_origin = np.array([0.8, -0.8, 0])
        slot = 1.0
        base = Line(chart_origin, chart_origin + 6 * slot * RIGHT).set_stroke(AXIS, 2)
        labels = VGroup(*(Tex(R"\%s" % n.split()[0] + (" " + n.split()[1] if " " in n else ""), font_size=26)
                          .move_to(chart_origin + (i + 0.5) * slot * RIGHT + 1.0 * DOWN) for i, (n, _) in enumerate(detectors)))
        self.play(ShowCreation(base), FadeIn(labels))
        self.say("כופלים את הפונקציה בגל הבודק, ומודדים את השטח. אם הגל לא נמצא בפונקציה, הכל מתקזז ויוצא אפס. "
                 "אם הוא כן נמצא, נשאר שטח, והוא מספר לנו בדיוק כמה.")
        low = pi_axes(width=6.4, height=2.0, y_range=(-2.5, 2.5, 1), labels=False).move_to(3.3 * LEFT + 0.6 * DOWN)
        self.play(FadeIn(low))
        previous = None
        for i, (name, g) in enumerate(detectors):
            det = curve(ax, g, color=BLUE, width=2.5)
            prod = lambda x, g=g: f(x) * g(x)
            area = signed_area(low, prod)
            pcurve = curve(low, prod, color=WHITE, width=2)
            val = integral(prod) / PI
            h = abs(val) * 0.9
            bar = Rectangle(width=0.6 * slot, height=max(h, 0.01)).set_stroke(width=0).set_fill(SUM_COLOR if val >= 0 else NEG, 0.9)
            bar.move_to(chart_origin + (i + 0.5) * slot * RIGHT + (h / 2 if val >= 0 else -h / 2) * UP)
            num = DecimalNumber(val, num_decimal_places=1, include_sign=False, font_size=28)
            if abs(val) < 0.05:
                num = Tex("0", font_size=28)
            num.next_to(bar, UP if val >= 0 else DOWN, buff=0.1)
            anims = [ShowCreation(det), FadeIn(area), ShowCreation(pcurve)]
            if previous is not None:
                anims = [FadeOut(m) for m in previous] + anims
            self.play(*anims, run_time=1.0 if i else 1.5)
            self.play(GrowFromEdge(bar, DOWN if val >= 0 else UP), FadeIn(num), run_time=0.8)
            self.wait(0.6 if i else 1.2)
            previous = (det, area, pcurve)
        self.wait_speech()
        reveal = Tex(R"f(x) = 1.5\,\sin x - 0.5\,\cos 2x + 0.8\,\sin 3x", font_size=36).set_color(F_COLOR)
        reveal.move_to(f_lab)
        self.caption("הגלאים מצאו את הפונקציה במדויק",
                     say="והנה התשובה: אחת וחצי סינוס איקס, פחות חצי קוסינוס שתיים איקס, ועוד שמונה עשיריות סינוס שלוש איקס. "
                         "הגלאים מצאו כל רכיב, בדיוק בכמות הנכונה.")
        self.play(*(FadeOut(m) for m in previous), FadeTransform(f_lab, reveal))
        self.wait_speech()
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None

        formulas = VGroup(
            Tex(R"a_n = \langle f, \cos nx \rangle = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x) \cos nx \, dx", font_size=44),
            Tex(R"b_n = \langle f, \sin nx \rangle = \frac{1}{\pi}\int_{-\pi}^{\pi} f(x) \sin nx \, dx", font_size=44),
        ).arrange(DOWN, buff=0.5).move_to(1.3 * UP)
        self.caption("הנוסחאות למקדמי פורייה",
                     say="ובאופן כללי, אלה הנוסחאות למקדמי פורייה. אַ אֶן הוא המכפלה הפנימית עם קוסינוס אן איקס, "
                         "ובֶּה אֶן עם סינוס אן איקס.")
        self.play(Write(formulas[0]), run_time=2)
        self.play(Write(formulas[1]), run_time=2)
        self.wait_speech()
        half = Tex(R"f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left( a_n \cos nx + b_n \sin nx \right)",
                   font_size=44).next_to(formulas, DOWN, buff=0.7)
        self.caption("החצי שליד a₀: האורך בריבוע של הקבוע 1 הוא 2",
                     say="ועכשיו מובן החצי המוזר שליד אַ אפס. המכפלה הפנימית עם הקבוע אחת מחזירה את המקדם שלו כפול שתיים, "
                         "כי זה האורך בריבוע של הקבוע. אז כותבים אַ אפס חלקי שתיים, ומרוויחים נוסחה אחת לכל המקדמים, בלי מקרים מיוחדים.")
        self.play(Write(half), run_time=2)
        self.play(Indicate(half[R"\frac{a_0}{2}"], color=ORANGE, scale_factor=1.4), run_time=1.5)
        self.wait_speech()
        self.fade_all()


class BestApprox(FourierScene):
    def construct(self):
        # וקטור מעל מישור: הצל הוא הנקודה הקרובה ביותר
        o = np.array([-3.4, -1.3, 0])
        u1, u2 = np.array([2.6, -0.5, 0]), np.array([1.3, 1.0, 0])
        plane = Polygon(o - 1.1 * u1 - 1.0 * u2, o + 1.3 * u1 - 1.0 * u2, o + 1.3 * u1 + 1.3 * u2, o - 1.1 * u1 + 1.3 * u2)
        plane.set_stroke(GREY_B, 1.5).set_fill(BLUE_E, 0.35)
        shadow = o + 0.75 * u1 + 0.55 * u2
        tip = shadow + 2.9 * UP
        v = Arrow(o, tip, buff=0).set_color(F_COLOR)
        proj = Arrow(o, shadow, buff=0).set_color(SUM_COLOR)
        drop = DashedLine(tip, shadow).set_stroke(RED, 3)
        self.caption("הצל על המישור הוא הנקודה הכי קרובה",
                     say="יש עוד סיבה שהמקדמים האלה נכונים, והיא מהפוסט של גדי על בסיסים אורתונורמליים. "
                         "נניח שיש לנו רק חלק מהבסיס, נגיד שני כיוונים, שפורשים מישור. "
                         "הצל של הוקטור על המישור, כלומר הסכום של המכפלות הפנימיות כפול הכיוונים, "
                         "הוא הנקודה במישור שהכי קרובה לוקטור.")
        self.play(FadeIn(plane), GrowArrow(v))
        self.play(GrowArrow(proj), ShowCreation(drop))
        self.play(FadeIn(right_angle_mark(shadow, UP, u1)))
        self.wait_speech()
        other = o + 0.3 * u1 + 0.95 * u2
        wrong = DashedLine(tip, other).set_stroke(GREY_B, 2)
        self.say("כל נקודה אחרת במישור רחוקה יותר. והשגיאה, הקו האדום, ניצבת למישור.")
        self.play(ShowCreation(wrong), FadeIn(Dot(other, fill_color=GREY_B)))
        self.wait_speech()

        bessel = Tex(R"\left\Vert f - \sum_{k=1}^{n} \langle f, e_k \rangle e_k \right\Vert^2 = \left\Vert f \right\Vert^2 - \sum_{k=1}^{n} \langle f, e_k \rangle^2",
                     font_size=36).move_to(3.0 * RIGHT + 2.4 * UP)
        self.caption("שוויון בסל: כל איבר נוסף רק מקטין את השגיאה",
                     say="ושוויון בסל אומר בדיוק כמה גדולה השגיאה: האורך בריבוע של הוקטור, פחות סכום הריבועים של המקדמים. "
                         "כל כיוון נוסף מוריד עוד משהו מהשגיאה, ואף פעם לא מגדיל אותה.")
        self.play(Write(bessel), run_time=2.5)
        self.wait_speech()
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None

        # אותו דבר בפונקציות: מנסים מקדם c לסינוס, והשגיאה מינימלית במקדם של פורייה
        ax = pi_axes(width=6.6, height=3.0, y_range=(-1.6, 1.6, 1)).move_to(3.3 * LEFT + 0.8 * UP)
        sq = square_graph(ax)
        c = ValueTracker(0.4)
        approx = always_redraw(lambda: curve(ax, lambda x: c.get_value() * np.sin(x), color=SUM_COLOR, width=4))
        err_area = always_redraw(lambda: diff_area(ax, square, lambda x: c.get_value() * np.sin(x)))
        err_ax = Axes((0, 2.2, 1), (0, 2.2, 1), width=5, height=3.2,
                      axis_config=dict(stroke_color=AXIS, stroke_width=2)).move_to(3.4 * RIGHT + 0.6 * UP)
        err_ax.add_coordinate_labels(font_size=20)
        err = lambda cc: (2 * PI - 2 * cc * 4 + cc * cc * PI) / PI
        err_curve = curve(err_ax, err, 0, 2.2, GREY_B, 2.5, step=0.01)
        err_dot = always_redraw(lambda: Dot(err_ax.c2p(c.get_value(), err(c.get_value())), fill_color=RED))
        c_lab = Tex("c", font_size=30).next_to(err_ax.x_axis, RIGHT, buff=0.1)
        e_lab = heb("השגיאה", 26, GREY_A).next_to(err_ax.y_axis, UP, buff=0.1)
        title = Tex(R"c \cdot \sin x", font_size=38).set_color(SUM_COLOR).next_to(ax, UP, buff=0.15)
        self.caption("מנסים לקרב את גל הריבוע עם c כפול sin x",
                     say="בואו נראה את זה בפונקציות. ננסה לקרב את גל הריבוע, הפונקציה הצהובה, "
                         "בעזרת סינוס אחד, כפול מקדם שאנחנו בוחרים. מימין, השגיאה כפונקציה של המקדם.")
        self.play(FadeIn(ax), ShowCreation(sq), FadeIn(title))
        self.add(err_area, approx)
        self.play(FadeIn(err_ax), ShowCreation(err_curve), FadeIn(c_lab), FadeIn(e_lab), FadeIn(err_dot))
        self.play(c.animate.set_value(2.0), run_time=4)
        self.play(c.animate.set_value(4 / PI), run_time=3)
        best = DecimalNumber(4 / PI, num_decimal_places=3, font_size=32).next_to(err_dot, UP, buff=0.3)
        best_tex = Tex(R"c = \frac{4}{\pi} \approx 1.273", font_size=36).move_to(err_ax.c2p(1.6, 2.0))
        self.wait_speech()
        self.caption("המינימום: בדיוק המקדם של פורייה",
                     say="השגיאה הכי קטנה בדיוק במקדם של פורייה, ארבע חלקי פאי. "
                         "הנוסחה לא רק נותנת מקדמים, היא נותנת את הקירוב הכי טוב שאפשר.")
        self.play(Write(best_tex), Flash(err_dot.get_center(), color=RED))
        self.wait_speech()
        for m in (approx, err_area, err_dot):
            m.clear_updaters()
        self.fade_all()


def right_angle_mark(vertex, d1, d2, size=0.22, color=GREY_B):
    d1, d2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
    return VMobject().set_points_as_corners(
        [vertex + size * d1, vertex + size * (d1 + d2), vertex + size * d2]).set_stroke(color, 2)


class SquareCoeffs(FourierScene):
    def construct(self):
        ax = pi_axes(width=6.4, height=2.0, y_range=(-1.4, 1.4, 1), labels=False).move_to(3.3 * LEFT + 2.1 * UP)
        sq = square_graph(ax)
        definition = Tex(R"f(x) = \begin{cases} -1 & -\pi \le x < 0 \\ \phantom{-}1 & 0 \le x \le \pi \end{cases}",
                         font_size=36).move_to(3.4 * RIGHT + 2.1 * UP)
        self.caption("גל הריבוע: 1 ומינוס 1 לסירוגין",
                     say="עכשיו לדוגמה האמיתית של גדי. פונקציה שמקבלת רק שני ערכים, אחת ומינוס אחת, "
                         "וקופצת ביניהם. היא ישרה וחדה, וסינוס עגול ונחמד. "
                         "איך סכום של הרבה סינוסים יכול לתת לנו את הזוועה הזאת?")
        self.play(FadeIn(ax), ShowCreation(sq), Write(definition))
        self.wait_speech()

        low = pi_axes(width=6.4, height=2.0, y_range=(-1.4, 1.4, 1)).move_to(3.3 * LEFT + 0.4 * DOWN)
        self.play(FadeIn(low))

        def show(g, color=BLUE, wait=1.0):
            det = curve(ax, g, color=color, width=2.5)
            prod = lambda x: square(x) * g(x)
            area = signed_area(low, prod)
            pcurve = curve(low, prod, color=WHITE, width=2, jumps=(0,))
            self.play(ShowCreation(det), FadeIn(area), ShowCreation(pcurve))
            self.wait(wait)
            return VGroup(det, area, pcurve)

        self.caption("קוסינוסים: כל השטחים מתקזזים",
                     say="נתחיל מהקוסינוסים. מכפלה של גל הריבוע בקוסינוס תמיד מתקזזת: "
                         "החלק השמאלי הוא תמונת ראי הפוכה של הימני. כל מקדמי הקוסינוס יוצאים אפס.")
        old = show(np.cos)
        self.play(FadeOut(old))
        old = show(lambda x: np.cos(2 * x))
        self.wait_speech()
        self.play(FadeOut(old))

        chart = bars([4 / (PI * k) if k % 2 else 0 for k in range(1, 10)], width=5.4, height=2.0,
                     labels=list(range(1, 10)), max_value=1.3)
        chart.move_to(3.4 * RIGHT + 0.8 * DOWN)
        b_lab = Tex("b_n", font_size=34).next_to(chart, UP, buff=0.2).shift(2 * LEFT)
        self.play(ShowCreation(chart[0]), FadeIn(chart[2]), FadeIn(b_lab))
        texts = {1: "סינוס איקס: המכפלה חיובית כולה. השטח ארבע, ואחרי חלוקה בפאי, ארבע חלקי פאי.",
                 2: "סינוס שתיים איקס: הכל מתקזז. אפס.",
                 3: "סינוס שלוש איקס: שתי גבעות למעלה ואחת למטה בכל צד. נשארת גבעה אחת, ומקבלים שליש מהקודם."}
        for k in range(1, 10):
            g = lambda x, k=k: np.sin(k * x)
            if k in texts:
                self.say(texts[k])
            old = show(g, wait=0.3 if k > 3 else 0.8)
            self.play(GrowFromEdge(chart.rects[k - 1], DOWN), run_time=0.6)
            if k in texts:
                self.wait_speech()
            self.play(FadeOut(old), run_time=0.4)
        result = Tex(R"b_n = \begin{cases} \dfrac{4}{\pi n} & n = 1, 3, 5, \dots \\[4pt] 0 & n = 2, 4, 6, \dots \end{cases}",
                     font_size=36).move_to(3.4 * RIGHT + 2.1 * UP)
        self.caption("המקדמים: 4 חלקי πn באי-זוגיים, 0 בזוגיים",
                     say="והתבנית ברורה. במספרים הזוגיים יוצא אפס, ובאי-זוגיים יוצא ארבע חלקי פאי אן. "
                         "אפשר לראות את זה גם מהחישוב: הכל תלוי בשאלה אם קוסינוס של אן פאי הוא אחת או מינוס אחת.")
        self.play(FadeOut(definition), Write(result))
        self.wait_speech()
        series = Tex(R"f(x) = \frac{4}{\pi} \left( \sin x + \frac{\sin 3x}{3} + \frac{\sin 5x}{5} + \frac{\sin 7x}{7} + \cdots \right)",
                     font_size=40).move_to(2.6 * DOWN)
        self.clear_caption()
        self.say("ואם נאסוף הכל, זה טור הפורייה של גל הריבוע.")
        self.play(Write(series), run_time=2.5)
        self.wait(1.5)
        self.fade_all()


class BuildSquare(FourierScene):
    def construct(self):
        ax = pi_axes((-2 * PI, 2 * PI), (-1.6, 1.6, 0.5), width=12, height=3.6).shift(1.4 * UP)
        sq = square_graph(ax, -2 * PI, 2 * PI).set_stroke(opacity=0.45)
        comp_ax = pi_axes((-2 * PI, 2 * PI), (-1.4, 1.4, 1), width=12, height=1.6, labels=False).shift(1.6 * DOWN)
        a, b = square_coeffs(401)
        counter_lab = heb("מספר האיברים:", 30, GREY_A)
        count = Integer(1, font_size=36).set_color(SUM_COLOR)
        counter = VGroup(counter_lab, count).arrange(LEFT, buff=0.2).to_corner(UR, buff=0.4)
        self.caption("מוסיפים את הגלים אחד אחד",
                     say="ועכשיו החלק הקסום. נבנה את גל הריבוע ונוסיף את הגלים אחד אחד. "
                         "למטה הגל שנכנס עכשיו, ולמעלה הסכום.")
        self.play(FadeIn(ax), ShowCreation(sq), FadeIn(comp_ax), FadeIn(counter))
        total = curve(ax, lambda x: partial(x, a, b, 1), -2 * PI, 2 * PI, SUM_COLOR, 4)
        comp = curve(comp_ax, lambda x: b[1] * np.sin(x), -2 * PI, 2 * PI, WAVES[0], 3)
        self.play(ShowCreation(comp))
        self.play(TransformFromCopy(comp, total))
        self.wait_speech()
        self.say("סינוס שלוש איקס חלקי שלוש מוסיף כתפיים ומיישר את הראש. חמש, שבע, תשע, "
                 "וכל גל קטן יותר ומהיר יותר, ומתקן פרטים עדינים יותר.")
        for j, k in enumerate(range(3, 16, 2)):
            new_comp = curve(comp_ax, lambda x, k=k: b[k] * np.sin(k * x), -2 * PI, 2 * PI, WAVES[(j + 1) % 8], 3)
            new_total = curve(ax, lambda x, k=k: partial(x, a, b, k), -2 * PI, 2 * PI, SUM_COLOR, 4)
            self.play(FadeOut(comp), FadeIn(new_comp), run_time=0.5)
            self.play(Transform(total, new_total), run_time=1.2)
            count.set_value(j + 2).next_to(counter_lab, LEFT, buff=0.2)
            comp = new_comp
            self.wait(0.4)
        self.wait_speech()
        self.caption("שמונה גלים, ועכשיו בלי לעצור",
                     say="שמונה גלים, וזה כבר נראה די טוב. והמבנה הכללי זהה לפונקציה שאנחנו מנסים לקרב. עכשיו בלי לעצור.")
        self.play(FadeOut(comp), FadeOut(comp_ax))
        n = ValueTracker(15)
        self.remove(total)
        live = always_redraw(lambda: curve(ax, lambda x: partial(x, a, b, n.get_value()), -2 * PI, 2 * PI,
                                           SUM_COLOR, 4, step=0.002))
        count.add_updater(lambda m: m.set_value((int(n.get_value()) + 1) // 2).next_to(counter_lab, LEFT, buff=0.2))
        self.add(live)
        self.play(n.animate.set_value(401), run_time=10, rate_func=lambda t: t ** 2.2)
        live.clear_updaters()
        count.clear_updaters()
        self.wait_speech()
        self.caption("סינוסים עגולים, וגל ריבוע ישר וחד",
                     say="מאתיים גלים עגולים לגמרי, וביחד הם נראים כמעט בדיוק כמו הפונקציה הישרה והחדה. "
                         "אבל שימו לב לקצוות של המדרגות. משהו שם לא נעלם. נחזור לזה.")
        self.play(*(Flash(ax.c2p(x, 1.18), color=RED, flash_radius=0.35) for x in (-2 * PI + 0.02, 0.02, -PI - 0.02, PI - 0.02)))
        self.wait_speech()
        self.fade_all()


class Epicycles(FourierScene):
    def construct(self):
        scale = 1.25
        center = np.array([-4.3, 0.3, 0])
        ax = Axes((0, 4 * PI, PI), (-1.5, 1.5, 1), width=7.6, height=3.75,
                  axis_config=dict(stroke_color=AXIS, stroke_width=2))
        ax.shift(np.array([-1.4, center[1], 0]) - ax.c2p(0, 0))
        t = ValueTracker(0)
        count = ValueTracker(1)

        def ks():
            return [2 * j + 1 for j in range(int(count.get_value()))]

        def chain():
            group = VGroup()
            p = center.copy()
            for k in ks():
                r = scale * 4 / (PI * k)
                group.add(Circle(radius=r).move_to(p).set_stroke(GREY_B, 1, 0.5))
                q = p + r * np.array([np.cos(k * t.get_value()), np.sin(k * t.get_value()), 0])
                group.add(Line(p, q).set_stroke(WHITE, 2))
                p = q
            group.add(Dot(p, radius=0.06, fill_color=YELLOW))
            group.tip = p
            return group

        def wave():
            tt = t.get_value()
            xs = np.linspace(0, min(tt, 4 * PI), max(int(min(tt, 4 * PI) / 0.01), 2))
            ys = sum(4 / (PI * k) * np.sin(k * (tt - xs)) for k in ks())
            return VMobject().set_points_as_corners(ax.c2p(xs, ys * scale / 1.25)).set_stroke(SUM_COLOR, 3)  # ציר: 1.25 ליחידה

        circles = always_redraw(chain)
        trace = always_redraw(wave)
        link = always_redraw(lambda: DashedLine(circles.tip, ax.c2p(0, (circles.tip[1] - center[1]) / 1.25))
                             .set_stroke(GREY_B, 1.5))
        self.caption("כל סינוס הוא נקודה שמסתובבת על מעגל",
                     say="יש דרך יפה לראות את זה. זוכרים שסינוס הוא גובה של נקודה שמסתובבת על מעגל? "
                         "גל עם מקדם ותדירות הוא מעגל ברדיוס של המקדם, שמסתובב במהירות של התדירות.")
        self.play(FadeIn(ax))
        self.add(circles, trace, link)
        self.play(t.animate.set_value(2 * PI), run_time=5, rate_func=linear)
        self.wait_speech()
        self.caption("שרשרת של מעגלים: כל מעגל רוכב על הקודם",
                     say="עכשיו נשרשר מעגלים: כל מעגל רוכב על הקצה של הקודם. "
                         "מעגל שלישי שמסתובב פי שלוש מהר, מעגל חמישי פי חמש, וכן הלאה. "
                         "הגובה של הקצה האחרון הוא הסכום של כל הסינוסים, והוא מצייר גל ריבוע.")
        for c_val in (2, 3, 5, 10, 25):
            self.play(count.animate.set_value(c_val), run_time=0.3)
            self.play(t.animate.increment_value(2 * PI), run_time=4, rate_func=linear)
        self.wait_speech()
        for m in (circles, trace, link):
            m.clear_updaters()
        self.fade_all()



class NormConvergence(FourierScene):
    def construct(self):
        ax = pi_axes(width=9, height=3.6, y_range=(-1.6, 1.6, 1)).move_to(1.6 * LEFT + 1.0 * UP)
        sq = square_graph(ax)
        a, b = square_coeffs(201)
        approx = curve(ax, lambda x: partial(x, a, b, 15), color=SUM_COLOR, width=3.5)
        zero = Dot(ax.c2p(0, 0), radius=0.09, fill_color=RED)
        f0 = Dot(ax.c2p(0, 1), radius=0.09, fill_color=F_COLOR)
        self.caption("ב-x = 0 הטור נותן 0, אבל f(0) = 1",
                     say="עכשיו צריך להיזהר. נציב איקס שווה אפס. כל הסינוסים מתאפסים שם, אז הטור נותן אפס. "
                         "אבל לפי ההגדרה, הפונקציה שווה שם אחת. אז השוויון לא נכון בכל נקודה!")
        self.play(FadeIn(ax), ShowCreation(sq), ShowCreation(approx))
        self.play(FadeIn(f0, scale=2), FadeIn(zero, scale=2))
        self.wait_speech()
        avg = Tex(R"\frac{-1 + 1}{2} = 0", font_size=38).next_to(ax.c2p(0, 0), RIGHT, buff=0.6).shift(0.5 * DOWN)
        self.say("שימו לב שהאפס הזה לא מקרי. הוא בדיוק הממוצע של שני הערכים משני צדי הקפיצה. "
                 "הטור הוא פונקציה רציפה, ובאמצע הקפיצה הוא עובר בדיוק באמצע.")
        self.play(Write(avg))
        self.wait_speech()
        self.play(FadeOut(avg))

        chart = bars([4 / (PI * k) if k % 2 else 0 for k in range(1, 10)], width=3.6, height=1.6,
                     labels=list(range(1, 10)), max_value=1.3, label_size=18).move_to(4.6 * RIGHT + 0.8 * UP)
        self.caption("משנים את f בנקודה אחת: האינטגרלים לא זזים",
                     say="והסיבה העמוקה: המקדמים מחושבים באינטגרלים. אם נשנה את הפונקציה בנקודה אחת, "
                         "שום אינטגרל לא ישתנה, והטור יישאר בדיוק אותו טור. "
                         "אותו טור מתאר הרבה פונקציות, שנבדלות זו מזו בנקודות בודדות.")
        self.play(FadeIn(chart))
        for y in (0.4, -0.8, 1.4, 1.0):
            self.play(f0.animate.move_to(ax.c2p(0, y)), run_time=1.2)
        self.wait_speech()

        self.play(FadeOut(chart), FadeOut(f0), FadeOut(zero))
        n = ValueTracker(1)
        self.remove(approx)
        live = always_redraw(lambda: curve(ax, lambda x: partial(x, a, b, round(n.get_value())), color=SUM_COLOR, width=3.5))
        area = always_redraw(lambda: diff_area(ax, square, lambda x: partial(x, a, b, round(n.get_value())), opacity=0.45))
        err = lambda m: 2 - sum(b[k] ** 2 for k in range(1, m + 1))
        err_tex = Tex(R"\left\Vert f - S_n \right\Vert^2 =", font_size=36)
        err_num = DecimalNumber(err(1), num_decimal_places=3, font_size=36)
        err_num.add_updater(lambda m: m.set_value(err(round(n.get_value()))))
        err_group = VGroup(err_tex, err_num).arrange(RIGHT).move_to(4.9 * RIGHT + 2.6 * UP)
        err_num.add_updater(lambda m: m.next_to(err_tex, RIGHT))
        self.caption("השוויון אומר: המרחק בין f לסכום שואף לאפס",
                     say="אז מה כן אומר סימן השוויון? הוא אומר שהמרחק בין הפונקציה לסכום החלקי, "
                         "לפי הנורמה שבאה מהמכפלה הפנימית, שואף לאפס. השטח האדום הוא השגיאה, והוא הולך ונעלם. "
                         "לזה קוראים התכנסות בנורמה.")
        self.add(area, live, err_group)
        for m in (3, 5, 9, 17, 33, 65, 129):
            self.play(n.animate.set_value(m), run_time=1.3)
            self.wait(0.3)
        self.wait_speech()
        for m in (live, area, err_num):
            m.clear_updaters()
        self.fade_all()


class Gibbs(FourierScene):
    def construct(self):
        ax = Axes((-0.6, 1.6, 0.5), (-1.3, 1.5, 0.5), width=11, height=5.2,
                  axis_config=dict(stroke_color=AXIS, stroke_width=2)).shift(0.4 * UP)
        sq = curve(ax, square, -0.6, 1.6, F_COLOR, 4, step=0.01, jumps=(0,))
        a, b = square_coeffs(1281)
        level = DashedLine(ax.c2p(-0.6, 1.179), ax.c2p(1.6, 1.179)).set_stroke(RED, 2)
        level_lab = Tex(R"1.179", font_size=32).set_color(RED).next_to(level, RIGHT, buff=0.1)
        one_lab = Tex("1", font_size=30).next_to(ax.c2p(-0.6, 1), LEFT, buff=0.1)

        def peak(m):
            xs = np.linspace(1e-4, 4 / (m + 1) + 0.05, 4000)
            ys = partial(xs, a, b, m)
            i = int(np.argmax(ys))
            return xs[i], ys[i]

        self.caption("אפקט גיבס: הבליטה ליד הקפיצה",
                     say="ועכשיו לצ'ופצ'יקים, כמו שגדי קורא להם: הבליטות הקטנות ליד הקפיצה. בואו נעשה זום.")
        self.play(FadeIn(ax), ShowCreation(sq), FadeIn(one_lab))
        m0 = 9
        graph = curve(ax, lambda x: partial(x, a, b, m0), -0.6, 1.6, SUM_COLOR, 3.5, step=0.002)
        px, py = peak(m0)
        pdot = Dot(ax.c2p(px, py), radius=0.08, fill_color=RED)
        pval = DecimalNumber(py, num_decimal_places=3, font_size=34).set_color(RED).next_to(pdot, UR, buff=0.1)
        label = heb("מספר הגלים:", 30, GREY_A)
        count = Integer((m0 + 1) // 2, font_size=34).set_color(SUM_COLOR)
        counter = VGroup(label, count).arrange(LEFT, buff=0.2).to_corner(UR, buff=0.4)
        self.play(ShowCreation(graph), FadeIn(counter))
        self.play(FadeIn(pdot), FadeIn(pval))
        self.wait_speech()
        self.caption("יותר גלים: הבליטה נהיית צרה יותר, אבל לא נמוכה יותר",
                     say="נוסיף עוד ועוד גלים. הבליטה נהיית צרה יותר ויותר, והיא נדחקת אל הקפיצה. "
                         "אבל תסתכלו על הגובה שלה. הוא לא יורד. הוא נשאר תקוע בערך על אחת נקודה אחת שבע תשע.")
        for m in (19, 39, 79, 159, 319, 639, 1279):
            px, py = peak(m)
            new = curve(ax, lambda x, m=m: partial(x, a, b, m), -0.6, 1.6, SUM_COLOR, 3.5, step=0.0008)
            new_val = DecimalNumber(py, num_decimal_places=3, font_size=34).set_color(RED)
            self.play(Transform(graph, new), pdot.animate.move_to(ax.c2p(px, py)),
                      Transform(pval, new_val.next_to(ax.c2p(px, py), UR, buff=0.1)), run_time=1.4)
            count.set_value((m + 1) // 2).next_to(label, LEFT, buff=0.2)
            if m == 79:
                self.play(ShowCreation(level), FadeIn(level_lab))
        self.wait_speech()
        pct = heb("בערך תשעה אחוזים מגובה הקפיצה, לנצח", 34, RED).move_to(ax.c2p(0.95, 0.55))
        self.caption("הקירוב לא טוב באותה מידה בכל מקום",
                     say="הבליטה היא בערך תשעה אחוזים מגובה הקפיצה, והיא לא נעלמת אף פעם. "
                         "בכל נקודה קבועה הטור כן מתכנס, כי הבליטה בורחת ממנה אל הקפיצה. "
                         "אבל תמיד יש איפשהו נקודה עם שגיאה גדולה. בשפה של אינפי: ההתכנסות היא לא במידה שווה.")
        self.play(FadeIn(pct))
        self.wait_speech()
        self.caption("ווילברהאם ב-1848, גיבס ב-1899",
                     say="את התופעה גילה הנרי ווילברהאם עוד באמצע המאה התשע עשרה, אבל אף אחד לא שם לב. "
                         "חמישים שנה אחר כך, ג'וזיה וילארד גיבס תיאר אותה מחדש, והשם שלו נשאר.")
        self.wait_speech()
        self.fade_all()


class Symmetry(FourierScene):
    def construct(self):
        left = pi_axes(width=5.6, height=2.6, y_range=(-1.4, 1.4, 1)).move_to(3.5 * LEFT + 1.4 * UP)
        right = pi_axes(width=5.6, height=2.6, y_range=(-0.5, 3.6, 1)).move_to(3.5 * RIGHT + 1.4 * UP)
        odd = square_graph(left)
        even = curve(right, tri, color=F_COLOR, width=4)
        odd_lab = heb("אי-זוגית: רק סינוסים", 30).next_to(left, DOWN, buff=0.3)
        even_lab = heb("זוגית: רק קוסינוסים", 30).next_to(right, DOWN, buff=0.3)
        self.caption("איפה הקוסינוסים של גל הריבוע?",
                     say="שאלה: בטור של גל הריבוע היו רק סינוסים. לאן נעלמו הקוסינוסים? "
                         "זה לא מקרי. גל הריבוע הוא פונקציה אי-זוגית: אם מסובבים אותה חצי סיבוב סביב הראשית, היא חוזרת לעצמה. "
                         "בדיוק כמו סינוס.")
        self.play(FadeIn(left), ShowCreation(odd))
        ghost = odd.copy().set_stroke(ORANGE, 3)
        self.play(Rotate(ghost, PI, about_point=left.c2p(0, 0)), run_time=2)
        self.play(FadeOut(ghost), FadeIn(odd_lab))
        self.wait_speech()
        self.say("ופונקציה זוגית, כמו הפונקציה הזאת, היא תמונת ראי של עצמה, בדיוק כמו קוסינוס. "
                 "לפונקציה זוגית יש רק קוסינוסים, ולאי-זוגית יש רק סינוסים.")
        self.play(FadeIn(right), ShowCreation(even))
        ghost = even.copy().set_stroke(ORANGE, 3)
        self.play(ghost.animate.flip(UP, about_point=right.c2p(0, 0)), run_time=2)
        self.play(FadeOut(ghost), FadeIn(even_lab))
        self.wait_speech()
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None

        ax = pi_axes(width=10, height=3.6, y_range=(-1.5, 2.2, 1)).shift(1.0 * UP)
        g = lambda x: np.exp(x / 2) / 2.5
        gm = lambda x: np.exp(-x / 2) / 2.5
        graph = curve(ax, g, color=F_COLOR, width=4)
        mirror = DashedVMobject(curve(ax, gm, color=GREY_B, width=2.5), num_dashes=60)
        ev = curve(ax, lambda x: (g(x) + gm(x)) / 2, color=BLUE, width=4)
        od = curve(ax, lambda x: (g(x) - gm(x)) / 2, color=ORANGE, width=4)
        formula = Tex(R"f(x) = \frac{f(x) + f(-x)}{2} + \frac{f(x) - f(-x)}{2}", font_size=42).move_to(1.9 * DOWN)
        formula[R"\frac{f(x) + f(-x)}{2}"].set_color(BLUE)
        formula[R"\frac{f(x) - f(-x)}{2}"].set_color(ORANGE)
        self.caption("כל פונקציה = חלק זוגי + חלק אי-זוגי",
                     say="ומה עם פונקציה שאינה זוגית ואינה אי-זוגית? מסתבר שכל פונקציה היא סכום של חלק זוגי וחלק אי-זוגי. "
                         "לוקחים את הפונקציה ואת תמונת הראי שלה. הממוצע שלהן הוא החלק הזוגי, וחצי ההפרש הוא האי-זוגי.")
        self.play(FadeIn(ax), ShowCreation(graph))
        self.play(ShowCreation(mirror))
        self.play(TransformFromCopy(VGroup(graph, mirror), ev), run_time=1.5)
        self.play(TransformFromCopy(VGroup(graph, mirror), od), run_time=1.5)
        self.play(Write(formula))
        self.wait_speech()
        self.say("אז בטור פורייה, הקוסינוסים מתארים את החלק הזוגי של הפונקציה, והסינוסים את החלק האי-זוגי.")
        self.play(ev.animate.set_stroke(width=6), run_time=0.8)
        self.play(od.animate.set_stroke(width=6), run_time=0.8)
        self.wait_speech()
        self.fade_all()


class Sawtooth(FourierScene):
    def construct(self):
        ax = pi_axes((-2 * PI, 2 * PI), (-3.6, 3.6, 1), width=12, height=3.4).shift(1.6 * UP)
        target = curve(ax, saw, -2 * PI, 2 * PI, F_COLOR, 4, step=0.02, jumps=(-PI, PI)).set_stroke(opacity=0.6)
        self.caption("הדוגמה של גדי: f(x) = x",
                     say="עוד דוגמה מהפוסט של גדי: הפונקציה הכי פשוטה שיש, אֶף של איקס שווה איקס, על הקטע ממינוס פאי עד פאי. "
                         "כשממשיכים אותה מחזורית, מקבלים שן של מסור.")
        self.play(FadeIn(ax), ShowCreation(target), run_time=2)
        self.wait_speech()
        steps = VGroup(
            Tex(R"b_n = \frac{2}{\pi} \int_0^{\pi} x \sin nx \, dx", font_size=38),
            Tex(R"\int_0^{\pi} x \sin nx \, dx = \left[ -\frac{x \cos nx}{n} \right]_0^{\pi} + \int_0^{\pi} \frac{\cos nx}{n} \, dx = (-1)^{n+1} \frac{\pi}{n}", font_size=34),
            Tex(R"b_n = \frac{2}{n} (-1)^{n+1}", font_size=40),
        ).arrange(DOWN, buff=0.35).move_to(1.6 * DOWN)
        self.caption("פונקציה אי-זוגית: רק סינוסים. אינטגרציה בחלקים",
                     say="זו פונקציה אי-זוגית, אז יש בה רק סינוסים, ומספיק לחשב על חצי מהקטע. "
                         "עם אינטגרציה בחלקים מקבלים שהמקדם הוא שתיים חלקי אן, עם סימן שמתחלף.")
        for st in steps:
            self.play(Write(st), run_time=1.6)
        self.wait_speech()
        series = Tex(R"x = 2\sin x - \sin 2x + \frac{2}{3}\sin 3x - \frac{1}{2}\sin 4x + \cdots", font_size=42)
        series.move_to(1.6 * DOWN)
        self.play(FadeOut(steps[:2]), ReplacementTransform(steps[2], series))
        a, b = saw_coeffs(200)
        n = ValueTracker(1)
        live = always_redraw(lambda: curve(ax, lambda x: partial(x, a, b, n.get_value()), -2 * PI, 2 * PI,
                                           SUM_COLOR, 3.5, step=0.003))
        chart = bars([b[k] for k in range(1, 13)], width=5, height=1.0, labels=list(range(1, 13)),
                     max_value=2, label_size=18).move_to(3.2 * DOWN + 0.4 * UP)
        self.caption("סינוסים בונים קו ישר",
                     say="ושוב הקסם: מגלים עגולים, שעולים ויורדים, נבנה קו ישר לגמרי. "
                         "ובקצוות, שם הפונקציה המחזורית קופצת ממינוס פאי לפאי, מחכה לנו שוב אפקט גיבס.")
        self.add(live)
        self.play(n.animate.set_value(6), run_time=5, rate_func=linear)
        self.play(n.animate.set_value(120), run_time=7, rate_func=lambda t: t ** 2)
        live.clear_updaters()
        self.wait_speech()
        self.fade_all()


class Smoothness(FourierScene):
    def construct(self):
        ax = pi_axes((-2 * PI, 2 * PI), (-0.4, 3.6, 1), width=12, height=3.0).shift(1.9 * UP)
        target = curve(ax, tri, -2 * PI, 2 * PI, F_COLOR, 4, step=0.02).set_stroke(opacity=0.6)
        a, b = tri_coeffs(40)
        formula = Tex(R"|x| = \frac{\pi}{2} - \frac{4}{\pi}\left( \cos x + \frac{\cos 3x}{9} + \frac{\cos 5x}{25} + \cdots \right)",
                      font_size=40).move_to(0.2 * DOWN)
        self.caption("פונקציית המשולש: בלי קפיצות, רק פינות",
                     say="עכשיו פונקציה בלי קפיצות בכלל, רק עם פינות: הערך המוחלט של איקס, שהופך מחזורית לגל משולש. "
                         "היא זוגית, ולכן יש בה רק קוסינוסים, והמקדמים קטנים כמו אחד חלקי אן בריבוע.")
        self.play(FadeIn(ax), ShowCreation(target))
        self.play(Write(formula), run_time=2)
        approx = curve(ax, lambda x: partial(x, a, b, 0), -2 * PI, 2 * PI, SUM_COLOR, 3.5)
        self.play(ShowCreation(approx))
        for m in (1, 3, 5):
            self.play(Transform(approx, curve(ax, lambda x, m=m: partial(x, a, b, m), -2 * PI, 2 * PI, SUM_COLOR, 3.5)),
                      run_time=1.4)
            self.wait(0.5)
        self.wait_speech()
        self.say("רק שלושה גלים, וכבר קשה להבדיל בין הקירוב לפונקציה. הרבה יותר מהר מגל הריבוע.")
        self.wait_speech()
        self.play(FadeOut(formula))

        sq_vals = [4 / (PI * k) if k % 2 else 0 for k in range(1, 16)]
        tri_vals = [4 / (PI * k * k) if k % 2 else 0 for k in range(1, 16)]
        smooth_a, smooth_b = coeffs_numeric(lambda x: np.exp(np.sin(x)), 15)
        sm_vals = [np.hypot(smooth_a[k], smooth_b[k]) for k in range(1, 16)]
        charts = VGroup()
        for vals, name, color in ((sq_vals, "קפיצה", RED), (tri_vals, "פינה", ORANGE), (sm_vals, "חלקה", GREEN)):
            ch = bars([v / vals[0] for v in vals], width=3.8, height=1.9, color=color, max_value=1)
            ch.add(heb(name, 30, color).next_to(ch, DOWN, buff=0.25))
            charts.add(ch)
        charts.arrange(RIGHT, buff=0.7).move_to(1.2 * DOWN)
        self.caption("ככל שהפונקציה חלקה יותר, המקדמים קטנים מהר יותר",
                     say="וזה כלל כללי. הנה גודל המקדמים של שלוש פונקציות. בגל הריבוע, שיש בו קפיצה, הם קטנים לאט, כמו אחד חלקי אן. "
                         "במשולש, שיש בו רק פינות, כמו אחד חלקי אן בריבוע. ובפונקציה חלקה לגמרי הם נעלמים כמעט מיד. "
                         "ככל שהפונקציה חלקה יותר, צריך פחות גלים כדי לתאר אותה.")
        for ch in charts:
            self.play(FadeIn(ch[0]), LaggedStart(*(GrowFromEdge(r, DOWN) for r in ch[1]), lag_ratio=0.08), FadeIn(ch[-1]),
                      run_time=1.6)
        self.wait_speech()
        self.say("ויש לזה הסבר יפה: הנגזרת של גל המשולש היא בדיוק גל הריבוע, וגזירה כופלת כל מקדם באן. "
                 "כל נגזרת שהפונקציה מוכנה לתת לנו, קונה עוד חזקה של אן במכנה.")
        self.wait_speech()
        self.fade_all()


class Fejer(FourierScene):
    def construct(self):
        rows = VGroup(
            heb("רציפה וגזירה: הטור מתכנס בכל נקודה", 34),
            heb("קפיצה: בנקודת הקפיצה, מתכנס לממוצע", 34),
            heb("רק רציפה: הטור עלול להתבדר!", 34, RED),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.4).move_to(1.3 * UP)
        self.caption("מתי הטור מתכנס בכל נקודה?",
                     say="אז מתי בכל זאת יש התכנסות בכל נקודה? יש כמה משפטים. "
                         "אחד הפשוטים: אם הפונקציה רציפה וגזירה, הטור מתכנס אליה בכל נקודה. "
                         "אם יש קפיצות, בנקודות הקפיצה הוא מתכנס לממוצע, כמו שראינו.")
        self.play(FadeIn(rows[0]))
        self.play(FadeIn(rows[1]))
        self.wait_speech()
        self.say("אבל אם הפונקציה רק רציפה, בלי שליטה על הנגזרת, היא יכולה להיות די משוגעת. "
                 "יש פונקציות רציפות שטור הפורייה שלהן לא מתכנס בכלל בנקודות מסוימות.")
        self.play(FadeIn(rows[2]))
        self.wait_speech()
        self.play(rows.animate.scale(0.6).to_corner(UR, buff=0.4).set_opacity(0.5))

        sums = Tex(R"\sigma_m(x) = \frac{S_0(x) + S_1(x) + \cdots + S_m(x)}{m + 1}", font_size=42).move_to(1.6 * UP + 1.5 * LEFT)
        self.caption("הפתרון: ממוצע של הסכומים החלקיים",
                     say="מה עושים? במקום להסתכל על הסכומים החלקיים עצמם, לוקחים את הממוצע שלהם. "
                         "לשיטה הזאת קוראים סכימת צזארו. ובשנת אלף תשע מאות הוכיח ליפוט פייר, אז סטודנט בן עשרים, "
                         "שעבור כל פונקציה רציפה הממוצעים האלה מתכנסים לפונקציה. בכל נקודה.")
        self.play(Write(sums), run_time=2)
        self.wait_speech()

        ax = pi_axes(width=11, height=3.2, y_range=(-1.5, 1.5, 1)).shift(1.4 * DOWN)
        sq = square_graph(ax).set_stroke(opacity=0.5)
        a, b = square_coeffs(200)
        key = VGroup(Tex("S_m", font_size=34).set_color(SUM_COLOR), Tex(R"\sigma_m", font_size=34).set_color(ORANGE))
        key.arrange(RIGHT, buff=0.4).next_to(ax, UP, buff=0.1).align_to(ax, RIGHT)
        m = ValueTracker(5)
        s_live = always_redraw(lambda: curve(ax, lambda x: partial(x, a, b, round(m.get_value())), color=SUM_COLOR, width=2.5))
        f_live = always_redraw(lambda: curve(ax, lambda x: fejer(x, a, b, round(m.get_value())), color=ORANGE, width=4))
        self.caption("בממוצעים: אין בליטות, אין אפקט גיבס",
                     say="והנה זה על גל הריבוע. בכחול, הסכום החלקי הרגיל, עם הבליטות. בכתום, הממוצעים. "
                         "הם קצת יותר עגולים ליד הקפיצה, אבל אין בהם שום בליטה. אפקט גיבס פשוט נעלם.")
        self.play(sums.animate.scale(0.8).to_edge(UP, buff=0.3).shift(1.5 * LEFT), FadeIn(ax), ShowCreation(sq))
        self.add(s_live, f_live)
        self.play(FadeIn(key))
        for v in (15, 40, 100, 200):
            self.play(m.animate.set_value(v), run_time=2)
            self.wait(0.6)
        self.wait_speech()
        for mob in (s_live, f_live):
            mob.clear_updaters()
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None

        dots_ax = Axes((0, 12, 1), (-0.2, 1.2, 0.5), width=10, height=3.2,
                       axis_config=dict(stroke_color=AXIS, stroke_width=2)).shift(0.4 * UP)
        series = Tex(R"1 - 1 + 1 - 1 + 1 - \cdots = \frac{1}{2}", font_size=44).to_edge(UP, buff=0.5)
        partials = [1 if k % 2 == 0 else 0 for k in range(12)]
        avgs = np.cumsum(partials) / np.arange(1, 13)
        pdots = VGroup(*(Dot(dots_ax.c2p(k + 0.5, v), radius=0.08, fill_color=SUM_COLOR) for k, v in enumerate(partials)))
        adots = VGroup(*(Dot(dots_ax.c2p(k + 0.5, v), radius=0.08, fill_color=ORANGE) for k, v in enumerate(avgs)))
        half = DashedLine(dots_ax.c2p(0, 0.5), dots_ax.c2p(12, 0.5)).set_stroke(ORANGE, 2)
        self.caption("באותה שיטה: 1 פחות 1 ועוד 1, וכן הלאה, שווה חצי",
                     say="אגב, זו אותה סכימת צזארו שגדי השתמש בה בפוסט על אחת ועוד שתיים ועוד שלוש, "
                         "ושנתנה את התוצאה המוזרה: אחת פחות אחת ועוד אחת פחות אחת וכן הלאה, שווה חצי. "
                         "הסכומים החלקיים קופצים בין אחת לאפס, והממוצעים שלהם מתייצבים על חצי. "
                         "גדי מקווה שהשימוש היפה הזה בטורי פורייה יפחית קצת את הסקפטיות כלפיה.")
        self.play(Write(series), FadeIn(dots_ax))
        self.play(LaggedStart(*(FadeIn(d) for d in pdots), lag_ratio=0.15))
        self.play(LaggedStart(*(FadeIn(d) for d in adots), lag_ratio=0.15), ShowCreation(half))
        self.wait_speech()
        self.fade_all()


class Parseval(FourierScene):
    def construct(self):
        u = 0.45
        A, B, C = np.array([-4.4, -0.6, 0]), np.array([-4.4 + 4 * u, -0.6, 0]), np.array([-4.4, -0.6 + 3 * u, 0])
        tri_mob = Polygon(A, B, C).set_stroke(WHITE, 3)
        sq_a = Square(3 * u).next_to(Line(A, C), LEFT, buff=0).set_fill(BLUE, 0.5).set_stroke(BLUE, 2)
        sq_b = Square(4 * u).next_to(Line(A, B), DOWN, buff=0).set_fill(GREEN, 0.5).set_stroke(GREEN, 2)
        hyp = Line(B, C)
        sq_c = Square(5 * u).set_fill(YELLOW, 0.35).set_stroke(YELLOW, 2)
        sq_c.rotate(hyp.get_angle()).move_to(hyp.get_center() + 2.5 * u * normalize(rotate_vector(C - B, -PI / 2)))
        pyth = Tex(R"a^2 + b^2 = c^2", font_size=44).move_to(2.3 * DOWN + 0.6 * LEFT)
        self.caption("פיתגורס: סכום הריבועים של הרכיבים = ריבוע האורך",
                     say="ולסיום החלק הזה, התוצאה האהובה עליי, וגם על גדי. משפט פיתגורס אומר שבמשולש ישר זווית, "
                         "סכום הריבועים של הניצבים שווה לריבוע היתר. במילים של וקטורים: "
                         "האורך בריבוע של וקטור הוא סכום הריבועים של הקואורדינטות שלו.")
        self.play(ShowCreation(tri_mob))
        self.play(DrawBorderThenFill(sq_a), DrawBorderThenFill(sq_b))
        self.play(DrawBorderThenFill(sq_c), Write(pyth))
        self.wait_speech()

        lines = VGroup(
            Tex(R"\Vert v \Vert^2 = v_1^2 + v_2^2 + v_3^2", font_size=40),
            Tex(R"\Vert f \Vert^2 = \frac{a_0^2}{2} + \sum_{n=1}^{\infty} \left( a_n^2 + b_n^2 \right)", font_size=42),
        ).arrange(DOWN, buff=0.6, aligned_edge=LEFT).move_to(3.0 * RIGHT + 0.8 * UP)
        self.caption("שוויון פרסבל: פיתגורס באינסוף ממדים",
                     say="זה עובד בשלושה ממדים, ובכל מספר סופי של ממדים. ושוויון פרסבל אומר שזה עובד גם באינסוף ממדים: "
                         "האורך בריבוע של פונקציה הוא סכום הריבועים של מקדמי הפורייה שלה.")
        self.play(Write(lines[0]))
        self.play(Write(lines[1]), run_time=2)
        self.wait_speech()
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None

        norm = Tex(R"\Vert x \Vert^2 = \frac{1}{\pi}\int_{-\pi}^{\pi} x^2 \, dx = \frac{2\pi^2}{3}", font_size=40)
        norm.move_to(2.7 * UP + 2.6 * RIGHT)
        coef = Tex(R"b_n^2 = \frac{4}{n^2}", font_size=40).next_to(norm, DOWN, buff=0.4)
        self.caption("נפעיל את פרסבל על f(x) = x",
                     say="עכשיו נפעיל את זה על הדוגמה של גדי, אֶף של איקס שווה איקס. "
                         "מצד אחד, אפשר לחשב את האורך בריבוע ישירות: זה אינטגרל של איקס בריבוע, ויוצא שתיים פאי בריבוע חלקי שלוש. "
                         "מצד שני, המקדמים הם שתיים חלקי אן, ובריבוע, ארבע חלקי אן בריבוע.")
        self.play(Write(norm), run_time=2)
        self.play(Write(coef))
        scale = 0.7
        base = np.array([-4.2, -3.0, 0])
        total = 2 * PI ** 2 / 3
        target = DashedLine(base + 1.4 * LEFT + total * scale * UP, base + 1.4 * RIGHT + total * scale * UP).set_stroke(YELLOW, 2.5)
        target_lab = Tex(R"\frac{2\pi^2}{3}", font_size=36).set_color(YELLOW).next_to(target, LEFT, buff=0.15)
        self.play(ShowCreation(target), FadeIn(target_lab))
        stack = VGroup()
        h = 0
        for k in range(1, 41):
            v = 4 / k ** 2
            r = Rectangle(width=1.6, height=v * scale).set_stroke(BG, 1 if k < 12 else 0)
            r.set_fill(interpolate_color(BLUE, GREEN, min(k / 12, 1)), 0.9)
            r.move_to(base + (h + v / 2) * scale * UP)
            h += v
            stack.add(r)
        self.play(LaggedStart(*(FadeIn(r, shift=0.3 * DOWN) for r in stack[:6]), lag_ratio=0.5, run_time=4))
        self.play(LaggedStart(*(FadeIn(r) for r in stack[6:]), lag_ratio=0.1, run_time=2))
        self.wait_speech()

        eq = Tex(R"\sum_{n=1}^{\infty} \frac{4}{n^2} = \frac{2\pi^2}{3}", font_size=44).move_to(0.6 * DOWN + 2.6 * RIGHT)
        result = Tex(R"\sum_{n=1}^{\infty} \frac{1}{n^2} = 1 + \frac{1}{4} + \frac{1}{9} + \cdots = \frac{\pi^2}{6}", font_size=46)
        result.move_to(eq.get_center() + 0.2 * DOWN)
        self.caption("מחלקים ב-4: סכום ההופכיים של הריבועים",
                     say="שני החישובים חייבים להיות שווים. ואם מחלקים בארבע, מקבלים: אחת ועוד רבע ועוד תשיעית, "
                         "וכן הלאה, שווה פאי בריבוע חלקי שש.")
        self.play(Write(eq))
        self.wait(1)
        self.play(TransformMatchingTex(eq, result), run_time=2)
        box = SurroundingRectangle(result, buff=0.2).set_stroke(YELLOW, 2.5)
        self.play(ShowCreation(box))
        self.wait_speech()
        self.caption("ופתאום צץ π, והשד יודע מאיפה",
                     say="באגף שמאל יש אחד הטורים הפשוטים ביותר, ולא ברור בכלל איך לחשב את הסכום שלו. "
                         "ופתאום יש לנו אותו בדיוק, וצץ שם פאי, השד יודע מאיפה, למרות שלסכום הזה אין שום קשר למעגלים. "
                         "זה המספר שאוילר מצא, וזה בדיוק זטה של שתיים מהסרטון שלנו על זטה והראשוניים.")
        self.wait_speech()
        self.fade_all()


def glyph_points(char, samples=2048, height=5.0):
    """נקודות לאורך קו המתאר של אות, כמספרים מרוכבים סביב הראשית."""
    path = Text(char, font=FONT, font_size=200).family_members_with_points()[0]
    pts = np.array([path.point_from_proportion(t) for t in np.linspace(0, 1, samples, endpoint=False)])
    pts -= pts.mean(axis=0)
    pts *= height / (pts[:, 1].max() - pts[:, 1].min())
    return pts[:, 0] + 1j * pts[:, 1]


class ComplexDraw(FourierScene):
    def construct(self):
        euler = Tex(R"e^{ix} = \cos x + i \sin x", font_size=52).move_to(2.4 * UP)
        plane = ComplexPlane((-2, 2), (-1.6, 1.6), background_line_style=dict(stroke_color=GREY_D, stroke_width=1))
        plane.set_height(3.6).move_to(1.0 * DOWN)
        theta = ValueTracker(0)
        arrow = always_redraw(lambda: Arrow(plane.c2p(0, 0), plane.c2p(np.cos(theta.get_value()), np.sin(theta.get_value())),
                                            buff=0).set_color(YELLOW))
        circle = Circle(radius=plane.c2p(1, 0)[0] - plane.c2p(0, 0)[0]).move_to(plane.c2p(0, 0)).set_stroke(GREY_B, 1.5)
        self.caption("נוסחת אוילר: e בחזקת ix מסתובב על מעגל",
                     say="בפוסט הבא שלו, גדי עובר למספרים מרוכבים. נוסחת אוילר אומרת שאֵה בחזקת איי איקס "
                         "הוא קוסינוס ועוד איי סינוס: חץ שמסתובב על מעגל היחידה במישור המרוכב. "
                         "כלומר הסינוס והקוסינוס הם שני הצללים של אותו סיבוב.")
        self.play(Write(euler), FadeIn(plane), ShowCreation(circle))
        self.add(arrow)
        self.play(theta.animate.set_value(TAU), run_time=4, rate_func=linear)
        arrow.clear_updaters()
        self.wait_speech()
        formulas = VGroup(
            Tex(R"f(x) = \sum_{n=-\infty}^{\infty} c_n\, e^{inx}", font_size=44),
            Tex(R"c_n = \frac{1}{2\pi} \int_{-\pi}^{\pi} f(x)\, e^{-inx} \, dx", font_size=44),
        ).arrange(RIGHT, buff=1.0).move_to(2.4 * UP)
        self.caption("הטור המרוכב: חץ לכל מספר שלם n",
                     say="ואז הטור נהיה פשוט יותר: מקדם אחד, מרוכב, לכל אן שלם, כולל שליליים. "
                         "בלי הפרדה לסינוסים וקוסינוסים, ובלי החצי המוזר של אַ אפס. "
                         "ובפונקציות ממשיות, כמו שגדי כותב, המרוכבים חכמים ויודעים מתי לצאת בנימוס מהבמה.")
        self.play(FadeOut(euler), FadeOut(plane), FadeOut(circle), FadeOut(arrow))
        self.play(Write(formulas[0]), run_time=1.5)
        self.play(Write(formulas[1]), run_time=1.5)
        self.wait_speech()
        self.play(FadeOut(formulas))

        z = glyph_points("π")
        M = len(z)
        c = np.fft.fft(z) / M
        freqs = np.fft.fftfreq(M, 1 / M).astype(int)
        order = sorted(range(M), key=lambda i: (abs(freqs[i]), -freqs[i]))
        center = np.array([0, -0.2, 0])
        s = ValueTracker(0)
        K = ValueTracker(2)

        def terms():
            k = int(K.get_value())
            return [i for i in order if abs(freqs[i]) <= k]

        def tip_path(upto):
            idx = terms()
            ts = np.linspace(0, upto, max(int(upto * 600), 2))
            w = sum(c[i] * np.exp(2j * PI * freqs[i] * ts) for i in idx)
            return np.stack([w.real, w.imag, 0 * ts], axis=1) + center

        def chain():
            group = VGroup()
            idx = terms()
            p = center + np.array([c[idx[0]].real, c[idx[0]].imag, 0])
            for i in idx[1:]:
                w = c[i] * np.exp(2j * PI * freqs[i] * s.get_value())
                q = p + np.array([w.real, w.imag, 0])
                r = abs(c[i])
                if r > 0.02:
                    group.add(Circle(radius=r).move_to(p).set_stroke(GREY_B, 1, 0.35))
                group.add(Line(p, q).set_stroke(WHITE, 1.6, 0.85))
                p = q
            group.add(Dot(p, radius=0.05, fill_color=YELLOW))
            return group

        ghost = VMobject().set_points_as_corners(np.stack([z.real, z.imag, 0 * z.real], 1) + center)
        ghost.set_stroke(GREY_B, 1.5, 0.35)
        arrows = always_redraw(chain)
        trace = always_redraw(lambda: VMobject().set_points_as_corners(tip_path(max(s.get_value(), 0.002)))
                              .set_stroke(YELLOW, 3.5))
        k_lab = heb("מספר החצים:", 30, GREY_A)
        k_num = Integer(5, font_size=34).set_color(YELLOW)
        counter = VGroup(k_lab, k_num).arrange(LEFT, buff=0.2).to_corner(UR, buff=0.4)
        self.caption("ציור עם חצים מסתובבים",
                     say="ועכשיו משהו יפה. כל קו סגור במישור הוא פונקציה מחזורית מרוכבת: לכל רגע, נקודה במישור. "
                         "אז אפשר לפרק אותו לטור פורייה. כל איבר בטור הוא חץ שמסתובב במהירות שלמה, "
                         "והחצים מחוברים זה לזה, קצה לזנב. ננסה לצייר את האות פאי.")
        self.play(ShowCreation(ghost), FadeIn(counter))
        self.add(arrows, trace)
        for kk, rt in ((2, 6), (8, 7), (40, 8)):
            K.set_value(kk)
            k_num.set_value(2 * kk + 1).next_to(k_lab, LEFT, buff=0.2)
            s.set_value(0)
            self.play(s.animate.set_value(1), run_time=rt, rate_func=linear)
            self.wait(0.6)
            if kk == 2:
                self.say("עם חמישה חצים, יוצא כתם עגלגל. שבעה עשר חצים, וכבר רואים צורה.")
            if kk == 8:
                self.say("ועם שמונים ואחד חצים, החצים המסתובבים מציירים את פאי כמעט בדיוק.")
        self.wait_speech()
        for m in (arrows, trace):
            m.clear_updaters()
        self.fade_all()


class History(FourierScene):
    def construct(self):
        line = Line([-6.4, 0.6, 0], [6.4, 0.6, 0]).set_stroke(GREY_B, 3)
        events = [
            (-5.4, "1807", "פורייה: גלים ומעבר חום", UP),
            (-2.7, "1829", "דיריכלה: הוכחת התכנסות", DOWN),
            (0.0, "1848", "ווילברהאם: הבליטה", UP),
            (2.7, "1899", "גיבס: שוב הבליטה", DOWN),
            (5.4, "1900", "פייר: ממוצעים", UP),
        ]
        texts = [
            "פורייה הגיע לרעיון כשחקר איך חום מתפשט במוט מתכת. ב-1807 הוא הציג אותו לאקדמיה בפריז, "
            "ולגראנז', מגדולי המתמטיקאים של התקופה, לא האמין שסכום של סינוסים חלקים יכול לתת פונקציה עם קפיצה.",
            "רק עשרים שנה אחר כך, פטר גוסטב לז'ן דיריכלה הוכיח לראשונה, בצורה מדויקת, מתי הטור באמת מתכנס.",
            "ווילברהאם ראה את הבליטה ליד הקפיצה, ואף אחד לא שם לב.",
            "וגיבס תיאר אותה שוב, חמישים שנה אחר כך.",
            "ופייר הראה שהממוצעים מתכנסים לכל פונקציה רציפה.",
        ]
        self.play(ShowCreation(line))
        for (x, year, what, side), text in zip(events, texts):
            self.say(text)
            p = np.array([x, 0.6, 0])
            tick = Dot(p, radius=0.1, fill_color=YELLOW)
            label = VGroup(heb(year, 30, YELLOW), heb(what, 26)).arrange(DOWN, buff=0.12).next_to(p, side, buff=0.35)
            self.play(GrowFromCenter(tick), FadeIn(label, shift=0.2 * side))
            self.wait_speech()
        self.play(*map(FadeOut, self.mobjects))
        uses = VGroup(*(heb(t, 34) for t in [
            "צלילים: כל צליל הוא סכום של תדרים",
            "דחיסת קבצים: MP3 ו-JPEG זורקים את הגלים שלא שומעים ולא רואים",
            "רפואה: מכשיר MRI בונה תמונה מגלים",
            "מחשבים קוונטיים: התמרת פורייה הקוונטית",
        ])).arrange(DOWN, aligned_edge=RIGHT, buff=0.45).move_to(0.5 * UP)
        self.say("והיום פורייה נמצא בכל מקום: בניתוח של צלילים, בדחיסה של מוזיקה ותמונות, "
                 "במכשירי אם אר איי בבתי חולים, ואפילו במחשבים קוונטיים, שגם עליהם גדי כתב. "
                 "וכל זה מתחיל ברעיון אחד: לפרק פונקציה לגלים.")
        self.play(LaggedStart(*(FadeIn(u, shift=0.2 * LEFT) for u in uses), lag_ratio=0.5, run_time=5))
        self.wait_speech()
        self.fade_all()


class Credits(FourierScene):
    def construct(self):
        thanks = heb("תודה", 60, SUM_COLOR)
        line1 = heb("הסרטון מבוסס על הפוסטים של", 34)
        line2 = heb("גדי אלכסנדרוביץ' בבלוג \"לא מדויק\"", 44, YELLOW)
        posts = VGroup(*(heb(t, 24, GREY_A) for t in [
            "2014: טורי פורייה - מה זה בכלל?",
            "2014: בסיסים אורתונורמליים במרחבי הילברט",
            "2014: התמרת פורייה - מה זה בכלל?",
            "2010: יום פאי שמח!",
            "2008: כמה מעלות טובות לרדיאן עלינו",
        ])).arrange(DOWN, buff=0.16)
        site = Text("gadial.net", font=FONT, font_size=34, fill_color=GREY_B)
        VGroup(thanks, line1, line2, posts, site).arrange(DOWN, buff=0.35)
        self.say("הסרטון הזה מבוסס על הפוסטים של גדי אלכסנדרוביץ' על טורי פורייה, בבלוג שלו, לא מדויק. "
                 "בעיקר הפוסט טורי פורייה, מה זה בכלל, ואיתו הפוסטים על בסיסים אורתונורמליים, על התמרת פורייה, "
                 "על יום פאי, ועל רדיאנים. תודה, גדי.")
        self.play(Write(thanks))
        self.play(LaggedStartMap(FadeIn, VGroup(line1, line2, posts, site), shift=0.2 * UP, lag_ratio=0.4, run_time=3))
        self.wait(3)
        self.fade_all(run_time=1.5)


SCENES = [Title, Decompose, SineCosine, Harmonics, TheQuestion, Vectors, Orthogonal, Detector, BestApprox,
          SquareCoeffs, BuildSquare, Epicycles, NormConvergence, Gibbs, Symmetry, Sawtooth, Smoothness, Fejer,
          Parseval, ComplexDraw, History, Credits]
