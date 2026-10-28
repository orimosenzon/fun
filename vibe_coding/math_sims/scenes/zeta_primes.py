"""פונקציית זטה של רימן והמספרים הראשוניים: סרטון ב-ManimGL.

הנתונים המספריים מחושבים מראש ב-compute/zeta_data.py.
רינדור כל הסצנות וחיבורן לסרטון אחד: ./render.sh
"""
import re
from pathlib import Path

import numpy as np

from manimlib import *

DATA = np.load(Path(__file__).resolve().parent.parent / "data" / "zeta_data.npz")
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


class HebrewScene(Scene):
    default_camera_config = dict(background_color=BG)

    def setup(self):
        super().setup()
        self.cap = None

    def caption(self, text, size=34, wait=1.5):
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


class Title(HebrewScene):
    def construct(self):
        title = heb("פונקציית זטה של רימן", 72)
        sub = heb("והסוד שמסתתר במספרים הראשוניים", 40, GREY_A).next_to(title, DOWN, buff=0.5)
        zeta = Tex(R"\zeta(s)", font_size=120).set_color(CURVE_COLOR).next_to(title, UP, buff=0.7)
        self.play(Write(zeta), run_time=1.5)
        self.play(FadeIn(title, shift=0.3 * UP))
        self.play(FadeIn(sub))
        self.wait(2)
        self.play(*map(FadeOut, [title, sub, zeta]))


class PrimeGrid(HebrewScene):
    def construct(self):
        cells = VGroup()
        for n in range(1, 101):
            square = Square(0.6).set_stroke(GREY_D, 1)
            num = Integer(n, font_size=24).move_to(square)
            cells.add(VGroup(square, num))
        cells.arrange_in_grid(10, 10, buff=0).to_edge(UP, buff=0.4)

        self.play(LaggedStartMap(FadeIn, cells, lag_ratio=0.01, run_time=2))
        self.caption("מספר ראשוני מתחלק רק ב-1 ובעצמו")

        primes = [cell for n, cell in enumerate(cells, 1) if is_prime(n)]
        others = [cell for n, cell in enumerate(cells, 1) if not is_prime(n)]
        self.play(
            LaggedStart(*(
                cell[0].animate.set_fill(PRIME_COLOR, 0.35).set_stroke(PRIME_COLOR, 2)
                for cell in primes
            ), lag_ratio=0.08, run_time=4),
            *(cell[1].animate.set_opacity(0.35) for cell in others),
        )
        self.caption("הם נראים מפוזרים בלי שום חוק", wait=1)
        self.caption("25 ראשוניים עד 100, ואחריהם מרווחים שגדלים ומתכווצים", wait=2)
        self.caption("השאלה של רימן: כמה ראשוניים יש עד x?", wait=2)
        self.play(FadeOut(cells), FadeOut(self.cap))


class PrimeCounting(HebrewScene):
    def construct(self):
        xs, pi = DATA["x_grid"], DATA["pi_exact"]
        axes = Axes((0, 100, 10), (0, 28, 5), width=11, height=5.2).to_edge(UP, buff=0.6)
        axes.add_coordinate_labels(font_size=20)
        label = Tex(R"\pi(x)", font_size=40).set_color(PRIME_COLOR)
        label.next_to(axes.c2p(0, 28), RIGHT)
        self.play(ShowCreation(axes), Write(label))

        stairs = staircase(axes, xs, pi, stroke_color=PRIME_COLOR, stroke_width=3)
        self.caption("פונקציית הספירה π(x): קופצת ב-1 בכל ראשוני")
        self.play(ShowCreation(stairs, rate_func=linear), run_time=5)
        self.wait()

        gauss = axes.get_graph(lambda x: x / np.log(x), x_range=(2.2, 100), color=BLUE)
        gauss_label = Tex(R"\frac{x}{\ln x}", font_size=34).set_color(BLUE)
        gauss_label.next_to(axes.c2p(100, 100 / np.log(100)), RIGHT, buff=0.1)
        self.caption("גאוס, בגיל 15: הצפיפות שלהם בערך 1 חלקי ln x", wait=0.5)
        self.play(ShowCreation(gauss), Write(gauss_label), run_time=2)
        self.wait()

        smooth = polyline(axes.c2p(xs, DATA["R_main"]), stroke_color=GREEN, stroke_width=3)
        smooth_label = Tex(R"R(x)", font_size=34).set_color(GREEN)
        smooth_label.next_to(smooth.get_end(), RIGHT, buff=0.1).shift(0.3 * UP)
        self.caption("הקירוב של רימן, R(x), תופס את המגמה מצוין...", wait=0.5)
        self.play(ShowCreation(smooth), Write(smooth_label), run_time=2)
        self.wait()
        self.caption("...אבל לא את המדרגות. איפה מסתתר המידע עליהן?", wait=2.5)
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None


class ZetaDefinition(HebrewScene):
    def construct(self):
        definition = Tex(R"\zeta(s) = 1 + \frac{1}{2^s} + \frac{1}{3^s} + \frac{1}{4^s} + \cdots",
                         font_size=60)
        definition.to_edge(UP, buff=1)
        self.play(Write(definition), run_time=2)
        self.caption("הכירו את פונקציית זטה: סכום של אחד חלקי n בחזקת s")

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
        self.play(n_tracker.animate.set_value(2000), run_time=5, rate_func=lambda t: t ** 3)

        target = Tex(R"\frac{\pi^2}{6} = 1.64493\ldots", font_size=48).set_color(CURVE_COLOR)
        target.next_to(row, DOWN, buff=0.5)
        self.caption("אוילר, 1734: הסכום מתכנס בדיוק ל-π²/6", wait=0.5)
        self.play(Write(target))
        self.wait(2)
        self.caption("אבל מה הקשר למספרים הראשוניים?", wait=1.5)
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None


class EulerProduct(HebrewScene):
    def construct(self):
        eq = Tex(
            R"\sum_{n=1}^{\infty} \frac{1}{n^s} = \prod_{p \text{ prime}} \frac{1}{1 - p^{-s}}",
            font_size=60,
        )
        eq[R"\prod_{p \text{ prime}} \frac{1}{1 - p^{-s}}"].set_color(PRIME_COLOR)
        eq.to_edge(UP, buff=0.8)
        self.play(Write(eq), run_time=2)
        self.caption("המכפלה של אוילר: הסכום על כל המספרים = מכפלה על הראשוניים בלבד", size=30)

        expansion = Tex(
            R"\left(1 + \tfrac{1}{2^s} + \tfrac{1}{4^s} + \tfrac{1}{8^s} + \cdots\right)"
            R"\left(1 + \tfrac{1}{3^s} + \tfrac{1}{9^s} + \cdots\right)"
            R"\left(1 + \tfrac{1}{5^s} + \cdots\right)\cdots",
            font_size=42,
        )
        expansion.next_to(eq, DOWN, buff=0.9)
        self.play(FadeIn(expansion, shift=DOWN))
        self.caption("כל גורם הוא טור הנדסי של חזקות של ראשוני אחד", size=30)

        example = Tex(R"\frac{1}{4^s} \cdot \frac{1}{3^s} = \frac{1}{12^s}", font_size=48)
        example.set_color(CURVE_COLOR).next_to(expansion, DOWN, buff=0.8)
        self.play(
            FlashAround(expansion[R"\tfrac{1}{4^s}"][0]),
            FlashAround(expansion[R"\tfrac{1}{3^s}"][0]),
            run_time=1.5,
        )
        self.play(Write(example))
        self.caption("כל מספר מתפרק באופן יחיד לראשוניים, 12 = 2·2·3", size=30)
        self.caption("ולכן כשפותחים סוגריים, כל 1/nˢ מופיע בדיוק פעם אחת", size=30, wait=2)
        self.caption("זטה יודעת הכול על הראשוניים. רימן הבין איך לחלץ את זה", size=30, wait=2)
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None


class ComplexZeros(HebrewScene):
    def construct(self):
        title = heb("רימן, 1859: נציב ב-s מספרים מרוכבים", 40).to_edge(UP, buff=0.3)
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

        self.caption("ב-s=1 זטה מתפוצצת לאינסוף (הטור ההרמוני)", wait=0.5)
        self.play(Write(pole))
        self.wait()

        trivial = VGroup(*(Dot(axes.c2p(x, 0), radius=0.07, fill_color=GREY_A) for x in (-2, -4, -6, -8)))
        self.caption("יש לה אפסים 'טריוויאליים' בכל המספרים הזוגיים השליליים", wait=0.5)
        self.play(LaggedStartMap(GrowFromCenter, trivial, lag_ratio=0.2))
        self.wait()

        self.caption("וכל השאר חיים ברצועה הקריטית, בין 0 ל-1", wait=0.5)
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
        self.caption("והפלא: כל האפסים שנמצאו יושבים בדיוק על הקו Re = ½", wait=0.5)
        self.play(ShowCreation(crit), run_time=1.5)
        self.wait(2.5)
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None


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
        self.caption("נטייל לאורך הקו הקריטי ונראה לאן זטה שולחת כל נקודה", size=30, wait=0.5)

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
                self.caption("הנה! זטה פוגעת בדיוק ב-0: האפס הראשון, t ≈ 14.13", size=30, wait=0.5)
                self.caption("כל מעבר דרך הראשית הוא עוד אפס", size=30, wait=0.3)

        self.caption("עשרה אפסים עד 50. ידועים יותר מ-10 טריליון, וכולם על הקו", size=30, wait=2.5)
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None


class ExplicitFormula(HebrewScene):
    def construct(self):
        xs, pi = DATA["x_grid"], DATA["pi_exact"]
        formula = Tex(R"\pi(x) \approx R(x) - \sum_{\rho} R(x^{\rho})", font_size=46)
        formula[R"\sum_{\rho} R(x^{\rho})"].set_color(ZERO_COLOR)
        formula.to_edge(UP, buff=0.3)
        self.play(Write(formula))
        self.caption("הנוסחה המפורשת של רימן: כל אפס ρ מוסיף גל לקירוב", size=30, wait=0.5)

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
        self.caption("זה הגל שתורם האפס הראשון:", size=30, wait=0.3)
        plot.save_state()
        # רק עמעום קווים: set_opacity על polyline היה צובע גם את המילוי מתחת לעקומה
        self.play(VGroup(axes, stairs, approx).animate.set_stroke(opacity=0.15),
                  counter.animate.set_opacity(0.15))
        self.play(ShowCreation(first_wave), Write(wave_label), run_time=2)
        self.wait(1.5)
        self.play(FadeOut(first_wave), FadeOut(wave_label), Restore(plot))

        for k in (1, 3, 10, 30, 100):
            self.play(Transform(approx, curve(DATA[f"approx_{k}"])), k_tracker.animate.set_value(k),
                      run_time=1.6)
            if k == 3:
                self.caption("עוד אפסים, עוד גלים, והמדרגות מתחילות להופיע", size=30, wait=0.3)
            else:
                self.wait(0.6)

        self.caption("עם 100 אפסים, המוזיקה של האפסים משחזרת את הראשוניים", size=30, wait=2.5)
        self.play(*map(FadeOut, self.mobjects))
        self.cap = None


class Hypothesis(HebrewScene):
    def construct(self):
        title = heb("השערת רימן", 64).set_color(CURVE_COLOR).to_edge(UP, buff=0.8)
        statement = heb("כל האפסים הלא טריוויאליים של זטה יושבים על הקו", 38)
        line = Tex(R"\mathrm{Re}(s) = \tfrac{1}{2}", font_size=56)
        VGroup(statement, line).arrange(DOWN, buff=0.4).next_to(title, DOWN, buff=0.8)
        self.play(Write(title))
        self.play(FadeIn(statement), Write(line))
        self.wait(1.5)

        meaning = heb("אם היא נכונה, הראשוניים סוטים מהממוצע הכי מעט שאפשר:", 32, GREY_A)
        bound = Tex(R"\left|\pi(x) - \mathrm{Li}(x)\right| \le \tfrac{1}{8\pi}\sqrt{x}\,\ln x", font_size=48)
        bound.set_color(PRIME_COLOR)
        VGroup(meaning, bound).arrange(DOWN, buff=0.35).next_to(line, DOWN, buff=0.8)
        self.play(FadeIn(meaning))
        self.play(Write(bound))
        self.wait(2)

        footer = heb("פתוחה מאז 1859. פרס המילניום: מיליון דולר", 34, GREY_B).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(footer, shift=UP))
        self.wait(3)
        self.play(*map(FadeOut, self.mobjects), run_time=1.5)


SCENES = [Title, PrimeGrid, PrimeCounting, ZetaDefinition, EulerProduct,
          ComplexZeros, CriticalLineSpiral, ExplicitFormula, Hypothesis]
