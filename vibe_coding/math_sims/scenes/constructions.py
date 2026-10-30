"""בניות בסרגל ובמחוגה: למה אי אפשר לחלק זווית לשלוש, להכפיל קובייה או לרבע את העיגול.

בהשראת סדרת הפוסטים של גדי אלכסנדרוביץ' בבלוג "לא מדויק" (gadial.net):
בניות בסרגל ומחוגה (2008), שני פוסטים על הרחבת שדות (2009),
אז למה אי אפשר לרבע את העיגול? (2009), ושלוש בעיות הבניה בסרגל ומחוגה של היוונים (2018).
רינדור: ./render.sh --hd scenes/constructions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: E402,F403

LINE = GREY_A
CIRCLE = BLUE_B
POINT = YELLOW
NUM = TEAL
GOOD = GREEN
BAD = RED
WOOD = "#d9c38f"


# ---------- גאומטריה ----------

def unit(v):
    return v / np.linalg.norm(v)


def perp(v):
    return rotate_vector(v, PI / 2)


def direction(angle):
    return np.array([np.cos(angle), np.sin(angle), 0])


def circle_circle(c1, r1, c2, r2):
    d = np.linalg.norm(c2 - c1)
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(max(r1 ** 2 - a ** 2, 0))
    u = unit(c2 - c1)
    m = c1 + a * u
    return m + h * perp(u), m - h * perp(u)


def line_circle(p, q, c, r):
    """שתי נקודות החיתוך של הישר pq עם המעגל, לפי הסדר מ-p לכיוון q."""
    u = unit(q - p)
    f = p - c
    b = np.dot(f, u)
    disc = np.sqrt(max(b * b - (np.dot(f, f) - r * r), 0))
    return p + (-b - disc) * u, p + (-b + disc) * u


def dot(p, color=POINT, radius=0.07):
    return Dot(p, radius=radius, fill_color=color)


def right_angle(vertex, d1, d2, size=0.22, color=GREY_B):
    d1, d2 = unit(d1), unit(d2)
    return VMobject().set_points_as_corners(
        [vertex + size * d1, vertex + size * (d1 + d2), vertex + size * d2]
    ).set_stroke(color, 2)


def angle_mark(vertex, p1, p2, radius=0.5, color=YELLOW, tex=None, size=30, label_buff=0.3):
    """קשת של הזווית הקטנה מבין שתי הזוויות שבין הכיוונים vertex→p1 ו-vertex→p2."""
    a1, a2 = angle_of_vector(p1 - vertex), angle_of_vector(p2 - vertex)
    sweep = (a2 - a1) % TAU
    if sweep > PI:
        a1, sweep = a2, TAU - sweep
    arc = Arc(start_angle=a1, angle=sweep, radius=radius, arc_center=vertex).set_stroke(color, 3)
    if tex is None:
        return arc
    label = Tex(tex, font_size=size).set_color(color)
    label.move_to(vertex + (radius + label_buff) * direction(a1 + sweep / 2))
    return VGroup(arc, label)


def compass(center, pen):
    """מחוגה פתוחה: רגל אחת נעוצה ב-center, העיפרון ב-pen."""
    d = pen - center
    r = np.linalg.norm(d)
    leg = max(0.62 * r, 0.9)
    u = unit(d) if r > 0 else RIGHT
    apex = center + d / 2 + np.sqrt(leg ** 2 - (r / 2) ** 2) * perp(u)
    return VGroup(
        Line(center, apex).set_stroke(GREY_A, 5),
        Line(apex, pen).set_stroke(GREY_A, 5),
        Line(apex, apex + 0.35 * perp(u)).set_stroke(GREY_B, 6),
        Circle(radius=0.08).move_to(apex).set_stroke(GREY_A, 3).set_fill(BG, 1),
        Dot(center, radius=0.04, fill_color=GREY_B),
        Dot(pen, radius=0.05, fill_color=POINT),
    )


def straightedge(start, end, width=0.3):
    """סרגל בלי שנתות, מונח כך שהשפה שלו על הקטע start→end."""
    u = unit(end - start)
    body = Rectangle(width=np.linalg.norm(end - start), height=width)
    body.set_fill(WOOD, 0.3).set_stroke(WOOD, 1.5)
    body.rotate(angle_of_vector(u)).move_to((start + end) / 2 - width / 2 * perp(u))
    return body


def iso_cube(corner, s, color=BLUE, depth=(0.45, 0.32)):
    """קובייה בהיטל אלכסוני: corner היא הפינה השמאלית התחתונה של הפאה הקדמית."""
    k = s * np.array([*depth, 0])
    x, y = s * RIGHT, s * UP
    faces = VGroup(
        Polygon(corner, corner + x, corner + x + y, corner + y).set_fill(color, 0.55),
        Polygon(corner + y, corner + x + y, corner + x + y + k, corner + y + k).set_fill(color, 0.8),
        Polygon(corner + x, corner + x + k, corner + x + y + k, corner + x + y).set_fill(color, 0.35),
    )
    return faces.set_stroke(WHITE, 1.5)


class ConstructionScene(HebrewScene):
    def ruler_line(self, p, q, extend=1.0, extend_start=None, color=LINE, run_time=1.0, tool=True):
        """ישר דרך p ו-q, עם הסרגל שמונח לאורכו."""
        u = unit(q - p)
        start = p - (extend if extend_start is None else extend_start) * u
        line = Line(start, q + extend * u).set_stroke(color, 2.5)
        if tool:
            edge = straightedge(line.get_start() - 0.3 * u, line.get_end() + 0.3 * u)
            self.play(FadeIn(edge, shift=0.15 * perp(u)), run_time=0.4)
            self.play(ShowCreation(line), run_time=run_time)
            self.play(FadeOut(edge), run_time=0.3)
        else:
            self.play(ShowCreation(line), run_time=run_time)
        return line

    def compass_arc(self, center, radius, start_angle=0, sweep=TAU, color=CIRCLE, run_time=1.4, tool=True):
        """קשת (או מעגל שלם) שהמחוגה משרטטת סביב center."""
        arc = Arc(start_angle=start_angle, angle=sweep, radius=radius, arc_center=center,
                  n_components=max(8, int(abs(sweep) / TAU * 24)))
        arc.set_stroke(color, 2.5)
        if not tool:
            self.play(ShowCreation(arc), run_time=run_time)
            return arc
        angle = ValueTracker(start_angle)
        tool_mob = always_redraw(lambda: compass(center, center + radius * direction(angle.get_value())))
        self.play(FadeIn(tool_mob), run_time=0.3)
        self.play(ShowCreation(arc), angle.animate.set_value(start_angle + sweep), run_time=run_time)
        tool_mob.clear_updaters()
        self.play(FadeOut(tool_mob), run_time=0.3)
        return arc

    def compass_circle(self, center, through, **kwargs):
        return self.compass_arc(center, np.linalg.norm(through - center),
                                start_angle=angle_of_vector(through - center), **kwargs)

    def transfer(self, a, b, center, end_angle=None, color=CIRCLE, sweep=None, **kwargs):
        """מחוגה שנפתחת על הקטע ab, מורמת ונעוצה ב-center, ומשרטטת שם קשת (או מעגל)."""
        tool_mob = compass(a, b)
        self.play(FadeIn(tool_mob), run_time=0.4)
        self.play(tool_mob.animate.shift(center - a), run_time=0.9)
        self.remove(tool_mob)
        start = angle_of_vector(b - a)
        if sweep is None:
            sweep = TAU if end_angle is None else end_angle - start
        return self.compass_arc(center, np.linalg.norm(b - a), start_angle=start, sweep=sweep,
                                color=color, **kwargs)

    def parallel_through(self, x, y, p, color, says=(None, None, None)):
        """ישר דרך p שמקביל ל-xy: משלימים את xyqp למקבילית בשתי קשתות, ומעבירים ישר דרך p ו-q.
        says: הקראה לקשת הראשונה, לשנייה ולישר."""
        q = p + (y - x)
        arcs = VGroup()
        for text, (a, b, c) in zip(says, ((x, y, p), (x, p, y))):
            if text:
                self.say(text)
            tool_mob = compass(a, b)
            self.play(FadeIn(tool_mob), run_time=0.4)
            self.play(tool_mob.animate.shift(c - a), run_time=0.9)
            ang = angle_of_vector(q - c)
            delta = (ang - 0.3 - angle_of_vector(b - a) + PI) % TAU - PI
            self.play(Rotate(tool_mob, delta, about_point=c), run_time=0.4)
            self.remove(tool_mob)
            arcs.add(self.compass_arc(c, np.linalg.norm(b - a), start_angle=ang - 0.3, sweep=0.6, run_time=1))
        if says[2]:
            self.say(says[2])
        q_dot = dot(q, ORANGE)
        self.play(Flash(q, color=color, flash_radius=0.3), FadeIn(q_dot))
        quad = Polygon(x, y, q, p).set_stroke(color, 1.5, opacity=0.6).set_fill(color, 0.12)
        self.play(FadeIn(quad))
        self.wait(0.6)
        return q, VGroup(arcs, q_dot, quad)


# ---------- סצנות ----------

class Title(ConstructionScene):
    def construct(self):
        center, r = np.array([-1.9, 1.7, 0]), 1.0
        self.say("סרגל ומחוגה. שלוש בעיות שהיוונים הציבו לפני יותר מאלפיים שנה, ואף אחד לא הצליח לפתור.")
        circle = self.compass_circle(center, center + r * RIGHT, color=NUM)
        disk = Circle(radius=r).move_to(center).set_fill(NUM, 0.25).set_stroke(width=0)
        square = Square(r * np.sqrt(PI)).move_to([1.9, 1.7, 0]).set_stroke(GOLD, 3).set_fill(GOLD, 0.25)
        q = Text("?", font=FONT, font_size=80).move_to([0, 1.7, 0])
        self.play(FadeIn(disk), DrawBorderThenFill(square))
        self.play(Write(q))
        title = heb("סרגל ומחוגה", 84).move_to(0.6 * DOWN)
        sub = heb("למה אי אפשר לחלק זווית לשלוש או לרבע את העיגול", 36, GREY_A)
        sub.next_to(title, DOWN, buff=0.45)
        self.play(FadeIn(title, shift=0.3 * UP))
        self.play(FadeIn(sub))
        self.say("ואחרי כל השנים האלה התברר שאין להן פתרון בכלל. והסיבה, באופן מפתיע, היא אלגברה.")
        self.wait(1)
        self.fade_all()


class Tools(ConstructionScene):
    def construct(self):
        body = Rectangle(width=9, height=0.8).set_fill(WOOD, 0.9).set_stroke("#a8925e", 2).move_to(1.6 * UP)
        top = body.get_top()[1]
        ticks, nums = VGroup(), VGroup()
        for i in range(41):
            x = -4.3 + i * 0.215
            ticks.add(Line([x, top, 0], [x, top - (0.3 if i % 5 == 0 else 0.15), 0]).set_stroke("#4a3b20", 2))
            if i % 10 == 0:
                nums.add(Text(str(i // 10), font=FONT, font_size=20, fill_color="#4a3b20").move_to([x, top - 0.48, 0]))
        name = Text("ruler", font=FONT, font_size=28, fill_color=GREY_B).next_to(body, DOWN, buff=0.3)
        self.play(FadeIn(body), ShowCreation(ticks), FadeIn(nums), FadeIn(name))
        self.caption("לסרגל שלנו יש שנתות: בעזרתן מודדים אורכים",
                     say="לסרגל שאנחנו מכירים יש שנתות, ובעזרתן מודדים אורכים. "
                         "אבל איך מי שבנה את הסרגל ידע איפה לסמן אותן? היוונים העדיפו לא להניח דבר כזה.")
        self.wait_speech()
        name2 = Text("straightedge", font=FONT, font_size=28, fill_color=GREY_B).move_to(name)
        self.play(FadeOut(ticks), FadeOut(nums), FadeTransform(name, name2))
        self.caption("הסרגל של היוונים: בלי שנתות. רק מעבירים קו ישר",
                     say="הסרגל של היוונים הוא בלי שנתות. אפשר רק להעביר בעזרתו קו ישר דרך שתי נקודות. "
                         "באנגלית יש לו שם משלו, סְטְרֵייט אֶדְג', ובעברית אין הבחנה.")
        self.wait_speech()
        self.play(FadeOut(body), FadeOut(name2))

        p, q = np.array([-6, -1.6, 0]), np.array([-2.2, 0.9, 0])
        self.play(FadeIn(dot(p)), FadeIn(dot(q)))
        self.ruler_line(p, q, extend=0.6, run_time=1.5)
        self.caption("והמחוגה: נועצים רגל אחת, ומסובבים את השנייה",
                     say="הכלי השני הוא המחוגה. נועצים רגל אחת במרכז, מסובבים את השנייה סביבו, ומקבלים מעגל.")
        center = np.array([3.2, -0.3, 0])
        self.play(FadeIn(dot(center, GREY_B)))
        self.compass_circle(center, center + 1.7 * direction(0.3), run_time=2.5)
        self.fade_all()


class Rules(ConstructionScene):
    def construct(self):
        rules = VGroup(
            heb("1. ישר דרך שתי נקודות קיימות", 28),
            heb("2. מעגל סביב נקודה קיימת, ברדיוס שהוא מרחק בין נקודות קיימות", 28),
            heb("3. נקודות חדשות נולדות רק בחיתוך של ישרים ומעגלים", 28),
        ).arrange(DOWN, aligned_edge=RIGHT, buff=0.22).to_corner(UR, buff=0.35)
        rules.set_fill(GREY_C)

        def highlight(i):
            return [rules[j].animate.set_fill(YELLOW if j == i else GREY_C) for j in range(3)]

        u = 1.8
        a, b = np.array([-u / 2, -0.7, 0]), np.array([u / 2, -0.7, 0])
        labels = VGroup(Tex("0", font_size=34).next_to(a, DL, buff=0.1), Tex("1", font_size=34).next_to(b, DR, buff=0.1))
        self.caption("מתחילים משתי נקודות. המרחק ביניהן הוא 1",
                     say="אלה כללי המשחק. מתחילים משתי נקודות על דף ריק, "
                         "והמרחק ביניהן ייחשב לאחת, יחידת המידה.")
        self.play(FadeIn(dot(a)), FadeIn(dot(b)), FadeIn(labels))
        self.play(FadeIn(rules[0]), *highlight(0))
        self.say("מותר להעביר ישר דרך שתי נקודות שכבר יש לנו.")
        line = self.ruler_line(a, b, extend=4.5)
        self.wait_speech()

        self.play(FadeIn(rules[1]), *highlight(1))
        self.say("מותר לשרטט מעגל שמרכזו בנקודה קיימת. את הרדיוס לוקחים מהמרחק בין שתי נקודות קיימות: "
                 "פותחים את המחוגה עליהן, ונועצים אותה איפה שרוצים.")
        self.compass_circle(a, b)
        self.compass_circle(b, a)
        self.wait_speech()

        self.play(FadeIn(rules[2]), *highlight(2))
        top, bottom = circle_circle(a, u, b, u)
        news = VGroup(*(dot(p, ORANGE) for p in (top, bottom, a - u * RIGHT, b + u * RIGHT)))
        self.say("ונקודות חדשות נולדות רק במקומות שבהם ישרים ומעגלים נחתכים. זהו, אלה כל הכללים.")
        self.play(LaggedStart(*(GrowFromCenter(d) for d in news), lag_ratio=0.3))
        self.play(*(Flash(d, color=ORANGE, flash_radius=0.25) for d in news))
        self.wait_speech()

        tri = Polygon(a, b, top).set_stroke(GOLD, 4).set_fill(GOLD, 0.25)
        self.caption("המשפט הראשון של אוקלידס: משולש שווה צלעות",
                     say="וכבר מהמהלכים האלה יוצא משהו: המשפט הראשון בספר יסודות של אוקלידס. "
                         "שני המעגלים נחתכים בנקודה שמרחקה משתי הנקודות הוא אחת, והנה משולש שווה צלעות.")
        self.play(DrawBorderThenFill(tri), run_time=1.5)
        self.wait(1)
        self.fade_all()


class Bisect(ConstructionScene):
    def construct(self):
        o = np.array([-2.6, -2.5, 0])
        alpha = 72 * DEGREES
        arm1, arm2 = o + 6 * RIGHT, o + 5.9 * direction(alpha)
        arms = VGroup(Line(o, arm1), Line(o, arm2)).set_stroke(WHITE, 3)
        mark = angle_mark(o, arm1, arm2, 0.6, YELLOW, R"\alpha")
        self.play(ShowCreation(arms), FadeIn(dot(o, WHITE)))
        self.play(FadeIn(mark))
        self.caption("לחצות זווית: מעגל סביב הקודקוד...",
                     say="הנה מה שהכלים האלה יודעים לעשות. נחצה זווית לשניים. "
                         "קודם, קשת סביב הקודקוד, שחותכת את שתי הזרועות.")
        r = 2.3
        self.compass_arc(o, r, -8 * DEGREES, alpha + 16 * DEGREES)
        p, q = o + r * RIGHT, o + r * direction(alpha)
        self.play(FadeIn(dot(p, ORANGE)), FadeIn(dot(q, ORANGE)))
        self.wait_speech()

        rho = 1.9
        rr = max(circle_circle(p, rho, q, rho), key=lambda x: np.linalg.norm(x - o))
        self.caption("...ועוד שתי קשתות באותו רדיוס",
                     say="עכשיו שתי קשתות באותו רדיוס, אחת סביב כל נקודה, ונקודת המפגש שלהן.")
        for c in (p, q):
            a0 = angle_of_vector(rr - c)
            self.compass_arc(c, rho, a0 - 25 * DEGREES, 50 * DEGREES, run_time=1)
        self.play(FadeIn(dot(rr, ORANGE)))
        self.wait_speech()

        self.say("הקו מהקודקוד לנקודת המפגש חוצה את הזווית בדיוק.")
        self.ruler_line(o, rr, extend=2.4, extend_start=0)
        halves = VGroup(
            angle_mark(o, arm1, rr, 1.15, GOOD, R"\alpha/2", 34, 0.45),
            angle_mark(o, rr, arm2, 1.45, GOOD, R"\alpha/2", 34, 0.45),
        )
        self.play(FadeIn(halves))
        self.caption("לחצות זווית לשתיים: קל", wait=1)
        self.wait_speech()

        keep = VGroup(arms, mark)
        self.play(*(FadeOut(m) for m in self.mobjects if m not in (*keep, self.cap)))
        thirds = VGroup(*(DashedLine(o, o + 5 * direction(alpha * k / 3)).set_stroke(YELLOW, 3) for k in (1, 2)))
        question = Text("?", font=FONT, font_size=72, fill_color=YELLOW).move_to(o + 3.6 * direction(alpha / 2) + RIGHT)
        self.caption("ולחלק זווית לשלוש?",
                     say="אז לחצות זווית קל. אבל לחלק זווית לשלושה חלקים שווים? זה כבר סיפור אחר לגמרי.")
        self.play(ShowCreation(thirds), Write(question))
        self.wait(1)
        self.fade_all()


class ThreeProblems(ConstructionScene):
    def construct(self):
        xs = [4.6, 0, -4.6]
        titles = VGroup(*(heb(t, 34, GOLD).move_to([x, 3.0, 0]) for t, x in
                          zip(["ריבוע העיגול", "חלוקת זווית לשלוש", "הכפלת הקובייה"], xs)))
        r = 0.85
        disk = Circle(radius=r).move_to([3.55, 0.6, 0]).set_stroke(NUM, 3).set_fill(NUM, 0.3)
        square = Square(r * np.sqrt(PI)).move_to([5.65, 0.6, 0]).set_stroke(GOLD, 3).set_fill(GOLD, 0.3)
        eq = Tex("=", font_size=48).move_to([4.6, 0.6, 0])
        circle_panel = VGroup(disk, eq, square, heb("ריבוע באותו שטח כמו העיגול", 26).move_to([4.6, -0.9, 0]))

        v, ang = np.array([-1.2, -0.3, 0]), 75 * DEGREES
        angle_panel = VGroup(
            Line(v, v + 2.6 * RIGHT).set_stroke(WHITE, 3), Line(v, v + 2.6 * direction(ang)).set_stroke(WHITE, 3),
            *(DashedLine(v, v + 2.3 * direction(ang * k / 3)).set_stroke(YELLOW, 2.5) for k in (1, 2)),
            Text("?", font=FONT, font_size=44, fill_color=YELLOW).move_to(v + 1.3 * RIGHT + 1.6 * UP),
            heb("זווית שהיא שליש מהזווית הנתונה", 26).move_to([0, -0.9, 0]),
        )

        cube_panel = VGroup(
            iso_cube(np.array([-6.5, -0.3, 0]), 1.0, BLUE),
            iso_cube(np.array([-4.6, -0.3, 0]), 1.26, PURPLE_B),
            Tex("V", font_size=34).move_to([-6.0, -0.55, 0]),
            Tex("2V", font_size=34).move_to([-4.0, -0.55, 0]),
            heb("קובייה בנפח כפול", 26).move_to([-4.6, -1.1, 0]),
        )

        self.say("אלה שלוש הבעיות המפורסמות.")
        self.play(FadeIn(titles[0]), FadeIn(circle_panel))
        self.say("ריבוע העיגול: בהינתן עיגול, לבנות ריבוע עם אותו שטח בדיוק.")
        self.wait_speech()
        self.play(FadeIn(titles[1]), FadeIn(angle_panel))
        self.say("חלוקת זווית לשלוש: בהינתן זווית, לבנות זווית שהיא שליש ממנה.")
        self.wait_speech()
        self.play(FadeIn(titles[2]), FadeIn(cube_panel))
        self.say("והכפלת הקובייה: בהינתן קובייה, לבנות קובייה שהנפח שלה כפול. "
                 "בקובייה זה בעצם לבנות קטע באורך של הצלע החדשה.")
        self.wait_speech()

        self.caption("ריבוע העיגול לא אומר לחתוך את העיגול לחתיכות", size=32,
                     say="ושימו לב: ריבוע העיגול לא דורש לחתוך את העיגול לחתיכות ולהרכיב מהן ריבוע. "
                         "רק לבנות ריבוע חדש, שהשטח שלו שווה. אין שום סיבה ברורה שזה יהיה בלתי אפשרי.")
        self.play(FlashAround(circle_panel[:3], color=GOLD), run_time=1.5)
        self.caption("אלפיים שנה אף אחד לא הצליח. במאה ה-19 הוכח שזה בלתי אפשרי", size=32,
                     say="כל השלוש נראות סבירות. ובכל זאת, במשך אלפיים שנה הן הביסו את כל מי שניסה, "
                         "עד שבמאה התשע עשרה הוכח שאין להן פתרון.")
        self.wait(1)
        self.fade_all()


class MarkedRuler(ConstructionScene):
    def construct(self):
        o, R = np.array([1.8, -0.9, 0]), 1.8
        theta = 60 * DEGREES
        a = o + R * direction(theta)
        circle = Circle(radius=R).move_to(o).set_stroke(CIRCLE, 2.5)
        diameter = Line([-6.8, o[1], 0], o + (R + 1.2) * RIGHT).set_stroke(LINE, 2)
        radius_oa = Line(o, a).set_stroke(WHITE, 3)
        theta_mark = angle_mark(o, o + RIGHT, a, 0.55, YELLOW, R"\theta", 32)
        self.caption("ארכימדס ידע לחלק כל זווית לשלוש. אבל הוא רימה",
                     say="לפני שנבין למה אי אפשר, הנה הפתעה. ארכימדס ידע לחלק כל זווית לשלוש. הוא רק רימה קצת.")
        self.play(ShowCreation(circle), ShowCreation(diameter), ShowCreation(radius_oa), FadeIn(dot(o, WHITE)))
        self.play(FadeIn(theta_mark), FadeIn(dot(a)))
        self.wait_speech()

        c_tracker = ValueTracker(3.4)

        def ruler():
            c = o + c_tracker.get_value() * R * LEFT
            u = unit(a - c)
            mark2 = c + R * u
            body = straightedge(c - 0.5 * u, a + 1.0 * u, width=0.32).set_fill(WOOD, 0.45)
            ticks = VGroup(*(Line(m, m - 0.32 * perp(u)).set_stroke(RED, 4) for m in (c, mark2)))
            return VGroup(body, Line(c - 0.5 * u, a + 1.0 * u).set_stroke(WOOD, 2), ticks,
                          Dot(c, radius=0.06, fill_color=RED), Dot(mark2, radius=0.06, fill_color=RED))

        tool = always_redraw(ruler)
        self.caption("על הסרגל שתי שנתות, במרחק של רדיוס אחד", size=32,
                     say="הוא סימן על הסרגל שתי שנתות, במרחק של רדיוס אחד זו מזו. "
                         "מעבירים את הסרגל דרך הנקודה שעל המעגל, ומחליקים אותו, "
                         "עד ששנתה אחת יושבת על הקוטר, והשנייה בדיוק על המעגל.")
        self.play(FadeIn(tool))
        phi = theta / 3
        self.play(c_tracker.animate.set_value(1.55), run_time=3)
        self.play(c_tracker.animate.set_value(2 * np.cos(phi)), run_time=2)
        tool.clear_updaters()
        c = o + 2 * np.cos(phi) * R * LEFT
        b = c + R * unit(a - c)
        self.play(Flash(b, color=RED, flash_radius=0.35))
        self.wait_speech()

        self.play(FadeOut(tool[0]), FadeOut(tool[2]))
        segs = VGroup(Line(c, a).set_stroke(WOOD, 3), Line(o, b).set_stroke(WHITE, 3))
        self.play(ShowCreation(segs))
        marks = VGroup(
            angle_mark(c, o, a, 0.75, GOOD, R"\varphi", 30),
            angle_mark(o, c, b, 0.45, GOOD, R"\varphi", 26, 0.22),
            angle_mark(b, o, a, 0.35, ORANGE, R"2\varphi", 26, 0.32),
            angle_mark(a, b, o, 0.45, ORANGE, R"2\varphi", 26, 0.32),
        )
        new_theta = angle_mark(o, o + RIGHT, a, 0.55, YELLOW, R"\theta = 3\varphi", 32, 0.55)
        self.caption("שני משולשים שווי שוקיים, וכל אחד מכפיל את הזווית", size=32,
                     say="למה זה עובד? המרחק בין השנתות שווה לרדיוס, אז יש כאן שני משולשים שווי שוקיים. "
                         "הזווית ליד הסרגל, נקרא לה פִי, מוכפלת בכל משולש: פי, אחר כך פעמיים פי, "
                         "ובסוף הזווית המקורית שווה לשלוש פי. הזווית הקטנה היא בדיוק שליש.")
        self.play(LaggedStart(*(FadeIn(m) for m in marks), lag_ratio=0.6, run_time=4))
        self.play(Transform(theta_mark, new_theta))
        self.wait_speech()
        self.caption("אבל לסרגל של אוקלידס אין שנתות. הבנייה הזאת לא חוקית", size=32,
                     say="אבל זה לא חוקי. לסרגל של אוקלידס אין שנתות, ובלעדיהן הטריק הזה נופל. "
                         "השאלה האמיתית היא מה אפשר לעשות לפי הכללים.")
        self.play(FlashAround(VGroup(tool[3], tool[4], tool[1]), color=RED), run_time=1.5)
        self.wait(1)
        self.fade_all()


class ToNumbers(ConstructionScene):
    def construct(self):
        u, y0 = 1.2, -0.6
        p0, p1 = np.array([-1.5, y0, 0]), np.array([-1.5 + u, y0, 0])
        self.caption("הרעיון הגדול: לתרגם את הגאומטריה למספרים",
                     say="ועכשיו הרעיון הגדול, שבזכותו אפשר בכלל להוכיח משהו: לתרגם את הגאומטריה למספרים. "
                         "נקרא לשתי הנקודות שלנו אפס ואחת, והישר שעובר דרכן יהיה ציר האיקס.")
        self.play(FadeIn(dot(p0)), FadeIn(dot(p1)))
        x_axis = self.ruler_line(p0, p1, extend=5.5, extend_start=5.3)
        nums = VGroup(Tex("0", font_size=30).next_to(p0, DR, buff=0.08), Tex("1", font_size=30).next_to(p1, DR, buff=0.08))
        self.play(FadeIn(nums))
        self.wait_speech()

        self.caption("שלושה מעגלים, וציר ה-y מאונך לו",
                     say="בעזרת שלושה מעגלים בונים ישר שמאונך לציר, ציר הוואי. וכבר יש לנו מערכת צירים.")
        pm = p0 - u * RIGHT
        c1 = self.compass_circle(p0, p1, run_time=1.1)
        helpers = VGroup(dot(pm, ORANGE))
        self.play(FadeIn(helpers[0]))
        c2 = self.compass_circle(pm, pm + 2 * u * RIGHT, run_time=1.1)
        c3 = self.compass_circle(p1, p1 - 2 * u * RIGHT, run_time=1.1)
        top, bottom = circle_circle(pm, 2 * u, p1, 2 * u)
        helpers.add(dot(top, ORANGE), dot(bottom, ORANGE))
        self.play(FadeIn(helpers[1:]))
        self.ruler_line(bottom, top, extend=2.4, extend_start=0.3)
        self.play(FadeOut(VGroup(c1, c2, c3, helpers)))
        self.wait_speech()

        p, q = np.array([2.8, 2.7, 0]), np.array([5.2, 1.5, 0])
        seg = VGroup(Line(p, q).set_stroke(GOLD, 4), dot(p, GOLD), dot(q, GOLD))
        self.caption("כל קטע שכבר בנינו אפשר להעתיק אל הציר",
                     say="ועוד דבר: כל קטע שכבר בנינו, איפה שהוא, אפשר להעתיק אל הציר. "
                         "פותחים את המחוגה על הקצוות שלו, נועצים אותה בראשית, ומסובבים עד הציר.")
        self.play(ShowCreation(seg))
        length = np.linalg.norm(q - p)
        self.transfer(p, q, p0, end_angle=0, color=GOLD, run_time=1.4)
        new = p0 + length * RIGHT
        self.play(FadeIn(dot(new, GOLD)), Write(Tex("a", font_size=34).set_color(GOLD).next_to(new, DR, buff=0.08)),
                  ShowCreation(Line(p0, new).set_stroke(GOLD, 5)))
        self.wait_speech()
        self.caption("השאלה עכשיו: אילו מספרים אפשר לסמן על הציר?",
                     say="ולכן כל המשחק הוא בעצם שאלה אחת: אילו מספרים אפשר לסמן על ציר האיקס?")
        self.wait(1)
        self.fade_all()


class AddSubtract(ConstructionScene):
    def construct(self):
        y0, x0 = 0.6, -5.6
        a_val, b_val = 4.0, 1.6

        def x(v):
            return np.array([x0 + v, y0, 0])

        axis = Line(x(-0.6), x(11.8)).set_stroke(LINE, 2)
        self.play(ShowCreation(axis))
        pts = VGroup(dot(x(0), WHITE), dot(x(1), WHITE), dot(x(b_val), BLUE), dot(x(a_val), GOLD))
        labels = VGroup(Tex("0", font_size=32).next_to(x(0), DOWN), Tex("1", font_size=32).next_to(x(1), DOWN),
                        Tex("b", font_size=36).set_color(BLUE).next_to(x(b_val), DOWN),
                        Tex("a", font_size=36).set_color(GOLD).next_to(x(a_val), DOWN))
        self.caption("חיבור וחיסור: מעגל ברדיוס b סביב a",
                     say="נניח שכבר בנינו שני מספרים, אֵיי וּבִּי. איך בונים את הסכום וההפרש? "
                         "מעתיקים את בי עם המחוגה, ומשרטטים מעגל ברדיוס בי סביב אֵיי.")
        self.play(FadeIn(pts), FadeIn(labels))
        self.transfer(x(0), x(b_val), x(a_val), color=BLUE, run_time=1.6)
        results = VGroup(dot(x(a_val + b_val), ORANGE), dot(x(a_val - b_val), ORANGE))
        rlabels = VGroup(Tex("a+b", font_size=36).set_color(ORANGE).next_to(x(a_val + b_val), UR, buff=0.1),
                         Tex("a-b", font_size=36).set_color(ORANGE).next_to(x(a_val - b_val), UL, buff=0.1))
        self.wait_speech()
        self.say("המעגל חותך את הציר בשתי נקודות: אֵיי ועוד בי, ואֵיי פחות בי.")
        self.play(GrowFromCenter(results[0]), GrowFromCenter(results[1]), Write(rlabels))
        self.wait(1)
        self.fade_all()


class MultiplyDivide(ConstructionScene):
    def construct(self):
        o, u = np.array([-5.6, -1.9, 0]), 1.6
        ang = 36 * DEGREES
        a_val, b_val = 1.6, 2.5
        d = direction(ang)

        def on_x(v):
            return o + v * u * RIGHT

        def on_ray(v):
            return o + v * u * d

        axis = Line(o, on_x(5.2)).set_stroke(LINE, 2)
        ray = Line(o, on_ray(4.6)).set_stroke(LINE, 2)
        self.caption("כפל: ישר בזווית, וקו מקביל",
                     say="כפל וחילוק מסובכים קצת יותר, אבל יפים. מעבירים מהראשית ישר נוסף בזווית כלשהי. "
                         "על ציר האיקס מסמנים אחת ובי, ועל הישר המשופע מסמנים אֵיי.")
        self.play(ShowCreation(axis), ShowCreation(ray), FadeIn(dot(o, WHITE)))
        marks = VGroup(dot(on_x(1), WHITE), dot(on_x(b_val), BLUE), dot(on_ray(a_val), GOLD))
        labels = VGroup(Tex("1", font_size=34).next_to(on_x(1), DOWN),
                        Tex("b", font_size=36).set_color(BLUE).next_to(on_x(b_val), DOWN),
                        Tex("a", font_size=36).set_color(GOLD).next_to(on_ray(a_val), UL, buff=0.1))
        self.play(FadeIn(marks), FadeIn(labels))
        self.wait_speech()

        self.say("מחברים את אחת עם אֵיי, ומעבירים דרך בי קו מקביל.")
        small = self.ruler_line(on_x(1), on_ray(a_val), extend=0.3)
        self.wait_speech()

        self.caption("קו מקביל: משלימים למקבילית", size=32,
                     say="איך מעבירים קו מקביל רק עם סרגל ומחוגה? בונים מקבילית.")
        ab = on_ray(a_val * b_val)
        q, helpers = self.parallel_through(on_x(1), on_ray(a_val), on_x(b_val), ORANGE, says=(
            "פותחים את המחוגה על הקטע שבין אחת לאֵיי, ומשרטטים קשת סביב בי.",
            "אחר כך פותחים אותה על הקטע שבין אחת לבי, ומשרטטים קשת סביב אֵיי.",
            "בנקודה שבה הקשתות נחתכות יש מרובע שהצלעות הנגדיות שלו שוות, כלומר מקבילית. "
            "ובמקבילית הצלעות הנגדיות מקבילות."))
        self.wait_speech()
        self.say("עכשיו מניחים את הסרגל על בי ועל הנקודה החדשה. זה הקו המקביל.")
        par = self.ruler_line(on_x(b_val), ab, extend=0.3, extend_start=0.3, color=ORANGE)
        self.play(FadeOut(helpers))
        ab_mark = VGroup(dot(ab, ORANGE), Tex("ab", font_size=38).set_color(ORANGE).next_to(ab, UL, buff=0.1))
        self.play(FadeIn(ab_mark[0]), Write(ab_mark[1]))
        self.wait_speech()

        tri_small = Polygon(o, on_x(1), on_ray(a_val)).set_stroke(width=0).set_fill(GOLD, 0.35)
        tri_big = Polygon(o, on_x(b_val), ab).set_stroke(width=0).set_fill(ORANGE, 0.18)
        self.caption("משולשים דומים: הכול גדל פי b", size=32,
                     say="למה? כי המשולש הקטן והמשולש הגדול דומים. הצלע אחת גדלה לבי, כלומר פי בי, "
                         "ולכן גם הצלע אֵיי גדלה פי בי. הנקודה החדשה היא בדיוק אֵיי כפול בי.")
        self.play(FadeIn(tri_big), FadeIn(tri_small))
        side_labels = VGroup(
            Tex(R"\times b", font_size=34).set_color(BLUE).next_to(on_x(b_val / 2 + 0.6), DOWN, buff=0.45),
            Tex(R"\times b", font_size=34).set_color(ORANGE).move_to(on_ray(a_val * b_val * 0.62) + 0.45 * perp(d)),
        )
        self.play(Write(side_labels))
        self.wait_speech()

        self.play(FadeOut(VGroup(small, par, tri_small, tri_big, side_labels, ab_mark)))
        self.caption("חילוק: אותו טריק, בתפקידים הפוכים", size=32,
                     say="חילוק עובד באותה דרך בתפקידים הפוכים: מחברים את אֵיי עם בי, "
                         "ומעבירים דרך אחת קו מקביל, שוב בעזרת מקבילית. מקבלים אֵיי חלקי בי.")
        line_ab = self.ruler_line(on_x(b_val), on_ray(a_val), extend=0.4)
        _, helpers2 = self.parallel_through(on_x(b_val), on_ray(a_val), on_x(1), GREEN)
        q = on_ray(a_val / b_val)
        far = on_x(1) + (on_ray(a_val) - on_x(b_val))
        par2 = self.ruler_line(on_x(1), far, extend=0.3, extend_start=0.3, color=GREEN)
        self.play(FadeOut(helpers2))
        self.play(FadeIn(dot(q, GREEN)), Write(Tex(R"\tfrac{a}{b}", font_size=40).set_color(GREEN).next_to(q, UL, buff=0.08)))
        self.wait_speech()
        self.wait(2)

        self.play(*(FadeOut(m) for m in self.mobjects if m is not self.cap))
        chain = Tex(R"1,\quad 2 = 1+1,\quad 3 = 2+1,\quad \tfrac{2}{3} = 2 : 3,\quad \ldots", font_size=52)
        rationals = Tex(R"\mathbb{Q} = \left\{ \tfrac{m}{n} \right\}", font_size=64).set_color(NUM)
        VGroup(chain, rationals).arrange(DOWN, buff=0.8).move_to(0.6 * UP)
        self.caption("מאחת, בארבע פעולות החשבון: כל השברים", size=32,
                     say="ועכשיו המשחק נהיה אלגברי. מאחת בונים את שתיים, אחת ועוד אחת, ואת שלוש, "
                         "ואז את שני שלישים, וכך כל שבר. כלומר את כל המספרים הרציונליים.")
        self.play(Write(chain), run_time=2.5)
        self.play(Write(rationals))
        self.wait(1)
        self.fade_all()


class SquareRoot(ConstructionScene):
    def construct(self):
        u, a_val = 1.6, 2.0
        y0, x0 = -1.9, -6.5
        p0, pa, p2 = (np.array([x0 + v * u, y0, 0]) for v in (0, a_val, a_val + 1))
        m = (p0 + p2) / 2
        rad = (a_val + 1) * u / 2
        top = pa + np.sqrt(a_val) * u * UP

        self.caption("המחוגה נותנת עוד פעולה: שורש ריבועי",
                     say="אבל המחוגה נותנת עוד פעולה אחת, קצת יותר מחוכמת: הוצאת שורש ריבועי. "
                         "שמים קטע באורך אֵיי, וממשיכים אותו באחת.")
        base = Line(p0 + 0.3 * LEFT, p2 + 0.3 * RIGHT).set_stroke(LINE, 2.5)
        self.play(ShowCreation(base), *(FadeIn(dot(p, WHITE)) for p in (p0, pa, p2)))
        braces = VGroup(Brace(Line(p0, pa), DOWN), Brace(Line(pa, p2), DOWN))
        blabels = VGroup(Tex("a", font_size=36).next_to(braces[0], DOWN, buff=0.1),
                         Tex("1", font_size=36).next_to(braces[1], DOWN, buff=0.1))
        self.play(GrowFromCenter(braces[0]), GrowFromCenter(braces[1]), FadeIn(blabels))
        self.wait_speech()

        self.say("משרטטים חצי מעגל שהקטע כולו הוא הקוטר שלו, ומעלים אנך בנקודה אֵיי.")
        self.play(FadeIn(dot(m, GREY_B)))
        self.compass_arc(m, rad, 0, PI, run_time=1.6)
        d_line = Line(pa, top).set_stroke(ORANGE, 5)
        self.play(ShowCreation(d_line))
        d_label = Tex("d", font_size=40).set_color(ORANGE).next_to(d_line, RIGHT, buff=0.1)
        self.play(FadeIn(dot(top, ORANGE)), Write(d_label))
        self.wait_speech()

        legs = VGroup(Line(p0, top).set_stroke(BLUE, 4), Line(top, p2).set_stroke(GREEN, 4))
        leg_labels = VGroup(Tex("x", font_size=38).set_color(BLUE).move_to((p0 + top) / 2 + 0.3 * UL),
                            Tex("y", font_size=38).set_color(GREEN).move_to((top + p2) / 2 + 0.3 * UR))
        corners = VGroup(right_angle(top, p0 - top, p2 - top), right_angle(pa, p2 - pa, top - pa))
        self.caption("זווית שנשענת על קוטר היא ישרה", size=32,
                     say="מחברים את הנקודה העליונה לשני הקצוות. זווית שנשענת על קוטר במעגל היא תמיד ישרה, "
                         "ולכן יש כאן משולש ישר זווית גדול, ובתוכו שניים קטנים.")
        self.play(ShowCreation(legs), FadeIn(leg_labels))
        self.play(ShowCreation(corners))
        self.wait_speech()

        eqs = VGroup(
            Tex(R"x^2 + y^2 = (a+1)^2", font_size=40),
            Tex(R"x^2 = d^2 + a^2", font_size=40),
            Tex(R"y^2 = d^2 + 1", font_size=40),
            Tex(R"2d^2 + a^2 + 1 = a^2 + 2a + 1", font_size=40),
            Tex(R"d^2 = a", font_size=56).set_color(ORANGE),
        ).arrange(DOWN, buff=0.36).move_to([3.6, 0.6, 0])
        eqs[1]["x^2"].set_color(BLUE)
        eqs[2]["y^2"].set_color(GREEN)
        self.caption("פיתגורס, שלוש פעמים", size=32,
                     say="ועכשיו פיתגורס שלוש פעמים. במשולש הגדול, איקס בריבוע ועוד וואי בריבוע הוא אֵיי ועוד אחת, בריבוע. "
                         "בקטנים, איקס בריבוע הוא די בריבוע ועוד אֵיי בריבוע, ווואי בריבוע הוא די בריבוע ועוד אחת.")
        for eq in eqs[:3]:
            self.play(Write(eq), run_time=1.2)
        self.wait_speech()
        self.say("מציבים, ומצמצמים. כמעט הכול מתבטל, ונשאר: די בריבוע שווה אֵיי. האנך הוא בדיוק שורש אֵיי.")
        self.play(Write(eqs[3]), run_time=1.5)
        self.play(Write(eqs[4]))
        box = SurroundingRectangle(eqs[4], buff=0.15).set_stroke(ORANGE, 2)
        self.play(ShowCreation(box))
        self.wait_speech()

        self.play(FadeOut(VGroup(*eqs[:4], braces, blabels, legs, leg_labels, corners, d_line, d_label)),
                  *(FadeOut(m) for m in self.mobjects if isinstance(m, (Dot, Arc))))
        a_tracker = ValueTracker(a_val)

        def figure():
            a = a_tracker.get_value()
            q0, qa, q2 = (np.array([x0 + v * u, y0, 0]) for v in (0, a, a + 1))
            t = qa + np.sqrt(a) * u * UP
            return VGroup(
                Arc(0, PI, radius=(a + 1) * u / 2, arc_center=(q0 + q2) / 2).set_stroke(CIRCLE, 2.5),
                Line(q0, q2).set_stroke(LINE, 2.5),
                Line(qa, t).set_stroke(ORANGE, 5),
                Dot(t, radius=0.07, fill_color=ORANGE), Dot(qa, radius=0.06), Dot(q0, radius=0.06), Dot(q2, radius=0.06),
            )

        fig = always_redraw(figure)
        a_num = DecimalNumber(a_val, num_decimal_places=2, font_size=44)
        d_num = DecimalNumber(np.sqrt(a_val), num_decimal_places=3, font_size=44).set_color(ORANGE)
        readout = VGroup(VGroup(Tex("a =", font_size=44), a_num).arrange(RIGHT, buff=0.2),
                         VGroup(Tex("d =", font_size=44).set_color(ORANGE), d_num).arrange(RIGHT, buff=0.2))
        readout.arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([3.6, 1.6, 0])
        a_num.add_updater(lambda n: n.set_value(a_tracker.get_value()))
        d_num.add_updater(lambda n: n.set_value(np.sqrt(a_tracker.get_value())))
        self.remove(base)
        self.add(fig, readout)
        self.caption("כש-a = 2 מקבלים √2, מספר שאינו רציונלי", size=32,
                     say="כשאֵיי שווה שתיים, מקבלים שורש שתיים, שכבר אינו מספר רציונלי. "
                         "ואפשר להזיז את אֵיי, והאנך תמיד יוצא שורש של אֵיי.")
        self.play(a_tracker.animate.set_value(0.5), run_time=2.5)
        self.play(a_tracker.animate.set_value(3.0), run_time=3)
        self.play(a_tracker.animate.set_value(2.0), run_time=2)
        self.wait(1)
        fig.clear_updaters()
        self.fade_all()


class WhyNothingMore(ConstructionScene):
    def construct(self):
        axes = Axes((-0.5, 5, 1), (-0.5, 4, 1), width=6.2, height=5.1).move_to([-3.6, 0.5, 0])
        self.play(ShowCreation(axes))
        right = np.array([3.4, 1.2, 0])

        def show_eqs(*lines):
            group = VGroup(*(Tex(t, font_size=40) for t in lines)).arrange(DOWN, buff=0.4).move_to(right)
            return group

        self.caption("למה אין עוד פעולות? כי יש רק ישרים ומעגלים", size=32,
                     say="האם יש עוד פעולות שאפשר לעשות עם סרגל ומחוגה? מתברר שלא. הסיבה: "
                         "נקודות חדשות נולדות רק מחיתוך של ישרים ומעגלים. "
                         "ובגאומטריה אנליטית, של דקארט, ישר ומעגל הם פשוט משוואות.")
        l1 = axes.get_graph(lambda t: 0.5 * t + 1, x_range=(-0.5, 5)).set_stroke(GOLD, 3)
        l2 = axes.get_graph(lambda t: -t + 4, x_range=(0, 4.5)).set_stroke(GREEN, 3)
        self.play(ShowCreation(l1), ShowCreation(l2))
        p = dot(axes.c2p(2, 2), ORANGE)
        self.play(GrowFromCenter(p))
        eqs = show_eqs(R"y = ax + b", R"y = cx + d",
                       R"x = \frac{d-b}{a-c},\quad y = \frac{ad-bc}{a-c}")
        eqs[0].set_color(GOLD)
        eqs[1].set_color(GREEN)
        self.wait_speech()
        self.caption("שני ישרים: רק חיבור, חיסור, כפל וחילוק", size=32,
                     say="ישר הוא משוואה ממעלה ראשונה. כשפותרים שני ישרים יחד, הנקודה המשותפת יוצאת "
                         "רק מחיבור, חיסור, כפל וחילוק של המקדמים.")
        self.play(Write(eqs[:2]))
        self.play(Write(eqs[2]), run_time=1.5)
        self.wait_speech()

        self.play(FadeOut(VGroup(l2, p, eqs)))
        center, r = axes.c2p(2.6, 2.0), 1.4 * axes.x_axis.get_unit_size()
        circ = Circle(radius=r).move_to(center).set_stroke(CIRCLE, 3)
        self.play(ShowCreation(circ))
        hits = line_circle(axes.c2p(-0.5, 0.75), axes.c2p(5, 3.5), center, r)
        dots2 = VGroup(*(dot(h, ORANGE) for h in hits))
        self.play(*(GrowFromCenter(d) for d in dots2))
        eqs = show_eqs(R"(x-p)^2 + (y-q)^2 = r^2", R"Ax^2 + Bx + C = 0",
                       R"x = \frac{-B \pm \sqrt{B^2 - 4AC}}{2A}")
        eqs[0].set_color(CIRCLE)
        self.caption("ישר ומעגל: משוואה ריבועית, ונכנס שורש ריבועי", size=32,
                     say="מעגל הוא משוואה ממעלה שנייה. מציבים בה את הישר, ומקבלים משוואה ריבועית. "
                         "ונוסחת השורשים מוסיפה לארבע הפעולות בדיוק דבר אחד: שורש ריבועי.")
        self.play(Write(eqs[0]))
        self.play(Write(eqs[1]))
        self.play(Write(eqs[2]), run_time=1.5)
        self.play(FlashAround(eqs[2][R"\sqrt{B^2 - 4AC}"], color=ORANGE))
        self.wait_speech()

        self.play(FadeOut(VGroup(l1, dots2, eqs)))
        c2, r2 = axes.c2p(1.4, 1.2), 1.2 * axes.x_axis.get_unit_size()
        circ2 = Circle(radius=r2).move_to(c2).set_stroke(GREEN, 3)
        self.play(ShowCreation(circ2))
        h1, h2 = circle_circle(center, r, c2, r2)
        radical = DashedLine(h1 + 1.2 * unit(h1 - h2), h2 + 1.2 * unit(h2 - h1)).set_stroke(ORANGE, 3)
        eqs = show_eqs(R"x^2 + y^2 + \ldots = \ldots \quad (1)", R"x^2 + y^2 + \ldots = \ldots \quad (2)",
                       R"(1) - (2):\quad ux + vy + w = 0")
        eqs[0].set_color(CIRCLE)
        eqs[1].set_color(GREEN)
        self.caption("שני מעגלים: מחסירים, הריבועים מתבטלים, ונשאר ישר", size=32,
                     say="ושני מעגלים? מחסירים את המשוואות זו מזו, איקס בריבוע ווואי בריבוע מתבטלים, "
                         "ונשארת משוואה של ישר שעובר דרך שתי נקודות החיתוך. חזרנו למקרה הקודם.")
        self.play(Write(eqs[:2]))
        self.play(Write(eqs[2]), ShowCreation(radical))
        self.play(GrowFromCenter(dot(h1, ORANGE)), GrowFromCenter(dot(h2, ORANGE)))
        self.wait_speech()
        self.play(*(FadeOut(m) for m in self.mobjects if m is not self.cap))

        head = heb("מספר ניתן לבנייה: כל מה שמתקבל מ-1 בפעולות", 38).move_to(2.9 * UP)
        ops = Tex(R"+ \qquad - \qquad \times \qquad : \qquad \sqrt{\ \ }", font_size=64).set_color(NUM)
        ops.next_to(head, DOWN, buff=0.45)
        chain = Tex(R"1 \to 2 \to \sqrt{2} \to 1 + \sqrt{2} \to \sqrt{1 + \sqrt{2}} \to \cdots", font_size=48)
        chain.next_to(ops, DOWN, buff=0.6)
        self.caption("זה כל הכוח של סרגל ומחוגה", size=32,
                     say="אז זה כל הכוח של סרגל ומחוגה. מספר ניתן לבנייה הוא מספר שמגיעים אליו מאחת, "
                         "בחיבור, חיסור, כפל, חילוק והוצאת שורש ריבועי. למשל, שורש של אחת ועוד שורש שתיים.")
        self.play(FadeIn(head))
        self.play(Write(ops))
        self.play(Write(chain), run_time=2.5)
        self.wait_speech()
        wanted = VGroup(
            VGroup(Tex(R"\sqrt[3]{2}", font_size=56), heb("הקובייה", 28, GREY_B)),
            VGroup(Tex(R"\cos 20^\circ", font_size=56), heb("הזווית", 28, GREY_B)),
            VGroup(Tex(R"\sqrt{\pi}", font_size=56), heb("העיגול", 28, GREY_B)),
        )
        for w in wanted:
            w.arrange(DOWN, buff=0.25)
        wanted.arrange(LEFT, buff=1.6).next_to(chain, DOWN, buff=0.8).set_color(YELLOW)
        for w in wanted:
            w[1].set_fill(GREY_B)
        self.caption("אבל איך מוכיחים שמספר אי אפשר לבנות?", size=32,
                     say="ושלוש הבעיות צריכות שלושה מספרים: הקובייה צריכה את השורש השלישי של שתיים, "
                         "הזווית את הקוסינוס של עשרים מעלות, והעיגול את שורש פאי. "
                         "אבל איך מוכיחים שמספר אי אפשר לבנות?")
        self.play(LaggedStart(*(FadeIn(w, shift=0.3 * UP) for w in wanted), lag_ratio=0.5, run_time=3))
        self.wait(1)
        self.fade_all()


class FieldPlane(ConstructionScene):
    def construct(self):
        head = heb("שדה: אוסף מספרים שסגור לארבע פעולות החשבון", 34).to_edge(UP, buff=0.35)
        self.caption("למשל, המספרים הרציונליים", size=32,
                     say="בשביל זה צריך מושג מהאלגברה: שדה. שדה הוא אוסף של מספרים שסגור לארבע פעולות החשבון. "
                         "מחברים, מחסרים, כופלים ומחלקים, ולא יוצאים ממנו. הדוגמה הבסיסית היא המספרים הרציונליים.")
        self.play(FadeIn(head))
        self.wait_speech()

        plane = NumberPlane((-3, 4, 1), (-2, 3, 1), background_line_style=dict(stroke_color=BLUE_E, stroke_width=1))
        plane.set_height(4.6).move_to([-3.1, -0.4, 0])
        xl = Tex(R"1", font_size=34).next_to(plane.c2p(4, 0), RIGHT, buff=0.15)
        yl = Tex(R"\sqrt{2}", font_size=34).next_to(plane.c2p(0, 3), UP, buff=0.1)
        form = Tex(R"\mathbb{Q}(\sqrt{2}) = \{\, a + b\sqrt{2} \,\}", font_size=48).set_color(NUM).move_to([3.4, 2.2, 0])
        self.caption("מוסיפים √2: כל מספר הוא a + b√2", size=32,
                     say="עכשיו נוסיף לרציונליים את שורש שתיים, ונסגור תחת ארבע הפעולות. "
                         "כל מספר שמתקבל הוא אֵיי ועוד בי כפול שורש שתיים, עם אֵיי ובי רציונליים. "
                         "כלומר, שני מספרים רציונליים מתארים אותו, והוא נקודה במישור.")
        self.play(ShowCreation(plane), FadeIn(xl), FadeIn(yl))
        self.play(Write(form))
        pts = {(1, 1): R"1 + \sqrt{2}", (2, 0): R"2", (0, 1): R"\sqrt{2}"}
        dots = VGroup(*(dot(plane.c2p(*k), GOLD) for k in pts))
        tags = VGroup(*(Tex(t, font_size=28).set_color(GOLD).next_to(plane.c2p(*k), UR, buff=0.06)
                        for k, t in pts.items()))
        self.play(FadeIn(dots), FadeIn(tags))
        self.wait_speech()

        facts = VGroup(
            Tex(R"(\sqrt{2})^2 = 2", font_size=40),
            Tex(R"(1+\sqrt{2})^2 = 3 + 2\sqrt{2}", font_size=40),
            Tex(R"\frac{1}{1+\sqrt{2}} = \sqrt{2} - 1", font_size=40),
        ).arrange(DOWN, buff=0.45, aligned_edge=LEFT).move_to([3.4, 0.0, 0])
        self.caption("כפל וחילוק לא מוציאים אותנו מהמישור", size=32,
                     say="ושום פעולה לא מוציאה אותנו מהמישור הזה. שורש שתיים בריבוע הוא שתיים, שחוזר לציר. "
                         "אחת ועוד שורש שתיים בריבוע, זה שלוש ועוד שתיים שורש שתיים. "
                         "ואפילו ההופכי שלו הוא שורש שתיים פחות אחת.")
        moves = [((0, 1), (2, 0)), ((1, 1), (3, 2)), ((1, 1), (-1, 1))]
        for fact, (src, dst) in zip(facts, moves):
            mover = dot(plane.c2p(*src), ORANGE)
            self.play(Write(fact), run_time=1.2)
            self.play(mover.animate.move_to(plane.c2p(*dst)).set_anim_args(path_arc=-0.6), run_time=1.2)
            self.add(dot(plane.c2p(*dst), ORANGE))
        self.wait_speech()
        dim = heb("המימד של השדה הזה: 2", 36, NUM).move_to([3.4, -2.2, 0])
        self.say("לכן אומרים שהמימד של השדה הזה, מעל הרציונליים, הוא שתיים.")
        self.play(FadeIn(dim))
        self.wait_speech()
        self.play(*(FadeOut(m) for m in self.mobjects if m not in (head, self.cap)))

        o = np.array([-4.8, -1.4, 0])
        arrows = VGroup(Arrow(o, o + 2.8 * RIGHT, buff=0), Arrow(o, o + 3.2 * UP, buff=0),
                        Arrow(o, o + 2.2 * direction(35 * DEGREES) + 0.3 * UP, buff=0))
        arrows.set_color(GOLD)
        alabels = VGroup(Tex("1", font_size=38).next_to(arrows[0], RIGHT),
                         Tex(R"\sqrt[3]{2}", font_size=38).next_to(arrows[1], UP),
                         Tex(R"\sqrt[3]{4}", font_size=38).next_to(arrows[2].get_end(), UR, buff=0.08))
        form3 = Tex(R"\mathbb{Q}(\sqrt[3]{2}) = \{\, a + b\sqrt[3]{2} + c\sqrt[3]{4} \,\}", font_size=44)
        form3.set_color(PURPLE_B).move_to([3.2, 1.8, 0])
        cube_back = Tex(R"(\sqrt[3]{2})^3 = 2", font_size=44).move_to([3.2, 0.6, 0])
        self.caption("עם ∛2 צריך שלושה כיוונים: המימד 3", size=32,
                     say="ועם השורש השלישי של שתיים? הפעם צריך שלושה מספרים: אחת, השורש השלישי של שתיים, "
                         "והשורש השלישי של ארבע. רק בחזקה השלישית חוזרים לרציונליים. המימד הוא שלוש.")
        self.play(LaggedStart(*(GrowArrow(a) for a in arrows), lag_ratio=0.4), FadeIn(alabels))
        self.play(Write(form3))
        self.play(Write(cube_back))
        self.wait_speech()

        rule = heb("המימד = המעלה של הפולינום הקטן ביותר שמתאפס על המספר", 30, YELLOW).move_to([0.4, -0.9, 0])
        examples = Tex(R"x^2 - 2 \;\to\; 2 \qquad\qquad x^3 - 2 \;\to\; 3", font_size=42).next_to(rule, DOWN, buff=0.35)
        rule.align_to(examples, RIGHT)
        VGroup(rule, examples).move_to([2.7, -2.0, 0])
        self.caption("המימד הוא המעלה של הפולינום המינימלי", size=32,
                     say="ויש לזה כלל: המימד של מספר שווה למעלה של הפולינום הקטן ביותר, עם מקדמים רציונליים, "
                         "שמתאפס עליו. לשורש שתיים, איקס בריבוע פחות שתיים. לשורש השלישי, איקס בשלישית פחות שתיים.")
        self.play(FadeIn(rule), Write(examples))
        self.wait(1)
        self.fade_all()


class Tower(ConstructionScene):
    def construct(self):
        law = Tex(R"[K:F] = [K:E]\cdot[E:F]", font_size=50).to_edge(UP, buff=0.35)
        self.caption("משפט המכפלה: המימדים מוכפלים", size=32,
                     say="ועכשיו משפט פשוט ויפה, משפט המכפלה. כשמרחיבים שדה, ואחר כך מרחיבים את התוצאה שוב, "
                         "המימדים מוכפלים זה בזה.")
        self.play(Write(law))
        self.wait_speech()

        bases = [["1"], ["1", R"\sqrt2"], ["1", R"\sqrt2", R"\sqrt3", R"\sqrt6"],
                 ["1", R"\sqrt2", R"\sqrt3", R"\sqrt5", R"\sqrt6", R"\sqrt{10}", R"\sqrt{15}", R"\sqrt{30}"]]
        names = [R"\mathbb{Q}", R"\mathbb{Q}(\sqrt2)", R"\mathbb{Q}(\sqrt2,\sqrt3)", R"\mathbb{Q}(\sqrt2,\sqrt3,\sqrt5)"]
        size = 0.72
        levels = VGroup()
        for i, (basis, name) in enumerate(zip(bases, names)):
            y = -2.4 + i * 1.15
            cells = VGroup(*(VGroup(Square(size).set_stroke(NUM, 2).set_fill(NUM, 0.2), Tex(t, font_size=22))
                             for t in basis)).arrange(RIGHT, buff=0)
            for cell in cells:
                cell[1].move_to(cell[0])
            cells.move_to([-1.6 + (len(basis) - 8) * size / 2, y, 0])
            cells.align_to(np.array([-4.5, 0, 0]), LEFT)
            label = Tex(name, font_size=32).move_to([3.0, y, 0]).align_to(np.array([1.6, 0, 0]), LEFT)
            dimn = Tex(str(len(basis)), font_size=40).set_color(YELLOW).move_to([-5.6, y, 0])
            levels.add(VGroup(cells, label, dimn))
        ups = VGroup(*(Tex(R"\times 2", font_size=30).set_color(YELLOW).move_to([-5.6, -2.4 + (i + 0.5) * 1.15, 0])
                       for i in range(3)))
        self.caption("כל שורש ריבועי: המימד כפול 2, או בלי שינוי", size=32,
                     say="כל הוצאת שורש ריבועי היא הרחבה ממימד שתיים, או אחת אם השורש כבר היה בשדה. "
                         "נוסיף לרציונליים שורש שתיים, אחר כך שורש שלוש, אחר כך שורש חמש: "
                         "המימד קופץ מאחת לשתיים, לארבע, לשמונה.")
        self.play(FadeIn(levels[0]))
        for i in range(1, 4):
            self.play(FadeIn(ups[i - 1]), TransformFromCopy(levels[i - 1][0], levels[i][0]),
                      FadeIn(levels[i][1:]), run_time=1.3)
        self.wait_speech()
        self.caption("כל מספר שאפשר לבנות חי בשדה שהמימד שלו 2 בחזקת k", size=32,
                     say="ומספר שאפשר לבנות נוצר ממספר סופי של צעדים כאלה. "
                         "לכן הוא חי בתוך שדה שהמימד שלו הוא חזקה של שתיים.")
        self.wait_speech()
        self.play(*(FadeOut(m) for m in self.mobjects if m not in (law, self.cap)))

        chain = Tex(R"\mathbb{Q} \subseteq \mathbb{Q}(\alpha) \subseteq F", font_size=54).move_to(1.8 * UP)
        prod = Tex(R"2^k = [F:\mathbb{Q}(\alpha)] \cdot [\mathbb{Q}(\alpha):\mathbb{Q}]", font_size=50).next_to(chain, DOWN, buff=0.5)
        prod[R"[\mathbb{Q}(\alpha):\mathbb{Q}]"].set_color(YELLOW)
        self.caption("והמימד של α מחלק את 2 בחזקת k", size=32,
                     say="עכשיו נסתכל על המספר אלפא עצמו. השדה שהוא יוצר יושב באמצע, ולפי משפט המכפלה, "
                         "המימד שלו כופל משהו ונותן חזקה של שתיים. ולכן גם הוא עצמו חזקה של שתיים.")
        self.play(Write(chain))
        self.play(Write(prod), run_time=1.5)
        self.wait_speech()

        blocks = VGroup(*(Square(0.6).set_stroke(WHITE, 2).set_fill(GREY_D, 0.6) for _ in range(8)))
        blocks.arrange(RIGHT, buff=0.12).move_to(1.1 * DOWN)
        self.caption("מימד 3, למשל, לא נכנס בשמונה", size=32,
                     say="מספר שהמימד שלו שלוש, למשל, לא יכול לשבת בתוך שדה ממימד שמונה: "
                         "שלוש לא מחלק את שמונה. תמיד יישאר עודף.")
        self.play(FadeIn(blocks))
        colors = [BLUE] * 3 + [GREEN] * 3 + [RED] * 2
        self.play(LaggedStart(*(b.animate.set_fill(c, 0.7) for b, c in zip(blocks, colors)), lag_ratio=0.25, run_time=2.5))
        self.play(*(Indicate(b, color=RED) for b in blocks[6:]))
        self.wait_speech()
        self.play(FadeOut(VGroup(blocks, chain, prod, law)))

        key = heb("אם מספר ניתן לבנייה, המימד שלו הוא חזקה של 2", 42, GOLD)
        frame = SurroundingRectangle(key, buff=0.35).set_stroke(GOLD, 3)
        self.caption("זה משפט המפתח", size=32,
                     say="וזה משפט המפתח. אם מספר ניתן לבנייה, המימד שלו הוא חזקה של שתיים. "
                         "ומכאן, כל שלוש הבעיות נופלות כמעט בבת אחת.")
        self.play(Write(key), ShowCreation(frame), run_time=2)
        self.wait(1)
        self.fade_all()


class DoubleCube(ConstructionScene):
    def construct(self):
        s = 1.25
        small = iso_cube(np.array([-6.2, -1.6, 0]), s, BLUE)
        self.caption("המלך מינוס הכפיל את הצלע, וקיבל פי שמונה", size=32,
                     say="נתחיל מהקובייה. יש סיפור יווני על המלך מינוס, שרצה להכפיל את הנפח של קבר בצורת קובייה. "
                         "הוא הכפיל את אורך הצלע. וזו טעות: קובייה עם צלע כפולה מכילה שמונה קוביות, לא שתיים.")
        self.play(FadeIn(small))
        big = VGroup()
        origin = np.array([-3.8, -1.6, 0])
        k = s * np.array([0.45, 0.32, 0])
        for depth in (1, 0):
            for yy in (0, 1):
                for xx in (0, 1):
                    big.add(iso_cube(origin + s * (xx * RIGHT + yy * UP) + depth * k, s, PURPLE_B))
        self.play(LaggedStart(*(FadeIn(c, scale=0.6) for c in big), lag_ratio=0.25, run_time=3))
        eight = Tex(R"2^3 = 8", font_size=48).next_to(big, UP, buff=0.3)
        self.play(Write(eight))
        self.wait_speech()

        self.play(FadeOut(big), FadeOut(eight))
        target = iso_cube(np.array([-3.8, -1.6, 0]), s * 2 ** (1 / 3), PURPLE_B).set_fill(opacity=0.15)
        target.set_stroke(PURPLE_B, 2)
        steps = VGroup(
            Tex(R"a^3 = 2 \quad\Rightarrow\quad a = \sqrt[3]{2} \approx 1.26", font_size=42),
            Tex(R"x^3 - 2", font_size=46).set_color(YELLOW),
            heb("ל-x³ − 2 אין שורש רציונלי, ולכן הוא לא מתפרק", 30),
            heb("המימד של ∛2 הוא 3, וזו לא חזקה של 2", 32, BAD),
        ).arrange(DOWN, buff=0.45).move_to([3.3, 0.6, 0])
        self.caption("הצלע הנכונה: ∛2", size=32,
                     say="קובייה בנפח שתיים צריכה צלע שהיא השורש השלישי של שתיים, בערך אחת נקודה עשרים ושש.")
        self.play(FadeIn(target))
        self.play(Write(steps[0]))
        self.wait_speech()
        self.caption("פולינום ממעלה 3 בלי שורש רציונלי הוא אי פריק", size=32,
                     say="המספר הזה מאפס את הפולינום איקס בשלישית פחות שתיים. אם הפולינום היה מתפרק, "
                         "אחד הגורמים היה ממעלה ראשונה, כלומר היה לו שורש רציונלי. "
                         "אבל אין שבר שהחזקה השלישית שלו היא בדיוק שתיים.")
        self.play(Write(steps[1]))
        self.play(FadeIn(steps[2]))
        self.wait_speech()
        self.caption("הכפלת הקובייה: בלתי אפשרית", size=32,
                     say="לכן הפולינום הזה הוא המינימלי, והמימד הוא שלוש. "
                         "שלוש אינו חזקה של שתיים. אי אפשר להכפיל את הקובייה.")
        self.play(FadeIn(steps[3]))
        self.play(ShowCreation(Cross(target)))
        self.wait(1)
        self.fade_all()


class AngleToCosine(ConstructionScene):
    def construct(self):
        o, R = np.array([-2.6, -1.4, 0]), 3.0
        axis = Line(o + 0.4 * LEFT, o + (R + 0.8) * RIGHT).set_stroke(LINE, 2)
        arc = Arc(0, PI / 2, radius=R, arc_center=o).set_stroke(CIRCLE, 2)
        alpha = ValueTracker(50 * DEGREES)

        def figure():
            a = alpha.get_value()
            p = o + R * direction(a)
            foot = np.array([p[0], o[1], 0])
            return VGroup(
                Line(o, p).set_stroke(WHITE, 3),
                DashedLine(p, foot).set_stroke(GREY_B, 2),
                Line(o, foot).set_stroke(NUM, 6),
                Dot(p, radius=0.07, fill_color=POINT),
                angle_mark(o, o + RIGHT, p, 0.6, YELLOW),
                right_angle(foot, LEFT, UP, 0.2),
                Tex("1", font_size=34).move_to((o + p) / 2 + 0.3 * perp(direction(a))),
                Tex(R"\cos\alpha", font_size=34).set_color(NUM).next_to(Line(o, foot), DOWN, buff=0.15),
            )

        fig = always_redraw(figure)
        self.caption("לבנות זווית = לבנות את הקוסינוס שלה", size=32,
                     say="עכשיו הזווית. איך הופכים זווית למספר? משרטטים משולש ישר זווית שהיתר שלו אחת. "
                         "הצלע שליד הזווית היא הקוסינוס שלה. ואם יש לנו את הקוסינוס, אפשר לבנות את הזווית בחזרה. "
                         "לבנות זווית, ולבנות את הקוסינוס שלה, זה אותו דבר.")
        self.play(ShowCreation(axis), ShowCreation(arc), FadeIn(fig))
        self.play(alpha.animate.set_value(75 * DEGREES), run_time=2)
        self.play(alpha.animate.set_value(30 * DEGREES), run_time=2)
        self.play(alpha.animate.set_value(60 * DEGREES), run_time=1.5)
        self.wait_speech()
        fig.clear_updaters()

        p60 = o + R * direction(60 * DEGREES)
        tri = Polygon(o, o + R * RIGHT, p60).set_stroke(GOLD, 3).set_fill(GOLD, 0.15)
        half = Tex(R"\cos 60^\circ = \tfrac12", font_size=46).move_to([3.9, 1.6, 0])
        self.caption("60° קל: משולש שווה צלעות", size=32,
                     say="שישים מעלות קל לבנות: זו הזווית של משולש שווה צלעות. והקוסינוס שלה הוא בדיוק חצי.")
        self.play(DrawBorderThenFill(tri))
        self.play(Write(half))
        self.wait_speech()

        rays = VGroup(*(Line(o, o + (R + 0.3) * direction(k * 30 * DEGREES)).set_stroke(GOOD, 3) for k in (1, 2, 3)))
        thirty = Tex(R"\tfrac{90^\circ}{3} = 30^\circ", font_size=46).set_color(GOOD).next_to(half, DOWN, buff=0.5)
        self.caption("90° אפשר לחלק לשלוש: חוצים את 60° ומקבלים 30°", size=32,
                     say="ומשם, אם חוצים את השישים, מקבלים שלושים. כלומר זווית ישרה אפשר לחלק לשלוש! "
                         "לפעמים זה כן אפשרי. השאלה היא אם יש שיטה שעובדת לכל זווית.")
        self.play(FadeOut(fig), FadeOut(tri), ShowCreation(rays))
        self.play(Write(thirty))
        self.wait_speech()
        self.caption("מספיקה זווית אחת שאי אפשר. ננסה את 60°", size=32,
                     say="כדי להראות שאין שיטה כללית, מספיקה זווית אחת שאי אפשר לחלק. ננסה את שישים מעלות. "
                         "שליש ממנה הוא עשרים מעלות, אז צריך לבנות את הקוסינוס של עשרים.")
        self.wait(1)
        self.fade_all()


class TrisectionCubic(ConstructionScene):
    def construct(self):
        formula = Tex(R"\cos 3x = 4\cos^3 x - 3\cos x", font_size=48).to_edge(UP, buff=0.3)
        plugged = Tex(R"\alpha = \cos 20^\circ:\qquad 4\alpha^3 - 3\alpha = \tfrac12", font_size=44)
        plugged.next_to(formula, DOWN, buff=0.3)
        self.caption("נוסחת הזווית המשולשת", size=32,
                     say="יש נוסחה לקוסינוס של זווית משולשת: ארבע קוסינוס בשלישית, פחות שלוש קוסינוס. "
                         "נציב שלוש איקס שווה שישים מעלות. הקוסינוס של שישים הוא חצי, ולכן הקוסינוס של עשרים, "
                         "נקרא לו אלפא, פותר משוואה ממעלה שלישית.")
        self.play(Write(formula), run_time=1.5)
        self.play(Write(plugged), run_time=1.5)
        self.wait_speech()

        o, R = np.array([-3.9, -0.7, 0]), 1.9
        circle = Circle(radius=R).move_to(o).set_stroke(CIRCLE, 2)
        xaxis = Line(o + (R + 0.3) * LEFT, o + (R + 0.3) * RIGHT).set_stroke(LINE, 1.5)
        axes = Axes((-1.2, 1.2, 0.5), (-1.3, 1.3, 0.5), width=2.4 * R, height=4.2).move_to([2.9, -0.7, 0])
        graph = axes.get_graph(lambda t: 4 * t ** 3 - 3 * t, x_range=(-1.03, 1.03)).set_stroke(WHITE, 3)
        level = DashedLine(axes.c2p(-1.2, 0.5), axes.c2p(1.2, 0.5)).set_stroke(GREY_B, 2)
        half = Tex(R"\tfrac12", font_size=30).next_to(axes.c2p(1.2, 0.5), RIGHT, buff=0.1)
        gl = Tex(R"4t^3 - 3t", font_size=30).next_to(axes.c2p(1.03, 4 * 1.03 ** 3 - 3 * 1.03), LEFT, buff=0.1)
        self.play(ShowCreation(circle), ShowCreation(xaxis), ShowCreation(axes))
        self.play(ShowCreation(graph), ShowCreation(level), FadeIn(half), FadeIn(gl))

        angles = [20, 140, 260]
        colors = [YELLOW, PINK, BLUE]
        ray60 = Line(o, o + R * direction(60 * DEGREES)).set_stroke(WHITE, 3)
        t60 = Tex(R"60^\circ", font_size=30).move_to(o + (R + 0.35) * direction(60 * DEGREES))
        self.caption("למשוואה יש שלושה פתרונות, לא אחד", size=32,
                     say="למה דווקא ממעלה שלישית? כי המשוואה לא יודעת לאיזו זווית התכוונו. "
                         "עשרים מעלות, מאה וארבעים, ומאתיים ושישים: כל אחת מהן, כפול שלוש, "
                         "נוחתת על שישים מעלות במעגל.")
        self.play(ShowCreation(ray60), FadeIn(t60))
        pts, roots = VGroup(), VGroup()
        for deg, color in zip(angles, colors):
            th = deg * DEGREES
            p = o + R * direction(th)
            c = np.cos(th)
            pts.add(VGroup(Line(o, p).set_stroke(color, 2.5), dot(p, color),
                           DashedLine(p, o + R * c * RIGHT).set_stroke(color, 2), dot(o + R * c * RIGHT, color, 0.06),
                           Tex(f"{deg}^\\circ", font_size=26).set_color(color).move_to(o + (R + 0.4) * direction(th))))
            roots.add(VGroup(dot(axes.c2p(c, 0.5), color),
                             DashedLine(axes.c2p(c, 0.5), axes.c2p(c, 0)).set_stroke(color, 2)))
        for deg, color, pt in zip(angles, colors, pts):
            th = deg * DEGREES
            self.play(FadeIn(pt), run_time=0.8)
            traveller = dot(o + R * direction(th), color, 0.09)
            tracker = ValueTracker(th)
            traveller.add_updater(lambda d, tr=tracker: d.move_to(o + R * direction(tr.get_value())))
            self.add(traveller)
            self.play(tracker.animate.set_value(3 * th), run_time=1.6)
            traveller.clear_updaters()
            self.play(Flash(traveller, color=color, flash_radius=0.3), FadeOut(traveller), run_time=0.6)
        self.wait_speech()
        self.caption("שלוש זוויות, שלושה קוסינוסים, שלושה פתרונות", size=32,
                     say="שלוש זוויות, שלושה קוסינוסים, ושלושה מקומות שבהם הגרף פוגש את החצי.")
        self.play(LaggedStart(*(FadeIn(r) for r in roots), lag_ratio=0.4, run_time=2))
        self.wait_speech()
        self.play(*(FadeOut(m) for m in self.mobjects if m not in (formula, self.cap)))

        beta = Tex(R"\beta = 2\alpha:\qquad \beta^3 - 3\beta - 1 = 0", font_size=50).set_color(YELLOW)
        beta.next_to(formula, DOWN, buff=0.7)
        self.caption("בטא = 2α: מקדמים שלמים", size=32,
                     say="נכפיל בשתיים ונסמן בטא שווה שתיים אלפא. מקבלים משוואה עם מקדמים שלמים: "
                         "בטא בשלישית, פחות שלוש בטא, פחות אחת, שווה אפס.")
        self.play(Write(beta), run_time=1.5)
        self.wait_speech()

        checks = VGroup(
            Tex(R"\beta = 1:\quad 1 - 3 - 1 = -3 \neq 0", font_size=44),
            Tex(R"\beta = -1:\quad -1 + 3 - 1 = 1 \neq 0", font_size=44),
        ).arrange(DOWN, buff=0.4).next_to(beta, DOWN, buff=0.7)
        self.caption("שורש רציונלי חייב לחלק את 1. נבדוק 1 ומינוס 1", size=32,
                     say="אם יש לה שורש רציונלי, הוא חייב להיות מספר שלם שמחלק את המקדם החופשי, אחת. "
                         "כלומר אחת או מינוס אחת. נבדוק: אחת נותן מינוס שלוש, ומינוס אחת נותן אחת. "
                         "אף אחד מהם לא אפס.")
        for c in checks:
            self.play(Write(c), run_time=1.3)
            self.play(c.animate.set_color(BAD), run_time=0.4)
        self.wait_speech()
        verdict = heb("המימד של cos 20° הוא 3, וזו לא חזקה של 2", 36, BAD).next_to(checks, DOWN, buff=0.6)
        self.caption("אין שורש רציונלי: הפולינום אי פריק", size=32,
                     say="אין שורש רציונלי, אז הפולינום לא מתפרק, והוא המינימלי. "
                         "המימד של קוסינוס עשרים הוא שלוש, ושלוש אינו חזקה של שתיים.")
        self.play(FadeIn(verdict))
        self.wait_speech()
        self.play(*(FadeOut(m) for m in self.mobjects if m is not self.cap))

        v = np.array([-1.5, -1.8, 0])
        arms = VGroup(Line(v, v + 4.5 * RIGHT), Line(v, v + 4.5 * direction(60 * DEGREES))).set_stroke(WHITE, 3)
        tries = VGroup(*(DashedLine(v, v + 4.2 * direction(k * 20 * DEGREES)).set_stroke(YELLOW, 3) for k in (1, 2)))
        mark = angle_mark(v, v + RIGHT, v + direction(60 * DEGREES), 0.8, WHITE, R"60^\circ", 34, 0.45)
        self.caption("אי אפשר לחלק 60° לשלוש בסרגל ומחוגה", size=32,
                     say="אי אפשר לבנות זווית של עשרים מעלות. ולכן אי אפשר לחלק שישים מעלות לשלוש, "
                         "ולא יכולה להיות שיטה שעובדת לכל זווית. "
                         "את זה הוכיח פייר וַנְצֶל, בשנת אלף שמונה מאות שלושים ושבע.")
        self.play(ShowCreation(arms), FadeIn(mark))
        self.play(ShowCreation(tries))
        self.play(ShowCreation(Cross(VGroup(tries, Dot(v + 4 * direction(40 * DEGREES))))))
        self.wait(1)
        self.fade_all()


class SquaringCircle(ConstructionScene):
    def construct(self):
        u = 1.4
        center = np.array([-4.4, 0.5, 0])
        disk = Circle(radius=u).move_to(center).set_stroke(NUM, 3).set_fill(NUM, 0.3)
        r_line = Line(center, center + u * RIGHT).set_stroke(WHITE, 2)
        r_label = Tex("1", font_size=30).next_to(r_line, UP, buff=0.08)
        side = u * np.sqrt(PI)
        square = Square(side).move_to([-0.9, 0.5, 0]).set_stroke(GOLD, 3).set_fill(GOLD, 0.3)
        s_label = Tex(R"\sqrt{\pi}", font_size=38).set_color(GOLD).next_to(square, DOWN, buff=0.15)
        area1 = Tex(R"S = \pi", font_size=36).next_to(disk, DOWN, buff=0.2)
        self.caption("עיגול ברדיוס 1, השטח שלו π", size=32,
                     say="ונשאר העיגול. עיגול שהרדיוס שלו אחת, השטח שלו פאי. "
                         "ריבוע באותו שטח צריך צלע שאורכה שורש פאי. כלומר צריך לבנות את המספר שורש פאי.")
        self.play(DrawBorderThenFill(disk), ShowCreation(r_line), FadeIn(r_label), FadeIn(area1))
        self.play(DrawBorderThenFill(square), FadeIn(s_label))
        self.wait_speech()
        self.play(square.animate.move_to(center).set_fill(opacity=0.2), FadeOut(s_label), run_time=1.5)
        self.wait(0.5)

        rows = VGroup(
            heb("אם אפשר לבנות את √π, המימד שלו סופי", 30),
            Tex(R"p(\sqrt{\pi}) = 0 \quad\Rightarrow\quad p(x)\,p(-x) = q(x^2) \quad\Rightarrow\quad q(\pi) = 0", font_size=36),
            heb("ואז גם π היה מאפס פולינום עם מקדמים רציונליים", 30),
        ).arrange(DOWN, buff=0.45).move_to([2.6, 1.6, 0])
        self.caption("אם √π ניתן לבנייה, גם π מאפס פולינום", size=32,
                     say="אם אפשר לבנות את שורש פאי, המימד שלו הוא חזקה של שתיים, ובפרט מספר סופי. "
                         "אז יש פולינום עם מקדמים רציונליים שמתאפס עליו. "
                         "וכשכופלים אותו בעצמו עם איקס הפוך בסימן, מקבלים פולינום שמתאפס על פאי עצמו.")
        self.play(FadeIn(rows[0]))
        self.play(Write(rows[1]), run_time=2)
        self.play(FadeIn(rows[2]))
        self.wait_speech()

        powers = Tex(R"1,\ \ \pi,\ \ \pi^2,\ \ \pi^3,\ \ \pi^4,\ \ \pi^5,\ \ \ldots", font_size=48).move_to([2.6, -0.6, 0])
        infinite = Tex(R"[\mathbb{Q}(\pi):\mathbb{Q}] = \infty", font_size=50).set_color(BAD).next_to(powers, DOWN, buff=0.45)
        who = heb("ב-1882 הוכיח לינדמן ש-π טרנסצנדנטי", 34, YELLOW).next_to(powers, UP, buff=0.4)
        self.caption("לינדמן, 1882: שום פולינום לא מתאפס על π", size=32,
                     say="אבל בשנת אלף שמונה מאות שמונים ושתיים הוכיח פרדיננד לינדמן, בעקבות עבודה של שארל הרמיט על המספר אֵה, "
                         "שפאי הוא טרנסצנדנטי. אין שום פולינום עם מקדמים רציונליים שמתאפס עליו.")
        self.play(FadeIn(who))
        self.play(Write(powers), run_time=2)
        self.wait_speech()
        self.caption("החזקות של π לא תלויות זו בזו: המימד אינסופי", size=32,
                     say="כלומר, החזקות של פאי לעולם לא תלויות זו בזו, והמימד שלו אינסופי. "
                         "ואינסוף בטח אינו חזקה של שתיים.")
        self.play(LaggedStart(*(Indicate(p, color=YELLOW) for p in powers), lag_ratio=0.1, run_time=2))
        self.play(Write(infinite))
        self.wait_speech()
        self.caption("אי אפשר לרבע את העיגול", size=36,
                     say="אי אפשר לרבע את העיגול. וההוכחה של לינדמן היא החלק היחיד בסיפור הזה שקשה באמת. "
                         "גם גדי, בבלוג, רק מספר על קיומה.")
        self.play(ShowCreation(Cross(square)))
        self.wait(1.5)
        self.fade_all()


class History(ConstructionScene):
    def construct(self):
        line = Line([-6.3, 0.4, 0], [6.3, 0.4, 0]).set_stroke(GREY_B, 3)
        events = [
            (-5.6, "בערך 300 לפנה״ס", "אוקלידס: יסודות", UP),
            (-1.0, "1637", "דקארט: גאומטריה אנליטית", DOWN),
            (2.4, "1837", "ונצל: הקובייה והזווית", UP),
            (5.4, "1882", "לינדמן: π וריבוע העיגול", DOWN),
        ]
        self.play(ShowCreation(line))
        lines = [
            "אוקלידס כינס את הגאומטריה היוונית בספר יסודות, בערך שלוש מאות שנה לפני הספירה.",
            "אלפיים שנה אחר כך, דקארט הפך ישרים ומעגלים למשוואות.",
            "פייר וַנְצֶל השתמש בזה, והוכיח שאי אפשר להכפיל את הקובייה ולחלק זווית לשלוש. "
            "את העיגול הוא לא יכול היה לסגור, כי אף אחד עוד לא ידע אם פאי אלגברי.",
            "ולינדמן סגר את הסיפור, כשהוכיח שפאי טרנסצנדנטי.",
        ]
        for (x, year, what, side), text in zip(events, lines):
            self.say(text)
            p = np.array([x, 0.4, 0])
            tick = dot(p, YELLOW, 0.1)
            label = VGroup(heb(year, 30, YELLOW), heb(what, 28)).arrange(DOWN, buff=0.15)
            label.next_to(p, side, buff=0.4)
            self.play(GrowFromCenter(tick), FadeIn(label, shift=0.2 * side))
            self.wait_speech()

        self.play(*(FadeOut(m) for m in self.mobjects))
        dead = heb("הבעיות האלה מתו ונקברו", 56, GOLD).move_to(0.9 * UP)
        run = heb("ואם מישהו טוען שהוא יודע לרבע את העיגול: ברחו מהר ככל האפשר!", 34).next_to(dead, DOWN, buff=0.6)
        self.say("גדי מסיים את הסדרה שלו בבקשה: הפיצו את הבשורה. הבעיות האלה מתו ונקברו. "
                 "וכשמישהו טוען שהוא יודע לחלק זווית לשלוש, בדרך כלל הוא פשוט שינה את כללי המשחק. "
                 "ואם הוא טוען שהוא יודע לרבע את העיגול, ברחו מהר ככל האפשר!")
        self.play(Write(dead), run_time=2)
        self.wait(5)
        self.play(FadeIn(run, shift=0.2 * UP))
        self.wait(1)
        self.fade_all()


class Credits(ConstructionScene):
    def construct(self):
        thanks = heb("תודה", 60, NUM)
        line1 = heb("הסרטון מבוסס על סדרת הפוסטים של", 34)
        line2 = heb("גדי אלכסנדרוביץ' בבלוג \"לא מדויק\"", 44, YELLOW)
        posts = VGroup(*(heb(t, 24, GREY_A) for t in [
            "2008: בניות בסרגל ומחוגה",
            "2009: הרחבה קלה על הרחבת שדות / עכשיו אפשר באמת להרחיב על הרחבת שדות",
            "2009: אז למה אי אפשר לרבע את העיגול?",
            "2018: שלוש בעיות הבניה בסרגל ומחוגה של היוונים",
        ])).arrange(DOWN, buff=0.18)
        site = Text("gadial.net", font=FONT, font_size=34, fill_color=GREY_B)
        VGroup(thanks, line1, line2, posts, site).arrange(DOWN, buff=0.4)
        self.say("הסרטון הזה מבוסס על סדרת הפוסטים של גדי אלכסנדרוביץ' על בניות בסרגל ומחוגה, "
                 "בבלוג שלו, לא מדויק. תודה, גדי.")
        self.play(Write(thanks))
        self.play(LaggedStartMap(FadeIn, VGroup(line1, line2, posts, site), shift=0.2 * UP, lag_ratio=0.4,
                                 run_time=3))
        self.wait(3)
        self.fade_all(run_time=1.5)


SCENES = [Title, Tools, Rules, Bisect, ThreeProblems, MarkedRuler, ToNumbers, AddSubtract, MultiplyDivide,
          SquareRoot, WhyNothingMore, FieldPlane, Tower, DoubleCube, AngleToCosine, TrisectionCubic,
          SquaringCircle, History, Credits]
