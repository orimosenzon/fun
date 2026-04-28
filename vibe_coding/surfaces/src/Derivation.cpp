#include "Derivation.h"

namespace {

// ─────────────────────────────────────────────────────────────
// SPHERE — explanation text
// ─────────────────────────────────────────────────────────────
QString sphereText() {
    return QStringLiteral(R"HTML(
<h3 style="color:#c084fc;">Sphere — איך מגיעים לנוסחה</h3>
<p>נקודה על כדור היחידה נקבעת ע"י <b>שתי זוויות</b>:</p>
<ul>
  <li><b style="color:#e879f9;">u</b> — הזווית מהציר העליון <i>+z</i>
      (קולטיטוד), 0 ≤ u ≤ π.</li>
  <li><b style="color:#22d3ee;">v</b> — הזווית סביב ציר <i>z</i> במישור
      <i>xy</i> (אזימוט), 0 ≤ v ≤ 2π.</li>
</ul>
<p>נסתכל על נקודה <i>P</i> על הכדור. במשולש הישר-זווית עם יתר 1
שצלעותיו הן הציר ה-<i>z</i> וההיטל למישור <i>xy</i>:</p>
<ul>
  <li>הקטטה האנכית = <span style="color:#e879f9;"><b>cos u</b></span> →
      זה <i>z</i> של הנקודה.</li>
  <li>הקטטה האופקית = <span style="color:#e879f9;"><b>sin u</b></span> →
      זה רדיוס ההיטל למישור <i>xy</i>.</li>
</ul>
<p>בהיטל למישור <i>xy</i> מקבלים מעגל ברדיוס <code>sin u</code>; הזווית
<b>v</b> אומרת איפה אנחנו עליו. לכן:</p>
<pre style="color:#e6e6ea;">x = sin u · cos v
y = sin u · sin v
z = cos u</pre>
)HTML");
}

// ─────────────────────────────────────────────────────────────
// SPHERE — SVG diagram
// We draw a perspective sphere viewed from slightly above-right.
// Layout numbers (computed once):
//   u = 50°, v = 35°.  sin u ≈ 0.766, cos u ≈ 0.643.
//   Center O at (260, 220), radius R = 150.
//   Z-axis goes straight up.  XY-plane is squashed (ry/rx = 0.32).
//   Point P projected:
//     Px = O.x + R*sinu*cosv*1     = 260 + 150*0.766*0.819 = 260 + 94.1 = 354.1
//     Py = O.y - R*cosu + R*sinu*sinv*0.32
//        = 220 - 96.5 + 150*0.766*0.574*0.32 = 220 - 96.5 + 21.1 = 144.6
//   Foot Q on xy-plane (z=0):
//     Qx = 354.1
//     Qy = 220 + 21.1 = 241.1
// ─────────────────────────────────────────────────────────────
QByteArray sphereSvg() {
    return QByteArray(R"SVG(
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 460" width="100%">
  <defs>
    <marker id="arrW" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 Z" fill="#cdd0d8"/>
    </marker>
    <marker id="arrU" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 Z" fill="#e879f9"/>
    </marker>
    <marker id="arrV" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 Z" fill="#22d3ee"/>
    </marker>
  </defs>

  <!-- title -->
  <text x="260" y="28" fill="#cdd0d8" font-size="15" text-anchor="middle"
        font-family="sans-serif">Sphere — שתי זוויות מגדירות נקודה</text>

  <!-- xy plane (perspective ellipse), faint -->
  <ellipse cx="260" cy="220" rx="200" ry="64" fill="#1f2027" stroke="#5a5c66"
           stroke-width="1" stroke-dasharray="3 3"/>

  <!-- back half of equator (dashed, behind sphere) -->
  <path d="M 60 220 A 200 64 0 0 1 460 220" fill="none" stroke="#5a5c66"
        stroke-width="1" stroke-dasharray="3 3"/>
  <!-- sphere outline (front silhouette) -->
  <circle cx="260" cy="220" r="150" fill="none" stroke="#9da0ab" stroke-width="1.5"/>
  <!-- equator (front half, solid) -->
  <path d="M 110 220 A 150 48 0 0 0 410 220" fill="none" stroke="#9da0ab" stroke-width="1.2"/>

  <!-- meridian containing P (great-circle arc tilted to azimuth v) -->
  <!-- to keep it simple, draw an ellipse rotated to azimuth v=35° -->
  <ellipse cx="260" cy="220" rx="150" ry="48"
           transform="rotate(-55 260 220)"
           fill="none" stroke="#9da0ab" stroke-width="1" stroke-dasharray="2 4"
           opacity="0.55"/>

  <!-- z-axis -->
  <line x1="260" y1="430" x2="260" y2="50" stroke="#cdd0d8"
        stroke-width="1.4" marker-end="url(#arrW)"/>
  <text x="270" y="56" fill="#cdd0d8" font-size="14" font-style="italic">z</text>

  <!-- x-axis (perspective) -->
  <line x1="260" y1="220" x2="478" y2="262" stroke="#cdd0d8"
        stroke-width="1" marker-end="url(#arrW)"/>
  <text x="484" y="270" fill="#cdd0d8" font-size="14" font-style="italic">x</text>
  <!-- y-axis -->
  <line x1="260" y1="220" x2="60" y2="262" stroke="#cdd0d8"
        stroke-width="1" marker-end="url(#arrW)"/>
  <text x="44" y="270" fill="#cdd0d8" font-size="14" font-style="italic">y</text>

  <!-- O label -->
  <circle cx="260" cy="220" r="2.5" fill="#cdd0d8"/>
  <text x="246" y="236" fill="#cdd0d8" font-size="13">O</text>

  <!-- N pole label -->
  <circle cx="260" cy="70" r="2.5" fill="#cdd0d8"/>
  <text x="240" y="64" fill="#9da0ab" font-size="12">+z</text>

  <!-- segment OP (radius = 1) -->
  <line x1="260" y1="220" x2="354.1" y2="144.6"
        stroke="#cdd0d8" stroke-width="1.8"/>
  <text x="298" y="178" fill="#cdd0d8" font-size="13"
        font-style="italic">1</text>

  <!-- height z = cos u  (from foot Q up to P) -->
  <line x1="354.1" y1="241.1" x2="354.1" y2="144.6"
        stroke="#e879f9" stroke-width="2"/>
  <text x="362" y="195" fill="#e879f9" font-size="14"
        font-weight="bold">cos u</text>

  <!-- horizontal radius sin u  (from O to Q in xy-plane) -->
  <line x1="260" y1="220" x2="354.1" y2="241.1"
        stroke="#fbbf24" stroke-width="2"/>
  <text x="284" y="244" fill="#fbbf24" font-size="14"
        font-weight="bold">sin u</text>

  <!-- right angle at Q -->
  <path d="M 346 241.1 L 346 234 L 354.1 234"
        fill="none" stroke="#cdd0d8" stroke-width="1"/>

  <!-- u angle arc near +z axis -->
  <!-- arc from (260,170) toward direction of OP, radius 50 -->
  <path d="M 260 170 A 50 50 0 0 1 298.3 188.6"
        fill="none" stroke="#e879f9" stroke-width="2"/>
  <text x="280" y="166" fill="#e879f9" font-size="15"
        font-weight="bold" font-style="italic">u</text>

  <!-- v angle arc on the xy-plane near +x axis -->
  <!-- center O, in the plane (rx=70, ry=22), from x-axis rotating toward azimuth -->
  <path d="M 326 233 A 70 22 0 0 1 314.5 240.5"
        fill="none" stroke="#22d3ee" stroke-width="2"/>
  <text x="316" y="262" fill="#22d3ee" font-size="15"
        font-weight="bold" font-style="italic">v</text>

  <!-- foot Q -->
  <circle cx="354.1" cy="241.1" r="3" fill="#fbbf24"/>
  <text x="358" y="258" fill="#fbbf24" font-size="12">Q  (היטל ל-xy)</text>

  <!-- point P -->
  <circle cx="354.1" cy="144.6" r="4" fill="#e879f9"/>
  <text x="362" y="142" fill="#e879f9" font-size="14"
        font-weight="bold">P</text>

  <!-- legend / summary box -->
  <g transform="translate(20,330)">
    <rect x="0" y="0" width="220" height="118" rx="6"
          fill="#0f1014" stroke="#3a3c44"/>
    <text x="12" y="22" fill="#cdd0d8" font-size="12" font-weight="bold">המשולש הישר-זווית O-Q-P:</text>
    <text x="12" y="44" fill="#e879f9" font-size="13">‎• OP = 1 (יתר)</text>
    <text x="12" y="64" fill="#e879f9" font-size="13">‎• QP = cos u (קטטה אנכית = z)</text>
    <text x="12" y="84" fill="#fbbf24" font-size="13">‎• OQ = sin u (קטטה אופקית)</text>
    <text x="12" y="104" fill="#22d3ee" font-size="13">‎• הזווית v סובבת את OQ במישור xy</text>
  </g>
</svg>
)SVG");
}

// ─────────────────────────────────────────────────────────────
// TORUS
// ─────────────────────────────────────────────────────────────
QString torusText() {
    return QStringLiteral(R"HTML(
<h3 style="color:#c084fc;">Torus — איך מגיעים לנוסחה</h3>
<p>טורוס נוצר ע"י <b>סיבוב מעגל</b> סביב ציר. נסמן:</p>
<ul>
  <li><b style="color:#fbbf24;">R</b> — הרדיוס הגדול: ממרכז הטורוס למרכז הצינור.</li>
  <li><b style="color:#22d3ee;">r</b> — הרדיוס הקטן: רדיוס חתך הצינור.</li>
  <li><b style="color:#e879f9;">v</b> — זווית בתוך החתך.</li>
  <li><b style="color:#22d3ee;">u</b> — זווית הסיבוב סביב ציר <i>z</i>.</li>
</ul>
<p>בחתך אנכי שעובר בציר: נקודה במעגל הקטן ברדיוס <i>r</i> וזווית <i>v</i>
נמצאת במרחק <code>R + r·cos v</code> מהציר ובגובה <code>r·sin v</code>.
מסובבים את כל החתך הזה בזווית <i>u</i> סביב ציר <i>z</i>:</p>
<pre style="color:#e6e6ea;">x = (R + r·cos v) · cos u
y = (R + r·cos v) · sin u
z =  r·sin v</pre>
)HTML");
}

QByteArray torusSvg() {
    return QByteArray(R"SVG(
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 380" width="100%">
  <defs>
    <marker id="arrT" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 Z" fill="#cdd0d8"/>
    </marker>
  </defs>

  <text x="260" y="26" fill="#cdd0d8" font-size="15" text-anchor="middle"
        font-family="sans-serif">Torus — חתך מאונך לציר הסיבוב</text>

  <!-- z-axis (axis of revolution) -->
  <line x1="80" y1="350" x2="80" y2="60" stroke="#cdd0d8"
        stroke-width="1.4" marker-end="url(#arrT)"/>
  <text x="62" y="68" fill="#cdd0d8" font-size="14" font-style="italic">z</text>

  <!-- horizontal radial axis (rho) -->
  <line x1="60" y1="200" x2="500" y2="200" stroke="#9da0ab"
        stroke-width="1" marker-end="url(#arrT)"/>
  <text x="492" y="220" fill="#9da0ab" font-size="13" font-style="italic">ρ (מרחק מהציר)</text>

  <!-- ghost cross-section (left side of axis, faint) -->
  <circle cx="80" cy="200" r="0" fill="none"/>
  <circle cx="-200" cy="200" r="60" fill="none"/>
  <!-- the small cross-section circle (right) -->
  <circle cx="320" cy="200" r="70" fill="none" stroke="#9da0ab" stroke-width="1.4"/>
  <circle cx="320" cy="200" r="2.5" fill="#cdd0d8"/>
  <text x="304" y="218" fill="#cdd0d8" font-size="12">C</text>

  <!-- R: distance from z-axis to circle center -->
  <line x1="80" y1="160" x2="320" y2="160" stroke="#fbbf24"
        stroke-width="2" marker-end="url(#arrT)" marker-start="url(#arrT)"/>
  <text x="170" y="152" fill="#fbbf24" font-size="15" font-weight="bold">R</text>

  <!-- r: from C to point on tube circle (v ≈ 50°) -->
  <!-- v = 50°: cos=0.643, sin=0.766 ; tube radius 70 -->
  <!-- P_tube = (320 + 70*0.643, 200 - 70*0.766) = (365.0, 146.4) -->
  <line x1="320" y1="200" x2="365.0" y2="146.4" stroke="#22d3ee"
        stroke-width="2"/>
  <text x="334" y="178" fill="#22d3ee" font-size="14" font-weight="bold">r</text>

  <!-- v angle arc (from +rho direction at C, sweep to point) -->
  <path d="M 350 200 A 30 30 0 0 0 339.3 176.0"
        fill="none" stroke="#e879f9" stroke-width="2"/>
  <text x="346" y="190" fill="#e879f9" font-size="14"
        font-weight="bold" font-style="italic">v</text>

  <!-- tube point P -->
  <circle cx="365.0" cy="146.4" r="4" fill="#e879f9"/>
  <text x="372" y="144" fill="#e879f9" font-size="14" font-weight="bold">P</text>

  <!-- horizontal projection from z-axis to P_x (R + r cos v) -->
  <line x1="80" y1="280" x2="365.0" y2="280" stroke="#fbbf24"
        stroke-width="2" stroke-dasharray="6 4"
        marker-end="url(#arrT)" marker-start="url(#arrT)"/>
  <text x="170" y="298" fill="#fbbf24" font-size="14" font-weight="bold">R + r·cos v</text>

  <!-- vertical from z=0 plane up to P (r sin v) -->
  <line x1="365.0" y1="200" x2="365.0" y2="146.4" stroke="#22d3ee"
        stroke-width="2" stroke-dasharray="4 3"/>
  <text x="372" y="178" fill="#22d3ee" font-size="13" font-weight="bold">r·sin v</text>

  <!-- horizontal dotted from P to its column -->
  <line x1="365.0" y1="146.4" x2="365.0" y2="200" stroke="#22d3ee"
        stroke-width="1" stroke-dasharray="2 3" opacity="0.55"/>
  <line x1="80" y1="200" x2="80" y2="280" stroke="#fbbf24"
        stroke-width="1" stroke-dasharray="2 3" opacity="0.55"/>
  <line x1="365.0" y1="200" x2="365.0" y2="280" stroke="#fbbf24"
        stroke-width="1" stroke-dasharray="2 3" opacity="0.55"/>

  <!-- caption: rotation -->
  <text x="80" y="345" fill="#9da0ab" font-size="12">
    סובב את כל החתך בזווית u סביב ציר z ⇒ הטורוס המלא.
  </text>
</svg>
)SVG");
}

// ─────────────────────────────────────────────────────────────
// HELIX TUBE
// ─────────────────────────────────────────────────────────────
QString helixText() {
    return QStringLiteral(R"HTML(
<h3 style="color:#c084fc;">3D Helix tube — איך מגיעים לנוסחה</h3>
<p>בנייה בשני שלבים:</p>
<p><b>שלב 1.</b> עקומת המרכז — ספירלה שמסתובבת ובו זמנית עולה:</p>
<pre style="color:#e6e6ea;">c(u) = (R·cos u,  R·sin u,  p·u)</pre>
<p>הקואורדינטות x, y מציירות מעגל ברדיוס <i>R</i>, ו-z עולה לינארית עם
קצב <i>p</i>. זה כמו בורג.</p>
<p><b>שלב 2.</b> בכל נקודה <code>c(u)</code> מצמידים מסגרת מקומית של
שלושה וקטורים ניצבים: כיוון התנועה <b style="color:#fbbf24;">T</b>
ושני הניצבים <b style="color:#22d3ee;">N</b> ו-<b style="color:#e879f9;">B</b>.
המשטח הוא צינור ברדיוס <i>tube</i> סביב c(u) — מעגל קטן במישור N–B:</p>
<pre style="color:#e6e6ea;">S(u, v) = c(u) + tube · (cos v · N(u) + sin v · B(u))</pre>
<p>הזווית <i>v</i> אומרת איפה על חתך הצינור אנחנו, בדיוק כמו ב-Torus.</p>
)HTML");
}

QByteArray helixSvg() {
    return QByteArray(R"SVG(
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 360" width="100%">
  <text x="260" y="24" fill="#cdd0d8" font-size="15" text-anchor="middle"
        font-family="sans-serif">Helix tube — מסגרת (T,N,B) שזזה לאורך הספירלה</text>

  <!-- ground / xy plane -->
  <ellipse cx="260" cy="280" rx="220" ry="48" fill="#1f2027"
           stroke="#5a5c66" stroke-width="1" stroke-dasharray="3 3"/>

  <!-- helix center curve (3 loops, sketched) -->
  <path d="M 60 290
           Q 110 240 160 290
           Q 210 340 260 290
           Q 310 240 360 290
           Q 410 340 460 290"
        fill="none" stroke="#9da0ab" stroke-width="1" stroke-dasharray="2 3"/>
  <!-- a 3D-looking helix, shifted up gradually -->
  <path d="M 60 270
           C 100 230 160 230 200 250
           C 240 270 260 240 300 210
           C 340 180 380 180 420 200
           C 440 210 460 200 470 190"
        fill="none" stroke="#cdd0d8" stroke-width="1.6"/>

  <!-- pick a point c(u) on the curve at (300,210) -->
  <circle cx="300" cy="210" r="3.5" fill="#cdd0d8"/>
  <text x="306" y="208" fill="#cdd0d8" font-size="13"
        font-style="italic">c(u)</text>

  <!-- tangent T (along curve direction) -->
  <line x1="300" y1="210" x2="356" y2="190" stroke="#fbbf24"
        stroke-width="2.2"/>
  <polygon points="356,190 348,182 348,196" fill="#fbbf24"/>
  <text x="360" y="184" fill="#fbbf24" font-size="14" font-weight="bold">T</text>
  <text x="312" y="176" fill="#fbbf24" font-size="11">(כיוון התנועה)</text>

  <!-- normal N (perpendicular, in plane of curve curvature) -->
  <line x1="300" y1="210" x2="262" y2="172" stroke="#22d3ee"
        stroke-width="2.2"/>
  <polygon points="262,172 270,178 256,180" fill="#22d3ee"/>
  <text x="240" y="166" fill="#22d3ee" font-size="14" font-weight="bold">N</text>

  <!-- binormal B (perpendicular to both) -->
  <line x1="300" y1="210" x2="316" y2="258" stroke="#e879f9"
        stroke-width="2.2"/>
  <polygon points="316,258 308,250 322,250" fill="#e879f9"/>
  <text x="320" y="266" fill="#e879f9" font-size="14" font-weight="bold">B</text>

  <!-- the cross-section circle (small, in N-B plane, perspective ellipse) -->
  <ellipse cx="300" cy="210" rx="18" ry="36" fill="none"
           stroke="#c084fc" stroke-width="1.6"
           transform="rotate(40 300 210)"/>

  <!-- a sample point on the cross section -->
  <!-- v ≈ 30° : along cos v * N + sin v * B from center  -->
  <!-- N direction: (-0.625, -0.78), B direction: (0.32, 0.96) (normalized-ish, sketchy) -->
  <!-- We'll just place a point visually -->
  <circle cx="282" cy="194" r="3.5" fill="#fbbf24"/>
  <text x="262" y="188" fill="#fbbf24" font-size="12" font-weight="bold">S(u,v)</text>

  <!-- v angle arc inside cross section -->
  <path d="M 290 218 A 14 14 0 0 1 286 200"
        fill="none" stroke="#e879f9" stroke-width="1.8"/>
  <text x="282" y="226" fill="#e879f9" font-size="13"
        font-style="italic" font-weight="bold">v</text>

  <!-- legend -->
  <g transform="translate(30,310)">
    <text x="0" y="0" fill="#9da0ab" font-size="12">u — לאורך הספירלה</text>
    <text x="170" y="0" fill="#9da0ab" font-size="12">v — סביב חתך הצינור</text>
    <text x="0" y="18" fill="#cdd0d8" font-size="11">המסגרת (T, N, B) מסתובבת יחד עם c(u) כך ש-T תמיד משיק.</text>
  </g>
</svg>
)SVG");
}

// ─────────────────────────────────────────────────────────────
// CONICAL SPIRAL
// ─────────────────────────────────────────────────────────────
QString conicalText() {
    return QStringLiteral(R"HTML(
<h3 style="color:#c084fc;">Conical spiral — איך מגיעים לנוסחה</h3>
<p>אותו רעיון כמו Helix, אבל ה<b>רדיוס מתכווץ</b> תוך כדי טיפוס:</p>
<pre style="color:#e6e6ea;">radius(u) = 1 − u/U          (U = הזווית הכוללת)
c(u) = (radius(u)·cos u,  radius(u)·sin u,  α·u)
S(u,v) = c(u) + tube(u) · (cos v · N + sin v · B)</pre>
<p>גם רדיוס הצינור עצמו הולך וקטן עם <i>u</i> — לכן המראה של קליפת חילזון.</p>
)HTML");
}

QByteArray conicalSvg() {
    return QByteArray(R"SVG(
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 320" width="100%">
  <text x="260" y="24" fill="#cdd0d8" font-size="15" text-anchor="middle"
        font-family="sans-serif">Conical spiral — רדיוס שמתכווץ עם הגובה</text>

  <!-- z-axis -->
  <line x1="80" y1="290" x2="80" y2="50" stroke="#cdd0d8" stroke-width="1.2"/>
  <text x="64" y="58" fill="#cdd0d8" font-size="13" font-style="italic">z</text>
  <!-- ground -->
  <line x1="60" y1="290" x2="500" y2="290" stroke="#5a5c66" stroke-width="1"/>

  <!-- spiral (top view collapsed in z): 3 loops shrinking -->
  <path d="M 80 270
           C 200 220 350 240 320 200
           C 290 165 180 175 200 145
           C 220 120 280 130 270 110
           C 263 95  225 100 230 85"
        fill="none" stroke="#22d3ee" stroke-width="2"/>

  <!-- tick markers showing radius shrinking -->
  <line x1="80" y1="270" x2="350" y2="270"
        stroke="#fbbf24" stroke-width="1.2" stroke-dasharray="3 3"/>
  <text x="170" y="286" fill="#fbbf24" font-size="12">radius גדול (u קטן)</text>

  <line x1="80" y1="110" x2="270" y2="110"
        stroke="#fbbf24" stroke-width="1.2" stroke-dasharray="3 3"/>
  <text x="160" y="104" fill="#fbbf24" font-size="12">radius קטן (u גדול)</text>

  <!-- height arrow -->
  <line x1="40" y1="270" x2="40" y2="85" stroke="#e879f9" stroke-width="1.6"/>
  <polygon points="40,85 36,93 44,93" fill="#e879f9"/>
  <text x="14" y="180" fill="#e879f9" font-size="12">α·u</text>
</svg>
)SVG");
}

// ─────────────────────────────────────────────────────────────
// MOBIUS
// ─────────────────────────────────────────────────────────────
QString mobiusText() {
    return QStringLiteral(R"HTML(
<h3 style="color:#c084fc;">Möbius strip — איך מגיעים לנוסחה</h3>
<p>קחו רצועת נייר, סובבו את הקצה ב-180°, הדביקו לקצה השני. זה משטח עם
<b>צד אחד</b>. נסחו זאת כפרמטריזציה:</p>
<ul>
  <li><b style="color:#22d3ee;">u</b> — היכן אנחנו לאורך הטבעת,
      0 ≤ u ≤ 2π.</li>
  <li><b style="color:#e879f9;">v</b> — היכן אנחנו לרוחב הרצועה,
      −0.4 ≤ v ≤ 0.4.</li>
</ul>
<p>נקודת הביקורת על הטבעת המרכזית היא <code>(cos u, sin u, 0)</code>.
המפתח: בכל סיבוב סביב הטבעת, הרצועה <b>מסתובבת בחצי-זווית u/2</b> סביב
הציר המשיק לטבעת. אחרי סיבוב מלא (u: 0→2π) הסיבוב הוא π — כלומר הרצועה
התהפכה. וקטור הרוחב:</p>
<pre style="color:#e6e6ea;">w(u) = cos(u/2) · ê_radial  +  sin(u/2) · ẑ</pre>
<p>נקודה כללית על המשטח:</p>
<pre style="color:#e6e6ea;">x = (1 + v·cos(u/2)) · cos u
y = (1 + v·cos(u/2)) · sin u
z =  v·sin(u/2)</pre>
)HTML");
}

QByteArray mobiusSvg() {
    return QByteArray(R"SVG(
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 320" width="100%">
  <text x="260" y="24" fill="#cdd0d8" font-size="15" text-anchor="middle"
        font-family="sans-serif">Möbius — הרצועה מסתובבת ב-u/2 סביב הטבעת</text>

  <!-- main ring (top view) -->
  <ellipse cx="260" cy="170" rx="180" ry="60" fill="none"
           stroke="#9da0ab" stroke-width="1.5"/>
  <ellipse cx="260" cy="170" rx="180" ry="60" fill="none"
           stroke="#5a5c66" stroke-width="1" stroke-dasharray="2 4"/>

  <!-- short tick marks (each is the strip's local "up" direction)
       At angle u around ring, tilt = u/2.
       We place them at u = 0, π/2, π, 3π/2, 2π. -->
  <!-- u=0   : right side, tilt 0 (vertical) -->
  <line x1="440" y1="170" x2="440" y2="140" stroke="#e879f9" stroke-width="2.6"/>
  <text x="448" y="148" fill="#e879f9" font-size="12">u=0</text>
  <!-- u=π/2 : top center, tilt 45°  -->
  <line x1="260" y1="110" x2="282" y2="88" stroke="#e879f9" stroke-width="2.6"/>
  <text x="270" y="80" fill="#e879f9" font-size="12">u=π/2</text>
  <!-- u=π   : left side, tilt 90° (horizontal!) -->
  <line x1="80" y1="170" x2="50" y2="170" stroke="#e879f9" stroke-width="2.6"/>
  <text x="20" y="186" fill="#e879f9" font-size="12">u=π</text>
  <!-- u=3π/2: bottom center, tilt 135° -->
  <line x1="260" y1="230" x2="282" y2="252" stroke="#e879f9" stroke-width="2.6"/>
  <text x="270" y="270" fill="#e879f9" font-size="12">u=3π/2</text>
  <!-- u=2π = back to start, tilt 180° = flipped -->
  <line x1="440" y1="170" x2="440" y2="200" stroke="#fbbf24" stroke-width="2.6"/>
  <text x="378" y="216" fill="#fbbf24" font-size="12">u=2π — התהפך!</text>

  <!-- caption -->
  <text x="40" y="304" fill="#9da0ab" font-size="12">
    כל סימן ורוד הוא חץ "מעלה" של הרצועה. הזווית שלו = u/2.
  </text>
</svg>
)SVG");
}

// ─────────────────────────────────────────────────────────────
// KLEIN
// ─────────────────────────────────────────────────────────────
QString kleinText() {
    return QStringLiteral(R"HTML(
<h3 style="color:#c084fc;">Klein bottle — איך מגיעים לנוסחה</h3>
<p>בקבוק קליין הוא משטח <b>סגור</b>, <b>חד-צדדי</b>, ו<b>בלי שפה</b>.
אי אפשר לטבע אותו ב-R³ בלי חיתוך-עצמי, אז משתמשים ב<b>אימוס</b>: ציור
שיש בו חיתוך עצמי אבל "נכון" טופולוגית.</p>
<p>בנייה במחשבה: לוקחים מלבן ומדביקים שני זוגות צלעות:</p>
<ul>
  <li>צלע שמאל אל צלע ימין <b>ישר</b> (כמו צילינדר).</li>
  <li>צלע למעלה אל צלע למטה <b>הפוך</b> (כמו רצועת מביוס).</li>
</ul>
<p>הציר u מטייל לאורך הצינור (הסגירה הצילינדרית), והציר v לרוחב החתך.
החיתוך לשני חלקים בנוסחה הוא הסגירה ההפוכה — הצינור חוזר אל עצמו אחרי
שעבר "דרך" עצמו.</p>
<pre style="color:#e6e6ea;">For 0 ≤ u < π :
   x = 3·cos u·(1+sin u) + r·cos u·cos v
   y = 8·sin u           + r·sin u·cos v
   z =                     r·sin v
For π ≤ u < 2π :
   x = 3·cos u·(1+sin u) + r·cos(v+π)
   y = 8·sin u
   z =                     r·sin v</pre>
)HTML");
}

QByteArray kleinSvg() {
    return QByteArray(R"SVG(
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 320" width="100%">
  <text x="260" y="24" fill="#cdd0d8" font-size="15" text-anchor="middle"
        font-family="sans-serif">Klein — מלבן עם זוג הדבקות, אחת ישרה ואחת הפוכה</text>

  <!-- left: rectangle with identification arrows -->
  <g transform="translate(40,60)">
    <rect x="0" y="0" width="180" height="120" fill="none" stroke="#cdd0d8" stroke-width="1.5"/>
    <!-- top edge: left arrow (will glue to bottom in opposite direction) -->
    <polygon points="80,-2 95,4 80,10 75,4" fill="#22d3ee"/>
    <polygon points="100,-2 115,4 100,10 95,4" fill="#22d3ee"/>
    <!-- bottom edge: same direction marks but glued FLIPPED -->
    <polygon points="80,118 65,124 80,130 85,124" fill="#22d3ee"/>
    <polygon points="100,118 85,124 100,130 105,124" fill="#22d3ee"/>
    <!-- left edge: single arrows (straight glue to right) -->
    <polygon points="-2,40 4,30 10,40 4,50" fill="#e879f9"/>
    <polygon points="-2,80 4,70 10,80 4,90" fill="#e879f9"/>
    <!-- right edge: same direction (straight glue) -->
    <polygon points="178,40 184,30 190,40 184,50" fill="#e879f9"/>
    <polygon points="178,80 184,70 190,80 184,90" fill="#e879f9"/>
    <!-- labels -->
    <text x="-22" y="64" fill="#e879f9" font-size="13">v</text>
    <text x="86" y="-12" fill="#22d3ee" font-size="13">u</text>
    <text x="0" y="150" fill="#cdd0d8" font-size="11">ורוד: הדבק ישר → צילינדר</text>
    <text x="0" y="168" fill="#cdd0d8" font-size="11">תכלת: הדבק הפוך → קליין</text>
  </g>

  <!-- right: 3D immersed rendering schematic -->
  <g transform="translate(280,40)">
    <!-- bulb -->
    <ellipse cx="100" cy="120" rx="80" ry="60" fill="none" stroke="#22d3ee" stroke-width="1.6"/>
    <!-- handle re-entering -->
    <path d="M 100 60
             C 30  20  -30 110  10 160
             C 50  210  130 180  130 165"
          fill="none" stroke="#e879f9" stroke-width="1.8"/>
    <!-- mark of self-intersection -->
    <circle cx="60" cy="140" r="4" fill="#fbbf24"/>
    <text x="68" y="138" fill="#fbbf24" font-size="11">חיתוך-עצמי</text>
    <text x="-30" y="220" fill="#cdd0d8" font-size="11">השרוול חוזר אל הבועה</text>
    <text x="-30" y="236" fill="#cdd0d8" font-size="11">— מבחוץ ויוצא מבפנים</text>
  </g>
</svg>
)SVG");
}

// ─────────────────────────────────────────────────────────────
// SADDLE
// ─────────────────────────────────────────────────────────────
QString saddleText() {
    return QStringLiteral(R"HTML(
<h3 style="color:#c084fc;">Saddle (z = u·v) — איך מגיעים לנוסחה</h3>
<p>זה המקרה הפשוט ביותר: <b>גרף</b> של פונקציה. הפרמטרים <i>u</i>, <i>v</i>
הם פשוט קואורדינטות <i>x</i>, <i>y</i>, והגובה הוא המכפלה ביניהם:</p>
<pre style="color:#e6e6ea;">S(u, v) = (u, v, u·v)</pre>
<p>למה זה אוכף? לאורך האלכסון <code>u = v</code> מקבלים
<code>z = u²</code> — קעור כלפי <b>מעלה</b> (קערה). לאורך האלכסון הנגדי
<code>u = −v</code> מקבלים <code>z = −u²</code> — קעור כלפי <b>מטה</b>
(כיפה הפוכה). אוכף = שני קימורים מנוגדים בכיוונים ניצבים.</p>
)HTML");
}

QByteArray saddleSvg() {
    return QByteArray(R"SVG(
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 300" width="100%">
  <text x="260" y="24" fill="#cdd0d8" font-size="15" text-anchor="middle"
        font-family="sans-serif">Saddle — שני אלכסונים, שני סוגי קימור</text>

  <!-- axes -->
  <line x1="60" y1="160" x2="460" y2="160" stroke="#9da0ab" stroke-width="1"/>
  <text x="466" y="164" fill="#9da0ab" font-size="13" font-style="italic">u</text>
  <line x1="260" y1="50" x2="260" y2="270" stroke="#9da0ab" stroke-width="1"/>
  <text x="248" y="48" fill="#9da0ab" font-size="13" font-style="italic">z</text>

  <!-- positive parabola: along u=v, z = u² (bowl up) -->
  <path d="M 80 240 Q 260 -40 440 240" fill="none"
        stroke="#22d3ee" stroke-width="2.4"/>
  <text x="80" y="260" fill="#22d3ee" font-size="13">u = v: z = u² ↑</text>

  <!-- negative parabola: along u=-v, z = -u² (bowl down) -->
  <path d="M 80 80 Q 260 360 440 80" fill="none"
        stroke="#e879f9" stroke-width="2.4"/>
  <text x="80" y="78" fill="#e879f9" font-size="13">u = −v: z = −u² ↓</text>

  <!-- saddle point at origin -->
  <circle cx="260" cy="160" r="5" fill="#fbbf24"/>
  <text x="270" y="184" fill="#fbbf24" font-size="12">נקודת אוכף</text>
</svg>
)SVG");
}

// ─────────────────────────────────────────────────────────────
// Dispatch
// ─────────────────────────────────────────────────────────────
struct Entry { QString text; QByteArray svg; };

Entry entryFor(const QString& name) {
    if (name == QStringLiteral("Sphere"))
        return { sphereText(),  sphereSvg()  };
    if (name == QStringLiteral("Torus"))
        return { torusText(),   torusSvg()   };
    if (name == QStringLiteral("3D Helix (tube)"))
        return { helixText(),   helixSvg()   };
    if (name == QStringLiteral("Conical spiral"))
        return { conicalText(), conicalSvg() };
    if (name == QStringLiteral("Mobius strip"))
        return { mobiusText(),  mobiusSvg()  };
    if (name == QStringLiteral("Klein bottle"))
        return { kleinText(),   kleinSvg()   };
    if (name == QStringLiteral("Saddle (z = u*v)"))
        return { saddleText(),  saddleSvg()  };
    return {};
}

} // namespace

QString    derivationTextFor(const QString& name) { return entryFor(name).text; }
QByteArray derivationSvgFor (const QString& name) { return entryFor(name).svg;  }
