#include "DiagramScene.h"

#include <QtMath>
#include <cmath>

namespace diag {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Color palette — matches the Phong colors used in SurfaceView/SVG diagrams.
const QColor cAxis    (205, 208, 216);
const QColor cFaint   ( 90,  92, 102);
const QColor cU       (232, 121, 249);   // pink for u and cos u
const QColor cSinU    (251, 191,  36);   // amber for sin u / R
const QColor cV       ( 34, 211, 238);   // cyan for v
const QColor cT       (251, 191,  36);
const QColor cN       ( 34, 211, 238);
const QColor cB       (232, 121, 249);
const QColor cPoint   (232, 121, 249);
const QColor cBowlUp  ( 34, 211, 238);
const QColor cBowlDn  (232, 121, 249);
const QColor cPlane   ( 50,  60,  80);

QVector3D sph(float u, float v) {
    return { std::sin(u)*std::cos(v),
             std::sin(u)*std::sin(v),
             std::cos(u) };
}

// ─────────────────────────────────────────────────────────────
// SPHERE
// ─────────────────────────────────────────────────────────────
Scene sphereScene() {
    Scene s;
    s.title = "Sphere — שתי זוויות מגדירות נקודה";

    const float u = 0.95f;       // ~54°
    const float v = 0.9f;        // ~52°
    const QVector3D O(0,0,0);
    const QVector3D P = sph(u, v);
    const QVector3D Q(P.x(), P.y(), 0);
    const QVector3D Zhat(0,0,1);
    const QVector3D Xhat(1,0,0);

    // xy plane
    s.planes.push_back({O, QVector3D(2.5f,0,0), QVector3D(0,2.5f,0),
                        QColor(40, 50, 70, 80)});

    // sphere outlines
    s.circles.push_back({O, QVector3D(1,0,0),  QVector3D(0,1,0),  cFaint, 1.0f, true});  // equator
    s.circles.push_back({O, QVector3D(1,0,0),  QVector3D(0,0,1),  cFaint, 1.0f, false}); // meridian xz
    s.circles.push_back({O, QVector3D(0,1,0),  QVector3D(0,0,1),  cFaint, 1.0f, false}); // meridian yz

    // axes
    s.segments.push_back({O - 1.4f*Xhat, O + 1.6f*Xhat, cAxis, 1.4f, false, "x", 1.62f*Xhat});
    s.segments.push_back({O - 1.4f*QVector3D(0,1,0), O + 1.6f*QVector3D(0,1,0), cAxis, 1.4f, false, "y", 1.62f*QVector3D(0,1,0)});
    s.segments.push_back({O - 1.4f*Zhat, O + 1.6f*Zhat, cAxis, 1.4f, false, "+z", 1.65f*Zhat});

    // OP — radius 1
    s.segments.push_back({O, P, cAxis, 2.0f, false, "1", 0.55f*P});

    // cos u — vertical from Q to P
    s.segments.push_back({Q, P, cU, 2.5f, false, "cos u", 0.5f*(Q+P) + QVector3D(0.08f,0.08f,0)});

    // sin u — horizontal from O to Q
    s.segments.push_back({O, Q, cSinU, 2.5f, false, "sin u", 0.5f*Q + QVector3D(0,0,-0.08f)});

    // angle u: at O, between +z and OP
    s.arcs.push_back({O, Zhat, P.normalized(), 0.45f, cU, "u"});

    // angle v: at O, between +x and OQ (in xy-plane)
    s.arcs.push_back({O, Xhat, Q.normalized(), 0.4f, cV, "v"});

    // point P, Q
    s.points.push_back({P, cPoint, 8.0f, "P"});
    s.points.push_back({Q, cSinU, 6.0f, "Q"});
    s.points.push_back({O, cAxis, 5.0f, "O"});
    s.points.push_back({Zhat, cFaint, 4.0f, ""});  // north pole tick

    s.boundsRadius = 2.6f;
    return s;
}

// ─────────────────────────────────────────────────────────────
// TORUS
// ─────────────────────────────────────────────────────────────
Scene torusScene() {
    Scene s;
    s.title = "Torus — חתך מאונך וסיבוב סביב ציר z";

    const float R = 1.2f;
    const float r = 0.45f;
    const float v = 0.9f;       // ~52°
    const QVector3D O(0,0,0);
    const QVector3D Zhat(0,0,1);
    const QVector3D Xhat(1,0,0);

    // xy plane
    s.planes.push_back({O, QVector3D(2.5f,0,0), QVector3D(0,2.5f,0),
                        QColor(40, 50, 70, 70)});

    // big circle on xy-plane (radius R) — the path traced by C as u varies
    s.circles.push_back({O, QVector3D(R,0,0), QVector3D(0,R,0), cFaint, 1.2f, true});

    // small circle (the cross-section) at azimuth u=0, in xz plane shifted to (R,0,0)
    QVector3D C(R, 0, 0);
    s.circles.push_back({C, QVector3D(r,0,0), QVector3D(0,0,r), cAxis, 1.4f, false});

    // axes
    s.segments.push_back({O, O + 2.0f*Zhat, cAxis, 1.4f, false, "z", 2.1f*Zhat});
    s.segments.push_back({O, O + 2.2f*Xhat, cAxis, 1.0f, false, "ρ", 2.3f*Xhat});

    // R: from O to C
    s.segments.push_back({O, C, cSinU, 2.5f, false, "R", 0.5f*C + QVector3D(0,0,0.1f)});

    // point P on tube at angle v
    QVector3D P(R + r*std::cos(v), 0, r*std::sin(v));
    // r: from C to P
    s.segments.push_back({C, P, cN, 2.5f, false, "r", 0.5f*(C+P) + QVector3D(0.08f,0.08f,0)});

    // angle v at C: from +x direction to (P-C) direction
    s.arcs.push_back({C, Xhat, (P-C).normalized(), 0.18f, cU, "v"});

    // horizontal projection: R + r cos v (drawn from O to (P.x, 0, 0))
    QVector3D Px(P.x(), 0, 0);
    s.segments.push_back({O, Px, cSinU, 1.6f, true, "R+r·cos v", 0.5f*Px + QVector3D(0,0,-0.18f)});

    // vertical: r sin v (from Px up to P)
    s.segments.push_back({Px, P, cN, 1.6f, true, "r·sin v", 0.5f*(Px+P) + QVector3D(0.05f,0.05f,0)});

    s.points.push_back({O, cAxis, 5.0f, "O"});
    s.points.push_back({C, cAxis, 5.0f, "C"});
    s.points.push_back({P, cPoint, 7.0f, "P"});

    s.boundsRadius = 2.6f;
    return s;
}

// ─────────────────────────────────────────────────────────────
// HELIX TUBE
// ─────────────────────────────────────────────────────────────
Scene helixScene() {
    Scene s;
    s.title = "Helix tube — מסגרת (T,N,B) שנעה לאורך הספירלה";

    const float R = 1.0f;
    const float p = 0.35f;
    const QVector3D O(0,0,0);

    // axes
    s.segments.push_back({O, QVector3D(0,0,3.0f), cAxis, 1.2f, false, "z", QVector3D(0,0,3.1f)});
    s.segments.push_back({O, QVector3D(1.6f,0,0), cAxis, 1.0f, false, "x", QVector3D(1.7f,0,0)});
    s.segments.push_back({O, QVector3D(0,1.6f,0), cAxis, 1.0f, false, "y", QVector3D(0,1.7f,0)});

    // helix center curve as a polyline of segments
    const int N = 160;
    const float uMax = 4.0f * 2.0f * kPi;
    QVector<QVector3D> pts;
    pts.reserve(N+1);
    for (int i = 0; i <= N; ++i) {
        float t = uMax * i / N;
        pts.push_back({R*std::cos(t), R*std::sin(t), p*t * 0.4f - 1.0f});
    }
    for (int i = 0; i < N; ++i) {
        s.segments.push_back({pts[i], pts[i+1], cFaint, 1.4f, false, "", {}});
    }

    // pick a point c(u) at u₀ = 6.5
    const float u0 = 6.5f;
    QVector3D c(R*std::cos(u0), R*std::sin(u0), p*u0*0.4f - 1.0f);
    QVector3D T(-R*std::sin(u0), R*std::cos(u0), p*0.4f);
    T.normalize();
    QVector3D upZ(0,0,1);
    QVector3D Nhat = upZ - QVector3D::dotProduct(upZ, T)*T;
    Nhat.normalize();
    QVector3D Bhat = QVector3D::crossProduct(T, Nhat);

    // T, N, B vectors
    s.segments.push_back({c, c + 0.7f*T, cT, 2.6f, false, "T", c + 0.78f*T});
    s.segments.push_back({c, c + 0.6f*Nhat, cN, 2.6f, false, "N", c + 0.68f*Nhat});
    s.segments.push_back({c, c + 0.6f*Bhat, cB, 2.6f, false, "B", c + 0.68f*Bhat});

    // tube cross-section circle in N–B plane
    s.circles.push_back({c, 0.18f*Nhat, 0.18f*Bhat, QColor(192,132,252), 1.6f, false});

    // a sample point S(u,v) on the tube cross-section, v=π/4
    const float vAng = kPi/4.0f;
    QVector3D S = c + 0.18f*(std::cos(vAng)*Nhat + std::sin(vAng)*Bhat);
    s.segments.push_back({c, S, cSinU, 1.4f, false, "", {}});
    s.arcs.push_back({c, Nhat, (S-c).normalized(), 0.10f, cU, "v"});

    s.points.push_back({c, cAxis, 6.0f, "c(u)"});
    s.points.push_back({S, cPoint, 6.0f, "S(u,v)"});

    s.boundsRadius = 2.6f;
    return s;
}

// ─────────────────────────────────────────────────────────────
// CONICAL SPIRAL
// ─────────────────────────────────────────────────────────────
Scene conicalScene() {
    Scene s;
    s.title = "Conical spiral — רדיוס מתכווץ עם הגובה";

    const QVector3D O(0,0,0);
    s.segments.push_back({O, QVector3D(0,0,2.8f), cAxis, 1.2f, false, "z", QVector3D(0,0,2.9f)});
    s.segments.push_back({O, QVector3D(1.6f,0,0), cAxis, 1.0f, false, "x", QVector3D(1.7f,0,0)});
    s.segments.push_back({O, QVector3D(0,1.6f,0), cAxis, 1.0f, false, "y", QVector3D(0,1.7f,0)});

    // spiral
    const int N = 240;
    const float uMax = 3.0f * 2.0f * kPi;
    QVector<QVector3D> pts;
    pts.reserve(N+1);
    for (int i = 0; i <= N; ++i) {
        float t = uMax * i / N;
        float radius = 1.4f * (1.0f - t / uMax * 0.95f);
        pts.push_back({radius*std::cos(t), radius*std::sin(t), 0.18f*t * 0.5f});
    }
    for (int i = 0; i < N; ++i) {
        s.segments.push_back({pts[i], pts[i+1], cV, 1.6f, false, "", {}});
    }

    // mark big-radius and small-radius circles
    s.circles.push_back({O, QVector3D(1.4f,0,0), QVector3D(0,1.4f,0), cSinU, 1.0f, true});
    s.circles.push_back({QVector3D(0,0,uMax*0.18f*0.5f),
                        QVector3D(0.07f,0,0), QVector3D(0,0.07f,0), cSinU, 1.0f, true});

    s.points.push_back({pts.first(), cSinU, 6.0f, "u קטן"});
    s.points.push_back({pts.last(),  cSinU, 6.0f, "u גדול"});

    s.boundsRadius = 2.8f;
    return s;
}

// ─────────────────────────────────────────────────────────────
// MOBIUS
// ─────────────────────────────────────────────────────────────
Scene mobiusScene() {
    Scene s;
    s.title = "Möbius — וקטור הרוחב מסתובב ב-u/2";

    const QVector3D O(0,0,0);
    s.segments.push_back({O, QVector3D(2.0f,0,0), cAxis, 1.0f, false, "x", QVector3D(2.1f,0,0)});
    s.segments.push_back({O, QVector3D(0,2.0f,0), cAxis, 1.0f, false, "y", QVector3D(0,2.1f,0)});
    s.segments.push_back({O, QVector3D(0,0,1.4f), cAxis, 1.0f, false, "z", QVector3D(0,0,1.5f)});

    // central ring (the strip's centerline)
    s.circles.push_back({O, QVector3D(1,0,0), QVector3D(0,1,0), cFaint, 1.4f, true});

    // tick marks at 5 sample u values, each tilted by u/2
    const int K = 8;
    for (int i = 0; i < K; ++i) {
        float u = 2.0f * kPi * i / K;
        float h = 0.5f * u;
        QVector3D base(std::cos(u), std::sin(u), 0);
        // strip tangent radial direction:
        QVector3D radial = base.normalized();
        // strip's local "up" rotates from radial (at u=0) toward z (at u=π) and back
        QVector3D upLocal = std::cos(h)*radial + std::sin(h)*QVector3D(0,0,1);
        QColor col = cU;
        if (i == 0)  col = QColor(34, 211, 238);   // start mark in cyan
        if (i == K-1 || (i==0)) {} // keep
        s.segments.push_back({base - 0.18f*upLocal, base + 0.18f*upLocal,
                              col, 2.4f, false, "", {}});
    }
    // explicit "after one loop" tick at u=2π → flipped
    {
        float u = 2.0f * kPi;
        float h = 0.5f * u;
        QVector3D base(std::cos(u), std::sin(u), 0);
        QVector3D radial = base.normalized();
        QVector3D upLocal = std::cos(h)*radial + std::sin(h)*QVector3D(0,0,1);
        s.segments.push_back({base - 0.20f*upLocal, base + 0.20f*upLocal,
                              cSinU, 3.0f, false, "u=2π (התהפך)", base + 0.36f*upLocal});
        s.points.push_back({base, cSinU, 6.0f, ""});
    }
    {
        QVector3D base(1, 0, 0);
        s.points.push_back({base, cV, 6.0f, "u=0"});
    }

    s.boundsRadius = 2.4f;
    return s;
}

// ─────────────────────────────────────────────────────────────
// KLEIN
// ─────────────────────────────────────────────────────────────
Scene kleinScene() {
    Scene s;
    s.title = "Klein — הדבקות של מלבן (אחת ישרה, אחת הפוכה)";

    // Draw a rectangle in 3D + arrow markings indicating glue directions.
    // Then a separate, smaller schematic of the immersed bottle to its right.
    const QVector3D O(0,0,0);

    // Rectangle in xy plane, centred at origin
    QVector3D A(-1.2f, -0.8f, 0);
    QVector3D B( 1.2f, -0.8f, 0);
    QVector3D C( 1.2f,  0.8f, 0);
    QVector3D D(-1.2f,  0.8f, 0);

    s.segments.push_back({A, B, cAxis, 1.6f});
    s.segments.push_back({B, C, cU,    2.0f});  // right edge — pink
    s.segments.push_back({C, D, cAxis, 1.6f});
    s.segments.push_back({D, A, cU,    2.0f});  // left edge — pink (same dir)

    // Top edge with arrows pointing right (glued to bottom flipped)
    s.segments.push_back({D, C, cV, 2.6f, false, "u (top→ , flipped)",  0.5f*(D+C)+QVector3D(0,0.18f,0)});
    s.segments.push_back({A, B, cV, 2.6f, false, "u (bottom→)", 0.5f*(A+B)+QVector3D(0,-0.22f,0)});

    // Mark left/right edge as identified straight (cylinder-like)
    s.points.push_back({0.5f*(A+D), cU, 5.0f, "v"});
    s.points.push_back({0.5f*(B+C), cU, 5.0f, "v"});

    // Mini schematic of the bottle (offset)
    QVector3D Q(2.7f, 0, 0);
    s.circles.push_back({Q, QVector3D(0.6f,0,0), QVector3D(0,0.6f,0), cV, 1.6f, false});  // bulb
    s.circles.push_back({Q + QVector3D(0,0,0.5f),
                         QVector3D(0.5f,0,0), QVector3D(0,0,0.5f), cU, 1.6f, false});    // handle

    s.points.push_back({Q + QVector3D(0,0,0.2f), cSinU, 5.0f, "חיתוך-עצמי"});

    s.boundsRadius = 3.5f;
    return s;
}

// ─────────────────────────────────────────────────────────────
// SADDLE
// ─────────────────────────────────────────────────────────────
Scene saddleScene() {
    Scene s;
    s.title = "Saddle z = u·v — שני אלכסונים, שני קימורים";

    const QVector3D O(0,0,0);
    s.segments.push_back({O, QVector3D(1.6f,0,0), cAxis, 1.0f, false, "u (=x)", QVector3D(1.7f,0,0)});
    s.segments.push_back({O, QVector3D(0,1.6f,0), cAxis, 1.0f, false, "v (=y)", QVector3D(0,1.7f,0)});
    s.segments.push_back({O, QVector3D(0,0,1.4f), cAxis, 1.0f, false, "z",      QVector3D(0,0,1.5f)});

    // Curve along u=v (z = u^2): polyline
    const int N = 60;
    QVector3D prev;
    for (int i = 0; i <= N; ++i) {
        float t = -1.0f + 2.0f*i/N;
        QVector3D q(t, t, t*t);
        if (i > 0) s.segments.push_back({prev, q, cBowlUp, 2.4f, false, "", {}});
        prev = q;
    }

    // Curve along u = -v (z = -u^2)
    for (int i = 0; i <= N; ++i) {
        float t = -1.0f + 2.0f*i/N;
        QVector3D q(t, -t, -t*t);
        if (i > 0) s.segments.push_back({prev, q, cBowlDn, 2.4f, false, "", {}});
        prev = q;
    }

    // u and v axis lines on z=0 (already drawn) — and labels for the bowls
    s.points.push_back({QVector3D(1, 1, 1),  cBowlUp, 6.0f, "u=v: z=u² ↑"});
    s.points.push_back({QVector3D(1,-1,-1),  cBowlDn, 6.0f, "u=−v: z=−u² ↓"});
    s.points.push_back({O, cSinU, 6.0f, "saddle point"});

    s.boundsRadius = 2.6f;
    return s;
}

}  // namespace

Scene sceneFor(const QString& name) {
    if (name == QStringLiteral("Sphere"))            return sphereScene();
    if (name == QStringLiteral("Torus"))             return torusScene();
    if (name == QStringLiteral("3D Helix (tube)"))   return helixScene();
    if (name == QStringLiteral("Conical spiral"))    return conicalScene();
    if (name == QStringLiteral("Mobius strip"))      return mobiusScene();
    if (name == QStringLiteral("Klein bottle"))      return kleinScene();
    if (name == QStringLiteral("Saddle (z = u*v)"))  return saddleScene();
    return {};
}

}  // namespace diag
