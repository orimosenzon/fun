#include "bike_mesh.hpp"
#include "mat4.hpp"
#include <array>
#include <cmath>

namespace fohn {

// ---------------- materials (those of bikeModel.js, number for number) ----------------
// colour (linear), metalness, roughness, roughness of the clear coat (0 = no coat)
struct Mat { V3 c; float metal, rough, coat; };
static V3 hex(unsigned h) { return srgbHex(h); }
static const float COAT0 = 0.0525f;      // a coat of roughness 0: the least three.js lets a surface have
static const Mat PAINT = {hex(0xc6cbd3), 0.55f, 0.3f, 0.06f}, BLUE = {hex(0x1b3a86), 0.3f, 0.3f, 0.08f};
static const Mat CAN = {hex(0xb9bdc4), 1.0f, 0.24f, 0}, ACCENT = {hex(0xe8500a), 0.1f, 0.35f, 0.1f};
static const Mat CARBON = {hex(0x17181b), 0.4f, 0.45f, 0}, STEEL = {hex(0x8d9299), 0.9f, 0.3f, 0};
static const Mat GOLD = {hex(0xb08a3c), 1.0f, 0.3f, 0}, HOT = {hex(0x4a4650), 0.95f, 0.4f, 0};
static const Mat DARK = {hex(0x0b0c0e), 0.3f, 0.7f, 0}, RUBBER = {hex(0x0c0c0d), 0, 0.9f, 0};
static const Mat SUIT = {hex(0x25272b), 0, 0.7f, 0}, SUIT2 = {hex(0xe8500a), 0, 0.7f, 0};
static const Mat HELMET = {hex(0x1b1c20), 0, 0.25f, COAT0}, VISOR = {hex(0x0a0a0c), 1.0f, 0.08f, 0};
static const Mat GLASS = {hex(0x1c2630), 0, 0.05f, 0};
static const Mat EMIT = {{0, 0, 0}, 0, 1, 0};        // what only shines: its light comes from its glow slot

// ---------------- affine transforms ----------------
struct Aff {
  M3 R{1, 0, 0, 0, 1, 0, 0, 0, 1};
  V3 t{0, 0, 0};
  V3 pt(const V3& p) const { return add(m3v(R, p), t); }
  V3 dir(const V3& d) const { return m3v(R, d); }
};
static M3 m3mul(const M3& a, const M3& b) {
  M3 r{};
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  return r;
}
static Aff operator*(const Aff& a, const Aff& b) { Aff r; r.R = m3mul(a.R, b.R); r.t = add(m3v(a.R, b.t), a.t); return r; }
static Aff T(double x, double y, double z) { Aff a; a.t = {x, y, z}; return a; }
static Aff rotZ(double ang) { Aff a; double c = std::cos(ang), s = std::sin(ang); a.R = {c, -s, 0, s, c, 0, 0, 0, 1}; return a; }
static const Aff I0 = Aff{};

// ---------------- mesh builder ----------------
struct MB {
  std::vector<float>* v;
  void vert(const V3& p, const V3& n, const Mat& m, float g) {
    v->push_back((float)p[0]); v->push_back((float)p[1]); v->push_back((float)p[2]);
    v->push_back((float)n[0]); v->push_back((float)n[1]); v->push_back((float)n[2]);
    v->push_back((float)m.c[0]); v->push_back((float)m.c[1]); v->push_back((float)m.c[2]);
    v->push_back(g);
    v->push_back(m.metal); v->push_back(m.rough); v->push_back(m.coat);
  }
  void tri(const V3& a, const V3& b, const V3& c, const V3& na, const V3& nb, const V3& nc, const Mat& col, float g = -1) {
    vert(a, na, col, g); vert(b, nb, col, g); vert(c, nc, col, g);
  }
  void flat(const V3& a, const V3& b, const V3& c, const Mat& col, float g = -1) {
    V3 n = norm(cross(sub(b, a), sub(c, a)));
    tri(a, b, c, n, n, n, col, g);
  }
  void quad(const V3& a, const V3& b, const V3& c, const V3& d, const Mat& col, float g = -1) { flat(a, b, c, col, g); flat(a, c, d, col, g); }

  void box(const Aff& X, const V3& c, const V3& s, const Mat& col, float g = -1) {
    V3 h = scale(s, 0.5);
    auto P = [&](double sx, double sy, double sz) { return X.pt({c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]}); };
    quad(P(1, -1, -1), P(1, 1, -1), P(1, 1, 1), P(1, -1, 1), col, g);
    quad(P(-1, -1, 1), P(-1, 1, 1), P(-1, 1, -1), P(-1, -1, -1), col, g);
    quad(P(-1, 1, -1), P(-1, 1, 1), P(1, 1, 1), P(1, 1, -1), col, g);
    quad(P(-1, -1, 1), P(-1, -1, -1), P(1, -1, -1), P(1, -1, 1), col, g);
    quad(P(-1, -1, 1), P(1, -1, 1), P(1, 1, 1), P(-1, 1, 1), col, g);
    quad(P(1, -1, -1), P(-1, -1, -1), P(-1, 1, -1), P(1, 1, -1), col, g);
  }

  // cylinder / cone frustum from a (radius ra) to b (radius rb), in the frame X
  void cyl(const Aff& X, const V3& a0, const V3& b0, double ra, double rb, const Mat& col, int seg = 16, bool caps = true, float g = -1) {
    V3 a = X.pt(a0), b = X.pt(b0);
    V3 ax = sub(b, a); double L = len(ax); if (L < 1e-9) return;
    V3 k = scale(ax, 1 / L);
    V3 ref = std::fabs(k[1]) < 0.9 ? V3{0, 1, 0} : V3{1, 0, 0};
    V3 u = norm(cross(k, ref)), w = cross(k, u);
    double slope = (ra - rb) / L;
    for (int i = 0; i < seg; i++) {
      double t0 = 2 * PI * i / seg, t1 = 2 * PI * (i + 1) / seg;
      V3 d0 = add(scale(u, std::cos(t0)), scale(w, std::sin(t0))), d1 = add(scale(u, std::cos(t1)), scale(w, std::sin(t1)));
      V3 n0 = norm(add(d0, scale(k, slope))), n1 = norm(add(d1, scale(k, slope)));
      V3 A0 = add(a, scale(d0, ra)), A1 = add(a, scale(d1, ra)), B0 = add(b, scale(d0, rb)), B1 = add(b, scale(d1, rb));
      tri(A0, B0, B1, n0, n0, n1, col, g); tri(A0, B1, A1, n0, n1, n1, col, g);
      if (caps) {
        V3 nk = scale(k, -1);
        tri(a, A1, A0, nk, nk, nk, col, g);
        tri(b, B0, B1, k, k, k, col, g);
      }
    }
  }
  // a flat disc (or annulus r0..r1) centred at c with normal n
  void disc(const Aff& X, const V3& c0, const V3& n0, double r1, const Mat& col, float g = -1, double r0 = 0, int seg = 20) {
    V3 c = X.pt(c0), n = norm(X.dir(n0));
    V3 ref = std::fabs(n[1]) < 0.9 ? V3{0, 1, 0} : V3{1, 0, 0};
    V3 u = norm(cross(n, ref)), w = cross(n, u);
    for (int i = 0; i < seg; i++) {
      double t0 = 2 * PI * i / seg, t1 = 2 * PI * (i + 1) / seg;
      V3 d0 = add(scale(u, std::cos(t0)), scale(w, std::sin(t0))), d1 = add(scale(u, std::cos(t1)), scale(w, std::sin(t1)));
      if (r0 <= 0) tri(c, add(c, scale(d0, r1)), add(c, scale(d1, r1)), n, n, n, col, g);
      else {
        V3 a = add(c, scale(d0, r0)), b = add(c, scale(d0, r1)), cc = add(c, scale(d1, r1)), d = add(c, scale(d1, r0));
        tri(a, b, cc, n, n, n, col, g); tri(a, cc, d, n, n, n, col, g);
      }
    }
  }
  // ellipsoid centred at c with radii r, in the frame X
  void ell(const Aff& X, const V3& c, const V3& r, const Mat& col, float g = -1, int seg = 16, int rings = 10) {
    auto P = [&](int i, int j, V3& pos, V3& nrm) {
      double th = PI * j / rings, ph = 2 * PI * i / seg;
      V3 s = {std::sin(th) * std::cos(ph), std::cos(th), std::sin(th) * std::sin(ph)};
      pos = X.pt({c[0] + s[0] * r[0], c[1] + s[1] * r[1], c[2] + s[2] * r[2]});
      nrm = norm(X.dir({s[0] / r[0], s[1] / r[1], s[2] / r[2]}));
    };
    for (int j = 0; j < rings; j++)
      for (int i = 0; i < seg; i++) {
        V3 p00, p10, p01, p11, n00, n10, n01, n11;
        P(i, j, p00, n00); P(i + 1, j, p10, n10); P(i, j + 1, p01, n01); P(i + 1, j + 1, p11, n11);
        tri(p00, p01, p11, n00, n01, n11, col, g); tri(p00, p11, p10, n00, n11, n10, col, g);
      }
  }
  // a beam (or a limb, capsule = rounded ends) between two points
  void beam(const Aff& X, const V3& a, const V3& b, double r, const Mat& col, bool capsule = false) {
    cyl(X, a, b, r, r, col, 10, !capsule);
    if (capsule) { ell(X, a, {r, r, r}, col, -1, 10, 6); ell(X, b, {r, r, r}, col, -1, 10, 6); }
  }
  // bodywork: a side profile (x,y) extruded across the bike, width w centred at z0, as three.js
  // extrudes it with a bevel: the two flat sides keep the profile as drawn, and between them the
  // panel swells out by bev in a quarter round of three flat strips. (So every panel is bev larger
  // all round than its drawing, and its edges catch the light.)
  void prof(const std::vector<std::array<double, 2>>& pts, double w, double z0, const Mat& col, double bev = 0.02) {
    int n = (int)pts.size();
    double area = 0;
    for (int i = 0; i < n; i++) { auto& p = pts[i]; auto& q = pts[(i + 1) % n]; area += p[0] * q[1] - q[0] * p[1]; }
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = area > 0 ? i : n - 1 - i;    // counter-clockwise
    double zf = z0 + w / 2, zb = z0 - w / 2;
    // caps by ear clipping
    std::vector<int> poly = idx;
    auto cr = [&](int a, int b, int c) { return (pts[b][0] - pts[a][0]) * (pts[c][1] - pts[a][1]) - (pts[b][1] - pts[a][1]) * (pts[c][0] - pts[a][0]); };
    auto inside = [&](int a, int b, int c, int p) { return cr(a, b, p) >= 0 && cr(b, c, p) >= 0 && cr(c, a, p) >= 0; };
    int guard = 0;
    while (poly.size() > 2 && guard++ < 200) {
      int m = (int)poly.size(); bool cut = false;
      for (int i = 0; i < m; i++) {
        int a = poly[(i + m - 1) % m], b = poly[i], c = poly[(i + 1) % m];
        if (cr(a, b, c) <= 1e-12) continue;
        bool ear = true;
        for (int k : poly) if (k != a && k != b && k != c && inside(a, b, c, k)) { ear = false; break; }
        if (!ear) continue;
        V3 A = {pts[a][0], pts[a][1], zf}, B = {pts[b][0], pts[b][1], zf}, C = {pts[c][0], pts[c][1], zf};
        tri(A, B, C, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, col);
        V3 A2 = {pts[a][0], pts[a][1], zb}, B2 = {pts[b][0], pts[b][1], zb}, C2 = {pts[c][0], pts[c][1], zb};
        tri(A2, C2, B2, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, col);
        poly.erase(poly.begin() + i); cut = true; break;
      }
      if (!cut) break;
    }
    // which way each corner moves as the panel swells: where its two edges, pushed out by one,
    // meet again (but no further than root 2: sharp tips are not drawn out into spikes)
    std::vector<std::array<double, 2>> mv(n);
    for (int i = 0; i < n; i++) {
      auto& a = pts[idx[(i + n - 1) % n]]; auto& p = pts[idx[i]]; auto& c = pts[idx[(i + 1) % n]];
      double e0x = p[0] - a[0], e0y = p[1] - a[1], e1x = c[0] - p[0], e1y = c[1] - p[1];
      double l0 = std::hypot(e0x, e0y), l1 = std::hypot(e1x, e1y);
      double n0x = e0y / l0, n0y = -e0x / l0, n1x = e1y / l1, n1y = -e1x / l1, dd = 1 + n0x * n1x + n0y * n1y;
      double mx = dd > 1e-9 ? (n0x + n1x) / dd : n0x, my = dd > 1e-9 ? (n0y + n1y) / dd : n0y, l2 = mx * mx + my * my;
      if (l2 > 2) { double k = std::sqrt(2 / l2); mx *= k; my *= k; }
      mv[i] = {mx, my};
    }
    const int BS = 3; const double hw = std::max(0.001, w / 2 - bev);
    auto ring = [&](int k, int i, double sgn) -> V3 {
      double t = k / (double)BS, bs = bev * std::sin(t * PI / 2), dz = bev * std::cos(t * PI / 2);
      return {pts[idx[i]][0] + mv[i][0] * bs, pts[idx[i]][1] + mv[i][1] * bs, z0 + sgn * (hw + dz)};
    };
    for (int i = 0; i < n; i++) {
      int j = (i + 1) % n;
      for (int k = 0; k < BS; k++) {
        quad(ring(k, i, 1), ring(k, j, 1), ring(k + 1, j, 1), ring(k + 1, i, 1), col);
        quad(ring(k + 1, i, -1), ring(k + 1, j, -1), ring(k, j, -1), ring(k, i, -1), col);
      }
      quad(ring(BS, i, -1), ring(BS, j, -1), ring(BS, j, 1), ring(BS, i, 1), col);      // the side wall
    }
  }
};

static V3 v3(double x, double y, double z) { return {x, y, z}; }

BikeMesh buildBikeMesh() {
  BikeMesh M;
  MB d{&M.body};
  const double EY = ENG.axisY, HUBY = 0.34, FX = LIFT[0].p[0], RX = LIFT[2].p[0];
  const double RY = REAR.p[1], RXN = REAR.p[0] + 0.25;
  const V3 IN = {INTAKE[0], INTAKE[1], INTAKE[2]};

  // ----- the fusion core: a drum between the rider's knees -----
  d.cyl(I0, v3(0.02, EY, -0.15), v3(0.02, EY, 0.15), 0.2, 0.2, CARBON, 28);
  for (int s : {-1, 1}) {
    d.cyl(I0, v3(0.02, EY, s * 0.165 - 0.0175), v3(0.02, EY, s * 0.165 + 0.0175), 0.17, 0.17, STEEL, 28);
    d.cyl(I0, v3(0.02, EY, s * 0.19 - 0.015), v3(0.02, EY, s * 0.19 + 0.015), 0.075, 0.075, GOLD, 20);
    for (int k = 0; k < 10; k++) {
      double a = k / 10.0 * 2 * PI, bx = 0.02 + std::cos(a) * 0.135, by = EY + std::sin(a) * 0.135;
      d.cyl(I0, v3(bx, by, s * 0.185 - 0.01), v3(bx, by, s * 0.185 + 0.01), 0.012, 0.012, STEEL, 6);
    }
    d.disc(I0, v3(0.02, EY, s * 0.206), v3(0, 0, s), 0.128, EMIT, 0, 0.102, 28);    // the window into the core
  }
  for (int k = -2; k <= 2; k++) d.cyl(I0, v3(0.02, EY, k * 0.05 - 0.006), v3(0.02, EY, k * 0.05 + 0.006), 0.213, 0.213, STEEL, 28);
  // compressor, fed by the intake in the nose
  d.cyl(T(0.4, EY + 0.08, 0) * rotZ(0.15), v3(0.2, 0, 0), v3(-0.2, 0, 0), 0.1, 0.14, STEEL, 20);
  const double ix = IN[0] - 0.2, iy = IN[1] + 0.03;
  d.cyl(I0, v3(ix + 0.1, iy, 0), v3(ix - 0.1, iy, 0), 0.105, 0.1, BLUE, 24, false);
  d.cyl(I0, v3(ix + 0.1, iy, 0), v3(ix - 0.1, iy, 0), 0.096, 0.09, DARK, 24, false);
  d.disc(I0, v3(ix - 0.02, iy, 0), v3(1, 0, 0), 0.07, STEEL);                       // fan face
  d.cyl(I0, v3(ix + 0.05, iy, 0), v3(ix - 0.02, iy, 0), 0.002, 0.029, ACCENT, 12);   // spinner
  // main engine: jet pipe slung under the tail, a polished can, then the nozzle
  d.beam(I0, v3(-0.16, EY + 0.02, 0), v3(-0.46, RY, 0), 0.07, HOT);
  d.cyl(I0, v3(-0.44, RY, 0), v3(-0.60, RY, 0), 0.07, 0.165, HOT, 28);
  d.cyl(I0, v3(-0.60, RY, 0), v3(RXN, RY, 0), 0.165, 0.165, CAN, 32, false);
  for (double x : {-0.62, (RXN - 0.6) / 2, RXN - 0.03})
    d.cyl(I0, v3(x + 0.015, RY, 0), v3(x - 0.015, RY, 0), 0.172, 0.172, x == RXN - 0.03 ? DARK : STEEL, 32);
  for (int s : {-1, 1}) d.beam(I0, v3(-0.62, 0.93, s * 0.07), v3(-0.7, RY + 0.15, s * 0.07), 0.014, STEEL);
  d.cyl(I0, v3(RXN, RY, 0), v3(RXN - 0.25, RY, 0), 0.15, 0.118, DARK, 24, false);   // the liner you look into
  d.cyl(I0, v3(RXN, RY, 0), v3(RXN - 0.26, RY, 0), 0.172, 0.122, HOT, 18, false);   // the ring of petals
  d.disc(I0, v3(RXN - 0.01, RY, 0), v3(-1, 0, 0), 0.14, EMIT, 5, 0, 24);            // the flame
  d.beam(I0, v3(0.2, 0.42, 0), v3(FX - 0.05, HUBY + 0.02, 0), 0.045, HOT);         // ducts to the two hubs
  d.beam(I0, v3(-0.14, 0.42, 0), v3(RX + 0.05, HUBY + 0.02, 0), 0.045, HOT);

  // ----- frame, fork and swing arm -----
  for (int s : {-1, 1}) {
    d.beam(I0, v3(0.30, 0.88, s * 0.13), v3(-0.2, 0.62, s * 0.15), 0.026, STEEL);
    d.beam(I0, v3(-0.2, 0.62, s * 0.15), v3(-0.62, 0.9, s * 0.09), 0.018, STEEL);
    d.beam(I0, v3(0.52, 1.03, s * 0.11), v3(FX, HUBY, s * 0.11), 0.026, GOLD);
    d.beam(I0, v3(0.61, 0.76, s * 0.11), v3(FX, HUBY, s * 0.11), 0.033, STEEL);
    d.beam(I0, v3(-0.2, 0.56, s * 0.13), v3(RX, HUBY, s * 0.13), 0.035, STEEL);
  }
  d.cyl(I0, v3(0.52, 1.03, -0.15), v3(0.52, 1.03, 0.15), 0.035, 0.035, STEEL, 12);
  d.beam(I0, v3(-0.3, 0.8, 0), v3(-0.52, 0.5, 0), 0.022, GOLD);
  for (double x : {FX, RX}) d.cyl(I0, v3(x, HUBY, -0.2), v3(x, HUBY, 0.2), 0.04, 0.04, STEEL, 16);
  // where the wheels were: a lift engine at each end of each axle
  for (int i = 0; i < 4; i++) {
    double gx = LIFT[i].p[0], gz = LIFT[i].p[2];
    d.cyl(I0, v3(gx, HUBY + 0.19, gz), v3(gx, HUBY - 0.06, gz), 0.094, 0.094, CAN, 20);
    d.ell(I0, v3(gx, HUBY + 0.19, gz), v3(0.094, 0.042, 0.094), CAN, -1, 20, 6);
    d.cyl(I0, v3(gx, HUBY + 0.161, gz), v3(gx, HUBY + 0.139, gz), 0.099, 0.099, ACCENT, 20);
    d.cyl(I0, v3(gx, HUBY - 0.034, gz), v3(gx, HUBY - 0.056, gz), 0.099, 0.099, STEEL, 20);
    d.cyl(I0, v3(gx, HUBY - 0.0505, gz), v3(gx, HUBY - 0.1255, gz), 0.1, 0.084, DARK, 20, false);
    d.disc(I0, v3(gx, HUBY - 0.12, gz), v3(0, -1, 0), 0.076, EMIT, (float)(1 + i), 0, 18);
  }
  // side nozzles: a short stub each side of the nose and of the jet pipe, blowing straight out
  for (int pair = 0; pair < 2; pair++)
    for (int sgn : {-1, 1}) {
      V3 p = {SIDE[pair].p[0], SIDE[pair].p[1], sgn * SIDE[pair].p[2]}, o = {0, 0, (double)sgn};
      d.cyl(I0, sub(p, scale(o, 0.045)), add(p, scale(o, 0.045)), 0.05, 0.04, HOT, 14);
      d.cyl(I0, add(p, scale(o, 0.037)), add(p, scale(o, 0.053)), 0.047, 0.047, STEEL, 14);
      d.disc(I0, add(p, scale(o, 0.055)), o, 0.036, EMIT, (float)(6 + pair * 2 + (sgn > 0 ? 1 : 0)), 0, 14);
    }

  // ----- bodywork -----
  d.prof({{0.99, 0.93}, {0.94, 1.04}, {0.74, 1.15}, {0.46, 1.09}, {0.42, 0.92}, {0.62, 0.85}, {0.88, 0.85}}, 0.34, 0, PAINT, 0.04);   // nose fairing
  d.prof({{0.96, 0.9}, {0.88, 0.87}, {0.64, 0.87}, {0.5, 0.93}, {0.9, 0.94}}, 0.36, 0, ACCENT, 0.015);                                     // its stripe
  for (int s : {-1, 1}) {
    d.prof({{0.66, 0.88}, {0.44, 0.96}, {0.26, 0.9}, {0.2, 0.74}, {0.3, 0.6}, {0.24, 0.47}, {0.5, 0.45}, {0.64, 0.56}, {0.72, 0.72}}, 0.03, s * 0.17, PAINT, 0.012);
    d.prof({{0.64, 0.56}, {0.5, 0.45}, {0.24, 0.47}, {-0.12, 0.38}, {-0.2, 0.33}, {0.5, 0.31}, {0.66, 0.42}}, 0.032, s * 0.165, BLUE, 0.012);
    d.prof({{0.6, 0.6}, {0.34, 0.53}, {0.3, 0.49}, {0.56, 0.52}}, 0.034, s * 0.172, ACCENT, 0.008);
  }
  d.prof({{0.54, 0.4}, {0.5, 0.31}, {-0.22, 0.31}, {-0.3, 0.38}, {-0.2, 0.41}}, 0.3, 0, BLUE, 0.03);                                  // belly pan
  { MB gl{&M.glass}; gl.ell(T(0.66, 1.17, 0) * rotZ(-0.75), v3(0, 0, 0), v3(0.2, 0.13, 0.145), GLASS, -1, 20, 14); }                  // screen (drawn after the rest, see-through)
  d.prof({{0.44, 1.05}, {0.3, 1.12}, {0.06, 1.1}, {-0.1, 0.99}, {-0.14, 0.9}, {0.42, 0.9}}, 0.3, 0, PAINT, 0.05);                       // tank
  d.prof({{0.36, 1.1}, {0.3, 1.135}, {0.08, 1.115}, {-0.04, 1.04}}, 0.1, 0, ACCENT, 0.012);
  d.prof({{-0.1, 0.96}, {-0.14, 0.89}, {-0.56, 0.9}, {-0.58, 0.97}, {-0.34, 0.94}}, 0.25, 0, RUBBER, 0.03);                             // seat
  d.prof({{-0.5, 0.9}, {-0.56, 0.99}, {-1.04, 1.2}, {-1.07, 1.15}, {-0.76, 0.97}}, 0.2, 0, PAINT, 0.035);                               // tail
  d.prof({{-0.62, 1.02}, {-1.02, 1.205}, {-1.04, 1.18}, {-0.7, 1.0}}, 0.21, 0, BLUE, 0.01);
  for (int s : {-1, 1}) d.ell(I0, v3(0.975, 0.97, s * 0.075), v3(0.02, 0.03, 0.05), EMIT, 10);                                // twin head lamps
  // bars, mirrors, pegs
  for (int s : {-1, 1}) {
    d.beam(I0, v3(0.52, 1.04, s * 0.1), v3(0.47, 1.04, s * 0.34), 0.014, STEEL);
    d.beam(I0, v3(0.47, 1.04, s * 0.24), v3(0.47, 1.04, s * 0.35), 0.021, RUBBER);
    d.beam(I0, v3(0.6, 1.08, s * 0.15), v3(0.68, 1.2, s * 0.28), 0.008, CARBON);
    d.ell(I0, v3(0.68, 1.21, s * 0.29), v3(0.012, 0.03, 0.05), CARBON);
    d.beam(I0, v3(-0.2, 0.6, s * 0.14), v3(-0.16, 0.46, s * 0.23), 0.012, STEEL);
    d.cyl(I0, v3(-0.16, 0.46, s * 0.26 - 0.05), v3(-0.16, 0.46, s * 0.26 + 0.05), 0.014, 0.014, STEEL, 8);
    d.box(I0, v3(-0.71, 0.95, s * 0.115), v3(0.3, 0.1, 0.012), CARBON);                                                         // air brake petals (closed)
  }
  d.box(T(0.58, 1.1, 0) * rotZ(-0.6), v3(0, 0, 0), v3(0.02, 0.08, 0.16), EMIT, 15);                                             // dash
  // lights: red to port, green to starboard, the tail lamp and the strobe
  d.ell(I0, v3(0.47, 1.04, -0.36), v3(0.014, 0.014, 0.014), EMIT, 11, 8, 6);
  d.ell(I0, v3(0.47, 1.04, 0.36), v3(0.014, 0.014, 0.014), EMIT, 12, 8, 6);
  d.ell(I0, v3(-1.07, 1.17, 0), v3(0.03, 0.03, 0.03), EMIT, 13, 8, 6);
  d.ell(I0, v3(-0.96, 1.2, 0), v3(0.014, 0.014, 0.014), EMIT, 14, 8, 6);

  // ----- rider: knees on the tank, chest over it -----
  Aff torso = T(-0.34, 1.02, 0) * rotZ(-0.95);
  d.ell(torso, v3(0, 0.3, 0), v3(0.132, 0.335, 0.178), SUIT);
  d.box(torso, v3(-0.13, 0.36, 0), v3(0.02, 0.3, 0.2), SUIT2);
  Aff head = torso * T(0.02, 0.66, 0) * rotZ(0.85);
  d.ell(head, v3(0, 0.06, 0), v3(0.135, 0.14, 0.125), HELMET);
  d.ell(head, v3(0.078, 0.065, 0), v3(0.075, 0.055, 0.108), VISOR);
  d.box(head, v3(-0.02, 0.2, 0), v3(0.2, 0.025, 0.02), ACCENT);
  for (int s : {-1, 1}) {
    d.beam(I0, v3(-0.34, 1.0, s * 0.13), v3(0.12, 0.87, s * 0.22), 0.075, SUIT, true);    // thigh
    d.beam(I0, v3(0.12, 0.87, s * 0.22), v3(-0.15, 0.54, s * 0.25), 0.058, SUIT, true);   // shin
    d.beam(I0, v3(-0.24, 0.5, s * 0.26), v3(-0.04, 0.48, s * 0.26), 0.045, RUBBER, true); // boot on the peg
    d.beam(I0, v3(0.06, 1.3, s * 0.2), v3(0.28, 1.15, s * 0.31), 0.052, SUIT, true);      // upper arm
    d.beam(I0, v3(0.28, 1.15, s * 0.31), v3(0.46, 1.07, s * 0.3), 0.044, SUIT, true);     // forearm
    d.ell(I0, v3(0.28, 1.15, s * 0.31), v3(0.058, 0.058, 0.058), SUIT2);                 // elbow guard
    d.ell(I0, v3(0.12, 0.87, s * 0.22), v3(0.078, 0.078, 0.078), SUIT2);                 // knee guard
  }

  // ----- the stand it rests on (shown only near the ground) -----
  MB st{&M.stand};
  for (int s : {-1, 1})
    for (int k : {0, 2}) {
      double x = SKIDS[k][0], out = std::fabs(SKIDS[k][2]) - 0.1;
      V3 g = {x * 0.7, 0.36, s * 0.1}, f = {g[0] + x * 0.3, g[1] - 0.34, s * out};
      st.beam(I0, g, f, 0.017, STEEL);
      st.box(I0, {f[0], g[1] - 0.348, f[2]}, {0.16, 0.02, 0.06}, RUBBER);
    }
  return M;
}

void bikeGlow(const Bike& b, float out[BIKE_GLOW_SLOTS * 3]) {
  const double K = 1.0;    // light, in the units of the scene: the tone mapping at the end of every shader brings it to the screen
  double heat = std::max(0.0, b.eng.N - 0.25) / 0.75, h2 = heat * heat, od = b.abOut;
  double flick = 0.92 + 0.08 * std::sin(b.t * 47) * std::sin(b.t * 13.1);
  auto set = [&](int i, double r, double g, double bl) { out[i * 3] = (float)(r * K); out[i * 3 + 1] = (float)(g * K); out[i * 3 + 2] = (float)(bl * K); };
  set(0, (0.25 + 1.6 * h2 + 5 * od) * flick, (0.9 + 4.2 * h2 + 4 * od) * flick, (1.4 + 6.5 * h2 + 6 * od) * flick);
  for (int i = 0; i < 4; i++) { double f = std::min(1.0, b.L[i] / 1600); set(1 + i, h2 * 1.2 * f, h2 * 1.9 * f, h2 * 3.2 * f); }
  double fr = std::min(1.0, b.R / 5000);
  set(5, h2 * 1.2 * fr + od * 7, h2 * 1.9 * fr + od * 6, h2 * 3.2 * fr + od * 11);
  for (int pair = 0; pair < 2; pair++)
    for (int side = 0; side < 2; side++) {
      double sgn = side ? 1 : -1;
      double f = std::min(1.0, std::max(0.0, -sgn * b.sd[pair]) / (0.6 * SIDE_MAX)), k = 1 + 5 * b.kick;
      set(6 + pair * 2 + side, 1.2 * f * k, 1.9 * f * k, 3.2 * f * k);
    }
  // the lamps: the tail lamp brightens with the air brakes, the strobe flashes twice a second
  set(10, 6, 5.4, 4.2);
  set(11, 8, 0.2, 0.1); set(12, 0.2, 8, 0.4);
  set(13, b.brake > 0.3 ? 16 : 3, 0.3, 0.2);
  double blink = std::fmod(b.t * 1.1, 1.0), st = blink < 0.06 || (blink > 0.14 && blink < 0.2) ? 14 : 0.05;
  set(14, st, st, st);
  V3 dash = hex(0x0a2a33); set(15, dash[0], dash[1], dash[2]);
}

} // namespace fohn
