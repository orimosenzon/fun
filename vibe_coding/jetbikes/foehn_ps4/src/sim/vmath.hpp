// Small vector / quaternion / matrix helpers. Faithful port of js/sim/vmath.js.
// Vectors are {x,y,z} (index 0,1,2), quaternions {w,x,y,z} (index 0,1,2,3).
// All double precision, matching JavaScript Number.
#pragma once
#include <array>
#include <cmath>

namespace fohn {

using V3 = std::array<double, 3>;
using V4 = std::array<double, 4>;   // quaternion [w,x,y,z]
using M3 = std::array<double, 9>;   // row-major 3x3

constexpr double PI = 3.14159265358979323846;

inline double clamp(double x, double a, double b) { return x < a ? a : (x > b ? b : x); }
inline double lerp(double a, double b, double t) { return a + (b - a) * t; }
inline double sstep(double a, double b, double x) { double t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }

inline V3 add(const V3& a, const V3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
inline V3 sub(const V3& a, const V3& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
inline V3 scale(const V3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }
inline V3 madd(const V3& a, const V3& b, double s) { return {a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s}; }
inline double dot(const V3& a, const V3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline V3 cross(const V3& a, const V3& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }
inline double len(const V3& a) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }
inline V3 norm(const V3& a) { double l = len(a); if (l == 0) l = 1; return {a[0] / l, a[1] / l, a[2] / l}; }
inline double wrapPi(double a) { a = std::fmod(a + PI, 2 * PI); if (a < 0) a += 2 * PI; return a - PI; }

inline V4 qmul(const V4& a, const V4& b) {
  return {
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  };
}
inline V4 qconj(const V4& q) { return {q[0], -q[1], -q[2], -q[3]}; }
inline V4 qnorm(const V4& q) { double l = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]); return {q[0] / l, q[1] / l, q[2] / l, q[3] / l}; }
// rotate v from body to world
inline V3 qrot(const V4& q, const V3& v) {
  double w = q[0], x = q[1], y = q[2], z = q[3];
  double tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
  return {v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx};
}
inline V3 qinvrot(const V4& q, const V3& v) { return qrot({q[0], -q[1], -q[2], -q[3]}, v); }
inline V4 qaxis(const V3& axis, double ang) { double s = std::sin(ang / 2); return {std::cos(ang / 2), axis[0] * s, axis[1] * s, axis[2] * s}; }
// exact exponential map: rotation vector (rad) -> quaternion
inline V4 qexp(const V3& r) {
  double a = len(r);
  if (a < 1e-9) return qnorm({1, r[0] / 2, r[1] / 2, r[2] / 2});
  double s = std::sin(a / 2) / a;
  return {std::cos(a / 2), r[0] * s, r[1] * s, r[2] * s};
}
// yaw (about +y, left positive), pitch (nose up positive), roll (right wing down positive)
// body axes: x forward, y up, z right
inline V4 qFromYPR(double yaw, double pitch, double roll) {
  return qmul(qmul(qaxis({0, 1, 0}, yaw), qaxis({0, 0, 1}, pitch)), qaxis({1, 0, 0}, roll));
}
inline V3 yprOf(const V4& q) {
  V3 f = qrot(q, {1, 0, 0}), r = qrot(q, {0, 0, 1});
  double yaw = std::atan2(-f[2], f[0]);
  double pitch = std::asin(clamp(f[1], -1, 1));
  V3 r0 = {std::sin(yaw), 0, std::cos(yaw)};
  V3 u0 = cross(r0, f);
  double roll = std::atan2(-dot(r, u0), dot(r, r0));
  return {yaw, pitch, roll};
}

// 3x3 matrices as flat arrays (row-major)
inline V3 m3v(const M3& m, const V3& v) {
  return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
          m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
          m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}
inline M3 m3inv(const M3& m) {
  double a = m[0], b = m[1], c = m[2], d = m[3], e = m[4], f = m[5], g = m[6], h = m[7], i = m[8];
  double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  double det = a * A + b * B + c * C;
  return {A / det, (c * h - b * i) / det, (b * f - c * e) / det,
          B / det, (a * i - c * g) / det, (c * d - a * f) / det,
          C / det, (b * g - a * h) / det, (a * e - b * d) / det};
}

// Solve A x = b for a small dense system (in place, Gauss with partial pivoting). n up to NU=12.
inline void solve(double* A, double* b, int n) {
  for (int c = 0; c < n; c++) {
    int p = c;
    for (int r = c + 1; r < n; r++) if (std::fabs(A[r * n + c]) > std::fabs(A[p * n + c])) p = r;
    if (p != c) {
      for (int k = 0; k < n; k++) { double t = A[c * n + k]; A[c * n + k] = A[p * n + k]; A[p * n + k] = t; }
      double t = b[c]; b[c] = b[p]; b[p] = t;
    }
    double d = A[c * n + c]; if (d == 0) d = 1e-12;
    for (int r = c + 1; r < n; r++) {
      double f = A[r * n + c] / d;
      if (f == 0) continue;
      for (int k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
      b[r] -= f * b[c];
    }
  }
  for (int r = n - 1; r >= 0; r--) {
    double s = b[r];
    for (int k = r + 1; k < n; k++) s -= A[r * n + k] * b[k];
    double d = A[r * n + r]; if (d == 0) d = 1e-12;
    b[r] = s / d;
  }
}

} // namespace fohn
