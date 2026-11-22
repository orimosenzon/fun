// Minimal 4x4 matrix math for GL (column-major float[16]). Right-handed, y-up world.
#pragma once
#include <cmath>
#include <array>
#include "../sim/vmath.hpp"

namespace fohn {

using Mat4 = std::array<float, 16>;

inline Mat4 mat4Identity() { return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; }

// column-major multiply: returns a*b
inline Mat4 mat4Mul(const Mat4& a, const Mat4& b) {
  Mat4 r{};
  for (int c = 0; c < 4; c++)
    for (int row = 0; row < 4; row++) {
      float s = 0;
      for (int k = 0; k < 4; k++) s += a[k * 4 + row] * b[c * 4 + k];
      r[c * 4 + row] = s;
    }
  return r;
}

inline Mat4 mat4Perspective(float fovy, float aspect, float znear, float zfar) {
  float f = 1.0f / std::tan(fovy / 2);
  Mat4 m{};
  m[0] = f / aspect; m[5] = f;
  m[10] = (zfar + znear) / (znear - zfar);
  m[11] = -1;
  m[14] = (2 * zfar * znear) / (znear - zfar);
  return m;
}

// orthographic box (the camera of the shadow map)
inline Mat4 mat4Ortho(float l, float r, float b, float t, float znear, float zfar) {
  Mat4 m = mat4Identity();
  m[0] = 2 / (r - l); m[5] = 2 / (t - b); m[10] = -2 / (zfar - znear);
  m[12] = -(r + l) / (r - l); m[13] = -(t + b) / (t - b); m[14] = -(zfar + znear) / (zfar - znear);
  return m;
}

inline Mat4 mat4LookAt(const V3& eye, const V3& center, const V3& up) {
  V3 f = norm(sub(center, eye));
  V3 s = norm(cross(f, up));
  V3 u = cross(s, f);
  Mat4 m = mat4Identity();
  m[0] = (float)s[0]; m[4] = (float)s[1]; m[8] = (float)s[2];
  m[1] = (float)u[0]; m[5] = (float)u[1]; m[9] = (float)u[2];
  m[2] = (float)-f[0]; m[6] = (float)-f[1]; m[10] = (float)-f[2];
  m[12] = (float)-dot(s, eye);
  m[13] = (float)-dot(u, eye);
  m[14] = (float)dot(f, eye);
  return m;
}

inline Mat4 mat4Translate(const V3& t) {
  Mat4 m = mat4Identity();
  m[12] = (float)t[0]; m[13] = (float)t[1]; m[14] = (float)t[2];
  return m;
}

// model matrix for a body posed at p with orientation q (body->world), optional per-axis scale
inline Mat4 mat4FromPose(const V4& q, const V3& p, const V3& scl = {1, 1, 1}) {
  V3 ex = qrot(q, {1, 0, 0}), ey = qrot(q, {0, 1, 0}), ez = qrot(q, {0, 0, 1});
  Mat4 m{};
  m[0] = (float)(ex[0] * scl[0]); m[1] = (float)(ex[1] * scl[0]); m[2] = (float)(ex[2] * scl[0]); m[3] = 0;
  m[4] = (float)(ey[0] * scl[1]); m[5] = (float)(ey[1] * scl[1]); m[6] = (float)(ey[2] * scl[1]); m[7] = 0;
  m[8] = (float)(ez[0] * scl[2]); m[9] = (float)(ez[1] * scl[2]); m[10] = (float)(ez[2] * scl[2]); m[11] = 0;
  m[12] = (float)p[0]; m[13] = (float)p[1]; m[14] = (float)p[2]; m[15] = 1;
  return m;
}

// a colour written as 0xRRGGBB in sRGB (as every colour of the web build is), as linear light
inline V3 srgbHex(unsigned h) {
  auto lin = [](unsigned v) { double c = v / 255.0; return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4); };
  return {lin((h >> 16) & 255), lin((h >> 8) & 255), lin(h & 255)};
}

} // namespace fohn
