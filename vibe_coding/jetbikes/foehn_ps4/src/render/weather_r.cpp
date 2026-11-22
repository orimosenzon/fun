#include "weather_r.hpp"
#include "env.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace fohn {

static const std::string WRAP = R"(
uniform vec3 uOff, uBox, cameraPosition;
vec3 wrapped(vec3 seed) { return mod(seed * uBox + uOff - cameraPosition, uBox) - 0.5 * uBox; }   // relative to the camera
float edgeFade(vec3 p) { float d = length(p); return (1.0 - smoothstep(0.36, 0.5, d / uBox.x)) * smoothstep(0.6, 3.0, d); }
)";
static const std::string RAIN_VS = "attribute vec3 aSeed; attribute float aEnd; uniform vec3 uStreak; uniform float uAmt; uniform mat4 uVP; varying float vA;\n" + WRAP + R"(
void main() {
  vec3 p = wrapped(aSeed);
  vA = edgeFade(p) * (1.0 - 0.75 * aEnd) * step(fract(aSeed.x * 37.0 + aSeed.z * 91.0), uAmt);
  gl_Position = uVP * vec4(p + cameraPosition - uStreak * aEnd, 1.0);
}
)";
static const char* RAIN_FS = R"(
uniform vec3 uLight; varying float vA;
void main() { gl_FragColor = outColor(uLight, vA * 0.32); }
)";
static const std::string SNOW_VS = "attribute vec3 aSeed; uniform float uAmt, uTime, uPx; uniform mat4 uView, uProj; varying float vA;\n" + WRAP + R"(
void main() {
  vec3 p = wrapped(aSeed);
  float ph = aSeed.x * 40.0 + aSeed.y * 17.0;
  p.xz += 0.35 * vec2(sin(uTime * 1.3 + ph), cos(uTime * 1.1 + ph * 1.7));
  vA = edgeFade(p) * step(fract(aSeed.y * 53.0 + aSeed.z * 29.0), uAmt);
  vec4 mv = uView * vec4(p + cameraPosition, 1.0);
  gl_PointSize = clamp(uPx * (0.035 + 0.03 * fract(ph)) / max(-mv.z, 0.3), 1.5, 14.0);
  gl_Position = uProj * mv;
}
)";
static const char* SNOW_FS = R"(
uniform vec3 uLight; varying float vA;
void main() { vec2 q = gl_PointCoord * 2.0 - 1.0; float r = dot(q, q); if (r > 1.0) discard; gl_FragColor = outColor(uLight, vA * (1.0 - r) * 0.85); }
)";

void WeatherR::init() {
  uint32_t s = 99991;
  auto rnd = [&]() { s = s * 1664525u + 1013904223u; return (float)(s / 4294967296.0); };
  // rain: line segments (both ends carry the drop's seed; the far end is pulled back along the streak)
  const int NR = 9000;
  std::vector<float> rv; rv.reserve((size_t)NR * 8);
  for (int i = 0; i < NR; i++) { float x = rnd(), y = rnd(), z = rnd(); rv.insert(rv.end(), {x, y, z, 0.0f, x, y, z, 1.0f}); }
  rain_.prog.build(RAIN_VS, GLSL_COMMON + RAIN_FS);
  rain_.count = NR * 2; rain_.box[0] = 64; rain_.box[1] = 40; rain_.box[2] = 64;
  glGenBuffers(1, &rain_.vbo); glBindBuffer(GL_ARRAY_BUFFER, rain_.vbo); glBufferData(GL_ARRAY_BUFFER, rv.size() * 4, rv.data(), GL_STATIC_DRAW);
  // snow: soft points that sway as they fall
  const int NS = 20000;
  std::vector<float> sv((size_t)NS * 3);
  for (auto& x : sv) x = rnd();
  snow_.prog.build(SNOW_VS, GLSL_COMMON + SNOW_FS);
  snow_.count = NS; snow_.box[0] = 56; snow_.box[1] = 40; snow_.box[2] = 56;
  glGenBuffers(1, &snow_.vbo); glBindBuffer(GL_ARRAY_BUFFER, snow_.vbo); glBufferData(GL_ARRAY_BUFFER, sv.size() * 4, sv.data(), GL_STATIC_DRAW);
}

void WeatherR::update(double dt, const Camera& cam, const V3& wind, double shelter, int viewH) {
  const WeatherDef& w = WEATHER[gEnv.wxIdx];
  bool isRain = w.wet > 0, isSnow = w.snow > 0;
  if ((!isRain && !isSnow) || dt <= 0) { first_ = true; return; }
  V3 d = sub(cam.eye, camPrev_); camPrev_ = cam.eye;
  if (first_ || len(d) > 250 * dt + 5) d = {0, 0, 0};                    // the camera was cut to another place
  first_ = false;
  double k = std::min(1.0, dt * 12);
  for (int i = 0; i < 3; i++) camVel_[i] += (d[i] / dt - camVel_[i]) * k;
  Layer& u = isRain ? rain_ : snow_;
  double fall = isRain ? 9.5 : 1.3, carry = isRain ? 0.75 : 1;
  double v[3] = {wind[0] * carry, -fall + wind[1] * (isRain ? 0.3 : 0.8), wind[2] * carry};
  for (int i = 0; i < 3; i++) { u.off[i] = std::fmod(u.off[i] + v[i] * dt, (double)u.box[i]); }
  const double tau = 0.02;
  for (int i = 0; i < 3; i++) u.streak[i] = (float)((v[i] - camVel_[i]) * tau);
  u.amt = (float)(1 - shelter);
  u.px = (float)(viewH / (2 * std::tan(cam.fovY / 2)));
  const float* a = gEnv.ambSky; double kk = isRain ? 2.4 : 3.4;
  u.light[0] = (float)std::min(1.6, a[0] * kk + 0.05); u.light[1] = (float)std::min(1.6, a[1] * kk + 0.05); u.light[2] = (float)std::min(1.6, a[2] * kk + 0.06);
}

void WeatherR::draw(const Camera& cam) {
  const WeatherDef& w = WEATHER[gEnv.wxIdx];
  bool isRain = w.wet > 0, isSnow = w.snow > 0;
  if (!isRain && !isSnow) return;
  Layer& u = isRain ? rain_ : snow_;
  Prog& p = u.prog;
  p.use();
  gEnv.apply(p, cam);
  p.m4("uVP", cam.vp); p.m4("uView", cam.view); p.m4("uProj", cam.proj);
  p.f3("uOff", (float)u.off[0], (float)u.off[1], (float)u.off[2]);
  glUniform3fv(p.U("uBox"), 1, u.box); glUniform3fv(p.U("uStreak"), 1, u.streak); glUniform3fv(p.U("uLight"), 1, u.light);
  p.f1("uAmt", u.amt); p.f1("uPx", u.px);
  glEnable(GL_BLEND); glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE); glDepthMask(GL_FALSE);
  glBindBuffer(GL_ARRAY_BUFFER, u.vbo);
  if (isRain) {
    p.attr("aSeed", 3, GL_FLOAT, false, 16, 0); p.attr("aEnd", 1, GL_FLOAT, false, 16, 12);
    glLineWidth(1.0f);
    glDrawArrays(GL_LINES, 0, u.count);
    p.attrOff("aSeed"); p.attrOff("aEnd");
  } else {
    p.attr("aSeed", 3, GL_FLOAT, false, 12, 0);
    glDrawArrays(GL_POINTS, 0, u.count);
    p.attrOff("aSeed");
  }
  glDepthMask(GL_TRUE); glDisable(GL_BLEND);
}

} // namespace fohn
