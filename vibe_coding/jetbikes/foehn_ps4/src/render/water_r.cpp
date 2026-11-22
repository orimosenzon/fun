#include "water_r.hpp"
#include "env.hpp"
#include "../sim/terrain.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>

namespace fohn {

// The lake is a grid on the nodes of the height field; each vertex carries the depth of the water
// there (GLES2 has no float texture to read the heights from in the fragment shader).
static const char* LAKE_VS = R"(
attribute vec3 aPos; attribute float aDepth; uniform mat4 uVP; varying vec3 vW; varying float vDepth;
void main() { vW = aPos; vDepth = aDepth; gl_Position = uVP * vec4(aPos, 1.0); }
)";
static const char* LAKE_FS = R"(
uniform sampler2D uAux, uRefl; uniform mat4 uReflMat; uniform vec4 uWorld; uniform float uLevel, uHasRefl; uniform vec4 uWash, uOutlet;
varying vec3 vW; varying float vDepth;
vec2 grad(vec2 p, float e) { return vec2(vnoise(p + vec2(e, 0.0)) - vnoise(p - vec2(e, 0.0)), vnoise(p + vec2(0.0, e)) - vnoise(p - vec2(0.0, e))); }
void main() {
  vec2 uv = (vW.xz + uWorld.x) / (2.0 * uWorld.x);
  float depth = vDepth;
  if (depth < -0.02) discard;
  if (uOutlet.z != 0.0 || uOutlet.w != 0.0) { vec2 o = vW.xz - uOutlet.xy; if (dot(o, uOutlet.zw) > 0.0 && abs(o.x * uOutlet.w - o.y * uOutlet.z) < 45.0) discard; }      // past the outlet the water belongs to the river
  if (uHole.z > 0.0 && distance(vW.xz, uHole.xy) < uHole.z) discard;
  vec4 aux = texture2D(uAux, uv);
  vec3 Vv = cameraPosition - vW; float dist = length(Vv); vec3 V = Vv / dist;
  // wind ripples at three scales, calmer far away and near the shore
  float calm = smoothstep(0.0, 2.5, depth);
  vec2 w1 = grad(vW.xz / 1.7 + uTime * vec2(0.55, 0.2), 0.6), w2 = grad(vW.xz / 6.5 - uTime * vec2(0.22, 0.35), 0.6), w3 = grad(vW.xz / 31.0 + uTime * vec2(0.1, -0.05), 0.6);
  float gust = 0.55 + 0.9 * vnoise(vW.xz / 240.0 + uTime * 0.02);
  vec2 g = (w1 * 0.55 / (1.0 + dist / 90.0) + w2 * 0.8 / (1.0 + dist / 500.0) + w3 * 0.6) * gust * (0.35 + 0.65 * calm);
  // jet wash: rings running out from under the bike
  vec2 wd = vW.xz - uWash.xy; float wr = length(wd);
  float ring = uWash.w * exp(-wr / (uWash.z + 1.0)) * sin(wr * 1.6 - uTime * 14.0);
  g += normalize(wd + 1e-4) * ring * 1.6;
  vec3 n = normalize(vec3(-g.x * 0.5, 1.0, -g.y * 0.5));
  float fres = 0.02 + 0.98 * pow(1.0 - max(dot(n, V), 0.0), 5.0);
  // reflection
  vec3 R = reflect(-V, n); R.y = abs(R.y);
  vec3 refl = skyLUT(R);
  if (uHasRefl > 0.5) {
    vec4 rc = uReflMat * vec4(vW + vec3(n.x, 0.0, n.z) * min(dist * 0.12, 26.0), 1.0);
    vec2 ruv = rc.xy / rc.w;
    if (rc.w > 0.0) { vec3 m = min(texture2D(uRefl, clamp(ruv, 0.002, 0.998)).rgb, vec3(0.985)); refl = m / (1.0 - m); }   // (the mirror was stored as c / (1 + c))
  }
  // what is below: lit lake bed fading into milky glacial turquoise
  float sunVis = aux.r * cloudShadow(vW);
  vec3 light = uSunCol * max(uSunDir.y, 0.0) * sunVis + uAmbSky;
  float caust = 0.75 + 0.5 * vnoise(vW.xz / 0.9 + g * 6.0 + uTime * 0.4);
  vec3 bed = vec3(0.34, 0.33, 0.27) * (0.6 + 0.6 * vnoise(vW.xz / 2.3)) * caust;
  float path = depth / max(V.y, 0.12);
  vec3 body = bed * exp(-path * vec3(0.42, 0.10, 0.085)) + vec3(0.015, 0.20, 0.21) * (1.0 - exp(-path * 0.20));
  vec3 col = mix(body * light, refl, fres);
  // sun glitter
  float sp = pow(max(dot(reflect(-uSunDir, n), V), 0.0), 520.0);
  col += uSunCol * sp * 28.0 * sunVis;
  // foam at the shore and in the jet wash
  float foam = (1.0 - smoothstep(0.0, 0.45, depth + 0.25 * (vnoise(vW.xz / 0.7 + uTime * 0.3) - 0.5))) * 0.75;
  foam = max(foam, clamp(uWash.w * exp(-wr / (uWash.z * 0.6 + 0.5)) * (0.4 + vnoise(vW.xz / 0.4 + uTime)) * 1.6, 0.0, 1.0));
  col = mix(col, vec3(0.85) * light, foam);
  gl_FragColor = outSolid(applyFog(col, vW));
}
)";
static const char* RIVER_VS = R"(
attribute vec3 aPos; attribute vec2 aUv; attribute float aSlope, aK; uniform mat4 uVP;
varying vec3 vW; varying vec2 vUv; varying float vSlope, vK;
void main() { vW = aPos; vUv = aUv; vSlope = aSlope; vK = aK; gl_Position = uVP * vec4(aPos, 1.0); }
)";
static const char* RIVER_FS = R"(
uniform sampler2D uAux; uniform vec4 uWorld;
varying vec3 vW; varying vec2 vUv; varying float vSlope, vK;
void main() {
  if (vW.y < uClipY) discard;
  vec2 uv = (vW.xz + uWorld.x) / (2.0 * uWorld.x);
  vec4 aux = texture2D(uAux, uv);
  float speed = 1.2 + 16.0 * min(vSlope, 0.9);
  vec2 f = vec2(vUv.x * 4.0, vUv.y * 0.6 - uTime * speed * 0.12);
  float n1 = vnoise(f * 3.0), n2 = vnoise(f * 9.0 + 4.0);
  vec3 V = normalize(cameraPosition - vW);
  vec3 n = normalize(vec3((n1 - 0.5) * 0.5, 1.0, (n2 - 0.5) * 0.5));
  float fres = 0.03 + 0.97 * pow(1.0 - max(dot(n, V), 0.0), 5.0);
  float sunVis = aux.r * cloudShadow(vW);
  vec3 light = uSunCol * max(uSunDir.y, 0.0) * sunVis + uAmbSky;
  vec3 R = reflect(-V, n); R.y = abs(R.y);
  vec3 col = mix(vec3(0.05, 0.25, 0.24) * light, skyLUT(R), fres);
  float white = smoothstep(0.035, 0.26, vSlope + (n1 * n2 - 0.25) * 0.12) + smoothstep(0.62, 0.9, n1 * 0.6 + n2 * 0.5) * 0.35;
  white = max(white, smoothstep(0.45, 0.9, vSlope) * (0.75 + 0.25 * vnoise(vec2(vUv.x * 14.0, vUv.y * 0.5 - uTime * 2.4))));     // where it falls, it is all white, in streaks
  col = mix(col, vec3(0.62) * light, clamp(white, 0.0, 1.0) * 0.8);
  col += uSunCol * pow(max(dot(reflect(-uSunDir, n), V), 0.0), 200.0) * 6.0 * sunVis;
  float a = smoothstep(0.0, 0.16, vUv.x) * smoothstep(1.0, 0.84, vUv.x);
  gl_FragColor = outColor(applyFog(col, vW), a * vK);
}
)";
// the spray that hangs where the water falls
static const char* MIST_VS = R"(
attribute vec3 aPos; attribute vec4 aSeed;
uniform float uTime, uPx; uniform mat4 uView, uProj; varying vec3 vW; varying float vA;
void main() {
  float t = fract(uTime * (0.10 + 0.08 * aSeed.x) + aSeed.y), h = aSeed.w;
  vec3 p = aPos + vec3((aSeed.z - 0.5) * (2.0 + 9.0 * t), t * (0.35 + 0.5 * aSeed.x) * h, (aSeed.x - 0.5) * (2.0 + 9.0 * t));
  vW = p; vA = smoothstep(0.0, 0.12, t) * (1.0 - t) * (1.0 - t);
  vec4 mv = uView * vec4(p, 1.0);
  gl_Position = uProj * mv;
  gl_PointSize = min(380.0, uPx * (3.0 + (5.0 + 0.5 * h) * t) / max(1.0, -mv.z));
}
)";
static const char* MIST_FS = R"(
varying vec3 vW; varying float vA;
void main() {
  if (vW.y < uClipY) discard;
  float d = length(gl_PointCoord - 0.5) * 2.0, k = (1.0 - smoothstep(0.2, 1.0, d)) * vA * 0.3;
  if (k < 0.004) discard;
  vec3 col = vec3(0.9, 0.93, 0.96) * (uSunCol * max(uSunDir.y, 0.0) * 0.5 * cloudShadow(vW) + uAmbSky * 1.3);
  gl_FragColor = outColor(applyFog(col, vW), k);
}
)";

static double sst(double a, double b, double x) { double t = std::min(1.0, std::max(0.0, (x - a) / (b - a))); return t * t * (3 - 2 * t); }

void WaterR::buildLakes() {
  const double c = gT.cell, half = gT.half;
  for (const Lake& L : gT.lakes) {
    int i0 = (int)std::floor((L.x0 + half) / c), i1 = (int)std::ceil((L.x1 + half) / c);
    int j0 = (int)std::floor((L.z0 + half) / c), j1 = (int)std::ceil((L.z1 + half) / c);
    int step = 1;
    while (((i1 - i0) / step + 2) * ((j1 - j0) / step + 2) > 65535) step *= 2;
    std::vector<int> is, js;
    for (int i = i0; i < i1; i += step) is.push_back(i);
    is.push_back(i1);
    for (int j = j0; j < j1; j += step) js.push_back(j);
    js.push_back(j1);
    int w = (int)is.size(), h = (int)js.size();
    std::vector<float> v; v.reserve((size_t)w * h * 4);
    for (int j : js) for (int i : is) {
      double x = clamp(-half + i * c, L.x0, L.x1), z = clamp(-half + j * c, L.z0, L.z1);
      v.insert(v.end(), {(float)x, (float)L.level, (float)z, (float)(L.level - gT.gridAt(x, z))});
    }
    std::vector<uint16_t> idx; idx.reserve((size_t)(w - 1) * (h - 1) * 6);
    for (int j = 0; j < h - 1; j++) for (int i = 0; i < w - 1; i++) {
      uint16_t a = (uint16_t)(j * w + i), b = (uint16_t)(a + 1), cc = (uint16_t)(a + w), d = (uint16_t)(cc + 1);
      idx.insert(idx.end(), {a, cc, b, b, cc, d});
    }
    LakeM lm; lm.level = (float)L.level; for (int k = 0; k < 4; k++) lm.out[k] = (float)L.out[k];
    lm.m.count = (int)idx.size();
    glGenBuffers(1, &lm.m.vbo); glBindBuffer(GL_ARRAY_BUFFER, lm.m.vbo); glBufferData(GL_ARRAY_BUFFER, v.size() * 4, v.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &lm.m.ibo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lm.m.ibo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * 2, idx.data(), GL_STATIC_DRAW);
    lakes_.push_back(lm);
  }
  level_ = gT.lakes.empty() ? 0 : (float)gT.lakes[0].level;
}

void WaterR::buildRivers() {
  // lift: how far above its level the water is drawn. shape(s, len): width 0..1 and opacity 0..1 at s metres from the start
  auto ribbon = [&](const std::vector<V3>& P, double width, double lift, const std::function<void(double, double, double&, double&)>& shape) {
    int n = (int)P.size();
    if (n < 2 || n * 2 > 65535) return;
    std::vector<double> S(n, 0.0);
    for (int i = 1; i < n; i++) S[i] = S[i - 1] + std::hypot(P[i][0] - P[i - 1][0], P[i][2] - P[i - 1][2]);
    std::vector<float> v; v.reserve((size_t)n * 14);
    std::vector<uint16_t> idx;
    for (int i = 0; i < n; i++) {
      const V3& a = P[std::max(0, i - 1)]; const V3& b = P[std::min(n - 1, i + 1)]; const V3& p = P[i];
      double ws = 1, al = 1; shape(S[i], S[n - 1], ws, al);
      double tx = b[0] - a[0], tz = b[2] - a[2], l = std::hypot(tx, tz); if (l == 0) l = 1;
      double w = (width / 2 + 1.0) * ws, slope = std::max(0.0, (a[1] - b[1]) / l);
      v.insert(v.end(), {(float)(p[0] - tz / l * w), (float)(p[1] + lift), (float)(p[2] + tx / l * w), 0.0f, (float)(S[i] / width), (float)slope, (float)al});
      v.insert(v.end(), {(float)(p[0] + tz / l * w), (float)(p[1] + lift), (float)(p[2] - tx / l * w), 1.0f, (float)(S[i] / width), (float)slope, (float)al});
      if (i < n - 1) { uint16_t k = (uint16_t)(i * 2); idx.insert(idx.end(), {k, (uint16_t)(k + 2), (uint16_t)(k + 1), (uint16_t)(k + 1), (uint16_t)(k + 2), (uint16_t)(k + 3)}); }
    }
    Mesh m; m.count = (int)idx.size();
    glGenBuffers(1, &m.vbo); glBindBuffer(GL_ARRAY_BUFFER, m.vbo); glBufferData(GL_ARRAY_BUFFER, v.size() * 4, v.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &m.ibo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ibo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * 2, idx.data(), GL_STATIC_DRAW);
    rivers_.push_back(m);
  };
  // a river comes out of a lake and goes into one without an edge; a spring begins as a trickle; a stream melts into the river it joins
  for (const River& r : gT.rivers) ribbon(r.pts, r.width, 0.12, [&r](double s, double len, double& w, double& a) {
    w = 1; a = 1;
    if (r.from == "spring") { w = 0.25 + 0.75 * sst(0, 45, s); a = sst(0, 4, s); }
    if (r.from == "lake") a = sst(0, 26, s);
    if (r.to == "lake") a *= sst(0, 30, len - s);
    if (r.to == "river") a *= sst(0, 14, len - s);
  });
  // beyond the map the river goes on, up the valley and down it, until the haze takes it
  for (size_t k = 0; k < gT.farRivers.size(); k++) {
    const River& r = gT.farRivers[k];
    ribbon(r.pts, r.width, 0, [k](double s, double len, double& w, double& a) { w = 1; a = sst(0, 2500, k ? len - s : s); });
  }
}

void WaterR::buildMist() {
  std::vector<float> v;
  uint64_t q = 7;
  auto rnd = [&]() { q = (q * 16807) % 2147483647; return (double)q / 2147483647.0; };
  for (const Fall& f : gT.falls) {
    double h = f.top - f.foot; int n = (int)std::lround(std::min(90.0, 14 + h * 1.6));
    for (int i = 0; i < n; i++) {
      double u = rnd(), side = (rnd() - 0.5) * (f.width + 2);
      double px = f.x - f.dz * side + f.dx * (rnd() * 6 - 1);
      double py = f.foot + 0.3; if (h > 30) { double a = rnd(), b = rnd(); py += a * b * h * 0.5; }
      double pz = f.z + f.dx * side + f.dz * (rnd() * 6 - 1);
      double s1 = rnd(), s2 = rnd();
      v.insert(v.end(), {(float)px, (float)py, (float)pz, (float)u, (float)s1, (float)s2, (float)std::min(h, 40.0)});
    }
  }
  mistCount_ = (int)(v.size() / 7);
  if (!mistCount_) return;
  glGenBuffers(1, &mistVbo_); glBindBuffer(GL_ARRAY_BUFFER, mistVbo_); glBufferData(GL_ARRAY_BUFFER, v.size() * 4, v.data(), GL_STATIC_DRAW);
}

void WaterR::init(GLuint auxTex) {
  auxTex_ = auxTex;
  lakeProg_.build(LAKE_VS, GLSL_COMMON + LAKE_FS);
  riverProg_.build(RIVER_VS, GLSL_COMMON + RIVER_FS);
  mistProg_.build(MIST_VS, GLSL_COMMON + MIST_FS);
  buildLakes(); buildRivers(); buildMist();
  // the mirror: colour texture + depth
  glGenTextures(1, &reflTex_); glBindTexture(GL_TEXTURE_2D, reflTex_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RW, RH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  const char* ext = (const char*)glGetString(GL_EXTENSIONS);
  bool d24 = ext && std::strstr(ext, "OES_depth24");
  glGenRenderbuffers(1, &reflDepth_); glBindRenderbuffer(GL_RENDERBUFFER, reflDepth_);
  glRenderbufferStorage(GL_RENDERBUFFER, d24 ? 0x81A6 /* GL_DEPTH_COMPONENT24_OES */ : GL_DEPTH_COMPONENT16, RW, RH);
  GLint prev = 0; glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev);
  glGenFramebuffers(1, &fbo_); glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, reflTex_, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, reflDepth_);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) { std::fprintf(stderr, "water: mirror frame buffer incomplete, reflections off\n"); glDeleteFramebuffers(1, &fbo_); fbo_ = 0; }
  glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev);
}

bool WaterR::beginMirror(const Camera& cam, Camera& mc) {
  hasRefl_ = false;
  const double L = level_;
  // skip when the lake cannot be seen or the camera is far above
  if (!fbo_ || lakes_.empty() || cam.eye[1] < L + 0.3 || cam.eye[1] > L + 1500) return false;
  // the camera, its target and its "up" mirrored in the water
  V3 p = {cam.eye[0], 2 * L - cam.eye[1], cam.eye[2]};
  V3 q = add(cam.eye, cam.fwd); q[1] = 2 * L - q[1];
  V3 up = {cam.up[0], -cam.up[1], cam.up[2]};
  mc = cam; mc.znear = 3.0f;      // (the mirror may only have a 16-bit depth buffer)
  mc.look(p, q, up);
  static const Mat4 BIAS = {0.5f, 0, 0, 0, 0, 0.5f, 0, 0, 0, 0, 0.5f, 0, 0.5f, 0.5f, 0.5f, 1};
  reflMat_ = mat4Mul(BIAS, mc.vp);
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo_); glGetIntegerv(GL_VIEWPORT, prevVp_);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  glViewport(0, 0, RW, RH);
  glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  hasRefl_ = true;
  return true;
}

void WaterR::endMirror() {
  glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prevFbo_);
  glViewport(prevVp_[0], prevVp_[1], prevVp_[2], prevVp_[3]);
}

void WaterR::drawLakes(const Camera& cam) {
  Prog& p = lakeProg_;
  p.use();
  gEnv.apply(p, cam);
  p.m4("uVP", cam.vp); p.m4("uReflMat", reflMat_);
  p.f4("uWorld", (float)gT.half, (float)gT.cell, (float)gT.n, 0);
  glUniform4fv(p.U("uWash"), 1, wash_);
  p.i1("uAux", TU_AUX); p.i1("uRefl", TU_REFL);
  glActiveTexture(GL_TEXTURE0 + TU_AUX); glBindTexture(GL_TEXTURE_2D, auxTex_);
  glActiveTexture(GL_TEXTURE0 + TU_REFL); glBindTexture(GL_TEXTURE_2D, reflTex_);
  glActiveTexture(GL_TEXTURE0);
  glDisable(GL_CULL_FACE);
  for (size_t i = 0; i < lakes_.size(); i++) {
    const LakeM& L = lakes_[i];
    p.f1("uLevel", L.level); p.f1("uHasRefl", i == 0 && hasRefl_ ? 1.0f : 0.0f);
    glUniform4fv(p.U("uOutlet"), 1, L.out);
    glBindBuffer(GL_ARRAY_BUFFER, L.m.vbo);
    p.attr("aPos", 3, GL_FLOAT, false, 16, 0); p.attr("aDepth", 1, GL_FLOAT, false, 16, 12);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, L.m.ibo);
    glDrawElements(GL_TRIANGLES, L.m.count, GL_UNSIGNED_SHORT, (void*)0);
  }
  p.attrOff("aPos"); p.attrOff("aDepth");
  glEnable(GL_CULL_FACE);
}

void WaterR::drawRivers(const Camera& cam, int viewH) {
  glEnable(GL_BLEND); glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);   // (the alpha that is there is the distance of what is solid: keep it)
  glDepthMask(GL_FALSE); glDisable(GL_CULL_FACE);
  Prog& p = riverProg_;
  p.use();
  gEnv.apply(p, cam);
  p.m4("uVP", cam.vp);
  p.f4("uWorld", (float)gT.half, (float)gT.cell, (float)gT.n, 0);
  p.i1("uAux", TU_AUX);
  glActiveTexture(GL_TEXTURE0 + TU_AUX); glBindTexture(GL_TEXTURE_2D, auxTex_); glActiveTexture(GL_TEXTURE0);
  for (const Mesh& m : rivers_) {
    glBindBuffer(GL_ARRAY_BUFFER, m.vbo);
    p.attr("aPos", 3, GL_FLOAT, false, 28, 0); p.attr("aUv", 2, GL_FLOAT, false, 28, 12);
    p.attr("aSlope", 1, GL_FLOAT, false, 28, 20); p.attr("aK", 1, GL_FLOAT, false, 28, 24);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m.ibo);
    glDrawElements(GL_TRIANGLES, m.count, GL_UNSIGNED_SHORT, (void*)0);
  }
  p.attrOff("aPos"); p.attrOff("aUv"); p.attrOff("aSlope"); p.attrOff("aK");
  if (mistCount_) {
    Prog& mp = mistProg_;
    mp.use();
    gEnv.apply(mp, cam);
    mp.m4("uView", cam.view); mp.m4("uProj", cam.proj);
    mp.f1("uPx", viewH * 0.5f * cam.proj[5]);
    glBindBuffer(GL_ARRAY_BUFFER, mistVbo_);
    mp.attr("aPos", 3, GL_FLOAT, false, 28, 0); mp.attr("aSeed", 4, GL_FLOAT, false, 28, 12);
    glDrawArrays(GL_POINTS, 0, mistCount_);
    mp.attrOff("aPos"); mp.attrOff("aSeed");
  }
  glDepthMask(GL_TRUE); glDisable(GL_BLEND); glEnable(GL_CULL_FACE);
}

} // namespace fohn
