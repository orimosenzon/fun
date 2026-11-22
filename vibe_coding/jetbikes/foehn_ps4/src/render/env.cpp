#include "env.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace fohn {

const TimeOfDay TIMES[N_TIMES] = {
  {"morning", 0.35, 0.33, 0.40, 1.25, 1.0},
  {"noon", 1.45, 1.0, 0.5, 0.8, 0.8},
  {"afternoon", 2.75, 0.52, 0.46, 1.0, 0.9},
  {"golden", 2.95, 0.2, 0.46, 1.2, 1.05},
  {"dusk", 3.05, 0.06, 0.55, 1.5, 1.6},
};
// "ov" is how overcast it is; under an overcast the sky table turns to an even grey (and with it
// the haze, the ambient light and every reflection), the sun all but goes, and the air thickens.
// Rain wets the ground; snow covers it down to the valley floor.
const WeatherDef WEATHER[N_WEATHER] = {
  {"clear", 0, 0, 1, 1, 1, 4e-5, 0, 0},
  {"rain", 1, 0.98, 0.07, 0.62, 26, 4.6e-4, 1, 0},
  {"snow", 1, 0.98, 0.12, 1.0, 40, 6.0e-4, 0, 1},
};
int timeIndex(const char* key) { for (int i = 0; i < N_TIMES; i++) if (!std::strcmp(TIMES[i].key, key)) return i; return -1; }
int weatherIndex(const char* key) { for (int i = 0; i < N_WEATHER; i++) if (!std::strcmp(WEATHER[i].key, key)) return i; return -1; }

Env gEnv;

// ---------------- atmosphere (Rayleigh + Mie + ozone, single scattering) ----------------
static const double RE = 6360e3, RA = 6460e3, HR = 8000, HM = 1200;
static const double BR[3] = {5.8e-6, 13.5e-6, 33.1e-6}, BM = 21e-6, BO[3] = {0.65e-6, 1.88e-6, 0.085e-6};

static double rayOut(const V3& o, const V3& d, double R) {
  double b = dot(o, d), c = dot(o, o) - R * R, disc = b * b - c;
  return disc < 0 ? -1 : -b + std::sqrt(disc);
}
static V3 optDepth(const V3& o, const V3& d, int n = 6) {
  double L = rayOut(o, d, RA), ds = L / n;
  double r = 0, m = 0, oz = 0;
  for (int i = 0; i < n; i++) {
    double t = (i + 0.5) * ds, h = len(madd(o, d, t)) - RE;
    if (h < -100) return {1e9, 1e9, 1e9};
    r += std::exp(-h / HR) * ds; m += std::exp(-h / HM) * ds; oz += std::max(0.0, 1 - std::fabs(h - 25000) / 15000) * ds;
  }
  return {r, m, oz};
}
// radiance arriving at height h0 from direction d, sun direction s
static V3 scatter(const V3& d, const V3& s, double h0, double mieScale) {
  V3 o = {0, RE + h0, 0};
  double L = rayOut(o, d, RA);
  double b = o[1] * d[1], c = o[1] * o[1] - RE * RE, disc = b * b - c;      // stop at the ground
  if (disc > 0 && -b - std::sqrt(disc) > 0) L = -b - std::sqrt(disc);
  const int N = 14;
  double ds = L / N, mu = dot(d, s);
  double pr = 3 / (16 * PI) * (1 + mu * mu), g = 0.78;
  double pm = 3 / (8 * PI) * ((1 - g * g) * (1 + mu * mu)) / ((2 + g * g) * std::pow(1 + g * g - 2 * g * mu, 1.5));
  V3 out = {0, 0, 0};
  double dr = 0, dm = 0, doz = 0;
  for (int i = 0; i < N; i++) {
    double t = (i + 0.5) * ds;
    V3 p = madd(o, d, t);
    double h = len(p) - RE;
    double er = std::exp(-h / HR) * ds, em = std::exp(-h / HM) * ds * mieScale;
    dr += er; dm += em; doz += std::max(0.0, 1 - std::fabs(h - 25000) / 15000) * ds;
    V3 sd = optDepth(p, s);
    if (sd[0] > 1e8) continue;
    for (int k = 0; k < 3; k++) {
      double tau = BR[k] * (dr + sd[0]) + 1.1 * BM * (dm + sd[1] * mieScale) + BO[k] * (doz + sd[2]);
      out[k] += std::exp(-tau) * (er * BR[k] * pr + em * BM * pm);
    }
  }
  return out;
}
static V3 sunTransmittance(const V3& s, double h0, double mieScale) {
  V3 od = optDepth({0, RE + h0, 0}, s, 12);
  V3 t;
  for (int k = 0; k < 3; k++) t[k] = std::exp(-(BR[k] * od[0] + 1.1 * BM * od[1] * mieScale + BO[k] * od[2]));
  return t;
}

static const int LW = 96, LH = 48;
static const double SUN_I = 22;
static double lutV(double el) { return std::sqrt(std::min(1.0, std::max(0.0, (el + 0.12) / (PI / 2 + 0.12)))); }

// the red channel of the noise texture, kept for the CPU (the statistics of the clouds, below)
static std::vector<float> gNoiseR;
static double vnoiseCpu(double x, double y) {
  const int N = 256;
  x -= 0.5; y -= 0.5;
  double fx = std::floor(x), fy = std::floor(y), tx = x - fx, ty = y - fy;
  int x0 = ((int)fx % N + N) % N, y0 = ((int)fy % N + N) % N, x1 = (x0 + 1) % N, y1 = (y0 + 1) % N;
  double a = gNoiseR[(size_t)y0 * N + x0], b = gNoiseR[(size_t)y0 * N + x1], c = gNoiseR[(size_t)y1 * N + x0], d = gNoiseR[(size_t)y1 * N + x1];
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}
static double cloudNCpu(double x, double y) {
  return 0.5 * vnoiseCpu(x * 16, y * 16) + 0.27 * vnoiseCpu(x * 37 + 11, y * 37 + 11) + 0.15 * vnoiseCpu(x * 83 + 5, y * 83 + 5) + 0.08 * vnoiseCpu(x * 190, y * 190);
}

void Env::init() {
  // noise: 256x256 random bytes, bilinear + mip-mapped: smooth value noise with period 256
  const int N = 256;
  std::vector<uint8_t> d((size_t)N * N * 4);
  uint32_t s = 1234567;
  for (size_t i = 0; i < d.size(); i++) { s = s * 1664525u + 1013904223u; d[i] = (uint8_t)std::floor(s / 4294967296.0 * 256); }
  gNoiseR.resize((size_t)N * N);
  for (size_t i = 0; i < gNoiseR.size(); i++) gNoiseR[i] = d[i * 4] / 255.0f;
  glGenTextures(1, &noiseTex);
  glBindTexture(GL_TEXTURE_2D, noiseTex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, N, N, 0, GL_RGBA, GL_UNSIGNED_BYTE, d.data());
  glGenerateMipmap(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // the sky table and the shadow map, in one texture (see env.hpp). Linear filtering, for the
  // table; the shadow look-ups ask for the middle of a texel, which is the texel itself.
  if (shadowSize >= 256) { atlasW = atlasH = shadowSize; shadowRows = shadowSize - 64; }
  else { atlasW = LW; atlasH = LH; shadowRows = 0; }
  glGenTextures(1, &lutTex);
  glBindTexture(GL_TEXTURE_2D, lutTex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, atlasW, atlasH, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

void Env::setTime(int ti, int wi) {
  timeIdx = ti; wxIdx = wi;
  const TimeOfDay& t = TIMES[ti];
  const WeatherDef& w = WEATHER[wi];
  V3 s = {std::cos(t.el) * std::cos(t.az), std::sin(t.el), std::cos(t.el) * std::sin(t.az)};
  sunDir = s; sunAz = (float)t.az;
  lut.assign((size_t)LW * LH * 4, 0.0f);
  std::vector<float>& f = lut;
  // table: u = azimuth from the sun (0..pi), v = elevation (denser near the horizon)
  for (int j = 0; j < LH; j++) {
    double v = (j + 0.5) / LH, el = v * v * (PI / 2 + 0.12) - 0.12;
    for (int i = 0; i < LW; i++) {
      double az = (i + 0.5) / LW * PI;
      V3 d = {std::cos(el) * std::cos(az), std::sin(el), std::cos(el) * std::sin(az)};
      V3 c = scatter(d, {std::cos(t.el), std::sin(t.el), 0}, 1400, t.haze);
      size_t k = ((size_t)j * LW + i) * 4;
      f[k] = (float)(c[0] * SUN_I); f[k + 1] = (float)(c[1] * SUN_I); f[k + 2] = (float)(c[2] * SUN_I); f[k + 3] = 1;
    }
  }
  if (w.ov > 0) {
    // overcast: the light of the whole sky, spread evenly, a little brighter overhead
    double M = 0; int n = 0;
    for (int j = (int)std::floor(LH * 0.3); j < LH; j++) for (int i = 0; i < LW; i += 2) { size_t q = ((size_t)j * LW + i) * 4; M += 0.3 * f[q] + 0.55 * f[q + 1] + 0.15 * f[q + 2]; n++; }
    M = M / n * w.sky;
    for (int j = 0; j < LH; j++) {
      double g = M * (0.72 + 0.5 * (j + 0.5) / LH);
      for (int i = 0; i < LW; i++) {
        size_t q = ((size_t)j * LW + i) * 4;
        f[q] += (float)((g * 0.93 - f[q]) * w.ov); f[q + 1] += (float)((g * 0.97 - f[q + 1]) * w.ov); f[q + 2] += (float)((g * 1.04 - f[q + 2]) * w.ov);
      }
    }
  }
  // upload as RGBM: colour = rgb * a^2 * range
  float range = 1e-6f;
  for (size_t q = 0; q < f.size(); q += 4) range = std::max(range, std::max(f[q], std::max(f[q + 1], f[q + 2])));
  lutRange = range;
  std::vector<uint8_t> px((size_t)LW * LH * 4);
  for (size_t q = 0; q < f.size(); q += 4) {
    float r = f[q] / range, g = f[q + 1] / range, b = f[q + 2] / range;
    float m = std::max(r, std::max(g, b));
    int a = (int)std::ceil(std::sqrt(m) * 255.0f); a = std::min(255, std::max(1, a));
    float mq = (a / 255.0f) * (a / 255.0f);
    auto enc = [&](float x) { return (uint8_t)std::min(255.0f, std::max(0.0f, std::round(x / mq * 255.0f))); };
    px[q] = enc(r); px[q + 1] = enc(g); px[q + 2] = enc(b); px[q + 3] = (uint8_t)a;
  }
  glBindTexture(GL_TEXTURE_2D, lutTex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, atlasH - LH, LW, LH, GL_RGBA, GL_UNSIGNED_BYTE, px.data());

  // direct sun
  V3 tr = sunTransmittance(s, 1400, t.haze);
  double k = SUN_I * 0.115 * w.sun;
  for (int c = 0; c < 3; c++) sunCol[c] = (float)(tr[c] * k);
  // ambient: mean of the upper sky, a little bounce from below
  double amb[3] = {0, 0, 0}; int n = 0;
  for (int j = (int)std::floor(LH * 0.35); j < LH; j++) for (int i = 0; i < LW; i += 4) { size_t q = ((size_t)j * LW + i) * 4; amb[0] += f[q]; amb[1] += f[q + 1]; amb[2] += f[q + 2]; n++; }
  double ak = 2.8 / n;
  // shade is lit by blue sky but also by clouds and bright slopes: pull the ambient towards neutral
  double am = (amb[0] + amb[1] + amb[2]) / 3;
  for (int c = 0; c < 3; c++) amb[c] = amb[c] * 0.62 + am * 0.38;
  for (int c = 0; c < 3; c++) ambSky[c] = (float)(amb[c] * ak);
  double up = std::max(0.05, s[1]);
  ambGnd[0] = (float)(0.10 * sunCol[0] * up + amb[0] * ak * 0.25);
  ambGnd[1] = (float)(0.10 * sunCol[1] * up + amb[1] * ak * 0.25);
  ambGnd[2] = (float)(0.07 * sunCol[2] * up + amb[2] * ak * 0.2);
  cloud[0] = (float)std::max(t.cloud, w.cloud); cloud[1] = 4300;
  fog[0] = (float)(2.0e-5 * t.haze * w.fog); fog[1] = w.ov > 0 ? 1.0f / 5000 : 1.0f / 1700; fog[2] = 900; fog[3] = 1;
  wx[0] = (float)w.ov; wx[1] = (float)w.wet; wx[2] = (float)w.snow; wx[3] = 0;
  exposure = (float)(t.exposure * (w.ov > 0 ? 1.25 : 1.0));

  // The clouds as a mirror on the bike sees them. The web build makes an environment map of the
  // whole sky dome, clouds and all, and blurs it; here the same clouds are evened out: how much of
  // the sky they cover, how much of them the sun reaches and how much of them is thin edge, taken
  // over the same noise the sky shader draws them from (Sky's fragment shader, term for term).
  {
    const double lo = 1 - cloud[0] - 0.1;
    double sx = s[0], sz = s[2], sl = std::hypot(sx, sz) + 1e-4, sk = 0.012 + 0.03 * (1 - s[1]);
    sx = sx / sl * sk; sz = sz / sl * sk;
    double D = 0, DS = 0, DE = 0, C = 0; const int M = 20000; uint32_t r = 2463534242u;
    auto rnd = [&]() { r ^= r << 13; r ^= r >> 17; r ^= r << 5; return r / 4294967296.0; };
    for (int i = 0; i < M; i++) {
      double x = rnd() * 16, y = rnd() * 16, n = cloudNCpu(x, y);
      double dens = sstep(lo, lo + 0.26, n);
      double shade = std::min(1.0, std::max(0.0, 1 - 2.6 * std::max(cloudNCpu(x + sx, y + sz) - lo, 0.0) - 1.6 * std::max(cloudNCpu(x + sx * 2.4, y + sz * 2.4) - lo, 0.0)));
      D += dens; DS += dens * shade; DE += dens * (1 - sstep(0.0, 0.5, dens));
      C += sstep(0.52, 0.8, 0.6 * vnoiseCpu(x * 40, y * 12) + 0.4 * vnoiseCpu(x * 90 + 3, y * 30 + 3));
    }
    cloudEnv[0] = (float)(D / M); cloudEnv[1] = (float)(D > 0 ? DS / D : 1); cloudEnv[2] = (float)(D > 0 ? DE / D : 0); cloudEnv[3] = (float)(C / M);
  }

  // The sky as light on a surface (for which the web build has that environment map again; here
  // the first two bands of its spherical harmonics, which is all a matte surface can tell of it):
  // mean radiance m0 and its first moment m1, and the light arriving on a normal n, over pi, is
  // m0 + 2 m1.n. Below the horizon the dome goes on in the colour of the haze.
  double m0[3] = {0, 0, 0}, m1[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, wsum = 0;
  const int NE = 24, NA = 48;
  for (int j = 0; j < NE; j++) {
    double el = ((j + 0.5) / NE - 0.5) * PI, ce = std::cos(el);
    for (int i = 0; i < NA; i++) {
      double az = (i + 0.5) / NA * 2 * PI;
      V3 d = {ce * std::cos(az), std::sin(el), ce * std::sin(az)}, L = skyEnv(d);
      for (int c = 0; c < 3; c++) { m0[c] += ce * L[c]; for (int k = 0; k < 3; k++) m1[k][c] += ce * L[c] * d[k]; }
      wsum += ce;
    }
  }
  for (int c = 0; c < 3; c++) {
    envSH[c] = (float)(m0[c] / wsum);
    for (int k = 0; k < 3; k++) envSH[3 + k * 3 + c] = (float)(2 * m1[k][c] / wsum);
  }
}

// the sky with its clouds evened out, in the direction d: skyEnv() of GLSL_PBR, on the CPU
V3 Env::skyEnv(const V3& d) const {
  V3 col = skyColor(std::atan2(d[2], d[0]), std::asin(std::min(1.0, std::max(-1.0, d[1]))));
  if (d[1] > 0.004) {
    double mu = std::max(0.0, dot(d, sunDir)), t = cloud[1] / d[1], far = std::exp(-t / 52000.0), S = cloudEnv[1];
    V3 hz = skyColor(std::atan2(d[2], d[0]), std::asin(0.03 / std::sqrt(d[0] * d[0] + d[2] * d[2] + 0.0009)));
    double w = cloudEnv[0] * sstep(0.004, 0.05, d[1]) * (0.35 + 0.65 * far) * 0.96;
    double t2 = 11000.0 / d[1];
    for (int c = 0; c < 3; c++) {
      double lit = sunCol[c] * (0.5 + 0.9 * S + 2.2 * std::pow(mu, 6.0) * cloudEnv[2]) + ambSky[c] * 0.9, dark = ambSky[c] * 0.55 + sunCol[c] * 0.06;
      double cc = (dark + (lit - dark) * (0.25 + 0.75 * S)) * 0.52;
      cc += (col[c] * (0.78 + 0.34 * S) - cc) * wx[0];
      cc = hz[c] + (cc - hz[c]) * far;
      double o = col[c] + (cc - col[c]) * w;
      o += cloudEnv[3] * (sunCol[c] * (0.10 + 0.5 * std::pow(mu, 8.0)) + ambSky[c] * 0.08) * std::exp(-t2 / 90000.0) * (1 - cloudEnv[0]);
      col[c] = o;
    }
  }
  return col;
}

V3 Env::skyColor(double az, double el) const {
  double u = std::fabs(std::fmod(az - sunAz + PI * 3, 2 * PI) - PI) / PI;
  int i = std::min(LW - 1, (int)std::floor(u * LW)), j = std::min(LH - 1, (int)std::floor(lutV(el) * LH));
  size_t k = ((size_t)j * LW + i) * 4;
  return {lut[k], lut[k + 1], lut[k + 2]};
}

void Env::apply(Prog& p, const Camera& cam, float clipY, bool mirror) const {
  p.v3("cameraPosition", cam.eye);
  p.v3("uSunDir", sunDir);
  glUniform3fv(p.U("uSunCol"), 1, sunCol);
  glUniform3fv(p.U("uAmbSky"), 1, ambSky);
  glUniform3fv(p.U("uAmbGnd"), 1, ambGnd);
  glUniform4fv(p.U("uFog"), 1, fog);
  glUniform4fv(p.U("uCloud"), 1, cloud);
  glUniform4fv(p.U("uWx"), 1, wx);
  glUniform4fv(p.U("uHole"), 1, hole);
  glUniform3fv(p.U("uEnvSH"), 4, envSH);
  glUniform4fv(p.U("uCloudEnv"), 1, cloudEnv);
  p.f4("uAtlas", 1.0f / atlasW, 1.0f / atlasH, (float)(atlasH - LH), (float)shadowRows);
  p.m4("uShadowMat", shadowMat);
  p.f1("uShadowOn", shadowOn ? 1.0f : 0.0f);
  p.f1("uTime", time);
  p.f1("uSunAz", sunAz);
  p.f1("uClipY", clipY);
  p.f1("uLutRange", lutRange);
  p.f2("uOut", exposure, mirror ? 1.0f : (float)outMode);
  p.i1("uSkyLUT", TU_SKY);
  p.i1("uNoise", TU_NOISE);
  glActiveTexture(GL_TEXTURE0 + TU_SKY); glBindTexture(GL_TEXTURE_2D, lutTex);
  glActiveTexture(GL_TEXTURE0 + TU_NOISE); glBindTexture(GL_TEXTURE_2D, noiseTex);
  glActiveTexture(GL_TEXTURE0);
}

// ---------------- shared GLSL ----------------
const std::string GLSL_COMMON = R"(
precision highp float;
uniform vec3 cameraPosition;
uniform vec3 uSunDir; uniform vec3 uSunCol; uniform vec3 uAmbSky; uniform vec3 uAmbGnd;
uniform sampler2D uSkyLUT; uniform sampler2D uNoise; uniform vec4 uFog; uniform float uTime; uniform vec4 uCloud;
uniform float uSunAz; uniform float uClipY; uniform float uLutRange; uniform vec4 uWx; uniform vec4 uHole; uniform vec2 uOut;
uniform vec4 uAtlas; uniform mat4 uShadowMat; uniform float uShadowOn;
const float PI = 3.14159265;
float vnoise(vec2 p) { return texture2D(uNoise, p / 256.0).r; }          // smooth value noise, period 256
vec4 vnoise4(vec2 p) { return texture2D(uNoise, p / 256.0); }
float fbm3(vec2 p) { return 0.57 * vnoise(p) + 0.29 * vnoise(p * 2.13 + 31.7) + 0.14 * vnoise(p * 4.37 + 71.3); }
vec3 skyLUT(vec3 d) {
  float az = atan(d.z, d.x) - uSunAz;
  float u = abs(mod(az + PI, 2.0 * PI) - PI) / PI;
  float el = asin(clamp(d.y, -1.0, 1.0));
  float v = sqrt(clamp((el + 0.12) / (PI * 0.5 + 0.12), 0.0, 1.0));
  // (the table is 96 x 48 texels in the top rows of the texture it shares with the shadow map;
  //  uAtlas = 1 / width, 1 / height, the table's first row, the rows of the shadow map)
  vec2 p = clamp(vec2(u, v) * vec2(96.0, 48.0), vec2(0.5), vec2(95.5, 47.5));
  vec4 t = texture2D(uSkyLUT, vec2(p.x, p.y + uAtlas.z) * uAtlas.xy);
  return t.rgb * (t.a * t.a * uLutRange);
}
// The shadow map near the bike (shadow.cpp): depth from the sun, packed into the four bytes of a
// colour, since GLES2 has no depth texture to count on. Five taps, as the web build's ground takes them.
vec4 packDepth(float d) { vec4 e = fract(d * vec4(1.0, 255.0, 65025.0, 16581375.0)); return e - e.yzww * vec4(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 0.0); }
float shadowTap(vec2 p, float z) {
  p = clamp(p, vec2(0.5), vec2(1.0 / uAtlas.x, uAtlas.w) - 0.5);
  return step(z, dot(texture2D(uSkyLUT, (floor(p) + 0.5) * uAtlas.xy), vec4(1.0, 1.0 / 255.0, 1.0 / 65025.0, 1.0 / 16581375.0)));
}
// how much of the sun reaches wp past what stands near the bike. bias: in units of the map's depth (899 m)
float realShadow(vec3 wp, float bias) {
  if (uShadowOn < 0.5) return 1.0;
  vec3 sc = (uShadowMat * vec4(wp, 1.0)).xyz;
  if (sc.x < 0.0 || sc.x > 1.0 || sc.y < 0.0 || sc.y > 1.0 || sc.z > 1.0) return 1.0;
  float z = sc.z - bias; vec2 p = sc.xy * vec2(1.0 / uAtlas.x, uAtlas.w);
  float s = shadowTap(p, z) * 2.0 + shadowTap(p + vec2(1.3, 0.4), z) + shadowTap(p + vec2(-0.4, 1.3), z) + shadowTap(p + vec2(-1.3, -0.4), z) + shadowTap(p + vec2(0.4, -1.3), z);
  vec2 e = min(sc.xy, 1.0 - sc.xy);
  return mix(1.0, s / 6.0, smoothstep(0.0, 0.08, min(e.x, e.y)));
}
// cloud cover seen from the ground at world xz (also drives cloud shadows)
float cloudDensity(vec2 xz) {
  vec2 p = xz / 9000.0 + vec2(uTime * 0.0012, uTime * 0.0005);
  float n = 0.5 * vnoise(p * 16.0) + 0.27 * vnoise(p * 37.0 + 11.0) + 0.15 * vnoise(p * 83.0 + 5.0) + 0.08 * vnoise(p * 190.0);
  return smoothstep(1.0 - uCloud.x - 0.1, 1.0 - uCloud.x + 0.16, n);
}
float cloudShadow(vec3 wp) {
  vec2 xz = wp.xz + uSunDir.xz / max(uSunDir.y, 0.08) * (uCloud.y - wp.y);
  return 1.0 - 0.62 * cloudDensity(xz);
}
// aerial perspective: exponential height fog (closed form), blue-shifted extinction,
// in-scatter colour taken from the sky table near the horizon in the viewing direction
vec3 applyFog(vec3 col, vec3 wp) {
  vec3 d = wp - cameraPosition; float dist = length(d); vec3 dir = d / max(dist, 1e-3);
  float k = dir.y * uFog.y * dist;
  float od = uFog.x * exp(-(cameraPosition.y - uFog.z) * uFog.y) * dist * (abs(k) > 1e-3 ? (1.0 - exp(-k)) / k : 1.0);
  vec3 T = exp(-od * vec3(0.72, 1.0, 1.42));
  vec3 hz = skyLUT(normalize(vec3(dir.x, 0.035 + 0.25 * max(dir.y, 0.0), dir.z)));
  return mix(hz * uFog.w, col, T);
}
// What leaves the shader. uOut = (exposure, mode):
//   0  the screen: exposure, ACES and the grade (when there is no post pass)
//   1  the lake's mirror: a reversible curve, so that the water can have the light back
//   2  the scene of the post pass: the light itself, packed into 8 bits as sqrt(c / (1 + c)), which
//      is about as fine as the screen from the shadows to the sun; the pass does the tone mapping.
//      What is solid also writes how far away it is into the alpha (1 / (1 + distance / 20 m)): the
//      pass blurs by it, and GLES2 has no depth texture to count on.
vec3 aces(vec3 x) { x *= 0.6; return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
vec3 screenCurve(vec3 col) {
  col = aces(col * uOut.x);
  col = pow(col, vec3(1.0 / 2.2));
  col = mix(col, col * col * (3.0 - 2.0 * col), 0.22);
  return mix(vec3(dot(col, vec3(0.3, 0.59, 0.11))), col, 1.10);
}
vec4 outColor(vec3 col, float a) {
  col = max(col, vec3(0.0));
  if (uOut.y > 1.5) return vec4(sqrt(col / (1.0 + col)), a);
  if (uOut.y > 0.5) return vec4(col / (1.0 + col), a);
  col = screenCurve(col);
  col += vec3(0.0, 0.004, 0.012) * (1.0 - dot(col, vec3(0.3, 0.59, 0.11)));
  return vec4(clamp(col, 0.0, 1.0), a);
}
// solid surfaces, and the sky (which is infinitely far)
vec4 outSolid(vec3 col) { vec4 c = outColor(col, 1.0); if (uOut.y > 1.5) c.a = gl_FragCoord.w / (gl_FragCoord.w + 0.05); return c; }
vec4 outSky(vec3 col) { return outColor(col, uOut.y > 1.5 ? 0.0 : 1.0); }
// light that is added to what is already there (the plumes). On the screen: the same curve without
// the lift of the blacks, so that no light adds nothing. In the post pass it goes to a buffer of
// its own, packed like the scene.
vec4 outAdd(vec3 col) {
  col = max(col, vec3(0.0));
  if (uOut.y > 1.5) return vec4(sqrt(col / (1.0 + col)), 1.0);
  return vec4(clamp(screenCurve(col), 0.0, 1.0), 1.0);
}
)";

// Surfaces that are made things (the bike, the pads): the model of three.js' MeshStandardMaterial
// and of the clear coat of MeshPhysicalMaterial, lit as the web build lights them: the sun, a
// hemisphere light at 0.75 and the sky itself as an environment at 0.55 (js/main.js). A metal has
// no colour of its own, it only mirrors; that is why the lift engines are dark under a dim sky.
const std::string GLSL_PBR = R"(
uniform vec3 uEnvSH[4]; uniform vec4 uCloudEnv;
// The sky a mirror sees: the table, and over it the clouds evened out (uCloudEnv: how much of the
// sky they cover, how much of them the sun reaches, how much is thin edge, and the cirrus). The
// same terms as the sky's own shader, from the origin, where the web build makes its environment map.
vec3 skyEnv(vec3 d) {
  vec3 col = skyLUT(d);
  if (d.y > 0.004) {
    float mu = max(dot(d, uSunDir), 0.0), t = uCloud.y / d.y, far = exp(-t / 52000.0);
    vec3 lit = uSunCol * (0.5 + 0.9 * uCloudEnv.y + 2.2 * pow(mu, 6.0) * uCloudEnv.z) + uAmbSky * 0.9;
    vec3 dark = uAmbSky * 0.55 + uSunCol * 0.06;
    vec3 cc = mix(dark, lit, 0.25 + 0.75 * uCloudEnv.y) * 0.52;
    cc = mix(cc, col * (0.78 + 0.34 * uCloudEnv.y), uWx.x);
    cc = mix(skyLUT(normalize(vec3(d.x, 0.03, d.z))), cc, far);
    col = mix(col, cc, uCloudEnv.x * smoothstep(0.004, 0.05, d.y) * (0.35 + 0.65 * far) * 0.96);
    col += uCloudEnv.w * (uSunCol * (0.10 + 0.5 * pow(mu, 8.0)) + uAmbSky * 0.08) * exp(-11000.0 / d.y / 90000.0) * (1.0 - uCloudEnv.x);
  }
  return col;
}
// the light of the whole sky on a surface facing n, over pi
vec3 envIrr(vec3 n) { return max(uEnvSH[0] + uEnvSH[1] * n.x + uEnvSH[2] * n.y + uEnvSH[3] * n.z, vec3(0.0)); }
// The sky as a surface of this roughness mirrors it (the web build's map is blurred by roughness):
// polished, a narrow look, three taps up and down so that the bright line of the horizon is not a
// hairline on a brushed can; rough, the light of that whole side of the sky.
vec3 envRad(vec3 r, float rough) {
  if (rough > 0.84) return envIrr(r);
  vec3 w = vec3(0.0, 0.5 * rough, 0.0);
  vec3 c = 0.5 * skyEnv(r) + 0.25 * (skyEnv(normalize(r + w)) + skyEnv(normalize(r - w)));
  return mix(c, envIrr(r), smoothstep(0.15, 0.85, rough));
}
// GGX: visibility * distribution * pi (the pi goes with the one in the sun's intensity)
float ggx(float rough, float nl, float nv, float nh) {
  float a = rough * rough, a2 = a * a;
  float d = nh * nh * (a2 - 1.0) + 1.0;
  float v = 0.5 / max(nl * sqrt(a2 + (1.0 - a2) * nv * nv) + nv * sqrt(a2 + (1.0 - a2) * nl * nl), 1e-4);
  return a2 / max(d * d, 1e-7) * v;
}
// how much of the environment a rough surface mirrors (scale and bias of F0), fitted curve
vec2 dfg(float nv, float rough) {
  vec4 r = rough * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
  float a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
  return vec2(-1.04, 1.04) * a004 + r.zw;
}
// albedo: linear colour. m = (metalness, roughness, roughness of the clear coat or 0 for none).
// n must face the eye; sunVis is how much of the sun reaches the place.
vec3 pbr(vec3 albedo, vec3 m, vec3 n, vec3 v, float sunVis) {
  float rough = max(m.y, 0.0525);
  vec3 diff = albedo * (1.0 - m.x), f0 = mix(vec3(0.04), albedo, m.x);
  vec3 h = normalize(uSunDir + v);
  float nv = clamp(dot(n, v), 0.0, 1.0), nl = clamp(dot(n, uSunDir), 0.0, 1.0), nh = clamp(dot(n, h), 0.0, 1.0), vh = clamp(dot(v, h), 0.0, 1.0);
  vec3 sun = uSunCol * (nl * sunVis);
  float fh = pow(1.0 - vh, 5.0), fv = pow(1.0 - nv, 5.0);
  vec3 col = sun * (diff + (f0 + (1.0 - f0) * fh) * ggx(rough, nl, nv, nh));
  vec3 irr = mix(uAmbGnd, uAmbSky, n.y * 0.5 + 0.5) * 0.75 + envIrr(n) * 0.55;
  vec2 ab = dfg(nv, rough);
  vec3 r = reflect(-v, n);
  col += diff * irr + envRad(normalize(mix(r, n, rough * rough)), rough) * 0.55 * (f0 * ab.x + ab.y);
  if (m.z > 0.0) {
    float cr = max(m.z, 0.0525);
    vec2 cab = dfg(nv, cr);
    vec3 coat = sun * ((0.04 + 0.96 * fh) * ggx(cr, nl, nv, nh)) + skyEnv(r) * 0.55 * (0.04 * cab.x + cab.y);
    col = col * (1.0 - (0.04 + 0.96 * fv)) + coat;
  }
  return col;
}
)";

// ---------------- sky dome ----------------
static const char* SKY_VS = R"(
attribute vec2 aPos;
uniform vec3 uCamR, uCamU, uCamF; uniform vec2 uTan;
varying vec3 vDir;
void main() { vDir = uCamF + uCamR * (aPos.x * uTan.x) + uCamU * (aPos.y * uTan.y); gl_Position = vec4(aPos, 1.0, 1.0); }
)";
static const char* SKY_FS = R"(
varying vec3 vDir;
float cloudN(vec2 p) { return 0.5 * vnoise(p * 16.0) + 0.27 * vnoise(p * 37.0 + 11.0) + 0.15 * vnoise(p * 83.0 + 5.0) + 0.08 * vnoise(p * 190.0); }
void main() {
  vec3 d = normalize(vDir);
  if (uClipY > -1e8) d.y = abs(d.y);
  vec3 col = skyLUT(d);
  float mu = dot(d, uSunDir);
  // sun disc with limb darkening, and a tight glow
  float disc = smoothstep(0.99985, 0.99996, mu);
  col += uSunCol * (disc * 60.0 + pow(max(mu, 0.0), 900.0) * 3.0 + pow(max(mu, 0.0), 60.0) * 0.12) * (1.0 - uWx.x);   // no sun disc through an overcast
  if (d.y > 0.004) {
    // cumulus deck
    float t = (uCloud.y - cameraPosition.y) / d.y;
    vec2 xz = cameraPosition.xz + d.xz * t;
    vec2 p = xz / 9000.0 + vec2(uTime * 0.0012, uTime * 0.0005);
    float n = cloudN(p);
    float lo = 1.0 - uCloud.x - 0.1;
    float dens = smoothstep(lo, lo + 0.26, n);
    // self-shadowing: density towards the sun
    vec2 sdir = normalize(uSunDir.xz + 1e-4) * (0.012 + 0.03 * (1.0 - uSunDir.y));
    float n2 = cloudN(p + sdir), n3 = cloudN(p + sdir * 2.4);
    float shade = clamp(1.0 - 2.6 * max(n2 - lo, 0.0) - 1.6 * max(n3 - lo, 0.0), 0.0, 1.0);
    float edge = 1.0 - smoothstep(0.0, 0.5, dens);
    float fwd = pow(max(mu, 0.0), 6.0);
    vec3 lit = uSunCol * (0.5 + 0.9 * shade + 2.2 * fwd * edge) + uAmbSky * 0.9;
    vec3 dark = uAmbSky * 0.55 + uSunCol * 0.06;
    vec3 cc = mix(dark, lit, 0.25 + 0.75 * shade) * 0.52;
    cc = mix(cc, skyLUT(d) * (0.78 + 0.34 * shade), uWx.x);                 // an overcast is lit from within, evenly
    float far = exp(-t / 52000.0);
    cc = mix(skyLUT(normalize(vec3(d.x, 0.03, d.z))), cc, far);
    col = mix(col, cc, dens * smoothstep(0.004, 0.05, d.y) * (0.35 + 0.65 * far) * 0.96);
    // thin cirrus far above
    float t2 = (11000.0 - cameraPosition.y) / d.y;
    vec2 q = (cameraPosition.xz + d.xz * t2) / 30000.0 + uTime * 0.0006;
    float ci = smoothstep(0.52, 0.8, 0.6 * vnoise(q * vec2(40.0, 12.0)) + 0.4 * vnoise(q * vec2(90.0, 30.0) + 3.0));
    col += ci * (uSunCol * (0.10 + 0.5 * pow(max(mu, 0.0), 8.0)) + uAmbSky * 0.08) * exp(-t2 / 90000.0) * (1.0 - dens);
  }
  gl_FragColor = outSky(col);
}
)";

void Sky::init() {
  prog_.build(SKY_VS, GLSL_COMMON + SKY_FS);
  const float tri[6] = {-1, -1, 3, -1, -1, 3};
  glGenBuffers(1, &vbo_); glBindBuffer(GL_ARRAY_BUFFER, vbo_); glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);
}

void Sky::draw(const Camera& cam, float clipY, bool mirror) {
  prog_.use();
  gEnv.apply(prog_, cam, clipY, mirror);
  prog_.v3("uCamR", cam.right); prog_.v3("uCamU", cam.up); prog_.v3("uCamF", cam.fwd);
  float ty = std::tan(cam.fovY / 2);
  prog_.f2("uTan", ty * cam.aspect, ty);
  glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  prog_.attr("aPos", 2, GL_FLOAT, false, 0, 0);
  glDrawArrays(GL_TRIANGLES, 0, 3);
  prog_.attrOff("aPos");
  glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE);
}

} // namespace fohn
