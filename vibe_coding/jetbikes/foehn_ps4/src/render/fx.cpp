#include "fx.hpp"
#include "env.hpp"
#include "../sim/terrain.hpp"
#include <algorithm>
#include <cmath>

namespace fohn {

// A plume is a tube along -x of the nozzle's frame, narrowing to its tip. aTube: how far along
// (0 at the nozzle .. 1 at the tip), and the cosine and sine of the angle around it.
static const char* PLUME_VS = R"(
attribute vec3 aTube;
uniform mat4 uVP, uNoz; uniform vec3 uDim;        // length, width, size of the nozzle
varying float vX; varying vec3 vN; varying vec3 vW;
void main() {
  float r = mix(0.08, 0.035, aTube.x) * uDim.z * uDim.y;
  vec4 w = uNoz * vec4(-aTube.x * uDim.x, r * aTube.y, r * aTube.z, 1.0);
  vW = w.xyz; vX = aTube.x;
  vN = (uNoz * vec4(-0.045 * uDim.z * uDim.y / uDim.x, aTube.y, aTube.z, 0.0)).xyz;
  gl_Position = uVP * w;
}
)";
static const char* PLUME_FS = R"(
uniform float uPow, uAb, uSeed;
varying float vX; varying vec3 vN; varying vec3 vW;
void main() {
  float x = vX;                                      // 0 at the nozzle .. 1 at the tip
  float core = pow(abs(dot(normalize(vN), normalize(cameraPosition - vW))), 1.6);
  float flick = 0.9 + 0.1 * sin(uTime * 61.0 + uSeed * 9.0 + x * 20.0);
  // shock diamonds: standing pressure pattern, spacing grows with thrust
  float dia = pow(0.5 + 0.5 * cos(x * (30.0 - 12.0 * uPow) - 0.6), 5.0) * exp(-x * 3.2);
  vec3 dry = vec3(0.35, 0.55, 1.0) * (0.10 * exp(-x * 5.0) + 0.25 * dia * uPow) * uPow;
  vec3 ab = (mix(vec3(0.75, 0.9, 1.0), vec3(0.55, 0.35, 1.0), smoothstep(0.0, 0.5, x)) * exp(-x * 2.3) * 2.4 + vec3(0.9, 1.1, 1.4) * dia * 4.0) * uAb;
  float fade = smoothstep(1.0, 0.75, x);
  gl_FragColor = outAdd((dry + ab) * core * flick * fade * 3.0);
}
)";

// (the web build: a point of 800 * size / distance pixels, at most 900, on a screen 1080 high)
static const char* PART_VS = R"(
attribute vec3 aPos; attribute vec2 aCorner; attribute vec4 aColor; attribute float aSize;
uniform mat4 uView, uProj;
varying vec4 vColor; varying vec2 vP;
void main() {
  vColor = aColor; vP = aCorner;
  vec4 mv = uView * vec4(aPos, 1.0);
  float d = min(aSize * 0.926, 1.04 * max(-mv.z, 0.5));
  mv.xy += aCorner * (0.5 * d);
  gl_Position = uProj * mv;
}
)";
static const char* PART_FS = R"(
varying vec4 vColor; varying vec2 vP;
void main() {
  float r = dot(vP, vP);
  if (r > 1.0) discard;
  float a = (1.0 - r); a *= a;
  gl_FragColor = outColor(vColor.rgb, vColor.a * a);
}
)";

void FX::Particles::init(int count) {
  n = count; i = 0;
  pos.assign((size_t)n * 3, 0); vel.assign((size_t)n * 3, 0); col.assign((size_t)n * 4, 0);
  for (auto* v : {&size, &life, &max, &grow, &drag, &grav, &a0}) v->assign(n, 0);
}
void FX::Particles::spawn(const V3& p, const V3& v, const V3& c, double a, double sz, double lf, double gr, double dr, double gv) {
  int k = i; i = (i + 1) % n;
  for (int q = 0; q < 3; q++) { pos[k * 3 + q] = (float)p[q]; vel[k * 3 + q] = (float)v[q]; col[k * 4 + q] = (float)c[q]; }
  col[k * 4 + 3] = (float)a;
  size[k] = (float)sz; life[k] = (float)lf; max[k] = (float)lf; grow[k] = (float)gr; drag[k] = (float)dr; grav[k] = (float)gv; a0[k] = (float)a;
}
void FX::Particles::update(double dt, const V3& wind) {
  for (int k = 0; k < n; k++) {
    if (life[k] <= 0) { col[k * 4 + 3] = 0; continue; }
    life[k] -= (float)dt;
    float e = (float)std::exp(-drag[k] * dt); int j = k * 3;
    vel[j] = (float)(wind[0] + (vel[j] - wind[0]) * e); vel[j + 1] = (float)(vel[j + 1] * e - grav[k] * dt); vel[j + 2] = (float)(wind[2] + (vel[j + 2] - wind[2]) * e);
    pos[j] += (float)(vel[j] * dt); pos[j + 1] += (float)(vel[j + 1] * dt); pos[j + 2] += (float)(vel[j + 2] * dt);
    size[k] += (float)(grow[k] * dt);
    float t = std::max(0.0f, life[k] / max[k]);
    col[k * 4 + 3] = a0[k] * std::min(1.0f, t * 1.6f) * std::min(1.0f, (1 - t) * 12);
  }
}

void FX::init() {
  soft_.init(2600);
  const double R[9] = {0.074, 0.074, 0.074, 0.074, 0.12, 0.05, 0.05, 0.05, 0.05};
  for (int i = 0; i < 9; i++) plumes_[i].k = R[i] / 0.078;

  plumeProg_.build(PLUME_VS, GLSL_COMMON + PLUME_FS);
  const int SEG = 16, ROWS = 12;
  std::vector<float> tv; std::vector<uint16_t> ti;
  for (int j = 0; j <= ROWS; j++) for (int i = 0; i <= SEG; i++) { double a = i / (double)SEG * 2 * PI; tv.insert(tv.end(), {(float)j / ROWS, (float)std::cos(a), (float)std::sin(a)}); }
  for (int j = 0; j < ROWS; j++) for (int i = 0; i < SEG; i++) {
    uint16_t a = (uint16_t)(j * (SEG + 1) + i), b = a + 1, c = (uint16_t)(a + SEG + 1), d = c + 1;
    ti.insert(ti.end(), {a, c, b, b, c, d});
  }
  tubeCount_ = (int)ti.size();
  glGenBuffers(1, &tubeVbo_); glBindBuffer(GL_ARRAY_BUFFER, tubeVbo_); glBufferData(GL_ARRAY_BUFFER, tv.size() * 4, tv.data(), GL_STATIC_DRAW);
  glGenBuffers(1, &tubeIbo_); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tubeIbo_); glBufferData(GL_ELEMENT_ARRAY_BUFFER, ti.size() * 2, ti.data(), GL_STATIC_DRAW);

  partProg_.build(PART_VS, GLSL_COMMON + PART_FS);
  std::vector<uint16_t> qi; qi.reserve((size_t)soft_.n * 6);
  for (int q = 0; q < soft_.n; q++) { uint16_t a = (uint16_t)(q * 4); qi.insert(qi.end(), {a, (uint16_t)(a + 1), (uint16_t)(a + 2), a, (uint16_t)(a + 2), (uint16_t)(a + 3)}); }
  glGenBuffers(1, &partIbo_); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, partIbo_); glBufferData(GL_ELEMENT_ARRAY_BUFFER, qi.size() * 2, qi.data(), GL_STATIC_DRAW);
  glGenBuffers(1, &partVbo_);
}

void FX::update(double dt, const Bike& b, const V3& wind, bool alive) {
  double wash = 0, wx = 0, wz = 0;
  const double ab = alive ? b.abOut : 0, sw = std::max(b.tilt, PI / 2);
  for (int i = 0; i < 5; i++) {
    // thrust of this nozzle, its exit and the direction its gas leaves in (body frame)
    const bool rear = i == 4;
    const double T = rear ? b.R : b.L[i];
    const double pw = alive ? std::min(1.0, rear ? b.R / (1 + ENG.abGain * b.eng.ab) / (ENG.tmax * 0.75) : T / 1500) : 0;
    Plume& pl = plumes_[i];
    pl.pow = (float)pw; pl.ab = (float)(rear ? ab * 0.55 : 0);
    pl.len = (float)(rear ? 0.6 + 1.5 * pw + 2.2 * ab : 0.35 + 1.5 * pw);
    pl.wide = (float)(1 + (rear ? 0.2 * ab : 0));
    pl.on = alive && pw > 0.02;
    if (!alive) continue;
    // where does this jet hit the surface?
    V3 dirB, exitD;
    if (rear) { double a = REAR.cant + b.rp, cy = std::cos(b.ry); dirB = {-std::cos(a) * cy, -std::sin(a) * cy, -std::sin(b.ry)}; exitD = {REAR.p[0], REAR.p[1], REAR.p[2]}; }
    else { double th = sw + b.va[i]; dirB = {-std::cos(th), -std::sin(th), 0}; exitD = {LIFT[i].p[0], LIFT[i].p[1], LIFT[i].p[2]}; }
    V3 dW = qrot(b.q, dirB);
    if (dW[1] > -0.2) continue;
    V3 exitW = b.toWorld(exitD);
    const double reach = 5 + 13 * pw;
    bool found = false, isWater = false; V3 hit{0, 0, 0}; double hitD = 0;
    for (double s = 0.5; s < reach; s += 0.8) {
      V3 q = madd(exitW, dW, s);
      double gy = groundAt(q[0], q[2]), wl = waterAt(q[0], q[2]), sy = std::max(gy, wl);
      if (q[1] <= sy) { hit = {q[0], sy, q[2]}; isWater = wl > gy; hitD = s; found = true; break; }
    }
    if (!found) continue;
    const double k = std::max(0.0, T / 1450 - 0.12) * (1 - hitD / reach);
    if (isWater) { wash += k; wx += hit[0] * k; wz += hit[2] * k; }
    acc_ += k * dt * 70;
    while (acc_ > 1) {
      acc_ -= 1;
      double a = rnd() * 6.283, sp = (6 + 16 * rnd()) * (0.4 + k);
      V3 v = {std::cos(a) * sp + b.v[0] * 0.3, 1.5 + 3.5 * rnd() * k, std::sin(a) * sp + b.v[2] * 0.3};
      V3 p = {hit[0] + std::cos(a) * 0.4, hit[1] + 0.15, hit[2] + std::sin(a) * 0.4};
      if (isWater) { double vy = 3 + 7 * rnd() * k, sz = 0.5 + rnd() * 0.5, lf = 0.7 + rnd() * 0.5; soft_.spawn(p, {v[0] * 0.8, vy, v[2] * 0.8}, {0.82, 0.88, 0.92}, 0.42, sz, lf, 2.8, 1.6, 7); }
      else if (hit[1] > 2380) { double sz = 0.6 + rnd() * 0.6, lf = 1.0 + rnd() * 0.9; soft_.spawn(p, v, {0.93, 0.95, 0.98}, 0.5, sz, lf, 3.2, 1.3, 1.2); }
      else { double c = 0.34 + 0.14 * rnd(), sz = 0.7 + rnd() * 0.7, lf = 1.2 + rnd() * 1.4; soft_.spawn(p, v, {c, c * 0.86, c * 0.64}, 0.34, sz, lf, 3.4, 1.1, 0.4); }
    }
  }
  // side nozzles: short plumes straight out, violet when the rider kicks
  for (int i = 5; i < 9; i++) {
    const int pair = (i - 5) / 2; const double sgn = (i - 5) % 2 ? 1 : -1;
    Plume& pl = plumes_[i];
    const double pw = alive ? std::min(1.0, std::max(0.0, -sgn * b.sd[pair]) / (0.7 * SIDE_MAX)) : 0;
    pl.pow = (float)std::min(1.0, pw * 1.6); pl.ab = (float)(b.kick * pw * 0.45);
    pl.len = (float)(0.2 + 0.9 * pw + 0.8 * b.kick * pw); pl.wide = (float)(1 + 0.25 * b.kick);
    pl.on = pw > 0.04;
  }
  washStrength += (float)(((wash > 0 ? std::min(1.0, wash * 0.8) : 0.0) - washStrength) * std::min(1.0, dt * 4));
  if (wash > 0) { washPos[0] = (float)(wx / wash); washPos[1] = (float)(wz / wash); }
  // vapour in hard, fast turns (the web build still sheds it where the tips of the wings were)
  if (alive) {
    double g = len(V3{b.acc[0], b.acc[1] + G, b.acc[2]}) / G;
    if (g > 2.0 && b.airspeed > 45) for (double z : {-1.27, 1.27}) if (rnd() < dt * 90) soft_.spawn(b.toWorld({-0.2, 0.86, z}), scale(b.v, 0.05), {0.95, 0.96, 1.0}, 0.16 * std::min(1.0, g - 1.9), 0.22, 0.55, 0.9, 0.5, 0);
  }
  // display smoke (aerobatic mode): a white trail that hangs where the bike has been
  if (alive && b.smoke) {
    double V = len(b.v); V3 ex = b.toWorld({REAR.p[0] - 0.6, REAR.p[1], 0}), bk = qrot(b.q, {-1, 0, 0});
    smokeAcc_ += dt * std::max(26.0, V / 0.75);
    while (smokeAcc_ > 1) {
      smokeAcc_ -= 1;
      double j = rnd() * dt;
      V3 p, v;
      for (int q = 0; q < 3; q++) { p[q] = ex[q] - b.v[q] * j + (rnd() - 0.5) * 0.25; }
      for (int q = 0; q < 3; q++) { v[q] = bk[q] * 5 + (rnd() - 0.5) * 1.5; }
      double sz = 0.5 + 0.3 * rnd(), lf = 6.5 + 2 * rnd();
      soft_.spawn(p, v, {0.97, 0.97, 1.0}, 0.46, sz, lf, 1.25, 1.8, -0.12);
    }
  }
  soft_.update(dt, wind);
}

static Mat4 rotZ4(double a) { Mat4 m = mat4Identity(); float c = (float)std::cos(a), s = (float)std::sin(a); m[0] = c; m[1] = s; m[4] = -s; m[5] = c; return m; }
static Mat4 rotY4(double a) { Mat4 m = mat4Identity(); float c = (float)std::cos(a), s = (float)std::sin(a); m[0] = c; m[2] = -s; m[8] = s; m[10] = c; return m; }

// dust, spray, vapour: the particles that are alive, as quads
void FX::drawParticles(const Camera& cam) {
  glEnable(GL_BLEND); glDepthMask(GL_FALSE); glDisable(GL_CULL_FACE);
  quads_.clear();
  static const float CORNER[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
  int live = 0;
  for (int k = 0; k < soft_.n; k++) {
    if (soft_.life[k] <= 0 || soft_.col[k * 4 + 3] <= 0) continue;
    for (int c = 0; c < 4; c++)
      quads_.insert(quads_.end(), {soft_.pos[k * 3], soft_.pos[k * 3 + 1], soft_.pos[k * 3 + 2], CORNER[c][0], CORNER[c][1],
                                   soft_.col[k * 4], soft_.col[k * 4 + 1], soft_.col[k * 4 + 2], soft_.col[k * 4 + 3], soft_.size[k]});
    live++;
  }
  if (live) {
    Prog& p = partProg_;
    p.use(); gEnv.apply(p, cam);
    p.m4("uView", cam.view); p.m4("uProj", cam.proj);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
    glBindBuffer(GL_ARRAY_BUFFER, partVbo_); glBufferData(GL_ARRAY_BUFFER, quads_.size() * 4, quads_.data(), GL_STREAM_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, partIbo_);
    p.attr("aPos", 3, GL_FLOAT, false, 40, 0); p.attr("aCorner", 2, GL_FLOAT, false, 40, 12);
    p.attr("aColor", 4, GL_FLOAT, false, 40, 20); p.attr("aSize", 1, GL_FLOAT, false, 40, 36);
    glDrawElements(GL_TRIANGLES, live * 6, GL_UNSIGNED_SHORT, (void*)0);
    p.attrOff("aPos"); p.attrOff("aCorner"); p.attrOff("aColor"); p.attrOff("aSize");
  }
  glEnable(GL_CULL_FACE); glDepthMask(GL_TRUE); glDisable(GL_BLEND);
}

// The plumes: light added over everything, each in the frame of its nozzle. With the post pass
// they are drawn into the buffer of added light (packed as a root, where "screen" blending is the
// nearest thing to a sum and never clips a colour to white); without it, straight onto the screen.
void FX::drawPlumes(const Camera& cam, const Bike& b, bool lightBuffer) {
  glEnable(GL_BLEND); glDepthMask(GL_FALSE); glDisable(GL_CULL_FACE);
  bool any = false;
  for (const Plume& pl : plumes_) any = any || pl.on;
  if (any) {
    Prog& p = plumeProg_;
    p.use(); gEnv.apply(p, cam);
    p.m4("uVP", cam.vp);
    if (lightBuffer) glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_COLOR); else glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ZERO, GL_ONE);
    glBindBuffer(GL_ARRAY_BUFFER, tubeVbo_); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tubeIbo_);
    p.attr("aTube", 3, GL_FLOAT, false, 12, 0);
    // design frame -> world, as the bike is drawn
    const Mat4 body = mat4Mul(mat4FromPose(b.q, b.p), mat4Translate(scale(b.mp.cg, -1)));
    const double HUBY = 0.34, sweep = std::max(b.tilt, PI / 2) - PI / 2;
    for (int i = 0; i < 9; i++) {
      const Plume& pl = plumes_[i];
      if (!pl.on) continue;
      Mat4 noz;
      if (i < 4) {
        // a lift engine swings about its hub; its pipe points straight down from there
        noz = mat4Mul(mat4Mul(mat4Translate({LIFT[i].p[0], HUBY, LIFT[i].p[2]}), rotZ4(sweep + b.va[i])), mat4Mul(mat4Translate({0, LIFT[i].p[1] - HUBY, 0}), rotZ4(PI / 2)));
      } else if (i == 4) {
        // the rear nozzle swivels a quarter of a metre ahead of its exit
        noz = mat4Mul(mat4Mul(mat4Translate({REAR.p[0] + 0.25, REAR.p[1], 0}), mat4Mul(rotY4(-b.ry), rotZ4(REAR.cant + b.rp))), mat4Translate({-0.25, 0, 0}));
      } else {
        const int pair = (i - 5) / 2; const double sgn = (i - 5) % 2 ? 1 : -1;
        noz = mat4Mul(mat4Mul(mat4Translate({SIDE[pair].p[0], SIDE[pair].p[1], sgn * SIDE[pair].p[2]}), rotY4(sgn * PI / 2)), mat4Translate({-0.05, 0, 0}));
      }
      p.m4("uNoz", mat4Mul(body, noz));
      p.f3("uDim", pl.len, pl.wide, (float)pl.k);
      p.f1("uPow", pl.pow); p.f1("uAb", pl.ab); p.f1("uSeed", (float)i);
      glDrawElements(GL_TRIANGLES, tubeCount_, GL_UNSIGNED_SHORT, (void*)0);
    }
    p.attrOff("aTube");
  }
  glEnable(GL_CULL_FACE); glDepthMask(GL_TRUE); glDisable(GL_BLEND);
}

} // namespace fohn
