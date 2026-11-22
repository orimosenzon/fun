#include "bike.hpp"
#include "terrain.hpp"
#include <algorithm>

namespace fohn {

const Eng ENG = Eng{};

const Vec3c LIFT[4] = {
  {{0.74, 0.22, -0.20}},   // FL
  {{0.74, 0.22, 0.20}},    // FR
  {{-0.70, 0.22, -0.20}},  // RL
  {{-0.70, 0.22, 0.20}},   // RR
};
Rear REAR = {{-1.27, 0.76, 0}, 0};
const Vec3c SIDE[2] = {{{0.93, 0.86, 0.17}}, {{-0.96, 0.86, 0.19}}};
const double INTAKE[3] = {0.95, 0.72, 0};

const double TILT_MAX = 116 * D2R, VANE_MAX = 12 * D2R, VEC_MAX = 15 * D2R;
const Lim LIM = {25 * D2R, 22 * D2R, 25 * D2R};

// ---------------- mass ----------------
// [mass kg, centre (design frame), box size x,y,z]
struct Item { double m; double c[3]; double s[3]; };
static const Item FIXED[] = {
  {140, {0.0, 0.60, 0}, {0.9, 0.5, 0.4}},
  {14, {0.0, 0.40, 0}, {1.5, 0.2, 0.4}},
  {26, {0.0, 0.75, 0}, {1.6, 0.2, 0.28}},
  {12, {0.62, 0.95, 0}, {0.6, 0.4, 0.4}},
  {12, {-0.55, 0.85, 0}, {0.7, 0.2, 0.28}},
  {6, {0, 0.12, 0}, {1.2, 0.2, 0.6}},
  {37, {-0.18, 1.12, 0}, {0.50, 0.40, 0.40}},
  {5.5, {0.10, 1.40, 0}, {0.26, 0.28, 0.24}},
  {27, {-0.08, 0.74, 0}, {0.60, 0.30, 0.52}},
  {8.5, {0.25, 1.12, 0}, {0.50, 0.14, 0.60}},
  {6, {-0.30, 0.86, 0}, {0.30, 0.15, 0.36}},
};
static const double TANK_C[3] = {0.18, 0.92, 0}, TANK_S[3] = {0.5, 0.25, 0.3};

MassProps massProps(double fuel) {
  Item items[12];
  int nI = 0;
  for (const auto& it : FIXED) items[nI++] = it;
  items[nI++] = Item{fuel, {TANK_C[0], TANK_C[1], TANK_C[2]}, {TANK_S[0], TANK_S[1], TANK_S[2]}};
  double m = 0; V3 cg{0, 0, 0};
  for (int i = 0; i < nI; i++) { double mi = items[i].m; m += mi; cg[0] += mi * items[i].c[0]; cg[1] += mi * items[i].c[1]; cg[2] += mi * items[i].c[2]; }
  cg[0] /= m; cg[1] /= m; cg[2] /= m;
  M3 I{0, 0, 0, 0, 0, 0, 0, 0, 0};
  for (int k = 0; k < nI; k++) {
    double mi = items[k].m, a = items[k].s[0], b = items[k].s[1], d = items[k].s[2];
    double x = items[k].c[0] - cg[0], y = items[k].c[1] - cg[1], z = items[k].c[2] - cg[2];
    I[0] += mi * (b * b + d * d) / 12 + mi * (y * y + z * z);
    I[4] += mi * (a * a + d * d) / 12 + mi * (x * x + z * z);
    I[8] += mi * (a * a + b * b) / 12 + mi * (x * x + y * y);
    I[1] -= mi * x * y; I[2] -= mi * x * z; I[5] -= mi * y * z;
  }
  I[3] = I[1]; I[6] = I[2]; I[7] = I[5];
  return MassProps{m, cg, I, m3inv(I)};
}

// the rear thrust line passes through the centre of gravity
static bool initRearCant() { V3 c = massProps(FUEL0).cg; REAR.cant = std::atan2(c[1] - REAR.p[1], c[0] - REAR.p[0]); return true; }
static bool _rearInit = initRearCant();

// ---------------- aerodynamics ----------------
// No wing, no fin (SURF empty). Bluff body of bike+rider per axis with separate centres of pressure.
struct Body { double cda[3]; double cp[3][3]; double brakeCda; double brakeCp[3]; };
static const Body BODY = {
  {0.40, 1.05, 0.95},
  {{0.05, 0.98, 0}, {-0.10, 0.85, 0}, {-0.22, 0.92, 0}},
  0.6,
  {-0.45, 0.95, 0},
};

Wrench aeroWrench(const V3& vB, const V3& w, double rho, const AeroCtl& ctl, const V3& cg) {
  (void)w;
  Wrench out;
  double Fx = 0, Fy = 0, Fz = 0, Tx = 0, Ty = 0, Tz = 0;
  // (SURF is empty: no strips)
  double V = std::sqrt(vB[0] * vB[0] + vB[1] * vB[1] + vB[2] * vB[2]);
  for (int k = 0; k < 3; k++) {
    double f = -0.5 * rho * BODY.cda[k] * V * vB[k];
    const double* r = BODY.cp[k];
    double rx = r[0] - cg[0], ry = r[1] - cg[1], rz = r[2] - cg[2];
    double fx = k == 0 ? f : 0, fy = k == 1 ? f : 0, fz = k == 2 ? f : 0;
    Fx += fx; Fy += fy; Fz += fz;
    Tx += ry * fz - rz * fy; Ty += rz * fx - rx * fz; Tz += rx * fy - ry * fx;
  }
  if (ctl.brake > 0.001) {
    double k = -0.5 * rho * BODY.brakeCda * ctl.brake * V;
    double fx = k * vB[0], fy = k * vB[1], fz = k * vB[2];
    const double* r = BODY.brakeCp;
    double rx = r[0] - cg[0], ry = r[1] - cg[1], rz = r[2] - cg[2];
    Fx += fx; Fy += fy; Fz += fz;
    Tx += ry * fz - rz * fy; Ty += rz * fx - rx * fz; Tz += rx * fy - ry * fx;
  }
  double kd = rho / RHO0 * (1 + V / 30);
  Tx -= 6 * kd * w[0]; Ty -= 9 * kd * w[1]; Tz -= 9 * kd * w[2];
  out.F = {Fx, Fy, Fz}; out.Tq = {Tx, Ty, Tz}; out.stall = 0;
  return out;
}

Wrench propWrench(const Noz& n, const V3& cg) {
  Wrench out;
  double Fx = 0, Fy = 0, Fz = 0, Tx = 0, Ty = 0, Tz = 0;
  double sw = std::max(n.tilt, PI / 2);
  for (int i = 0; i < 4; i++) {
    double a = n.va[i], th = sw + a;
    double t = n.L[i] * (1 - 0.3 * a * a);
    double fx = std::cos(th) * t, fy = std::sin(th) * t;
    const double* p = LIFT[i].p;
    double rx = p[0] - cg[0], ry = p[1] - cg[1], rz = p[2] - cg[2];
    Fx += fx; Fy += fy;
    Tx += -rz * fy; Ty += rz * fx; Tz += rx * fy - ry * fx;
  }
  {
    double a = REAR.cant + n.rp, cy = std::cos(n.ry);
    double t = n.R * (1 - 0.3 * (n.rp * n.rp + n.ry * n.ry));
    double fx = std::cos(a) * cy * t, fy = std::sin(a) * cy * t, fz = std::sin(n.ry) * t;
    const double* p = REAR.p;
    double rx = p[0] - cg[0], ry = p[1] - cg[1], rz = p[2] - cg[2];
    Fx += fx; Fy += fy; Fz += fz;
    Tx += ry * fz - rz * fy; Ty += rz * fx - rx * fz; Tz += rx * fy - ry * fx;
  }
  for (int i = 0; i < 2; i++) {
    double fz = n.sd[i]; const double* p = SIDE[i].p;
    double rx = p[0] - cg[0], ry = p[1] - cg[1];
    Fz += fz; Tx += ry * fz; Ty -= rx * fz;
  }
  out.F = {Fx, Fy, Fz}; out.Tq = {Tx, Ty, Tz};
  return out;
}

Wrench ramWrench(const V3& vB, const V3& w, double Tdry, const V3& cg) {
  Wrench out;
  double md = Tdry / ENG.ve;
  double rx = INTAKE[0] - cg[0], ry = INTAKE[1] - cg[1], rz = INTAKE[2] - cg[2];
  double fx = -md * (vB[0] + w[1] * rz - w[2] * ry);
  double fy = -md * (vB[1] + w[2] * rx - w[0] * rz);
  double fz = -md * (vB[2] + w[0] * ry - w[1] * rx);
  out.F = {fx, fy, fz};
  out.Tq = {ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx};
  return out;
}

// ---------------- turbofan ----------------
void Turbofan::update(double dt, double Gcmd, double abCmd, double sigma) {
  double N0 = N;
  if (on) {
    ab += clamp((abCmd - ab) / 0.12, -6, 5) * dt;
    double Nc = clamp(nOfThrust(Gcmd, sigma), ENG.idleN, 1);
    double d = clamp((Nc - N) / 0.16, -1.1, 0.3 + 0.55 * N);
    N = clamp(N + d * dt, ENG.idleN, 1);
  } else {
    ab = 0;
    N = std::max(0.0, N - (N * 0.35 + 0.004) * dt);
  }
  dNdt = (N - N0) / dt;
  Tdry = on ? thrustOfN(N, sigma) : 0;
}

// ---------------- geometry ----------------
const V3 SKIDS[4] = {{0.50, 0, -0.42}, {0.50, 0, 0.42}, {-0.52, 0, -0.42}, {-0.52, 0, 0.42}};
const V3 PROBES[8] = {
  {1.10, 0.80, 0}, {-1.3, 0.76, 0}, {0.12, 1.58, 0}, {-1.0, 1.15, 0},
  {0.50, 1.04, -0.36}, {0.50, 1.04, 0.36},
  {0.74, 0.2, 0}, {-0.70, 0.2, 0},
};

static inline double servo(double x, double target, double lag, double rate, double dt) {
  return x + clamp((target - x) / lag, -rate, rate) * dt;
}

void Bike::reset(const V3& pos, double yaw, double speed, double fuelKg) {
  fuel = fuelKg;
  mp = massProps(fuel); _mpAge = 0;
  p = {pos[0], pos[1] + (speed > 0 ? 0 : mp.cg[1] - 0.01), pos[2]};
  q = {std::cos(yaw / 2), 0, std::sin(yaw / 2), 0};
  v = {std::cos(yaw) * speed, 0, -std::sin(yaw) * speed};
  w = {0, 0, 0};
  tilt = speed > 40 ? 1.0 : PI / 2;
  va[0] = va[1] = va[2] = va[3] = 0;
  sh[0] = sh[1] = sh[2] = sh[3] = 0.25;
  rp = ry = 0;
  sd[0] = sd[1] = 0; sdv[0] = sdv[1] = 0; kick = 0;
  L[0] = L[1] = L[2] = L[3] = 0; R = 0; abOut = 0;
  da = de = dr = brake = 0;
  smoke = false;
  heat = 0; abLock = false;
  t = 0;
  HprevSet = false; Hprev = {0, 0, 0};
  contactN = 0; skidF[0] = skidF[1] = skidF[2] = skidF[3] = 0;
  acc = {0, 0, 0}; crash = nullptr; scrape = 0;
  airspeed = speed; vAirB = {speed, 0, 0}; sigma = 1; rho = RHO0; stall = 0;
  gfx[0] = gfx[1] = gfx[2] = gfx[3] = 1;
  agl = 0; groundY = pos[1]; overWater = false;
  _gAge = 0;
  eng.on = true; eng.ab = 0; eng.N = speed > 0 ? 0.66 : ENG.idleN;
  eng.update(1e-3, 0, 0, 1);
}

void Bike::step(double dt, const Cmd& cmd, const V3& windW) {
  if (_mpAge-- <= 0) { mp = massProps(fuel); _mpAge = 30; }
  V3 cg = mp.cg; V4 q = this->q; V3 w = this->w;
  if (!(_gAge-- > 0)) {
    double g = groundAt(p[0], p[2]), wl = waterAt(p[0], p[2]);
    overWater = wl > g; groundY = std::max(g, wl); _gAge = 4;
    agl = p[1] - cg[1] - groundY;
  }
  double rho = airDensity(p[1]), sigma = rho / RHO0;

  // --- actuators ---
  tilt = servo(tilt, clamp(cmd.tilt, 0, TILT_MAX), 0.08, 1.7, dt);
  const double* cs = cmd.s;
  double csum = std::max(cs[0], 0.0) + std::max(cs[1], 0.0) + std::max(cs[2], 0.0) + std::max(cs[3], 0.0);
  double ssum = 0;
  for (int i = 0; i < 4; i++) {
    va[i] = servo(va[i], clamp(cmd.va[i], -VANE_MAX, VANE_MAX), 0.03, 3.2, dt);
    double want = csum > 1e-6 ? clamp(std::max(cs[i], 0.0) / csum, SHARE_MIN, SHARE_MAX) : 0.25;
    sh[i] = servo(sh[i], want, 0.035, 5, dt); ssum += sh[i];
  }
  for (int i = 0; i < 4; i++) sh[i] /= ssum;
  rp = servo(rp, clamp(cmd.rp, -VEC_MAX, VEC_MAX), 0.04, 2.5, dt);
  ry = servo(ry, clamp(cmd.ry, -VEC_MAX, VEC_MAX), 0.04, 2.5, dt);
  da = servo(da, clamp(cmd.da, -LIM.da, LIM.da), 0.04, 2.5, dt);
  de = servo(de, clamp(cmd.de, -LIM.de, LIM.de), 0.04, 2.5, dt);
  dr = servo(dr, clamp(cmd.dr, -LIM.dr, LIM.dr), 0.04, 2.5, dt);
  brake = servo(brake, clamp(cmd.brake, 0, 1), 0.12, 4, dt);

  // --- after-burner thermal budget ---
  double abWanted = (cmd.ab > 0.5 && !abLock && fuel > 0) ? 1 : 0;
  double abNow = eng.ab;
  kick = servo(kick, abLock ? 0 : clamp(cmd.kick, 0, 1), 0.05, 12, dt);
  double hot = std::max(abNow, 0.0), kickNow = kick * clamp((std::fabs(sd[0]) + std::fabs(sd[1])) / 1500, 0, 1);
  heat = clamp(heat + (hot * 0.15 + kickNow * 0.24 - (1 - std::max(hot, kickNow)) * (0.055 + 0.06 * std::min(1.0, airspeed / 80))) * dt, 0, 1);
  if (heat >= 1) abLock = true; else if (heat < 0.45) abLock = false;

  // --- engine and gas distribution ---
  eng.update(dt, cmd.G, abWanted, sigma);
  double sab = 1 + SIDE_AB * kick;
  const double* csd = cmd.sd;
  for (int i = 0; i < 2; i++) sdv[i] = servo(sdv[i], clamp(csd[i], -SIDE_MAX, SIDE_MAX), 0.03, 90000, dt);
  double gasWant = (std::fabs(sdv[0]) + std::fabs(sdv[1])) / sab, gasCap = SIDE_BLEED * eng.Tdry;
  double ks = gasWant > gasCap ? gasCap / gasWant : 1, gasS = gasWant * ks;
  sd[0] = sdv[0] * ks; sd[1] = sdv[1] * ks;
  double Tmain = eng.Tdry - gasS;
  double fl = liftFrac(tilt), s0 = std::sin(std::max(tilt, PI / 2));
  double gasR = Tmain * (1 - fl), gasL = Tmain * fl * ENG.ductEta;
  R = gasR * (1 + ENG.abGain * eng.ab);
  abOut = eng.ab * clamp(gasR / (0.12 * ENG.tmax), 0, 1);
  for (int i = 0; i < 4; i++) {
    double hn = std::max(0.0, agl + LIFT[i].p[1]);
    gfx[i] = 1 - (0.05 * std::exp(-hn / 0.8) + 0.03 * std::exp(-hn / 2.4)) * s0;
    L[i] = gasL * sh[i] * gfx[i];
  }
  double flow = 0;
  V3 H = {ENG.rotorI * ENG.omegaMax * eng.N, 0, 0};

  // --- air data ---
  V3 vAirW = sub(v, windW);
  V3 vB = qinvrot(q, vAirW);
  vAirB = vB; airspeed = len(vB); this->sigma = sigma; this->rho = rho;

  // --- forces (body frame) ---
  AeroCtl ac{da, de, dr, brake};
  Wrench A = aeroWrench(vB, w, rho, ac, cg);
  Noz noz; noz.tilt = tilt; noz.R = R; noz.rp = rp; noz.ry = ry;
  for (int i = 0; i < 4; i++) { noz.L[i] = L[i]; noz.va[i] = va[i]; }
  noz.sd[0] = sd[0]; noz.sd[1] = sd[1];
  Wrench P = propWrench(noz, cg);
  Wrench Rw = ramWrench(vB, w, eng.Tdry, cg);
  stall = airspeed > 18 ? A.stall : 0;
  V3 Fb = {A.F[0] + P.F[0] + Rw.F[0], A.F[1] + P.F[1] + Rw.F[1], A.F[2] + P.F[2] + Rw.F[2]};
  V3 Tq = {A.Tq[0] + P.Tq[0] + Rw.Tq[0], A.Tq[1] + P.Tq[1] + Rw.Tq[1], A.Tq[2] + P.Tq[2] + Rw.Tq[2]};
  if (HprevSet) Tq = madd(Tq, sub(H, Hprev), -1 / dt);
  Hprev = H; HprevSet = true;

  V3 Fw = qrot(q, Fb);
  Fw[1] -= mp.m * G;

  // --- ground contact ---
  contactN = 0;
  if (agl < 4) {
    for (int i = 0; i < 4; i++) {
      V3 r = arm(SKIDS[i]);
      V3 pw = add(p, qrot(q, r));
      double gy = groundAt(pw[0], pw[2]);
      double pen = gy - pw[1];
      skidF[i] = 0;
      if (waterAt(pw[0], pw[2]) - 0.15 > std::max(pw[1], gy)) { if (!crash) crash = "water"; continue; }
      if (pen <= 0) continue;
      V3 nrm = normalAt(pw[0], pw[2]);
      V3 vp = add(v, qrot(q, cross(w, r)));
      double vn = dot(vp, nrm);
      if (vn < -6.5) { if (!crash) crash = "hard"; }
      double Fn = std::max(0.0, 60000 * pen - 3300 * vn);
      V3 vt = madd(vp, nrm, -vn); double vtl = len(vt);
      if (vtl > 34) { if (!crash) crash = "ground"; }
      double mu = 0.6 * std::min(1.0, vtl / 0.04);
      V3 fW = vtl > 1e-6 ? madd(scale(nrm, Fn), vt, -mu * Fn / vtl) : scale(nrm, Fn);
      Fw = add(Fw, fW);
      Tq = add(Tq, cross(r, qinvrot(q, fW)));
      skidF[i] = Fn; contactN += Fn;
    }
    for (int i = 0; i < N_PROBES; i++) {
      V3 r = arm(PROBES[i]);
      V3 pw = add(p, qrot(q, r));
      double gy = groundAt(pw[0], pw[2]), wl = waterAt(pw[0], pw[2]);
      if (wl > gy && pw[1] < wl) { if (!crash) crash = "water"; continue; }
      double pen = gy - pw[1];
      if (pen <= 0) continue;
      V3 nrm = normalAt(pw[0], pw[2]);
      V3 vp = add(v, qrot(q, cross(w, r)));
      double vn = dot(vp, nrm);
      if (vn < -2.2 || len(vp) > 12 || pen > 0.35) { if (!crash) crash = "ground"; continue; }
      V3 fW = scale(nrm, std::max(0.0, 30000 * pen - 2500 * vn));
      Fw = add(Fw, fW); Tq = add(Tq, cross(r, qinvrot(q, fW)));
      scrape += dt;
    }
  }

  // --- integrate ---
  acc = scale(Fw, 1 / mp.m);
  v = madd(v, acc, dt);
  p = madd(p, v, dt);
  V3 wdot = m3v(mp.Iinv, sub(Tq, cross(w, add(m3v(mp.I, w), H))));
  w = madd(w, wdot, dt);
  this->w = w;
  this->q = qnorm(qmul(q, qexp(scale(w, dt))));
  fuel = std::max(0.0, fuel - flow * dt);
  if (fuel <= 0) eng.on = false;
  fuelFlow = flow;
  agl = p[1] - cg[1] - groundY;
  t += dt;
}

} // namespace fohn
