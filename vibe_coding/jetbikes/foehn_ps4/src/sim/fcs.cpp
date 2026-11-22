#include "fcs.hpp"
#include "terrain.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace fohn {

Feel FEEL = Feel{};
const Feel FEEL_DEFAULT = Feel{};

// the nose may lead the flight path by this much in aerobatic mode (rad)
static const double A_LIM = 0.2;
struct AcroK { double rollAcc = 30, sideArm = 0.12, rollSlew = 15, pitchSlew = 7, yawSlew = 5; };
static const AcroK ACRO = AcroK{};
struct NetK { double t0 = 0.35, rate = 2.6, fall = 0.8, pull = 0.75, h0 = 4, lead = 0.5; };
static const NetK NET = NetK{};

double levelYaw(const V4& q) {
  V3 f = qrot(q, {1, 0, 0}), u = qrot(q, {0, 1, 0});
  return std::atan2(-(f[2] - u[2] * f[1]), f[0] - u[0] * f[1]);
}

// row weights: [sum of lift (N), side force (N), Mx, My, Mz (N m)]
static const double WV[NR] = {0.5, 3.0 / 1000, 6.0 / 200, 2.5 / 200, 6.0 / 200};
// effector weights (cost of leaving the preferred position)
static const double WU[NU] = {4e-4, 4e-4, 4e-4, 4e-4, 1.2, 1.2, 1.2, 1.2, 0.3, 0.3, 5e-4, 5e-4};

// Bounded weighted least squares by an active set. Returns du[NU]; sets *satOut.
static std::array<double, NU> wls(const double* J, const double* e, const double* wv, const double* wu,
                                  const double* pref, const double* lo, const double* hi, int* satOut) {
  static double A[NU * NU], bb[NU], res[NR];
  std::array<double, NU> du{};
  int fix[NU];
  for (int i = 0; i < NU; i++) { du[i] = 0; fix[i] = 0; }
  int sat = 0;
  for (int pass = 0; pass < NU; pass++) {
    for (int r = 0; r < NR; r++) { double s = e[r]; for (int c = 0; c < NU; c++) if (fix[c]) s -= J[r * NU + c] * du[c]; res[r] = s; }
    int idx[NU], n = 0;
    for (int c = 0; c < NU; c++) if (!fix[c]) idx[n++] = c;
    if (!n) break;
    for (int a = 0; a < n; a++) {
      int ca = idx[a];
      double rhs = wu[ca] * wu[ca] * pref[ca];
      for (int r = 0; r < NR; r++) rhs += J[r * NU + ca] * wv[r] * wv[r] * res[r];
      bb[a] = rhs;
      for (int c = 0; c < n; c++) {
        int cc = idx[c];
        double s = (a == c) ? wu[ca] * wu[ca] : 0;
        for (int r = 0; r < NR; r++) s += J[r * NU + ca] * wv[r] * wv[r] * J[r * NU + cc];
        A[a * n + c] = s;
      }
    }
    solve(A, bb, n);
    int worst = -1; double over = 1e-9;
    for (int a = 0; a < n; a++) {
      int c = idx[a]; double x = bb[a], span = hi[c] - lo[c] + 1e-9;
      du[c] = x;
      double o = x < lo[c] ? (lo[c] - x) / span : (x > hi[c] ? (x - hi[c]) / span : 0);
      if (o > over) { over = o; worst = c; }
    }
    if (worst < 0) break;
    du[worst] = du[worst] < lo[worst] ? lo[worst] : hi[worst]; fix[worst] = 1; sat++;
  }
  for (int c = 0; c < NU; c++) du[c] = clamp(du[c], lo[c], hi[c]);
  if (satOut) *satOut = sat;
  return du;
}

FCS::FCS(Bike* b) : bike(b) { reset(); }

void FCS::reset() {
  turn = 0; pitch = 0; strafe = 0; hold = 0; rollNeed = 0;
  iV = 0; iZ = 0; yawRefSet = false; tiltCmd = bike->tilt;
  idle = true; downT = 1; low = false;
  acro = false; rec = false; recT = 0; bail = nullptr; qRefSet = false; kickA = 0; wStick = {0, 0, 0};
  tele = Tele{};
  cmd = Cmd{};
  cmd.G = 0; for (int i = 0; i < 4; i++) { cmd.s[i] = 1; cmd.va[i] = 0; }
  cmd.sd[0] = cmd.sd[1] = 0; cmd.kick = 0; cmd.rp = cmd.ry = 0;
  cmd.da = cmd.de = cmd.dr = cmd.brake = cmd.ab = 0; cmd.tilt = bike->tilt;
}

// the least height over the ground (and obstacles): here and where the bike is going, T seconds ahead
double FCS::clearance(double T) {
  Bike* b = bike; const V3& v = b->v;
  bool coarse = len(v) > 12;
  double hBike = b->p[1] - b->mp.cg[1];
  auto top = [&](double x, double z) {
    if (obstacle) return std::max(surfaceAt(x, z), obstacle(x, z, coarse) + 1.5);
    return surfaceAt(x, z);
  };
  double slack = std::max(0.0, 0.25 * std::hypot(v[0], v[2]));
  double h = std::min(std::min(b->agl, hBike - top(b->p[0], b->p[2])),
                      std::min(hBike - top(b->p[0] + v[0] * 0.6, b->p[2] + v[2] * 0.6),
                               hBike - top(b->p[0] + v[0] * 1.3, b->p[2] + v[2] * 1.3) + slack));
  for (double t = 2; t < T + 0.7; t += 0.7) h = std::min(h, hBike - top(b->p[0] + v[0] * t, b->p[2] + v[2] * t) + slack);
  return h;
}

bool FCS::setAcro(bool on, const char* why) {
  Bike* b = bike;
  on = on && b->contactN == 0 && !(idle && len(b->v) < 5);
  if (on) {
    if (acro && !rec) return true;
    if (rec) return false;
    acro = true; bail = nullptr; qRef = b->q; qRefSet = true; kickA = 0; wStick = {0, 0, 0};
    return true;
  }
  if (!acro) return false;
  bool isGround = why && std::strcmp(why, "ground") == 0;
  bool isDone = why && std::strcmp(why, "done") == 0;
  if (!rec && !isGround && !isDone) {
    V3 bx = qrot(b->q, {1, 0, 0}), by = qrot(b->q, {0, 1, 0});
    bail = why;
    if (by[1] < 0.6 || std::fabs(bx[1]) > 0.55 || b->v[1] < -12) { rec = true; recT = 0; return false; }
  }
  acro = rec = false;
  iV = iZ = 0; yawRef = levelYaw(b->q); yawRefSet = true; low = false;
  return false;
}

FCS::AcroOut FCS::acroLaw(double dt, const Input& inp, bool boost) {
  AcroOut out;
  Bike* b = bike; double m = b->mp.m; const V3& v = b->v; double V = len(v);
  V3 bx = qrot(b->q, {1, 0, 0}), by = qrot(b->q, {0, 1, 0}), bz = qrot(b->q, {0, 0, 1});
  double gy = 0.9 * thrustOfN(1, b->sigma) * ENG.ductEta / m;
  V3 ypr = yprOf(b->q); double pt = ypr[1], rl = ypr[2];
  bool steep = std::fabs(pt) > 1.25;
  V3 vI = qinvrot(b->q, v); double u = std::max(vI[0], 0.0), uq = std::max(u, 6.0), fast = sstep(8, 22, V);
  if (FEEL.net && !inp.noFloor && !rec) {
    double sink = -v[1] + NET.lead * std::max(0.0, -b->acc[1]);
    double tRoll = NET.t0 + (steep ? 0 : std::fabs(rl)) / NET.rate;
    double Vh = std::hypot(v[0], v[2]);
    double sink1 = sink + NET.fall * G * tRoll, V1 = std::hypot(Vh, sink1);
    double need = sink1 <= 0 ? 0 : std::max(0.0, sink * tRoll + 0.5 * NET.fall * G * tRoll * tRoll) + V1 * V1 / (NET.pull * gy - G) * (1 - Vh / std::max(V1, 1.0));
    double extra = sink1 > 0 ? V1 * std::acos(std::min(1.0, Vh / std::max(V1, 1.0))) / (NET.pull * gy - G) : 0;
    if (clearance(tRoll + extra + 0.6) < need + NET.h0 + 0.07 * V) setAcro(false, "net");
  }
  double sTurn = inp.turn, sPitch = inp.pitch, sYaw = inp.strafe, vt = inp.vt;
  if (rec) {
    recT += dt;
    if ((by[1] > 0.6 && v[1] > -4 && bx[1] > -0.5 && bx[1] < 0.57) || vI[0] < 6 || recT > 9) { setAcro(false, "done"); out.valid = false; return out; }
    sYaw = 0; boost = false;
    if (bx[1] > 0.87 || (by[1] < 0 && bx[1] > 0.26)) { sTurn = 0; sPitch = 1; }
    else if (by[1] >= 0.6 && bx[1] >= 0.57) { sTurn = clamp(-2.2 * rl, -1, 1); sPitch = -1; vt = std::min(vt, 25.0); }
    else { sTurn = steep ? 0 : clamp(-2.2 * rl, -1, 1); sPitch = (steep || std::fabs(rl) < 1.0) && (v[1] < -2 || bx[1] < -0.05) ? 1 : 0; }
  } else if (!acro) { out.valid = false; return out; }
  double aPos = gy - G * by[1], aNeg = std::max(0.0, G * by[1] - 0.3 * G);
  double alpha = -vI[1] / std::max(V, 10.0), beta = vI[2] / std::max(V, 10.0);
  double qMax = aPos / uq + lerp(9, 2.5 * (A_LIM - alpha), fast), qMin = -aNeg / uq - lerp(9, 2.5 * (A_LIM + alpha), fast);
  auto slew = [&](double x, double to, double a) { return x + clamp(to - x, -a * dt, a * dt); };
  double w0 = wStick[0], w1 = wStick[1], w2 = wStick[2];
  wStick[0] = slew(wStick[0], clamp(sTurn, -1, 1) * FEEL.acroRoll, ACRO.rollSlew);
  wStick[1] = slew(wStick[1], -clamp(sYaw, -1, 1) * FEEL.acroYaw, ACRO.yawSlew);
  wStick[2] = slew(wStick[2], clamp(clamp(sPitch, -1, 1) * FEEL.acroPitch, std::min(qMin, qMax), qMax), ACRO.pitchSlew);
  double pc = wStick[0], qc = wStick[2];
  double rc = wStick[1] - clamp(1.2 * beta, -0.5, 0.5) * fast;
  double det = FEEL.detent * PI / 180;
  if (det > 0 && !rec) {
    V3 ypr2 = yprOf(qRef); double yr = ypr2[0], pr = ypr2[1], rr = ypr2[2], cp = std::cos(pr);
    if (std::fabs(sTurn) < 0.05 && std::fabs(wStick[0]) < 0.6 && cp > 0.3) {
      double e = wrapPi(std::round(rr / (PI / 2)) * (PI / 2) - rr);
      if (std::fabs(e) < 1.7 * det) pc += clamp(4 * e, -0.6, 0.6);
    }
    if (std::fabs(sPitch) < 0.05 && std::fabs(wStick[2]) < 0.3 && std::fabs(sYaw) < 0.05 && cp > 0.02) {
      bool has = std::fabs(pr) < det || (PI / 2 - std::fabs(pr) < det);
      double tgt = std::fabs(pr) < det ? 0 : (PI / 2 - std::fabs(pr) < det ? (pr < 0 ? -1 : 1) * PI / 2 : 0);
      if (has) { double k = clamp(3 * (tgt - pr), -0.35, 0.35); V3 wv = qinvrot(qRef, {std::sin(yr) * k, 0, std::cos(yr) * k}); rc += wv[1]; qc += wv[2]; }
    }
  }
  qc = clamp(qc, std::min(qMin, qMax), qMax);
  qRef = qnorm(qmul(qRef, qexp({pc * dt, rc * dt, qc * dt})));
  V4 qe = qmul(qconj(b->q), qRef);
  if (qe[0] < 0) qe = {-qe[0], -qe[1], -qe[2], -qe[3]};
  double ang = 2 * std::acos(std::min(1.0, qe[0]));
  if (ang > 0.4) { double k = std::sin(0.2) / std::sin(ang / 2); qRef = qnorm(qmul(b->q, {std::cos(0.2), qe[1] * k, qe[2] * k, qe[3] * k})); }
  double eV = vt - vI[0];
  iV = clamp(iV + 0.25 * eV * dt, -1.5, 1.5);
  double aF = boost ? clamp(1.2 * (V_MAX + 30 - vI[0]), -4, 20) : clamp(0.85 * eV + iV, -14, 14);
  double fx = aF + G * bx[1];
  double fy = clamp(clamp(-vI[1] / 0.3, -20, 20) + qc * u + G * by[1], 0, gy);
  double fz = clamp(clamp(-vI[2] / 0.35, -16, 16) - rc * u + G * bz[1], -16, 16);
  kickA += ((b->abLock ? 0 : clamp((m * std::fabs(fz) - 700) / 1200, 0, 1)) - kickA) * std::min(1.0, dt * 8);
  tele.aCmd = {aF, fy - G * by[1], fz - G * bz[1]}; tele.vzDes = v[1]; tele.floor = false; tele.phiDes = yprOf(qRef)[2];
  out.valid = true;
  out.Fb = {m * fx, m * fy, m * fz}; out.qDes = qRef;
  out.wff = {pc, rc, qc}; out.aff = {(wStick[0] - w0) / dt, (wStick[1] - w1) / dt, (wStick[2] - w2) / dt};
  out.eV = eV; out.kickCmd = kickA;
  return out;
}

const Cmd& FCS::update(double dt, const Input& inp) {
  Bike* b = bike; const MassProps& mp = b->mp; double m = mp.m; const V3& cg = mp.cg; double W = m * G;
  double k1 = 1 - std::exp(-dt / FEEL.stick);
  turn += (clamp(inp.turn, -1, 1) - turn) * k1;
  pitch += (clamp(inp.pitch, -1, 1) - pitch) * k1;
  strafe += (clamp(inp.strafe, -1, 1) - strafe) * k1;
  hold = std::fabs(inp.turn) > 0.9 ? std::min(hold + dt, 2.0) : std::max(0.0, hold - 3 * dt);
  double hair = FEEL.hairpin ? sstep(HAIR_T0, HAIR_T1, hold) : 0;
  double vtIn = inp.vt > V_CORNER ? lerp(inp.vt, V_CORNER, hair) : inp.vt;

  V3 ypr = yprOf(b->q); double yaw = ypr[0], pitchA = ypr[1];
  if (std::fabs(pitchA) > 1.2) yaw = levelYaw(b->q);
  V3 fh = {std::cos(yaw), 0, -std::sin(yaw)}, rh = {std::sin(yaw), 0, std::cos(yaw)};
  const V3& v = b->v; double Vf = dot(v, fh), Vr = dot(v, rh), Vu = v[1];
  double Vh = std::hypot(v[0], v[2]), V = len(v), Va = b->airspeed;
  bool onGround = b->contactN > 0.25 * W;
  if (acro && b->contactN > 0) setAcro(false, "ground");
  if (!acro) rec = false;

  bool wantsUp = inp.pitch > 0.05 || std::fabs(inp.vt) > 0.5 || std::fabs(inp.strafe) > 0.05 || inp.boost;
  bool down = (onGround || (b->contactN > 0.06 * W && pitch < 0.02)) && !wantsUp && V < 4;
  downT = down ? downT + dt : 0;
  if (down && (idle || downT > 0.35)) {
    idle = true;
    cmd.G = 0; for (int i = 0; i < 4; i++) { cmd.s[i] = 1; cmd.va[i] = 0; } cmd.sd[0] = cmd.sd[1] = 0;
    cmd.kick = 0; cmd.rp = cmd.ry = 0; cmd.da = cmd.de = cmd.dr = 0; cmd.ab = 0; cmd.brake = 0; cmd.tilt = PI / 2;
    iV = iZ = 0; yawRef = yaw; yawRefSet = true; tiltCmd = PI / 2;
    return cmd;
  }
  idle = false;
  if (!yawRefSet) { yawRef = yaw; yawRefSet = true; }

  bool boost = inp.boost && !b->abLock && !inp.brake;
  AcroOut ac = acro ? acroLaw(dt, inp, boost) : AcroOut{};
  bool acroOn = ac.valid;
  V3 Fb, wff{0, 0, 0}, aff{0, 0, 0}; V4 qDes; double eV = 0, kickCmd = 0;
  if (acroOn) { Fb = ac.Fb; qDes = ac.qDes; wff = ac.wff; aff = ac.aff; eV = ac.eV; kickCmd = ac.kickCmd; }
  else {
    // ----- 1. wanted acceleration -----
    eV = vtIn - Vf;
    iV = clamp(iV + 0.25 * eV * dt, -1.5, 1.5);
    double aF = boost ? clamp(1.2 * (V_MAX + 30 - Vf), -4, 20) : clamp(0.85 * eV + iV, -14, 14);
    double Tavail = thrustOfN(1, b->sigma) * ENG.ductEta;
    double liftCap = 0.9 * Tavail;
    double phiLim = clamp(std::acos(std::min(1.0, W / (0.96 * liftCap))), 0.62, FEEL.bank);
    double rateMax = std::min(FEEL.yawRate, G * std::tan(phiLim) / std::max(Vh, 6.0));
    double yawRateCmd = -turn * rateMax;
    double lowf = 1 - sstep(16, 30, std::fabs(Vf));
    double aR = -Vf * yawRateCmd + lowf * clamp(1.6 * (strafe * FEEL.strafe - Vr), -7, 7);
    double kickIn = b->abLock ? 0 : (1 - lowf) * strafe, aK = kickIn * FEEL.kick;

    double dive = std::max(0.0, -pitch) * sstep(8, 20, Va) * (inp.noDive ? 0 : FEEL.dive);
    double vzDes = pitch * std::max(8.0, lerp(0.62 * V, std::min(1.0 * std::max(Vh, 12.0), std::sqrt(std::max(0.0, V_MAX * V_MAX - Vh * Vh))), dive));
    bool landing = pitch < -0.2 && std::min(Va, V) < 16;
    double ax = b->p[0] + v[0] * 0.6, az = b->p[2] + v[2] * 0.6;
    bool coarse = V > 12;
    auto top = [&](double x, double z) {
      if (obstacle) return std::max(surfaceAt(x, z), obstacle(x, z, coarse) + 1.5);
      return surfaceAt(x, z);
    };
    double ax2 = b->p[0] + v[0] * 1.3, az2 = b->p[2] + v[2] * 1.3;
    double hBike = b->p[1] - cg[1];
    double aglAhead = std::min(hBike - top(ax, az), hBike - top(ax2, az2) + std::max(0.0, 0.25 * Vh));
    double hHere = obstacle ? std::min(b->agl, hBike - top(b->p[0], b->p[2])) : b->agl;
    double hmin = std::min(hHere, aglAhead), floorH = 1.6 + 0.07 * Va;
    tele.floor = false;
    if (landing && hmin < floorH + 0.5) low = true;
    else if (inp.pitch > 0.05 || inp.boost || V > 12 || hmin > floorH + 4) low = false;
    if (!landing && !inp.noFloor) {
      double room = hmin - (low ? 0.6 : floorH), roomL = std::max(0.0, room + 0.25 * std::min(0.0, Vu));
      bool elev = !inp.noDive && FEEL.dive > 0;
      double dn = elev ? std::max(0.0, std::sin(-pitchA)) : 0, aArr = 14 * (1 - 0.6 * dn), roomD = std::max(0.0, roomL + 1.2 * dn * std::min(0.0, Vu));
      double f = room < 0 ? -2.2 * room : -std::min(1.5 * room, lerp(std::sqrt(2 * aArr * roomD), 1.5 * room, sstep(70, 100, elev ? std::hypot(Vh, std::min(V, 0.62 * Vh)) : V))) - 0.5;
      if (f > vzDes) { vzDes = f; tele.floor = true; }
    }
    if (onGround && pitch <= 0.05) vzDes = std::min(vzDes, 0.0);
    if (down) vzDes = std::min(vzDes, -0.4);
    if (!inp.noFloor && std::min(Va, V) < 25) vzDes = std::max(vzDes, -(0.6 + 0.48 * std::max(0.0, (landing ? b->agl : hmin) + 0.5 * std::min(0.0, Vu))));
    double eZ = vzDes - Vu;
    iZ = clamp(iZ + 0.35 * eZ * dt, -2, 2);
    double noseDn = std::max(0.0, std::sin(-pitchA));
    double diveNow = (tele.floor || landing || low) ? 0 : dive;
    double aU = clamp(1.9 * eZ + iZ, -(1.0 + 1.3 * noseDn * std::min(1.0, 3 * diveNow)) * G, 1.9 * G);

    double need = m * std::hypot(aR, aU + G);
    if (need > liftCap) {
      double kk = std::sqrt(std::max(0.0, liftCap * liftCap / (m * m) - (aU + G) * (aU + G)));
      if (std::fabs(aR) > kk) aR = (aR < 0 ? -1 : 1) * std::max(kk, 0.25 * std::fabs(aR));
    }
    double aRmax = std::tan(phiLim) * std::max(aU + G, 0.35 * G);
    aR = clamp(aR, -aRmax, aRmax);
    V3 f = {aF * fh[0] + (aR + aK) * rh[0], aU + G, aF * fh[2] + (aR + aK) * rh[2]};
    tele.aCmd = {aF, aU, aR}; tele.vzDes = vzDes; tele.wb = 0;

    // ----- 2. attitude -----
    double phiDes = clamp(std::atan2(aR, std::max(aU + G, 0.35 * G)), -phiLim, phiLim);
    double lvl = inp.noFloor ? 1 : clamp(0.12 + 0.11 * b->agl, 0, 1);
    double thLow = clamp(-0.5 * aF / G, -0.20, 0.42) * (aF > 0 ? 1 - 0.6 * sstep(8, 30, Vf) : 1 - 0.85 * sstep(22, 55, Vf)) * lvl;
    double gamma = std::atan2(Vu, std::max(Vh, 6.0));
    double gammaCmd = std::atan2(std::min(vzDes, 0.0), std::max(Vh, 6.0));
    double thetaH = clamp(thLow + lerp(gamma, std::min(gamma, 1.15 * gammaCmd), std::min(1.0, 1.5 * diveNow)) * sstep(10, 30, Va), -1.0, 1.0);
    double beta = b->vAirB[2] / std::max(Va, 12.0);
    double yawRateH = yawRateCmd - clamp(2.2 * beta, -1.1, 1.1) * sstep(8, 26, Va) * (1 - std::fabs(kickIn));
    yawRef = yaw + clamp(wrapPi(yawRef + yawRateH * dt - yaw), -0.3, 0.3);
    qDes = qFromYPR(yawRef, thetaH, phiDes);
    Fb = qinvrot(b->q, {m * f[0], m * f[1], m * f[2]});
    wff = qinvrot(b->q, {0, yawRateH, 0});
    kickCmd = clamp(std::fabs(kickIn) * 1.5, 0, 1);
    tele.phiDes = phiDes;
  }

  // ----- shared tail: moment demand and allocation -----
  double Tmin = std::max(0.3 * W, std::min(1.6 * W, (std::fabs(rollNeed) + (acroOn ? ACRO.sideArm * (std::fabs(b->sd[0]) + std::fabs(b->sd[1])) : 0)) / 0.10));
  double thetaDes = std::asin(clamp(2 * (qDes[0] * qDes[3] + qDes[1] * qDes[2]), -1, 1));
  V4 qe = qmul(qconj(b->q), qDes);
  if (qe[0] < 0) qe = {-qe[0], -qe[1], -qe[2], -qe[3]};
  tele.attErr = 2 * std::acos(std::min(1.0, qe[0]));
  V3 wDes = {clamp(5.0 * 2 * qe[1], -FEEL.rollRate, FEEL.rollRate) + wff[0],
             clamp(3.2 * 2 * qe[2], -1.2, 1.2) + wff[1],
             clamp(4.6 * 2 * qe[3], -1.3, 1.3) + wff[2]};
  double aL0 = acroOn ? ACRO.rollAcc : FEEL.rollAcc, aL1 = FEEL.yawAcc, aL2 = acroOn ? 9 : 4;
  V3 wdot = {clamp(11 * (wDes[0] - b->w[0]) + aff[0], -aL0, aL0),
             clamp(6.5 * (wDes[1] - b->w[1]) + aff[1], -aL1, aL1),
             clamp(10 * (wDes[2] - b->w[2]) + aff[2], -aL2, aL2)};
  V3 Iw = m3v(mp.I, b->w), gyro = cross(b->w, Iw);
  V3 tauDes = m3v(mp.I, wdot); tauDes[0] += gyro[0]; tauDes[1] += gyro[1]; tauDes[2] += gyro[2];
  double Hx = ENG.rotorI * ENG.omegaMax * b->eng.N; tauDes[1] += b->w[2] * Hx; tauDes[2] -= b->w[1] * Hx;
  tele.thetaDes = thetaDes;

  // ----- 3. thrust vector -----
  AeroCtl ctl{b->da, b->de, b->dr, b->brake};
  Wrench A0 = aeroWrench(b->vAirB, b->w, b->rho, ctl, cg);
  Wrench R0 = ramWrench(b->vAirB, b->w, b->eng.Tdry, cg);
  double Fx = Fb[0] - A0.F[0] - R0.F[0];
  double Fy = std::max(Fb[1] - A0.F[1] - R0.F[1], Tmin);
  double ab = 1 + ENG.abGain * b->eng.ab, eta = ENG.ductEta, Gmax = thrustOfN(1, b->sigma);
  double cant = std::cos(REAR.cant);
  if (Fx > 0) Fx = std::min(Fx, std::max(0.12 * Gmax, Gmax - std::max(Fy, 0.0) / eta) * ab * cant);
  double mag = std::hypot(Fx, Fy);
  double tiltNew = clamp(Fy <= 0 ? (Fx > 0 ? 0 : TILT_MAX) : std::atan2(Fy, Fx), 0, TILT_MAX);
  tiltCmd = lerp(tiltCmd, tiltNew, sstep(120, 700, mag));
  cmd.tilt = tiltCmd;
  cmd.ab = boost ? 1 : 0;
  cmd.brake = (inp.brake || (eV < -10 && Va > 20)) ? 1 : 0;
  double tilt = b->tilt, ct = std::cos(tilt), st = std::sin(tilt), fl = liftFrac(tilt);
  double per = tilt < PI / 2 ? (1 - fl) * ab * cant * ct + fl * eta * st : eta;
  double Gmain = clamp((Fx * ct + Fy * st) / std::max(per, 0.3), Tmin, Gmax);
  double sab = 1 + SIDE_AB * b->kick;
  cmd.kick = kickCmd;
  cmd.G = clamp(Gmain + (std::fabs(b->sd[0]) + std::fabs(b->sd[1])) / sab, Tmin, Gmax);
  double sideGas = clamp(Gmax - Gmain, (0.015 + 0.10 * cmd.kick) * Gmax, std::max(SIDE_BLEED * b->eng.Tdry, 0.05 * Gmax));
  double sideCap = std::min(SIDE_MAX, 0.6 * sab * sideGas);

  // ----- 4. allocation -----
  double u0[NU], lo[NU], hi[NU], pref[NU];
  Noz n; double Ltot = b->lift();
  n.tilt = tilt; n.R = b->R; n.rp = b->rp; n.ry = b->ry;
  for (int i = 0; i < 4; i++) {
    n.L[i] = b->L[i]; n.va[i] = b->va[i];
    u0[i] = b->L[i]; lo[i] = std::min(SHARE_MIN * 1.05 * Ltot * b->gfx[i] - b->L[i], 0.0); hi[i] = std::max(SHARE_MAX * 0.95 * Ltot * b->gfx[i] - b->L[i], 0.0);
    u0[4 + i] = b->va[i]; lo[4 + i] = -VANE_MAX * 0.95 - b->va[i]; hi[4 + i] = VANE_MAX * 0.95 - b->va[i];
  }
  u0[8] = b->rp; lo[8] = -VEC_MAX * 0.95 - b->rp; hi[8] = VEC_MAX * 0.95 - b->rp;
  u0[9] = b->ry; lo[9] = -VEC_MAX * 0.95 - b->ry; hi[9] = VEC_MAX * 0.95 - b->ry;
  for (int i = 0; i < 2; i++) { n.sd[i] = b->sd[i]; u0[10 + i] = b->sd[i]; lo[10 + i] = std::min(-sideCap - b->sd[i], 0.0); hi[10 + i] = std::max(sideCap - b->sd[i], 0.0); }

  auto row = [&](const Wrench& wv, const Wrench& k, double s, int col, double sum) {
    J[col] = sum; J[NU + col] = (wv.F[2] - k.F[2]) / s;
    J[2 * NU + col] = (wv.Tq[0] - k.Tq[0]) / s; J[3 * NU + col] = (wv.Tq[1] - k.Tq[1]) / s; J[4 * NU + col] = (wv.Tq[2] - k.Tq[2]) / s;
  };
  Wrench P0 = propWrench(n, cg);
  for (int i = 0; i < 4; i++) {
    n.L[i] += 20; row(propWrench(n, cg), P0, 20, i, 1); n.L[i] -= 20;
    n.va[i] += 0.01; row(propWrench(n, cg), P0, 0.01, 4 + i, 0); n.va[i] -= 0.01;
  }
  n.rp += 0.01; row(propWrench(n, cg), P0, 0.01, 8, 0); n.rp -= 0.01;
  n.ry += 0.01; row(propWrench(n, cg), P0, 0.01, 9, 0); n.ry -= 0.01;
  for (int i = 0; i < 2; i++) { n.sd[i] += 20; row(propWrench(n, cg), P0, 20, 10 + i, 0); n.sd[i] -= 20; }

  for (int i = 0; i < 4; i++) { pref[i] = Ltot * 0.25 - u0[i]; pref[4 + i] = -u0[4 + i]; }
  for (int i = 8; i < NU; i++) pref[i] = -u0[i];
  V3 tq = {tauDes[0] - A0.Tq[0] - R0.Tq[0], tauDes[1] - A0.Tq[1] - R0.Tq[1], tauDes[2] - A0.Tq[2] - R0.Tq[2]};
  rollNeed += (tq[0] - rollNeed) * 0.5;
  double FzDes = Fb[2] - (acroOn ? 1 : FEEL.gust) * (A0.F[2] + R0.F[2]);
  double e[NR] = {0, FzDes - P0.F[2], tq[0] - P0.Tq[0], tq[1] - P0.Tq[1], tq[2] - P0.Tq[2]};
  int sat = 0;
  std::array<double, NU> du = wls(J, e, WV, WU, pref, lo, hi, &sat);
  for (int i = 0; i < 4; i++) { cmd.s[i] = std::max(1e-3, (u0[i] + du[i]) / b->gfx[i]); cmd.va[i] = u0[4 + i] + du[4 + i]; }
  cmd.rp = u0[8] + du[8]; cmd.ry = u0[9] + du[9];
  cmd.sd[0] = u0[10] + du[10]; cmd.sd[1] = u0[11] + du[11];
  tele.sat = sat; tele.tau = tauDes; tele.tq = tq; tele.F[0] = Fx; tele.F[1] = Fy;
  return cmd;
}

} // namespace fohn
