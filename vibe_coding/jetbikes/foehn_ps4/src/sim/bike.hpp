// FÖHN F-5 "fusion" jet motorcycle: geometry, mass, the engine and its five nozzles, aerodynamics
// and the 6-DOF rigid-body integrator. Port of js/sim/bike.js.
//
// Body / design frame: x forward, y up, z to the rider's RIGHT. Design-frame origin on the ground
// under the middle of the bike; the physics works about the centre of gravity.
#pragma once
#include <array>
#include <cmath>
#include "vmath.hpp"

namespace fohn {

constexpr double G = 9.80665;
constexpr double RHO0 = 1.225;
constexpr double D2R = PI / 180.0;

struct Eng {
  double tmax = 14000, abGain = 0.5, ductEta = 0.94, exp = 3.2, idleN = 0.30, ve = 900;
  double rotorI = 0.012, omegaMax = 4600, axisY = 0.60, radius = 0.2;
};
extern const Eng ENG;

struct Vec3c { double p[3]; };
extern const Vec3c LIFT[4];           // lift pipe exits (design frame)
struct Rear { double p[3]; double cant; };
extern Rear REAR;                      // rear nozzle; cant set once the CG is known
extern const Vec3c SIDE[2];           // nose + tail side nozzles
extern const double INTAKE[3];

constexpr double SIDE_MAX = 2400;
constexpr double SIDE_BLEED = 0.30;
constexpr double SIDE_AB = 1.5;
extern const double TILT_MAX, VANE_MAX, VEC_MAX;
constexpr double SHARE_MIN = 0.10, SHARE_MAX = 0.45;
struct Lim { double da, de, dr; };
extern const Lim LIM;

inline double thrustOfN(double N, double sigma) { return ENG.tmax * std::pow(sigma, 0.85) * std::pow(std::max(N, 0.0), ENG.exp); }
inline double nOfThrust(double T, double sigma) { return std::pow(std::max(T, 0.0) / (ENG.tmax * std::pow(sigma, 0.85)), 1.0 / ENG.exp); }
inline double liftFrac(double tilt) { if (tilt >= PI / 2) return 1; double c = std::cos(tilt), s = std::sin(tilt); return s / (c + s); }

// mass properties
struct MassProps { double m; V3 cg; M3 I; M3 Iinv; };
constexpr double FUEL0 = 10;
MassProps massProps(double fuel);

// wrench and control structs
struct Wrench { V3 F{0, 0, 0}; V3 Tq{0, 0, 0}; double stall = 0; };
struct AeroCtl { double da = 0, de = 0, dr = 0, brake = 0; };
struct Noz { double tilt = 0; double L[4]{0, 0, 0, 0}; double va[4]{0, 0, 0, 0}; double R = 0; double rp = 0; double ry = 0; double sd[2]{0, 0}; };

Wrench aeroWrench(const V3& vB, const V3& w, double rho, const AeroCtl& ctl, const V3& cg);
Wrench propWrench(const Noz& n, const V3& cg);
Wrench ramWrench(const V3& vB, const V3& w, double Tdry, const V3& cg);

// turbofan spool
struct Turbofan {
  double N, ab = 0, Tdry = 0, dNdt = 0; bool on = true;
  Turbofan() : N(ENG.idleN) {}
  void update(double dt, double Gcmd, double abCmd, double sigma);
};

// command the FCS hands the bike each tick
struct Cmd {
  double G = 0, ab = 0, tilt = PI / 2;
  double s[4]{1, 1, 1, 1};
  double va[4]{0, 0, 0, 0};
  double rp = 0, ry = 0;
  double sd[2]{0, 0};
  double kick = 0, da = 0, de = 0, dr = 0, brake = 0;
};

// probe/skid geometry
extern const V3 SKIDS[4];
extern const V3 PROBES[8];
constexpr int N_PROBES = 8;

class Bike {
 public:
  Turbofan eng;
  double fuel = FUEL0;
  MassProps mp;
  V3 p{0, 0, 0}, v{0, 0, 0}, w{0, 0, 0};
  V4 q{1, 0, 0, 0};
  double tilt = PI / 2;
  double va[4]{0, 0, 0, 0}, sh[4]{0.25, 0.25, 0.25, 0.25}, rp = 0, ry = 0;
  double sd[2]{0, 0}, sdv[2]{0, 0}, kick = 0;
  double L[4]{0, 0, 0, 0}, R = 0, abOut = 0;
  double da = 0, de = 0, dr = 0, brake = 0;
  bool smoke = false;
  double heat = 0; bool abLock = false;
  double t = 0;
  bool HprevSet = false; V3 Hprev{0, 0, 0};
  double contactN = 0; double skidF[4]{0, 0, 0, 0};
  V3 acc{0, 0, 0};
  const char* crash = nullptr; double scrape = 0;
  double airspeed = 0; V3 vAirB{0, 0, 0}; double sigma = 1, rho = RHO0, stall = 0;
  double gfx[4]{1, 1, 1, 1};
  double agl = 0, groundY = 0; bool overWater = false;
  double fuelFlow = 0;

  int _mpAge = 0, _gAge = 0;

  Bike() { reset({0, 0, 0}, 0); }
  void reset(const V3& pos, double yaw, double speed = 0, double fuelKg = FUEL0);

  V3 arm(const V3& d) const { return {d[0] - mp.cg[0], d[1] - mp.cg[1], d[2] - mp.cg[2]}; }
  V3 toWorld(const V3& d) const { return add(p, qrot(q, arm(d))); }
  double lift() const { return L[0] + L[1] + L[2] + L[3]; }
  double thrust() const { return lift() + R + std::fabs(sd[0]) + std::fabs(sd[1]); }

  void step(double dt, const Cmd& cmd, const V3& windW);
};

} // namespace fohn
