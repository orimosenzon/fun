// Flight control system of the F-5: a unified STOVL-style control law. The stick asks for a
// velocity vector; the computer finds the attitude and thrust vector, then allocates 12 effectors
// by bounded weighted least squares. Port of js/sim/fcs.js.
#pragma once
#include <array>
#include <functional>
#include "vmath.hpp"
#include "bike.hpp"

namespace fohn {

// How the bike feels in the hand (the ?tune sliders). Mutable so the host can retune at runtime.
struct Feel {
  double stick = 0.07;
  double rollRate = 3.2, rollAcc = 14;
  double yawRate = 2.0, yawAcc = 6;
  double bank = 1.27;
  double kick = 13, strafe = 9, gust = 0.5;
  double hairpin = 1;
  double recoil = 1.0, muzzle = 300;
  double dive = 1;
  double acroRoll = 4.0, acroPitch = 1.7, acroYaw = 0.9;
  double detent = 7;
  double net = 1;
};
extern Feel FEEL;
extern const Feel FEEL_DEFAULT;

constexpr int FCS_HZ = 120;
constexpr double V_MAX = 95, V_MIN = -8;
constexpr double V_CORNER = 24, HAIR_T0 = 0.5, HAIR_T1 = 1.3;
constexpr int NU = 12, NR = 5;

// the stick / pads, already mapped to the law's vocabulary
struct Input {
  double turn = 0;     // + right
  double pitch = 0;    // + climb
  double strafe = 0;   // + right (hover: side-step; speed: side kick)
  double vt = 0;       // forward speed target (m/s)
  bool boost = false, brake = false;
  bool noFloor = false, noDive = false;
};

struct Tele {
  double phiDes = 0, thetaDes = 0; V3 aCmd{0, 0, 0}; double wb = 0; int sat = 0;
  double vzDes = 0; bool floor = false; double attErr = 0; V3 tau{0, 0, 0}; V3 tq{0, 0, 0}; double F[2]{0, 0};
};

class FCS {
 public:
  Bike* bike;
  Cmd cmd;
  Tele tele;
  std::function<double(double, double, bool)> obstacle;  // tree/roof tops; null in the open

  explicit FCS(Bike* b);
  void reset();
  const Cmd& update(double dt, const Input& inp);

  bool aerobatic() const { return acro && !rec; }
  bool setAcro(bool on, const char* why = nullptr);

 private:
  double turn = 0, pitch = 0, strafe = 0, hold = 0, rollNeed = 0, iV = 0, iZ = 0;
  double yawRef = 0; bool yawRefSet = false;
  double tiltCmd = PI / 2;
  bool idle = true; double downT = 1; bool low = false;
  bool acro = false, rec = false; double recT = 0; const char* bail = nullptr;
  V4 qRef{1, 0, 0, 0}; bool qRefSet = false;
  double kickA = 0; V3 wStick{0, 0, 0};
  double J[NR * NU];

  struct AcroOut { bool valid = false; V3 Fb{0, 0, 0}; V4 qDes{1, 0, 0, 0}; V3 wff{0, 0, 0}; V3 aff{0, 0, 0}; double eV = 0, kickCmd = 0; };
  double clearance(double T);
  AcroOut acroLaw(double dt, const Input& inp, bool boost);
};

double levelYaw(const V4& q);

} // namespace fohn
