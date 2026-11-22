// The rider's hands -> what the flight computer is asked for. Port of the control step of
// js/sim/game.js (Game.update) and of the key mapping of js/ui/input.js.
//
// The throttle is a lever that moves a target speed; the brake pulls the target down under the
// speed flown; the overdrive lifts it over the top speed; and resting on the skids with nothing
// asked for cuts the engine (any leftover target would keep the flight computer awake and lift
// the bike off again).
#pragma once
#include <algorithm>
#include "bike.hpp"
#include "fcs.hpp"

namespace fohn {

struct Ctl {
  double turn = 0;       // + right
  double pitch = 0;      // + climb: the stick pulled BACK (the down arrow), as on the web build
  double strafe = 0;     // + right
  double throttle = 0;   // -1..1
  bool boost = false, brake = false;
};

struct Rider {
  double vt = 0;         // the target speed along the nose (m/s)

  Input step(double dt, const Ctl& c, const Bike& b) {
    const double fwd = dot(b.v, qrot(b.q, {1, 0, 0}));
    vt = clamp(vt + c.throttle * 20 * dt, V_MIN, V_MAX);
    if (c.brake) vt = std::max(0.0, std::min(vt, fwd) - 30 * dt);
    if (c.boost && !b.abLock) vt = clamp(std::max(vt, fwd + 3), V_MIN, V_MAX + 30);
    else if (vt > V_MAX) vt = std::max(V_MAX, std::min(vt, fwd));
    if (b.contactN > 0.25 * b.mp.m * G && len(b.v) < 4 && !c.boost && c.pitch <= 0.05 && !(c.throttle > 0)) vt = 0;
    Input in;
    in.turn = c.turn; in.pitch = c.pitch; in.strafe = c.strafe; in.vt = vt; in.boost = c.boost; in.brake = c.brake;
    return in;
  }
};

} // namespace fohn
