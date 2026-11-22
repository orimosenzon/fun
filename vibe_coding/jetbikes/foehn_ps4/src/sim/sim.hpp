// Fixed-step simulation driver: physics at 600 Hz, flight computer at 120 Hz. Port of js/sim/sim.js.
#pragma once
#include "bike.hpp"
#include "fcs.hpp"
#include "wind.hpp"

namespace fohn {

constexpr int PHYS_HZ = 600, SUB = 5;   // 5 physics steps per control tick
constexpr double H = 1.0 / PHYS_HZ;

class Sim {
 public:
  Bike bike;
  FCS fcs;
  Wind wind;
  V3 windW{0, 0, 0};
  double acc = 0, t = 0;

  Sim() : bike(), fcs(&bike), wind(calmWind()) {}
  explicit Sim(const Wind& w) : bike(), fcs(&bike), wind(w) {}

  void setWind(const Wind& w) { wind = w; }
  void reset(const V3& pos, double yaw, double speed = 0) { bike.reset(pos, yaw, speed); fcs.reset(); acc = 0; }

  void tick(const Input& inp) {
    windW = wind(bike.p, t, bike.agl);
    const Cmd& cmd = fcs.update(SUB * H, inp);
    for (int i = 0; i < SUB; i++) bike.step(H, cmd, windW);
    t += SUB * H;
  }

  int advance(double dt, const Input& inp) {
    acc += std::min(dt, 0.1);
    int n = 0;
    while (acc >= SUB * H) { tick(inp); acc -= SUB * H; n++; }
    return n;
  }
};

} // namespace fohn
