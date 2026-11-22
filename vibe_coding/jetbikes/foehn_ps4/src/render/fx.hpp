// Effects. Port of js/render/fx.js: a plume on each of the nine nozzles, sized by the thrust that
// really leaves it (shock diamonds, the overdrive's flame on the rear one), dust / spray / snow
// kicked up where a jet really hits the surface, vapour in hard turns, display smoke.
// Not yet: the explosion and its debris (they come with the crash sequence of the game).
//
// The web build draws its particles as points. Here they are small quads facing the camera: the
// largest point a GLES2 driver must draw is one pixel, and a cloud of dust next to the camera is
// hundreds across.
#pragma once
#include <cstdint>
#include <vector>
#include "prog.hpp"
#include "../sim/bike.hpp"

namespace fohn {

class FX {
 public:
  void init();
  // once per simulated frame, drawn or not (no GL in here): the plumes follow the engine, the jets
  // are traced to the surface, the particles move
  void update(double dt, const Bike& b, const V3& wind, bool alive = true);
  void drawParticles(const Camera& cam);
  void drawPlumes(const Camera& cam, const Bike& b, bool lightBuffer);

  // where the jets meet water, for the rings on the lake
  float washStrength = 0, washPos[2]{0, 0};

 private:
  struct Particles {
    int n = 0, i = 0;
    std::vector<float> pos, vel, col, size, life, max, grow, drag, grav, a0;
    void init(int count);
    void spawn(const V3& p, const V3& v, const V3& c, double a, double size, double life, double grow = 0, double drag = 1, double grav = 0);
    void update(double dt, const V3& wind);
  };
  struct Plume { double k = 1; float pow = 0, ab = 0, len = 1, wide = 1; bool on = false; };

  Particles soft_;
  Plume plumes_[9];                 // 4 lift, rear, 4 side
  double acc_ = 0, smokeAcc_ = 0;
  uint32_t seed_ = 20261122u;
  double rnd() { seed_ = seed_ * 1664525u + 1013904223u; return seed_ / 4294967296.0; }

  Prog plumeProg_, partProg_;
  GLuint tubeVbo_ = 0, tubeIbo_ = 0, partVbo_ = 0, partIbo_ = 0;
  int tubeCount_ = 0;
  std::vector<float> quads_;
};

} // namespace fohn
