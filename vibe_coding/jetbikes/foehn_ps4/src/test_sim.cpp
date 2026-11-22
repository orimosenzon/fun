// Headless check that the ported simulation flies: load the real terrain, spawn on the ground,
// hover, climb, then accelerate forward. Prints a trace. No graphics.
#include <cstdio>
#include <cmath>
#include "sim/sim.hpp"
#include "sim/terrain.hpp"

using namespace fohn;

int main(int argc, char** argv) {
  const char* bin = "assets/terrain.bin";
  const char* meta = "assets/meta.json";
  if (!gT.load(bin, meta)) { std::fprintf(stderr, "failed to load terrain\n"); return 1; }
  std::printf("terrain: n=%d cell=%.0f half=%.0f hscale=%.5f lakes=%zu hole=%s\n",
              gT.n, gT.cell, gT.half, gT.hscale, gT.lakes.size(), gT.hole.on ? "yes" : "no");

  // spawn on the "base" pad (flat, radius 75), as the game does
  double px = -1800, pz = 2330;
  double g0 = gT.groundAt(px, pz);
  std::printf("base pad ground = %.2f m; slope sample: g(px+5)=%.2f g(px-5)=%.2f g(pz+5)=%.2f g(pz-5)=%.2f\n",
              g0, gT.groundAt(px + 5, pz), gT.groundAt(px - 5, pz), gT.groundAt(px, pz + 5), gT.groundAt(px, pz - 5));

  Sim sim;                       // calm wind
  sim.reset({px, g0, pz}, 0);    // on the ground
  std::printf("spawn: p=(%.2f, %.2f, %.2f) agl=%.3f mass=%.1f kg cg_y=%.3f\n",
              sim.bike.p[0], sim.bike.p[1], sim.bike.p[2], sim.bike.agl, sim.bike.mp.m, sim.bike.mp.cg[1]);

  Input in;
  double dt = 1.0 / 60;          // 60 fps host
  auto trace = [&](const char* phase) {
    const Bike& b = sim.bike;
    V3 ypr = yprOf(b.q);
    std::printf("t=%5.2f %-7s agl=%7.2f  spd=%6.2f  vt=%5.1f  p=(%7.1f,%7.1f,%7.1f)  pitch=%5.1f roll=%5.1f  N=%.2f thr=%.0f%s\n",
                sim.t, phase, b.agl, len(b.v), in.vt, b.p[0], b.p[1], b.p[2],
                ypr[1] * 180 / PI, ypr[2] * 180 / PI, b.eng.N, b.thrust(), b.crash ? b.crash : "");
  };

  // phase 0: sit still on the pad for 2 s (should stay put, no crash)
  in = Input{};
  for (int f = 0; f < 120; f++) { sim.advance(dt, in); if (f % 60 == 0) trace("sit"); }
  trace("sit");

  // phase 1: hover up (climb) for 3 s
  in = Input{}; in.pitch = 0.6;
  for (int f = 0; f < 180; f++) { sim.advance(dt, in); if (f % 30 == 0) trace("climb"); }
  trace("climb");

  // phase 2: level and accelerate forward for 5 s
  in = Input{}; in.vt = 30;
  for (int f = 0; f < 300; f++) { sim.advance(dt, in); if (f % 30 == 0) trace("fwd"); }
  trace("fwd");

  // phase 3: right turn for 3 s
  in = Input{}; in.vt = 30; in.turn = 0.8;
  for (int f = 0; f < 180; f++) { sim.advance(dt, in); if (f % 30 == 0) trace("turn"); }
  trace("turn");

  // phase 4: descend and settle for 5 s
  in = Input{}; in.vt = 0; in.pitch = -0.4;
  for (int f = 0; f < 300; f++) { sim.advance(dt, in); if (f % 60 == 0) trace("land"); }
  trace("land");

  std::printf("\nresult: %s\n", sim.bike.crash ? "CRASHED" : "flew clean");
  return sim.bike.crash ? 2 : 0;
}
