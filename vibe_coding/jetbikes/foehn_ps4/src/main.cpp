// FÖHN for PS4 — PC host. Loads the real alpine terrain, flies the ported F-5, draws it in GLES2.
// Interactive:  ./foehn [--time afternoon] [--wx clear|rain|snow] [--wind 0..3] [--invert] [--nopost] [--shadow 0|1024|2048]
// Screenshot:   ./foehn --shot out.ppm [--frames N] [--side] [--boost] [--pos x,z,height,yawDeg,speed] [--aniso 1..16] [--bench]
//               [--cam ex,ey,ez,tx,ty,tz]   a fixed camera, in world coordinates (to set beside dbg.free(...) of the web build)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <chrono>
#include "sim/sim.hpp"
#include "sim/rider.hpp"
#include "sim/terrain.hpp"
#include "render/renderer.hpp"
#include "platform/platform.hpp"

using namespace fohn;

// base pad: flat, radius 75
static const V3 PAD = {-1800, 992.69, 2330};

struct Cam { V3 eye, target; };
static Cam chase(const Bike& b) {
  V3 ypr = yprOf(b.q);
  double yaw = ypr[0];
  V3 fwd = {std::cos(yaw), 0, -std::sin(yaw)};
  V3 up = {0, 1, 0};
  V3 focus = add(b.p, scale(up, 1.2));
  Cam c;
  c.target = add(focus, scale(fwd, 4.0));
  c.eye = add(sub(focus, scale(fwd, 6.5)), scale(up, 2.0));
  double gy = gT.surfaceAt(c.eye[0], c.eye[2]) + 1.0;
  if (c.eye[1] < gy) c.eye[1] = gy;
  return c;
}

// keys -> stick, as js/ui/input.js reads them: the down arrow pulls the stick back and climbs
static Ctl readKeys(const Keys& k, bool invert) {
  Ctl c;
  c.turn = (k.right ? 1.0 : 0.0) - (k.left ? 1.0 : 0.0);
  c.pitch = ((k.down ? 1.0 : 0.0) - (k.up ? 1.0 : 0.0)) * (invert ? -1 : 1);
  c.strafe = (k.strafeR ? 1.0 : 0.0) - (k.strafeL ? 1.0 : 0.0);
  c.throttle = (k.thrUp ? 1.0 : 0.0) - (k.thrDn ? 1.0 : 0.0);
  c.boost = k.boost; c.brake = k.brake;
  return c;
}

int main(int argc, char** argv) {
  const char* shot = nullptr;
  int frames = 300;
  bool sideView = false;     // --side: a close three-quarter view of the bike for stills
  bool bench = false;        // --bench (with --shot): draw every frame and report the time per frame
  bool boost = false;        // --boost (with --shot): hold the overdrive, to see its flame
  bool invert = false;       // --invert: the up arrow climbs (the web build's "invert" setting)
  int timeIdx = 2, wxIdx = 0, windLevel = 1;
  bool hasPos = false; double pos[5] = {0, 0, 30, 0, 30};
  bool hasCam = false; double camAt[6] = {0, 0, 0, 0, 0, 0};
  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "--shot") && i + 1 < argc) shot = argv[++i];
    else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc) frames = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--side")) sideView = true;
    else if (!std::strcmp(argv[i], "--bench")) bench = true;
    else if (!std::strcmp(argv[i], "--boost")) boost = true;
    else if (!std::strcmp(argv[i], "--invert")) invert = true;
    else if (!std::strcmp(argv[i], "--nopost")) Renderer::postWanted = false;
    else if (!std::strcmp(argv[i], "--shadow") && i + 1 < argc) gEnv.shadowSize = std::atoi(argv[++i]);     // the side of the shadow map (2048), 0 for none
    else if (!std::strcmp(argv[i], "--aniso") && i + 1 < argc) TerrainR::photoAniso = (float)std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--time") && i + 1 < argc) { int t = timeIndex(argv[++i]); if (t >= 0) timeIdx = t; else std::fprintf(stderr, "unknown --time (morning, noon, afternoon, golden, dusk)\n"); }
    else if (!std::strcmp(argv[i], "--wx") && i + 1 < argc) { int w = weatherIndex(argv[++i]); if (w >= 0) wxIdx = w; else std::fprintf(stderr, "unknown --wx (clear, rain, snow)\n"); }
    else if (!std::strcmp(argv[i], "--wind") && i + 1 < argc) windLevel = std::min(3, std::max(0, std::atoi(argv[++i])));
    else if (!std::strcmp(argv[i], "--cam") && i + 1 < argc) { hasCam = std::sscanf(argv[++i], "%lf,%lf,%lf,%lf,%lf,%lf", &camAt[0], &camAt[1], &camAt[2], &camAt[3], &camAt[4], &camAt[5]) == 6; }
    else if (!std::strcmp(argv[i], "--pos") && i + 1 < argc) { hasPos = std::sscanf(argv[++i], "%lf,%lf,%lf,%lf,%lf", &pos[0], &pos[1], &pos[2], &pos[3], &pos[4]) >= 2; }
  }

  if (!gT.load("assets/terrain.bin", "assets/meta.json")) { std::fprintf(stderr, "no terrain\n"); return 1; }
  if (!gT.loadBackdrop("assets/backdrop.bin")) std::fprintf(stderr, "no backdrop (far mountains): continuing without\n");

  Sim sim;
  auto setWind = [&]() { sim.setWind(windFor(0.25, 4, 1, windLevel, 7)); };
  auto spawn = [&]() { sim.reset({PAD[0], gT.groundAt(PAD[0], PAD[2]), PAD[2]}, 0); };
  setWind(); spawn();
  Rider rider;

  Platform plat;
  Renderer rend;
  bool ok = shot ? plat.initOffscreen(1280, 720) : plat.initWindow(1280, 720, "FOHN \xE2\x80\x94 PS4 port (PC host)");
  if (!ok) { std::fprintf(stderr, "platform init failed\n"); return 1; }
  rend.resize(plat.width(), plat.height());
  gEnv.timeIdx = timeIdx; gEnv.wxIdx = wxIdx;
  rend.init();

  if (shot) {
    double dt = 1.0 / 60;
    Cam c;
    auto benchT0 = std::chrono::steady_clock::now();
    if (hasPos) {
      // a still from a given place: in the air at that height over the surface, flying straight
      double yaw = pos[3] * PI / 180;
      sim.reset({pos[0], gT.surfaceAt(pos[0], pos[1]) + pos[2], pos[1]}, yaw, pos[4]);
      rider.vt = pos[4];
    }
    for (int f = 0; f < frames; f++) {
      Ctl c0;
      if (!hasPos) {
        // scripted flight: sit, climb, accelerate, bank
        if (f < 40) { /* sit */ }
        else if (f < 150) { c0.pitch = 1; }
        else if (f < 260) { c0.throttle = 1; }
        else { c0.throttle = 1; c0.turn = 1; }
      }
      c0.boost = boost;
      Input in = rider.step(dt, c0, sim.bike);
      sim.advance(dt, in);
      rend.simulate(dt, sim.bike, sim.windW);
      // (the last frames are drawn, so that the rain has a camera speed and the clouds a time)
      if (bench && f == 10) { glFinish(); benchT0 = std::chrono::steady_clock::now(); }
      if (f >= frames - 3 || bench) {
        c = chase(sim.bike);
        if (sideView) {
          const Bike& b = sim.bike;
          double yaw = yprOf(b.q)[0];
          V3 fwd = {std::cos(yaw), 0, -std::sin(yaw)}, right = {std::sin(yaw), 0, std::cos(yaw)};
          c.target = add(b.p, {0, 0.2, 0});
          c.eye = add(add(add(b.p, scale(fwd, 2.2)), scale(right, 3.0)), {0, 0.9, 0});
        }
        if (hasCam) { c.eye = {camAt[0], camAt[1], camAt[2]}; c.target = {camAt[3], camAt[4], camAt[5]}; }
        rend.frame(c.eye, c.target, sim.bike, f == frames - 3 ? f * dt : dt, sim.windW);
      }
    }
    glFinish();
    if (bench && frames > 10) {
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - benchT0).count() / (frames - 10);
      std::printf("bench: %.2f ms per frame (sim + draw), %.0f fps, %dx%d\n", ms, 1000 / ms, plat.width(), plat.height());
    }
    if (plat.savePPM(shot)) std::printf("wrote %s (%dx%d)  %s/%s%s  bike agl=%.1f spd=%.1f at (%.1f, %.1f, %.1f)  camera (%.1f, %.1f, %.1f) -> (%.1f, %.1f, %.1f)%s\n", shot, plat.width(), plat.height(), TIMES[gEnv.timeIdx].key, WEATHER[gEnv.wxIdx].key, (std::string(rend.post() ? "" : "  no post") + (rend.shadows() ? "" : "  no shadows")).c_str(),
                                        sim.bike.agl, len(sim.bike.v), sim.bike.p[0], sim.bike.p[1], sim.bike.p[2], c.eye[0], c.eye[1], c.eye[2], c.target[0], c.target[1], c.target[2], sim.bike.crash ? "  CRASHED" : "");
    else std::fprintf(stderr, "savePPM failed\n");
    plat.shutdown();
    return 0;
  }

  // interactive
  std::printf("controls (as on the web build): left/right turn, DOWN arrow climbs, UP arrow dives%s,\n"
              "          W/S throttle, A/D side-step, Space overdrive, Shift or B brake, R respawn,\n"
              "          T hour, X weather, N wind, Esc quit\n", invert ? " (inverted)" : "");
  auto prev = std::chrono::steady_clock::now();
  Keys k;
  while (plat.poll(k)) {
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - prev).count();
    prev = now;
    if (dt > 0.1) dt = 0.1;
    if (k.nextTime || k.nextWx) {
      if (k.nextTime) timeIdx = (timeIdx + 1) % N_TIMES;
      if (k.nextWx) wxIdx = (wxIdx + 1) % N_WEATHER;
      rend.setTime(timeIdx, wxIdx);
      std::printf("%s, %s\n", TIMES[timeIdx].key, WEATHER[wxIdx].key);
    }
    if (k.nextWind) { windLevel = (windLevel + 1) % 4; setWind(); std::printf("wind level %d\n", windLevel); }
    if (k.respawn) { spawn(); rider.vt = 0; }
    Input in = rider.step(dt, readKeys(k, invert), sim.bike);
    sim.advance(dt, in);
    if (sim.bike.crash) { spawn(); rider.vt = 0; }
    rend.simulate(dt, sim.bike, sim.windW);
    rend.resize(plat.width(), plat.height());
    Cam c = chase(sim.bike);
    rend.frame(c.eye, c.target, sim.bike, dt, sim.windW);
    plat.swap();
  }
  std::printf("at exit: agl=%.1f m  speed=%.1f m/s  target=%.1f m/s\n", sim.bike.agl, len(sim.bike.v), rider.vt);
  plat.shutdown();
  return 0;
}
