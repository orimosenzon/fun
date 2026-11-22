// Mountain wind: mean valley wind with a log boundary layer, orographic lift/sink, and a
// frozen-turbulence gust field. Port of js/sim/wind.js. Deterministic (seeded), so runs replay.
#pragma once
#include <array>
#include <cmath>
#include <vector>
#include "vmath.hpp"
#include "noise.hpp"
#include "terrain.hpp"

namespace fohn {

struct WindMode { double f, amp; double ph[3]; double lat[2]; };

struct Wind {
  bool calm = false;
  V3 dir = {1, 0, 0};
  double speed = 0, gust = 1;
  double U = 1.5, nrm = 1;
  std::vector<WindMode> modes;

  V3 operator()(const V3& p, double t, double agl) const {
    if (calm) return {0, 0, 0};
    agl = agl < 0.3 ? 0.3 : agl;
    double prof = clamp(std::log(agl / 0.05) / std::log(40.0 / 0.05), 0.25, 1.35);
    V3 n = normalAt(p[0], p[2]);
    double slope = -(n[0] * dir[0] + n[2] * dir[2]) / n[1];
    double steep = std::sqrt(n[0] * n[0] + n[2] * n[2]) / n[1];
    double decay = std::exp(-agl / 280);
    V3 w = {dir[0] * speed * prof, clamp(speed * prof * slope * decay, -7, 7), dir[2] * speed * prof};
    double sig = std::min(gust * (0.35 + 0.11 * speed) * (1 + 1.4 * std::min(steep, 1.5) * decay), 10.0);
    double along = p[0] * dir[0] + p[2] * dir[2], across = -p[0] * dir[2] + p[2] * dir[0];
    for (const auto& m : modes) {
      double k = 6.283 * m.f / U;
      double arg = 6.283 * m.f * t - k * along + 0.6 * k * across * m.lat[0] + 0.4 * k * p[1] * m.lat[1];
      double a = sig * m.amp / nrm;
      w[0] += a * std::sin(arg + m.ph[0]);
      w[1] += 0.6 * a * std::min(1.0, agl / 12) * std::sin(arg * 1.07 + m.ph[1]);
      w[2] += a * std::sin(arg * 0.93 + m.ph[2]);
    }
    return w;
  }
};

inline Wind makeWind(double heading, double speed, double gust, uint32_t seed) {
  Wind W;
  W.calm = false;
  W.dir = {std::cos(heading), 0, std::sin(heading)};
  W.speed = speed; W.gust = gust;
  Mulberry32 r(seed);
  double ampSq = 0;
  for (int k = 0; k < 12; k++) {
    double f = 0.03 * std::pow(1.5, k);
    double amp = std::pow(f / 0.03, -1.0 / 3.0);
    double a = r() * 6.283;
    WindMode m;
    m.f = f; m.amp = amp;
    m.ph[0] = r() * 6.28; m.ph[1] = r() * 6.28; m.ph[2] = r() * 6.28;
    m.lat[0] = std::cos(a); m.lat[1] = std::sin(a);
    W.modes.push_back(m);
    ampSq += amp * amp;
  }
  W.nrm = std::sqrt(ampSq / 2);
  W.U = std::max(speed, 1.5);
  return W;
}

inline Wind calmWind() { Wind W; W.calm = true; W.dir = {1, 0, 0}; W.speed = 0; return W; }

// The rider's wind-level choice (0..3). ev gives the per-event base heading/speed/gust.
inline Wind windFor(double heading, double evSpeed, double evGust, int level, uint32_t seed) {
  if (level <= 0) return calmWind();
  static const double kk[] = {0, 1, 2.2, 3.4}, gg[] = {0, 1, 2.2, 3.4}, s0[] = {0, 0, 9, 15}, g0[] = {0, 0, 2.2, 3.4};
  double k = kk[level], g = gg[level];
  return makeWind(heading, std::max(evSpeed * k, s0[level]), std::max(evGust * g, g0[level]), seed);
}

} // namespace fohn
