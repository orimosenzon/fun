// The baked alpine terrain: one height field shared by physics and rendering.
// Port of js/sim/terrain.js. Loads assets/terrain.bin (Uint16 heightfield) + assets/meta.json.
#pragma once
#include <vector>
#include <string>
#include <cmath>
#include "vmath.hpp"

namespace fohn {

struct Lake { double level, x0, x1, z0, z1; double out[4]; };   // out: outlet x, z and its direction (0 = none)
struct River { std::string name, from, to; double width = 0; std::vector<V3> pts; };
struct Fall { double width, x, z, top, foot, dx, dz; };
struct Hole { bool on = false; double x = 0, z = 0, r = 0, y = 0, rim = 0; };
struct Pad { std::string name; double x, y, z, r; };              // a landing place: its centre and the radius of the flat ground

struct Terrain {
  int n = 0;
  double cell = 12, half = 0, hscale = 1;
  std::vector<float> H;          // n*n heights (metres)
  std::vector<Lake> lakes;
  Hole hole;
  bool ready = false;

  // meta kept for the renderer / scenery
  int bn = 0; double bcell = 0, bhalf = 0;
  std::vector<River> rivers, farRivers;
  std::vector<Fall> falls;
  std::vector<Pad> pads;
  std::vector<float> B;          // backdrop heights, bn*bn (far mountains beyond the playable square)
  bool loadBackdrop(const std::string& binPath);

  bool load(const std::string& binPath, const std::string& metaPath);

  double gridAt(double x, double z) const;
  double groundAt(double x, double z) const;
  V3 normalAt(double x, double z) const;
  double waterAt(double x, double z) const;
  double surfaceAt(double x, double z) const { double g = groundAt(x, z), w = waterAt(x, z); return g > w ? g : w; }
  bool isWater(double x, double z) const { return waterAt(x, z) > groundAt(x, z); }
};

// one shared world, as in the JS module scope
extern Terrain gT;

inline double groundAt(double x, double z) { return gT.groundAt(x, z); }
inline V3 normalAt(double x, double z) { return gT.normalAt(x, z); }
inline double waterAt(double x, double z) { return gT.waterAt(x, z); }
inline double surfaceAt(double x, double z) { return gT.surfaceAt(x, z); }

// ISA troposphere
inline double airDensity(double h) { double b = 1 - 2.25577e-5 * h; if (b < 0.1) b = 0.1; return 1.225 * std::pow(b, 4.2559); }
inline double airTemp(double h) { return 288.15 - 0.0065 * h; }

} // namespace fohn
