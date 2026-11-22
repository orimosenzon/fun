#include "terrain.hpp"
#include "json.hpp"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>

namespace fohn {

Terrain gT;

static std::string readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool Terrain::load(const std::string& binPath, const std::string& metaPath) {
  std::string metaStr = readFile(metaPath);
  if (metaStr.empty()) { std::fprintf(stderr, "terrain: cannot read %s\n", metaPath.c_str()); return false; }
  Json m = parseJson(metaStr);
  n = (int)m["n"].n(0);
  cell = m["cell"].n(12);
  half = m["half"].n(0);
  hscale = m["hscale"].n(1);
  bn = (int)m["bn"].n(0);
  bcell = m["bcell"].n(0);
  bhalf = m["bhalf"].n(0);

  lakes.clear();
  const Json& ls = m["lakes"];
  for (size_t i = 0; i < ls.size(); i++) {
    const Json& L = ls[i];
    Lake lk{L["level"].n(), L["x0"].n(), L["x1"].n(), L["z0"].n(), L["z1"].n(), {0, 0, 0, 0}};
    for (int k = 0; k < 4; k++) lk.out[k] = L["out"][(size_t)k].n(0);
    lakes.push_back(lk);
  }
  auto readRivers = [](const Json& rs, std::vector<River>& out) {
    out.clear();
    for (size_t i = 0; i < rs.size(); i++) {
      const Json& r = rs[i];
      River rv; rv.name = r["name"].str; rv.from = r["from"].str; rv.to = r["to"].str; rv.width = r["width"].n();
      const Json& ps = r["pts"];
      rv.pts.reserve(ps.size());
      for (size_t k = 0; k < ps.size(); k++) rv.pts.push_back({ps[k][(size_t)0].n(), ps[k][(size_t)1].n(), ps[k][(size_t)2].n()});
      out.push_back(std::move(rv));
    }
  };
  readRivers(m["rivers"], rivers);
  readRivers(m["far"], farRivers);
  falls.clear();
  const Json& fs = m["falls"];
  for (size_t i = 0; i < fs.size(); i++) {
    const Json& f = fs[i];
    falls.push_back({f["width"].n(), f["x"].n(), f["z"].n(), f["top"].n(), f["foot"].n(), f["dx"].n(), f["dz"].n()});
  }
  pads.clear();
  for (const auto& kv : m["pads"].obj) {
    const Json& q = kv.second;
    pads.push_back({kv.first, q[(size_t)0].n(), q[(size_t)1].n(), q[(size_t)2].n(), q[(size_t)3].n()});
  }
  if (m.has("hole")) {
    const Json& h = m["hole"];
    hole.on = true; hole.x = h["x"].n(); hole.z = h["z"].n(); hole.r = h["r"].n(); hole.y = h["y"].n(); hole.rim = h["rim"].n();
  }

  // height field: Uint16 little-endian, n*n, scaled by hscale
  std::string bin = readFile(binPath);
  if ((int)bin.size() < n * n * 2) { std::fprintf(stderr, "terrain: %s too small (%zu)\n", binPath.c_str(), bin.size()); return false; }
  const uint8_t* u = reinterpret_cast<const uint8_t*>(bin.data());
  H.resize((size_t)n * n);
  for (size_t i = 0; i < (size_t)n * n; i++) {
    uint16_t v = (uint16_t)(u[i * 2] | (u[i * 2 + 1] << 8));
    H[i] = (float)(v * hscale);
  }
  ready = true;
  return true;
}

bool Terrain::loadBackdrop(const std::string& binPath) {
  std::string bin = readFile(binPath);
  if (bn <= 0 || (int)bin.size() < bn * bn * 2) return false;
  const uint8_t* u = reinterpret_cast<const uint8_t*>(bin.data());
  B.resize((size_t)bn * bn);
  for (size_t i = 0; i < (size_t)bn * bn; i++) B[i] = (float)((uint16_t)(u[i * 2] | (u[i * 2 + 1] << 8)) * hscale);
  return true;
}

double Terrain::gridAt(double x, double z) const {
  double gx = (x + half) / cell, gz = (z + half) / cell;
  gx = clamp(gx, 0, n - 1.0001); gz = clamp(gz, 0, n - 1.0001);
  int i = (int)gx, j = (int)gz;
  double fx = gx - i, fz = gz - j;
  size_t k = (size_t)j * n + i;
  double h00 = H[k], h10 = H[k + 1], h01 = H[k + n], h11 = H[k + n + 1];
  if (fx + fz <= 1) return h00 + (h10 - h00) * fx + (h01 - h00) * fz;
  return h11 + (h01 - h11) * (1 - fx) + (h10 - h11) * (1 - fz);
}

double Terrain::groundAt(double x, double z) const {
  if (hole.on && (x - hole.x) * (x - hole.x) + (z - hole.z) * (z - hole.z) < hole.r * hole.r) return hole.y;
  return gridAt(x, z);
}

V3 Terrain::normalAt(double x, double z) const {
  double e = cell * 0.5;
  double nx = gridAt(x - e, z) - gridAt(x + e, z), nz = gridAt(x, z - e) - gridAt(x, z + e);
  double l = std::sqrt(nx * nx + (2 * e) * (2 * e) + nz * nz);
  return {nx / l, 2 * e / l, nz / l};
}

double Terrain::waterAt(double x, double z) const {
  const double NINF = -std::numeric_limits<double>::infinity();
  if (hole.on && (x - hole.x) * (x - hole.x) + (z - hole.z) * (z - hole.z) < hole.r * hole.r) return NINF;
  for (const auto& L : lakes) if (x > L.x0 && x < L.x1 && z > L.z0 && z < L.z1) return L.level;
  return NINF;
}

} // namespace fohn
