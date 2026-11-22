// The F-5 and its rider as static triangle meshes in the physics design frame
// (x forward, y up, z right; origin on the ground under the bike). Port of js/render/bikeModel.js,
// built from the same primitives and dimensions, with the same materials. Vertex: pos3, nrm3,
// col3 (linear), glow slot (-1 = none), then metalness, roughness and the roughness of the clear
// coat (0 = none). What has a glow slot only shines: lamps and the mouths of the nozzles.
#pragma once
#include <vector>
#include "../sim/bike.hpp"

namespace fohn {

constexpr int BIKE_VSTRIDE = 13;        // floats per vertex
constexpr int BIKE_GLOW_SLOTS = 16;     // 0 core, 1-4 lift nozzles, 5 rear nozzle, 6-9 side exits, 10 head lamps, 11-12 nav, 13 tail, 14 strobe, 15 dash

struct BikeMesh {
  std::vector<float> body;   // always drawn
  std::vector<float> stand;  // the landing stand, drawn only near the ground
  std::vector<float> glass;  // the screen: drawn last, blended
};

BikeMesh buildBikeMesh();

// emissive colour of each glow slot from the simulated engine and nozzles (rgb * BIKE_GLOW_SLOTS)
void bikeGlow(const Bike& b, float out[BIKE_GLOW_SLOTS * 3]);

} // namespace fohn
