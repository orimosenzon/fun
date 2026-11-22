// Rain and snow: a box of drops that travels with the camera. Each drop keeps its place in the
// world (the box wraps around), falls, and is carried by the wind that the physics feels, so a
// gust is something you can see coming across the valley. Rain is drawn as streaks whose length
// and direction are the drop's motion relative to the camera during the exposure: standing still
// it falls, at 300 km/h it comes at you almost level. Port of js/render/weather.js.
#pragma once
#include "prog.hpp"

namespace fohn {

class WeatherR {
 public:
  void init();
  // wind: what the physics feels at the bike (m/s, world). shelter: 0 in the open .. 1 under a roof
  void update(double dt, const Camera& cam, const V3& wind, double shelter, int viewH);
  void draw(const Camera& cam);

 private:
  struct Layer { Prog prog; GLuint vbo = 0; int count = 0; float box[3]; double off[3]{0, 0, 0}; float streak[3]{0, 0, 0}; float amt = 0, light[3]{1, 1, 1}, px = 900; };
  Layer rain_, snow_;
  V3 camPrev_{0, 0, 0}, camVel_{0, 0, 0};
  bool first_ = true;
};

} // namespace fohn
