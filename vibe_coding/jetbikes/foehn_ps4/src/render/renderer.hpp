// The frame: mirror pass for the lake, then sky, ground, the pads, the bike, water, dust and the weather into
// the scene, the plumes into the buffer of added light, and the post pass that puts it on the screen.
// All of it GLES2, the same path the PS4 build takes (Piglet / PS4GL_GNM).
#pragma once
#include <GLES2/gl2.h>
#include "prog.hpp"
#include "env.hpp"
#include "terrain_r.hpp"
#include "water_r.hpp"
#include "weather_r.hpp"
#include "fx.hpp"
#include "post.hpp"
#include "props_r.hpp"
#include "shadow.hpp"
#include "../sim/bike.hpp"

namespace fohn {

class Renderer {
 public:
  void init();
  void resize(int w, int h) { width_ = w; height_ = h; }
  // the hour (index into TIMES) and the weather (index into WEATHER): rebuilds the sky table and
  // re-bakes the shadows of the mountains
  void setTime(int timeIdx, int wxIdx);
  // once per simulated frame, whether it is drawn or not: plumes, dust and spray under the jets,
  // and the rings they leave on the water
  void simulate(double dt, const Bike& bike, const V3& wind);
  // draw one frame. dt advances the clouds, the ripples and the rain; wind is what the bike feels
  void frame(const V3& eye, const V3& target, const Bike& bike, double dt, const V3& wind);

  // false (--nopost): draw straight to the screen, tone-mapped in each shader. Set before init().
  static bool postWanted;
  bool post() const { return postOn_; }
  bool shadows() const { return shadowOk_; }

  int width() const { return width_; }
  int height() const { return height_; }

 private:
  int width_ = 1280, height_ = 720;
  Sky sky_;
  TerrainR terrain_;
  WaterR water_;
  WeatherR weather_;
  FX fx_;
  PropsR props_;
  ShadowMap shadow_;
  bool shadowOk_ = false;
  Post post_;
  bool postOn_ = false;
  // the F-5 (static meshes in the design frame)
  Prog bikeProg_;
  GLuint bikeVbo_ = 0, standVbo_ = 0, glassVbo_ = 0; int bikeVerts_ = 0, standVerts_ = 0, glassVerts_ = 0;
  void buildBike();
  void drawMesh(GLuint vbo, int verts);
  void drawBike(const Camera& cam, const Bike& b);
  void drawShadowCasters(const Bike& b);
};

} // namespace fohn
