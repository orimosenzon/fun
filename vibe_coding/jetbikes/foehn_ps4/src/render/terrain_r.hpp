// Terrain rendering. Port of js/render/terrain.js (without the real-time shadow map, which comes
// later): the whole 12 km square as 16x16 chunks with four levels
// of detail, shaded in the fragment shader (rock strata, scree, meadows, snow, shores, wet ground),
// photographs of real ground for the detail up close, the shadows of the mountains baked into a
// texture, cloud shadows and aerial perspective; and the far mountains beyond the playable square.
#pragma once
#include <cstdint>
#include <vector>
#include "prog.hpp"

namespace fohn {

class TerrainR {
 public:
  void init();
  void bakeSun(const V3& sun);          // call when the hour changes
  void draw(const Camera& cam, float clipY = -1e9f, bool mirror = false, float lodBias = 1);
  GLuint auxTex() const { return auxTex_; }
  // how far the photographs are filtered along a shallow view (1 = off, the web build asks for 8);
  // set before init(). It costs frame time, and is the first thing to turn down on the console
  static float photoAniso;

 private:
  static constexpr int CH = 64, W = CH + 1, NV = W * W + 4 * W, TEX = 1024;
  struct Chunk { GLuint vbo; float cx, cz, y0, y1; };
  std::vector<Chunk> chunks_;
  GLuint ibo_[4]{0, 0, 0, 0}; int idxCount_[4]{0, 0, 0, 0};
  Prog prog_, photoProg_, backProg_;          // painted ground; the same with the photographs; far mountains
  GLuint normalTex_ = 0, auxTex_ = 0;
  GLuint photoTex_[4]{0, 0, 0, 0}; bool photos_ = false;      // rock, turf, gravel, both normal maps
  std::vector<uint8_t> nodeSun_, nodeAO_, nodeBed_, aux_;   // per grid node; aux_ is the texture (per cell)
  struct Band { GLuint vbo, ibo; int count; };
  std::vector<Band> back_;

  void buildChunks();
  void buildTextures();
  void uploadAux();
  void buildBackdrop();
  bool loadPhotos(const char* path);
};

} // namespace fohn
