// Lakes (planar reflection, depth-dependent glacial colour, shore foam, sun glitter, jet wash),
// rivers (flowing ribbons with white water on the steep reaches) and the spray at the falls.
// Port of js/render/water.js.
#pragma once
#include <vector>
#include "prog.hpp"

namespace fohn {

class WaterR {
 public:
  void init(GLuint auxTex);
  // The mirror of the main lake. beginMirror() returns false when the lake cannot be seen from
  // here; otherwise it binds the mirror's frame buffer and fills `mc` with the mirrored camera:
  // draw the sky and the ground with clipY() and mirror = true, then call endMirror().
  bool beginMirror(const Camera& cam, Camera& mc);
  void endMirror();
  float clipY() const { return level_ - 0.4f; }
  void setWash(float x, float z, float radius, float strength) { wash_[0] = x; wash_[1] = z; wash_[2] = radius; wash_[3] = strength; }
  void drawLakes(const Camera& cam);
  void drawRivers(const Camera& cam, int viewH);     // transparent: after everything solid

 private:
  struct Mesh { GLuint vbo = 0, ibo = 0; int count = 0; };
  struct LakeM { Mesh m; float level; float out[4]; };
  std::vector<LakeM> lakes_;
  std::vector<Mesh> rivers_;
  GLuint mistVbo_ = 0; int mistCount_ = 0;
  Prog lakeProg_, riverProg_, mistProg_;
  GLuint auxTex_ = 0, fbo_ = 0, reflTex_ = 0, reflDepth_ = 0;
  GLint prevFbo_ = 0, prevVp_[4]{0, 0, 0, 0};
  static constexpr int RW = 1024, RH = 576;
  Mat4 reflMat_ = mat4Identity();
  bool hasRefl_ = false;
  float level_ = 0, wash_[4]{0, 0, 0, 0};

  void buildLakes();
  void buildRivers();
  void buildMist();
};

} // namespace fohn
