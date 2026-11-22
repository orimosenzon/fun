// The shadow map near the bike. In the web build this is three.js' own: an orthographic box 140 m
// across that follows the bike, looked into from 400 m up the sun, 2048 texels a side. Here the
// same box is drawn by hand into the texture it shares with the sky table (env.hpp), as depth
// packed into a colour. What casts: the bike (the houses and the trees when they come). What takes
// the shadow: the ground, the pads and the bike itself, through realShadow() of GLSL_COMMON.
#pragma once
#include <GLES2/gl2.h>
#include "prog.hpp"

namespace fohn {

class ShadowMap {
 public:
  bool init();                                   // false: no shadows (asked off, or the frame buffer is refused)
  void begin(const V3& centre, const V3& sunDir);   // bind and clear the map; the box is centred on `centre`
  // the program the casters are drawn with: attribute aPos, uniform uModel (begin() has set the rest)
  Prog& prog() { return prog_; }
  void end();                                    // back to where the frame was going

 private:
  GLuint fbo_ = 0, depth_ = 0;
  Prog prog_;
  GLint prevFbo_ = 0, prevVp_[4]{0, 0, 0, 0};
};

} // namespace fohn
