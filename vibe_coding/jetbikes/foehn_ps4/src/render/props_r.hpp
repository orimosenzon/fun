// What people have put in the valley. Port of js/render/props.js; so far the landing pads and the
// wooden pier (the houses, the church and the hangar come later).
//
// The web build paints each pad on a canvas and uses it as a texture. Here the same painting is
// worked out in the fragment shader (the asphalt, the yellow rim, the white circle and the letter:
// F on the base, H elsewhere), which needs no texture unit and stays sharp however near the camera.
#pragma once
#include <GLES2/gl2.h>
#include "prog.hpp"

namespace fohn {

class PropsR {
 public:
  void init(GLuint auxTex);                       // auxTex: the ground's texture, for the shadows of the mountains
  void draw(const Camera& cam, int screenH);
 private:
  Prog prog_;
  GLuint vbo_ = 0, aux_ = 0;
  int padVerts_ = 0, woodVerts_ = 0;
};

} // namespace fohn
