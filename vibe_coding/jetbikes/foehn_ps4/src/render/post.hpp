// The post pass. Port of js/render/post.js: the scene is drawn into a frame buffer, a bloom
// pyramid is built from what is really bright in it, and one last pass puts it on the screen with
// the bloom, the rays of the sun, the blur of the camera's own motion, the streaks of speed, the
// exposure, ACES, the grade, a vignette and grain.
//
// The web build has a half-float scene with a depth texture and 4x MSAA. Basic GLES2 promises none
// of the three, so here:
//  - the scene is 8 bits, the light packed as sqrt(c / (1 + c)) (outColor in env.cpp);
//  - what is solid writes its distance into the scene's alpha, and the blur reads it from there;
//  - light that is added (the plumes) has a buffer of its own that shares the scene's depth: in a
//    packed buffer a sum over an unknown background cannot be had from blending, over black it can;
//  - the bloom pyramid is 8 bits too, as the root of the light over a fixed range;
//  - no anti-aliasing yet.
#pragma once
#include <GLES2/gl2.h>
#include "prog.hpp"

namespace fohn {

class Post {
 public:
  // false: this GL has no 24-bit depth for a frame buffer. The frame is then drawn straight to
  // the screen, tone-mapped in each shader, as before the pass existed.
  bool init();
  bool resize(int w, int h);        // (re)allocates the targets when the size changes
  void beginScene();                // bind and clear the scene; remembers where the frame was going
  void beginLight();                // from here on: added light, tested against the scene's depth

  struct Opts { float blur = 0.5f, speed = 0, time = 0, exposure = 1, rays = 1; };
  void finish(const Camera& cam, const V3& sunDir, const Opts& o);   // back to the screen

 private:
  struct Target { GLuint fbo = 0, tex = 0; int w = 0, h = 0; };
  enum { LEVELS = 6 };
  bool make(Target& t, int w, int h, GLuint depth);
  void drop(Target& t);
  void quad(Prog& p);

  int w_ = 0, h_ = 0;
  GLenum depthFormat_ = 0; bool stencilToo_ = false;
  GLuint depth_ = 0, tri_ = 0;
  Target scene_, light_, down_[LEVELS], up_[LEVELS - 1];
  Prog first_, down_p_, up_p_, comp_;
  GLint outFbo_ = 0;
  Mat4 prevVP_{}; bool hasPrev_ = false;
};

} // namespace fohn
