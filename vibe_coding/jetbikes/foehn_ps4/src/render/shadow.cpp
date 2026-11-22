#include "shadow.hpp"
#include "env.hpp"
#include <cmath>

namespace fohn {

static const char* VS = R"(
attribute vec3 aPos; uniform mat4 uLVP; uniform mat4 uModel;
void main() { gl_Position = uLVP * (uModel * vec4(aPos, 1.0)); }
)";
static const char* FS = R"(
precision highp float;
vec4 packDepth(float d) { vec4 e = fract(d * vec4(1.0, 255.0, 65025.0, 16581375.0)); return e - e.yzww * vec4(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 0.0); }
void main() { gl_FragColor = packDepth(gl_FragCoord.z); }
)";

// the box of the web build's shadow camera
static const float HALF = 70, DIST = 400, ZNEAR = 1, ZFAR = 900;

bool ShadowMap::init() {
  if (gEnv.shadowRows <= 0) return false;
  GLint prev = 0; glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev);
  glGenRenderbuffers(1, &depth_); glBindRenderbuffer(GL_RENDERBUFFER, depth_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, gEnv.atlasW, gEnv.atlasH);     // 16 bits over 899 m sort the casters; the depth that is compared is the packed one
  glGenFramebuffers(1, &fbo_); glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gEnv.lutTex, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_);
  bool ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
  glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev);
  if (!ok) return false;
  prog_.build(VS, FS);
  return true;
}

void ShadowMap::begin(const V3& centre, const V3& sunDir) {
  const int W = gEnv.atlasW, R = gEnv.shadowRows;
  // the box moves in whole texels, so that the edge of a shadow that stands still does not crawl
  V3 f = scale(sunDir, -1), s = norm(cross(f, {0, 1, 0})), u = cross(s, f);
  double tx = 2.0 * HALF / W, ty = 2.0 * HALF / R, x = dot(centre, s), y = dot(centre, u);
  V3 c = add(centre, add(scale(s, std::round(x / tx) * tx - x), scale(u, std::round(y / ty) * ty - y)));
  Mat4 lvp = mat4Mul(mat4Ortho(-HALF, HALF, -HALF, HALF, ZNEAR, ZFAR), mat4LookAt(madd(c, sunDir, DIST), c, {0, 1, 0}));
  // world -> 0..1 in the map and in depth
  Mat4 half = mat4Identity(); half[0] = half[5] = half[10] = 0.5f; half[12] = half[13] = half[14] = 0.5f;
  gEnv.shadowMat = mat4Mul(half, lvp); gEnv.shadowOn = true;

  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFbo_); glGetIntegerv(GL_VIEWPORT, prevVp_);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  // only the rows of the map: the sky table above them must stay as it is
  glViewport(0, 0, W, R); glEnable(GL_SCISSOR_TEST); glScissor(0, 0, W, R);
  glClearColor(1, 1, 1, 1); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_BLEND); glDisable(GL_CULL_FACE);        // thin panels and open tubes cast from both sides
  prog_.use(); prog_.m4("uLVP", lvp);
}

void ShadowMap::end() {
  glDisable(GL_SCISSOR_TEST); glEnable(GL_CULL_FACE);
  glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prevFbo_);
  glViewport(prevVp_[0], prevVp_[1], prevVp_[2], prevVp_[3]);
}

} // namespace fohn
