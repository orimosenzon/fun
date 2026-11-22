#include "post.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace fohn {

static const char* QUAD_VS = R"(
attribute vec2 aPos; varying vec2 vUv;
void main() { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }
)";

// the light back out of a packed pixel (see outColor): e = sqrt(c / (1 + c))
#define UNPACK "vec3 hdr(vec3 e) { vec3 r = min(e * e, vec3(0.99)); return r / (1.0 - r); }\n"

// Bloom, first step down: the scene and the added light, and of them only what is really bright.
// From here on the pyramid holds sqrt(light / 64).
static const char* FIRST_FS = R"(
precision highp float; uniform sampler2D tSrc, tLight; uniform vec2 uTexel; varying vec2 vUv;
)" UNPACK R"(
vec3 tap(vec2 uv) { return hdr(texture2D(tSrc, uv).rgb) + hdr(texture2D(tLight, uv).rgb); }
void main() {
  vec3 s = (tap(vUv + uTexel * vec2(-1.0, -1.0)) + tap(vUv + uTexel * vec2(1.0, -1.0)) + tap(vUv + uTexel * vec2(-1.0, 1.0)) + tap(vUv + uTexel * vec2(1.0, 1.0)) + tap(vUv)) * 0.2;
  s = min(s, vec3(40.0)); float l = dot(s, vec3(0.2126, 0.7152, 0.0722)); s *= smoothstep(0.9, 2.2, l);   // only what is really bright
  gl_FragColor = vec4(sqrt(s / 64.0), 1.0);
}
)";
static const char* DOWN_FS = R"(
precision highp float; uniform sampler2D tSrc; uniform vec2 uTexel; varying vec2 vUv;
vec3 tap(vec2 uv) { vec3 e = texture2D(tSrc, uv).rgb; return e * e; }
void main() {
  vec3 s = (tap(vUv + uTexel * vec2(-1.0, -1.0)) + tap(vUv + uTexel * vec2(1.0, -1.0)) + tap(vUv + uTexel * vec2(-1.0, 1.0)) + tap(vUv + uTexel * vec2(1.0, 1.0)) + tap(vUv)) * 0.2;
  gl_FragColor = vec4(sqrt(s), 1.0);
}
)";
static const char* UP_FS = R"(
precision highp float; uniform sampler2D tSrc, tAdd; uniform vec2 uTexel; varying vec2 vUv;
vec3 tap(sampler2D t, vec2 uv) { vec3 e = texture2D(t, uv).rgb; return e * e; }
void main() {
  vec3 s = tap(tSrc, vUv + uTexel * vec2(-1.0, -1.0)) + tap(tSrc, vUv + uTexel * vec2(1.0, -1.0)) + tap(tSrc, vUv + uTexel * vec2(-1.0, 1.0)) + tap(tSrc, vUv + uTexel * vec2(1.0, 1.0));
  gl_FragColor = vec4(sqrt(s * 0.25 + tap(tAdd, vUv)), 1.0);
}
)";
static const char* COMP_FS = R"(
precision highp float;
uniform sampler2D tScene, tLight, tBloom, tBloomLo;
uniform mat4 uPrevVP; uniform vec3 uCamR, uCamU, uCamF; uniform vec2 uTan;
uniform vec3 uSunUV; uniform vec2 uRes;
uniform float uBlur, uSpeed, uTime, uExposure, uRays;
varying vec2 vUv;
)" UNPACK R"(
vec3 bloomAt(sampler2D t, vec2 uv) { vec3 e = texture2D(t, uv).rgb; return e * e * 64.0; }
vec3 aces(vec3 x) { x *= 0.6; return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void main() {
  // how far away is this pixel? (the alpha of the scene is 1 / (1 + distance / 20), the sky is 0)
  float a = max(texture2D(tScene, vUv).a, 0.002);
  float w = (1.0 / a - 1.0) * 20.0;
  // camera motion blur: where was this point on screen a frame ago? (uPrevVP is relative to the camera of this frame)
  vec2 ndc = vUv * 2.0 - 1.0;
  vec4 pp = uPrevVP * vec4((uCamF + uCamR * (ndc.x * uTan.x) + uCamU * (ndc.y * uTan.y)) * w, 1.0);
  vec2 vel = (vUv - (pp.xy / pp.w * 0.5 + 0.5)) * uBlur;
  float nearMask = smoothstep(0.970, 0.984, 1.0 - 0.3 / max(w, 0.3));   // keep the bike itself sharp (the numbers are depth-buffer values of a camera whose near plane is 0.3 m)
  vel *= nearMask * step(0.0, pp.w);
  float vl = length(vel); if (vl > 0.006) vel *= 0.006 / vl;
  // radial speed streaks towards the edges
  vec2 c = vUv - 0.5;
  vel += c * dot(c, c) * uSpeed * 0.022 * nearMask;
  float j = hash(vUv * uRes + uTime) - 0.5;
  vec3 col = vec3(0.0);
  for (int i = 0; i < 7; i++) col += hdr(texture2D(tScene, vUv - vel * (float(i) + 0.5 * j) / 6.0).rgb);
  col /= 7.0;
  col += hdr(texture2D(tLight, vUv).rgb);
  // bloom
  col += bloomAt(tBloom, vUv) * 0.05;
  // crepuscular rays: march the blurred bright sky towards the sun
  if (uSunUV.z > 0.0 && uRays > 0.0) {
    vec2 d = (uSunUV.xy - vUv);
    float fall = smoothstep(1.25, 0.1, length(d * vec2(uRes.x / uRes.y, 1.0)));
    vec3 rays = vec3(0.0); float wt = 1.0;
    for (int i = 1; i <= 14; i++) { rays += bloomAt(tBloomLo, vUv + d * (float(i) + j) / 15.0) * wt; wt *= 0.93; }
    col += rays * 0.0045 * fall * uRays;
  }
  col *= uExposure;
  col = aces(col);
  // grade: a touch of contrast and warmth, slightly lifted blue shadows
  col = pow(col, vec3(1.0 / 2.2));
  col = mix(col, col * col * (3.0 - 2.0 * col), 0.22);
  float l = dot(col, vec3(0.3, 0.59, 0.11));
  col = mix(vec3(l), col, 1.10);
  col += vec3(0.0, 0.004, 0.012) * (1.0 - l);
  col *= 1.0 - 0.36 * pow(dot(c, c) * 2.0, 1.6);
  col += (hash(vUv * uRes + uTime * 7.0) - 0.5) * 0.012;
  gl_FragColor = vec4(col, 1.0);
}
)";

bool Post::init() {
  const char* ext = (const char*)glGetString(GL_EXTENSIONS);
  if (ext && std::strstr(ext, "OES_depth24")) depthFormat_ = 0x81A6;                                         // GL_DEPTH_COMPONENT24_OES
  else if (ext && std::strstr(ext, "OES_packed_depth_stencil")) { depthFormat_ = 0x88F0; stencilToo_ = true; }   // GL_DEPTH24_STENCIL8_OES
  else {
    // (16 bits between 0.3 m and 60 km would flicker everywhere)
    std::fprintf(stderr, "post: no 24-bit depth for a frame buffer on this GL, drawing straight to the screen\n");
    return false;
  }
  first_.build(QUAD_VS, FIRST_FS); down_p_.build(QUAD_VS, DOWN_FS); up_p_.build(QUAD_VS, UP_FS); comp_.build(QUAD_VS, COMP_FS);
  const float tri[6] = {-1, -1, 3, -1, -1, 3};
  glGenBuffers(1, &tri_); glBindBuffer(GL_ARRAY_BUFFER, tri_); glBufferData(GL_ARRAY_BUFFER, sizeof(tri), tri, GL_STATIC_DRAW);
  return true;
}

void Post::drop(Target& t) {
  if (t.fbo) glDeleteFramebuffers(1, &t.fbo);
  if (t.tex) glDeleteTextures(1, &t.tex);
  t = Target{};
}

// (sizes that are not powers of two: allowed in GLES2 without mip-maps and with clamping)
bool Post::make(Target& t, int w, int h, GLuint depth) {
  t.w = w; t.h = h;
  glGenTextures(1, &t.tex); glBindTexture(GL_TEXTURE_2D, t.tex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glGenFramebuffers(1, &t.fbo); glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.tex, 0);
  if (depth) {
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
    if (stencilToo_) glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depth);
  }
  return glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
}

bool Post::resize(int w, int h) {
  if (w == w_ && h == h_) return true;
  GLint prev = 0; glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev);
  drop(scene_); drop(light_);
  for (Target& t : down_) drop(t);
  for (Target& t : up_) drop(t);
  if (depth_) glDeleteRenderbuffers(1, &depth_);
  w_ = w; h_ = h; hasPrev_ = false;
  glGenRenderbuffers(1, &depth_); glBindRenderbuffer(GL_RENDERBUFFER, depth_);
  glRenderbufferStorage(GL_RENDERBUFFER, depthFormat_, w, h);
  bool ok = make(scene_, w, h, depth_) && make(light_, w, h, depth_);
  int bw = w, bh = h;
  for (int i = 0; i < LEVELS && ok; i++) {
    bw = std::max(2, bw >> 1); bh = std::max(2, bh >> 1);
    ok = make(down_[i], bw, bh, 0) && (i == LEVELS - 1 || make(up_[i], bw, bh, 0));
  }
  glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev);
  if (!ok) { std::fprintf(stderr, "post: frame buffer incomplete at %dx%d, drawing straight to the screen\n", w, h); w_ = h_ = 0; }
  return ok;
}

void Post::beginScene() {
  glGetIntegerv(GL_FRAMEBUFFER_BINDING, &outFbo_);
  glBindFramebuffer(GL_FRAMEBUFFER, scene_.fbo);
  glViewport(0, 0, w_, h_);
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Post::beginLight() {
  glBindFramebuffer(GL_FRAMEBUFFER, light_.fbo);
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT);
}

void Post::quad(Prog& p) {
  glBindBuffer(GL_ARRAY_BUFFER, tri_);
  p.attr("aPos", 2, GL_FLOAT, false, 0, 0);
  glDrawArrays(GL_TRIANGLES, 0, 3);
  p.attrOff("aPos");
}

static void bindTex(int unit, GLuint tex) { glActiveTexture(GL_TEXTURE0 + unit); glBindTexture(GL_TEXTURE_2D, tex); }

void Post::finish(const Camera& cam, const V3& sunDir, const Opts& o) {
  glDisable(GL_DEPTH_TEST); glDepthMask(GL_FALSE); glDisable(GL_BLEND); glDisable(GL_CULL_FACE);

  // bloom pyramid: down, each step half the size, then back up, each step adding the one below
  const Target* src = &scene_;
  for (int i = 0; i < LEVELS; i++) {
    Prog& p = i == 0 ? first_ : down_p_;
    glBindFramebuffer(GL_FRAMEBUFFER, down_[i].fbo); glViewport(0, 0, down_[i].w, down_[i].h);
    p.use(); p.i1("tSrc", 0); bindTex(0, src->tex);
    if (i == 0) { p.i1("tLight", 1); bindTex(1, light_.tex); }
    p.f2("uTexel", 1.0f / src->w, 1.0f / src->h);
    quad(p);
    src = &down_[i];
  }
  const Target* acc = &down_[LEVELS - 1];
  for (int i = LEVELS - 2; i >= 0; i--) {
    Prog& p = up_p_;
    glBindFramebuffer(GL_FRAMEBUFFER, up_[i].fbo); glViewport(0, 0, up_[i].w, up_[i].h);
    p.use(); p.i1("tSrc", 0); p.i1("tAdd", 1); bindTex(0, acc->tex); bindTex(1, down_[i].tex);
    p.f2("uTexel", 0.5f / down_[i + 1].w, 0.5f / down_[i + 1].h);
    quad(p);
    acc = &up_[i];
  }

  // composite, to where the frame was going
  glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)outFbo_); glViewport(0, 0, w_, h_);
  Prog& p = comp_;
  p.use();
  p.i1("tScene", 0); p.i1("tLight", 1); p.i1("tBloom", 2); p.i1("tBloomLo", 3);
  bindTex(0, scene_.tex); bindTex(1, light_.tex); bindTex(2, up_[0].tex); bindTex(3, up_[2].tex);
  glActiveTexture(GL_TEXTURE0);
  // the camera of the last frame, taken relative to where this one stands (small numbers: a
  // position in the valley has no sub-pixel left in a float)
  if (!hasPrev_) { prevVP_ = cam.vp; hasPrev_ = true; }
  p.m4("uPrevVP", mat4Mul(prevVP_, mat4Translate(cam.eye)));
  p.v3("uCamR", cam.right); p.v3("uCamU", cam.up); p.v3("uCamF", cam.fwd);
  const float ty = std::tan(cam.fovY / 2);
  p.f2("uTan", ty * cam.aspect, ty);
  // where the sun is on the screen
  const Mat4& m = cam.vp;
  const float sx = m[0] * (float)sunDir[0] + m[4] * (float)sunDir[1] + m[8] * (float)sunDir[2];
  const float sy = m[1] * (float)sunDir[0] + m[5] * (float)sunDir[1] + m[9] * (float)sunDir[2];
  const float sw = m[3] * (float)sunDir[0] + m[7] * (float)sunDir[1] + m[11] * (float)sunDir[2];
  const bool facing = dot(sunDir, cam.fwd) > 0.05 && sw > 1e-6f;
  p.f3("uSunUV", facing ? sx / sw * 0.5f + 0.5f : 0.5f, facing ? sy / sw * 0.5f + 0.5f : 0.5f, facing ? 1.0f : 0.0f);
  p.f2("uRes", (float)w_, (float)h_);
  p.f1("uBlur", o.blur); p.f1("uSpeed", o.speed); p.f1("uTime", std::fmod(o.time, 100.0f)); p.f1("uExposure", o.exposure); p.f1("uRays", o.rays);
  quad(p);
  prevVP_ = cam.vp;

  glEnable(GL_DEPTH_TEST); glDepthMask(GL_TRUE); glEnable(GL_CULL_FACE);
}

} // namespace fohn
