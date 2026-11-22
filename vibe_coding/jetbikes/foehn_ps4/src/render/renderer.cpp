#include "renderer.hpp"
#include "bike_mesh.hpp"
#include "../sim/terrain.hpp"
#include <cmath>

namespace fohn {

bool Renderer::postWanted = true;

// The bike: every part has the material bikeModel.js gives it (metalness, roughness, clear coat),
// lit by pbr() of env.cpp. Two-sided (open tubes, thin panels). What has a glow slot only shines,
// with the light the simulated engine and the lamps give it.
// (GLES2: the glow slot is picked in the vertex shader with a constant-bound loop.)
static const char* BIKE_VS = R"(
attribute vec3 aPos; attribute vec3 aNrm; attribute vec3 aCol; attribute float aGlow; attribute vec3 aMat;
uniform mat4 uVP; uniform mat4 uModel; uniform vec3 uGlow[16];
varying vec3 vN; varying vec3 vC; varying vec3 vW; varying vec3 vE; varying vec4 vM;
void main(){
  vec4 w = uModel * vec4(aPos,1.0); vW = w.xyz; gl_Position = uVP * w;
  vN = (uModel * vec4(aNrm,0.0)).xyz; vC = aCol; vE = vec3(0.0); vM = vec4(aMat, 1.0);
  if (aGlow > -0.5) { vM.w = 0.0; for (int k = 0; k < 16; k++) { if (abs(float(k) - aGlow) < 0.5) vE = uGlow[k]; } }
}
)";
static const char* BIKE_FS = R"(
uniform float uAlpha;
varying vec3 vN; varying vec3 vC; varying vec3 vW; varying vec3 vE; varying vec4 vM;
void main(){
  vec3 v = normalize(cameraPosition - vW);
  vec3 n = normalize(vN);
  float facing = dot(n, v);
  if (uAlpha < 1.0 && facing < 0.0) discard;          // the screen: only the side towards the eye
  if (facing < 0.0) n = -n;
  vec3 col = vE;
  // (its own shadow on itself, with the margins three.js gives a material by default: half a
  //  metre along the sun, 3 cm off the surface. At 7 cm to the texel only the large shapes shade.)
  if (vM.w > 0.5) col = pbr(vC, vM.xyz, n, v, cloudShadow(vW) * realShadow(vW + n * 0.03, 0.0006));
  vec4 o = outSolid(applyFog(col, vW));
  if (uAlpha < 1.0) o.a = uAlpha;
  gl_FragColor = o;
}
)";

void Renderer::buildBike() {
  bikeProg_.build(BIKE_VS, GLSL_COMMON + GLSL_PBR + BIKE_FS);
  BikeMesh m = buildBikeMesh();
  bikeVerts_ = (int)(m.body.size() / BIKE_VSTRIDE);
  standVerts_ = (int)(m.stand.size() / BIKE_VSTRIDE);
  glGenBuffers(1, &bikeVbo_); glBindBuffer(GL_ARRAY_BUFFER, bikeVbo_);
  glBufferData(GL_ARRAY_BUFFER, m.body.size() * 4, m.body.data(), GL_STATIC_DRAW);
  glGenBuffers(1, &standVbo_); glBindBuffer(GL_ARRAY_BUFFER, standVbo_);
  glBufferData(GL_ARRAY_BUFFER, m.stand.size() * 4, m.stand.data(), GL_STATIC_DRAW);
  glassVerts_ = (int)(m.glass.size() / BIKE_VSTRIDE);
  glGenBuffers(1, &glassVbo_); glBindBuffer(GL_ARRAY_BUFFER, glassVbo_);
  glBufferData(GL_ARRAY_BUFFER, m.glass.size() * 4, m.glass.data(), GL_STATIC_DRAW);
}

void Renderer::drawMesh(GLuint vbo, int verts) {
  const int S = BIKE_VSTRIDE * 4;
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  bikeProg_.attr("aPos", 3, GL_FLOAT, false, S, 0);
  bikeProg_.attr("aNrm", 3, GL_FLOAT, false, S, 12);
  bikeProg_.attr("aCol", 3, GL_FLOAT, false, S, 24);
  bikeProg_.attr("aGlow", 1, GL_FLOAT, false, S, 36);
  bikeProg_.attr("aMat", 3, GL_FLOAT, false, S, 40);
  glDrawArrays(GL_TRIANGLES, 0, verts);
  bikeProg_.attrOff("aPos"); bikeProg_.attrOff("aNrm"); bikeProg_.attrOff("aCol"); bikeProg_.attrOff("aGlow"); bikeProg_.attrOff("aMat");
}

void Renderer::drawBike(const Camera& cam, const Bike& b) {
  bikeProg_.use();
  gEnv.apply(bikeProg_, cam);
  bikeProg_.m4("uVP", cam.vp);
  // design frame -> world: the pose about the centre of gravity, the design origin shifted by -cg
  bikeProg_.m4("uModel", mat4Mul(mat4FromPose(b.q, b.p), mat4Translate(scale(b.mp.cg, -1))));
  float glow[BIKE_GLOW_SLOTS * 3];
  bikeGlow(b, glow);
  glUniform3fv(bikeProg_.U("uGlow"), BIKE_GLOW_SLOTS, glow);
  bikeProg_.f1("uAlpha", 1.0f);
  glDisable(GL_CULL_FACE);
  drawMesh(bikeVbo_, bikeVerts_);
  if (b.agl < 2.5) drawMesh(standVbo_, standVerts_);
  // the screen, see-through (opacity 0.6 in the web build). The blend leaves the alpha of what is
  // behind it alone: in the post pass that is where the distance is kept.
  bikeProg_.f1("uAlpha", 0.6f);
  glEnable(GL_BLEND); glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
  drawMesh(glassVbo_, glassVerts_);
  glDisable(GL_BLEND);
  glEnable(GL_CULL_FACE);
}

// what stands in the sun's way near the bike: so far the bike itself, screen and stand and all
void Renderer::drawShadowCasters(const Bike& b) {
  Prog& p = shadow_.prog();
  p.m4("uModel", mat4Mul(mat4FromPose(b.q, b.p), mat4Translate(scale(b.mp.cg, -1))));
  auto mesh = [&](GLuint vbo, int verts) {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    p.attr("aPos", 3, GL_FLOAT, false, BIKE_VSTRIDE * 4, 0);
    glDrawArrays(GL_TRIANGLES, 0, verts);
  };
  mesh(bikeVbo_, bikeVerts_); mesh(glassVbo_, glassVerts_);
  if (b.agl < 2.5) mesh(standVbo_, standVerts_);
  p.attrOff("aPos");
}

void Renderer::init() {
  gEnv.init();
  shadowOk_ = shadow_.init();
  if (gT.hole.on) { gEnv.hole[0] = (float)gT.hole.x; gEnv.hole[1] = (float)gT.hole.z; gEnv.hole[2] = (float)gT.hole.r; }
  sky_.init();
  terrain_.init();
  water_.init(terrain_.auxTex());
  props_.init(terrain_.auxTex());
  weather_.init();
  fx_.init();
  postOn_ = postWanted && post_.init();
  buildBike();
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glEnable(GL_CULL_FACE); glCullFace(GL_BACK); glFrontFace(GL_CCW);
  setTime(gEnv.timeIdx, gEnv.wxIdx);
}

void Renderer::setTime(int timeIdx, int wxIdx) {
  gEnv.setTime(timeIdx, wxIdx);
  terrain_.bakeSun(gEnv.sunDir);
}

void Renderer::simulate(double dt, const Bike& bike, const V3& wind) {
  fx_.update(dt, bike, wind, !bike.crash);
  water_.setWash(fx_.washPos[0], fx_.washPos[1], 6, fx_.washStrength);
}

void Renderer::frame(const V3& eye, const V3& target, const Bike& bike, double dt, const V3& wind) {
  gEnv.time += (float)dt;
  Camera cam;
  cam.fovY = 64.0f * 3.14159265f / 180; cam.aspect = (float)width_ / (height_ > 0 ? height_ : 1);
  cam.znear = 0.3f; cam.zfar = 60000.0f;
  cam.look(eye, target);
  weather_.update(dt, cam, wind, 0, height_);
  if (postOn_ && !post_.resize(width_, height_)) postOn_ = false;
  gEnv.outMode = postOn_ ? 2 : 0;

  if (shadowOk_) { shadow_.begin(bike.p, gEnv.sunDir); drawShadowCasters(bike); shadow_.end(); }

  // the world upside down, for the main lake
  Camera mc;
  if (water_.beginMirror(cam, mc)) {
    sky_.draw(mc, water_.clipY(), true);
    terrain_.draw(mc, water_.clipY(), true, 2.5f);
    water_.endMirror();
  }

  if (postOn_) post_.beginScene();
  else {
    glViewport(0, 0, width_, height_);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  }
  sky_.draw(cam);
  terrain_.draw(cam);
  props_.draw(cam, height_);
  water_.drawLakes(cam);
  drawBike(cam, bike);
  water_.drawRivers(cam, height_);
  fx_.drawParticles(cam);
  weather_.draw(cam);
  if (postOn_) post_.beginLight();
  fx_.drawPlumes(cam, bike, postOn_);
  if (postOn_) {
    // (the numbers of js/main.js: the blur of play, streaks from 126 km/h and with the overdrive, no rays through an overcast)
    Post::Opts o;
    o.blur = 0.5f;
    o.speed = (float)(clamp((bike.airspeed - 35) / 80, 0.0, 1.0) + bike.abOut * 0.3);
    o.time = gEnv.time; o.exposure = gEnv.exposure; o.rays = 1.0f - gEnv.wx[0];
    post_.finish(cam, gEnv.sunDir, o);
  }
}

} // namespace fohn
