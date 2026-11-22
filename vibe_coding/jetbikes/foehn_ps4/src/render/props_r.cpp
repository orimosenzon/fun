#include "props_r.hpp"
#include "env.hpp"
#include "../sim/terrain.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace fohn {

// vertex: position (world), normal, then the place on the painting: x to the right and y down in
// the canvas' own pixels from its centre (256 to the pad's radius), what it is (0 a pad with an F,
// 1 a pad with an H, 2 wood) and how many of those pixels there are to a metre
static const char* VS = R"(
attribute vec3 aPos; attribute vec3 aNrm; attribute vec4 aLoc;
uniform mat4 uVP;
varying vec3 vW; varying vec3 vN; varying vec4 vL;
void main() { vW = aPos; vN = aNrm; vL = aLoc; gl_Position = uVP * vec4(aPos, 1.0); }
)";
static const char* FS = R"(
uniform sampler2D uAux; uniform vec4 uWorld; uniform float uPix;
uniform vec3 uAsphalt, uYellow, uRing, uLetter, uWood;
varying vec3 vW; varying vec3 vN; varying vec4 vL;
// how much of a pixel of width w lies inside a rectangle (x0, y0, x1, y1), and inside a band of half-width hw
float inBox(vec2 p, vec4 r, float w) { vec2 d = min(p - r.xy, r.zw - p); return clamp(min(d.x, d.y) / w + 0.5, 0.0, 1.0); }
float inBand(float d, float hw, float w) { return clamp((hw - abs(d)) / w + 0.5, 0.0, 1.0) * min(1.0, 2.0 * hw / w); }
void main() {
  vec3 e = cameraPosition - vW; float dist = length(e); vec3 v = e / dist;
  vec3 n = normalize(vN);
  vec3 alb = uWood;
  if (vL.z < 1.5) {
    // the width of a screen pixel on the painting (there is no texture here to be filtered for us)
    float w = max(dist * uPix * vL.w / sqrt(max(abs(v.y), 0.02)), 0.05);
    vec2 q = vL.xy; float rho = length(q);
    // old asphalt: a scatter of lighter and darker grit
    vec4 gr = vnoise4(q * 0.83);
    float near = clamp(1.5 - w * 0.4, 0.0, 1.0);
    alb = uAsphalt * (1.0 + near * (0.25 * smoothstep(0.7, 0.8, gr.r) - 0.07 * smoothstep(0.7, 0.8, gr.g)));
    alb = mix(alb, uYellow, inBand(rho - 230.0, 8.0, w));
    alb = mix(alb, uRing, inBand(rho - 133.12, 4.0, w));
    // the letter, in the units of the type it was set in (bold sans, 200 px to 2048 units, its
    // baseline 81.2 px under the centre): both letters are three bars
    vec2 g = vec2(q.x, 81.2 - q.y) * 10.24; float gw = w * 10.24, let;
    if (vL.z < 0.5) { g.x += 699.5; let = max(inBox(g, vec4(188.0, 0.0, 573.0, 1493.0), gw), max(inBox(g, vec4(188.0, 1202.0, 1227.0, 1493.0), gw), inBox(g, vec4(188.0, 633.0, 1188.0, 924.0), gw))); }
    else { g.x += 857.0; let = max(inBox(g, vec4(188.0, 0.0, 573.0, 1493.0), gw), max(inBox(g, vec4(1141.0, 0.0, 1526.0, 1493.0), gw), inBox(g, vec4(188.0, 633.0, 1526.0, 924.0), gw))); }
    alb = mix(alb, uLetter, let);
  }
  vec2 uv = (vW.xz + uWorld.x) / (2.0 * uWorld.x);
  float sunVis = texture2D(uAux, uv).r * realShadow(vW, 0.0001) * cloudShadow(vW);
  if (dot(n, v) < 0.0) n = -n;
  gl_FragColor = outSolid(applyFog(pbr(alb, vec3(0.0, 0.9, 0.0), n, v, sunVis), vW));
}
)";

void PropsR::init(GLuint auxTex) {
  aux_ = auxTex;
  prog_.build(VS, GLSL_COMMON + GLSL_PBR + FS);
  std::vector<float> v;
  auto vert = [&](double x, double y, double z, const V3& n, double qx, double qy, double tag, double ppm) {
    v.push_back((float)x); v.push_back((float)y); v.push_back((float)z);
    v.push_back((float)n[0]); v.push_back((float)n[1]); v.push_back((float)n[2]);
    v.push_back((float)qx); v.push_back((float)qy); v.push_back((float)tag); v.push_back((float)ppm);
  };
  // the pads: a disc 6 cm over the ground. The canvas is 512 px across and the asphalt stops 4 px
  // short of its edge; the top of the canvas is towards -z, so the letter reads from the south.
  const int SEG = 48; const double EDGE = 252.0;
  const Pad* pier = nullptr;
  for (const Pad& p : gT.pads) {
    if (p.name == "pier") pier = &p;
    double r = p.name == "village" ? 16 : std::min(p.r * 0.62, 18.0), ppm = 256 / r, tag = p.name == "base" ? 0 : 1, y = p.y + 0.06;
    for (int i = 0; i < SEG; i++) {
      double a0 = 2 * PI * i / SEG, a1 = 2 * PI * (i + 1) / SEG;
      double q0x = std::cos(a0) * EDGE, q0y = std::sin(a0) * EDGE, q1x = std::cos(a1) * EDGE, q1y = std::sin(a1) * EDGE;
      vert(p.x, y, p.z, {0, 1, 0}, 0, 0, tag, ppm);
      vert(p.x + q1x / ppm, y, p.z + q1y / ppm, {0, 1, 0}, q1x, q1y, tag, ppm);
      vert(p.x + q0x / ppm, y, p.z + q0y / ppm, {0, 1, 0}, q0x, q0y, tag, ppm);
    }
  }
  padVerts_ = (int)(v.size() / 10);
  // the wooden pier: a deck 34 m square under its pad
  if (pier) {
    const double hx = 17, hy = 0.25, hz = 17, cx = pier->x, cy = pier->y - 0.3, cz = pier->z;
    auto face = [&](const V3& n, const V3& a, const V3& b) {      // a, b: the two half-edges that span the face
      V3 c = {n[0] * hx, n[1] * hy, n[2] * hz};
      V3 P[4] = {sub(sub(c, a), b), sub(add(c, a), b), add(add(c, a), b), add(sub(c, a), b)};
      for (int k : {0, 1, 2, 0, 2, 3}) vert(cx + P[k][0], cy + P[k][1], cz + P[k][2], n, 0, 0, 2, 1);
    };
    face({0, 1, 0}, {hx, 0, 0}, {0, 0, hz}); face({0, -1, 0}, {hx, 0, 0}, {0, 0, hz});
    face({1, 0, 0}, {0, hy, 0}, {0, 0, hz}); face({-1, 0, 0}, {0, hy, 0}, {0, 0, hz});
    face({0, 0, 1}, {hx, 0, 0}, {0, hy, 0}); face({0, 0, -1}, {hx, 0, 0}, {0, hy, 0});
  }
  woodVerts_ = (int)(v.size() / 10) - padVerts_;
  glGenBuffers(1, &vbo_); glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferData(GL_ARRAY_BUFFER, v.size() * 4, v.data(), GL_STATIC_DRAW);
}

void PropsR::draw(const Camera& cam, int screenH) {
  if (!padVerts_ && !woodVerts_) return;
  Prog& p = prog_;
  p.use();
  gEnv.apply(p, cam);
  p.m4("uVP", cam.vp);
  p.f4("uWorld", (float)gT.half, (float)gT.cell, (float)gT.n, 0);
  p.f1("uPix", 2 * std::tan(cam.fovY / 2) / (float)std::max(1, screenH));
  p.v3("uAsphalt", srgbHex(0x3a3d40)); p.v3("uYellow", srgbHex(0xf2c230)); p.v3("uRing", srgbHex(0xe8e8e8));
  p.v3("uLetter", srgbHex(0xf0f0f0)); p.v3("uWood", srgbHex(0x6b4a2e));
  p.i1("uAux", TU_AUX);
  glActiveTexture(GL_TEXTURE0 + TU_AUX); glBindTexture(GL_TEXTURE_2D, aux_);
  glActiveTexture(GL_TEXTURE0);
  const int S = 10 * 4;
  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  p.attr("aPos", 3, GL_FLOAT, false, S, 0);
  p.attr("aNrm", 3, GL_FLOAT, false, S, 12);
  p.attr("aLoc", 4, GL_FLOAT, false, S, 24);
  glDisable(GL_CULL_FACE);
  if (woodVerts_) glDrawArrays(GL_TRIANGLES, padVerts_, woodVerts_);
  // the pads lie on the ground: pulled towards the eye so that the two do not fight over the depth
  glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(-4, -4);
  glDrawArrays(GL_TRIANGLES, 0, padVerts_);
  glDisable(GL_POLYGON_OFFSET_FILL);
  glEnable(GL_CULL_FACE);
  p.attrOff("aPos"); p.attrOff("aNrm"); p.attrOff("aLoc");
}

} // namespace fohn
