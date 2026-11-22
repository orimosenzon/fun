// Sun, sky and air. Port of js/render/env.js + js/render/sky.js.
// A single-scattering atmosphere is ray-marched on the CPU into a small sky table whenever the
// hour or the weather changes; every shader reads its sky colour, its haze and its ambient light
// from that one table, so distant mountains fade into exactly the sky that is behind them.
//
// GLES2 has no float textures to count on, so the table is stored as RGBM in 8 bits (colour *
// alpha^2 * range) and every shader ends in outColor(): the light packed into 8 bits for the post
// pass (post.cpp), or, without that pass, the tone mapping and the grade themselves; in the lake's
// mirror a reversible encoding.
#pragma once
#include <string>
#include <vector>
#include "prog.hpp"

namespace fohn {

struct TimeOfDay { const char* key; double az, el, cloud, haze, exposure; };
struct WeatherDef { const char* key; double ov, cloud, sun, sky, fog, fog2, wet, snow; };
constexpr int N_TIMES = 5, N_WEATHER = 3;
extern const TimeOfDay TIMES[N_TIMES];
extern const WeatherDef WEATHER[N_WEATHER];
int timeIndex(const char* key);      // -1 if unknown
int weatherIndex(const char* key);

// texture units shared by all programs
// (GLES2 promises 8 in a fragment shader. The ground with its photographs uses all 8: units 4..7
// are the photographs there, and unit 4 is the lake's mirror in the water shader.)
enum { TU_SKY = 0, TU_NOISE = 1, TU_NORMAL = 2, TU_AUX = 3, TU_REFL = 4, TU_PHOTO = 4 };

struct Env {
  int timeIdx = 2, wxIdx = 0;
  V3 sunDir{0, 1, 0};
  float sunCol[3]{1, 1, 1}, ambSky[3]{0.3f, 0.4f, 0.6f}, ambGnd[3]{0.1f, 0.1f, 0.08f};
  float fog[4]{2.2e-5f, 1.0f / 1500, 900, 1};
  float cloud[4]{0.46f, 4300, 0, 0};
  float wx[4]{0, 0, 0, 0};             // overcast, wet ground, snow cover, -
  float hole[4]{0, 0, 0, 0};           // the shaft on the island: x, z, radius
  float sunAz = 0, time = 0, exposure = 1, lutRange = 1;
  float envSH[12]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};   // the sky as light on a surface: mean, then x, y, z (see setTime)
  float cloudEnv[4]{0, 1, 0, 0};       // the clouds evened out, for mirrors: cover, sunlit part, thin edge, cirrus (see setTime)
  int outMode = 0;                     // 0: shaders tone-map for the screen; 2: they pack the light for the post pass (see GLSL_COMMON)
  GLuint lutTex = 0, noiseTex = 0;
  // The sky table does not have a texture to itself: it sits in the top rows of the one the
  // shadow map is drawn into (shadow.cpp). The ground's shader already uses all eight texture
  // units GLES2 promises, and this way the shadows cost it none. shadowSize is the side of that
  // texture (0: no shadows, the texture is the table alone); set it before init().
  int shadowSize = 2048;
  int atlasW = 96, atlasH = 48, shadowRows = 0;      // the texture; the rows of it that are shadow map
  Mat4 shadowMat = mat4Identity(); bool shadowOn = false;   // world -> shadow map (0..1), once a map has been drawn
  std::vector<float> lut;              // the table in floats, for look-ups on the CPU

  void init();
  void setTime(int timeIdx, int wxIdx);
  V3 skyColor(double az, double el) const;
  V3 skyEnv(const V3& d) const;        // the sky with its clouds evened out, as GLSL_PBR's skyEnv()
  // common uniforms of a pass. clipY: nothing below it is drawn (the mirror); mirror: reversible output
  void apply(Prog& p, const Camera& cam, float clipY = -1e9f, bool mirror = false) const;
};
extern Env gEnv;

// shared GLSL for fragment shaders (declares precision, the common uniforms and helpers)
extern const std::string GLSL_COMMON;
// after GLSL_COMMON: pbr(), the surface model of the made things (metalness, roughness, clear coat)
extern const std::string GLSL_PBR;

class Sky {
 public:
  void init();
  void draw(const Camera& cam, float clipY = -1e9f, bool mirror = false);
 private:
  Prog prog_;
  GLuint vbo_ = 0;
};

} // namespace fohn
