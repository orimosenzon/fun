#include "terrain_r.hpp"
#include "env.hpp"
#include "../sim/terrain.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace fohn {

static const int LODS[4] = {1, 2, 4, 8};     // grid step per level

static const char* VERT = R"(
attribute vec3 aPos; uniform mat4 uVP; varying vec3 vW;
void main() { vW = aPos; gl_Position = uVP * vec4(aPos, 1.0); }
)";
static const char* FRAG = R"(
uniform sampler2D uNormal, uAux; uniform vec4 uWorld;
uniform vec4 uLake; uniform vec4 uLakeBox, uLakeOut;
varying vec3 vW;
#ifdef PHOTO
// Photographs of real ground (assets/ground of the web build, from Poly Haven, CC0). Each one is
// divided by its own mean colour, so it adds only what the painted ground lacks (blades, pebbles,
// cracks) and the colours stay the ones chosen below. From far away the mip-maps return the mean:
// nothing changes. The textures hold the square root of the linear colour (GLES2 has no sRGB
// textures to count on); uTPhotoN carries both normal maps, rock in xy and turf in zw.
uniform sampler2D uTRock, uTGrass, uTGravel, uTPhotoN;
const vec3 ROCK_MEAN = vec3(0.2323, 0.1455, 0.0838), GRASS_MEAN = vec3(0.2918, 0.2482, 0.1198), GRAVEL_MEAN = vec3(0.2980, 0.2361, 0.1510);
// (half of the photograph's own colour is kept: all of it would tint the pale limestone with the brown of another rock)
vec3 photo(sampler2D t, vec2 uv, vec3 mean) { vec3 c = texture2D(t, uv).rgb; vec3 r = min(c * c / mean, vec3(2.6)); return mix(vec3(dot(r, vec3(0.3, 0.55, 0.15))), r, 0.5); }
#endif
void main() {
  if (vW.y < uClipY) discard;
  if (uHole.z > 0.0 && distance(vW.xz, uHole.xy) < uHole.z) discard;
  vec2 uv = (vW.xz + uWorld.x) / (2.0 * uWorld.x);
  vec3 n = texture2D(uNormal, uv, -1.0).xyz; n.xz = n.xz * 2.0 - 1.0; n = normalize(n);
  vec4 aux = texture2D(uAux, uv);
  float h = vW.y, dist = length(vW - cameraPosition);
  float nearK = 1.0 - smoothstep(60.0, 420.0, dist), midK = 1.0 - smoothstep(500.0, 3500.0, dist);
  // noise at several scales
  float nA = fbm3(vW.xz / 310.0), nB = fbm3(vW.xz / 41.0), nC = vnoise(vW.xz / 5.3), nD = vnoise(vW.xz / 0.9), nE = fbm3(vW.xz / 900.0 + 40.0);
  // large-scale slope decides the material (the fine normal only shades it)
  vec3 nL = texture2D(uNormal, uv, 1.5).xyz; nL.xz = nL.xz * 2.0 - 1.0; nL = normalize(nL);
  float slopeL = 1.0 - nL.y;
  float rockW = smoothstep(0.17, 0.26, slopeL * 0.8 + (1.0 - n.y) * 0.2 + (nB - 0.5) * 0.07 + (nC - 0.5) * 0.04 * nearK);
  // bedding planes: strata dip gently across the range and catch the light as ledges
  float strataH = h + vW.x * 0.11 + vW.z * 0.05 + (nA - 0.5) * 120.0;
  float st1 = vnoise(vec2(strataH * 0.55, nB * 5.0 + nA * 3.0)), st2 = vnoise(vec2(strataH * 0.09, 7.0 + nA * 2.0));
  float ledge = (vnoise(vec2((strataH + 0.8) * 0.55, nB * 5.0 + nA * 3.0)) - st1);
  // on steep faces project the detail from the side, or it smears down the wall
  float steep = smoothstep(0.22, 0.45, slopeL);
  float wX = abs(nL.x) + 1e-3, wZ = abs(nL.z) + 1e-3;
  float sC = (vnoise(vW.zy / 4.1) * wX + vnoise(vW.xy / 4.1) * wZ) / (wX + wZ);
  float sD = (fbm3(vW.zy / 17.0 + 3.0) * wX + fbm3(vW.xy / 17.0 + 3.0) * wZ) / (wX + wZ);
  float sE = (vnoise(vW.zy / 0.8) * wX + vnoise(vW.xy / 0.8) * wZ) / (wX + wZ);
  nC = mix(nC, sC, steep);
  // detail normals at three scales
  vec2 dn = vec2(vnoise(vW.xz / 2.3 + 7.0) - vnoise(vW.xz / 2.3 - 7.0), vnoise(vW.zx / 2.3 + 3.0) - vnoise(vW.zx / 2.3 - 3.0));
  vec2 dm = vec2(fbm3(vW.xz / 19.0 + 5.0) - fbm3(vW.xz / 19.0 - 5.0), fbm3(vW.zx / 19.0 + 9.0) - fbm3(vW.zx / 19.0 - 9.0));
  vec2 df = vec2(vnoise(vW.xz / 0.37 + 3.0) - vnoise(vW.xz / 0.37 - 3.0), vnoise(vW.zx / 0.37 + 5.0) - vnoise(vW.zx / 0.37 - 5.0));
  vec3 tang = normalize(vec3(n.x, 0.0, n.z) + 1e-4);
#ifdef PHOTO
  // the relief of the photographs: turf seen from above, rock from the side on the walls
  float photoN = 1.0 - smoothstep(40.0, 260.0, dist);
  vec2 gn = texture2D(uTPhotoN, vW.xz / 3.1).zw * 2.0 - 1.0;
  vec2 rT = texture2D(uTPhotoN, vW.xz / 9.0).xy * 2.0 - 1.0, rX = texture2D(uTPhotoN, vW.zy / 9.0).xy * 2.0 - 1.0, rZ = texture2D(uTPhotoN, vW.xy / 9.0).xy * 2.0 - 1.0;
  vec3 rockN = mix(vec3(rT.x, 0.0, rT.y), (vec3(0.0, rX.y, rX.x) * wX + vec3(rZ.x, rZ.y, 0.0) * wZ) / (wX + wZ), steep);
  n = normalize(n + (vec3(gn.x, 0.0, gn.y) * 0.45 * (1.0 - rockW) + rockN * 0.9 * rockW) * photoN);
#endif
  n = normalize(n + vec3(dm.x, 0.0, dm.y) * (1.5 * rockW + 0.10) * midK + vec3(dn.x, 0.0, dn.y) * (1.0 * rockW + 0.22) * nearK
      + vec3(df.x, 0.0, df.y) * (0.5 * rockW + 0.25) * (1.0 - smoothstep(20.0, 110.0, dist))
      + (tang * 0.9 - vec3(0.0, 0.5, 0.0)) * ledge * 1.3 * rockW * midK);
  float slope = 1.0 - n.y;
  // --- materials ---
  float alp = smoothstep(1500.0, 2150.0, h + (nA - 0.5) * 300.0);
  vec3 grass = mix(vec3(0.075, 0.135, 0.030), vec3(0.150, 0.185, 0.045), nA);
  grass = mix(grass, vec3(0.205, 0.195, 0.060), smoothstep(0.45, 0.75, nE) * 0.65);         // drier pasture
  grass = mix(grass, vec3(0.055, 0.105, 0.030), smoothstep(0.55, 0.8, 1.0 - nE) * 0.5);     // lush hollows
  grass = mix(grass, vec3(0.235, 0.225, 0.085), alp * 0.85);                                // alpine turf
  grass *= 0.78 + 0.44 * nB; grass *= 0.86 + 0.28 * nC * nearK; grass *= 0.9 + 0.2 * nD * (1.0 - smoothstep(15.0, 80.0, dist));
  // flowers in the meadows, seen close
  float fl = smoothstep(0.80, 0.95, vnoise(vW.xz / 0.22)) * smoothstep(0.5, 0.8, nD) * (1.0 - smoothstep(25.0, 140.0, dist)) * smoothstep(0.52, 0.66, nB) * (1.0 - alp);
  grass = mix(grass, mix(vec3(0.75, 0.62, 0.10), vec3(0.72, 0.68, 0.78), step(0.5, nC)), fl * 0.55);
  vec3 forest = vec3(0.028, 0.058, 0.024) * (0.8 + 0.5 * nB);
  vec3 col = mix(grass, forest, smoothstep(0.1, 0.7, aux.b) * 0.92);
#ifdef PHOTO
  // two sizes of every photograph, multiplied: the small one is the ground under the bike, the
  // large one breaks the repetition that the eye finds from the air
  float photoK = 1.0 - smoothstep(1200.0, 3200.0, dist);
  // (the photograph of the turf is flat, lit from straight above: its contrast is raised to read at all)
  vec3 turf = photo(uTGrass, vW.xz / 3.1, GRASS_MEAN) * mix(vec3(1.0), photo(uTGrass, vW.xz / 23.0, GRASS_MEAN), 0.8);
  col *= mix(vec3(1.0), pow(turf, vec3(1.45)), photoK);
  vec3 gravelP = mix(vec3(1.0), photo(uTGravel, vW.xz / 4.3, GRAVEL_MEAN) * mix(vec3(1.0), photo(uTGravel, vW.xz / 31.0, GRAVEL_MEAN), 0.6), 0.9 * photoK);
  vec3 rockNear = mix(photo(uTRock, vW.xz / 9.0, ROCK_MEAN), (photo(uTRock, vW.zy / 9.0, ROCK_MEAN) * wX + photo(uTRock, vW.xy / 9.0, ROCK_MEAN) * wZ) / (wX + wZ), steep);
  vec3 rockFar = mix(photo(uTRock, vW.xz / 67.0, ROCK_MEAN), (photo(uTRock, vW.zy / 67.0, ROCK_MEAN) * wX + photo(uTRock, vW.xy / 67.0, ROCK_MEAN) * wZ) / (wX + wZ), steep);
  vec3 rockP = mix(vec3(1.0), rockNear * mix(vec3(1.0), rockFar, 0.7), 0.9 * photoK);
#else
  vec3 gravelP = vec3(1.0), rockP = vec3(1.0);
#endif
  // rock: banded limestone, weathered streaks, lichen
  vec3 rock = mix(vec3(0.27, 0.25, 0.225), vec3(0.47, 0.44, 0.40), st1 * 0.35 + st2 * 0.65);
  rock = mix(rock, vec3(0.42, 0.33, 0.24), smoothstep(0.5, 0.85, nA) * 0.45);               // iron-stained bands
  float streak = (vnoise(vec2(vW.z, h * 0.2) / 3.0) * wX + vnoise(vec2(vW.x, h * 0.2) / 3.0) * wZ) / (wX + wZ);
  rock *= mix(1.0, 0.62 + 0.7 * streak, steep);                                             // water streaks down the walls
  rock *= 0.72 + 0.56 * mix(fbm3(vW.xz / 6.0), sD, steep);
  rock *= 0.88 + 0.24 * mix(nD, sE, steep) * nearK;
  float cav = mix(0.5, sD * 0.6 + sC * 0.4, rockW);                                         // crevices take less sky light
  vec3 scree = vec3(0.37, 0.355, 0.33) * (0.72 + 0.4 * nC) * (0.85 + 0.3 * nD) * gravelP;
  col = mix(col, scree, smoothstep(0.09, 0.19, slopeL + (nB - 0.5) * 0.1) * (0.3 + 0.7 * alp) * (1.0 - 0.6 * smoothstep(0.1, 0.6, aux.b)));
  col = mix(col, rock * rockP, rockW);
  // shore and lake bed
  float inLake = step(uLakeBox.x, vW.x) * step(vW.x, uLakeBox.z) * step(uLakeBox.y, vW.z) * step(vW.z, uLakeBox.w);
  float shore = inLake * (1.0 - smoothstep(0.3, 2.6, h - uLake.x)) * (1.0 - rockW);
  if (uLakeOut.z != 0.0 || uLakeOut.w != 0.0) shore *= 1.0 - smoothstep(-20.0, 10.0, dot(vW.xz - uLakeOut.xy, uLakeOut.zw));     // past the outlet it is the river's gravel, not a beach
  col = mix(col, vec3(0.42, 0.40, 0.34) * (0.7 + 0.5 * nC) * gravelP, shore);
  // the gravel of the rivers: pale stones in the bed and on the bars beside the water
  col = mix(col, vec3(0.33, 0.32, 0.295) * (0.6 + 0.6 * nC) * (0.75 + 0.5 * nD) * gravelP, smoothstep(0.3, 0.62, aux.a * (0.55 + 0.9 * nB) + (nC - 0.5) * 0.25) * 0.9 * (1.0 - 0.7 * rockW));
  // snow: above the snow line, less on steep and sun-facing rock, drifts in gullies
  col *= 1.0 - 0.3 * uWx.y;                                                                  // rain: wet ground is darker
  float snowLine = 2330.0 + (nA - 0.5) * 420.0 - (1.0 - aux.g) * 500.0 - n.z * 60.0 - uWx.z * 2200.0;   // (snowfall: down to the valley floor)
  float snow = smoothstep(snowLine - 60.0, snowLine + 90.0, h) * (1.0 - smoothstep(0.38 + 0.26 * uWx.z, 0.62 + 0.3 * uWx.z, slope + (nB - 0.5) * 0.2));
  snow = max(snow, smoothstep(snowLine - 300.0, snowLine - 80.0, h) * smoothstep(0.5, 0.75, nA * 0.6 + nB * 0.4) * (1.0 - rockW) * 0.85);
  col = mix(col, vec3(0.86, 0.89, 0.93), snow);
  // --- light ---
  float sunVis = aux.r * realShadow(vW, 0.0001) * cloudShadow(vW);
  float ndl = max(dot(n, uSunDir), 0.0);
  float ao = aux.g * (1.0 - 0.35 * smoothstep(0.2, 0.8, aux.b) * (1.0 - rockW));
  // sky light with direction: slopes facing the bright part of the sky get more of it
  vec3 amb = (mix(uAmbGnd, uAmbSky, n.y * 0.5 + 0.5) * 0.45 + skyLUT(normalize(n + vec3(0.0, 0.55, 0.0))) * 0.75) * ao * (0.55 + 0.9 * cav) * 1.25;
  amb += uSunCol * 0.085 * (1.0 - aux.r) * ao * (0.6 + 0.4 * n.y);                                             // light bounced off the sunlit side
  vec3 light = uSunCol * ndl * sunVis + amb;
  // snow glows a little in shade and sparkles in the sun
  light += snow * uAmbSky * 0.25;
  vec3 V = normalize(cameraPosition - vW), Hh = normalize(V + uSunDir);
  float spec = pow(max(dot(n, Hh), 0.0), 60.0) * snow * 0.5 * sunVis;
  col = col * light + uSunCol * spec;
  gl_FragColor = outSolid(applyFog(col, vW));
}
)";
static const char* BACK_VS = R"(
attribute vec3 aPos; attribute vec3 aNrm; uniform mat4 uVP; varying vec3 vW; varying vec3 vN;
void main() { vW = aPos; vN = aNrm; gl_Position = uVP * vec4(aPos, 1.0); }
)";
static const char* BACK_FS = R"(
varying vec3 vW; varying vec3 vN;
void main() {
  if (vW.y < uClipY) discard;
  float nA = fbm3(vW.xz / 310.0), nB = fbm3(vW.xz / 41.0);
  vec2 dm = vec2(fbm3(vW.xz / 60.0 + 5.0) - fbm3(vW.xz / 60.0 - 5.0), fbm3(vW.zx / 60.0 + 9.0) - fbm3(vW.zx / 60.0 - 9.0));
  vec3 n = normalize(vN + vec3(dm.x, 0.0, dm.y) * 1.2);
  float slope = 1.0 - normalize(vN).y + (nB - 0.5) * 0.1;
  vec3 col = mix(vec3(0.07, 0.13, 0.04), vec3(0.30, 0.29, 0.10), smoothstep(1500.0, 2150.0, vW.y));
  col = mix(col, vec3(0.37, 0.355, 0.33) * (0.8 + 0.4 * nA), smoothstep(0.15, 0.3, slope));
  col *= 1.0 - 0.3 * uWx.y;
  float sl = 2330.0 + (nA - 0.5) * 420.0 - uWx.z * 2200.0;
  col = mix(col, vec3(0.86, 0.89, 0.93), smoothstep(sl - 60.0, sl + 90.0, vW.y) * (1.0 - smoothstep(0.4 + 0.26 * uWx.z, 0.62 + 0.3 * uWx.z, slope + (nB - 0.5) * 0.2)));
  vec3 light = uSunCol * max(dot(n, uSunDir), 0.0) * cloudShadow(vW) + mix(uAmbGnd, uAmbSky, n.y * 0.5 + 0.5) * 0.45 + skyLUT(normalize(n + vec3(0.0, 0.55, 0.0))) * 0.75;
  gl_FragColor = outSolid(applyFog(col * light, vW));
}
)";

// two box blurs via running sums (as in the web build)
static std::vector<float> blur(const std::vector<float>& src, int n, int r) {
  std::vector<float> tmp((size_t)n * n), out((size_t)n * n);
  for (int j = 0; j < n; j++) {
    double s = 0;
    for (int i = -r; i < n; i++) {
      if (i + r < n) s += src[(size_t)j * n + i + r];
      if (i - r - 1 >= 0) s -= src[(size_t)j * n + i - r - 1];
      if (i >= 0) tmp[(size_t)j * n + i] = (float)(s / (std::min(n - 1, i + r) - std::max(0, i - r) + 1));
    }
  }
  for (int i = 0; i < n; i++) {
    double s = 0;
    for (int j = -r; j < n; j++) {
      if (j + r < n) s += tmp[(size_t)(j + r) * n + i];
      if (j - r - 1 >= 0) s -= tmp[(size_t)(j - r - 1) * n + i];
      if (j >= 0) out[(size_t)j * n + i] = (float)(s / (std::min(n - 1, j + r) - std::max(0, j - r) + 1));
    }
  }
  return out;
}

static void texParams(bool mips) {
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mips ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

// Anisotropic filtering where the driver has it (an extension in GLES2): it is what keeps the
// photographs sharp on ground seen at a shallow angle. Without it they only blur sooner.
float TerrainR::photoAniso = 8;
static void setAniso() {
  if (TerrainR::photoAniso <= 1) return;
  const char* ext = (const char*)glGetString(GL_EXTENSIONS);
  if (!ext || !std::strstr(ext, "GL_EXT_texture_filter_anisotropic")) return;
  GLfloat mx = 1; glGetFloatv(0x84FF /* MAX_TEXTURE_MAX_ANISOTROPY_EXT */, &mx);
  glTexParameterf(GL_TEXTURE_2D, 0x84FE /* TEXTURE_MAX_ANISOTROPY_EXT */, std::min(TerrainR::photoAniso, mx));
}
static void photoParams() {
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  setAniso();
}

// The ground photographs, baked by tools/bake_ground.py: three colour textures and one with both
// normal maps. A colour arrives as sRGB bytes; every mip level is averaged in linear light (so
// that from far away it is exactly the mean) and stored as its square root, which the shader squares.
bool TerrainR::loadPhotos(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  char magic[4]; uint32_t size = 0, count = 0;
  bool ok = std::fread(magic, 1, 4, f) == 4 && !std::memcmp(magic, "FGT1", 4) && std::fread(&size, 4, 1, f) == 1 && std::fread(&count, 4, 1, f) == 1
            && count == 4 && size >= 1 && size <= 4096 && (size & (size - 1)) == 0;
  float lin[256];
  for (int v = 0; v < 256; v++) { double c = v / 255.0; lin[v] = (float)(c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4)); }
  std::vector<uint8_t> raw, px;
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  for (int t = 0; ok && t < 4; t++) {
    uint32_t ch = 0;
    ok = std::fread(&ch, 4, 1, f) == 1 && ch == (t < 3 ? 3u : 4u);
    if (!ok) break;
    raw.resize((size_t)size * size * ch);
    ok = std::fread(raw.data(), 1, raw.size(), f) == raw.size();
    if (!ok) break;
    glGenTextures(1, &photoTex_[t]); glBindTexture(GL_TEXTURE_2D, photoTex_[t]);
    if (t == 3) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, raw.data());
      glGenerateMipmap(GL_TEXTURE_2D);
    } else {
      std::vector<float> cur(raw.size()), nx;
      for (size_t i = 0; i < raw.size(); i++) cur[i] = lin[raw[i]];
      for (int level = 0, s = (int)size;; level++, s /= 2) {
        px.resize((size_t)s * s * 3);
        for (size_t i = 0; i < px.size(); i++) px[i] = (uint8_t)std::lround(std::sqrt(cur[i]) * 255.0f);
        glTexImage2D(GL_TEXTURE_2D, level, GL_RGB, s, s, 0, GL_RGB, GL_UNSIGNED_BYTE, px.data());
        if (s == 1) break;
        int h = s / 2; nx.resize((size_t)h * h * 3);
        for (int j = 0; j < h; j++) for (int i = 0; i < h; i++) for (int c = 0; c < 3; c++) {
          size_t a = ((size_t)(2 * j) * s + 2 * i) * 3 + c;
          nx[((size_t)j * h + i) * 3 + c] = 0.25f * (cur[a] + cur[a + 3] + cur[a + (size_t)s * 3] + cur[a + (size_t)s * 3 + 3]);
        }
        cur.swap(nx);
      }
    }
    photoParams();
  }
  glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
  std::fclose(f);
  return ok;
}

void TerrainR::buildTextures() {
  const int n = gT.n; const std::vector<float>& H = gT.H; const double c = gT.cell;
  // normals at the nodes, then one texel per cell (1024 x 1024, a power of two: GLES2 mip-maps)
  std::vector<uint8_t> nd((size_t)n * n * 3);
  for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) {
    size_t k = (size_t)j * n + i;
    double hx = (H[(size_t)j * n + std::min(n - 1, i + 1)] - H[(size_t)j * n + std::max(0, i - 1)]) / (2 * c);
    double hz = (H[(size_t)std::min(n - 1, j + 1) * n + i] - H[(size_t)std::max(0, j - 1) * n + i]) / (2 * c);
    double l = 1 / std::sqrt(hx * hx + 1 + hz * hz);
    nd[k * 3] = (uint8_t)((-hx * l * 0.5 + 0.5) * 255); nd[k * 3 + 1] = (uint8_t)(l * 255); nd[k * 3 + 2] = (uint8_t)((-hz * l * 0.5 + 0.5) * 255);
  }
  std::vector<uint8_t> nt((size_t)TEX * TEX * 4);
  for (int j = 0; j < TEX; j++) for (int i = 0; i < TEX; i++) {
    size_t a = (size_t)j * n + i, t = ((size_t)j * TEX + i) * 4;
    for (int ch = 0; ch < 3; ch++) nt[t + ch] = (uint8_t)((nd[a * 3 + ch] + nd[(a + 1) * 3 + ch] + nd[(a + n) * 3 + ch] + nd[(a + n + 1) * 3 + ch] + 2) / 4);
    nt[t + 3] = 255;
  }
  glGenTextures(1, &normalTex_); glBindTexture(GL_TEXTURE_2D, normalTex_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, TEX, TEX, 0, GL_RGBA, GL_UNSIGNED_BYTE, nt.data());
  glGenerateMipmap(GL_TEXTURE_2D); texParams(true);

  // ambient occlusion from the difference to a blurred copy of the terrain
  std::vector<float> b1 = blur(H, n, 3), b2 = blur(H, n, 14);
  nodeAO_.resize((size_t)n * n); nodeSun_.assign((size_t)n * n, 255); nodeBed_.resize((size_t)n * n);
  for (size_t k = 0; k < (size_t)n * n; k++) {
    double ao = 1 - std::min(0.55, std::max(0.0, (b1[k] - H[k]) * 0.05 + (b2[k] - H[k]) * 0.008));
    nodeAO_[k] = (uint8_t)(ao * 255);
  }
  // the bed and the gravel of the rivers
  std::vector<float> bed((size_t)n * n, 0.0f);
  for (const auto& r : gT.rivers) for (const auto& p : r.pts) {
    double R = r.width / 2 + 5; int i0 = (int)std::lround((p[0] + gT.half) / c), j0 = (int)std::lround((p[2] + gT.half) / c), q = (int)std::ceil(R / c);
    for (int j = std::max(0, j0 - q); j <= std::min(n - 1, j0 + q); j++) for (int i = std::max(0, i0 - q); i <= std::min(n - 1, i0 + q); i++)
      if (std::hypot(-gT.half + i * c - p[0], -gT.half + j * c - p[2]) <= R) bed[(size_t)j * n + i] = 1;
  }
  std::vector<float> bb = blur(bed, n, 1);
  for (size_t k = 0; k < (size_t)n * n; k++) nodeBed_[k] = (uint8_t)(std::min(1.0f, bb[k] * 1.6f) * 255);
  glGenTextures(1, &auxTex_);
  aux_.resize((size_t)TEX * TEX * 4);
}

// aux texture: R sun visibility (baked), G ambient occlusion, B forest cover (none yet), A river bed
void TerrainR::uploadAux() {
  const int n = gT.n;
  auto cell = [&](const std::vector<uint8_t>& v, size_t a) { return (uint8_t)((v[a] + v[a + 1] + v[a + n] + v[a + n + 1] + 2) / 4); };
  for (int j = 0; j < TEX; j++) for (int i = 0; i < TEX; i++) {
    size_t a = (size_t)j * n + i, t = ((size_t)j * TEX + i) * 4;
    aux_[t] = cell(nodeSun_, a); aux_[t + 1] = cell(nodeAO_, a); aux_[t + 2] = 0; aux_[t + 3] = cell(nodeBed_, a);
  }
  glBindTexture(GL_TEXTURE_2D, auxTex_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, TEX, TEX, 0, GL_RGBA, GL_UNSIGNED_BYTE, aux_.data());
  glGenerateMipmap(GL_TEXTURE_2D); texParams(true);
}

// Shadows of the mountains: march from every node towards the sun over the height field.
void TerrainR::bakeSun(const V3& sun) {
  const int n = gT.n; const std::vector<float>& H = gT.H; const double c = gT.cell;
  double hl = std::hypot(sun[0], sun[2]); if (hl == 0) hl = 1e-6;
  double dx = sun[0] / hl, dz = sun[2] / hl, rise = sun[1] / hl;
  const int STEPS = 44;
  std::vector<uint8_t>& ad = nodeSun_;
  for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) {
    double h0 = H[(size_t)j * n + i] + 1.5;
    double vis = 1, d = c * 1.2;
    for (int s = 0; s < STEPS; s++) {
      double x = i + dx * d / c, z = j + dz * d / c;
      if (x < 0 || z < 0 || x > n - 1 || z > n - 1) break;
      double hr = h0 + rise * d, ht = H[(size_t)((int)z) * n + (int)x];
      double m = (hr - ht) / (d * 0.045);          // soft edge: 0 at the ray .. 1 when well clear
      if (m < vis) { vis = m; if (vis <= 0) { vis = 0; break; } }
      if (hr > 3600) break;
      d *= 1.16;
    }
    ad[(size_t)j * n + i] = (uint8_t)(std::max(0.0, std::min(1.0, vis)) * 255);
  }
  // 3x3 blur of the visibility to hide the grid
  std::vector<uint8_t> tmp((size_t)n * n);
  for (int j = 1; j < n - 1; j++) for (int i = 1; i < n - 1; i++) { int s = 0; for (int b = -1; b <= 1; b++) for (int a = -1; a <= 1; a++) s += ad[(size_t)(j + b) * n + i + a]; tmp[(size_t)j * n + i] = (uint8_t)(s / 9); }
  for (int j = 1; j < n - 1; j++) for (int i = 1; i < n - 1; i++) ad[(size_t)j * n + i] = tmp[(size_t)j * n + i];
  uploadAux();
}

void TerrainR::buildChunks() {
  const int n = gT.n; const std::vector<float>& H = gT.H; const double c = gT.cell, half = gT.half;
  const int nc = (n - 1) / CH;
  const float drop = LODS[3] * 5 + 2;
  std::vector<float> pos((size_t)NV * 3);
  for (int cj = 0; cj < nc; cj++) for (int ci = 0; ci < nc; ci++) {
    float ymin = 1e9f, ymax = -1e9f;
    for (int j = 0; j < W; j++) for (int i = 0; i < W; i++) {
      int gi = ci * CH + i, gj = cj * CH + j; float y = H[(size_t)gj * n + gi]; size_t k = ((size_t)j * W + i) * 3;
      pos[k] = (float)(-half + gi * c); pos[k + 1] = y; pos[k + 2] = (float)(-half + gj * c);
      ymin = std::min(ymin, y); ymax = std::max(ymax, y);
    }
    // skirt ring: the four edges again, dropped, to close the cracks between levels of detail
    for (int e = 0; e < 4; e++) for (int k = 0; k < W; k++) {
      int src = e == 0 ? k : e == 1 ? (W - 1) * W + k : e == 2 ? k * W : k * W + W - 1;
      size_t d = ((size_t)W * W + e * W + k) * 3;
      pos[d] = pos[(size_t)src * 3]; pos[d + 1] = pos[(size_t)src * 3 + 1] - drop; pos[d + 2] = pos[(size_t)src * 3 + 2];
    }
    Chunk ch;
    glGenBuffers(1, &ch.vbo); glBindBuffer(GL_ARRAY_BUFFER, ch.vbo);
    glBufferData(GL_ARRAY_BUFFER, pos.size() * 4, pos.data(), GL_STATIC_DRAW);
    ch.cx = pos[0] + (float)(CH * c / 2); ch.cz = pos[2] + (float)(CH * c / 2); ch.y0 = ymin - drop; ch.y1 = ymax;
    chunks_.push_back(ch);
  }
  // one index buffer per level of detail, shared by all chunks: it picks every step-th vertex
  for (int L = 0; L < 4; L++) {
    int s = LODS[L], m = CH / s;
    std::vector<uint16_t> idx;
    auto V = [&](int i, int j) { return (uint16_t)(j * s * W + i * s); };
    // same diagonal as the physics: (0,0)-(0,1)-(1,0) and (1,0)-(0,1)-(1,1)
    for (int j = 0; j < m; j++) for (int i = 0; i < m; i++) {
      uint16_t a = V(i, j), b = V(i + 1, j), cc = V(i, j + 1), d = V(i + 1, j + 1);
      idx.insert(idx.end(), {a, cc, b, b, cc, d});
    }
    auto strip = [&](int e, bool flip) {
      for (int i = 0; i < m; i++) {
        uint16_t a, b;
        if (e == 0) { a = V(i, 0); b = V(i + 1, 0); } else if (e == 1) { a = V(i, m); b = V(i + 1, m); }
        else if (e == 2) { a = V(0, i); b = V(0, i + 1); } else { a = V(m, i); b = V(m, i + 1); }
        uint16_t cc = (uint16_t)(W * W + e * W + i * s), d = (uint16_t)(W * W + e * W + (i + 1) * s);
        if (flip) idx.insert(idx.end(), {a, b, cc, b, d, cc}); else idx.insert(idx.end(), {a, cc, b, b, cc, d});
      }
    };
    strip(0, true); strip(1, false); strip(2, false); strip(3, true);
    idxCount_[L] = (int)idx.size();
    glGenBuffers(1, &ibo_[L]); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_[L]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * 2, idx.data(), GL_STATIC_DRAW);
  }
}

// far mountains beyond the playable square (coarse grid with a hole in the middle), in bands that
// fit 16-bit indices
void TerrainR::buildBackdrop() {
  if (gT.B.empty()) return;
  const int n = gT.bn; const double c = gT.bcell, half = gT.bhalf; const std::vector<float>& u = gT.B;
  const double inner = gT.half - c;      // overlap one cell under the main terrain
  const int rowsPer = 65535 / n - 1;
  for (int j0 = 0; j0 < n - 1; j0 += rowsPer) {
    int j1 = std::min(n - 1, j0 + rowsPer);
    std::vector<float> v; v.reserve((size_t)(j1 - j0 + 1) * n * 6);
    for (int j = j0; j <= j1; j++) for (int i = 0; i < n; i++) {
      double hx = (u[(size_t)j * n + std::min(n - 1, i + 1)] - u[(size_t)j * n + std::max(0, i - 1)]) / (2 * c);
      double hz = (u[(size_t)std::min(n - 1, j + 1) * n + i] - u[(size_t)std::max(0, j - 1) * n + i]) / (2 * c);
      double l = 1 / std::sqrt(hx * hx + 1 + hz * hz);
      v.insert(v.end(), {(float)(-half + i * c), u[(size_t)j * n + i] - 6.0f, (float)(-half + j * c), (float)(-hx * l), (float)l, (float)(-hz * l)});
    }
    std::vector<uint16_t> idx;
    for (int j = j0; j < j1; j++) for (int i = 0; i < n - 1; i++) {
      double x0 = -half + i * c, z0 = -half + j * c;
      if (x0 > -inner && x0 + c < inner && z0 > -inner && z0 + c < inner) continue;
      uint16_t a = (uint16_t)((j - j0) * n + i), b = a + 1, d = (uint16_t)(a + n), e = d + 1;
      idx.insert(idx.end(), {a, d, b, b, d, e});
    }
    Band bd; bd.count = (int)idx.size();
    glGenBuffers(1, &bd.vbo); glBindBuffer(GL_ARRAY_BUFFER, bd.vbo); glBufferData(GL_ARRAY_BUFFER, v.size() * 4, v.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &bd.ibo); glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bd.ibo); glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * 2, idx.data(), GL_STATIC_DRAW);
    back_.push_back(bd);
  }
}

void TerrainR::init() {
  prog_.build(VERT, GLSL_COMMON + FRAG);
  // with the photographs: a second program, for the main view only (the lake's mirror, rippled
  // and at a quarter of the pixels, would not show them)
  photos_ = loadPhotos("assets/ground.bin");
  if (photos_) photoProg_.build(VERT, GLSL_COMMON + "#define PHOTO\n" + FRAG);
  else std::fprintf(stderr, "no assets/ground.bin (run tools/bake_ground.py): the ground stays painted, without photographs\n");
  backProg_.build(BACK_VS, GLSL_COMMON + BACK_FS);
  buildTextures();
  buildChunks();
  buildBackdrop();
}

// is the box outside one of the six planes of the view frustum?
static bool culled(const Mat4& m, float x0, float y0, float z0, float x1, float y1, float z1) {
  for (int p = 0; p < 6; p++) {
    int r = p / 2; float sg = (p & 1) ? -1.0f : 1.0f;
    float a = m[3] + sg * m[r], b = m[7] + sg * m[4 + r], c = m[11] + sg * m[8 + r], d = m[15] + sg * m[12 + r];
    if (a * (a > 0 ? x1 : x0) + b * (b > 0 ? y1 : y0) + c * (c > 0 ? z1 : z0) + d < 0) return true;
  }
  return false;
}

void TerrainR::draw(const Camera& cam, float clipY, bool mirror, float lodBias) {
  const bool photo = photos_ && !mirror;
  Prog& p = photo ? photoProg_ : prog_;
  p.use();
  gEnv.apply(p, cam, clipY, mirror);
  p.m4("uVP", cam.vp);
  p.f4("uWorld", (float)gT.half, (float)gT.cell, (float)gT.n, 0);
  if (!gT.lakes.empty()) {
    const Lake& L = gT.lakes[0];
    p.f4("uLake", (float)L.level, 0, 0, 0);
    p.f4("uLakeBox", (float)L.x0, (float)L.z0, (float)L.x1 + 300, (float)L.z1);
    p.f4("uLakeOut", (float)L.out[0], (float)L.out[1], (float)L.out[2], (float)L.out[3]);
  }
  p.i1("uNormal", TU_NORMAL); p.i1("uAux", TU_AUX);
  glActiveTexture(GL_TEXTURE0 + TU_NORMAL); glBindTexture(GL_TEXTURE_2D, normalTex_);
  glActiveTexture(GL_TEXTURE0 + TU_AUX); glBindTexture(GL_TEXTURE_2D, auxTex_);
  if (photo) {
    static const char* NAMES[4] = {"uTRock", "uTGrass", "uTGravel", "uTPhotoN"};
    for (int t = 0; t < 4; t++) { p.i1(NAMES[t], TU_PHOTO + t); glActiveTexture(GL_TEXTURE0 + TU_PHOTO + t); glBindTexture(GL_TEXTURE_2D, photoTex_[t]); }
  }
  glActiveTexture(GL_TEXTURE0);
  glDisable(GL_CULL_FACE);                 // (the web build draws the ground double-sided as well)
  GLint aPos = p.A("aPos");
  glEnableVertexAttribArray(aPos);
  const float hs = (float)(CH * gT.cell / 2);
  int bound = -1;
  for (const Chunk& ch : chunks_) {
    if (ch.y1 < clipY) continue;
    if (culled(cam.vp, ch.cx - hs, ch.y0, ch.cz - hs, ch.cx + hs, ch.y1, ch.cz + hs)) continue;
    float dx = std::max(0.0f, std::fabs((float)cam.eye[0] - ch.cx) - hs), dz = std::max(0.0f, std::fabs((float)cam.eye[2] - ch.cz) - hs);
    float dy = std::max(0.0f, std::max((float)cam.eye[1] - ch.y1, ch.y0 - (float)cam.eye[1]));
    float d = std::sqrt(dx * dx + dz * dz + dy * dy) * lodBias;
    int lod = d < 520 ? 0 : d < 1500 ? 1 : d < 3600 ? 2 : 3;
    if (lod != bound) { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_[lod]); bound = lod; }
    glBindBuffer(GL_ARRAY_BUFFER, ch.vbo);
    glVertexAttribPointer(aPos, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glDrawElements(GL_TRIANGLES, idxCount_[lod], GL_UNSIGNED_SHORT, (void*)0);
  }
  glDisableVertexAttribArray(aPos);

  if (!back_.empty()) {
    backProg_.use();
    gEnv.apply(backProg_, cam, clipY, mirror);
    backProg_.m4("uVP", cam.vp);
    for (const Band& b : back_) {
      glBindBuffer(GL_ARRAY_BUFFER, b.vbo);
      backProg_.attr("aPos", 3, GL_FLOAT, false, 24, 0);
      backProg_.attr("aNrm", 3, GL_FLOAT, false, 24, 12);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b.ibo);
      glDrawElements(GL_TRIANGLES, b.count, GL_UNSIGNED_SHORT, (void*)0);
    }
    backProg_.attrOff("aPos"); backProg_.attrOff("aNrm");
  }
  glEnable(GL_CULL_FACE);
}

} // namespace fohn
