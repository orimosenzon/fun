// The field on the GPU: one shader that turns a rectangle of the plane into (smooth count,
// distance estimate, gradient direction) per pixel. The map draws its picture from that, the
// landscape takes its heights from it, the mini-map is a small one of the same.
//
// The arithmetic is that of core/field.js, in single precision. Everything a pixel carries is
// kept relative to uScale (so a difference of 1e-30 is the number 1.0 here): that is what lets a
// 32-bit float follow a zoom of 10^32.
import { THREE } from '../foehn.js';
import { R2, SLC, BLA_W, BLA_LEVELS } from '../core/field.js';

export const FS_VERT = /* glsl */`
in vec3 position;
void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const FIELD_FRAG = /* glsl */`
precision highp float; precision highp int; precision highp sampler2D;
out vec4 outColor;
uniform sampler2D uRef, uBlaAB, uBlaR;
uniform int uRefLen, uIter, uPack, uBlaMax;
uniform int uBlaOff[${BLA_LEVELS + 1}], uBlaCnt[${BLA_LEVELS + 1}];
uniform float uScale;            // units of the plane per unit of uv
uniform vec2 uOff;               // centre of the rectangle from the reference point, in uScale
uniform vec2 uHalf;              // half size of the rectangle, in uScale
uniform vec2 uRes, uJit;         // size of the target in pixels, sub-pixel offset of the sample
uniform vec3 uC0;                // the reference point as plain floats; z = 1 when the view is shallow enough to use them
const float R2 = ${R2.toFixed(1)}, SLC = ${SLC.toFixed(1)};

vec2 refAt(int i) { return texelFetch(uRef, ivec2(i & 2047, i >> 11), 0).xy; }
vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
float amax(vec2 v) { return max(abs(v.x), abs(v.y)); }

void main() {
  vec2 uv = ((gl_FragCoord.xy + uJit) / uRes * 2.0 - 1.0) * uHalf;
  vec2 dc = uOff + uv;
  float sl = -1.0, dn = 0.0, ang = 0.0;
  bool inside = false;
  if (uC0.z > 0.5) {
    // the cardioid and the period-2 disc have closed formulas
    vec2 c = uC0.xy + dc * uScale;
    float xq = c.x - 0.25, q = xq * xq + c.y * c.y;
    inside = q * (q + xq) < 0.25 * c.y * c.y * 0.99999 || (c.x + 1.0) * (c.x + 1.0) + c.y * c.y < 0.0625 * 0.99999;
  }
  if (!inside) {
    vec2 dz = vec2(0.0), z = vec2(0.0), der = vec2(0.0);
    int m = 0, n = 0;
    float dcA = amax(dc) * 1.4143 * uScale;
    for (int step = 0; step < uIter; step++) {
      if (n >= uIter) break;
      bool hit = false;
      if (m > 0 && uBlaMax > 0) {
        // skip a whole block of the reference orbit while this pixel is still glued to it
        int q = m - 1;
        float dA = amax(dz) * 1.4143 * uScale;
        for (int l = uBlaMax; l >= 1; l--) {
          if ((q & ((1 << l) - 1)) != 0) continue;
          int j = q >> l;
          if (j >= uBlaCnt[l]) continue;
          int k = uBlaOff[l] + j;
          ivec2 tc = ivec2(k & ${BLA_W - 1}, k >> 11);
          vec2 rr = texelFetch(uBlaR, tc, 0).xy;
          if (!(dA < rr.x && dcA < rr.y)) continue;
          vec4 ab = texelFetch(uBlaAB, tc, 0);
          dz = cmul(ab.xy, dz) + cmul(ab.zw, dc);
          der = cmul(ab.xy, der) + ab.zw * uScale;
          m += 1 << l; n += 1 << l;
          z = refAt(m) + dz * uScale;
          if (m >= uRefLen - 1) { dz = z / uScale; m = 0; }
          hit = true; break;
        }
      }
      if (hit) continue;
      der = 2.0 * cmul(z, der) + vec2(uScale, 0.0);
      vec2 Z = refAt(m);
      dz = 2.0 * cmul(Z, dz) + cmul(dz * uScale, dz) + dc;
      m++; n++;
      z = refAt(m) + dz * uScale;
      float zz = dot(z, z);
      if (zz > R2) {
        float lz = 0.5 * log(zz);
        sl = float(n) - log2(lz) + SLC;
        float dm = amax(der);
        dn = sqrt(zz) * lz / (dm * length(der / dm));
        vec2 u = vec2(z.x * der.x + z.y * der.y, z.y * der.x - z.x * der.y);
        ang = atan(u.y, u.x);
        break;
      }
      if (amax(z) < amax(dz) * uScale || m >= uRefLen - 1) { dz = z / uScale; m = 0; }
    }
  }
  if (uPack == 1) {
    // half floats: the count is split so that its fraction keeps six bits wherever it is
    if (sl < 0.0) outColor = vec4(-1.0, 0.0, 0.0, 0.0);
    else { float hi = floor(sl / 32.0); outColor = vec4(hi, sl - hi * 32.0, dn, ang); }
  } else {
    outColor = vec4(sl < 0.0 ? float(uIter) + SLC : sl, dn, 0.0, 1.0);
  }
}`;

const tri = new THREE.BufferGeometry();
tri.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));

// a full-screen triangle with any raw material, into any target
export class Quad {
  constructor(renderer) {
    this.r = renderer; this.scene = new THREE.Scene(); this.cam = new THREE.Camera();
    this.mesh = new THREE.Mesh(tri, null); this.mesh.frustumCulled = false; this.scene.add(this.mesh);
  }
  draw(material, target = null, layer = 0) {
    this.mesh.material = material;
    this.r.setRenderTarget(target, layer);
    this.r.render(this.scene, this.cam);
  }
}
export const rawMat = (frag, uniforms, extra = {}) => new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, vertexShader: FS_VERT, fragmentShader: frag, uniforms, depthTest: false, depthWrite: false, ...extra });

const dataTex = (arr, w, h, format) => {
  const t = new THREE.DataTexture(arr, w, h, format, THREE.FloatType);
  t.minFilter = t.magFilter = THREE.NearestFilter; t.generateMipmaps = false; t.needsUpdate = true;
  return t;
};

export class FieldPass {
  constructor(renderer, field) {
    this.r = renderer; this.field = field; this.quad = new Quad(renderer); this.version = -1;
    this.u = {
      uRef: { value: null }, uBlaAB: { value: null }, uBlaR: { value: null },
      uRefLen: { value: 2 }, uIter: { value: 100 }, uPack: { value: 0 }, uBlaMax: { value: 0 },
      uBlaOff: { value: new Int32Array(BLA_LEVELS + 1) }, uBlaCnt: { value: new Int32Array(BLA_LEVELS + 1) },
      uScale: { value: 1 }, uOff: { value: new THREE.Vector2() }, uHalf: { value: new THREE.Vector2(1, 1) },
      uRes: { value: new THREE.Vector2(1, 1) }, uJit: { value: new THREE.Vector2() }, uC0: { value: new THREE.Vector3() },
    };
    this.mat = rawMat(FIELD_FRAG, this.u);
  }
  // the reference orbit and its skip table, as textures (again whenever the field built a new one)
  sync() {
    const f = this.field;
    if (this.version === f.version) return;
    this.version = f.version;
    for (const k of ['uRef', 'uBlaAB', 'uBlaR']) if (this.u[k].value) this.u[k].value.dispose();
    const rows = Math.max(1, Math.ceil(f.len / 2048)), ref = new Float32Array(rows * 2048 * 2);
    for (let i = 0, n = f.len * 2; i < n; i++) ref[i] = f.arr[i];
    this.u.uRef.value = dataTex(ref, 2048, rows, THREE.RGFormat);
    const b = f.bla;
    this.u.uBlaAB.value = dataTex(b.ab, BLA_W, b.rows, THREE.RGBAFormat);
    this.u.uBlaR.value = dataTex(b.rr, BLA_W, b.rows, THREE.RGBAFormat);
    this.u.uBlaOff.value = b.off; this.u.uBlaCnt.value = b.cnt; this.blaMax = b.maxL;
    this.u.uRefLen.value = f.len;
  }
  // o: { scale, offX, offY (centre from the reference, in units of the plane), halfX, halfY (in scale), iter, pack, jitX, jitY,
  //      strip: [y0, y1] rows of the target to fill (optional) }
  render(target, layer, o) {
    this.sync();
    const u = this.u, w = target.width, h = target.height;
    u.uScale.value = o.scale; u.uOff.value.set(o.offX / o.scale, o.offY / o.scale); u.uHalf.value.set(o.halfX, o.halfY);
    u.uRes.value.set(w, h); u.uJit.value.set(o.jitX || 0, o.jitY || 0);
    u.uIter.value = o.iter; u.uPack.value = o.pack ? 1 : 0;
    // skipping pays only where pixels stay close to the reference for long: in deep views
    u.uBlaMax.value = o.scale * Math.max(o.halfX, o.halfY) < 1e-4 ? this.blaMax : 0;
    const shallow = o.scale * Math.max(o.halfX, o.halfY) > 2e-4;
    u.uC0.value.set(this.field.c0[0], this.field.c0[1], shallow ? 1 : 0);
    if (o.strip) { target.scissor.set(0, o.strip[0], w, o.strip[1] - o.strip[0]); target.scissorTest = true; } else target.scissorTest = false;
    this.quad.draw(this.mat, target, layer);
    target.scissorTest = false;
  }
}
