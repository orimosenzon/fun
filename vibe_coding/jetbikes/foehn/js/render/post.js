// Post-processing, hand written: HDR scene (MSAA) -> bloom pyramid -> composite with
// crepuscular rays, camera motion blur from the depth buffer, speed streaks, ACES tone
// mapping, colour grade, vignette and grain.
import * as THREE from '../../lib/three.module.min.js';

const FS_VERT = 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }';
const tri = new THREE.BufferGeometry();
tri.setAttribute('position', new THREE.Float32BufferAttribute([-1, -1, 0, 3, -1, 0, -1, 3, 0], 3));
tri.setAttribute('uv', new THREE.Float32BufferAttribute([0, 0, 2, 0, 0, 2], 2));

const DOWN = /* glsl */`
precision highp float; uniform sampler2D tSrc; uniform vec2 uTexel; uniform float uFirst; varying vec2 vUv;
void main() {
  vec3 a = texture2D(tSrc, vUv + uTexel * vec2(-1.0, -1.0)).rgb, b = texture2D(tSrc, vUv + uTexel * vec2(1.0, -1.0)).rgb;
  vec3 c = texture2D(tSrc, vUv + uTexel * vec2(-1.0, 1.0)).rgb, d = texture2D(tSrc, vUv + uTexel * vec2(1.0, 1.0)).rgb;
  vec3 e = texture2D(tSrc, vUv).rgb;
  vec3 s = (a + b + c + d) * 0.2 + e * 0.2;
  if (uFirst > 0.5) { s = min(s, vec3(40.0)); float l = dot(s, vec3(0.2126, 0.7152, 0.0722)); s *= smoothstep(0.9, 2.2, l); }   // only what is really bright
  gl_FragColor = vec4(s, 1.0);
}`;
const UP = /* glsl */`
precision highp float; uniform sampler2D tSrc, tAdd; uniform vec2 uTexel; varying vec2 vUv;
void main() {
  vec3 s = texture2D(tSrc, vUv + uTexel * vec2(-1.0, -1.0)).rgb + texture2D(tSrc, vUv + uTexel * vec2(1.0, -1.0)).rgb + texture2D(tSrc, vUv + uTexel * vec2(-1.0, 1.0)).rgb + texture2D(tSrc, vUv + uTexel * vec2(1.0, 1.0)).rgb;
  gl_FragColor = vec4(s * 0.25 + texture2D(tAdd, vUv).rgb, 1.0);
}`;
const COMP = /* glsl */`
precision highp float;
uniform sampler2D tScene, tDepth, tBloom, tBloomLo;
uniform mat4 uInvVP, uPrevVP; uniform vec3 uSunUV; uniform vec3 uSunCol; uniform vec2 uRes;
uniform float uBlur, uSpeed, uTime, uExposure, uFlash, uRays;
varying vec2 vUv;
vec3 aces(vec3 x) { x *= 0.6; return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void main() {
  float z = texture2D(tDepth, vUv).x;
  // camera motion blur: where was this point on screen a frame ago?
  vec4 wp = uInvVP * vec4(vUv * 2.0 - 1.0, z * 2.0 - 1.0, 1.0); wp /= wp.w;
  vec4 pp = uPrevVP * vec4(wp.xyz, 1.0);
  vec2 vel = (vUv - (pp.xy / pp.w * 0.5 + 0.5)) * uBlur;
  float nearMask = smoothstep(0.970, 0.984, z);             // keep the bike itself sharp
  vel *= nearMask * step(0.0, pp.w);
  float vl = length(vel); if (vl > 0.006) vel *= 0.006 / vl;
  // radial speed streaks towards the edges
  vec2 c = vUv - 0.5;
  vel += c * dot(c, c) * uSpeed * 0.022 * nearMask;
  float j = hash(vUv * uRes + uTime) - 0.5;
  vec3 col = vec3(0.0);
  for (int i = 0; i < 7; i++) col += texture2D(tScene, vUv - vel * (float(i) + 0.5 * j) / 6.0).rgb;
  col /= 7.0;
  // bloom
  vec3 bloom = texture2D(tBloom, vUv).rgb;
  col += bloom * 0.05;
  // crepuscular rays: march the blurred bright sky towards the sun
  if (uSunUV.z > 0.0 && uRays > 0.0) {
    vec2 d = (uSunUV.xy - vUv);
    float fall = smoothstep(1.25, 0.1, length(d * vec2(uRes.x / uRes.y, 1.0)));
    vec3 rays = vec3(0.0); float w = 1.0;
    for (int i = 1; i <= 14; i++) { rays += texture2D(tBloomLo, vUv + d * (float(i) + j) / 15.0).rgb * w; w *= 0.93; }
    col += rays * 0.0045 * fall * uRays;
  }
  col = mix(col, vec3(4.0, 3.0, 2.0), uFlash);
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
}`;

export class Post {
  constructor(renderer) {
    this.r = renderer; this.scene = new THREE.Scene(); this.cam = new THREE.Camera();
    this.quad = new THREE.Mesh(tri, null); this.quad.frustumCulled = false; this.scene.add(this.quad);
    const mk = (fs, uniforms) => new THREE.ShaderMaterial({ vertexShader: FS_VERT, fragmentShader: fs, uniforms, depthTest: false, depthWrite: false });
    this.mDown = mk(DOWN, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() }, uFirst: { value: 0 } });
    this.mUp = mk(UP, { tSrc: { value: null }, tAdd: { value: null }, uTexel: { value: new THREE.Vector2() } });
    this.mComp = mk(COMP, {
      tScene: { value: null }, tDepth: { value: null }, tBloom: { value: null }, tBloomLo: { value: null },
      uInvVP: { value: new THREE.Matrix4() }, uPrevVP: { value: new THREE.Matrix4() }, uSunUV: { value: new THREE.Vector3() }, uSunCol: { value: new THREE.Color() },
      uRes: { value: new THREE.Vector2() }, uBlur: { value: 1 }, uSpeed: { value: 0 }, uTime: { value: 0 }, uExposure: { value: 1 }, uFlash: { value: 0 }, uRays: { value: 1 },
    });
    this.vp = new THREE.Matrix4(); this.prevVP = new THREE.Matrix4(); this.first = true;
    this.msaa = 4;
  }
  setSize(w, h) {
    this.w = w; this.h = h;
    if (this.main) { this.main.dispose(); this.main.depthTexture.dispose(); this.down.forEach((t) => t.dispose()); this.up.forEach((t) => t.dispose()); }
    const depth = new THREE.DepthTexture(w, h); depth.type = THREE.UnsignedIntType; depth.minFilter = depth.magFilter = THREE.NearestFilter;
    this.main = new THREE.WebGLRenderTarget(w, h, { type: THREE.HalfFloatType, samples: this.msaa, depthTexture: depth, depthBuffer: true });
    this.main.texture.minFilter = this.main.texture.magFilter = THREE.LinearFilter;
    this.down = []; this.up = [];
    let bw = w, bh = h;
    for (let i = 0; i < 6; i++) {
      bw = Math.max(2, bw >> 1); bh = Math.max(2, bh >> 1);
      const o = { type: THREE.HalfFloatType, depthBuffer: false };
      this.down.push(new THREE.WebGLRenderTarget(bw, bh, o)); this.up.push(new THREE.WebGLRenderTarget(bw, bh, o));
    }
    this.mComp.uniforms.uRes.value.set(w, h);
  }
  pass(mat, target) { this.quad.material = mat; this.r.setRenderTarget(target); this.r.render(this.scene, this.cam); }
  // scene has been rendered into this.main by the caller
  finish(cam, sunDir, o) {
    const r = this.r;
    // bloom pyramid
    let src = this.main.texture, sw = this.w, sh = this.h;
    for (let i = 0; i < this.down.length; i++) {
      this.mDown.uniforms.tSrc.value = src; this.mDown.uniforms.uTexel.value.set(1 / sw, 1 / sh); this.mDown.uniforms.uFirst.value = i === 0 ? 1 : 0;
      this.pass(this.mDown, this.down[i]); src = this.down[i].texture; sw = this.down[i].width; sh = this.down[i].height;
    }
    let acc = this.down[this.down.length - 1].texture;
    for (let i = this.down.length - 2; i >= 0; i--) {
      this.mUp.uniforms.tSrc.value = acc; this.mUp.uniforms.tAdd.value = this.down[i].texture; this.mUp.uniforms.uTexel.value.set(0.5 / this.down[i + 1].width, 0.5 / this.down[i + 1].height);
      this.pass(this.mUp, this.up[i]); acc = this.up[i].texture;
    }
    // composite
    const u = this.mComp.uniforms;
    this.vp.multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    if (this.first) { this.prevVP.copy(this.vp); this.first = false; }
    u.uInvVP.value.copy(this.vp).invert(); u.uPrevVP.value.copy(this.prevVP);
    const s = new THREE.Vector3(sunDir[0], sunDir[1], sunDir[2]).multiplyScalar(1e5).add(cam.position).project(cam);
    const facing = new THREE.Vector3(sunDir[0], sunDir[1], sunDir[2]).dot(cam.getWorldDirection(new THREE.Vector3()));
    u.uSunUV.value.set(s.x * 0.5 + 0.5, s.y * 0.5 + 0.5, facing > 0.05 ? 1 : 0);
    u.tScene.value = this.main.texture; u.tDepth.value = this.main.depthTexture; u.tBloom.value = this.up[0].texture; u.tBloomLo.value = this.up[2].texture;
    u.uBlur.value = o.blur; u.uSpeed.value = o.speed; u.uTime.value = o.time % 100; u.uExposure.value = o.exposure; u.uFlash.value = o.flash || 0; u.uRays.value = o.rays;
    this.pass(this.mComp, null);
    this.prevVP.copy(this.vp);
  }
}
