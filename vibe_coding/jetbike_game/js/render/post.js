// Post pipeline (no EffectComposer):
//   scene -> HDR target (+ depth) ; haze volumes -> distortion target (depth-tested)
//   composite (refraction offset + bloom) -> HDR ; accumulate N sub-frames (motion blur,
//   sub-pixel jitter = anti-aliasing) ; present: exposure, ACES filmic, sRGB, vignette, grain.

import * as THREE from 'three';

const quadGeo = new THREE.PlaneGeometry(2, 2);
const orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const VS = /* glsl */`varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

function pass(frag, uniforms, extra = {}) {
  const m = new THREE.ShaderMaterial({ vertexShader: VS, fragmentShader: frag, uniforms, depthTest: false, depthWrite: false, ...extra });
  const mesh = new THREE.Mesh(quadGeo, m);
  mesh.frustumCulled = false;
  const sc = new THREE.Scene(); sc.add(mesh);
  // additive passes must not clear their target first
  const additive = extra.blending === THREE.AdditiveBlending;
  return {
    m, u: m.uniforms,
    draw(renderer, target) {
      const ac = renderer.autoClear;
      renderer.autoClear = !additive;
      renderer.setRenderTarget(target);
      renderer.render(sc, orthoCam);
      renderer.autoClear = ac;
    },
  };
}

export class Post {
  constructor(renderer, w, h) {
    this.r = renderer;
    this.setSize(w, h);
    this.down = pass(/* glsl */`
      uniform sampler2D tSrc; uniform vec2 uTexel; uniform float uThreshold; uniform bool uFirst;
      varying vec2 vUv;
      vec3 s(vec2 o){ return texture2D(tSrc, vUv + o*uTexel).rgb; }
      void main(){
        vec3 c = (s(vec2(-1,-1))+s(vec2(1,-1))+s(vec2(-1,1))+s(vec2(1,1)))*0.125
               + (s(vec2(-2,0))+s(vec2(2,0))+s(vec2(0,-2))+s(vec2(0,2)))*0.0625
               + s(vec2(0))*0.25;
        if (uFirst) { float l = max(c.r, max(c.g, c.b)); c *= max(0.0, l - uThreshold) / max(l, 1e-4); c = min(c, vec3(60.0)); }
        gl_FragColor = vec4(c, 1.0);
      }`, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() }, uThreshold: { value: 1.2 }, uFirst: { value: false } });
    this.up = pass(/* glsl */`
      uniform sampler2D tSrc; uniform vec2 uTexel; varying vec2 vUv;
      void main(){
        vec3 c = vec3(0.0);
        for (int i=-1;i<=1;i++) for (int j=-1;j<=1;j++) {
          float w = (i==0?2.0:1.0)*(j==0?2.0:1.0);
          c += texture2D(tSrc, vUv + vec2(float(i),float(j))*uTexel).rgb * w;
        }
        gl_FragColor = vec4(c/16.0, 1.0);
      }`, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() } }, { blending: THREE.AdditiveBlending, transparent: true });
    // composite: heat-haze refraction, camera motion blur (depth reprojection into the previous
    // frame; nearby pixels, i.e. the bike riding with the camera, are excluded), bloom
    this.comp = pass(/* glsl */`
      uniform sampler2D tScene, tHaze, tBloom, tDepth; uniform float uBloom, uBlur;
      uniform mat4 uInvViewProj, uPrevViewProj; uniform vec3 uCam;
      varying vec2 vUv;
      void main(){
        vec2 off = texture2D(tHaze, vUv).rg;
        vec2 uv = vUv + off;
        float d = texture2D(tDepth, uv).x;
        vec4 wp = uInvViewProj * vec4(uv*2.0-1.0, d*2.0-1.0, 1.0);
        vec3 w = wp.xyz / wp.w;
        vec4 pp = uPrevViewProj * vec4(w, 1.0);
        vec2 prevUv = pp.xy / pp.w * 0.5 + 0.5;
        vec2 vel = (uv - prevUv) * uBlur * smoothstep(5.0, 14.0, length(w - uCam));
        float vl = length(vel);
        if (vl > 0.06) vel *= 0.06 / vl;
        vec3 c = vec3(0.0);
        for (int i = 0; i < 8; i++) {
          float t = float(i) / 7.0 - 0.5;
          c += texture2D(tScene, uv - vel * t).rgb;
        }
        c /= 8.0;
        c += texture2D(tBloom, vUv).rgb * uBloom;
        gl_FragColor = vec4(c, 1.0);
      }`, {
      tScene: { value: null }, tHaze: { value: null }, tBloom: { value: null }, tDepth: { value: null }, uBloom: { value: 0.05 },
      uBlur: { value: 0 }, uInvViewProj: { value: new THREE.Matrix4() }, uPrevViewProj: { value: new THREE.Matrix4() }, uCam: { value: new THREE.Vector3() },
    });
    this.prevVP = null;
    this.accumP = pass(/* glsl */`
      uniform sampler2D tSrc; uniform float uW; varying vec2 vUv;
      void main(){ gl_FragColor = vec4(texture2D(tSrc, vUv).rgb * uW, 1.0); }`,
    { tSrc: { value: null }, uW: { value: 1 } }, { blending: THREE.AdditiveBlending, transparent: true });
    this.present = pass(/* glsl */`
      uniform sampler2D tSrc; uniform float uExposure, uTime, uFade; uniform vec2 uRes; varying vec2 vUv;
      vec3 RRTAndODTFit(vec3 v){ vec3 a = v*(v+0.0245786)-0.000090537; vec3 b = v*(0.983729*v+0.4329510)+0.238081; return a/b; }
      vec3 aces(vec3 c){
        const mat3 i = mat3(vec3(0.59719,0.07600,0.02840), vec3(0.35458,0.90834,0.13383), vec3(0.04823,0.01566,0.83777));
        const mat3 o = mat3(vec3(1.60475,-0.10208,-0.00327), vec3(-0.53108,1.10813,-0.07276), vec3(-0.07367,-0.00605,1.07602));
        c = i*c; c = RRTAndODTFit(c); c = o*c; return clamp(c,0.0,1.0);
      }
      float h(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233)))*43758.5453); }
      void main(){
        // slight lateral chromatic aberration toward the edges
        vec2 d = vUv - 0.5;
        vec3 c;
        c.r = texture2D(tSrc, vUv - d*0.0016).r;
        c.g = texture2D(tSrc, vUv).g;
        c.b = texture2D(tSrc, vUv + d*0.0016).b;
        c = aces(c * uExposure / 0.6);
        c = pow(c, vec3(1.0/2.2));
        // filmic grade: gentle warm highlights, cool shadows
        c = mix(c, c*vec3(1.02,1.0,0.97), smoothstep(0.4,1.0,c.g));
        c = mix(c*vec3(0.96,0.99,1.04), c, smoothstep(0.0,0.35,c.g));
        float vig = smoothstep(0.95, 0.25, length(d*vec2(1.0, uRes.y/uRes.x)*1.25));
        c *= mix(0.72, 1.0, vig);
        c += (h(vUv*uRes + uTime*61.0) - 0.5) * 0.018;
        c *= uFade;
        gl_FragColor = vec4(c, 1.0);
      }`, { tSrc: { value: null }, uExposure: { value: 1 }, uTime: { value: 0 }, uRes: { value: new THREE.Vector2() }, uFade: { value: 1 } });
  }

  setSize(w, h) {
    this.w = w; this.h = h;
    const hf = { type: THREE.HalfFloatType, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter };
    this.depthTexture = new THREE.DepthTexture(w, h);
    this.depthTexture.type = THREE.FloatType;
    this.rtScene = new THREE.WebGLRenderTarget(w, h, { ...hf, depthTexture: this.depthTexture });
    this.rtHaze = new THREE.WebGLRenderTarget(w >> 1, h >> 1, hf);
    this.rtComp = new THREE.WebGLRenderTarget(w, h, hf);
    this.rtAccum = new THREE.WebGLRenderTarget(w, h, hf);  // half float: blendable everywhere (float32 needs EXT_float_blend)
    this.bloom = [];
    let bw = w >> 1, bh = h >> 1;
    for (let i = 0; i < 6; i++) { this.bloom.push(new THREE.WebGLRenderTarget(Math.max(bw, 1), Math.max(bh, 1), hf)); bw >>= 1; bh >>= 1; }
  }

  // render one sub-frame into rtComp
  renderFrame(scene, camera, hazeScene) {
    const r = this.r;
    r.setRenderTarget(this.rtScene);
    r.clear();
    r.render(scene, camera);
    // haze
    r.setRenderTarget(this.rtHaze);
    r.setClearColor(0x000000, 0); r.clear();
    r.render(hazeScene, camera);
    // bloom chain
    let src = this.rtScene.texture, sw = this.w, sh = this.h;
    for (let i = 0; i < this.bloom.length; i++) {
      this.down.u.tSrc.value = src; this.down.u.uTexel.value.set(1 / sw, 1 / sh); this.down.u.uFirst.value = i === 0;
      this.down.draw(r, this.bloom[i]);
      src = this.bloom[i].texture; sw = this.bloom[i].width; sh = this.bloom[i].height;
    }
    for (let i = this.bloom.length - 2; i >= 0; i--) {
      const s = this.bloom[i + 1];
      this.up.u.tSrc.value = s.texture; this.up.u.uTexel.value.set(1 / s.width, 1 / s.height);
      this.up.draw(r, this.bloom[i]);
    }
    const u = this.comp.u;
    u.tScene.value = this.rtScene.texture; u.tHaze.value = this.rtHaze.texture; u.tBloom.value = this.bloom[0].texture;
    u.tDepth.value = this.depthTexture;
    const vp = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    u.uInvViewProj.value.copy(vp).invert();
    u.uPrevViewProj.value.copy(this.prevVP && !this.cut ? this.prevVP : vp);
    u.uCam.value.copy(camera.position);
    u.uBlur.value = this.blur ?? 0;
    this.prevVP = vp; this.cut = false;
    this.comp.draw(r, this.rtComp);
  }

  clearAccum() { this.r.setRenderTarget(this.rtAccum); this.r.setClearColor(0x000000, 1); this.r.clear(); }
  accumulate(weight) { this.accumP.u.tSrc.value = this.rtComp.texture; this.accumP.u.uW.value = weight; this.accumP.draw(this.r, this.rtAccum); }
  toScreen(exposure, time, fade = 1, fromAccum = true) {
    const u = this.present.u;
    u.tSrc.value = fromAccum ? this.rtAccum.texture : this.rtComp.texture;
    u.uExposure.value = exposure; u.uTime.value = time; u.uRes.value.set(this.w, this.h); u.uFade.value = fade;
    this.present.draw(this.r, null);
  }
}
