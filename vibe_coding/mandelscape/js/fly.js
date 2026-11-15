// The flight: the F-5 of FÖHN over the Mandelbrot land. See core/world.js for the idea
// (the lake, the land that rises with the distance from it, the logarithmic altimeter).
import { THREE, Sim, FoehnTerrain, G, V_MAX, V_MIN, len, yprOf, qrot, clamp, lerp } from './foehn.js';
import { buildBike, FX, Post, FoehnU, Input, EngineAudio } from './foehn-render.js';
import { World, K_LO, K_HI, S_MAX, S_MIN, Y_TOP, VIEW_M } from './core/world.js';
import { Clip, Terrain } from './gpu/terrain.js';
import { makeColorize, floatRT } from './gpu/colorize.js';
import { HOME, HOME_SCALE } from './map.js';
import { bpAddD, bpSubD, fixToDec } from './core/fixed.js';

const CAMS = ['chase', 'wide', 'close', 'cockpit'];

export class FlyView {
  constructor(app) {
    const r = app.renderer, $ = app.$;
    this.app = app; this.$ = $;
    const w = this.world = new World(app.field, app.tone, app.opt);
    w.slope = app.slope;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.3, 90000);
    this.post = new Post(r); this.post.msaa = 4;
    this.clip = new Clip(r, app.fp, w, app.opt.floatLinear);
    this.terrain = new Terrain(this.scene, this.clip, w, app.pal, app.toneU, app.opt.floatLinear);
    const s = this.sunDir = this.terrain.U.uSun.value;
    // light for the bike itself (the land has its own shader): the same sun, and the sky as surroundings
    const sun = new THREE.DirectionalLight(0xfff0dc, Math.PI * 1.15); sun.position.set(s.x * 100, s.y * 100, s.z * 100); this.scene.add(sun);
    this.scene.add(new THREE.HemisphereLight(0x9db4e0, 0x2c3038, Math.PI * 0.55));
    const pm = new THREE.PMREMGenerator(r), es = new THREE.Scene(); es.add(this.terrain.sky.clone());
    this.scene.environment = pm.fromScene(es, 0, 1, 80000).texture; this.scene.environmentIntensity = 0.7;
    this.model = buildBike(); this.scene.add(this.model.root);
    this.fx = new FX(this.scene, this.model);
    this.sim = new Sim(); this.bike = this.sim.bike;
    this.input = new Input(); this.audio = new EngineAudio();
    this.ext = { groundAt: (x, z) => w.groundAt(x, z), normalAt: (x, z) => w.normalAt(x, z), waterAt: () => 0 };
    this.state = 'fly'; this.stateT = 0; this.vt = 30; this.time = 0; this.lean = 0; this.paused = false;
    this.cam = 'chase'; this.camPos = new THREE.Vector3(); this.camLook = new THREE.Vector3(); this.camInit = false; this.fov = 60;
    this.dyn = 1; this.fpsAcc = 0; this.fpsN = 0; this.frameN = 0; this.viewW = 1; this.viewH = 1;
    this.last = null; this.msg = ''; this.msgT = 0; this.coordT = 0; this.wOff = [0, 0]; this.active = false;
    // mini-map: the flat map of the neighbourhood, from the same field shader
    this.mini = { rt: floatRT(192, 192), toneU: { value: new THREE.Vector4(2, 6, 0.55, 0) }, half: 600 }; this.mini.col = makeColorize(app.pal, this.mini.toneU);
    this.touchUI = app.Q.has('touch') || matchMedia('(pointer: coarse)').matches;
    if (this.touchUI) { this.input.enableTouch($('touch')); $('touch').classList.remove('on'); }
    const wake = () => { if (this.active) this.audio.start(); };
    addEventListener('pointerdown', wake); addEventListener('keydown', wake);
    this._q = new THREE.Quaternion(); this._v = new THREE.Vector3();
  }

  // ---------- entering, leaving, placing the bike ----------
  enter(center, scale) {
    this.active = true; FoehnTerrain.ext = this.ext;
    if (this.touchUI) this.$('touch').classList.add('on');
    const L = this.last;
    if (L && Math.abs(scale / L.scale - 1) < 1e-6 && bpSubD(center, L.center).every((d) => Math.abs(d) < 1e-6 * scale)) this.put(L.center, L.S, L.y, L.yaw);
    else this.spawn(center, scale);
    this.keysT = 14; this.$('keys').classList.remove('off');
  }
  leave() {
    this.active = false; FoehnTerrain.ext = null;
    this.$('touch').classList.remove('on');
    if (this.audio.ctx) this.audio.engine(this.bike, false);
  }
  // where the bike is, as a map would show it
  place() {
    const w = this.world, y = this.bike.p[1], scale = Math.min(HOME_SCALE, w.S * Math.max(VIEW_M, y / 0.9));
    this.last = { center: w.origin(), scale, S: w.S, y, yaw: yprOf(this.bike.q)[0] };
    return { center: this.last.center, scale };
  }
  // Arrive at a place chosen on the map. The bike is put on the high side of it, nose towards the
  // lake, so that the place itself lies ahead and below, where one can see it.
  spawn(center, scale) {
    const w = this.world, o = {};
    let S = scale / VIEW_M, y = 0.5 * (K_LO + K_HI);
    let far = 0.9;
    if (S > S_MAX) { y = clamp(scale / S_MAX * 0.75, K_HI, Y_TOP * 0.85); S = S_MAX; far = 1.5; }      // a wide map: stand well back and high, to see all of it
    w.place(center, S);
    w.f.eval(0, 0, w.bud.probe, o);
    let ux = 1, uy = 0;                                    // over the lake itself: come from the east
    if (o.sl >= 0) { ux = o.ux; uy = o.uy; }
    const back = far * Math.max(scale, VIEW_M * w.S);
    this.put(bpAddD(w.origin(), ux * back, uy * back), w.S, y, Math.atan2(-uy, -ux));
  }
  put(center, S, y, yaw) {
    const w = this.world, o = {};
    w.place(center, S);
    // the colours and the iteration budget of the new place, before anything is drawn
    for (let k = 0; k < 6; k++) { w.sample(40); w.ensureRef(); }
    this.app.tone.settle(0);
    // never inside a mountain: fly above it, and if the mountain is taller than the band allows, be larger
    w.f.eval(0, 0, w.bud.probe, o);
    let g = o.sl < 0 ? 0 : w.hOf(o.d, o.sl);
    if (g + 45 > y) {
      if (g > K_HI - 50 && w.S < S_MAX) { const f = Math.min(S_MAX / w.S, g / (K_HI - 50)); w.place(center, w.S * f); g /= f; for (let k = 0; k < 3; k++) { w.sample(40); w.ensureRef(); } }
      y = g + 45;
    }
    this.sim.reset([0, y, 0], yaw, 30); this.bike.crash = null;
    this.vt = 30; this.state = 'fly'; this.camInit = false; this.model.root.visible = true;
    this.clip.work(0, Math.cos(yaw), Math.sin(yaw), true);
    for (const L of w.levels) L.fade = 1;
    w.sampleLevels(60, 0);
    this.mini.due = 0;
  }
  jump(center, scale) {
    const fade = this.$('fade'); fade.classList.add('on');
    setTimeout(() => { this.spawn(center, scale); setTimeout(() => fade.classList.remove('on'), 80); }, 290);
  }
  reset() { this.jump(HOME.center, HOME_SCALE); }
  respawn() {
    const w = this.world, yaw = yprOf(this.bike.q)[0], g = Math.max(0, w.groundAt(0, 0));
    this.sim.reset([0, Math.max(g + 45, K_LO + 30), 0], yaw, 30); this.bike.crash = null;
    this.vt = 30; this.state = 'fly'; this.camInit = false; this.model.root.visible = true;
  }
  setSlope(v) { this.world.slope = v; if (this.active && this.state === 'fly' && this.bike.p[1] < this.world.groundAt(0, 0) + 3) this.respawn(); }
  setFine(on) { const w = this.world; if ((w.N === 1024) === !!on) return; w.setRes(on ? 1024 : 512, w.NL); this.clip.alloc(); if (this.active) { const yaw = yprOf(this.bike.q)[0]; this.clip.work(0, Math.cos(yaw), Math.sin(yaw), true); for (const L of w.levels) L.fade = 1; } }
  say(text, secs = 2.5) { this.msg = text; this.msgT = secs; }

  resize(viewW, viewH) {
    this.viewW = viewW; this.viewH = viewH;
    const pr = Math.min(devicePixelRatio || 1, 1.5) * this.dyn * (+this.app.Q.get('scale') || 1);
    const w = Math.max(2, Math.round(viewW * pr)), h = Math.max(2, Math.round(viewH * pr));
    this.app.renderer.setSize(w, h, false); this.post.setSize(w, h);
    this.camera.aspect = viewW / viewH; this.camera.updateProjectionMatrix();
  }

  // ---------- one frame ----------
  update(now, dtRaw) {
    const dt = Math.min(0.05, dtRaw), w = this.world, b = this.bike, $ = this.$, app = this.app;
    for (const k of this.input.pressed) {
      if (k === 'KeyC') { this.cam = CAMS[(CAMS.indexOf(this.cam) + 1) % CAMS.length]; this.camInit = false; }
      else if (k === 'KeyR') this.respawn();
      else if (k === 'KeyH') $('helpOverlay').classList.toggle('show');
      else if (k === 'KeyP') this.paused = !this.paused;
    }
    this.input.pressed.length = 0;
    const held = this.paused || $('helpOverlay').classList.contains('show');
    if (this.audio.ctx) { if (held && this.audio.ctx.state === 'running') this.audio.ctx.suspend(); else if (!held && this.audio.ctx.state === 'suspended') this.audio.ctx.resume(); }
    if (!held) {
      const [yaw] = yprOf(b.q), fx = Math.cos(yaw), fy = Math.sin(yaw);
      if (this.state === 'fly') {
        const c = this.input.read(), nose = qrot(b.q, [1, 0, 0]), fwd = b.v[0] * nose[0] + b.v[1] * nose[1] + b.v[2] * nose[2];
        // throttle lever -> wanted speed (as in FÖHN)
        this.vt = clamp(this.vt + (c.throttle || 0) * 20 * dt, V_MIN, V_MAX);
        if (c.brake) this.vt = Math.max(0, Math.min(this.vt, fwd) - 30 * dt);
        if (c.boost && !b.abLock) this.vt = clamp(Math.max(this.vt, fwd + 3), V_MIN, V_MAX + 30); else if (this.vt > V_MAX) this.vt = Math.max(V_MAX, Math.min(this.vt, fwd));
        if (b.contactN > 0.25 * b.mp.m * G && len(b.v) < 4 && !c.boost && (c.pitch || 0) <= 0.05 && !(c.throttle > 0)) this.vt = 0;
        this.sim.advance(dt, { turn: c.turn || 0, pitch: c.pitch || 0, strafe: c.strafe || 0, vt: this.vt, boost: !!c.boost, brake: !!c.brake });
        this.lean += ((c.turn || 0) - this.lean) * Math.min(1, dt * 5);
        if (b.crash) {
          this.fx.explode(b.p, b.v, b.crash === 'water'); this.audio.crash(b.crash === 'water');
          this.state = 'crashed'; this.stateT = 2.0; this.model.root.visible = false;
          this.say(b.crash === 'water' ? (w.S <= S_MIN * 1.01 ? 'הגעת לתחתית. מכאן נגמרים המספרים' : 'נחיתה במים') : 'התרסקות', 2);
        } else {
          if (w.S >= S_MAX && b.p[1] > Y_TOP) b.v[1] -= (b.p[1] - Y_TOP) * 0.5 * dt + Math.max(0, b.v[1]) * 2 * dt;      // the ceiling, at full size
          this.carry(w.follow(b.p, b.contactN > 0, dt));
        }
      } else { this.stateT -= dt; if (this.stateT <= 0) this.respawn(); }
      // reference orbit, colour levels, iteration budget, and a strip of whichever height map is due
      w.ensureRef(); w.sample(5); w.ensureRef();
      app.tone.settle(dt, 0.8); w.sampleLevels(2, dt);
      this.clip.work(dtRaw, fx, fy);
      this.time += dt; FoehnU.uTime.value = this.time;
      const alive = this.state === 'fly';
      this.model.update(b, this.time, this.lean, dt);
      this.fx.update(dt, b, [0, 0, 0], alive);
      this.updateCamera(dt);
      this.audio.engine(b, alive, this.camera.position.distanceTo(this.model.root.position), this.cam === 'cockpit');
    }
    this.draw(dt);
    this.hud(dt);
    // resolution follows the frame rate
    this.fpsAcc += dtRaw; this.fpsN++;
    if (this.fpsAcc > 1.5) {
      const fps = this.fpsN / this.fpsAcc; this.fps = fps; this.fpsAcc = 0; this.fpsN = 0;
      if (!app.Q.has('fixed')) { if (fps < 42 && this.dyn > 0.55) { this.dyn = Math.max(0.55, this.dyn - 0.1); this.resize(this.viewW, this.viewH); } else if (fps > 57 && this.dyn < 1) { this.dyn = Math.min(1, this.dyn + 0.05); this.resize(this.viewW, this.viewH); } }
    }
  }
  // the origin moved under the bike and the world may have changed size: take along what else lives in local metres
  carry({ sx, sz, f }) {
    if (sx === 0 && sz === 0 && f === 1) return;
    const k = 1 / f;
    for (const P of [this.fx.soft, this.fx.fire]) {
      const p = P.pos;
      for (let i = 0, n = P.n; i < n; i++) { if (P.life[i] <= 0) continue; const j = i * 3; p[j] = (p[j] - sx) * k; p[j + 1] *= k; p[j + 2] = (p[j + 2] - sz) * k; P.size[i] *= k; }
    }
    for (const d of this.fx.debris) { const p = d.m.position; p.set((p.x - sx) * k, p.y * k, (p.z - sz) * k); }
    for (const v of [this.camPos, this.camLook]) v.set((v.x - sx) * k, v.y * k, (v.z - sz) * k);
    this.wOff[0] = (this.wOff[0] + sx) % 20000; this.wOff[1] = (this.wOff[1] + sz) % 20000;
  }

  updateCamera(dt) {
    const b = this.bike, cam = this.camera, V = len(b.v), [yaw, , roll] = yprOf(b.q), tv = this._v;
    const k = clamp((V - 6) / 20, 0, 1), fh = [Math.cos(yaw), 0, -Math.sin(yaw)];
    let dir = [lerp(fh[0], b.v[0] / (V || 1), k), lerp(0, b.v[1] / (V || 1), k) * 0.6, lerp(fh[2], b.v[2] / (V || 1), k)];
    const dl = Math.hypot(dir[0], dir[1], dir[2]) || 1; dir = dir.map((v) => v / dl);
    const shake = this.fx.shake * 0.5 + b.abOut * 0.04 + Math.max(0, V - 80) * 0.0006;
    let targetFov = 56 + Math.min(15, V * 0.15) + b.abOut * 7;
    if (this.cam === 'cockpit') {
      const p = b.toWorld([0.16, 1.5, 0]);
      cam.position.set(p[0], p[1], p[2]);
      this._q.set(b.q[1], b.q[2], b.q[3], b.q[0]);
      cam.quaternion.copy(this._q).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, -Math.PI / 2, 0))).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(-0.10, -this.lean * 0.25, 0)));
      targetFov += 12;
    } else {
      const dist = this.cam === 'close' ? 3.1 : this.cam === 'wide' ? 12 + V * 0.02 : 4.8 + V * 0.008, h = this.cam === 'close' ? 0.95 : this.cam === 'wide' ? 4.2 : 1.5;
      tv.set(b.p[0] - dir[0] * dist, b.p[1] - dir[1] * dist + h, b.p[2] - dir[2] * dist);
      if (!this.camInit) { this.camPos.copy(tv); this.camInit = true; }
      this.camPos.lerp(tv, 1 - Math.exp(-dt * (5 + V * 0.06)));
      const dx = this.camPos.x - b.p[0], dy = this.camPos.y - b.p[1], dz = this.camPos.z - b.p[2], dd = Math.hypot(dx, dy, dz), mx = dist + 2.5;
      if (dd > mx) this.camPos.set(b.p[0] + dx / dd * mx, b.p[1] + dy / dd * mx, b.p[2] + dz / dd * mx);
      this.camPos.y = Math.max(this.camPos.y, Math.max(0, this.world.groundAt(this.camPos.x, this.camPos.z)) + 0.9);
      cam.position.copy(this.camPos);
      // the higher, the more the camera looks down: from far up, the land is below, not ahead
      const y = b.p[1], sm = (a, c, x) => { const t = clamp((x - a) / (c - a), 0, 1); return t * t * (3 - 2 * t); };
      const down = Math.tan(0.13 + 0.34 * sm(K_HI, 700, y) + 0.22 * sm(700, Y_TOP, y));
      cam.position.y += dist * down * 0.55;
      this.camLook.set(b.p[0] + dir[0] * 9, b.p[1] + dir[1] * 9 + 1.0 - 9 * down * 0.6, b.p[2] + dir[2] * 9);
      cam.up.set(Math.sin(roll * 0.28) * fh[2] * -1, 1, Math.sin(roll * 0.28) * fh[0]).normalize();
      cam.lookAt(this.camLook);
    }
    if (shake > 0.001) { cam.position.x += (Math.random() - 0.5) * shake; cam.position.y += (Math.random() - 0.5) * shake; cam.position.z += (Math.random() - 0.5) * shake; }
    this.fov += (targetFov - this.fov) * Math.min(1, dt * 3);
    cam.fov = this.fov; cam.updateProjectionMatrix(); cam.updateMatrixWorld();
  }

  // ---------- drawing ----------
  draw(dt = 0) {
    const r = this.app.renderer, w = this.world, b = this.bike, T = this.terrain, U = T.U, s = this.sunDir, app = this.app;
    T.update(dt, this.time);
    app.syncTone();
    // the bike's shadow, where the sun would put it
    const g = Math.max(0, w.groundAt(0, 0)), agl = Math.max(0, b.p[1] - b.mp.cg[1] - g), back = Math.min(agl / s.y, 400);
    U.uShadow.value.set(-s.x * back, -s.z * back, 1.7 + 0.1 * agl, this.state === 'fly' ? 0.8 * clamp(1 - agl / 110, 0, 1) : 0);
    const air = 14000 + 12 * b.p[1];
    U.uFog.value.x = 1 / (T.skirt.visible ? air : Math.min(air, 0.4 * T.reach));
    U.uWater.value.set(this.wOff[0], this.wOff[1]);
    T.sky.position.set(0, 0, 0);
    r.setRenderTarget(this.post.main); r.setClearColor(0x000000, 1); r.clear();
    r.render(this.scene, this.camera);
    const film = this.cam === 'cockpit' ? 0.35 : 0.5;
    this.post.finish(this.camera, [s.x, s.y, s.z], { blur: film, speed: clamp((b.airspeed - 35) / 80, 0, 1) + b.abOut * 0.3, time: this.time, exposure: 1.0, flash: Math.min(0.6, Math.max(0, this.fx.flashT - 0.42) * 6), rays: 0.6 });
    this.drawMini();
  }
  drawMini() {
    const app = this.app, r = app.renderer, w = this.world, m = this.mini, el = this.$('mini').getBoundingClientRect(), cv = this.app.canvas;
    if (el.width < 8) return;
    const k = cv.width / this.viewW, x = Math.round(el.left * k), wd = Math.round(el.width * k), ht = Math.round(el.height * k), y = cv.height - Math.round(el.top * k) - ht;
    if (!(m.due-- > 0)) {
      m.due = 5; m.half = Math.max(500, 2.5 * this.bike.p[1]);
      const sc = m.half * w.S;
      this.app.fp.render(m.rt, 0, { scale: sc, offX: w.ox, offY: w.oy, halfX: 1, halfY: 1, iter: w.bud.iter, pack: true });
      this.$('miniScale').textContent = fmtLen(2 * m.half);
    }
    const u = m.col.u, g = app.toneU.value;
    // coloured like the level of the land that is about its size
    let L = w.levels[0]; for (const X of w.levels) if (X.h <= m.half * w.S * 1.01) { L = X; break; }
    if (L) m.toneU.value.set(L.tone.lo, L.tone.w, g.z, g.w);
    u.uCount.value = 0; u.uMotion.value = m.rt.texture; u.uDataRes.value.set(m.rt.width, m.rt.height); u.uRect.value.set(x, y, wd, ht);
    u.uPx.value = ht / 2; u.uRelief.value = 0.3; u.uEdge.value = 0.9;
    r.setRenderTarget(null); r.setViewport(x, y, wd, ht); r.setScissor(x, y, wd, ht); r.setScissorTest(true);
    this.app.quad.draw(m.col.mat, null);
    r.setScissorTest(false); r.setViewport(0, 0, cv.width, cv.height);
  }

  hud(dt) {
    const $ = this.$, w = this.world, b = this.bike, y = b.p[1];
    if (this.keysT > 0 && (this.keysT -= dt) <= 0) $('keys').classList.add('off');
    this.coordT -= dt;
    if (this.coordT <= 0) {
      this.coordT = 0.2;
      const zoom = HOME_SCALE / (w.S * VIEW_M);
      $('hZoomV').textContent = this.app.fmtZoom(zoom);
      $('hSpeedV').textContent = Math.round(len(b.v) * 3.6);
      const c = w.origin(), dg = Math.min(36, Math.max(10, Math.ceil(-Math.log10(w.S * VIEW_M)) + 4));
      this.app.setCoords(`x = ${fixToDec(c.x, c.bits, dg)}   y = ${fixToDec(c.y, c.bits, dg)}   ${w.bud.iter} איטרציות`);
      const [yaw] = yprOf(b.q);
      $('miniArrow').style.transform = `rotate(${Math.atan2(Math.cos(yaw), Math.sin(yaw)) * 180 / Math.PI}deg)`;
    }
    // the altimeter: the band between the two lines is ordinary flight; below it the bike shrinks, above it it grows
    const H = $('hGauge').clientHeight || 230, top = 1.5 * K_HI, px = (v) => H * (1 - clamp(v / top, 0, 1));
    if (this.gaugeH !== H) { this.gaugeH = H; $('gBand').style.top = px(K_HI) + 'px'; $('gBand').style.height = (px(K_LO) - px(K_HI)) + 'px'; $('gHi').style.bottom = (H - px(K_HI) + 4) + 'px'; $('gLo').style.bottom = (H - px(K_LO) - 16) + 'px'; }
    $('gMe').style.top = px(y) + 'px'; $('gVal').style.bottom = (H - px(y)) + 'px'; $('gVal').textContent = Math.round(y) + ' מ\'';
    $('hHeat').firstElementChild.style.width = Math.round(b.heat * 100) + '%';
    let msg = '';
    if (this.msgT > 0) { this.msgT -= dt; msg = this.msg; }
    else if (this.state === 'fly') {
      if (w.zooming < 0) msg = 'צוללים פנימה';
      else if (w.zooming > 0) msg = 'מתרחקים';
      else if (w.S <= S_MIN * 1.01 && y < K_LO) msg = 'זו התחתית: עומק של 10 בחזקת 31';
      else if (b.contactN > 0 && len(b.v) < 1) msg = 'נחתת. חץ למטה ו־W להמראה';
    }
    if (msg !== this.shown) { this.shown = msg; $('hMsg').textContent = msg; }
  }
}
// Debug (window.dbg.fly.gpuCheck()): the height map of a level, read back from the GPU, against
// the same points computed on the CPU in doubles. Returns the worst differences found.
FlyView.prototype.gpuCheck = function (li = 0, n = 400) {
  const w = this.world, L = w.levels[li], N = w.N, r = this.app.renderer, o = {};
  const rt = new THREE.WebGLRenderTarget(N, N, { type: THREE.FloatType, format: THREE.RGBAFormat, depthBuffer: false });
  const mat = new THREE.RawShaderMaterial({ glslVersion: THREE.GLSL3, depthTest: false, uniforms: { uLev: { value: this.clip.rt.texture }, uLayer: { value: L.layer } },
    vertexShader: 'in vec3 position; void main(){ gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: 'precision highp float; precision highp sampler2DArray; uniform sampler2DArray uLev; uniform int uLayer; out vec4 o; void main(){ o = vec4(texelFetch(uLev, ivec3(ivec2(gl_FragCoord.xy), uLayer), 0).xy, 0.0, 1.0); }' });
  this.app.quad.draw(mat, rt);
  const px = new Float32Array(N * N * 4); r.readRenderTargetPixels(rt, 0, 0, N, N, px); r.setRenderTarget(null); rt.dispose(); mat.dispose();
  let wSl = 0, wD = 0, flips = 0, used = 0, inside = 0;
  for (let k = 0; k < n; k++) {
    const i = Math.floor(Math.random() * N), j = Math.floor(Math.random() * N), g = (j * N + i) * 4;
    w.f.eval(L.cx - L.h + (i + 0.5) * L.tau, L.cy - L.h + (j + 0.5) * L.tau, L.iter, o);
    const gIn = px[g + 1] === 0, cIn = o.sl < 0;
    if (gIn !== cIn) { flips++; continue; }
    if (cIn) { inside++; continue; }
    if (o.d < 0.05 * L.tau) continue;                                    // a twentieth of a texel from the set: single precision cannot tell
    used++; wSl = Math.max(wSl, Math.abs(px[g] - o.sl)); wD = Math.max(wD, Math.abs(px[g + 1] * L.h / o.d - 1));
  }
  return { level: L.l, iter: L.iter, used, inside, flips, worstCount: +wSl.toFixed(4), worstDistRel: +wD.toExponential(2) };
};
const fmtLen = (m) => (m >= 1000 ? (m / 1000).toFixed(1) + ' ק"מ' : Math.round(m) + ' מ\'');
