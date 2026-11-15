// The map: the classic flat explorer. Wheel to zoom, drag to move, dive to 10^32.
//
// What it does for sharpness:
//   * while the view moves, one sample per pixel (at a lower resolution if the GPU is busy);
//   * the moment it rests, further samples are computed, each shifted by a fraction of a pixel,
//     and averaged: up to eight per pixel. Edges and filaments come out clean, not jagged.
//   * the filaments themselves are drawn from the distance estimate, as lines one pixel wide;
//   * the iteration budget follows the place (see Tone.need), so deep spots are not left black.
import { THREE, clamp } from './foehn.js';
import { makeColorize, layersRT, floatRT } from './gpu/colorize.js';
import { bp, bpAlign, bpAddD, bpSubD, bitsFor, fixToDec } from './core/fixed.js';
import { iterCount, Budget } from './core/field.js';

export const HOME_SCALE = 1.35, MIN_SCALE = 1e-32;
export const HOME = { center: bp('-0.6', '0', 96), scale: HOME_SCALE };
const halton = (i, b) => { let f = 1, r = 0; while (i > 0) { f /= b; r += f * (i % b); i = Math.floor(i / b); } return r; };

export class MapView {
  // app: { renderer, canvas, field, fp, quad, pal, tone, toneU, opt: { quality, cap, mobile } }
  constructor(app) {
    this.app = app; this.active = true;
    this.state = { center: HOME.center, scale: HOME.scale };
    this.target = { center: HOME.center, scale: HOME.scale };
    this.flight = null; this.recording = false;
    this.dirty = true; this.recolor = true;
    this.resScale = 1; this.count = 0; this.job = null; this.strips = 1;
    this.w = 2; this.h = 2; this.layers = null; this.motion = null; this.nLayers = 1;
    this.off = [0, 0]; this.gi = 0;
    this.slow = 0; this.fast = 0; this.rendered = false; this.settleN = 0; this.settling = false; this.fresh = true; this.iterMin = 0;
    this.bud = new Budget(app.tone, app.opt); this.at = { center: HOME.center, scale: HOME.scale };
    this.col = makeColorize(app.pal, app.toneU);
    this.onStatus = null;
    this._o = {};
    this.attach();
  }

  // ---------- size ----------
  resize(w, h) {
    if (w === this.w && h === this.h && this.layers) return;
    this.w = w; this.h = h;
    if (this.layers) this.layers.dispose();
    // as many refined samples as fit in about 200 MB of video memory
    const max = this.app.opt.mobile ? 4 : 8;
    this.nLayers = clamp(Math.floor(200e6 / (w * h * 8)), 1, max);
    this.layers = layersRT(w, h, this.nLayers, THREE.RGBAFormat, THREE.HalfFloatType);
    this.sizeMotion();
    this.dirty = true;
  }
  sizeMotion() {
    const w = Math.max(2, Math.round(this.w * this.resScale)), h = Math.max(2, Math.round(this.h * this.resScale));
    if (this.motion && this.motion.width === w && this.motion.height === h) return;
    if (this.motion) this.motion.dispose();
    this.motion = floatRT(w, h);
  }

  // ---------- the reference orbit, statistics, iteration budget ----------
  get iter() { return this.bud.iter; }
  ensureRef() {
    const f = this.app.field, s = this.state, probe = this.bud.probe;
    let off = f.ref ? bpSubD(s.center, f.ref.center) : [0, 0];
    const stale = !f.ref || Math.max(Math.abs(off[0]), Math.abs(off[1])) > 40 * s.scale || f.ref.center.bits < bitsFor(s.scale) + 32 || (!f.ref.escaped && f.len < probe + 2);
    if (stale) {
      // during a dive the reference sits at the destination: one orbit serves the whole way down
      const c = this.flight ? this.flight.c1 : s.center, sc = this.flight ? this.flight.s1 : s.scale;
      f.build(bpAlign(c, bitsFor(sc) + 32), sc, probe);
      off = bpSubD(s.center, f.ref.center);
    }
    this.off = off;
  }
  // a few points of the view on the CPU: they feed the colour levels and the iteration budget
  sample(n) {
    const f = this.app.field, s = this.state, o = this._o, asp = this.w / this.h, probe = this.bud.probe;
    for (let k = 0; k < n; k++) {
      const g = this.gi++ % 96, gx = g % 12, gy = (g / 12) | 0;
      const dx = this.off[0] + ((gx + Math.random()) / 12 * 2 - 1) * asp * s.scale, dy = this.off[1] + ((gy + Math.random()) / 8 * 2 - 1) * s.scale;
      f.eval(dx, dy, probe, o);
      this.bud.note(o.sl);
    }
  }
  budget() {
    const s = this.state, at = this.at;
    const [mx, my] = bpSubD(s.center, at.center);
    if (Math.abs(Math.log(s.scale / at.scale)) > 0.7 || Math.max(Math.abs(mx), Math.abs(my)) > 1.5 * s.scale) { this.at = { center: s.center, scale: s.scale }; this.bud.forget(); }
    return this.bud.settle(s.scale, this.flight ? this.iterMin : 0);
  }

  // ---------- one frame ----------
  update(now, dt) {
    const s = this.state, t = this.target, app = this.app;
    this.adapt(dt);
    if (this.flight) {
      const fl = this.flight, u = Math.min(1, (now - fl.t0) / fl.dur), e = u * u * (3 - 2 * u);
      const sc = fl.s0 * Math.pow(fl.s1 / fl.s0, e), f = fl.s0 === fl.s1 ? 1 - e : (sc - fl.s1) / (fl.s0 - fl.s1);
      s.scale = sc; s.center = bpAddD(bpAlign(fl.c1, bitsFor(sc)), fl.dx * f, fl.dy * f);
      t.scale = s.scale; t.center = s.center; this.dirty = true;
      this.iterMin = Math.round(fl.i0 + (fl.i1 - fl.i0) * e);
      if (u >= 1) { const cb = fl.onDone; this.flight = null; if (cb) cb(); }
    } else {
      const k = 1 - Math.exp(-dt * 10), lz0 = Math.log(s.scale), lz1 = Math.log(t.scale);
      const [dx, dy] = bpSubD(t.center, s.center);
      if (Math.abs(lz1 - lz0) > 1e-6 || Math.abs(dx) > s.scale * 1e-6 || Math.abs(dy) > s.scale * 1e-6) {
        s.scale = Math.exp(lz0 + (lz1 - lz0) * k);
        s.center = bpAddD(bpAlign(s.center, bitsFor(s.scale)), dx * k, dy * k);
        this.dirty = true;
      } else if (this.dirty) { s.scale = t.scale; s.center = t.center; }
    }
    if (app.cycle) this.recolor = true;

    this.ensureRef();
    // statistics: a burst when the view changes, a trickle while it rests
    if (this.dirty || this.settling) { const b0 = this.bud.boost; this.sample(this.dirty ? 14 : 6); if (this.budget()) this.dirty = true; if (this.bud.boost !== b0) this.settleN = 60; this.ensureRef(); }
    this.settling = app.tone.settle(this.fresh ? 0 : dt) || (this.settleN = this.dirty ? Math.max(40, this.settleN) : this.settleN - 1) > 0;
    this.fresh = false;
    if (this.settling) this.recolor = true;

    const fp = app.fp, base = { scale: s.scale, offX: this.off[0], offY: this.off[1], halfX: this.w / this.h, halfY: 1, iter: this.iter, pack: true };
    this.rendered = false;
    if (this.dirty) {
      this.job = null; this.rendered = true;
      if (this.resScale >= 1) { fp.render(this.layers, 0, base); this.count = 1; }
      else { this.sizeMotion(); fp.render(this.motion, 0, base); this.count = 0; }
      this.dirty = false; this.recolor = true; this.idle = 0;
      if (this.onStatus) this.onStatus();
    } else if (this.count < this.nLayers && !this.recording) {
      // refine: the next sample, a strip of it per frame when a whole one would stall the page
      if (!this.job) {
        const k = this.count;
        this.job = { k, y: 0, jx: k ? halton(k, 2) - 0.5 : 0, jy: k ? halton(k, 3) - 0.5 : 0 };
        if (k === 0) this.strips = Math.max(this.strips, Math.ceil(1 / (this.resScale * this.resScale)));
      }
      const j = this.job, hStrip = Math.ceil(this.h / this.strips), y1 = Math.min(this.h, j.y + hStrip);
      fp.render(this.layers, j.k, { ...base, jitX: j.jx, jitY: j.jy, strip: [j.y, y1] });
      j.y = y1;
      if (j.y >= this.h) { this.count = j.k + 1; this.job = null; this.recolor = true; }
      // keep the page responsive: thinner strips when frames are slow, thicker when there is time
      if (dt > 0.05 && this.strips < 64) this.strips *= 2; else if (dt < 0.022 && this.strips > 1) this.strips = Math.max(1, this.strips >> 1);
    }
    if (this.recolor) { this.paint(); this.recolor = false; }
  }
  paint(target = null) {
    const u = this.col.u, app = this.app;
    u.uCount.value = this.count; u.uData.value = this.layers.texture; u.uMotion.value = this.motion.texture;
    u.uRect.value.set(0, 0, this.w, this.h);
    u.uDataRes.value.set(this.count ? this.w : this.motion.width, this.count ? this.h : this.motion.height);
    u.uPx.value = this.h / 2; u.uRelief.value = app.relief; u.uEdge.value = app.edge;
    app.syncTone();
    app.quad.draw(this.col.mat, target);
  }
  // resolution of the moving picture follows the frame rate
  adapt(dt) {
    if (!this.rendered || this.recording) return;
    if (dt > 0.042) { this.slow++; this.fast = 0; } else if (dt < 0.02) { this.fast++; this.slow = 0; }
    if (this.slow > 4 && this.resScale > 0.25) { this.resScale = Math.max(0.25, this.resScale * 0.8); this.slow = 0; }
    else if (this.fast > 40 && this.resScale < 1) { this.resScale = Math.min(1, this.resScale * 1.25); this.fast = 0; }
  }

  // ---------- moving ----------
  jump(centerBP, scale) {
    scale = clamp(scale, MIN_SCALE, HOME_SCALE * 2.2);
    const c = bpAlign(centerBP, bitsFor(scale));
    this.state.center = this.target.center = c; this.state.scale = this.target.scale = scale;
    this.flight = null; this.dirty = true; this.fresh = true; this.app.tone.reset();
  }
  flyTo(centerBP, scale, dur, onDone) {
    const o = this.app.opt, f = this.app.field, out = this._o;
    const c1 = bpAlign(centerBP, bitsFor(scale));
    // The orbit of the destination, long enough for what will be found there: a few points around
    // it are tried first, so that the dive never has to stop and compute a longer one.
    let capIt = Math.min(o.cap, Math.round(2 * iterCount(scale) * o.quality + 2000)), need = 0;
    for (let pass = 0; pass < 4; pass++) {
      f.build(bpAlign(c1, bitsFor(scale) + 32), scale, capIt);
      const v = [], probe = f.ref.escaped ? capIt : f.len - 2;
      for (let k = 0; k < 48; k++) { const r = scale * (0.15 + 0.85 * k / 48); f.eval(Math.cos(k * 2.4) * r * 1.5, Math.sin(k * 2.4) * r, probe, out); if (out.sl >= 0) v.push(out.sl); }
      v.sort((a, b) => a - b);
      const hi = v.length >= 10 ? v[Math.floor(v.length * 0.9)] : capIt;
      need = Math.min(o.cap, Math.round(Math.max(iterCount(scale), 1.6 * hi + 250) * o.quality * 1.15));
      if (f.ref.escaped || f.len >= need + 2 || capIt >= o.cap) break;
      capIt = Math.min(o.cap, 2 * need + 1000);
    }
    const [dx, dy] = bpSubD(this.state.center, c1);
    this.flight = { t0: performance.now(), dur, s0: this.state.scale, s1: scale, c1, dx, dy, i0: this.iter, i1: need, onDone: onDone || null };
  }
  zoomAt(px, py, factor) {
    const t = this.target, ns = clamp(t.scale * factor, MIN_SCALE, HOME_SCALE * 2.2);
    factor = ns / t.scale;
    const r = this.app.canvas.getBoundingClientRect();
    const ox = ((px - r.left) / r.height * 2 - r.width / r.height) * t.scale, oy = -((py - r.top) / r.height * 2 - 1) * t.scale;
    t.center = bpAddD(bpAlign(t.center, bitsFor(ns)), ox * (1 - factor), oy * (1 - factor));
    t.scale = ns;
  }
  panPx(dpx, dpy) {
    const t = this.target, r = this.app.canvas.getBoundingClientRect();
    t.center = bpAddD(bpAlign(t.center, bitsFor(t.scale)), -dpx / r.height * 2 * t.scale, dpy / r.height * 2 * t.scale);
  }
  // the point of the plane under a place on the screen, as an offset from the centre (doubles)
  offsetAt(px, py) {
    const r = this.app.canvas.getBoundingClientRect(), s = this.state.scale;
    return [((px - r.left) / r.height * 2 - r.width / r.height) * s, -((py - r.top) / r.height * 2 - 1) * s];
  }
  coords() {
    const s = this.state, dg = Math.min(36, Math.max(12, Math.ceil(-Math.log10(s.scale)) + 4));
    return { x: fixToDec(s.center.x, s.center.bits, dg), y: fixToDec(s.center.y, s.center.bits, dg), zoom: HOME_SCALE / s.scale };
  }

  attach() {
    const cv = this.app.canvas, P = new Map();
    let pinch = 0, mid = null, tap = { t: 0, x: 0, y: 0 };
    const busy = () => !this.active || this.flight;
    cv.addEventListener('pointerdown', (e) => {
      if (!this.active) return;
      cv.setPointerCapture(e.pointerId);
      P.set(e.pointerId, { x: e.clientX, y: e.clientY, sx: e.clientX, sy: e.clientY });
      if (P.size === 2) { const [a, b] = [...P.values()]; pinch = Math.hypot(a.x - b.x, a.y - b.y); mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }
      cv.classList.add('dragging');
    });
    cv.addEventListener('pointermove', (e) => {
      if (!P.has(e.pointerId)) return;
      const prev = P.get(e.pointerId), cur = { x: e.clientX, y: e.clientY, sx: prev.sx, sy: prev.sy };
      P.set(e.pointerId, cur);
      if (busy()) return;
      if (P.size === 1) { this.panPx(cur.x - prev.x, cur.y - prev.y); this.state.center = this.target.center; this.dirty = true; }       // the picture sticks to the finger
      else if (P.size === 2) {
        const [a, b] = [...P.values()], d = Math.hypot(a.x - b.x, a.y - b.y), m = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
        if (mid) this.panPx(m.x - mid.x, m.y - mid.y);
        if (pinch > 0 && d > 0) this.zoomAt(m.x, m.y, pinch / d);
        pinch = d; mid = m;
      }
    });
    const end = (e) => {
      const p = P.get(e.pointerId); P.delete(e.pointerId);
      if (P.size < 2) { pinch = 0; mid = null; }
      if (P.size === 0) cv.classList.remove('dragging');
      if (p && e.type === 'pointerup' && e.pointerType === 'touch' && !busy()) {
        const now = performance.now();
        if (Math.hypot(e.clientX - p.sx, e.clientY - p.sy) < 14) {
          if (now - tap.t < 320 && Math.hypot(e.clientX - tap.x, e.clientY - tap.y) < 40) { this.zoomAt(e.clientX, e.clientY, 0.25); tap.t = 0; } else tap = { t: now, x: e.clientX, y: e.clientY };
        }
      }
    };
    cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
    cv.addEventListener('wheel', (e) => { if (!this.active) return; e.preventDefault(); if (!this.flight) this.zoomAt(e.clientX, e.clientY, Math.pow(1.0016, e.deltaY)); }, { passive: false });
    cv.addEventListener('dblclick', (e) => { if (!busy()) this.zoomAt(e.clientX, e.clientY, 0.2); });
    addEventListener('keydown', (e) => {
      if (busy() || e.target.tagName === 'INPUT') return;
      const t = this.target, pan = this.state.scale * 0.12;
      if (e.key === '+' || e.key === '=') t.scale = Math.max(MIN_SCALE, t.scale * 0.7);
      else if (e.key === '-') t.scale = Math.min(HOME_SCALE * 2.2, t.scale / 0.7);
      else if (e.key === 'ArrowLeft') t.center = bpAddD(t.center, -pan, 0);
      else if (e.key === 'ArrowRight') t.center = bpAddD(t.center, pan, 0);
      else if (e.key === 'ArrowUp') t.center = bpAddD(t.center, 0, pan);
      else if (e.key === 'ArrowDown') t.center = bpAddD(t.center, 0, -pan);
    });
  }
}
