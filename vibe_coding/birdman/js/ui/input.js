// Keyboard, touch and gamepad → one intention: { pitch, roll, fwd, flap, tuck }.
import { clamp } from '../sim/vmath.js';

export class Input {
  constructor() {
    this.keys = new Set(); this.pitch = 0; this.roll = 0; this.invert = false; this.touch = { x: 0, y: 0, flap: false, tuck: false, on: false };
    this.handlers = {};
    try { this.invert = localStorage.getItem('birdman.invert') === '1'; } catch {}
    addEventListener('keydown', (e) => {
      if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault();
      if (!e.repeat && this.handlers[e.code]) this.handlers[e.code](e);
      if (!e.repeat && this.handlers.any) this.handlers.any(e);
      this.keys.add(e.code);
    });
    addEventListener('keyup', (e) => this.keys.delete(e.code));
    addEventListener('blur', () => this.keys.clear());
  }
  on(code, fn) { this.handlers[code] = fn; }
  setInvert(v) { this.invert = v; try { localStorage.setItem('birdman.invert', v ? '1' : '0'); } catch {} }
  bindTouch(stick, knob, flapBtn, tuckBtn) {
    let id = null, cx = 0, cy = 0;
    const R = 52;
    stick.addEventListener('pointerdown', (e) => { id = e.pointerId; const r = stick.getBoundingClientRect(); cx = r.left + r.width / 2; cy = r.top + r.height / 2; stick.setPointerCapture(id); move(e); this.touch.on = true; });
    const move = (e) => { if (e.pointerId !== id) return; const dx = clamp((e.clientX - cx) / R, -1, 1), dy = clamp((e.clientY - cy) / R, -1, 1); this.touch.x = dx; this.touch.y = dy; knob.style.transform = `translate(${dx * R}px, ${dy * R}px)`; };
    stick.addEventListener('pointermove', move);
    const end = (e) => { if (e.pointerId !== id) return; id = null; this.touch.x = this.touch.y = 0; knob.style.transform = ''; };
    stick.addEventListener('pointerup', end); stick.addEventListener('pointercancel', end);
    const hold = (el, key) => { el.addEventListener('pointerdown', (e) => { e.preventDefault(); this.touch[key] = true; this.touch.on = true; el.classList.add('on'); if (this.handlers.any) this.handlers.any(e); }); const up = () => { this.touch[key] = false; el.classList.remove('on'); }; el.addEventListener('pointerup', up); el.addEventListener('pointercancel', up); el.addEventListener('pointerleave', up); };
    hold(flapBtn, 'flap'); hold(tuckBtn, 'tuck');
  }
  // dt-smoothed sticks so a tap is a nudge, not a kick
  read(dt) {
    const k = this.keys, has = (...c) => c.some((x) => k.has(x));
    let ud = (has('ArrowUp', 'KeyW') ? 1 : 0) - (has('ArrowDown', 'KeyS') ? 1 : 0);
    let lr = (has('ArrowRight', 'KeyD') ? 1 : 0) - (has('ArrowLeft', 'KeyA') ? 1 : 0);
    let flap = has('Space'), tuck = has('ShiftLeft', 'ShiftRight');
    let analog = false;
    if (this.touch.x || this.touch.y) { ud = -this.touch.y; lr = this.touch.x; analog = true; }
    flap = flap || this.touch.flap; tuck = tuck || this.touch.tuck;
    const gp = navigator.getGamepads ? [...navigator.getGamepads()].find((g) => g) : null;
    if (gp) {
      const dz = (v) => (Math.abs(v) < 0.12 ? 0 : v);
      if (dz(gp.axes[0]) || dz(gp.axes[1])) { lr = dz(gp.axes[0]); ud = -dz(gp.axes[1]); analog = true; }
      flap = flap || gp.buttons[0]?.pressed || gp.buttons[7]?.value > 0.3; tuck = tuck || gp.buttons[1]?.pressed || gp.buttons[6]?.value > 0.3;
    }
    const fwd = ud;
    const pt = this.invert ? -ud : ud;
    const rate = analog ? 12 : 4.5;
    this.pitch += clamp(pt - this.pitch, -rate * dt, rate * dt);
    this.roll += clamp(lr - this.roll, -rate * dt, rate * dt);
    return { pitch: this.pitch, roll: this.roll, fwd, flap, tuck, jump: false };
  }
}
