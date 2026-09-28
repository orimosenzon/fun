// Keyboard (smoothed to analog), gamepad and touch -> one pilot input object.
// Keys (kept close to fable): arrows = bank / pitch (up = nose down, like an aircraft),
// W/S throttle, E/D climb/descend, Z/X rudder, Space emergency power.

const K = new Set();
const once = new Set();
addEventListener('keydown', (e) => {
  if (!K.has(e.code)) once.add(e.code);
  K.add(e.code);
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) e.preventDefault();
});
addEventListener('keyup', (e) => K.delete(e.code));
addEventListener('blur', () => K.clear());

const approach = (x, target, rate, dt) => x + Math.max(-rate * dt, Math.min(rate * dt, target - x));
const expo = (x, k = 0.45) => x * (1 - k + k * x * x);
const dz = (x, d = 0.12) => (Math.abs(x) < d ? 0 : (x - Math.sign(x) * d) / (1 - d));

export class Input {
  constructor() {
    this.state = { stickX: 0, stickY: 0, rudder: 0, climb: 0, throttle: 0, boost: false, assist: true };
    this.kx = 0; this.ky = 0; this.kr = 0; this.kc = 0;
    this.touch = null;
    this.usingPad = false;
  }
  pressed(code) { const h = once.has(code); once.delete(code); return h; }
  endFrame() { once.clear(); }

  update(dt) {
    const s = this.state;
    const key = (c) => (K.has(c) ? 1 : 0);
    // keyboard: ramp toward the target so taps are gentle and holds are full deflection
    this.kx = approach(this.kx, key('ArrowRight') - key('ArrowLeft'), (key('ArrowRight') || key('ArrowLeft')) ? 2.6 : 5, dt);
    this.ky = approach(this.ky, key('ArrowUp') - key('ArrowDown'), (key('ArrowUp') || key('ArrowDown')) ? 2.4 : 5, dt);
    this.kr = approach(this.kr, key('KeyX') - key('KeyZ'), 4, dt);
    this.kc = approach(this.kc, key('KeyE') - key('KeyD'), 4, dt);
    if (K.has('KeyW')) s.throttle = Math.min(1, s.throttle + 0.7 * dt);
    if (K.has('KeyS')) s.throttle = Math.max(0, s.throttle - 0.9 * dt);
    let x = this.kx, y = this.ky, r = this.kr, c = this.kc, boost = K.has('Space');

    // gamepad (standard mapping)
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    const gp = [...pads].find((p) => p && p.connected);
    if (gp) {
      const ax = gp.axes;
      const lx = dz(ax[0] || 0), ly = dz(ax[1] || 0), rx = dz(ax[2] || 0), ry = dz(ax[3] || 0);
      const rt = gp.buttons[7]?.value || 0, lt = gp.buttons[6]?.value || 0;
      if (Math.abs(lx) + Math.abs(ly) + Math.abs(rx) + Math.abs(ry) + rt + lt > 0.05) this.usingPad = true;
      if (this.usingPad) {
        // stick up (ly < 0) = push forward = nose down = stickY +1
        x = expo(lx); y = expo(-ly);
        r = expo(rx); c = -expo(ry);
        s.throttle = Math.max(s.throttle - 0.9 * dt * lt, Math.min(1, s.throttle + (rt > 0.05 ? (rt - s.throttle) * 3 * dt : 0)));
        boost = boost || !!gp.buttons[0]?.pressed;
        this.padButtons = gp.buttons.map((b) => b.pressed);
      }
    }
    // touch overlay
    if (this.touch && this.touch.active) {
      const t = this.touch;
      x = expo(t.lx); y = expo(t.ly); r = expo(t.rx); c = expo(t.ry);
      s.throttle = t.throttle; boost = boost || t.boost;
    }
    s.stickX = Math.max(-1, Math.min(1, x));
    s.stickY = Math.max(-1, Math.min(1, y));
    s.rudder = Math.max(-1, Math.min(1, r));
    s.climb = Math.max(-1, Math.min(1, c));
    s.boost = boost;
    return s;
  }
  padPressed(i) {
    const now = this.padButtons?.[i] || false;
    this._prevPad ??= [];
    const was = this._prevPad[i] || false;
    this._prevPad[i] = now;
    return now && !was;
  }
}

// Minimal touch controls: left stick (bank/pitch), right stick (rudder/climb),
// throttle slider, boost button.
export function makeTouch(root, input) {
  const t = { active: false, lx: 0, ly: 0, rx: 0, ry: 0, throttle: 0, boost: false };
  input.touch = t;
  const box = document.createElement('div');
  box.className = 'touch';
  box.innerHTML = `<div class="tstick" id="tl"><div class="knob"></div></div><div class="tstick" id="tr"><div class="knob"></div></div>
    <div class="tthr"><div class="tfill"></div><span>מצערת</span></div><div class="tboost">חירום</div>`;
  root.appendChild(box);
  const bindStick = (el, set) => {
    const knob = el.querySelector('.knob');
    let id = null;
    const move = (e) => {
      const r = el.getBoundingClientRect();
      const x = ((e.clientX - r.left) / r.width) * 2 - 1, y = ((e.clientY - r.top) / r.height) * 2 - 1;
      const l = Math.hypot(x, y), k = l > 1 ? 1 / l : 1;
      set(x * k, y * k);
      knob.style.transform = `translate(${x * k * 40}px, ${y * k * 40}px)`;
    };
    el.addEventListener('pointerdown', (e) => { id = e.pointerId; el.setPointerCapture(id); t.active = true; move(e); });
    el.addEventListener('pointermove', (e) => { if (e.pointerId === id) move(e); });
    const up = (e) => { if (e.pointerId === id) { id = null; set(0, 0); knob.style.transform = ''; } };
    el.addEventListener('pointerup', up); el.addEventListener('pointercancel', up);
  };
  bindStick(box.querySelector('#tl'), (x, y) => { t.lx = x; t.ly = -y; });   // push up = nose down
  bindStick(box.querySelector('#tr'), (x, y) => { t.rx = x; t.ry = -y; });
  const thr = box.querySelector('.tthr'), fill = box.querySelector('.tfill');
  const setThr = (e) => { const r = thr.getBoundingClientRect(); t.throttle = Math.max(0, Math.min(1, 1 - (e.clientY - r.top) / r.height)); fill.style.height = `${t.throttle * 100}%`; t.active = true; };
  thr.addEventListener('pointerdown', (e) => { thr.setPointerCapture(e.pointerId); setThr(e); });
  thr.addEventListener('pointermove', (e) => { if (e.buttons) setThr(e); });
  const bb = box.querySelector('.tboost');
  bb.addEventListener('pointerdown', () => { t.boost = true; t.active = true; });
  bb.addEventListener('pointerup', () => { t.boost = false; });
  return t;
}
