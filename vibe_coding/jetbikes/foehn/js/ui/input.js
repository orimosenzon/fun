// Keyboard, gamepad and touch -> one control state.
export class Input {
  constructor() {
    this.keys = new Set(); this.pressed = [];        // key-down edges, drained by the game
    this.invert = false; this.touch = { turn: 0, pitch: 0, throttle: 0, boost: false, brake: false, on: false };
    addEventListener('keydown', (e) => {
      if (e.repeat) return;
      if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) e.preventDefault();
      this.keys.add(e.code); this.pressed.push(e.code);
    });
    addEventListener('keyup', (e) => this.keys.delete(e.code));
    addEventListener('blur', () => this.keys.clear());
    this.padPrev = [];
  }
  read() {
    const k = (c) => this.keys.has(c);
    const c = {
      turn: (k('ArrowRight') ? 1 : 0) - (k('ArrowLeft') ? 1 : 0),
      pitch: ((k('ArrowDown') ? 1 : 0) - (k('ArrowUp') ? 1 : 0)) * (this.invert ? -1 : 1),
      strafe: (k('KeyD') ? 1 : 0) - (k('KeyA') ? 1 : 0),
      throttle: (k('KeyW') ? 1 : 0) - (k('KeyS') ? 1 : 0),
      boost: k('Space'), brake: k('ShiftLeft') || k('ShiftRight') || k('KeyB'),
    };
    // gamepad: left stick turn / pitch, right stick x side-step, triggers throttle, A boost, B brake
    const gp = navigator.getGamepads ? [...navigator.getGamepads()].find((g) => g && g.connected) : null;
    if (gp) {
      const dz = (v) => (Math.abs(v) < 0.12 ? 0 : (v - Math.sign(v) * 0.12) / 0.88);
      const tx = dz(gp.axes[0] || 0), ty = dz(gp.axes[1] || 0), sx = dz(gp.axes[2] || 0);
      if (tx) c.turn = tx; if (ty) c.pitch = ty * (this.invert ? -1 : 1) * -1 * -1; if (sx) c.strafe = sx;
      const rt = gp.buttons[7] ? gp.buttons[7].value : 0, lt = gp.buttons[6] ? gp.buttons[6].value : 0;
      if (rt > 0.05 || lt > 0.05) c.throttle = rt - lt;
      if (gp.buttons[0] && gp.buttons[0].pressed) c.boost = true;
      if (gp.buttons[1] && gp.buttons[1].pressed) c.brake = true;
      const edge = (i, code) => { const p = gp.buttons[i] && gp.buttons[i].pressed; if (p && !this.padPrev[i]) this.pressed.push(code); this.padPrev[i] = p; };
      edge(3, 'KeyC'); edge(2, 'KeyR'); edge(9, 'Escape'); edge(0, 'PadA');
    }
    const t = this.touch;
    if (t.on) { if (t.turn) c.turn = t.turn; if (t.pitch) c.pitch = t.pitch; if (t.throttle) c.throttle = t.throttle; c.boost = c.boost || t.boost; c.brake = c.brake || t.brake; }
    return c;
  }
  // on-screen stick and buttons for phones
  enableTouch(root) {
    const t = this.touch; t.on = true; root.classList.add('on');
    const stick = root.querySelector('.stick'), knob = stick.querySelector('i');
    const move = (e) => { const r = stick.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * 2 - 1, y = (e.clientY - r.top) / r.height * 2 - 1; const l = Math.max(1, Math.hypot(x, y)); t.turn = x / l; t.pitch = y / l * (this.invert ? -1 : 1); knob.style.transform = `translate(${x / l * 40}px,${y / l * 40}px)`; };
    stick.addEventListener('pointerdown', (e) => { stick.setPointerCapture(e.pointerId); move(e); });
    stick.addEventListener('pointermove', (e) => { if (e.buttons || e.pointerType === 'touch') move(e); });
    const end = () => { t.turn = t.pitch = 0; knob.style.transform = ''; };
    stick.addEventListener('pointerup', end); stick.addEventListener('pointercancel', end);
    for (const b of root.querySelectorAll('[data-k]')) {
      const k = b.dataset.k, set = (v) => { if (k === 'up') t.throttle = v ? 1 : 0; else if (k === 'down') t.throttle = v ? -1 : 0; else if (k === 'key') { if (v) this.pressed.push(b.dataset.code); } else t[k] = v; b.classList.toggle('act', v); };
      b.addEventListener('pointerdown', (e) => { e.preventDefault(); set(true); }); b.addEventListener('pointerup', () => set(false)); b.addEventListener('pointercancel', () => set(false)); b.addEventListener('pointerleave', () => set(false));
    }
  }
}
