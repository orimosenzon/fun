// קלט: מקלדת ובקר משחק (Gamepad API, מיפוי סטנדרטי). מחזיר ערכים גולמיים בין −1 ל-1.
export class Input {
  constructor() {
    this.keys = new Set();
    this.pressed = new Set();   // לחיצות חד-פעמיות לפריים הנוכחי
    this.touch = { throttle: 0, brake: 0, steer: 0, lift: 0, boost: 0 };
    addEventListener('keydown', (e) => {
      if (e.target && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return;
      if (!this.keys.has(e.code)) this.pressed.add(e.code);
      this.keys.add(e.code);
      if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Tab'].includes(e.code)) e.preventDefault();
    });
    addEventListener('keyup', (e) => this.keys.delete(e.code));
    addEventListener('blur', () => this.keys.clear());
    this.padPrev = [];
  }
  k(...codes) { return codes.some(c => this.keys.has(c)) ? 1 : 0; }
  // לחיצה חד-פעמית (מקלדת או כפתור בבקר)
  hit(code) { return this.pressed.has(code); }
  endFrame() { this.pressed.clear(); }

  read() {
    const r = {
      throttle: this.k('KeyW', 'ArrowUp'),
      brake: this.k('KeyS', 'ArrowDown'),
      steer: this.k('KeyA', 'ArrowLeft') - this.k('KeyD', 'ArrowRight'),
      boost: this.k('ShiftLeft', 'ShiftRight'),
      lift: this.k('Space'),
      pitch: this.k('KeyW', 'ArrowUp') - this.k('KeyS', 'ArrowDown'),
      roll: this.k('KeyD', 'ArrowRight') - this.k('KeyA', 'ArrowLeft'),
      yaw: this.k('KeyQ') - this.k('KeyE'),
      handbrake: this.k('KeyZ'),
    };
    // מגע
    const t = this.touch;
    r.throttle = Math.max(r.throttle, t.throttle); r.brake = Math.max(r.brake, t.brake);
    if (t.steer) { r.steer = t.steer; r.roll = -t.steer; }
    r.lift = Math.max(r.lift, t.lift); r.boost = Math.max(r.boost, t.boost);
    // בקר משחק
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (const p of pads) {
      if (!p || !p.connected) continue;
      const ax = (i) => { const v = p.axes[i] || 0; return Math.abs(v) < 0.12 ? 0 : v; };
      const bt = (i) => (p.buttons[i] ? p.buttons[i].value : 0);
      const btn = (i) => !!(p.buttons[i] && p.buttons[i].pressed);
      r.throttle = Math.max(r.throttle, bt(7));
      r.brake = Math.max(r.brake, bt(6));
      if (ax(0)) { r.steer = -ax(0); r.roll = ax(0); }
      if (ax(1)) r.pitch = -ax(1);
      if (ax(2)) r.yaw = -ax(2);
      r.lift = Math.max(r.lift, bt(0));
      r.boost = Math.max(r.boost, bt(5));
      r.handbrake = Math.max(r.handbrake, bt(4));
      // כפתורים חד-פעמיים
      const map = { 2: 'KeyK', 3: 'KeyC', 1: 'KeyF', 8: 'KeyR', 9: 'Escape', 12: 'KeyV', 13: 'KeyX' };
      for (const [i, code] of Object.entries(map)) {
        const now = btn(+i);
        if (now && !this.padPrev[i]) this.pressed.add(code);
        this.padPrev[i] = now;
      }
      break;
    }
    return r;
  }
}
