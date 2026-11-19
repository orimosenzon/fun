// Keyboard, gamepads and touch → C64 keyboard matrix and joystick ports.

import { KEYS, JOY } from './vm/c64.js';

// Two key sets that act as joysticks. Set A: arrows + Space. Set B: WASD + F.
const STICK_A = { ArrowUp: JOY.UP, ArrowDown: JOY.DOWN, ArrowLeft: JOY.LEFT, ArrowRight: JOY.RIGHT, Space: JOY.FIRE, Numpad0: JOY.FIRE };
const STICK_B = { KeyW: JOY.UP, KeyS: JOY.DOWN, KeyA: JOY.LEFT, KeyD: JOY.RIGHT, KeyF: JOY.FIRE, ShiftLeft: JOY.FIRE };

// A second button for each set, which the C64 joystick never had. Game extensions use it
// (in Archon: the dragon's fire breath). Reported through onSpecial(port index).
const SPECIAL_A = new Set(['Enter', 'ShiftRight', 'NumpadEnter']);
const SPECIAL_B = new Set(['KeyE', 'KeyG', 'KeyX']);

// Everything else goes to the C64 keyboard by physical position.
const KEYMAP = {
  F1: ['F1'], F3: ['F3'], F5: ['F5'], F7: ['F7'],
  F2: ['LSHIFT', 'F1'], F4: ['LSHIFT', 'F3'], F6: ['LSHIFT', 'F5'], F8: ['LSHIFT', 'F7'],
  Escape: ['STOP'], Enter: ['RETURN'], Backspace: ['DEL'], ShiftRight: ['RSHIFT'], ControlLeft: ['CTRL'], Tab: ['CTRL'], AltLeft: ['CBM'],
  Home: ['HOME'], Minus: ['-'], Equal: ['='], Comma: [','], Period: ['.'], Slash: ['/'], Semicolon: [';'], Quote: [':'],
  BracketLeft: ['@'], BracketRight: ['*'], Backslash: ['UP_ARROW'], Backquote: ['LEFT_ARROW'],
};
for (const ch of 'ABCDEFGHIJKLMNOPQRSTUVWXYZ') KEYMAP['Key' + ch] = [ch];
for (const ch of '0123456789') KEYMAP['Digit' + ch] = [ch];

export class Input {
  constructor() {
    this.machine = null;
    this.twoPlayers = false;
    this.sticks = { a: 0, b: 0 };        // from the keyboard
    this.touch = 0;                      // from the on-screen pad
    this.pads = [0, 0];
    this.held = new Map();               // code → C64 keys it is holding
    this.onSpecial = null;               // (port index) => {}, second button pressed
    this.padSpecial = [false, false];
  }

  // Second button of set 'a' or 'b': same routing to the ports as the sticks.
  special(set) {
    if (!this.onSpecial) return;
    if (!this.twoPlayers) { this.onSpecial(0); this.onSpecial(1); }
    else this.onSpecial(set === 'a' ? 1 : 0);
  }

  attach(target = window) {
    target.addEventListener('keydown', (e) => this.key(e, true));
    target.addEventListener('keyup', (e) => this.key(e, false));
    window.addEventListener('blur', () => this.releaseAll());
  }

  key(e, down) {
    if (e.ctrlKey || e.metaKey || e.altKey) { if (!down) this.releaseAll(); return; }
    const code = e.code;
    if (code in STICK_A) { this.sticks.a = down ? this.sticks.a | STICK_A[code] : this.sticks.a & ~STICK_A[code]; e.preventDefault(); this.apply(); return; }
    if (code in STICK_B) { this.sticks.b = down ? this.sticks.b | STICK_B[code] : this.sticks.b & ~STICK_B[code]; e.preventDefault(); this.apply(); return; }
    if (this.onSpecial && (SPECIAL_A.has(code) || SPECIAL_B.has(code))) { if (down && !e.repeat) this.special(SPECIAL_A.has(code) ? 'a' : 'b'); e.preventDefault(); return; }
    const keys = KEYMAP[code];
    if (!keys) return;
    e.preventDefault();
    if (down) this.held.set(code, keys); else this.held.delete(code);
    this.applyKeys();
  }

  // Press a C64 key for a moment (on-screen buttons).
  tap(name, ms = 120) {
    const id = 'tap:' + name + Math.random();
    this.held.set(id, [name]); this.applyKeys();
    setTimeout(() => { this.held.delete(id); this.applyKeys(); }, ms);
  }

  applyKeys() {
    if (!this.machine) return;
    this.machine.keyCols.fill(0);
    for (const keys of this.held.values()) for (const k of keys) this.machine.setKey(KEYS[k], true);
  }

  // One player: every control drives both ports, so it works for either side.
  // Two players: WASD + F is port 1 (light, on the left), arrows + Space port 2 (dark).
  apply() {
    if (!this.machine) return;
    const a = this.sticks.a | this.touch | this.pads[0], b = this.sticks.b | this.pads[1];
    if (this.twoPlayers) { this.machine.joy[0] = b; this.machine.joy[1] = a; }
    else this.machine.joy[0] = this.machine.joy[1] = a | b;
  }

  pollGamepads() {
    if (!navigator.getGamepads) return;
    let i = 0, changed = false;
    for (const p of navigator.getGamepads()) {
      if (!p || i > 1) continue;
      const ax = p.axes, bt = (n) => p.buttons[n] && p.buttons[n].pressed;
      let v = 0;
      if (ax[1] < -0.5 || bt(12)) v |= JOY.UP;
      if (ax[1] > 0.5 || bt(13)) v |= JOY.DOWN;
      if (ax[0] < -0.5 || bt(14)) v |= JOY.LEFT;
      if (ax[0] > 0.5 || bt(15)) v |= JOY.RIGHT;
      if (bt(0) || (!this.onSpecial && (bt(1) || bt(2) || bt(3)))) v |= JOY.FIRE;
      const sp = !!(bt(1) || bt(2) || bt(3));
      if (sp && !this.padSpecial[i]) this.special(i ? 'b' : 'a');
      this.padSpecial[i] = sp;
      if (this.pads[i] !== v) { this.pads[i] = v; changed = true; }
      i++;
    }
    if (changed) this.apply();
  }

  releaseAll() { this.sticks.a = this.sticks.b = 0; this.touch = 0; this.held.clear(); this.applyKeys(); this.apply(); }
}
