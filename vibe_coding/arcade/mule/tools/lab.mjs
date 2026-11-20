// Scripted play sessions. A Session wraps a machine and counts time in seconds:
//   const s = new Session('name'); s.wait(3); s.joy(1, 'F', 0.3); s.key('F7'); s.shot('label');
// Screenshots go to reference/shots/<name>_<nnn>_<label>.png. Everything is deterministic, so a
// script that grows step by step replays the same game every time.
import fs from 'node:fs';
import { startGame, shot, KEYS, JOY } from './node64.mjs';

const BITS = { U: JOY.UP, D: JOY.DOWN, L: JOY.LEFT, R: JOY.RIGHT, F: JOY.FIRE };
export const bits = (s) => [...(s || '')].reduce((b, ch) => b | BITS[ch], 0);

export class Session {
  constructor(name = 'lab', opts = {}) {
    this.name = name;
    this.m = startGame(opts);
    this.fps = this.m.model.hz / (this.m.model.lines * this.m.model.cpl);
    this.n = 0; this.frames = 0; this.every = 0; this.quiet = !!opts.quiet;
    this.dir = new URL('../reference/shots/', import.meta.url).pathname;
    fs.mkdirSync(this.dir, { recursive: true });
  }
  get t() { return this.frames / this.fps; }
  frame() {
    this.m.runFrame(); this.frames++;
    if (this.m.cpu.jammed) throw new Error('CPU jammed at $' + this.m.cpu.pc.toString(16));
    if (this.every && this.frames % Math.round(this.every * this.fps) === 0) this.shot('t' + Math.round(this.t));
    if (this.onFrame) this.onFrame(this);
  }
  wait(secs) { const n = Math.round(secs * this.fps); for (let i = 0; i < n; i++) this.frame(); return this; }
  // Hold a joystick (port 1 or 2) in a state for a while, then release.
  joy(port, state, secs = 0.2, after = 0.1) { this.m.joy[port - 1] = bits(state); this.wait(secs); this.m.joy[port - 1] = 0; return this.wait(after); }
  hold(port, state) { this.m.joy[port - 1] = bits(state); return this; }
  key(name, secs = 0.15, after = 0.2) { this.m.setKey(KEYS[name], true); this.wait(secs); this.m.setKey(KEYS[name], false); return this.wait(after); }
  shot(label = '') { if (this.quiet) return this; shot(this.m, `${this.dir}${this.name}_${String(this.n++).padStart(3, '0')}_${label}.png`); return this; }
  // Wait until fn(session) is true (checked every frame), at most max seconds.
  until(fn, max = 60) { const end = this.frames + max * this.fps; while (!fn(this) && this.frames < end) this.frame(); return this; }
  // The game's text screen. Its character set is ASCII with the top bit set; double-width
  // letters are two cells (code|$80, code), collapsed here to one.
  text() {
    const base = ((this.m.vic.reg[0x18] & 0xf0) << 6) | this.m.vic.bank, ram = this.m.ram, lines = [];
    for (let r = 0; r < 25; r++) {
      let line = '';
      for (let c = 0; c < 40; c++) {
        const v = ram[base + r * 40 + c], ch = v & 0x7f;
        if (c && !(v & 0x80) && ram[base + r * 40 + c - 1] === (v | 0x80)) continue;
        line += ch >= 0x20 && ch < 0x60 ? String.fromCharCode(ch) : ' ';
      }
      lines.push(line.replace(/\s+/g, ' ').trim());
    }
    return lines.filter(Boolean).join(' | ');
  }
  sees(...words) { const t = this.text(); return words.every((w) => t.includes(w)); }
  untilSees(words, max = 120) { words = [].concat(words); const t0 = this.t; this.until((s) => s.frames % 5 === 0 && s.sees(...words), max); if (!this.sees(...words)) throw new Error('never saw ' + words + ' (from ' + t0.toFixed(0) + 's): ' + this.text()); return this; }
  sprites() { const r = this.m.vic.reg, out = []; for (let i = 0; i < 8; i++) out.push({ on: (r[0x15] >> i) & 1, x: r[i * 2] | (((r[0x10] >> i) & 1) << 8), y: r[i * 2 + 1], color: r[0x27 + i] & 15 }); return out; }
}

// Title -> options -> start with the given number of humans (0-4) and level (0 beginner, 1 standard, 2 tournament).
export function begin(s, humans = 1, level = 0) {
  s.wait(3).joy(1, 'F', 0.3).wait(13);
  for (let i = 0; i < level; i++) s.key('F3', 0.15, 0.6);
  for (let i = 4; i > humans; i--) s.key('F5', 0.15, 0.6);
  s.wait(0.5).shot('options').key('F7').wait(2);
  return s;
}

// Land grant: the moving frame is sprite 1, at x = 40 + 32*col, y = 66 + 32*row (9 columns, 5 rows).
export const plotX = (col) => 40 + 32 * col, plotY = (row) => 66 + 32 * row;
export function grab(s, col, row, port = 1, max = 40) {
  s.until((s) => { const p = s.sprites()[1]; return p.on && p.x === plotX(col) && p.y === plotY(row); }, max);
  return s.joy(port, 'F', 0.2);
}
// Colour of the frame around a plot (0 or 1 while nobody owns it).
export const owner = (s, col, row) => s.m.colorRam[(2 + 4 * row) * 40 + 2 + 4 * col] & 15;
