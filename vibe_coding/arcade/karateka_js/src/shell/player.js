// Runs the game in real time. The game itself has no clock: every pass of its main loop says how
// many processor cycles the original would have needed (game.cycles), which frames it showed
// and when the speaker flipped. This plays that back at the original's speed, and runs the next
// pass only when the time for it has come, so that keys are read at the right moment.
import { Game } from '../index.js';
import { buildMemory } from '../loader.js';
import { Display } from './display.js';
import { Speaker } from './speaker.js';

export const CPU_HZ = 1020484;
const LEAD = 0.07;   // seconds between working a pass out and showing it: room to schedule its sound

export class Player {
  constructor(canvas, hooks = {}) {
    this.display = new Display(canvas); this.speaker = new Speaker(CPU_HZ); this.hooks = hooks;
    this.muted = false; this.running = false; this.state = 'idle'; this.queue = [];
    this.newGame();
  }
  newGame() {
    this.game = new Game(buildMemory(1)); this.game.rB700();
    this.queue.length = 0; this.state = 'play'; this.sync();
  }
  sync() { this.t0 = performance.now() / 1000; this.c0 = this.game.cycles; this.a0 = this.speaker.now; }
  clock() { return this.c0 + (performance.now() / 1000 - this.t0) * CPU_HZ; }          // where the original would be by now
  key(code) {
    const g = this.game;
    if (this.state === 'over') { this.overKey = true; return; }
    if (g.paused) { g.paused = false; this.sync(); return; }                              // any key ends the pause, and is swallowed
    const now = this.clock() - LEAD * CPU_HZ;
    if (g.deaf.some(([a, b]) => now >= a && now < b)) return;                             // the original swallows keys while a tune plays
    g.key = code | 0x80;
  }
  start() { this.speaker.start(); if (this.running) return; this.running = true; this.sync(); const tick = () => { if (!this.running) return; requestAnimationFrame(tick); this.frame(); }; requestAnimationFrame(tick); }
  stop() { this.running = false; }
  pause(on) { if (on) this.running = false; else if (!this.running) this.start(); }

  frame() {
    const g = this.game;
    let now = this.clock();
    if (now - g.cycles > CPU_HZ) { this.sync(); now = this.clock(); }                    // the tab was asleep: do not try to catch up
    for (let guard = 0; g.cycles <= now && guard < 8 && !g.paused && this.state !== 'stopped'; guard++) this.pass();
    const shown = now - LEAD * CPU_HZ;
    let f = null; while (this.queue.length && this.queue[0].at <= shown) f = this.queue.shift();
    if (f) this.display.draw(f.page, 0);
  }
  pass() {
    const g = this.game, from = g.cycles;
    if (this.state === 'over') {                                                           // the game-over screen waits for a key, about two seconds
      g.cycles += 0.1 * CPU_HZ;
      if (this.overKey || g.cycles - this.overAt > 2 * CPU_HZ) { g.again(); this.state = 'play'; }
    } else {
      const res = g.step();
      if (res === 'over') { g.gameOver(); this.state = 'over'; this.overKey = false; this.overAt = g.cycles; g.key &= 0x7f; }
      else if (res === 'next') { this.state = 'stopped'; this.hooks.onEnd?.('next'); }
      else if (res === 'restart') this.newGame();
    }
    for (const f of g.flips) this.queue.push(f); g.flips.length = 0;
    if (g.clicks.length) { if (!this.muted) this.speaker.play(g.clicks, from, g.cycles, this.a0 + (from - this.c0) / CPU_HZ + LEAD); g.clicks.length = 0; }
    const horizon = g.cycles - 30 * CPU_HZ; while (g.deaf.length && g.deaf[0][1] < horizon) g.deaf.shift();
  }
}
