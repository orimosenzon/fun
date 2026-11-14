// Karateka, rewritten in JavaScript. No emulator and none of the original program runs here:
// every routine of the game is a JavaScript method below (and in the files that extend this class).
//
// The game keeps its state where the original kept it: one array of 64K bytes, `m`. That is a
// deliberate choice. It lets every routine be checked against the original, byte for byte
// (tools/check.mjs), and it lets the game read its pictures and tables from the player's own
// disk, which is not part of this project. Method names carry the address of the routine they
// re-implement (r7366 = the routine at $7366) next to a comment saying what it does.
//
// a, x, y and c stand for what the original passed between routines in the processor's
// registers and carry flag; a routine sets them only where its callers read them.
import { Gfx } from './gfx.js';

export class Game {
  constructor(mem) {
    this.m = mem;
    this.g = new Gfx(mem);
    this.a = 0; this.x = 0; this.y = 0; this.c = 0;
    this.key = 0;            // the keyboard latch: bit 7 set = a key is waiting
    this.clicks = [];        // when the speaker flips, in cycles
    this.flips = [];         // when each finished frame is shown, and which page it is on
    this.deaf = [];          // stretches of time (music) during which the original swallowed every key
    this.paused = false;
    this.cycles = 0;         // an estimate of the time the original would have spent
  }
}
// helpers shared by the routine files
export const u8 = (v) => v & 0xff;
export const s8 = (v) => (v & 0x80 ? v - 256 : v);
export function extend(methods) { for (const k of Object.getOwnPropertyNames(methods)) Object.defineProperty(Game.prototype, k, Object.getOwnPropertyDescriptor(methods, k)); }
