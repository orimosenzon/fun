// Differential test: run an original 6502 routine in the reference emulator and the JS
// port on identical memory, and require identical memory afterwards.
import fs from 'node:fs';
import path from 'node:path';
import { Machine } from './emu/machine.mjs';
import { loadRoms } from './emu/apple2.mjs';
import { Gfx } from '../src/gfx.js';

const root = path.resolve(import.meta.dirname, '../reference');
const roms = loadRoms(path.join(root, 'roms'));
const SENT = 0x02f0;

export function makeOriginal(ramFile = 'game.ram') {
  const m = new Machine({ ...roms, dsk: null });
  m.ram.set(fs.readFileSync(path.join(root, ramFile)));
  m.call = (addr, maxCycles = 5e7) => {
    const c = m.cpu; c.sp = 0xff; c.push((SENT - 1) >> 8); c.push((SENT - 1) & 0xff); c.pc = addr; c.d = 0;
    const lim = c.cycles + maxCycles;
    while (c.pc !== SENT) { c.step(); if (c.cycles > lim) throw new Error('runaway at ' + c.pc.toString(16)); }
  };
  return m;
}

function firstDiff(a, b, lo = 0, hi = 0xc000) {
  for (let i = lo; i < hi; i++) if (a[i] !== b[i] && !(i >= 0x100 && i < 0x200) && !(i >= SENT - 2 && i < SENT + 2) && !(i >= 0x1900 && i < 0x1c78)) return i; // $19xx: self-modified opcodes
  return -1;
}

const ENTRY = { blit: 0x1903, blitR: 0x1906, blitM: 0x1909, blitMR: 0x190c, fill: 0x0a00, fill2: 0x0a03, clear: 0x1900 };
const PORT = { blit: (g) => g.blit(), blitR: (g) => g.blitRight(), blitM: (g) => g.blitMirror(), blitMR: (g) => g.blitMirrorRight(), fill: (g) => g.fill(false), fill2: (g) => g.fill(true), clear: (g) => g.clear() };

function runSuite(name) {
  const { calls, shapes } = JSON.parse(fs.readFileSync(path.join(root, `${name}.calls.json`)));
  const orig = makeOriginal();
  const mem = new Uint8Array(orig.ram);
  const gfx = new Gfx(mem);
  let n = 0, fails = 0;
  const one = (r) => {
    if (!ENTRY[r.e]) return;
    for (const z of [orig.ram, mem]) {
      z[5] = r.x; z[6] = r.y; z[7] = r.page; z[0x10] = r.shift; z[0x0f] = r.mode;
      if (r.ptr !== undefined) { z[3] = r.ptr & 0xff; z[4] = r.ptr >> 8; z.set(Buffer.from(shapes[r.ptr.toString(16)].data, 'hex'), r.ptr); if (r.h !== undefined) z[r.ptr] = r.h; }
      if (r.x2 !== undefined) { z[9] = r.x2; z[8] = r.y2; z[2] = r.a; z[0x11] = r.b; z[0x12] = r.c; z[0x13] = r.d; }
    }
    try { orig.call(ENTRY[r.e]); } catch (e) { console.log('  ' + e.message, JSON.stringify(r), shapes[r.ptr?.toString(16)]?.h, shapes[r.ptr?.toString(16)]?.w); process.exit(1); }
    PORT[r.e](gfx);
    n++;
    // scratch zero page ($00,$01,$0A,$0B,$17-$1A) is not part of the contract
    let d = firstDiff(orig.ram, mem, 0x200);
    if (d < 0) for (const a of [3, 4, 5, 6, 7, 8, 9, 0x0d, 0x0e, 0x0f, 0x10]) if (orig.ram[a] !== mem[a]) { d = a; break; }
    if (d >= 0) {
      if (fails++ < 5) console.log(`  MISMATCH ${name} #${n}`, JSON.stringify(r), 'at', d.toString(16), 'orig', orig.ram[d].toString(16), 'port', mem[d].toString(16));
      mem.set(orig.ram);
    }
  };
  for (const r of calls) one(r);
  const recorded = n;
  // fuzz: every shape at random positions, shifts, modes, incl. clipping on both sides
  let seed = 12345; const rnd = (k) => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % k; };
  const keys = Object.keys(shapes).filter((k) => parseInt(k, 16) < 0x2000 || parseInt(k, 16) >= 0x6000);
  for (let i = 0; i < 20000; i++) {
    const k = keys[rnd(keys.length)], sh = shapes[k];
    one({ e: ['blit', 'blitR', 'blitM', 'blitMR'][rnd(4)], ptr: parseInt(k, 16), x: (rnd(64) - 14) & 0xff, y: rnd(Math.max(1, 192 - sh.h)), page: rnd(2) ? 0x20 : 0x40, shift: rnd(7), mode: [0, 0x80, 1, 0xff][rnd(4)] });
  }
  console.log(`${name}: ${recorded} recorded + ${n - recorded} fuzz calls, ${fails} mismatches`);
  return fails;
}

if (process.argv[1] === import.meta.filename) {
  let fails = 0;
  for (const s of ['game', 'demo']) fails += runSuite(s);
  process.exit(fails ? 1 : 0);
}
