// Development only: which bytes of a scene's starting memory does the game read before writing them
// itself, and which routine reads them? Pictures (re/catalog.json) and the screens are left out, so
// what is printed are the tables. Run tools/catalog.mjs first.
// usage: node tools/needs.mjs [--seeds N] [--frames N] [--ours]
import fs from 'node:fs';
import { Game } from '../src/index.js';
import { memoryFor, botPlay } from './lib.mjs';
const args = process.argv.slice(2), opt = (k, d) => { const i = args.indexOf('--' + k); return i < 0 ? d : args[i + 1]; };
const need = new Uint8Array(0x10000), who = new Map();
let ctx = '';
for (const k of Object.getOwnPropertyNames(Game.prototype)) if (/^r[0-9A-F]{4}$/.test(k)) {
  const f = Game.prototype[k];
  Game.prototype[k] = function (...a) { const was = ctx; ctx = k; try { return f.apply(this, a); } finally { ctx = was; } };
}
function trace(mem) {
  const own = new Uint8Array(0x10000);
  const read = (i) => { if (!own[i] && !need[i]) { need[i] = 1; who.set(i, ctx); } };
  return new Proxy(mem, {
    get(t, k) {
      if (typeof k === 'string' && k.charCodeAt(0) < 58) { read(+k); return t[+k]; }
      const v = t[k]; if (typeof v !== 'function') return v;
      if (k === 'fill') return (val, a, b) => { own.fill(1, a, b); return t.fill(val, a, b); };
      if (k === 'copyWithin') return (d, s, e) => { for (let i = s; i < e; i++) read(i); own.fill(1, d, d + e - s); return t.copyWithin(d, s, e); };
      if (k === 'set') return (src, o = 0) => { own.fill(1, o, o + src.length); return t.set(src, o); };
      if (k === 'slice' || k === 'subarray') return (a, b) => { for (let i = a; i < b; i++) read(i); return t.slice(a, b); };
      throw new Error('untraced method ' + k);
    },
    set(t, k, v) { own[+k] = 1; t[+k] = v; return true; },
  });
}
for (let seed = 1; seed <= +opt('seeds', 4); seed++) for (const cheat of [true, false]) botPlay(new Game(trace(memoryFor(args.includes('--ours')))), { seed, cheat, frames: +opt('frames', 6000), end: true });
need.fill(0, 0x2000, 0x6000);
for (const [p, e] of Object.entries(JSON.parse(fs.readFileSync(new URL('../re/catalog.json', import.meta.url))))) need.fill(0, parseInt(p, 16), parseInt(p, 16) + 2 + e.h * e.w);
const hex = (v) => v.toString(16).padStart(4, '0');
let total = 0; const out = [];
for (let a = 0; a < 0x10000; a++) if (need[a]) {
  let b = a; while (b < 0x10000 && (need[b] || need[b + 1] || need[b + 2] || need[b + 3])) b++;
  const by = new Set(); let cnt = 0; for (let i = a; i < b; i++) if (need[i]) { cnt++; by.add(who.get(i)); }
  total += cnt; out.push('$' + hex(a) + '-$' + hex(b - 1) + '  ' + (b - a) + ' (' + cnt + ')  ' + [...by].join(' ')); a = b;
}
console.log(out.join('\n')); console.log('total read:', total);
fs.writeFileSync(new URL('../re/needs.bin', import.meta.url), need);
