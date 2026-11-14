// Development only: where the reference places each pose (boxes in pixels relative to the fighter's
// own column; rows relative to the top of the fighters' strip), to place our own drawings alike.
import { memoryFor } from './lib.mjs';
const m = memoryFor(false), s8 = (v) => (v << 24) >> 24, w = (lo, hi) => m[lo] | (m[hi] << 8);
const L = [], U = [];
for (let x = 0; x < 42; x++) {
  const p = w(0x6000 + x, 0x602a + x), sp = w(0x60a8 + x, 0x60d2 + x);
  L.push([x, 'legs x', s8(m[0x6054 + x]) * 7, '..', (s8(m[0x6054 + x]) + m[p + 1]) * 7, 'y', m[0x607e + x], '..', m[0x607e + x] + m[p], 'anchor', s8(m[0x6150 + x]) * 7 + s8(m[0x617a + x]), 'shadow', sp >> 8 ? [(s8(m[0x6054 + x]) + s8(m[0x60fc + x])) * 7, m[sp + 1] * 7, m[0x6126 + x], m[sp]].join('/') : '-', m[0x62be + x] ? 'stored mirrored' : ''].join(' '));
}
for (let x = 0; x < 38; x++) {
  const k = m[0x61a4 + x], p = w(0x6226 + x, 0x624c + x);
  U.push([x, 'head kind', k >= 7 ? k - 6 : 0, 'x', s8(m[0x61da + x]) * 7 + (k < 7 ? k : 0), 'y', m[0x6200 + x], 'body x', s8(m[0x6272 + x]) * 7, '..', (s8(m[0x6272 + x]) + m[p + 1]) * 7, 'y', m[0x6298 + x], '..', m[0x6298 + x] + m[p]].join(' '));
}
console.log(L.join('\n')); console.log(U.join('\n'));
const head = (lo, hi) => [0, 1, 2].map((y) => { const p = w(lo + y, hi + y); return p >> 8 ? m[p] + 'x' + m[p + 1] : '-'; }).join(' ');
console.log('heads: player', head(0x61cd, 0x61ca), '| masks', head(0x61d9, 0x61d6), '| opponents', [[0x61d3, 0x61d0], [0x62eb, 0x62e8], [0x62f1, 0x62ee], [0x62f7, 0x62f4], [0x62fd, 0x62fa], [0x6aed, 0x6aea], [0x6af3, 0x6af0]].map(([a, b]) => head(a, b)).join(' | '));
