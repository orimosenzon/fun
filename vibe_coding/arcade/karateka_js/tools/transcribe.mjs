// Development only, run once per scene: writes src/data/rules<N>.js, the game's rule tables as plain
// numbers (how a move unfolds, which blow reaches which pose, how each opponent behaves, where the
// pieces of a scene stand). These are numbers that define how the game plays, not pictures and not
// program; they are taken from the reference memory at the ranges the game was seen to read
// (tools/needs.mjs). Everything that is a picture or a tune is made by tools/art and is not copied.
// usage: node tools/transcribe.mjs
import fs from 'node:fs';
import { memoryFor } from './lib.mjs';
const m = memoryFor(false);
const TABLES = [
  [0x6300, 0x63f9, 'how every move unfolds: the pose of each step, for the legs and for the body'],
  [0x6c6d, 0x6c6d, 'the opponent: how often he thinks'],
  [0x70c6, 0x70ef, 'which class every pose belongs to when it is hit'],
  [0x7300, 0x7365, 'the hit tables: for each attack and distance, which poses it reaches and how hard'],
  [0x7446, 0x745a, 'what a blow does to the one who takes it'],
  [0x74f7, 0x74ff, 'where the flash of a blow is drawn, by height'],
  [0x7815, 0x781a, 'the hall: places of the figures'], [0x7820, 0x7825, ''], [0x782b, 0x7830, ''], [0x7836, 0x783b, ''],
  [0x7d0c, 0x7d0c, 'the hall: its pieces, where they stand'], [0x7d12, 0x7d1d, ''], [0x7d23, 0x7d2e, ''], [0x7d34, 0x7d3f, ''], [0x7d45, 0x7d4f, ''],
  [0xac33, 0xac3e, 'the cliff: its pieces'],
  [0xadf7, 0xae3e, 'the road: fence posts and the pieces of the background, where they stand'],
  [0xb19a, 0xb1ad, 'the mountain: its pieces'],
  [0xb60f, 0xb61a, 'the gate house: its pieces'], [0xb62f, 0xb63a, ''], [0xb64f, 0xb65a, ''], [0xb66f, 0xb67a, ''],
  [0xbb75, 0xbb86, 'the opponents of this scene: strength and character of each'],
  [0xbc06, 0xbc49, 'the gate: its pieces'],
  [0xbfe8, 0xbff6, 'the opponent: how keen he is to attack, to retreat, to block, by opponent'], [0xbffb, 0xbffd, ''],
];
const hex = (v, n = 2) => v.toString(16).padStart(n, '0');
const lines = TABLES.map(([a, b, what]) => `  [0x${hex(a, 4)}, '${[...m.subarray(a, b + 1)].map((v) => hex(v)).join('')}'],${what ? '   // ' + what : ''}`);
fs.writeFileSync(new URL('../src/data/rules1.js', import.meta.url), `// The rule tables of the first scene: [address, bytes in hexadecimal]. Numbers that say how the game
// plays (see tools/transcribe.mjs); the pictures and the tunes are our own and live in art1.js and tunes.js.
export default [
${lines.join('\n')}
];
`);
const low = JSON.parse(JSON.stringify([...m.subarray(0, 0x100)]));
console.log('variables read before written:', [0x07, 0x3f, 0x48, 0x4f, 0x63, 0x64, 0x6e, 0x73, 0x74, 0x7e, 0x86, 0x8b, 0x9b, 0x9c, 0xc4, 0xd0, 0xe4, 0xe5].map((a) => hex(a) + '=' + hex(low[a])).join(' '));
console.log('tune settings', [0x0d5e, 0x0d79, 0x0d7a, 0x0d88].map((a) => hex(m[a])).join(' '), 'sound', [...m.subarray(0x0dfb, 0x0dfe)].map((v) => hex(v)).join(' '), 'arrow', [...m.subarray(0x0b12, 0x0b1b)].map((v) => hex(v)).join(' '));
console.log('masks', [...m.subarray(0x1a34, 0x1a42)].map((v) => hex(v)).join(' '));
console.log('rows', [...m.subarray(0x800, 0x808)].map((v) => hex(v)).join(' '), [...m.subarray(0x8c0, 0x8c8)].map((v) => hex(v)).join(' '), 'flip', [...m.subarray(0x900, 0x904)].map((v) => hex(v)).join(' '), [...m.subarray(0x980, 0x988)].map((v) => hex(v)).join(' '));
