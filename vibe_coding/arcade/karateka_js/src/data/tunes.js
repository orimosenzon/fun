// The music and the sound of a blow. Our own: short phrases in a Japanese pentatonic scale, written
// here as notes and turned into the three bytes per note that the tune player (music.js) reads:
// how many times to repeat, the coarse and the fine half-period of the speaker's square wave.
//
// A note is [name, length]; '-' is a rest; the length is counted in steps
// of 4.5 ms times the tune's tempo.
const SEMITONE = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const hz = (name) => { const k = /^([A-G])(#|b)?(\d)$/.exec(name); return 440 * 2 ** ((SEMITONE[k[1]] + (k[2] === '#' ? 1 : k[2] === 'b' ? -1 : 0) - 9 + (k[3] - 4) * 12) / 12); };
const PASSES_PER_SECOND = 1020484 / 18;      // one pass of the player's inner loop takes 18 cycles of the 1 MHz processor

function compile({ voices = 2, tempo = 6, notes }, room) {
  const out = [tempo, voices === 2 ? 0xc0 : 0];
  for (const [name, length] of notes) {
    if (name === '-') { out.push(length, 0x81, 0x40); continue; }
    const half = Math.max(2, Math.round(PASSES_PER_SECOND / (2 * hz(name))));      // passes between two flips of the speaker
    const coarse = 1 + Math.floor((half - 1) / 256), fine = half - 256 * (coarse - 1);
    out.push(length, coarse, Math.min(255, fine));
  }
  out.push(0xff);
  if (out.length > room) throw new Error('tune too long: ' + out.length + ' > ' + room);
  return Uint8Array.from(out);
}

// scene 1. In scale: D E F A Bb (the "in" scale on D), which sounds like a koto more than like a bugle.
const SCENE1 = {
  // the scene opens: a rising call, answered lower
  0x1040: [75, { tempo: 13, notes: [['D4', 6], ['A4', 6], ['Bb4', 3], ['A4', 3], ['F4', 6], ['-', 3], ['E4', 3], ['F4', 3], ['A4', 6], ['D5', 12], ['-', 4], ['Bb4', 3], ['A4', 3], ['F4', 3], ['E4', 3], ['D4', 14]] }],
  // the hero falls: the same call turned downwards, slow
  0x108b: [52, { tempo: 15, notes: [['D5', 8], ['Bb4', 8], ['A4', 8], ['F4', 12], ['-', 4], ['E4', 6], ['F4', 6], ['E4', 6], ['D4', 20], ['-', 4], ['D3', 24]] }],
  // an opponent is down: three quick steps up
  0x10bf: [37, { tempo: 10, notes: [['A4', 3], ['Bb4', 3], ['D5', 3], ['-', 2], ['A4', 3], ['D5', 3], ['E5', 3], ['F5', 10], ['-', 2], ['D5', 12]] }],
  // the hall of the warlord appears: one dark note
  0x10e4: [16, { voices: 1, tempo: 10, notes: [['D3', 10], ['Eb3', 10], ['D3', 22]] }],
  // he points: go
  0x10f4: [40, { tempo: 10, notes: [['A3', 4], ['Bb3', 4], ['A3', 4], ['-', 2], ['D4', 5], ['E4', 5], ['F4', 16]] }],
};

export function tunes(scene) {
  if (scene !== 1) return [];
  return Object.entries(SCENE1).map(([at, [room, tune]]) => [+at, compile(tune, room)]);
}

// The sound of a blow landing: 2048 one-bit samples. A thump whose pitch drops fast, roughened
// with a little noise, then silence.
export function thud() {
  const out = new Uint8Array(256);
  let level = 0, left = 2, seed = 0x5a17, n = 0;
  for (let i = 0; i < 2048; i++) {
    if (i < 1100 && --left <= 0) {
      level ^= 1; n++;
      seed = (seed * 1103515245 + 12345) >>> 0;
      left = 2 + Math.floor(n / 6) + ((seed >>> 16) % 3);          // each half-wave a little longer than the last
    }
    if (level) out[i >> 3] |= 0x80 >> (i & 7);
  }
  return out;
}
