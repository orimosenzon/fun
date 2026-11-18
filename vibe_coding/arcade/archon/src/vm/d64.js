// D64 disk image reader (35 tracks, 683 sectors). Files only; no drive emulation.

const SECTORS = (t) => (t <= 17 ? 21 : t <= 24 ? 19 : t <= 30 ? 18 : 17);
const TYPES = ['DEL', 'SEQ', 'PRG', 'USR', 'REL'];

export class D64 {
  constructor(bytes) {
    this.bytes = bytes;
    this.offsets = [0, 0];
    for (let t = 1, o = 0; t <= 40; t++) { this.offsets[t] = o; o += SECTORS(t) * 256; }
  }
  sector(t, s) { const o = this.offsets[t] + s * 256; return this.bytes.subarray(o, o + 256); }

  directory() {
    const out = [];
    let t = 18, s = 1, guard = 0;
    while (t && guard++ < 40) {
      const sec = this.sector(t, s);
      for (let i = 0; i < 8; i++) {
        const e = sec.subarray(i * 32, i * 32 + 32);
        const type = e[2];
        if (!type) continue;
        let name = [];
        for (let k = 5; k < 21 && e[k] !== 0xa0; k++) name.push(e[k]);
        out.push({ name: Uint8Array.from(name), text: String.fromCharCode(...name), type: TYPES[type & 7] || '???', track: e[3], sector: e[4], blocks: e[30] | (e[31] << 8) });
      }
      t = sec[0]; s = sec[1];
    }
    return out;
  }

  // Follow a sector chain and return the file's bytes (for a PRG: load address first).
  readChain(t, s) {
    const parts = [];
    let guard = 0;
    while (t && guard++ < 800) {
      const sec = this.sector(t, s);
      if (sec[0]) parts.push(sec.subarray(2, 256)); else parts.push(sec.subarray(2, sec[1] + 1));
      t = sec[0]; s = sec[1];
    }
    const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }

  // CBM-style match: '*' ends the pattern, '?' matches one character.
  find(pattern) {
    const match = (name) => {
      for (let i = 0; i < pattern.length; i++) {
        if (pattern[i] === 0x2a) return true;
        if (i >= name.length) return false;
        if (pattern[i] !== 0x3f && pattern[i] !== name[i]) return false;
      }
      return name.length === pattern.length;
    };
    return this.directory().find((e) => e.type === 'PRG' && match(e.name)) || null;
  }

  load(pattern) { const e = this.find(pattern); return e ? this.readChain(e.track, e.sector) : null; }
}
