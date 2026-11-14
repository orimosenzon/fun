// One-bit speaker: turns the moments the game flipped the speaker (in processor cycles) into audio.
export class Speaker {
  constructor(hz) { this.hz = hz; this.ac = null; this.level = 0.2; }
  start() { if (!this.ac) this.ac = new AudioContext(); else this.ac.resume(); }
  get now() { return this.ac ? this.ac.currentTime : 0; }
  // clicks: sorted cycle stamps within [c0, c1); the stretch starts playing at audio time `at`
  play(clicks, c0, c1, at) {
    if (!this.ac || !clicks.length) return;
    const rate = this.ac.sampleRate, n = Math.ceil((c1 - c0) / this.hz * rate) + 1, per = this.hz / rate;
    const buf = this.ac.createBuffer(1, n, rate), d = buf.getChannelData(0);
    let ti = 0, level = this.level;
    for (let i = 0; i < n; i++) {
      const endC = c0 + (i + 1) * per; let prev = c0 + i * per, acc = 0;
      while (ti < clicks.length && clicks[ti] < endC) { acc += level * (clicks[ti] - prev); prev = clicks[ti]; level = -level; ti++; }
      d[i] = (acc + level * (endC - prev)) / per;
    }
    // let the cone settle back to rest instead of holding a level between sounds
    const tail = Math.min(n, 64); for (let i = 0; i < tail; i++) d[n - 1 - i] *= i / tail;
    const src = this.ac.createBufferSource(); src.buffer = buf; src.connect(this.ac.destination);
    src.start(Math.max(at, this.ac.currentTime));
  }
}
