// One-bit speaker: turns a list of click times (in CPU cycles) into audio.
export class Speaker {
  constructor(hz) { this.hz = hz; this.ac = null; this.level = 0.25; this.playAt = 0; }
  start() { if (!this.ac) this.ac = new AudioContext(); else this.ac.resume(); }
  // clicks: sorted cycle stamps within [c0, c1)
  push(clicks, c0, c1) {
    if (!this.ac) return;
    const rate = this.ac.sampleRate, n = Math.round((c1 - c0) / this.hz * rate);
    if (n <= 0 || !clicks.length) { if (!clicks.length) return; }
    const buf = this.ac.createBuffer(1, n, rate), d = buf.getChannelData(0), per = this.hz / rate;
    let ti = 0;
    for (let i = 0; i < n; i++) {
      const endC = c0 + (i + 1) * per; let prev = c0 + i * per, acc = 0;
      while (ti < clicks.length && clicks[ti] < endC) { acc += this.level * (clicks[ti] - prev); prev = clicks[ti]; this.level = -this.level; ti++; }
      d[i] = (acc + this.level * (endC - prev)) / per;
    }
    const src = this.ac.createBufferSource(); src.buffer = buf; src.connect(this.ac.destination);
    const now = this.ac.currentTime;
    if (this.playAt < now + 0.02 || this.playAt > now + 0.4) this.playAt = now + 0.08;
    src.start(this.playAt); this.playAt += n / rate;
  }
}
