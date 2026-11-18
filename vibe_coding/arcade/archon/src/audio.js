// Sound output: the SID's samples are handed to an AudioWorklet that plays
// them from a small queue (about 60 ms), dropping or padding when the
// emulation and the sound card drift apart.

const WORKLET = `
class SidOut extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = []; this.pos = 0; this.queued = 0; this.primed = false; this.last = 0;
    this.port.onmessage = (e) => {
      this.queue.push(e.data); this.queued += e.data.length;
      const max = sampleRate * 0.25;
      while (this.queued > max && this.queue.length > 1) { this.queued -= this.queue[0].length - this.pos; this.queue.shift(); this.pos = 0; }
    };
  }
  process(inputs, outputs) {
    const out = outputs[0][0];
    if (!this.primed && this.queued >= sampleRate * 0.05) this.primed = true;
    for (let i = 0; i < out.length; i++) {
      if (this.primed && this.queue.length) {
        const chunk = this.queue[0];
        this.last = chunk[this.pos++]; this.queued--;
        if (this.pos >= chunk.length) { this.queue.shift(); this.pos = 0; }
      } else { this.primed = false; this.last *= 0.98; }
      out[i] = this.last;
    }
    return true;
  }
}
registerProcessor('sid-out', SidOut);`;

export class AudioOut {
  constructor() { this.ctx = null; this.node = null; this.gain = null; this.muted = false; }
  get sampleRate() { return this.ctx ? this.ctx.sampleRate : 44100; }

  // Must be called from a user gesture.
  async start() {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return false;
    try {
      this.ctx = new Ctx({ latencyHint: 'interactive' });
      const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
      await this.ctx.audioWorklet.addModule(url);
      this.node = new AudioWorkletNode(this.ctx, 'sid-out', { numberOfInputs: 0, outputChannelCount: [1] });
      this.gain = this.ctx.createGain();
      this.node.connect(this.gain).connect(this.ctx.destination);
      await this.ctx.resume();
      return true;
    } catch (e) { console.warn('no sound:', e); this.node = null; return false; }
  }

  push(samples) { if (this.node && samples.length) this.node.port.postMessage(samples, [samples.buffer]); }
  setMuted(m) { this.muted = m; if (this.gain) this.gain.gain.value = m ? 0 : 1; }
  resume() { if (this.ctx && this.ctx.state !== 'running') this.ctx.resume(); }
}
