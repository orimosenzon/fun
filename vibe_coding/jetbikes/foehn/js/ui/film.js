// Turns the replay on screen into a video file. Every drawn frame is copied onto a 16:9 canvas
// (the picture between the black bars, a caption under it) and that canvas is recorded together
// with the sound of the game. It records in real time: a two-minute flight takes two minutes.
const TYPES = ['video/mp4;codecs=avc1.640028,mp4a.40.2', 'video/mp4;codecs=avc1.42E01E,mp4a.40.2', 'video/mp4', 'video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm'];
export const filmType = () => (typeof MediaRecorder === 'undefined' ? null : TYPES.find((m) => MediaRecorder.isTypeSupported(m)) || null);

export class Film {
  // src: the game's canvas. caption: { left, right } text for the lower bar. sound: a MediaStream or null
  constructor(src, caption, sound) {
    this.src = src; this.caption = caption;
    const big = src.width >= 1700;
    const c = this.canvas = document.createElement('canvas'); c.width = big ? 1920 : 1280; c.height = big ? 1080 : 720;
    this.g = c.getContext('2d', { alpha: false });
    this.mime = filmType();
    const stream = c.captureStream(60);
    if (sound) for (const tr of sound.getAudioTracks()) stream.addTrack(tr);
    this.rec = new MediaRecorder(stream, { mimeType: this.mime, videoBitsPerSecond: big ? 10e6 : 6e6, audioBitsPerSecond: 160e3 });
    this.chunks = []; this.rec.ondataavailable = (e) => { if (e.data && e.data.size) this.chunks.push(e.data); };
    this.started = false;
  }
  get ext() { return this.mime.startsWith('video/mp4') ? 'mp4' : 'webm'; }
  // copy the frame that was just drawn. fade: 0 picture .. 1 black
  grab(fade) {
    const g = this.g, W = this.canvas.width, H = this.canvas.height, s = this.src;
    // the same strip of the picture that the black bars leave on screen
    const sh = Math.min(s.height * 0.9, s.width / 2.39), sw = Math.min(s.width, sh * 2.39), dh = Math.round(W * sh / sw), dy = Math.round((H - dh) / 2);
    g.fillStyle = '#000'; g.fillRect(0, 0, W, H);
    g.drawImage(s, (s.width - sw) / 2, (s.height - sh) / 2, sw, sh, 0, dy, W, dh);
    if (fade > 0) { g.fillStyle = `rgba(0,0,0,${Math.min(1, fade)})`; g.fillRect(0, dy, W, dh); }
    const k = H / 720, y = dy + dh + (H - dy - dh) / 2;
    g.textBaseline = 'middle'; g.fillStyle = 'rgba(255,255,255,.82)';
    g.font = `900 ${22 * k}px system-ui, sans-serif`; g.direction = 'ltr'; g.textAlign = 'left'; g.fillText('FÖHN', 36 * k, y);
    g.font = `600 ${17 * k}px system-ui, sans-serif`; g.direction = this.caption.rtl ? 'rtl' : 'ltr'; g.textAlign = 'right'; g.fillText(this.caption.text, W - 36 * k, y);
    if (!this.started) { this.started = true; this.rec.start(1000); }
  }
  // resolves with the finished file
  finish() {
    return new Promise((res) => {
      if (this.rec.state === 'inactive') return res(null);
      this.rec.onstop = () => res(new Blob(this.chunks, { type: this.mime.split(';')[0] }));
      this.rec.stop();
    });
  }
  cancel() { if (this.rec.state !== 'inactive') { this.rec.onstop = null; this.rec.stop(); } this.chunks = []; }
}
