/* engine.js: מנוע הניגון וההרכבה
 *
 * אחראי על: הקצאת אלמנטי <video> לקליפים שנמצאים בזמן הנוכחי (או קרוב אליו),
 * סנכרון שלהם לשעון הטיימליין, ציור הפריים המורכב על הקנבס, וניתוב האודיו
 * דרך WebAudio (כדי שיהיו עוצמה, דעיכות ומיקס בין ערוצים).
 */

import { clamp } from './util.js';
import {
  state, proj, fps, duration, clipAt, prevAdjacent, clipEnd, mediaById,
} from './state.js';
import { rt, getAudioCtx } from './media.js';
import { drawTransition } from './transitions.js';

const HARD_SYNC = 0.22;    // סטייה שמעליה עושים seek
const SOFT_SYNC = 0.02;    // סטייה שמעליה מתקנים במהירות הניגון
const PREROLL   = 1.2;     // כמה שניות מראש להכין קליפ הבא
const MAX_PLAYERS = 10;

export class Engine {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.players = new Map();       // clipId → player
    this.playing = false;
    this.clockStart = 0;
    this.clockBase = 0;
    this.rafId = 0;
    this.onTick = null;             // (t) => void
    this.onEnd = null;
    this.rate = 1;                  // מהירות ניגון גלובלית (J/K/L)
    this.stopAt = null;             // עצירה אוטומטית בזמן מסוים
    this.audio = null;
    this.needsRender = true;
    this._dirtySeek = false;
  }

  /* ─────────── אודיו ─────────── */

  initAudio() {
    if (this.audio) return this.audio;
    const ac = getAudioCtx();
    const master = ac.createGain();
    master.gain.value = 1;
    master.connect(ac.destination);
    this.audio = { ac, master, stream: null };
    return this.audio;
  }

  /** יעד הקלטה לייצוא: מחזיר MediaStream של המיקס */
  captureAudioStream() {
    const a = this.initAudio();
    if (!a.stream) {
      a.streamDest = a.ac.createMediaStreamDestination();
      a.master.connect(a.streamDest);
      a.stream = a.streamDest.stream;
    }
    return a.stream;
  }

  /* ─────────── ניהול נגנים ─────────── */

  player(clip) {
    let p = this.players.get(clip.id);
    if (p && p.mediaId === clip.mediaId) { p.lastUse = performance.now(); return p; }
    if (p) this.release(clip.id);

    const media = mediaById(clip.mediaId);
    const r = media && rt(media.id);
    if (!media || !r?.url) return null;

    if (media.type === 'image') {
      p = { kind: 'image', el: r.image, mediaId: media.id, lastUse: performance.now() };
      this.players.set(clip.id, p);
      return p;
    }

    const el = document.createElement(media.type === 'audio' ? 'audio' : 'video');
    el.src = r.url;
    el.preload = 'auto';
    el.playsInline = true;
    el.crossOrigin = 'anonymous';
    el.volume = 1;
    el.muted = false;

    p = { kind: media.type, el, mediaId: media.id, lastUse: performance.now(), ready: false, wantPlay: false };
    // כשמגיע פריים חדש ואנחנו לא בניגון, מרעננים את התצוגה
    const refresh = () => { p.ready = true; if (!this.playing) this.render(state.playhead); };
    el.addEventListener('loadeddata', refresh);
    el.addEventListener('canplay', refresh);
    el.addEventListener('seeked', refresh);
    el.addEventListener('error', () => { p.broken = true; });

    // ניתוב אודיו
    if (media.hasAudio) {
      try {
        const a = this.initAudio();
        p.src = a.ac.createMediaElementSource(el);
        p.gain = a.ac.createGain();
        p.gain.gain.value = 0;
        p.src.connect(p.gain).connect(a.master);
      } catch (e) { console.warn('audio route failed', e); }
    }

    this.players.set(clip.id, p);
    return p;
  }

  release(clipId) {
    const p = this.players.get(clipId);
    if (!p) return;
    if (p.kind !== 'image' && p.el) {
      try { p.el.pause(); } catch {}
      try { p.gain?.disconnect(); p.src?.disconnect(); } catch {}
      p.el.removeAttribute('src');
      try { p.el.load(); } catch {}
    }
    this.players.delete(clipId);
  }

  releaseAll() {
    [...this.players.keys()].forEach((id) => this.release(id));
  }

  gc(now) {
    if (this.players.size <= MAX_PLAYERS) return;
    const old = [...this.players.entries()]
      .filter(([, p]) => now - p.lastUse > 2500)
      .sort((a, b) => a[1].lastUse - b[1].lastUse);
    while (this.players.size > MAX_PLAYERS && old.length) this.release(old.shift()[0]);
  }

  /* ─────────── מה צריך להתנגן עכשיו ─────────── */

  /**
   * מחזיר לכל ערוץ את הקליפים הרלוונטיים בזמן t.
   * לכל אחד: {clip, track, srcTime, mix} כאשר mix הוא משקל האודיו/וידאו (מעבר).
   */
  activeAt(t) {
    const out = [];
    for (const track of proj().tracks) {
      const c = clipAt(track, t);
      if (!c) continue;
      const entry = { track, clip: c, srcTime: srcTimeOf(c, t), mix: 1, role: 'main' };
      // מעבר בראש הקליפ: צריך גם את הקליפ הקודם
      if (c.tin && t < c.start + c.tin.dur) {
        const prev = prevAdjacent(track, c);
        if (prev) {
          const p = clamp((t - c.start) / c.tin.dur, 0, 1);
          entry.mix = p;
          entry.trans = { type: c.tin.type, p, prev };
          out.push({ track, clip: prev, srcTime: srcTimeOf(prev, t), mix: 1 - p, role: 'prev' });
        } else {
          entry.trans = null;
        }
      }
      out.push(entry);
    }
    return out;
  }

  /** קליפים שמתחילים בקרוב, שמכינים מראש כדי למנוע גמגום */
  upcomingAt(t) {
    const out = [];
    for (const track of proj().tracks) {
      for (const c of track.clips) {
        if (c.start > t && c.start < t + PREROLL) out.push(c);
      }
    }
    return out;
  }

  /* ─────────── סנכרון ─────────── */

  syncTo(t, hard = false) {
    const now = performance.now();
    const act = this.activeAt(t);
    const activeIds = new Set(act.map((a) => a.clip.id));

    for (const { clip, srcTime } of act) {
      const p = this.player(clip);
      if (!p || p.kind === 'image') continue;
      p.lastUse = now;
      const el = p.el;
      const want = clamp(srcTime, 0, Math.max(0, (el.duration || 1e9) - 0.03));
      const drift = el.currentTime - want;

      if (!this.playing) {
        if (Math.abs(drift) > 0.012) { el.pause(); el.currentTime = want; }
        continue;
      }
      if (hard || Math.abs(drift) > HARD_SYNC || el.readyState < 2) {
        el.currentTime = want;
        el.playbackRate = (clip.speed || 1) * this.rate;
      } else if (Math.abs(drift) > SOFT_SYNC) {
        // תיקון עדין: מאיצים/מאטים קצת עד שמתיישרים
        const corr = clamp(-drift * 0.6, -0.06, 0.06);
        el.playbackRate = clamp((clip.speed || 1) * this.rate * (1 + corr), 0.25, 4);
      } else {
        el.playbackRate = clamp((clip.speed || 1) * this.rate, 0.25, 4);
      }
      if (el.paused) el.play().catch(() => {});
    }

    // הכנה מראש
    if (this.playing) {
      for (const c of this.upcomingAt(t)) {
        if (activeIds.has(c.id)) continue;
        const p = this.player(c);
        if (!p || p.kind === 'image' || p.prerolled === c.inPoint) continue;
        p.prerolled = c.inPoint;
        try { p.el.currentTime = clamp(c.inPoint, 0, Math.max(0, (p.el.duration || 1e9) - 0.03)); } catch {}
      }
    }

    // עצירת מי שלא פעיל
    for (const [id, p] of this.players) {
      if (activeIds.has(id) || p.kind === 'image') continue;
      if (!p.el.paused) p.el.pause();
      if (p.gain) p.gain.gain.value = 0;
    }

    this.applyGains(act, t);
    this.gc(now);
  }

  applyGains(act, t) {
    if (!this.audio) return;
    const ac = this.audio.ac;
    const now = ac.currentTime;
    for (const a of act) {
      const p = this.players.get(a.clip.id);
      if (!p?.gain) continue;
      const g = this.playing ? clipGain(a.clip, a.track, t, a.role === 'prev') * a.mix : 0;
      const cur = p.gain.gain.value;
      if (Math.abs(cur - g) > 0.001) {
        p.gain.gain.cancelScheduledValues(now);
        p.gain.gain.setValueAtTime(cur, now);
        p.gain.gain.linearRampToValueAtTime(g, now + 0.03);
      }
    }
  }

  /* ─────────── ציור ─────────── */

  render(t = state.playhead) {
    const W = proj().width, H = proj().height;
    if (this.canvas.width !== W || this.canvas.height !== H) {
      this.canvas.width = W; this.canvas.height = H;
    }
    const ctx = this.ctx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.filter = 'none';
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, W, H);

    // ערוצי וידאו: מציירים מלמטה למעלה (הראשון במערך הוא העליון)
    const vTracks = proj().tracks.filter((tr) => tr.kind === 'video' && !tr.hidden);
    for (let i = vTracks.length - 1; i >= 0; i--) {
      this.renderTrack(ctx, vTracks[i], t, W, H);
    }
    this.needsRender = false;
  }

  renderTrack(ctx, track, t, W, H) {
    const c = clipAt(track, t);
    if (!c) return;

    const drawMain = (cx) => this.drawClip(cx, c, t, W, H);

    if (c.tin && t < c.start + c.tin.dur) {
      const prev = prevAdjacent(track, c);
      if (prev) {
        const p = clamp((t - c.start) / c.tin.dur, 0, 1);
        const drawPrev = (cx) => this.drawClip(cx, prev, t, W, H, true);
        drawTransition(ctx, c.tin.type, p, drawPrev, drawMain, W, H);
        return;
      }
    }
    drawMain(ctx);
  }

  /** ציור קליפ בודד. tail=true הוא הזנב של קליפ שכבר הסתיים וממשיך לרוץ בתוך מעבר.
   *  בזנב מדלגים על דעיכות הווידאו, אחרת הן היו מאפסות את התמונה בזמן המעבר. */
  drawClip(ctx, clip, t, W, H, tail = false) {
    const base = ctx.globalAlpha;
    let alpha = clip.opacity ?? 1;

    if (!tail) {
      const local = t - clip.start;
      if (clip.vFadeIn > 0 && local < clip.vFadeIn) alpha *= clamp(local / clip.vFadeIn, 0, 1);
      if (clip.vFadeOut > 0) {
        const toEnd = clipEnd(clip) - t;
        if (toEnd < clip.vFadeOut) alpha *= clamp(toEnd / clip.vFadeOut, 0, 1);
      }
    }
    if (alpha <= 0.001) return;

    ctx.save();
    ctx.globalAlpha = base * alpha;

    if (clip.kind === 'title') {
      drawTitle(ctx, clip, W, H);
      ctx.restore();
      return;
    }

    const p = this.player(clip);
    if (!p) { ctx.restore(); return; }
    const el = p.el;
    const media = mediaById(clip.mediaId);
    if (!media || media.type === 'audio') { ctx.restore(); return; }

    let sw = media.width || el?.videoWidth || 0;
    let sh = media.height || el?.videoHeight || 0;
    if (p.kind === 'image') { sw = el.naturalWidth; sh = el.naturalHeight; }
    if (!sw || !sh) { ctx.restore(); return; }
    if (p.kind !== 'image' && el.readyState < 2) { ctx.restore(); return; }

    const f = clip.filters || {};
    const parts = [];
    if (f.brightness != null && f.brightness !== 100) parts.push(`brightness(${f.brightness}%)`);
    if (f.contrast != null && f.contrast !== 100) parts.push(`contrast(${f.contrast}%)`);
    if (f.saturate != null && f.saturate !== 100) parts.push(`saturate(${f.saturate}%)`);
    if (f.blur) parts.push(`blur(${f.blur}px)`);
    if (parts.length) ctx.filter = parts.join(' ');

    const mode = clip.fit || 'contain';
    let s;
    if (mode === 'cover') s = Math.max(W / sw, H / sh);
    else if (mode === 'stretch') s = 1;
    else s = Math.min(W / sw, H / sh);

    const dw = mode === 'stretch' ? W : sw * s;
    const dh = mode === 'stretch' ? H : sh * s;

    ctx.translate(W / 2 + (clip.posX || 0) * W, H / 2 + (clip.posY || 0) * H);
    if (clip.rotation) ctx.rotate((clip.rotation * Math.PI) / 180);
    const sc = clip.scale || 1;
    ctx.scale(sc * (clip.flipH ? -1 : 1), sc);

    if (mode === 'cover') {
      ctx.beginPath();
      ctx.rect(-W / (2 * sc), -H / (2 * sc), W / sc, H / sc);
      ctx.clip();
    }
    try { ctx.drawImage(el, -dw / 2, -dh / 2, dw, dh); } catch {}
    ctx.restore();
  }

  /* ─────────── שליטה ─────────── */

  seek(t, hard = true) {
    this.syncTo(t, hard);
    this.render(t);
  }

  async play() {
    if (this.playing) return;
    const total = duration();
    if (total <= 0) return;
    if (state.playhead >= total - 1 / fps()) state.playhead = 0;
    this.initAudio();
    if (this.audio.ac.state === 'suspended') { try { await this.audio.ac.resume(); } catch {} }
    this.playing = true;
    state.playing = true;
    this.clockBase = state.playhead;
    this.clockStart = performance.now();
    this.onPlayState?.(true);
    this.syncTo(state.playhead, true);
    this.loop();
  }

  pause() {
    if (!this.playing) return;
    this.playing = false;
    state.playing = false;
    this.onPlayState?.(false);
    cancelAnimationFrame(this.rafId);
    for (const [, p] of this.players) {
      if (p.kind !== 'image') { try { p.el.pause(); } catch {} }
      if (p.gain) p.gain.gain.value = 0;
    }
    this.syncTo(state.playhead, true);
    this.render(state.playhead);
  }

  toggle() { this.playing ? this.pause() : this.play(); }

  setRate(r) {
    this.rate = r;
    if (this.playing) { this.clockBase = state.playhead; this.clockStart = performance.now(); }
  }

  loop = () => {
    if (!this.playing) return;
    const elapsed = ((performance.now() - this.clockStart) / 1000) * this.rate;
    let t = this.clockBase + elapsed;
    const total = duration();
    const limit = this.stopAt != null ? Math.min(this.stopAt, total) : total;

    if (t >= limit) {
      state.playhead = limit;
      this.pause();
      this.onTick?.(state.playhead);
      this.onEnd?.();
      return;
    }
    if (t < 0) { t = 0; this.clockBase = 0; this.clockStart = performance.now(); }

    state.playhead = t;
    this.syncTo(t);
    this.render(t);
    this.onTick?.(t);
    this.rafId = requestAnimationFrame(this.loop);
  };

  /** ממתין שכל הנגנים הפעילים יגיעו לפריים הנכון (לצילום/ייצוא מדויק) */
  async waitReady(t, timeout = 2500) {
    const act = this.activeAt(t);
    const waits = [];
    for (const { clip, srcTime } of act) {
      const p = this.player(clip);
      if (!p || p.kind === 'image') continue;
      const el = p.el;
      const want = clamp(srcTime, 0, Math.max(0, (el.duration || 1e9) - 0.03));
      if (Math.abs(el.currentTime - want) < 0.008 && el.readyState >= 2) continue;
      waits.push(new Promise((res) => {
        let done = false;
        const fin = () => { if (done) return; done = true; el.removeEventListener('seeked', fin); res(); };
        el.addEventListener('seeked', fin);
        setTimeout(fin, timeout);
        try { el.currentTime = want; } catch { fin(); }
      }));
    }
    if (waits.length) await Promise.all(waits);
  }
}

/* ─────────── עזרים ─────────── */

export function srcTimeOf(clip, t) {
  return clip.inPoint + (t - clip.start) * (clip.speed || 1);
}

/** עוצמת האודיו של קליפ בזמן t (כולל דעיכות והשתקת ערוץ).
 *  tail: הקליפ היוצא בתוך מעבר, שהדעיכה שלו מוחלפת בקרוספייד של המעבר. */
export function clipGain(clip, track, t, tail = false) {
  if (track.muted || clip.mute) return 0;
  let g = clip.volume ?? 1;
  if (!tail) {
    const local = t - clip.start;
    if (clip.aFadeIn > 0 && local < clip.aFadeIn) g *= clamp(local / clip.aFadeIn, 0, 1);
    if (clip.aFadeOut > 0) {
      const toEnd = clipEnd(clip) - t;
      if (toEnd < clip.aFadeOut) g *= clamp(toEnd / clip.aFadeOut, 0, 1);
    }
  }
  return clamp(g, 0, 4);
}

/* ─────────── כותרות ─────────── */

function drawTitle(ctx, clip, W, H) {
  const size = (clip.fontSize || 72) * (clip.scale || 1) * (W / 1920);
  const lines = String(clip.text || '').split('\n');
  const lh = size * 1.25;
  const cx = W / 2 + (clip.posX || 0) * W;
  let cy = H / 2 + (clip.posY || 0) * H;

  ctx.save();
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `600 ${size}px "Noto Sans Hebrew", "Segoe UI", Arial, sans-serif`;
  ctx.direction = 'rtl';

  if (clip.titleStyle === 'lower-third') {
    cy = H * 0.82 + (clip.posY || 0) * H;
    const wMax = Math.max(...lines.map((l) => ctx.measureText(l).width));
    ctx.fillStyle = 'rgba(0,0,0,.62)';
    ctx.fillRect(cx - wMax / 2 - size * 0.5, cy - lh * lines.length / 2 - size * 0.22,
      wMax + size, lh * lines.length + size * 0.44);
    ctx.fillStyle = clip.color || '#fff';
  } else if (clip.titleStyle === 'box') {
    const wMax = Math.max(...lines.map((l) => ctx.measureText(l).width));
    ctx.fillStyle = 'rgba(0,0,0,.6)';
    ctx.fillRect(cx - wMax / 2 - size * 0.4, cy - lh * lines.length / 2 - size * 0.2,
      wMax + size * 0.8, lh * lines.length + size * 0.4);
    ctx.fillStyle = clip.color || '#fff';
  } else if (clip.titleStyle === 'shadow') {
    ctx.shadowColor = 'rgba(0,0,0,.85)';
    ctx.shadowBlur = size * 0.16;
    ctx.shadowOffsetY = size * 0.045;
    ctx.fillStyle = clip.color || '#fff';
  } else {
    ctx.fillStyle = clip.color || '#fff';
  }

  const y0 = cy - (lh * (lines.length - 1)) / 2;
  lines.forEach((line, i) => ctx.fillText(line, cx, y0 + i * lh));
  ctx.restore();
}
