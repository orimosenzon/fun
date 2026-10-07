/* timeline.js: הטיימליין, מצויר כולו על קנבס אחד.
 *
 * שורות מלמעלה למטה: סרגל זמנים, תמונות ממוזערות, צורת גל, ופס סקירה של כל הסרטון.
 * הזמן תמיד זורם משמאל לימין, גם בממשק עברי, כמו בכל תוכנת עריכה.
 *
 * אינטראקציות:
 *   סרגל            גרירה = הזזת סמן הניגון
 *   פס הקטעים       לחיצה על קטע מהרשימה = בחירה שלו
 *   תמונות/גל       לחיצה = קפיצה; גרירה = קטע חדש; גרירת קצה = שינוי התחלה/סוף
 *                   (גרירה בתוך הקטע יוצרת קטע חדש ולא מזיזה אותו: כשהקטע הוא כל
 *                   הסרטון אין שום מקום "מחוצה לו", ובחירה מחדש היא הפעולה הנפוצה)
 *   פס סקירה        לחיצה/גרירה = מעבר לאזור אחר בסרטון
 *   גלגלת           גלילה הצידה; עם Ctrl = זום סביב העכבר
 */
window.C = window.C || {};

C.Timeline = function (canvas, wrap, cb) {
  const RULER = 24, LANE = 18, THUMB = 56, WAVE = 38, GAP = 8, OVER = 14;
  const H = RULER + LANE + THUMB + WAVE + GAP + OVER;
  const LANE_TOP = RULER, TRACK_TOP = RULER + LANE, TRACK_BOT = TRACK_TOP + THUMB + WAVE;
  const OVER_TOP = TRACK_BOT + GAP;
  const MAX_PPS = 400;
  const EDGE_PX = 8;
  const MIN_SEL = 0.1;

  const ctx = canvas.getContext('2d');
  let W = 0, dpr = 1;
  let dur = 0;
  let pps = 1, viewStart = 0;
  let playhead = 0;
  let selIn = 0, selOut = 0;
  let thumbs = [], thumbCount = 0, thumbAspect = 16 / 9;
  let peaks = null;
  let ranges = [];   // קטעים מהרשימה: { a, b, n, key, on, cur }
  let drag = null;
  let dirty = true;

  const css = getComputedStyle(document.documentElement);
  const col = (name, fb) => (css.getPropertyValue(name).trim() || fb);
  const C_ACCENT = col('--accent', '#f5b301');
  const C_PLAY = col('--playhead', '#ff4d5e');

  // ── גיאומטריה ──────────────────────────────────────────────────────────
  const minPps = () => (dur > 0 ? W / dur : 1);
  const x2t = (x) => viewStart + x / pps;
  const t2x = (t) => (t - viewStart) * pps;
  const viewLen = () => W / pps;

  function clampView() {
    pps = C.clamp(pps, minPps(), Math.max(minPps(), MAX_PPS));
    viewStart = C.clamp(viewStart, 0, Math.max(0, dur - viewLen()));
  }

  function resize() {
    dpr = window.devicePixelRatio || 1;
    const nw = wrap.clientWidth;
    if (!nw) return;
    const keepFit = W === 0 || Math.abs(pps - minPps()) < 1e-9;
    W = nw;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    if (keepFit) pps = minPps();
    clampView();
    dirty = true;
  }
  new ResizeObserver(resize).observe(wrap);

  // ── ציור ───────────────────────────────────────────────────────────────
  function niceStep() {
    const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1200, 1800];
    for (const s of steps) if (s * pps >= 80) return s;
    return 3600;
  }

  function label(t, step) {
    const h = Math.floor(t / 3600);
    const m = Math.floor((t - h * 3600) / 60);
    const s = t - h * 3600 - m * 60;
    const dec = step < 1 ? 1 : 0;
    const ss = s.toFixed(dec).padStart(dec ? 4 : 2, '0');
    return h ? `${h}:${String(m).padStart(2, '0')}:${ss}` : `${m}:${ss}`;
  }

  function nearestThumb(t) {
    if (!thumbCount) return null;
    const i0 = C.clamp(Math.floor((t / dur) * thumbCount), 0, thumbCount - 1);
    for (let d = 0; d < thumbCount; d++) {
      if (thumbs[i0 - d]) return thumbs[i0 - d];
      if (thumbs[i0 + d]) return thumbs[i0 + d];
    }
    return null;
  }

  function draw() {
    dirty = false;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#15171c';
    ctx.fillRect(0, 0, W, H);
    if (!dur) {
      ctx.fillStyle = '#20232a';
      ctx.fillRect(0, TRACK_TOP, W, TRACK_BOT - TRACK_TOP);
      return;
    }

    // סרגל
    ctx.fillStyle = '#1c1f26';
    ctx.fillRect(0, 0, W, RULER);
    const step = niceStep();
    const minor = step / (step >= 60 ? 6 : 5);
    ctx.strokeStyle = '#4a4f5c';
    ctx.fillStyle = '#9aa1b2';
    ctx.font = '11px system-ui, sans-serif';
    ctx.textBaseline = 'top';
    ctx.beginPath();
    for (let t = Math.floor(viewStart / minor) * minor; t <= viewStart + viewLen() + minor; t += minor) {
      const x = Math.round(t2x(t)) + 0.5;
      const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
      ctx.moveTo(x, RULER);
      ctx.lineTo(x, RULER - (major ? 10 : 5));
      if (major) ctx.fillText(label(t, step), x + 3, 3);
    }
    ctx.stroke();

    // תמונות ממוזערות
    const tw = THUMB * thumbAspect;
    ctx.fillStyle = '#20232a';
    ctx.fillRect(0, TRACK_TOP, W, THUMB);
    const first = Math.floor((viewStart * pps) / tw);
    const offset = viewStart * pps;
    for (let k = first; k * tw - offset < W; k++) {
      const x = k * tw - offset;
      const t = (k * tw + tw / 2) / pps;
      if (t > dur) break;
      const b = nearestThumb(t);
      if (b) ctx.drawImage(b, x, TRACK_TOP, tw, THUMB);
    }
    // קצה הסרטון (אחרי dur)
    const xEnd = t2x(dur);
    if (xEnd < W) { ctx.fillStyle = '#15171c'; ctx.fillRect(xEnd, TRACK_TOP, W - xEnd, TRACK_BOT - TRACK_TOP); }

    // צורת גל
    const wy = TRACK_TOP + THUMB;
    ctx.fillStyle = '#1a2a33';
    ctx.fillRect(0, wy, Math.min(W, xEnd), WAVE);
    if (peaks) {
      ctx.fillStyle = '#4fb3d9';
      const mid = wy + WAVE / 2;
      for (let x = 0; x < Math.min(W, xEnd); x++) {
        const a = Math.floor(x2t(x) * peaks.rate);
        const b = Math.max(a + 1, Math.floor(x2t(x + 1) * peaks.rate));
        let m = 0;
        for (let i = a; i < b && i < peaks.peaks.length; i++) if (peaks.peaks[i] > m) m = peaks.peaks[i];
        const h = Math.max(1, Math.min(1, Math.sqrt(m)) * (WAVE - 4));
        ctx.fillRect(x, mid - h / 2, 1, h);
      }
    }

    // הקטע הנבחר: מחשיכים את מה שמחוץ לו
    const xi = t2x(selIn), xo = t2x(selOut);
    ctx.fillStyle = 'rgba(8,9,12,0.62)';
    if (xi > 0) ctx.fillRect(0, TRACK_TOP, Math.min(W, xi), TRACK_BOT - TRACK_TOP);
    if (xo < W) ctx.fillRect(Math.max(0, xo), TRACK_TOP, W - Math.max(0, xo), TRACK_BOT - TRACK_TOP);
    if (selOut > selIn) {
      ctx.strokeStyle = C_ACCENT;
      ctx.lineWidth = 2;
      ctx.strokeRect(xi + 1, TRACK_TOP + 1, xo - xi - 2, TRACK_BOT - TRACK_TOP - 2);
      // ידיות
      ctx.fillStyle = C_ACCENT;
      for (const [x, dir] of [[xi, 1], [xo, -1]]) {
        const hx = dir > 0 ? x : x - 7;
        ctx.fillRect(hx, TRACK_TOP, 7, TRACK_BOT - TRACK_TOP);
        ctx.fillStyle = 'rgba(0,0,0,0.55)';
        const gy = (TRACK_TOP + TRACK_BOT) / 2;
        ctx.fillRect(hx + 2, gy - 8, 1, 16);
        ctx.fillRect(hx + 4, gy - 8, 1, 16);
        ctx.fillStyle = C_ACCENT;
      }
      // סימון הקטע גם על הסרגל
      ctx.fillStyle = C_ACCENT + '55';
      ctx.fillRect(xi, RULER - 4, xo - xi, 4);
    }

    // פס הקטעים מהרשימה
    ctx.fillStyle = '#181b21';
    ctx.fillRect(0, LANE_TOP, W, LANE);
    ctx.font = 'bold 11px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'center';
    for (const r of [...ranges].sort((p, q) => p.cur - q.cur)) {
      const x1 = t2x(r.a), x2 = t2x(r.b);
      if (x2 < 0 || x1 > W) continue;
      const w = Math.max(3, x2 - x1);
      ctx.fillStyle = r.cur ? C_ACCENT : r.on ? '#3d6fb8' : '#3a3f4a';
      ctx.fillRect(x1, LANE_TOP + 3, w, LANE - 6);
      if (r.cur) { ctx.fillStyle = C_ACCENT + '22'; ctx.fillRect(x1, TRACK_TOP, w, TRACK_BOT - TRACK_TOP); }
      if (w > 16) {
        ctx.fillStyle = r.cur ? '#1a1400' : '#e7eaf0';
        ctx.fillText(String(r.n), x1 + w / 2, LANE_TOP + LANE / 2 + 0.5);
      }
    }
    ctx.textAlign = 'start';

    // סמן ניגון
    const xp = Math.round(t2x(playhead)) + 0.5;
    if (xp >= -6 && xp <= W + 6) {
      ctx.strokeStyle = C_PLAY;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(xp, 4);
      ctx.lineTo(xp, TRACK_BOT);
      ctx.stroke();
      ctx.fillStyle = C_PLAY;
      ctx.beginPath();
      ctx.moveTo(xp - 6, 0); ctx.lineTo(xp + 6, 0); ctx.lineTo(xp, 9);
      ctx.closePath();
      ctx.fill();
    }

    // פס סקירה
    ctx.fillStyle = '#20232a';
    ctx.fillRect(0, OVER_TOP, W, OVER);
    const k = W / dur;
    ctx.fillStyle = C_ACCENT + 'aa';
    ctx.fillRect(selIn * k, OVER_TOP + 3, Math.max(2, (selOut - selIn) * k), OVER - 6);
    for (const r of ranges) {
      ctx.fillStyle = r.cur ? C_ACCENT : r.on ? '#5b8fd6' : '#4a4f5c';
      ctx.fillRect(r.a * k, OVER_TOP + 1, Math.max(2, (r.b - r.a) * k), 3);
    }
    ctx.strokeStyle = '#dfe3ec';
    ctx.lineWidth = 1;
    ctx.strokeRect(viewStart * k + 0.5, OVER_TOP + 0.5, Math.max(4, viewLen() * k) - 1, OVER - 1);
    ctx.fillStyle = C_PLAY;
    ctx.fillRect(playhead * k - 1, OVER_TOP, 2, OVER);
  }

  (function loop() {
    if (dirty) draw();
    requestAnimationFrame(loop);
  })();

  // ── אינטראקציה ─────────────────────────────────────────────────────────
  function zone(x, y) {
    if (y >= OVER_TOP) return { kind: 'over' };
    if (y < RULER) return { kind: 'ruler' };
    if (y < TRACK_TOP) {
      const t = x2t(x);
      const r = ranges.find((q) => t >= q.a && t <= q.b) || ranges.find((q) => Math.abs(t2x(q.a) - x) < 4 || Math.abs(t2x(q.b) - x) < 4);
      return r ? { kind: 'lane', key: r.key } : { kind: 'ruler' };
    }
    if (selOut > selIn) {
      const xi = t2x(selIn), xo = t2x(selOut);
      if (Math.abs(x - xi) <= EDGE_PX && x <= (xi + xo) / 2) return { kind: 'in' };
      if (Math.abs(x - xo) <= EDGE_PX) return { kind: 'out' };
    }
    return { kind: 'track' };
  }

  const CURSORS = { over: 'pointer', lane: 'pointer', ruler: 'col-resize', in: 'ew-resize', out: 'ew-resize', track: 'crosshair' };

  function setSel(a, b, fromUser) {
    a = C.clamp(a, 0, dur); b = C.clamp(b, 0, dur);
    if (b < a) [a, b] = [b, a];
    selIn = a; selOut = b;
    dirty = true;
    if (fromUser) cb.onSelect?.(selIn, selOut);
  }

  function centerOverview(x) {
    viewStart = (x / W) * dur - viewLen() / 2;
    clampView();
    dirty = true;
  }

  canvas.addEventListener('pointermove', (e) => {
    if (drag || !dur) return;
    const r = canvas.getBoundingClientRect();
    canvas.style.cursor = CURSORS[zone(e.clientX - r.left, e.clientY - r.top).kind];
  });

  canvas.addEventListener('pointerdown', (e) => {
    if (!dur || e.button !== 0) return;
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const z = zone(x, y);
    canvas.setPointerCapture(e.pointerId);
    drag = { kind: z.kind, x0: x, t0: x2t(x), in0: selIn, out0: selOut, moved: false };
    if (z.kind === 'lane') { drag = null; cb.onRange?.(z.key); return; }
    if (z.kind === 'over') centerOverview(x);
    else if (z.kind === 'ruler') { cb.onScrub?.(true); cb.onSeek?.(C.clamp(x2t(x), 0, dur)); }
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const t = C.clamp(x2t(x), 0, dur);
    if (Math.abs(x - drag.x0) > 3) drag.moved = true;
    switch (drag.kind) {
      case 'over': centerOverview(x); break;
      case 'ruler': cb.onSeek?.(t); break;
      case 'in': setSel(Math.min(t, selOut - MIN_SEL), selOut, true); cb.onSeek?.(selIn); break;
      case 'out': setSel(selIn, Math.max(t, selIn + MIN_SEL), true); cb.onSeek?.(selOut); break;
      case 'track':
        if (drag.moved) { setSel(drag.t0, t, true); cb.onSeek?.(t); }
        break;
    }
    edgeScroll(x);
  });

  function endDrag(e) {
    if (!drag) return;
    const d = drag;
    drag = null;
    if (d.kind === 'ruler') cb.onScrub?.(false);
    if (d.kind === 'track' && !d.moved) cb.onSeek?.(C.clamp(d.t0, 0, dur));
    if (d.kind === 'track' && d.moved && selOut - selIn < MIN_SEL) setSel(d.in0, d.out0, true);
    canvas.style.cursor = '';
  }
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);

  // גרירה ליד הקצה גוללת את התצוגה
  function edgeScroll(x) {
    const m = 30;
    if (x < m) viewStart -= (m - x) / pps / 4;
    else if (x > W - m) viewStart += (x - (W - m)) / pps / 4;
    else return;
    clampView();
    dirty = true;
  }

  canvas.addEventListener('wheel', (e) => {
    if (!dur) return;
    e.preventDefault();
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    if (e.ctrlKey || e.metaKey) {
      const t = x2t(x);
      pps *= Math.exp(-e.deltaY * 0.0025);
      clampView();
      viewStart = t - x / pps;
      clampView();
    } else {
      const d = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
      viewStart += d / pps;
      clampView();
    }
    dirty = true;
    cb.onZoom?.();
  }, { passive: false });

  // ── API ────────────────────────────────────────────────────────────────
  return {
    setDuration(d, aspect) {
      dur = d; thumbs = []; thumbCount = 0; peaks = null;
      thumbAspect = aspect || 16 / 9;
      playhead = 0;
      resize();
      pps = minPps(); viewStart = 0;
      dirty = true;
    },
    setThumbCount(n) { thumbCount = n; thumbs = new Array(n); dirty = true; },
    setThumb(i, bmp) { thumbs[i] = bmp; dirty = true; },
    setPeaks(p) { peaks = p; dirty = true; },
    setRanges(list) { ranges = list || []; dirty = true; },
    setSelection(a, b) { setSel(a, b, false); },
    setPlayhead(t, follow) {
      playhead = t;
      if (follow && dur) {
        const x = t2x(t);
        if (x > W - 20 || x < 0) { viewStart = t - viewLen() * 0.1; clampView(); }
      }
      dirty = true;
    },
    zoomFit() { pps = minPps(); viewStart = 0; dirty = true; cb.onZoom?.(); },
    zoomTo(a, b) {
      const pad = (b - a) * 0.08 + 0.2;
      pps = W / (b - a + 2 * pad); clampView();
      viewStart = a - pad; clampView();
      dirty = true; cb.onZoom?.();
    },
    /** זום כמספר 0..1 (לוגריתמי), בשביל המחוון */
    getZoom01() {
      const lo = minPps(), hi = Math.max(lo * 1.0001, MAX_PPS);
      return C.clamp(Math.log(pps / lo) / Math.log(hi / lo), 0, 1);
    },
    setZoom01(z) {
      const lo = minPps(), hi = Math.max(lo, MAX_PPS);
      const center = viewStart + viewLen() / 2;
      const c = playhead >= viewStart && playhead <= viewStart + viewLen() ? playhead : center;
      const frac = (c - viewStart) / viewLen();
      pps = lo * Math.pow(hi / lo, z);
      clampView();
      viewStart = c - frac * viewLen();
      clampView();
      dirty = true;
    },
    redraw() { dirty = true; },
  };
};
