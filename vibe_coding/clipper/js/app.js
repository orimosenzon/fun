/* app.js: מחבר את הכול: טעינת סרטון, תצוגה מקדימה, טיימליין, מיתוג וייצוא. */
(() => {
  const { $, t, clamp, fmtTime, parseTime, toast } = C;

  const src = $('srcVideo');
  const introV = $('introVideo');
  const outroV = $('outroVideo');
  const preview = $('preview');
  const pctx = preview.getContext('2d', { alpha: false });

  // ── מצב ────────────────────────────────────────────────────────────────
  const S = {
    file: null,          // הסרטון הפתוח
    dur: 0,
    selIn: 0, selOut: 0,
    brand: { logo: null, intro: null, outro: null },   // Blob לכל אחד
    brandFrom: {},       // 'folder' אם הגיע מתיקיית brand/ ולא מהמשתמש
    logoImg: null,
    set: {               // הגדרות שנשמרות בין ביקורים
      platform: 'ig_reels', format: '9:16', guides: true, quality: '1080', fades: true,
      zoom: 1, px: 0.5, py: 0.5, fill: 'blur', fillColor: '#000000',      // מסגור: אחד לכל הסרטון
      textOn: false, textTop: '', textBottom: '', textSize: 7, textColor: '#ffffff',
      batchText: '',
      corner: 'tr', logoSize: 18, logoOpacity: 0.9, logoAll: false,
      volume: 1, muted: false,
    },
    mode: 'idle',        // idle | play | sequence | export
    seq: null,
    scrubbing: false,
  };
  window.__clipper = S;  // לבדיקות אוטומטיות

  // ── עזרים ──────────────────────────────────────────────────────────────
  const plat = () => C.platform(S.set.platform);
  const safe = () => plat()?.safe || C.DEFAULT_SAFE;
  const look = () => ({
    zoom: S.set.zoom, px: S.set.px, py: S.set.py, fill: S.set.fill, fillColor: S.set.fillColor,
    text: S.set.textOn ? { top: S.set.textTop, bottom: S.set.textBottom, size: S.set.textSize, color: S.set.textColor } : null,
    logo: S.logoImg, safe: safe(),
    corner: S.set.corner, logoSize: S.set.logoSize, logoOpacity: S.set.logoOpacity,
  });
  const platName = () => (plat() ? plat().name[C.lang] : t('platCustom'));

  function outSize(short) {
    return C.compose.outputSize(S.set.format, src.videoWidth, src.videoHeight, short ?? +S.set.quality);
  }

  function saveSettings() { C.store.set('settings', S.set); }

  /** התצוגה המקדימה בגודל מוקטן (חצי מהפלט) כדי לחסוך עבודה בניגון */
  function sizePreview() {
    const { W, H } = outSize(540);
    if (preview.width !== W || preview.height !== H) { preview.width = W; preview.height = H; }
    const full = outSize();
    $('formatHint').textContent = t('fmtHint', full.W, full.H, t('fmtFor')[S.set.format]);
    $('rngFrameZoom').max = Math.max(4, Math.ceil(coverZoom() * 1.5));
    $('platformNote').textContent = plat() ? plat().note[C.lang] : '';
    // אינסטגרם ורוב הרשתות דורשות AAC. כרום על לינוקס יודע רק Opus.
    const fmt = C.sequencer.pickFormat();
    const noAac = plat()?.needsAac && fmt && !fmt.mime.includes('mp4a');
    $('codecWarn').hidden = !noAac;
    if (noAac) $('codecWarn').textContent = t('noAacWarn', platName());
  }

  /** האורך הכולל (פתיחה + קטע + סיום) */
  const brandSeconds = () => (S.brand.intro && introV.duration ? introV.duration : 0) + (S.brand.outro && outroV.duration ? outroV.duration : 0);
  function totalSeconds() { return Math.max(0, S.selOut - S.selIn) + brandSeconds(); }

  const tooLong = () => !!(S.file && plat()?.maxSec && totalSeconds() > plat().maxSec + 0.05);

  /** סימון האזורים שהאפליקציה של הרשת מכסה. רק בתצוגה, אף פעם לא בקובץ. */
  function drawGuides(ctx, W, H) {
    const p = plat();
    if (!S.set.guides || !p) return;
    const z = p.safe;
    ctx.save();
    ctx.fillStyle = 'rgba(255, 77, 94, 0.18)';
    ctx.fillRect(0, 0, W, z.top * H);
    ctx.fillRect(0, H - z.bottom * H, W, z.bottom * H);
    ctx.fillRect(0, z.top * H, z.left * W, H * (1 - z.top - z.bottom));
    ctx.fillRect(W - z.right * W, z.top * H, z.right * W, H * (1 - z.top - z.bottom));
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.setLineDash([6, 5]);
    ctx.lineWidth = 1;
    ctx.strokeRect(z.left * W + 0.5, z.top * H + 0.5, W * (1 - z.left - z.right) - 1, H * (1 - z.top - z.bottom) - 1);
    ctx.restore();
  }

  function renderPlatforms() {
    const box = $('platforms');
    box.innerHTML = '';
    for (const p of C.PLATFORMS) {
      const b = document.createElement('button');
      b.dataset.id = p.id;
      const m = p.maxSec ? ` · ${t('maxLen', p.maxSec)}` : '';
      b.innerHTML = `<span class="pi">${p.icon}</span><span></span><span class="pm">${p.format}${m}</span>`;
      b.children[1].textContent = p.name[C.lang];
      b.classList.toggle('on', p.id === S.set.platform);
      box.appendChild(b);
    }
  }

  function drawPreview() {
    if (S.mode === 'sequence' || S.mode === 'export') { drawGhost(); return; }
    C.compose.draw(pctx, S.file ? src : null, preview.width, preview.height, { ...look(), logoOn: !!S.logoImg, fade: 0 });
    drawGuides(pctx, preview.width, preview.height);
    setBadge(S.file ? 'stageMain' : '');
    drawGhost();
  }

  // ── מסגור: גרירה וזום ישירות על התצוגה ───────────────────────────────────
  const ghost = $('ghost');
  const gctx = ghost.getContext('2d');
  const coverZoom = () => (S.file ? C.compose.coverZoom(src, preview.width, preview.height) : 1);
  const ZMIN = 0.5;
  const zmax = () => +$('rngFrameZoom').max;

  /** איפה הפריים של התוצאה מוצג בתוך הבמה (הקנבס מוצג ב-object-fit: contain) */
  function shownRect() {
    const b = preview.getBoundingClientRect();
    const s = Math.min(b.width / preview.width, b.height / preview.height);
    return { s, left: b.left + (b.width - preview.width * s) / 2, top: b.top + (b.height - preview.height * s) / 2, b };
  }

  /** החלקים שנחתכים מחוץ למסגרת, בשקיפות, כדי שיהיה ברור מה נשאר בחוץ */
  function drawGhost() {
    const dpr = window.devicePixelRatio || 1;
    const { s, left, top, b } = shownRect();
    const gw = Math.round(b.width * dpr), gh = Math.round(b.height * dpr);
    if (ghost.width !== gw || ghost.height !== gh) { ghost.width = gw; ghost.height = gh; }
    gctx.setTransform(1, 0, 0, 1, 0, 0);
    gctx.clearRect(0, 0, gw, gh);
    if (!S.file || S.mode === 'sequence' || S.mode === 'export') return;
    gctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const ox = left - b.left, oy = top - b.top;
    const r = C.compose.place(src, preview.width, preview.height, look());
    gctx.globalAlpha = 0.28;
    gctx.drawImage(src, ox + r.x * s, oy + r.y * s, r.w * s, r.h * s);
    gctx.globalAlpha = 1;
    gctx.strokeStyle = 'rgba(255,255,255,0.5)';
    gctx.lineWidth = 1;
    gctx.strokeRect(ox - 0.5, oy - 0.5, preview.width * s + 1, preview.height * s + 1);
  }

  /** px/py חדשים כך שהנקודה (X, Y) בפריים תישאר מתחת לאצבע אחרי שינוי זום */
  function zoomAround(z, X, Y) {
    const W = preview.width, H = preview.height;
    const r0 = C.compose.place(src, W, H, look());
    const u = (X - r0.x) / r0.w, v = (Y - r0.y) / r0.h;
    S.set.zoom = clamp(z, ZMIN, zmax());
    const r1 = C.compose.place(src, W, H, look());
    const sx = W - r1.w, sy = H - r1.h;
    S.set.px = Math.abs(sx) > 0.5 ? clamp((X - u * r1.w) / sx, 0, 1) : 0.5;
    S.set.py = Math.abs(sy) > 0.5 ? clamp((Y - v * r1.h) / sy, 0, 1) : 0.5;
  }

  let saveTimer = null;
  function frameChanged() {
    $('rngFrameZoom').value = S.set.zoom;
    $('zoomVal').textContent = `${Math.round(S.set.zoom * 100)}%`;
    if (S.mode !== 'play') drawPreview();
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveSettings, 400);
  }

  const pointers = new Map();
  let drag = null;
  const canFrame = () => S.file && S.mode !== 'sequence' && S.mode !== 'export';
  const toFrame = (e) => { const { s, left, top } = shownRect(); return [(e.clientX - left) / s, (e.clientY - top) / s]; };

  preview.addEventListener('pointerdown', (e) => {
    if (!canFrame()) return;
    preview.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, toFrame(e));
    const r = C.compose.place(src, preview.width, preview.height, look());
    drag = { start: toFrame(e), px: S.set.px, py: S.set.py, sx: preview.width - r.w, sy: preview.height - r.h, pinch: null };
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      drag.pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), zoom: S.set.zoom };
    }
    preview.classList.add('grabbing');
  });
  preview.addEventListener('pointermove', (e) => {
    if (!drag || !pointers.has(e.pointerId)) return;
    const p = toFrame(e);
    pointers.set(e.pointerId, p);
    if (drag.pinch && pointers.size === 2) {
      const [a, b] = [...pointers.values()];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (drag.pinch.d > 10) zoomAround(drag.pinch.zoom * (d / drag.pinch.d), (a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
    } else if (!drag.pinch) {
      // גרירה ימינה מזיזה את התמונה ימינה, בין אם היא קטנה מהמסגרת ובין אם גדולה ממנה
      if (Math.abs(drag.sx) > 0.5) S.set.px = clamp(drag.px + (p[0] - drag.start[0]) / drag.sx, 0, 1);
      if (Math.abs(drag.sy) > 0.5) S.set.py = clamp(drag.py + (p[1] - drag.start[1]) / drag.sy, 0, 1);
    }
    frameChanged();
  });
  const endDrag = (e) => {
    pointers.delete(e.pointerId);
    if (pointers.size === 0) { drag = null; preview.classList.remove('grabbing'); }
  };
  preview.addEventListener('pointerup', endDrag);
  preview.addEventListener('pointercancel', endDrag);
  preview.addEventListener('wheel', (e) => {
    if (!canFrame()) return;
    e.preventDefault();
    // צביטה בטאצ'פד מגיעה ככגלגלת עם ctrlKey וצעדים קטנים
    const k = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
    const [X, Y] = toFrame(e);
    zoomAround(S.set.zoom * k, X, Y);
    frameChanged();
  }, { passive: false });
  preview.addEventListener('dblclick', () => { if (canFrame()) { S.set.zoom = 1; S.set.px = S.set.py = 0.5; frameChanged(); } });
  window.addEventListener('resize', () => drawGhost());

  function setBadge(key) {
    const b = $('stageBadge');
    b.textContent = key ? t(key) : '';
    b.hidden = !key;
  }

  function updateTimes() {
    $('timecode').textContent = `${fmtTime(src.currentTime || 0)} / ${fmtTime(S.dur)}`;
    if (document.activeElement !== $('inIn')) $('inIn').value = fmtTime(S.selIn);
    if (document.activeElement !== $('inOut')) $('inOut').value = fmtTime(S.selOut);
    $('selLen').textContent = fmtTime(Math.max(0, S.selOut - S.selIn));
    $('totalLen').textContent = fmtTime(totalSeconds());
    const long = tooLong();
    $('totalLen').classList.toggle('bad', long);
    $('lenWarn').hidden = !long;
    if (long) $('lenWarn').textContent = t('tooLong', fmtTime(totalSeconds()), fmtTime(plat().maxSec), platName());
  }

  function setSelection(a, b, fromTimeline) {
    a = clamp(a, 0, S.dur); b = clamp(b, 0, S.dur);
    if (b < a) [a, b] = [b, a];
    S.selIn = a; S.selOut = b;
    if (!fromTimeline) tl.setSelection(a, b);
    updateTimes();
  }

  // ── טיימליין ───────────────────────────────────────────────────────────
  const tl = C.Timeline($('timeline'), $('tlWrap'), {
    onSeek: (tt) => { stopAll(); src.currentTime = tt; tl.setPlayhead(tt); updateTimes(); },
    onSelect: (a, b) => setSelection(a, b, true),
    onScrub: (on) => { S.scrubbing = on; },
    onZoom: () => { $('rngZoom').value = tl.getZoom01(); },
  });

  src.addEventListener('seeked', () => { drawPreview(); tl.setPlayhead(src.currentTime); updateTimes(); });
  src.addEventListener('timeupdate', updateTimes);

  // ── טעינת סרטון ────────────────────────────────────────────────────────
  async function openFile(file) {
    if (!file) return;
    if (!file.type.startsWith('video/') && !/\.(mp4|mov|m4v|webm|mkv|avi)$/i.test(file.name)) { toast(t('badFile')); return; }
    stopAll();
    toast(t('loading'), 1500);
    try {
      await C.loadVideo(src, file);
    } catch (e) {
      toast(t('loadFailed'), 6000);
      return;
    }
    S.file = file;
    S.dur = src.duration;
    $('fileName').textContent = file.name;
    $('dropzone').hidden = true;
    document.body.classList.add('has-video');
    sizePreview();
    tl.setDuration(S.dur, src.videoWidth / src.videoHeight);
    $('rngZoom').value = 0;
    // ברירת מחדל: כל הסרטון, עד דקה
    setSelection(0, Math.min(S.dur, 60));
    drawPreview();
    updateTimes();
    B.status.clear();
    renderBatch();

    const n = clamp(Math.round(S.dur / 2), 24, 160);
    tl.setThumbCount(n);
    C.media.buildThumbs(file, S.dur, n, 56 * (window.devicePixelRatio || 1), (i, _t, bmp) => tl.setThumb(i, bmp));
    C.media.buildPeaks(file).then((p) => { if (S.file === file && p) tl.setPeaks(p); });
  }

  $('fileInput').addEventListener('change', (e) => { openFile(e.target.files[0]); e.target.value = ''; });
  $('btnOpen').addEventListener('click', () => $('fileInput').click());
  $('dropzone').addEventListener('click', () => $('fileInput').click());
  document.addEventListener('dragover', (e) => { e.preventDefault(); document.body.classList.add('dragging'); });
  document.addEventListener('dragleave', (e) => { if (!e.relatedTarget) document.body.classList.remove('dragging'); });
  document.addEventListener('drop', (e) => {
    e.preventDefault();
    document.body.classList.remove('dragging');
    const f = e.dataTransfer.files[0];
    if (f) openFile(f);
  });

  // ── ניגון חופשי ────────────────────────────────────────────────────────
  let playStopAt = null;   // בניגון קטע: עוצרים בסוף הקטע

  function playLoop() {
    if (S.mode !== 'play') return;
    drawPreview();
    tl.setPlayhead(src.currentTime, true);
    updateTimes();
    if (playStopAt != null && src.currentTime >= playStopAt) {
      stopAll();
      src.currentTime = playStopAt;
      return;
    }
    if (src.ended) { stopAll(); return; }
    requestAnimationFrame(playLoop);
  }

  function play(from, to) {
    if (!S.file) { toast(t('noVideo')); return; }
    stopAll();
    const g = C.audio.gainFor(src);
    g.gain.cancelScheduledValues(0);
    g.gain.value = 1;
    if (from != null) src.currentTime = from;
    playStopAt = to ?? null;
    S.mode = 'play';
    $('btnPlay').textContent = '⏸';
    src.play().then(() => requestAnimationFrame(playLoop)).catch((e) => { console.warn(e); stopAll(); });
  }

  function stopAll() {
    if (S.seq) { S.seq.abort(); S.seq = null; }
    if (S.mode === 'play') src.pause();
    if (S.mode !== 'export') S.mode = 'idle';
    playStopAt = null;
    $('btnPlay').textContent = '▶';
  }

  $('btnPlay').addEventListener('click', () => (S.mode === 'play' || S.mode === 'sequence' ? stopAll() : play()));
  $('btnBack').addEventListener('click', () => { stopAll(); src.currentTime = S.selIn; });
  const stepFrame = (d) => { stopAll(); src.currentTime = clamp(src.currentTime + d, 0, S.dur); };
  $('btnFrameBack').addEventListener('click', () => stepFrame(-1 / 30));
  $('btnFrameFwd').addEventListener('click', () => stepFrame(1 / 30));
  $('btnJumpBack').addEventListener('click', () => stepFrame(-5));
  $('btnJumpFwd').addEventListener('click', () => stepFrame(5));

  // עוצמה: נשלטת בגיין הראשי של WebAudio, כי הקול של כל הסרטונים עובר דרכו
  function applyVolume() {
    const v = S.set.muted ? 0 : S.set.volume;
    C.audio.setVolume(v);
    $('btnMute').textContent = v === 0 ? '🔇' : '🔊';
    $('btnMute').classList.toggle('off', v === 0);
    $('rngVolume').value = S.set.volume;
  }
  $('btnMute').addEventListener('click', () => { S.set.muted = !S.set.muted; applyVolume(); saveSettings(); });
  $('rngVolume').addEventListener('input', (e) => { S.set.volume = +e.target.value; S.set.muted = false; applyVolume(); saveSettings(); });
  $('btnSetIn').addEventListener('click', () => setSelection(src.currentTime, Math.max(src.currentTime + 0.1, S.selOut)));
  $('btnSetOut').addEventListener('click', () => setSelection(Math.min(S.selIn, src.currentTime - 0.1), src.currentTime));
  $('btnPlaySel').addEventListener('click', () => play(S.selIn, S.selOut));

  // ── רצף מלא (פתיחה + קטע + סיום) ────────────────────────────────────────
  /** ranges: טווחי הקטע בסרטון המקור. כמה טווחים = כמה חלקים ברצף, עם דהייה ביניהם. */
  function buildParts(ranges = [[S.selIn, S.selOut]]) {
    const parts = [];
    // פתיחה וסיום אף פעם לא נחתכים: יש בהם טקסט וכתובות שחייבים להיראות במלואם
    const brandLook = { zoom: 1, px: 0.5, py: 0.5, text: null };
    if (S.brand.intro && introV.duration) parts.push({ video: introV, from: 0, to: introV.duration, logo: S.set.logoAll && !!S.logoImg, look: brandLook, label: t('stageIntro') });
    ranges.forEach(([from, to], k) => parts.push({
      video: src, from, to, logo: !!S.logoImg,
      label: ranges.length > 1 ? `${t('stageMain')} ${k + 1}/${ranges.length}` : t('stageMain'),
    }));
    if (S.brand.outro && outroV.duration) parts.push({ video: outroV, from: 0, to: outroV.duration, logo: S.set.logoAll && !!S.logoImg, look: brandLook, label: t('stageOutro') });
    return parts;
  }

  function checkReady() {
    if (!S.file) { toast(t('noVideo')); return false; }
    if (S.selOut - S.selIn < 0.2) { toast(t('selTooShort')); return false; }
    return true;
  }

  async function playSequence(parts) {
    stopAll();
    S.mode = 'sequence';
    drawGhost();
    $('btnPlay').textContent = '⏸';
    S.seq = C.sequencer.run({
      parts, canvas: preview, look: look(), fades: S.set.fades, record: null,
      onProgress: (_f, label) => { if (label) $('stageBadge').textContent = label; $('stageBadge').hidden = !label; tl.setPlayhead(src.currentTime); },
    });
    try { await S.seq.done; } catch (e) { console.warn(e); }
    if (S.mode === 'sequence') S.mode = 'idle';
    S.seq = null;
    $('btnPlay').textContent = '▶';
    drawPreview();
  }

  $('btnPlayAll').addEventListener('click', () => {
    if (S.mode === 'sequence') { stopAll(); drawPreview(); return; }
    if (!checkReady()) return;
    playSequence(buildParts());
  });

  // ── ייצוא ──────────────────────────────────────────────────────────────
  /** מקליט רצף אחד לקובץ. ההתקדמות מדווחת דרך onProgress(שבר, תווית). */
  async function makeVideo(parts, onProgress) {
    const fmt = C.sequencer.pickFormat();
    const { W, H } = outSize();
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    // ~0.15 ביט לפיקסל בשנייה ב-30fps: 1080×1920 ≈ 9.3Mbps. מספיק בשביל לשרוד את הדחיסה החוזרת של הרשתות.
    const bitrate = Math.round(W * H * 30 * 0.15);
    const t0 = performance.now();
    const seq = S.seq = C.sequencer.run({
      parts, canvas: cv, look: look(), fades: S.set.fades,
      record: { mime: fmt.mime, bitrate },
      onFrame: (c) => pctx.drawImage(c, 0, 0, preview.width, preview.height),
      onProgress: (f, label) => {
        onProgress(f, label);
        if (label) { $('stageBadge').textContent = label; $('stageBadge').hidden = false; }
      },
    });
    let blob = null, err = null;
    try { blob = await seq.done; } catch (e) { err = e; console.error(e); }
    S.seq = null;
    const seconds = parts.reduce((s, p) => s + p.to - p.from, 0);
    return { blob, err, fmt, W, H, seconds, took: (performance.now() - t0) / 1000, stats: seq.stats };
  }

  function beginExport() {
    if (S.mode === 'export') return false;
    if (!C.sequencer.pickFormat()) { toast(t('noRecorder'), 6000); return false; }
    stopAll();
    S.mode = 'export';
    document.body.classList.add('exporting');
    drawGhost();
    return true;
  }
  function endExport() {
    S.mode = 'idle';
    document.body.classList.remove('exporting');
    drawPreview();
  }

  $('btnExport').addEventListener('click', async () => {
    if (S.mode === 'export') return;
    if (!checkReady()) return;
    if (tooLong() && !confirm(t('confirmTooLong'))) return;
    if (!beginExport()) return;
    $('result').hidden = true;
    $('progress').hidden = false;
    $('progFill').style.width = '0%';

    const parts = buildParts();
    const r = await makeVideo(parts, (f, label) => {
      $('progFill').style.width = `${(f * 100).toFixed(1)}%`;
      $('progText').textContent = label ? t('exporting', Math.round(f * 100), label) : t('finishing');
    });
    $('progress').hidden = true;
    endExport();

    if (r.err) { toast(t('exportFailed', r.err.message || r.err), 8000); return; }
    if (!r.blob) { toast(t('cancelled')); return; }

    const url = URL.createObjectURL(r.blob);
    const rv = $('resultVideo');
    if (rv._url) URL.revokeObjectURL(rv._url);
    rv._url = url;
    rv.src = url;
    const name = `${C.baseName(S.file.name)}_${plat() ? plat().id : S.set.format.replace(':', 'x')}.${r.fmt.ext}`;
    const a = $('btnDownload');
    a.href = url;
    a.download = name;
    $('resultInfo').textContent = t('resultInfo', C.fmtBytes(r.blob.size), fmtTime(r.seconds), `${r.fmt.label} · ${r.W}×${r.H}`)
      + (r.fmt.warn ? `\n${t(r.fmt.warn)}` : '');
    $('result').hidden = false;
    S.lastExport = { blob: r.blob, name, mime: r.fmt.mime, W: r.W, H: r.H, seconds: r.took, stats: r.stats };
    toast(t('done'));
  });

  // ── רשימת קטעים: הרבה סרטונים מאותו מקור בבת אחת ─────────────────────────
  const B = { items: [], off: new Set(), cur: null, status: new Map(), cancel: false, sim: new Map() };
  window.__batch = B;
  const short = (x) => fmtTime(x).replace(/\.00$/, '');
  const stamp = (x) => { x = Math.floor(x); return `${Math.floor(x / 3600)}-${String(Math.floor(x / 60) % 60).padStart(2, '0')}-${String(x % 60).padStart(2, '0')}`; };
  const okItem = (it) => it.n && !it.errors.length && !(S.file && it.ranges.some(([, b]) => b > S.dur + 0.05));
  const chosen = () => B.items.filter((it) => okItem(it) && !B.off.has(it.text));

  function renderBatch() {
    B.items = C.batch.parse(S.set.batchText);
    B.sim = C.batch.similarTo(B.items);
    const box = $('batchList');
    box.innerHTML = '';
    const max = plat()?.maxSec;
    for (const it of B.items) {
      const row = document.createElement('div');
      row.className = 'bitem';
      row.classList.toggle('cur', B.cur === it.text);
      const msgs = it.errors.map((k) => t(k));
      const warns = [];
      if (S.file && it.ranges.some(([, b]) => b > S.dur + 0.05)) msgs.push(t('bPastEnd', short(S.dur)));
      const len = C.batch.length(it) + brandSeconds();
      if (it.n && max && len > max + 0.05) warns.push(t('bTooLong', short(len), short(max)));
      if (B.sim.has(it)) warns.push(t('bSimilar', B.sim.get(it)));
      const bad = !okItem(it);
      row.classList.toggle('bad', bad);
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = !bad && !B.off.has(it.text);
      cb.disabled = bad;
      cb.addEventListener('change', () => { if (cb.checked) B.off.delete(it.text); else B.off.add(it.text); updateBatchSum(); });
      cb.addEventListener('click', (e) => e.stopPropagation());
      const n = document.createElement('span'); n.className = 'bn'; n.textContent = it.n || '!';
      const r = document.createElement('span'); r.className = 'br';
      r.textContent = it.ranges.length ? it.ranges.map(([a, b]) => `${short(a)}–${short(b)}`).join(' + ') : it.text;
      r.title = `${t('batchLine', it.line)}: ${it.text}`;
      const l = document.createElement('span'); l.className = 'bl'; l.textContent = it.n ? short(C.batch.length(it)) : '';
      const pb = document.createElement('button'); pb.className = 'bplay'; pb.textContent = '▶'; pb.title = t('batchPlay');
      pb.hidden = bad || !S.file;
      pb.addEventListener('click', (e) => { e.stopPropagation(); if (S.mode !== 'export') { B.cur = it.text; markCur(); playSequence(buildParts(it.ranges)); } });
      row.append(cb, n, r, l, pb);
      if (msgs.length || warns.length) {
        const m = document.createElement('div'); m.className = 'bmsg'; m.textContent = [...msgs, ...warns].join(' · ');
        row.appendChild(m);
      }
      const st = B.status.get(it.text);
      if (st) row.appendChild(st);
      if (!bad) {
        row.title = t('batchShow');
        row.addEventListener('click', () => showItem(it));
      }
      it.row = row;
      box.appendChild(row);
    }
    updateBatchSum();
  }

  function markCur() { for (const it of B.items) it.row?.classList.toggle('cur', B.cur === it.text); }

  function showItem(it) {
    if (!S.file || S.mode === 'export') return;
    stopAll();
    B.cur = it.text; markCur();
    // הטווח הראשון נבחר על הטיימליין; בשורה עם And רואים את כולם ב-▶
    const [a, b] = it.ranges[0];
    setSelection(a, b);
    src.currentTime = a;
    const pad = Math.max(2, (b - a) * 0.15);
    tl.zoomTo(Math.max(0, a - pad), Math.min(S.dur, b + pad));
    $('rngZoom').value = tl.getZoom01();
  }

  function updateBatchSum() {
    const c = chosen();
    const any = B.items.length > 0;
    $('batchSum').hidden = !any;
    $('batchSum').textContent = any ? t('batchSum', c.length, short(c.reduce((s, it) => s + C.batch.length(it) + brandSeconds(), 0)))
      + (B.items.some((it) => !okItem(it)) ? `\n${t('batchFixErrors')}` : '') : '';
    $('btnBatchRun').hidden = !any;
    $('btnBatchRun').textContent = c.length ? t('batchRunN', c.length) : t('batchRun');
    $('btnBatchRun').disabled = !c.length;
  }

  let batchTimer = null;
  // Pages מפריד שורות בתוך פסקה בתו U+2028, ותיבת טקסט מציגה אותו כרווח. ממירים לשורה רגילה.
  const normLines = (x) => x.replace(/\r\n?|\u2028|\u2029/g, '\n');
  $('inBatch').addEventListener('input', (e) => {
    if (/[\r\u2028\u2029]/.test(e.target.value)) e.target.value = normLines(e.target.value);
    S.set.batchText = e.target.value;
    clearTimeout(batchTimer);
    batchTimer = setTimeout(() => { renderBatch(); saveSettings(); }, 150);
  });
  $('btnBatchFile').addEventListener('click', () => $('batchInput').click());
  $('btnBatchClear').addEventListener('click', () => {
    S.set.batchText = ''; $('inBatch').value = ''; B.status.clear(); B.off.clear(); B.cur = null;
    renderBatch(); saveSettings();
  });
  $('batchInput').addEventListener('change', async (e) => {
    const f = e.target.files[0];
    e.target.value = '';
    if (!f) return;
    if (/\.(pages|docx?|rtf)$/i.test(f.name)) { toast(t('batchPagesFile'), 7000); return; }
    S.set.batchText = normLines(await f.text());
    $('inBatch').value = S.set.batchText;
    B.status.clear();
    renderBatch(); saveSettings();
    toast(t('batchFileRead', f.name, B.items.filter((it) => it.n).length));
  });

  function statusLine(html) {
    const d = document.createElement('div');
    d.className = 'bstat';
    if (typeof html === 'string') d.textContent = html; else d.append(...html);
    return d;
  }

  $('btnBatchRun').addEventListener('click', async () => {
    if (!S.file) { toast(t('noVideo')); return; }
    const list = chosen();
    if (!list.length) { toast(t('batchNoneSelected')); return; }
    const max = plat()?.maxSec;
    if (max && list.some((it) => C.batch.length(it) + brandSeconds() > max + 0.05) && !confirm(t('batchConfirmLong'))) return;
    // בכרום ובאדג' שומרים ישר לתיקייה, וכל קובץ משתחרר מהזיכרון מיד. בשאר הדפדפנים: הורדה רגילה לכל קובץ.
    let dir = null;
    if (window.showDirectoryPicker) {
      try { dir = await window.showDirectoryPicker({ id: 'clipper-batch', mode: 'readwrite' }); }
      catch (e) { if (e.name === 'AbortError') return; dir = null; }
    }
    if (!beginExport()) return;
    B.cancel = false;
    S.lastBatch = [];
    $('inBatch').readOnly = true;
    $('btnBatchRun').hidden = true;
    $('batchProg').hidden = false;
    const base = C.baseName(S.file.name);
    const suffix = plat() ? plat().id : S.set.format.replace(':', 'x');
    let done = 0;
    for (let i = 0; i < list.length && !B.cancel; i++) {
      const it = list[i];
      B.cur = it.text; markCur();
      it.row?.scrollIntoView({ block: 'nearest' });
      const r = await makeVideo(buildParts(it.ranges), (f, label) => {
        const all = (i + f) / list.length;
        $('batchFill').style.width = `${(all * 100).toFixed(1)}%`;
        $('batchText').textContent = label ? t('batchProgress', i + 1, list.length, Math.round(f * 100)) : t('finishing');
      });
      if (r.err) { B.status.set(it.text, statusLine(t('exportFailed', r.err.message || r.err))); renderBatch(); continue; }
      if (!r.blob) break;
      const name = `${base}_${String(it.n).padStart(2, '0')}_${stamp(it.ranges[0][0])}_${suffix}.${r.fmt.ext}`;
      const info = `${name} · ${C.fmtBytes(r.blob.size)} · ${short(r.seconds)}`;
      if (dir) {
        try {
          const fh = await dir.getFileHandle(name, { create: true });
          const w = await fh.createWritable();
          await w.write(r.blob);
          await w.close();
          B.status.set(it.text, statusLine(`✓ ${t('batchSaved')} · ${info}`));
        } catch (e) { console.error(e); dir = null; }
      }
      if (!dir) {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(r.blob);
        a.download = name;
        a.textContent = `⬇ ${info}`;
        a.addEventListener('click', (e) => e.stopPropagation());
        B.status.set(it.text, statusLine([a]));
        a.click();
      }
      done++;
      renderBatch();
      S.lastBatch = (S.lastBatch || []).concat({ name, blob: dir ? null : r.blob, size: r.blob.size, seconds: r.seconds, W: r.W, H: r.H });
    }
    $('batchProg').hidden = true;
    $('inBatch').readOnly = false;
    endExport();
    renderBatch();
    toast(B.cancel ? t('cancelled') : t('batchDone', done), 5000);
  });
  $('btnBatchCancel').addEventListener('click', () => { B.cancel = true; if (S.seq) S.seq.abort(); });

  $('btnCancel').addEventListener('click', () => { if (S.seq) S.seq.abort(); });

  // ── הגדרות פורמט ───────────────────────────────────────────────────────
  function syncControls() {
    renderPlatforms();
    $('chkGuides').checked = S.set.guides;
    document.querySelectorAll('#segFormat button').forEach((b) => b.classList.toggle('on', b.dataset.v === S.set.format));
    document.querySelectorAll('#corners button').forEach((b) => b.classList.toggle('on', b.dataset.v === S.set.corner));
    $('selFill').value = S.set.fill;
    $('inFillColor').value = S.set.fillColor;
    $('inFillColor').hidden = S.set.fill !== 'color';
    $('rngFrameZoom').value = S.set.zoom;
    $('zoomVal').textContent = `${Math.round(S.set.zoom * 100)}%`;
    $('chkText').checked = S.set.textOn;
    $('textOpts').hidden = !S.set.textOn;
    $('inTextTop').value = S.set.textTop;
    $('inTextBottom').value = S.set.textBottom;
    $('rngTextSize').value = S.set.textSize;
    $('inTextColor').value = S.set.textColor;
    $('inBatch').value = S.set.batchText;
    $('selQuality').value = S.set.quality;
    $('chkFades').checked = S.set.fades;
    $('rngLogoSize').value = S.set.logoSize;
    $('rngLogoOpacity').value = S.set.logoOpacity;
    $('chkLogoAll').checked = S.set.logoAll;
  }

  function changed() { saveSettings(); sizePreview(); drawPreview(); updateTimes(); renderBatch(); }

  $('segFormat').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    S.set.format = b.dataset.v;
    if (plat()?.format !== S.set.format) S.set.platform = null;   // שינוי ידני = כבר לא פלטפורמה מוכנה
    syncControls(); changed();
  });
  $('platforms').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    S.set.platform = b.dataset.id;
    S.set.format = plat().format;
    syncControls(); changed();
  });
  $('chkGuides').addEventListener('change', (e) => { S.set.guides = e.target.checked; changed(); });
  $('corners').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    S.set.corner = b.dataset.v; syncControls(); changed();
  });
  $('rngFrameZoom').addEventListener('input', (e) => {
    if (S.file) zoomAround(+e.target.value, preview.width / 2, preview.height / 2);
    else S.set.zoom = +e.target.value;
    frameChanged();
  });
  document.querySelectorAll('[data-frame]').forEach((b) => b.addEventListener('click', () => {
    const k = b.dataset.frame;
    if (k === 'whole') { S.set.zoom = 1; S.set.px = S.set.py = 0.5; }
    else if (k === 'fill') { S.set.zoom = coverZoom(); S.set.px = S.set.py = 0.5; }
    else { S.set.px = S.set.py = 0.5; }
    frameChanged();
  }));
  $('selFill').addEventListener('change', (e) => { S.set.fill = e.target.value; syncControls(); changed(); });
  $('inFillColor').addEventListener('input', (e) => { S.set.fillColor = e.target.value; changed(); });
  $('chkText').addEventListener('change', (e) => { S.set.textOn = e.target.checked; syncControls(); changed(); });
  $('inTextTop').addEventListener('input', (e) => { S.set.textTop = e.target.value; changed(); });
  $('inTextBottom').addEventListener('input', (e) => { S.set.textBottom = e.target.value; changed(); });
  $('rngTextSize').addEventListener('input', (e) => { S.set.textSize = +e.target.value; changed(); });
  $('inTextColor').addEventListener('input', (e) => { S.set.textColor = e.target.value; changed(); });
  $('selQuality').addEventListener('change', (e) => { S.set.quality = e.target.value; changed(); });
  $('chkFades').addEventListener('change', (e) => { S.set.fades = e.target.checked; changed(); });
  $('rngLogoSize').addEventListener('input', (e) => { S.set.logoSize = +e.target.value; changed(); });
  $('rngLogoOpacity').addEventListener('input', (e) => { S.set.logoOpacity = +e.target.value; changed(); });
  $('chkLogoAll').addEventListener('change', (e) => { S.set.logoAll = e.target.checked; changed(); });

  // שדות זמן: מקבלים "1:15.5" או "75.5"
  for (const [id, which] of [['inIn', 'in'], ['inOut', 'out']]) {
    const el = $(id);
    const commit = () => {
      const v = parseTime(el.value);
      if (!isNaN(v)) {
        if (which === 'in') setSelection(v, Math.max(S.selOut, v + 0.1));
        else setSelection(Math.min(S.selIn, v - 0.1), v);
        src.currentTime = which === 'in' ? S.selIn : S.selOut;
      }
      el.value = fmtTime(which === 'in' ? S.selIn : S.selOut);
    };
    el.addEventListener('change', commit);
    el.addEventListener('keydown', (e) => { if (e.key === 'Enter') { commit(); el.blur(); } });
  }

  // ── מיתוג ──────────────────────────────────────────────────────────────
  const SLOT = {
    logo: { stat: 'statLogo', thumb: 'thumbLogo', accept: 'image/*' },
    intro: { stat: 'statIntro', thumb: 'thumbIntro', accept: 'video/*' },
    outro: { stat: 'statOutro', thumb: 'thumbOutro', accept: 'video/*' },
  };

  async function applyBrand(kind, blob, from) {
    S.brand[kind] = blob || null;
    S.brandFrom[kind] = blob ? from : null;
    const th = $(SLOT[kind].thumb);
    th.innerHTML = '';
    if (kind === 'logo') {
      S.logoImg = null;
      if (blob) {
        try {
          S.logoImg = await C.loadImage(blob);
          const im = new Image(); im.src = S.logoImg.src; th.appendChild(im);
        } catch { S.brand.logo = null; }
      }
    } else {
      const v = kind === 'intro' ? introV : outroV;
      if (blob) {
        try {
          await C.loadVideo(v, blob);
          const c = document.createElement('canvas');
          c.width = 96; c.height = Math.round((96 * v.videoHeight) / v.videoWidth) || 54;
          await C.seekTo(v, Math.min(0.5, v.duration / 2));
          c.getContext('2d').drawImage(v, 0, 0, c.width, c.height);
          th.appendChild(c);
        } catch (e) { console.warn(kind, e); S.brand[kind] = null; }
      } else {
        v.removeAttribute('src'); v.load();
      }
    }
    renderBrandStatus();
    drawPreview();
    updateTimes();
    renderBatch();
  }

  function renderBrandStatus() {
    for (const kind of Object.keys(SLOT)) {
      const b = S.brand[kind];
      let txt = t('notSet');
      if (b) {
        const name = b.name || '';
        if (kind === 'logo') txt = name || 'PNG';
        else {
          const v = kind === 'intro' ? introV : outroV;
          txt = `${name ? name + ' · ' : ''}${t('seconds', (v.duration || 0).toFixed(1))}`;
        }
        if (S.brandFrom[kind] === 'folder') txt += ` · ${t('fromFolder')}`;
      }
      $(SLOT[kind].stat).textContent = txt;
      document.querySelector(`.slot[data-slot="${kind}"]`).classList.toggle('set', !!b);
    }
  }

  let pickKind = null;
  document.querySelectorAll('[data-pick]').forEach((b) => b.addEventListener('click', () => {
    pickKind = b.dataset.pick;
    $('brandInput').accept = SLOT[pickKind].accept;
    $('brandInput').click();
  }));
  $('brandInput').addEventListener('change', async (e) => {
    const f = e.target.files[0];
    e.target.value = '';
    if (!f || !pickKind) return;
    const want = pickKind === 'logo' ? 'image/' : 'video/';
    if (f.type && !f.type.startsWith(want)) { toast(t('badFile')); return; }
    await applyBrand(pickKind, f, 'user');
    if (S.brand[pickKind]) { await C.store.set('brand.' + pickKind, f); toast(t('savedBrand', t(pickKind === 'logo' ? 'slotLogo' : pickKind === 'intro' ? 'slotIntro' : 'slotOutro'))); }
  });
  document.querySelectorAll('[data-clear]').forEach((b) => b.addEventListener('click', async () => {
    const kind = b.dataset.clear;
    await C.store.set('brand.' + kind, null);
    await applyBrand(kind, null);
  }));

  /** ברירות מחדל מתיקיית brand/ (רק כשהאתר מוגש מ-http, כי file:// חוסם fetch) */
  async function folderBrand(kind) {
    if (!location.protocol.startsWith('http')) return null;
    try {
      if (!folderBrand.cfg) {
        const r = await fetch('brand/brand.json', { cache: 'no-cache' });
        folderBrand.cfg = r.ok ? await r.json() : {};
      }
      const file = folderBrand.cfg[kind];
      if (!file) return null;
      const r = await fetch('brand/' + file);
      if (!r.ok) return null;
      const b = await r.blob();
      return new File([b], file, { type: b.type });
    } catch { return null; }
  }

  // ── מקלדת ──────────────────────────────────────────────────────────────
  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, select, textarea')) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (e.key === ' ') { e.preventDefault(); $('btnPlay').click(); }
    else if (k === 'i') $('btnSetIn').click();
    else if (k === 'o') $('btnSetOut').click();
    else if (k === 'p') $('btnPlaySel').click();
    else if (k === 'm') $('btnMute').click();
    else if (e.key === 'ArrowLeft') { e.preventDefault(); stepFrame(e.shiftKey ? -5 : -1 / 30); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); stepFrame(e.shiftKey ? 5 : 1 / 30); }
    else if (e.key === 'Home') { stopAll(); src.currentTime = S.selIn; }
    else if (e.key === 'End') { stopAll(); src.currentTime = S.selOut; }
  });

  // ── זום ────────────────────────────────────────────────────────────────
  $('rngZoom').addEventListener('input', (e) => tl.setZoom01(+e.target.value));
  $('btnZoomIn').addEventListener('click', () => { tl.setZoom01(Math.min(1, tl.getZoom01() + 0.1)); $('rngZoom').value = tl.getZoom01(); });
  $('btnZoomOut').addEventListener('click', () => { tl.setZoom01(Math.max(0, tl.getZoom01() - 0.1)); $('rngZoom').value = tl.getZoom01(); });
  $('btnZoomFit').addEventListener('click', () => { tl.zoomFit(); });
  $('btnZoomSel').addEventListener('click', () => { if (S.selOut > S.selIn) tl.zoomTo(S.selIn, S.selOut); });

  // ── שפה ────────────────────────────────────────────────────────────────
  $('btnLang').addEventListener('click', () => {
    C.setLang(C.lang === 'he' ? 'en' : 'he');
    renderPlatforms(); sizePreview(); renderBrandStatus(); drawPreview(); updateTimes(); renderBatch();
  });

  // ── אתחול ──────────────────────────────────────────────────────────────
  (async () => {
    C.applyI18n();
    const saved = await C.store.get('settings');
    if (saved) { delete saved.fit; delete saved.focus; Object.assign(S.set, saved); }   // fit/focus: לפני שהיה מסגור חופשי
    if (S.set.platform && C.platform(S.set.platform)?.format !== S.set.format) S.set.platform = null;
    syncControls();
    renderBatch();
    applyVolume();
    sizePreview();
    drawPreview();
    for (const kind of Object.keys(SLOT)) {
      let blob = await C.store.get('brand.' + kind);
      let from = 'user';
      if (!blob) { blob = await folderBrand(kind); from = 'folder'; }
      if (blob) await applyBrand(kind, blob, from);
    }
    renderBrandStatus();
    document.body.classList.add('ready');
  })();
})();
