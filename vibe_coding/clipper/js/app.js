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
      platform: 'ig_reels', format: '9:16', guides: true, fit: 'blur', focus: 0.5, quality: '1080', fades: true,
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
    fit: S.set.fit, focus: S.set.focus, logo: S.logoImg, safe: safe(),
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
    $('rowFocus').hidden = S.set.fit !== 'cover';
    $('platformNote').textContent = plat() ? plat().note[C.lang] : '';
    // אינסטגרם ורוב הרשתות דורשות AAC. כרום על לינוקס יודע רק Opus.
    const fmt = C.sequencer.pickFormat();
    const noAac = plat()?.needsAac && fmt && !fmt.mime.includes('mp4a');
    $('codecWarn').hidden = !noAac;
    if (noAac) $('codecWarn').textContent = t('noAacWarn', platName());
  }

  /** האורך הכולל (פתיחה + קטע + סיום) */
  function totalSeconds() {
    const extra = (S.brand.intro && introV.duration ? introV.duration : 0) + (S.brand.outro && outroV.duration ? outroV.duration : 0);
    return Math.max(0, S.selOut - S.selIn) + extra;
  }

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
    if (S.mode === 'sequence' || S.mode === 'export') return;
    C.compose.draw(pctx, S.file ? src : null, preview.width, preview.height, { ...look(), logoOn: !!S.logoImg, fade: 0 });
    drawGuides(pctx, preview.width, preview.height);
    setBadge(S.file ? 'stageMain' : '');
  }

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
  function buildParts() {
    const parts = [];
    // פתיחה וסיום אף פעם לא נחתכים: יש בהם טקסט וכתובות שחייבים להיראות במלואם
    const brandLook = { fit: S.set.fit === 'cover' ? 'blur' : S.set.fit };
    if (S.brand.intro && introV.duration) parts.push({ video: introV, from: 0, to: introV.duration, logo: S.set.logoAll && !!S.logoImg, look: brandLook, label: t('stageIntro') });
    parts.push({ video: src, from: S.selIn, to: S.selOut, logo: !!S.logoImg, label: t('stageMain') });
    if (S.brand.outro && outroV.duration) parts.push({ video: outroV, from: 0, to: outroV.duration, logo: S.set.logoAll && !!S.logoImg, look: brandLook, label: t('stageOutro') });
    return parts;
  }

  function checkReady() {
    if (!S.file) { toast(t('noVideo')); return false; }
    if (S.selOut - S.selIn < 0.2) { toast(t('selTooShort')); return false; }
    return true;
  }

  $('btnPlayAll').addEventListener('click', async () => {
    if (S.mode === 'sequence') { stopAll(); drawPreview(); return; }
    if (!checkReady()) return;
    stopAll();
    S.mode = 'sequence';
    $('btnPlay').textContent = '⏸';
    const parts = buildParts();
    S.seq = C.sequencer.run({
      parts, canvas: preview, look: look(), fades: S.set.fades, record: null,
      onProgress: (_f, label) => { if (label) $('stageBadge').textContent = label; $('stageBadge').hidden = !label; tl.setPlayhead(src.currentTime); },
    });
    try { await S.seq.done; } catch (e) { console.warn(e); }
    if (S.mode === 'sequence') S.mode = 'idle';
    S.seq = null;
    $('btnPlay').textContent = '▶';
    drawPreview();
  });

  // ── ייצוא ──────────────────────────────────────────────────────────────
  $('btnExport').addEventListener('click', async () => {
    if (S.mode === 'export') return;
    if (!checkReady()) return;
    if (tooLong() && !confirm(t('confirmTooLong'))) return;
    const fmt = C.sequencer.pickFormat();
    if (!fmt) { toast(t('noRecorder'), 6000); return; }
    stopAll();
    S.mode = 'export';
    document.body.classList.add('exporting');
    $('result').hidden = true;
    $('progress').hidden = false;
    $('progFill').style.width = '0%';

    const { W, H } = outSize();
    const cv = document.createElement('canvas');
    cv.width = W; cv.height = H;
    // ~0.15 ביט לפיקסל בשנייה ב-30fps: 1080×1920 ≈ 9.3Mbps. מספיק בשביל לשרוד את הדחיסה החוזרת של הרשתות.
    const bitrate = Math.round(W * H * 30 * 0.15);
    const parts = buildParts();
    const t0 = performance.now();
    const seq = S.seq = C.sequencer.run({
      parts, canvas: cv, look: look(), fades: S.set.fades,
      record: { mime: fmt.mime, bitrate },
      onFrame: (c) => pctx.drawImage(c, 0, 0, preview.width, preview.height),
      onProgress: (f, label) => {
        $('progFill').style.width = `${(f * 100).toFixed(1)}%`;
        $('progText').textContent = label ? t('exporting', Math.round(f * 100), label) : t('finishing');
        if (label) { $('stageBadge').textContent = label; $('stageBadge').hidden = false; }
      },
    });
    let blob = null, err = null;
    try { blob = await S.seq.done; } catch (e) { err = e; console.error(e); }
    S.seq = null;
    S.mode = 'idle';
    document.body.classList.remove('exporting');
    $('progress').hidden = true;
    drawPreview();

    if (err) { toast(t('exportFailed', err.message || err), 8000); return; }
    if (!blob) { toast(t('cancelled')); return; }

    const url = URL.createObjectURL(blob);
    const rv = $('resultVideo');
    if (rv._url) URL.revokeObjectURL(rv._url);
    rv._url = url;
    rv.src = url;
    const name = `${C.baseName(S.file.name)}_${plat() ? plat().id : S.set.format.replace(':', 'x')}.${fmt.ext}`;
    const a = $('btnDownload');
    a.href = url;
    a.download = name;
    const total = parts.reduce((s, p) => s + p.to - p.from, 0);
    $('resultInfo').textContent = t('resultInfo', C.fmtBytes(blob.size), fmtTime(total), `${fmt.label} · ${W}×${H}`)
      + (fmt.warn ? `\n${t(fmt.warn)}` : '');
    $('result').hidden = false;
    S.lastExport = { blob, name, mime: fmt.mime, W, H, seconds: (performance.now() - t0) / 1000, stats: seq.stats };
    toast(t('done'));
  });

  $('btnCancel').addEventListener('click', () => { if (S.seq) S.seq.abort(); });

  // ── הגדרות פורמט ───────────────────────────────────────────────────────
  function syncControls() {
    renderPlatforms();
    $('chkGuides').checked = S.set.guides;
    document.querySelectorAll('#segFormat button').forEach((b) => b.classList.toggle('on', b.dataset.v === S.set.format));
    document.querySelectorAll('#corners button').forEach((b) => b.classList.toggle('on', b.dataset.v === S.set.corner));
    $('selFit').value = S.set.fit;
    $('rngFocus').value = S.set.focus;
    $('selQuality').value = S.set.quality;
    $('chkFades').checked = S.set.fades;
    $('rngLogoSize').value = S.set.logoSize;
    $('rngLogoOpacity').value = S.set.logoOpacity;
    $('chkLogoAll').checked = S.set.logoAll;
  }

  function changed() { saveSettings(); sizePreview(); drawPreview(); updateTimes(); }

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
  $('selFit').addEventListener('change', (e) => { S.set.fit = e.target.value; changed(); });
  $('rngFocus').addEventListener('input', (e) => { S.set.focus = +e.target.value; changed(); });
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
    renderPlatforms(); sizePreview(); renderBrandStatus(); drawPreview(); updateTimes();
  });

  // ── אתחול ──────────────────────────────────────────────────────────────
  (async () => {
    C.applyI18n();
    const saved = await C.store.get('settings');
    if (saved) Object.assign(S.set, saved);
    if (S.set.platform && C.platform(S.set.platform)?.format !== S.set.format) S.set.platform = null;
    syncControls();
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
