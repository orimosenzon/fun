/* util.js: עזרים קטנים שכל השאר משתמש בהם */
window.C = window.C || {};

C.$ = (id) => document.getElementById(id);
C.clamp = (v, a, b) => Math.min(b, Math.max(a, v));

/** 75.5 → "1:15.50", ומעל שעה 4211 → "1:10:11.00" (הקלטות של מפגשים נמשכות שעות) */
C.fmtTime = (s) => {
  if (!isFinite(s) || s < 0) s = 0;
  s = Math.round(s * 100) / 100;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s - h * 3600) / 60);
  const r = (s - h * 3600 - m * 60).toFixed(2).padStart(5, '0');
  return h ? `${h}:${String(m).padStart(2, '0')}:${r}` : `${m}:${r}`;
};

/** "1:15.5" / "75.5" / "0:01:15" → שניות, או NaN */
C.parseTime = (str) => {
  const parts = String(str).trim().split(':').map((p) => p.trim());
  if (!parts.length || parts.some((p) => p === '' || isNaN(+p))) return NaN;
  return parts.reduce((acc, p) => acc * 60 + +p, 0);
};

C.fmtBytes = (n) => {
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
};

let toastTimer = null;
C.toast = (msg, ms = 3500) => {
  const el = C.$('toast');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, ms);
};

/** מחכה לאירוע אחד על אלמנט, עם timeout כדי שלא ניתקע לנצח */
C.once = (el, ev, ms = 8000) => new Promise((res, rej) => {
  const t = setTimeout(() => { el.removeEventListener(ev, h); rej(new Error(`timeout: ${ev}`)); }, ms);
  const h = () => { clearTimeout(t); res(); };
  el.addEventListener(ev, h, { once: true });
});

/** קפיצה לזמן בווידאו ומחכה שהפריים באמת מוכן */
C.seekTo = async (v, t) => {
  t = C.clamp(t, 0, v.duration || 0);
  if (Math.abs(v.currentTime - t) < 0.001 && v.readyState >= 2) return;
  const p = C.once(v, 'seeked');
  v.currentTime = t;
  await p;
};

/** טוען Blob לתוך <video> ומחכה למטא-דאטה */
C.loadVideo = async (v, blob) => {
  if (v._url) URL.revokeObjectURL(v._url);
  v._url = URL.createObjectURL(blob);
  const p = C.once(v, 'loadeddata', 30000);
  const err = new Promise((_, rej) => v.addEventListener('error', () => rej(new Error('decode')), { once: true }));
  v.src = v._url;
  v.load();
  await Promise.race([p, err]);
};

C.loadImage = (blob) => new Promise((res, rej) => {
  const img = new Image();
  img.onload = () => res(img);
  img.onerror = () => rej(new Error('image'));
  img.src = URL.createObjectURL(blob);
});

C.baseName = (name) => (name || 'video').replace(/\.[^.]+$/, '');
