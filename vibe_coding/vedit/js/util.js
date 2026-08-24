/* util.js: פונקציות עזר כלליות */

export const $  = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
export const lerp  = (a, b, t) => a + (b - a) * t;

let idCounter = 0;
export function uid(prefix = 'id') {
  idCounter++;
  return `${prefix}_${Date.now().toString(36)}_${idCounter.toString(36)}`;
}

/** timecode בפורמט HH:MM:SS:FF */
export function tc(seconds, fps = 30) {
  if (!isFinite(seconds) || seconds < 0) seconds = 0;
  const totalFrames = Math.round(seconds * fps);
  const f = totalFrames % fps;
  const totalSec = Math.floor(totalFrames / fps);
  const s = totalSec % 60;
  const m = Math.floor(totalSec / 60) % 60;
  const h = Math.floor(totalSec / 3600);
  const p = (n) => String(n).padStart(2, '0');
  return `${p(h)}:${p(m)}:${p(s)}:${p(f)}`;
}

/** תצוגה קצרה של משך: 1:23 או 1:02:03 */
export function shortDur(seconds) {
  if (!isFinite(seconds)) return '?';
  const s = Math.floor(seconds % 60);
  const m = Math.floor(seconds / 60) % 60;
  const h = Math.floor(seconds / 3600);
  const p = (n) => String(n).padStart(2, '0');
  return h ? `${h}:${p(m)}:${p(s)}` : `${m}:${p(s)}`;
}

export function bytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(0)} KB`;
  if (n < 1073741824) return `${(n / 1048576).toFixed(1)} MB`;
  return `${(n / 1073741824).toFixed(2)} GB`;
}

/** עיגול לפריים הקרוב */
export const snapFrame = (t, fps) => Math.round(t * fps) / fps;

export function toast(msg, kind = '', ms = 2600) {
  const host = $('#toasts');
  if (!host) return;
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  host.appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity .3s';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 320);
  }, ms);
}

/** מנגנון אירועים פשוט */
export class Emitter {
  constructor() { this._m = new Map(); }
  on(evt, fn) {
    if (!this._m.has(evt)) this._m.set(evt, new Set());
    this._m.get(evt).add(fn);
    return () => this.off(evt, fn);
  }
  off(evt, fn) { this._m.get(evt)?.delete(fn); }
  emit(evt, payload) {
    this._m.get(evt)?.forEach((fn) => {
      try { fn(payload); } catch (e) { console.error(`[${evt}]`, e); }
    });
  }
}

/** גרירה נוחה: מחזיר ניקוי. onMove מקבל (dx, dy, ev) */
export function drag(startEv, { onMove, onEnd, cursor }) {
  startEv.preventDefault();
  const x0 = startEv.clientX, y0 = startEv.clientY;
  const prevCursor = document.body.style.cursor;
  if (cursor) document.body.style.cursor = cursor;
  let moved = false;

  const mm = (e) => {
    const dx = e.clientX - x0, dy = e.clientY - y0;
    if (!moved && Math.abs(dx) + Math.abs(dy) > 2) moved = true;
    onMove?.(dx, dy, e, moved);
  };
  const mu = (e) => {
    window.removeEventListener('pointermove', mm);
    window.removeEventListener('pointerup', mu);
    document.body.style.cursor = prevCursor;
    onEnd?.(moved, e);
  };
  window.addEventListener('pointermove', mm);
  window.addEventListener('pointerup', mu);
}

/** debounce */
export function debounce(fn, ms = 120) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

/** המתנה לאירוע בודד עם timeout */
export function once(target, evt, ms = 5000) {
  return new Promise((resolve) => {
    let done = false;
    const h = () => { if (done) return; done = true; cleanup(); resolve(true); };
    const timer = setTimeout(() => { if (done) return; done = true; cleanup(); resolve(false); }, ms);
    function cleanup() { clearTimeout(timer); target.removeEventListener(evt, h); }
    target.addEventListener(evt, h, { once: true });
  });
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** האם הפוקוס בשדה קלט (כדי לא לחטוף קיצורי מקלדת) */
export function inField(el = document.activeElement) {
  if (!el) return false;
  const t = el.tagName;
  return t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT' || el.isContentEditable;
}

export function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

/** שם קובץ בלי סיומת */
export const baseName = (n) => n.replace(/\.[^.]+$/, '');
