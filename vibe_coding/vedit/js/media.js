/* media.js: ייבוא קבצים, קריאת מטא-דאטה, תמונות ממוזערות, צורת גל ואחסון מקומי
 *
 * המידע הסריאליזבילי על כל פריט מדיה נשמר ב-state.proj.media.
 * הקובץ עצמו, ה-objectURL, הפילמסטריפ וצורת הגל חיים כאן ב-registry.
 */

import { uid, toast, once, baseName } from './util.js';
import { state, proj, emitter, beginChange, commit } from './state.js';

/** mediaId → { file, url, thumbs:[ImageBitmap], peaks:Float32Array, ready:Promise } */
const registry = new Map();
export const rt = (id) => registry.get(id);

const THUMB_COUNT = 14;      // כמה תמונות בפילמסטריפ
const THUMB_W = 96;
const PEAKS_PER_SEC = 120;

/* ─────────────── ייבוא ─────────────── */

const kindOf = (file) => {
  const t = file.type || '';
  if (t.startsWith('video')) return 'video';
  if (t.startsWith('audio')) return 'audio';
  if (t.startsWith('image')) return 'image';
  // גיבוי לפי סיומת (יש דפדפנים שלא ממלאים type)
  const ext = file.name.split('.').pop().toLowerCase();
  if (['mp4', 'webm', 'mov', 'mkv', 'avi', 'm4v', 'ogv'].includes(ext)) return 'video';
  if (['mp3', 'wav', 'ogg', 'm4a', 'aac', 'flac', 'opus'].includes(ext)) return 'audio';
  if (['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'avif'].includes(ext)) return 'image';
  return null;
};

export async function importFiles(fileList) {
  const files = [...fileList];
  const added = [];
  for (const file of files) {
    const type = kindOf(file);
    if (!type) { toast(`דילגתי על "${file.name}": סוג קובץ לא נתמך`, 'err'); continue; }
    try {
      const item = await probe(file, type);
      added.push(item);
    } catch (err) {
      console.error(err);
      toast(`לא הצלחתי לפתוח את "${file.name}"`, 'err');
    }
  }
  if (added.length) {
    beginChange();
    proj().media.push(...added);
    commit('import');
    emitter.emit('media');
    // עיבוד רקע: פילמסטריפ + צורת גל
    added.forEach((m) => buildAssets(m));
  }
  return added;
}

/** קריאת מטא-דאטה בסיסית של קובץ */
async function probe(file, type) {
  const url = URL.createObjectURL(file);
  const item = {
    id: uid('med'), name: file.name, type,
    duration: 0, width: 0, height: 0, hasAudio: type !== 'image',
    size: file.size,
  };
  registry.set(item.id, { file, url, thumbs: [], peaks: null });

  if (type === 'image') {
    const img = new Image();
    img.src = url;
    await img.decode().catch(() => {});
    item.width = img.naturalWidth || 1920;
    item.height = img.naturalHeight || 1080;
    item.duration = 5;
    item.hasAudio = false;
    registry.get(item.id).image = img;
    return item;
  }

  const el = document.createElement(type === 'audio' ? 'audio' : 'video');
  el.preload = 'metadata';
  el.src = url;
  el.muted = true;
  const ok = await once(el, 'loadedmetadata', 15000);
  if (!ok || !isFinite(el.duration) || el.duration <= 0) {
    // חלק מהקבצים (למשל webm מהקלטה) לא מדווחים duration עד שמדלגים לסוף
    el.currentTime = 1e6;
    await once(el, 'timeupdate', 3000);
    el.currentTime = 0;
  }
  if (!isFinite(el.duration) || el.duration <= 0) throw new Error('no duration');
  item.duration = el.duration;
  item.width = el.videoWidth || 0;
  item.height = el.videoHeight || 0;
  if (type === 'video' && !item.width) throw new Error('no video track');
  // מניחים שיש פס קול; buildPeaks יכבה את הדגל אם הפענוח ייכשל
  item.hasAudio = true;
  return item;
}

/* ─────────────── פילמסטריפ ─────────────── */

async function buildAssets(item) {
  const r = registry.get(item.id);
  if (!r) return;
  if (item.type === 'video') { await buildThumbs(item).catch(() => {}); }
  if (item.hasAudio) { await buildPeaks(item).catch(() => {}); }
  emitter.emit('assets', item.id);
}

async function buildThumbs(item) {
  const r = registry.get(item.id);
  const v = document.createElement('video');
  v.src = r.url; v.muted = true; v.preload = 'auto';
  if (!(await once(v, 'loadedmetadata', 12000))) return;

  const h = Math.max(24, Math.round(THUMB_W * (item.height / Math.max(1, item.width))));
  const cv = document.createElement('canvas');
  cv.width = THUMB_W; cv.height = h;
  const ctx = cv.getContext('2d');
  const n = Math.min(THUMB_COUNT, Math.max(3, Math.ceil(item.duration)));
  const thumbs = [];

  for (let i = 0; i < n; i++) {
    const t = (item.duration * (i + 0.5)) / n;
    v.currentTime = Math.min(t, Math.max(0, item.duration - 0.05));
    if (!(await once(v, 'seeked', 6000))) break;
    ctx.drawImage(v, 0, 0, cv.width, cv.height);
    try { thumbs.push(await createImageBitmap(cv)); } catch { break; }
  }
  r.thumbs = thumbs;
  r.thumbAspect = THUMB_W / h;
  v.src = '';
  emitter.emit('assets', item.id);
}

/* ─────────────── צורת גל ─────────────── */

let audioCtx = null;
export function getAudioCtx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  return audioCtx;
}

/* פענוח האודיו כולו לזיכרון כדי לחשב צורת גל יקר בזיכרון: שעה של סטריאו =
   מאות מגה-בייט. בקבצים ארוכים או כבדים מוותרים על הגל. האודיו עצמו עדיין מתנגן. */
const PEAKS_MAX_SECONDS = 12 * 60;
const PEAKS_MAX_BYTES = 300 * 1024 * 1024;

async function buildPeaks(item) {
  const r = registry.get(item.id);
  if (!r) return;
  if (item.duration > PEAKS_MAX_SECONDS || item.size > PEAKS_MAX_BYTES) {
    console.info(`[vedit] דילוג על צורת גל עבור "${item.name}" (קובץ ארוך/כבד)`);
    return;
  }
  const buf = await r.file.arrayBuffer();
  let audio;
  try {
    audio = await getAudioCtx().decodeAudioData(buf);
  } catch {
    // אין פס קול או שהדפדפן לא יודע לפענח אותו
    if (item.type === 'video') { item.hasAudio = false; emitter.emit('assets', item.id); }
    return;
  }
  const n = Math.max(1, Math.floor(audio.duration * PEAKS_PER_SEC));
  const peaks = new Float32Array(n);
  const chs = Math.min(2, audio.numberOfChannels);
  const step = audio.length / n;
  for (let ch = 0; ch < chs; ch++) {
    const d = audio.getChannelData(ch);
    for (let i = 0; i < n; i++) {
      const a = Math.floor(i * step), b = Math.min(d.length, Math.floor((i + 1) * step));
      let peak = 0;
      for (let j = a; j < b; j += 2) { const v = d[j] < 0 ? -d[j] : d[j]; if (v > peak) peak = v; }
      if (peak > peaks[i]) peaks[i] = peak;
    }
  }
  r.peaks = peaks;
  r.peaksPerSec = PEAKS_PER_SEC;
  item.hasAudio = true;
  emitter.emit('assets', item.id);
}

/* ─────────────── מחיקת מדיה ─────────────── */

export function removeMedia(id) {
  beginChange();
  proj().media = proj().media.filter((m) => m.id !== id);
  for (const t of proj().tracks) t.clips = t.clips.filter((c) => c.mediaId !== id);
  commit('remove-media');
  emitter.emit('media');
  idbDelete(id);
}

/* ─────────────── אחסון מקומי (IndexedDB) ─────────────── */

const DB_NAME = 'vedit';
const DB_VER = 1;
let dbPromise = null;

function db() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VER);
    req.onupgradeneeded = () => {
      const d = req.result;
      if (!d.objectStoreNames.contains('files')) d.createObjectStore('files', { keyPath: 'id' });
      if (!d.objectStoreNames.contains('meta')) d.createObjectStore('meta');
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

async function tx(store, mode, fn) {
  const d = await db();
  return new Promise((resolve, reject) => {
    const t = d.transaction(store, mode);
    const s = t.objectStore(store);
    const out = fn(s);
    t.oncomplete = () => resolve(out?.result ?? out);
    t.onerror = () => reject(t.error);
  });
}

export async function idbSaveFile(item) {
  const r = registry.get(item.id);
  if (!r?.file) return;
  try { await tx('files', 'readwrite', (s) => s.put({ id: item.id, name: item.name, blob: r.file })); }
  catch (e) { console.warn('idb save failed', e); }
}

async function idbDelete(id) {
  try { await tx('files', 'readwrite', (s) => s.delete(id)); } catch {}
}

export async function idbSaveProject(json) {
  try { await tx('meta', 'readwrite', (s) => s.put(json, 'project')); } catch {}
}

export async function idbLoadProject() {
  try { return await tx('meta', 'readonly', (s) => s.get('project')); } catch { return null; }
}

export async function idbClear() {
  try {
    await tx('files', 'readwrite', (s) => s.clear());
    await tx('meta', 'readwrite', (s) => s.clear());
  } catch {}
}

/** משחזר את הקבצים מ-IndexedDB אחרי טעינת פרויקט שמור */
export async function rehydrate(mediaList) {
  const missing = [];
  for (const m of mediaList) {
    if (registry.has(m.id)) continue;
    let rec = null;
    try { rec = await tx('files', 'readonly', (s) => s.get(m.id)); } catch {}
    if (!rec?.blob) { missing.push(m); continue; }
    const url = URL.createObjectURL(rec.blob);
    registry.set(m.id, { file: rec.blob, url, thumbs: [], peaks: null });
    if (m.type === 'image') {
      const img = new Image(); img.src = url;
      await img.decode().catch(() => {});
      registry.get(m.id).image = img;
    }
    buildAssets(m);
  }
  return missing;
}

/** קישור מחדש של קובץ לפריט מדיה קיים (אחרי פתיחת פרויקט מקובץ) */
export async function relink(item, file) {
  const url = URL.createObjectURL(file);
  registry.set(item.id, { file, url, thumbs: [], peaks: null });
  if (item.type === 'image') {
    const img = new Image(); img.src = url;
    await img.decode().catch(() => {});
    registry.get(item.id).image = img;
  }
  buildAssets(item);
  idbSaveFile(item);
  emitter.emit('media');
}

export function hasFile(id) { return !!registry.get(id)?.url; }

export function mediaLabel(m) {
  if (m.type === 'image') return `תמונה · ${m.width}×${m.height}`;
  const res = m.width ? ` · ${m.height}p` : '';
  return `${m.type === 'audio' ? 'אודיו' : 'וידאו'}${res}`;
}

export { baseName };
