/* state.js: מודל הפרויקט, בחירה, והיסטוריית בטל/בצע-שוב
 *
 * המודל כולו נתונים סריאליזביליים (JSON). הקבצים עצמם (Blob / <video>) יושבים
 * במרשם נפרד ב-media.js וממופים לפי mediaId, כך שאפשר לצלם snapshot של
 * הפרויקט להיסטוריה בלי לשכפל וידאו.
 */

import { Emitter, uid, clamp, snapFrame } from './util.js';
import { log } from './logger.js';

const L = log.tag('edit');

/** קליפים שנמחקו לאחרונה, כדי שאפשר יהיה להסביר לאן הם נעלמו */
export const lastRemoved = [];
function noteRemoved(clips, reason) {
  if (!clips.length) return;
  for (const c of clips) {
    lastRemoved.push({ name: c.name, start: +c.start.toFixed(2), dur: +c.duration.toFixed(2), reason, at: Date.now() });
  }
  while (lastRemoved.length > 20) lastRemoved.shift();
  L.warn('clips removed', {
    reason, count: clips.length,
    clips: clips.map((c) => `${c.name}@${c.start.toFixed(2)}+${c.duration.toFixed(2)}`),
  });
}

export const EPS = 1e-4;                 // סובלנות להשוואת זמנים
export const MIN_CLIP = 0.04;            // אורך קליפ מינימלי (שניות)

export const emitter = new Emitter();
export const on = (e, f) => emitter.on(e, f);

/* ─────────────── מבנה ברירת מחדל ─────────────── */

function newTrack(kind, name) {
  return {
    id: uid('trk'), kind, name,
    height: kind === 'video' ? 62 : 48,
    muted: false, hidden: false, locked: false,
    clips: [],
  };
}

function blankProject() {
  return {
    name: 'פרויקט חדש',
    width: 1920, height: 1080, fps: 30,
    media: [],
    tracks: [newTrack('video', 'V1'), newTrack('audio', 'A1')],
  };
}

/* ─────────────── מצב חי ─────────────── */

export const state = {
  proj: blankProject(),
  playhead: 0,
  selection: new Set(),      // מזהי קליפים נבחרים
  selTransition: null,       // מזהה קליפ שהמעבר שבראשו נבחר
  activeTrack: null,         // ערוץ יעד להדבקה/הוספה
  inPoint: null,
  outPoint: null,
  zoom: 60,                  // פיקסלים לשנייה
  tool: 'select',
  snap: true,
  playing: false,
};

export const proj = () => state.proj;
export const fps = () => state.proj.fps;

/* ─────────────── היסטוריה ─────────────── */

const history = { past: [], future: [], limit: 80, suspended: false };

function snapshot() {
  return JSON.stringify({
    name: state.proj.name, width: state.proj.width, height: state.proj.height,
    fps: state.proj.fps, media: state.proj.media, tracks: state.proj.tracks,
  });
}

function restore(json) {
  const o = JSON.parse(json);
  state.proj.name = o.name;
  state.proj.width = o.width; state.proj.height = o.height; state.proj.fps = o.fps;
  state.proj.media = o.media; state.proj.tracks = o.tracks;
  // ניקוי בחירות שכבר לא קיימות
  const live = new Set(allClips().map((c) => c.id));
  [...state.selection].forEach((id) => { if (!live.has(id)) state.selection.delete(id); });
  if (state.selTransition && !live.has(state.selTransition)) state.selTransition = null;
}

let pending = null;   // הצילום שנלקח לפני השינוי הנוכחי

/** לקרוא *לפני* שינוי במודל */
export function beginChange() {
  if (history.suspended || pending) return;
  pending = snapshot();
}

/** לקרוא *אחרי* השינוי: סוגר צעד היסטוריה ומודיע לממשק */
export function commit(what = 'change') {
  if (pending && !history.suspended) {
    const now = snapshot();
    if (now !== pending) {
      history.past.push(pending);
      if (history.past.length > history.limit) history.past.shift();
      history.future.length = 0;
    }
  }
  pending = null;
  emitter.emit('change', what);
  emitter.emit('history');
}

/** שינוי עטוף: beginChange → fn() → commit */
export function change(what, fn) {
  beginChange();
  const r = fn();
  commit(what);
  return r;
}

/** שינוי בלי רישום בהיסטוריה (למשל גרירה חיה, לפני שחרור העכבר) */
export function silent(fn) {
  history.suspended = true;
  try { fn(); } finally { history.suspended = false; }
}

export function undo() {
  if (!history.past.length) return false;
  history.future.push(snapshot());
  restore(history.past.pop());
  emitter.emit('change', 'undo');
  emitter.emit('history');
  return true;
}

export function redo() {
  if (!history.future.length) return false;
  history.past.push(snapshot());
  restore(history.future.pop());
  emitter.emit('change', 'redo');
  emitter.emit('history');
  return true;
}

export const canUndo = () => history.past.length > 0;
export const canRedo = () => history.future.length > 0;
export function clearHistory() { history.past.length = 0; history.future.length = 0; pending = null; }

/* ─────────────── שאילתות ─────────────── */

export const allClips = () => state.proj.tracks.flatMap((t) => t.clips);
export const trackById = (id) => state.proj.tracks.find((t) => t.id === id);
export const clipById = (id) => { for (const t of state.proj.tracks) { const c = t.clips.find((c) => c.id === id); if (c) return c; } return null; };
export const trackOfClip = (id) => state.proj.tracks.find((t) => t.clips.some((c) => c.id === id));
export const mediaById = (id) => state.proj.media.find((m) => m.id === id);

export const clipEnd = (c) => c.start + c.duration;

export function duration() {
  let d = 0;
  for (const t of state.proj.tracks) for (const c of t.clips) d = Math.max(d, clipEnd(c));
  return d;
}

export function sortTrack(t) { t.clips.sort((a, b) => a.start - b.start); }

/** הקליפ שנמצא בזמן t בערוץ נתון */
export function clipAt(track, t) {
  return track.clips.find((c) => t >= c.start - EPS && t < clipEnd(c) - EPS) || null;
}

/** הקליפ הקודם הצמוד (נוגע בהתחלה של c) */
export function prevAdjacent(track, c) {
  let best = null;
  for (const o of track.clips) {
    if (o === c) continue;
    if (Math.abs(clipEnd(o) - c.start) < 0.02) { if (!best || o.start > best.start) best = o; }
  }
  return best;
}

/* ─────────────── יצירת קליפים ─────────────── */

export function makeClip(media, { trackId, start = 0, inPoint = 0, duration: dur } = {}) {
  const isAudioOnly = media.type === 'audio';
  return {
    id: uid('clip'),
    kind: media.type === 'image' ? 'image' : isAudioOnly ? 'audio' : 'video',
    mediaId: media.id,
    name: media.name,
    trackId,
    start,
    inPoint,
    duration: dur ?? (media.type === 'image' ? 5 : media.duration),
    speed: 1,
    // אודיו
    volume: 1, mute: false, aFadeIn: 0, aFadeOut: 0,
    // וידאו
    opacity: 1, scale: 1, posX: 0, posY: 0, rotation: 0,
    flipH: false,
    vFadeIn: 0, vFadeOut: 0,
    filters: { brightness: 100, contrast: 100, saturate: 100, blur: 0 },
    tin: null,   // מעבר בראש הקליפ: {type, dur}
  };
}

export function makeTitleClip({ trackId, start = 0, duration: dur = 4, text = 'כותרת',
  size = 72, color = '#ffffff', style = 'shadow' } = {}) {
  return {
    id: uid('clip'), kind: 'title', mediaId: null, name: text.split('\n')[0].slice(0, 24) || 'כותרת',
    trackId, start, inPoint: 0, duration: dur, speed: 1,
    volume: 1, mute: true, aFadeIn: 0, aFadeOut: 0,
    opacity: 1, scale: 1, posX: 0, posY: 0, rotation: 0, flipH: false,
    vFadeIn: 0.3, vFadeOut: 0.3,
    filters: { brightness: 100, contrast: 100, saturate: 100, blur: 0 },
    tin: null,
    text, fontSize: size, color, titleStyle: style, align: 'center',
  };
}

/** משך המקור הזמין לקליפ (בשניות של הטיימליין, אחרי מהירות) */
export function sourceRoom(clip) {
  if (clip.kind === 'title') return Infinity;
  const m = mediaById(clip.mediaId);
  if (!m) return clip.duration;
  if (m.type === 'image') return Infinity;
  return Math.max(0, (m.duration - clip.inPoint) / (clip.speed || 1));
}

/* ─────────────── עריכה ─────────────── */

/** פינוי מקום בערוץ: חותך/מקצר/מוחק כל מה שחופף לתחום [a,b), בהתנהגות "דריסה" */
export function carve(track, a, b, exceptId = null) {
  const out = [];
  const swallowed = [];
  for (const c of track.clips) {
    if (c.id === exceptId) { out.push(c); continue; }
    const s = c.start, e = clipEnd(c);
    if (e <= a + EPS || s >= b - EPS) { out.push(c); continue; }        // אין חפיפה
    if (s >= a - EPS && e <= b + EPS) { swallowed.push(c); continue; }  // נבלע לגמרי → נמחק
    if (s < a - EPS && e > b + EPS) {                                    // הקטע נופל באמצע → פיצול
      const right = JSON.parse(JSON.stringify(c));
      right.id = uid('clip');
      const cutLeft = a - s, cutRight = b - s;
      c.duration = cutLeft;
      right.start = b;
      right.inPoint = c.inPoint + cutRight * (c.speed || 1);
      right.duration = e - b;
      right.tin = null;
      out.push(c, right);
      continue;
    }
    if (s < a - EPS) { c.duration = a - s; out.push(c); continue; }      // חותכים את הזנב
    // חותכים את הראש
    const delta = b - s;
    c.inPoint += delta * (c.speed || 1);
    c.start = b;
    c.duration = e - b;
    c.tin = null;
    out.push(c);
  }
  const slivers = out.filter((c) => c.duration <= MIN_CLIP / 2);
  track.clips = out.filter((c) => c.duration > MIN_CLIP / 2);
  sortTrack(track);
  noteRemoved(swallowed, 'overwritten');       // נדרס על ידי קליפ שהונח מעליו
  noteRemoved(slivers, 'too-short');
  if (swallowed.length) emitter.emit('overwrote', swallowed.length);
}

/** הוספת קליפ לערוץ בדריסה */
export function placeClip(track, clip) {
  clip.trackId = track.id;
  carve(track, clip.start, clipEnd(clip), clip.id);
  if (!track.clips.includes(clip)) track.clips.push(clip);
  sortTrack(track);
  validateTransitions(track);
  return clip;
}

/** פיצול קליפ בזמן t. מחזיר את החלק הימני או null */
export function splitClip(track, clip, t) {
  if (t <= clip.start + MIN_CLIP || t >= clipEnd(clip) - MIN_CLIP) return null;
  const right = JSON.parse(JSON.stringify(clip));
  right.id = uid('clip');
  const offset = t - clip.start;
  right.start = t;
  right.inPoint = clip.inPoint + offset * (clip.speed || 1);
  right.duration = clip.duration - offset;
  right.tin = null;
  right.aFadeIn = 0; right.vFadeIn = 0;
  clip.duration = offset;
  clip.aFadeOut = Math.min(clip.aFadeOut, clip.duration);
  clip.vFadeOut = Math.min(clip.vFadeOut, clip.duration);
  track.clips.push(right);
  sortTrack(track);
  return right;
}

/** חיתוך בכל הערוצים שאינם נעולים בנקודת הזמן t */
export function splitAll(t, onlySelected = false) {
  let n = 0;
  for (const track of state.proj.tracks) {
    if (track.locked) continue;
    const c = clipAt(track, t);
    if (!c) continue;
    if (onlySelected && state.selection.size && !state.selection.has(c.id)) continue;
    if (splitClip(track, c, t)) n++;
  }
  return n;
}

export function removeClips(ids) {
  const set = new Set(ids);
  const gone = [];
  for (const t of state.proj.tracks) {
    if (t.locked) continue;
    gone.push(...t.clips.filter((c) => set.has(c.id)));
    t.clips = t.clips.filter((c) => !set.has(c.id));
    validateTransitions(t);
  }
  noteRemoved(gone, 'deleted');
  ids.forEach((id) => state.selection.delete(id));
}

/** מחיקה עם סגירת החור: כל מה שאחרי הקליפ באותו ערוץ נגרר אחורה */
export function rippleDelete(ids) {
  const set = new Set(ids);
  for (const t of state.proj.tracks) {
    if (t.locked) continue;
    const doomed = t.clips.filter((c) => set.has(c.id)).sort((a, b) => b.start - a.start);
    noteRemoved(doomed, 'ripple-deleted');
    for (const d of doomed) {
      const gap = d.duration, after = clipEnd(d);
      t.clips = t.clips.filter((c) => c.id !== d.id);
      for (const c of t.clips) if (c.start >= after - EPS) c.start -= gap;
    }
    sortTrack(t);
    validateTransitions(t);
  }
  ids.forEach((id) => state.selection.delete(id));
}

/** הסרת קטע זמן מכל הערוצים וסגירת החור (ripple delete של טווח) */
export function rippleRange(a, b) {
  if (b <= a + EPS) return;
  const gap = b - a;
  for (const t of state.proj.tracks) {
    if (t.locked) continue;
    carve(t, a, b);
    for (const c of t.clips) if (c.start >= b - EPS) c.start -= gap;
    sortTrack(t);
    validateTransitions(t);
  }
}

/** סגירת חורים בערוץ: הצמדת כל הקליפים זה לזה מתחילת הערוץ */
export function closeGaps(track) {
  sortTrack(track);
  let t = 0;
  for (const c of track.clips) { c.start = t; t += c.duration; }
}

/* ─────────────── מעברים ─────────────── */

export const TRANSITIONS = [
  { id: 'dissolve',  name: 'מעבר הדרגתי' },
  { id: 'fadeblack', name: 'דרך שחור' },
  { id: 'fadewhite', name: 'דרך לבן' },
  { id: 'wipeleft',  name: 'מחיקה שמאלה' },
  { id: 'wiperight', name: 'מחיקה ימינה' },
  { id: 'wipeup',    name: 'מחיקה למעלה' },
  { id: 'slideleft', name: 'החלקה שמאלה' },
  { id: 'slideup',   name: 'החלקה למעלה' },
  { id: 'zoomin',    name: 'זום' },
  { id: 'circle',    name: 'עיגול' },
  { id: 'blinds',    name: 'תריסים' },
  { id: 'push',      name: 'דחיפה' },
];
export const transitionName = (id) => TRANSITIONS.find((t) => t.id === id)?.name || id;

/** המשך מותר למעבר בראש הקליפ */
export function maxTransition(track, clip) {
  const prev = prevAdjacent(track, clip);
  if (!prev) return 0;
  return Math.max(0, Math.min(clip.duration * 0.9, prev.duration * 0.9, 4));
}

export function setTransition(track, clip, type, dur) {
  const max = maxTransition(track, clip);
  if (max <= 0.02) return false;
  clip.tin = { type, dur: clamp(dur ?? Math.min(1, max), 0.06, max) };
  return true;
}

/** ניקוי מעברים שאיבדו את הקליפ הצמוד שלפניהם */
export function validateTransitions(track) {
  for (const c of track.clips) {
    if (!c.tin) continue;
    const max = maxTransition(track, c);
    if (max <= 0.02) c.tin = null;
    else c.tin.dur = Math.min(c.tin.dur, max);
  }
}

/* ─────────────── ערוצים ─────────────── */

export function addTrack(kind) {
  const same = state.proj.tracks.filter((t) => t.kind === kind);
  const t = newTrack(kind, `${kind === 'video' ? 'V' : 'A'}${same.length + 1}`);
  if (kind === 'video') {
    // ערוצי וידאו נערמים מלמעלה: החדש נכנס בראש הרשימה
    state.proj.tracks.unshift(t);
  } else {
    state.proj.tracks.push(t);
  }
  return t;
}

export function removeTrack(id) {
  const i = state.proj.tracks.findIndex((t) => t.id === id);
  if (i < 0) return;
  const kind = state.proj.tracks[i].kind;
  if (state.proj.tracks.filter((t) => t.kind === kind).length <= 1) return; // תמיד נשאר אחד
  state.proj.tracks[i].clips.forEach((c) => state.selection.delete(c.id));
  state.proj.tracks.splice(i, 1);
}

/** הערוץ הראשון מסוג נתון שבו יש מקום פנוי בטווח, אחרת חדש */
export function findFreeTrack(kind, start, end) {
  const cands = state.proj.tracks.filter((t) => t.kind === kind && !t.locked);
  for (let i = cands.length - 1; i >= 0; i--) {
    const t = cands[i];
    const busy = t.clips.some((c) => c.start < end - EPS && clipEnd(c) > start + EPS);
    if (!busy) return t;
  }
  return addTrack(kind);
}

/* ─────────────── בחירה וסמן ─────────────── */

export function select(ids, additive = false) {
  if (!additive) state.selection.clear();
  (Array.isArray(ids) ? ids : [ids]).forEach((i) => i && state.selection.add(i));
  state.selTransition = null;
  emitter.emit('selection');
}

export function toggleSelect(id) {
  if (state.selection.has(id)) state.selection.delete(id); else state.selection.add(id);
  state.selTransition = null;
  emitter.emit('selection');
}

export function clearSelection() {
  state.selection.clear();
  state.selTransition = null;
  emitter.emit('selection');
}

export function selectTransition(clipId) {
  state.selection.clear();
  state.selTransition = clipId;
  emitter.emit('selection');
}

export function setPlayhead(t, why = '') {
  const d = Math.max(duration(), 0);
  const v = clamp(snapFrame(t, state.proj.fps), 0, Math.max(d, 0.0001) + 5);
  if (Math.abs(v - state.playhead) < 1e-9) return;
  state.playhead = v;
  emitter.emit('playhead', why);
}

/* ─────────────── סריאליזציה ─────────────── */

export function serialize() {
  return {
    app: 'vedit', version: 1,
    name: state.proj.name, width: state.proj.width, height: state.proj.height, fps: state.proj.fps,
    media: state.proj.media.map((m) => ({
      id: m.id, name: m.name, type: m.type, duration: m.duration,
      width: m.width, height: m.height, hasAudio: m.hasAudio, size: m.size,
    })),
    tracks: state.proj.tracks,
    inPoint: state.inPoint, outPoint: state.outPoint,
  };
}

export function load(data) {
  if (!data || data.app !== 'vedit') throw new Error('קובץ פרויקט לא מוכר');
  state.proj = {
    name: data.name || 'פרויקט', width: data.width || 1920, height: data.height || 1080,
    fps: data.fps || 30, media: data.media || [], tracks: data.tracks || [],
  };
  if (!state.proj.tracks.length) state.proj.tracks = [newTrack('video', 'V1'), newTrack('audio', 'A1')];
  state.selection.clear();
  state.selTransition = null;
  state.playhead = 0;
  state.inPoint = data.inPoint ?? null;
  state.outPoint = data.outPoint ?? null;
  clearHistory();
  emitter.emit('change', 'load');
  emitter.emit('history');
}

export function reset() {
  state.proj = blankProject();
  state.selection.clear();
  state.selTransition = null;
  state.playhead = 0; state.inPoint = null; state.outPoint = null;
  clearHistory();
  emitter.emit('change', 'reset');
  emitter.emit('history');
}
