/* timeline.js: ציור הטיימליין וכל אינטראקציות העריכה שבו */

import { $, $$, clamp, drag, tc, shortDur, snapFrame, toast } from './util.js';
import {
  state, proj, fps, duration, allClips, trackById, clipById, trackOfClip, mediaById,
  clipEnd, sortTrack, clipAt, prevAdjacent, EPS, MIN_CLIP,
  beginChange, commit, silent, emitter, select, toggleSelect, clearSelection,
  selectTransition, setPlayhead, placeClip, carve, splitClip, splitAll, removeClips,
  rippleDelete, closeGaps, sourceRoom, validateTransitions, maxTransition, setTransition,
  transitionName, TRANSITIONS, addTrack, removeTrack, makeClip, findFreeTrack,
} from './state.js';
import { rt } from './media.js';

const SNAP_PX = 8;
const RULER_H = 26;          // חייב להתאים ל---ruler-h שב-CSS
const MIN_ZOOM = 2, MAX_ZOOM = 900;

let engine = null;
let els = {};
let dragCtx = null;          // מצב גרירה פעיל
export let pendingDrop = null;   // פריט מדיה/מעבר שנגרר מהפאנל השמאלי

export const setPendingDrop = (v) => { pendingDrop = v; };

/* ─────────── מיפוי זמן ↔ פיקסלים ─────────── */

export const t2x = (t) => t * state.zoom;
export const x2t = (x) => x / state.zoom;

function contentWidth() {
  const scrollW = els.scroll?.clientWidth || 800;
  return Math.max(duration() * state.zoom + 300, scrollW);
}

/* ─────────── אתחול ─────────── */

export function initTimeline(eng) {
  engine = eng;
  els = {
    scroll: $('#tlScroll'),
    inner: $('#tlInner'),
    ruler: $('#ruler'),
    tracks: $('#tracks'),
    heads: $('#trackHeads'),
    headsWrap: $('.tl-heads'),
    playhead: $('#playhead'),
    band: $('#inoutBand'),
    ghost: $('#dropGhost'),
  };

  els.scroll.addEventListener('scroll', () => {
    els.headsWrap.scrollTop = els.scroll.scrollTop;
    renderRuler();
  });

  // סרגל הזמן: לחיצה וגרירה מזיזות את הסמן
  els.ruler.addEventListener('pointerdown', (e) => {
    scrubFrom(e);
  });
  $('.ph-head').addEventListener('pointerdown', (e) => { e.stopPropagation(); scrubFrom(e); });

  // לחיצה על אזור ריק
  els.tracks.addEventListener('pointerdown', onTracksPointerDown);
  els.tracks.addEventListener('dblclick', onTracksDblClick);
  els.tracks.addEventListener('contextmenu', onContextMenu);

  // גלגלת: זום עם Ctrl, גלילה אופקית עם Shift
  els.scroll.addEventListener('wheel', (e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const rect = els.scroll.getBoundingClientRect();
      const anchorT = x2t(e.clientX - rect.left + els.scroll.scrollLeft);
      zoomBy(e.deltaY < 0 ? 1.18 : 1 / 1.18, anchorT);
    } else if (e.shiftKey) {
      e.preventDefault();
      els.scroll.scrollLeft += e.deltaY;
    }
  }, { passive: false });

  // גרירה מהפאנל השמאלי אל הטיימליין
  els.scroll.addEventListener('dragover', onDragOver);
  els.scroll.addEventListener('dragleave', () => { els.ghost.classList.add('hidden'); });
  els.scroll.addEventListener('drop', onDrop);

  emitter.on('change', () => { renderAll(); });
  emitter.on('selection', () => { refreshSelection(); });
  emitter.on('assets', () => { paintAllClipContent(); });
  emitter.on('playhead', () => { updatePlayhead(); });

  renderAll();
}

/* ─────────── ציור ─────────── */

export function renderAll() {
  renderHeads();
  renderTracks();
  renderRuler();
  updatePlayhead();
  updateBand();
}

function renderHeads() {
  const host = els.heads;
  host.innerHTML = '';
  for (const t of proj().tracks) {
    const d = document.createElement('div');
    d.className = `thead ${t.kind}${state.activeTrack === t.id ? ' active' : ''}`;
    d.style.height = `${t.height}px`;
    d.dataset.track = t.id;
    d.innerHTML = `
      <span class="tname">${t.name}</span>
      ${t.kind === 'video' ? `<button data-act="hide" class="${t.hidden ? 'on' : ''}" title="הסתרת הערוץ">${t.hidden ? '🚫' : '👁'}</button>` : ''}
      <button data-act="mute" class="${t.muted ? 'on' : ''}" title="השתקה">${t.muted ? '🔇' : '🔊'}</button>
      <button data-act="lock" class="${t.locked ? 'on' : ''}" title="נעילה">${t.locked ? '🔒' : '🔓'}</button>`;
    d.addEventListener('pointerdown', () => { state.activeTrack = t.id; renderHeads(); });
    d.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      openMenu(e.clientX, e.clientY, [
        { label: `שינוי שם "${t.name}"`, fn: () => {
          const n = prompt('שם הערוץ:', t.name);
          if (n) { beginChange(); t.name = n; commit('rename-track'); }
        } },
        { label: 'ניקוי הערוץ', fn: () => { beginChange(); t.clips = []; commit('clear-track'); } },
        { label: 'סגירת חורים בערוץ', fn: () => { beginChange(); closeGaps(t); commit('close-gaps'); } },
        { sep: true },
        { label: 'מחיקת הערוץ', fn: () => { beginChange(); removeTrack(t.id); commit('remove-track'); } },
      ]);
    });
    d.querySelectorAll('button').forEach((b) => {
      b.addEventListener('click', (e) => {
        e.stopPropagation();
        beginChange();
        const a = b.dataset.act;
        if (a === 'mute') t.muted = !t.muted;
        if (a === 'hide') t.hidden = !t.hidden;
        if (a === 'lock') t.locked = !t.locked;
        commit('track-flag');
      });
    });
    host.appendChild(d);
  }
}

function renderTracks() {
  const host = els.tracks;
  host.innerHTML = '';
  els.inner.style.width = `${contentWidth()}px`;

  for (const t of proj().tracks) {
    const tr = document.createElement('div');
    tr.className = `track ${t.kind}`;
    tr.style.height = `${t.height}px`;
    tr.dataset.track = t.id;
    for (const c of t.clips) tr.appendChild(buildClip(c, t));
    for (const c of t.clips) { const x = buildTransition(c, t); if (x) tr.appendChild(x); }
    host.appendChild(tr);
  }
  paintAllClipContent();
}

function buildClip(c, track) {
  const d = document.createElement('div');
  d.className = `clip ${c.kind}${state.selection.has(c.id) ? ' sel' : ''}`;
  d.dataset.clip = c.id;
  d.style.left = `${t2x(c.start)}px`;
  d.style.width = `${Math.max(2, t2x(c.duration))}px`;
  if (track.locked) d.style.opacity = '.6';

  const w = Math.max(2, t2x(c.duration));
  const h = track.height - 4;

  const cv = document.createElement('canvas');
  cv.className = c.kind === 'audio' ? 'cwave' : 'cthumbs';
  cv.style.width = '100%'; cv.style.height = '100%';
  cv.width = Math.min(3000, Math.max(1, Math.round(w)));
  cv.height = Math.max(1, Math.round(h));
  d.appendChild(cv);

  const label = document.createElement('div');
  label.className = 'clabel';
  const sp = (c.speed && c.speed !== 1) ? ` · ${c.speed}×` : '';
  label.textContent = `${c.name}${sp}`;
  d.appendChild(label);

  if ((c.mute || c.volume === 0) && c.kind !== 'title') {
    const m = document.createElement('div');
    m.className = 'muted-badge'; m.textContent = '🔇';
    d.appendChild(m);
  }

  // סימוני דעיכה
  if (c.aFadeIn > 0 || c.vFadeIn > 0) {
    const f = document.createElement('div');
    f.className = 'fade in';
    f.style.width = `${t2x(Math.max(c.aFadeIn, c.vFadeIn))}px`;
    d.appendChild(f);
  }
  if (c.aFadeOut > 0 || c.vFadeOut > 0) {
    const f = document.createElement('div');
    f.className = 'fade out';
    f.style.width = `${t2x(Math.max(c.aFadeOut, c.vFadeOut))}px`;
    d.appendChild(f);
  }

  // ידיות חיתוך
  const hl = document.createElement('div'); hl.className = 'handle l';
  const hr = document.createElement('div'); hr.className = 'handle r';
  d.appendChild(hl); d.appendChild(hr);
  hl.addEventListener('pointerdown', (e) => { e.stopPropagation(); startTrim(e, c, track, 'l'); });
  hr.addEventListener('pointerdown', (e) => { e.stopPropagation(); startTrim(e, c, track, 'r'); });

  d.addEventListener('pointerdown', (e) => onClipPointerDown(e, c, track));

  return d;
}

/** אלמנט המעבר יושב על הערוץ (לא בתוך הקליפ) כדי שיוכל לחרוג מגבולותיו */
function buildTransition(c, track) {
  if (!c.tin) return null;
  const tw = t2x(c.tin.dur);
  const tEl = document.createElement('div');
  tEl.className = `trans${state.selTransition === c.id ? ' sel' : ''}`;
  tEl.style.left = `${t2x(c.start)}px`;
  tEl.style.width = `${Math.max(6, tw)}px`;
  tEl.dataset.trans = c.id;
  tEl.innerHTML = `<span>${tw > 46 ? transitionName(c.tin.type) : '⇄'}</span>`;
  tEl.title = `${transitionName(c.tin.type)} · ${c.tin.dur.toFixed(2)} שנ׳. גררו לשינוי האורך, לחיצה ימנית להחלפה`;
  tEl.addEventListener('pointerdown', (e) => { e.stopPropagation(); startTransResize(e, c, track); });
  tEl.addEventListener('contextmenu', (e) => {
    e.preventDefault(); e.stopPropagation();
    transitionMenu(e.clientX, e.clientY, c, track);
  });
  return tEl;
}

/* ─────────── תוכן הקליפ: פילמסטריפ וצורת גל ─────────── */

function paintAllClipContent() {
  for (const t of proj().tracks) {
    for (const c of t.clips) paintClipContent(c, t);
  }
}

function paintClipContent(c, track) {
  const el = els.tracks.querySelector(`.clip[data-clip="${c.id}"]`);
  if (!el) return;
  const cv = el.querySelector('canvas');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);

  if (c.kind === 'title') {
    ctx.fillStyle = 'rgba(255,255,255,.14)';
    ctx.fillRect(0, 0, W, H);
    return;
  }

  const m = mediaById(c.mediaId);
  const r = m && rt(m.id);
  if (!m || !r) return;

  const srcSpan = c.duration * (c.speed || 1);
  const showThumbs = (c.kind === 'video' || c.kind === 'image') && track.kind === 'video';
  const showWave = m.hasAudio && r.peaks;

  if (showThumbs) {
    if (m.type === 'image' && r.image) {
      const ih = H, iw = ih * (r.image.naturalWidth / Math.max(1, r.image.naturalHeight));
      for (let x = 0; x < W; x += iw) ctx.drawImage(r.image, x, 0, iw, ih);
    } else if (r.thumbs?.length) {
      const n = r.thumbs.length;
      const tw = Math.max(28, H * (r.thumbAspect || 1.6));
      for (let x = 0; x < W; x += tw) {
        const srcT = c.inPoint + (x / W) * srcSpan;
        const i = clamp(Math.floor((srcT / Math.max(0.001, m.duration)) * n), 0, n - 1);
        const bmp = r.thumbs[i];
        if (bmp) ctx.drawImage(bmp, x, 0, tw, H);
      }
      ctx.fillStyle = 'rgba(20,40,70,.18)';
      ctx.fillRect(0, 0, W, H);
    }
  }

  if (showWave) {
    const peaks = r.peaks, pps = r.peaksPerSec || 120;
    const baseY = showThumbs ? H : H / 2;
    const amp = showThumbs ? H * 0.32 : H * 0.44;
    ctx.fillStyle = showThumbs ? 'rgba(160,220,255,.55)' : 'rgba(190,255,220,.75)';
    for (let x = 0; x < W; x++) {
      const t0 = c.inPoint + (x / W) * srcSpan;
      const t1 = c.inPoint + ((x + 1) / W) * srcSpan;
      let peak = 0;
      const i0 = Math.floor(t0 * pps), i1 = Math.max(i0 + 1, Math.floor(t1 * pps));
      for (let i = i0; i < i1 && i < peaks.length; i++) if (peaks[i] > peak) peak = peaks[i];
      const hh = Math.max(0.5, peak * amp);
      if (showThumbs) ctx.fillRect(x, baseY - hh, 1, hh);
      else ctx.fillRect(x, baseY - hh, 1, hh * 2);
    }
  }
}

/* ─────────── סרגל הזמן ─────────── */

function renderRuler() {
  const cv = els.ruler;
  const vw = els.scroll.clientWidth;
  const dpr = window.devicePixelRatio || 1;
  // מציירים רק את החלק הנראה ומזיזים אותו עם הגלילה, כדי לא להחזיק קנבס ענק.
  // חשוב לא לחרוג מרוחב התוכן, אחרת אזור הגלילה גדל בכל גלילה.
  const cssW = Math.max(50, Math.min(contentWidth() - els.scroll.scrollLeft, vw + 200));
  cv.width = Math.round(cssW * dpr);
  cv.height = Math.round(RULER_H * dpr);
  cv.style.width = `${cssW}px`;
  cv.style.height = `${RULER_H}px`;      // חובה: אחרת בצגי Retina הסרגל תופס גובה כפול
  // הסרגל "צף" עם הגלילה כדי לא לצייר קנבס ענק
  cv.style.transform = `translateX(${els.scroll.scrollLeft}px)`;

  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, RULER_H);
  ctx.fillStyle = '#24272f';
  ctx.fillRect(0, 0, cssW, RULER_H);

  const t0 = x2t(els.scroll.scrollLeft);
  const t1 = x2t(els.scroll.scrollLeft + cssW);
  const step = niceStep(state.zoom);
  const minor = step / 5;

  ctx.strokeStyle = '#3b404a';
  ctx.fillStyle = '#7d8592';
  ctx.font = '10px -apple-system, "Segoe UI", sans-serif';
  ctx.textBaseline = 'alphabetic';
  ctx.beginPath();
  for (let t = Math.floor(t0 / minor) * minor; t < t1; t += minor) {
    const x = Math.round(t2x(t) - els.scroll.scrollLeft) + 0.5;
    const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
    ctx.moveTo(x, major ? 8 : 18);
    ctx.lineTo(x, RULER_H);
  }
  ctx.stroke();

  for (let t = Math.ceil(t0 / step) * step; t < t1; t += step) {
    const x = Math.round(t2x(t) - els.scroll.scrollLeft);
    ctx.fillText(labelFor(t, step), x + 3, 12);
  }

  // סימוני כניסה/יציאה
  if (state.inPoint != null || state.outPoint != null) {
    ctx.fillStyle = '#4a9eff';
    if (state.inPoint != null) mark(ctx, t2x(state.inPoint) - els.scroll.scrollLeft, true);
    if (state.outPoint != null) mark(ctx, t2x(state.outPoint) - els.scroll.scrollLeft, false);
  }
}

function mark(ctx, x, isIn) {
  ctx.beginPath();
  if (isIn) { ctx.moveTo(x, 14); ctx.lineTo(x + 8, 20); ctx.lineTo(x, RULER_H); }
  else { ctx.moveTo(x, 14); ctx.lineTo(x - 8, 20); ctx.lineTo(x, RULER_H); }
  ctx.closePath(); ctx.fill();
}

function niceStep(zoom) {
  const targets = [1 / 30, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800, 3600];
  for (const s of targets) if (s * zoom >= 64) return s;
  return targets[targets.length - 1];
}

function labelFor(t, step) {
  if (step < 1) return `${t.toFixed(1)}s`;
  return shortDur(t);
}

/* ─────────── סמן הזמן ─────────── */

export function updatePlayhead() {
  els.playhead.style.left = `${t2x(state.playhead)}px`;
  const cur = $('#tcCur');
  if (cur) cur.textContent = tc(state.playhead, fps());
}

export function updateBand() {
  const a = state.inPoint, b = state.outPoint;
  if (a == null && b == null) { els.band.classList.add('hidden'); renderRuler(); return; }
  const s = a ?? 0, e = b ?? duration();
  els.band.classList.remove('hidden');
  els.band.style.left = `${t2x(s)}px`;
  els.band.style.width = `${Math.max(1, t2x(e - s))}px`;
  renderRuler();
}

export function ensureVisible(t) {
  const x = t2x(t);
  const sl = els.scroll.scrollLeft, vw = els.scroll.clientWidth;
  if (x < sl + 40) els.scroll.scrollLeft = Math.max(0, x - 60);
  else if (x > sl + vw - 60) els.scroll.scrollLeft = x - vw * 0.75;
}

function scrubFrom(e) {
  const rect = els.inner.getBoundingClientRect();
  const move = (cx) => {
    let t = x2t(cx - rect.left);
    t = clamp(t, 0, Math.max(duration(), 0.001));
    if (state.snap) t = snapTime(t, [], true);
    setPlayhead(t, 'scrub');
    engine?.seek(state.playhead, true);
  };
  move(e.clientX);
  drag(e, { onMove: (dx, dy, ev) => move(ev.clientX), cursor: 'ew-resize' });
}

/* ─────────── הצמדה ─────────── */

function snapTargets(excludeIds = new Set()) {
  const pts = [0];
  for (const t of proj().tracks) {
    for (const c of t.clips) {
      if (excludeIds.has(c.id)) continue;
      pts.push(c.start, clipEnd(c));
    }
  }
  pts.push(state.playhead);
  if (state.inPoint != null) pts.push(state.inPoint);
  if (state.outPoint != null) pts.push(state.outPoint);
  return pts;
}

function snapTime(t, excludeIds = [], skipPlayhead = false) {
  if (!state.snap) return snapFrame(t, fps());
  const ex = new Set(excludeIds);
  const tol = SNAP_PX / state.zoom;
  let best = null, bestD = tol;
  for (const p of snapTargets(ex)) {
    if (skipPlayhead && Math.abs(p - state.playhead) < 1e-9) continue;
    const d = Math.abs(p - t);
    if (d < bestD) { bestD = d; best = p; }
  }
  return best != null ? best : snapFrame(t, fps());
}

/* ─────────── אינטראקציות בקליפ ─────────── */

function onClipPointerDown(e, c, track) {
  if (e.button === 2) return;                    // תפריט הקשר מטופל בנפרד
  e.stopPropagation();

  if (state.tool === 'razor') {
    const rect = els.inner.getBoundingClientRect();
    const t = snapTime(x2t(e.clientX - rect.left), [c.id]);
    beginChange();
    const r = splitClip(track, c, t);
    commit('split');
    if (r) select(r.id);
    return;
  }
  if (state.tool === 'hand') { startPan(e); return; }
  if (track.locked) { toast('הערוץ נעול', 'err'); return; }

  const additive = e.shiftKey || e.ctrlKey || e.metaKey;
  if (additive) toggleSelect(c.id);
  else if (!state.selection.has(c.id)) select(c.id);

  state.activeTrack = track.id;
  startMove(e, c, track);
}

function startMove(e, clip, track) {
  const ids = state.selection.has(clip.id) ? [...state.selection] : [clip.id];
  const items = ids.map((id) => {
    const c = clipById(id), tr = trackOfClip(id);
    return c && tr && !tr.locked ? { c, tr, start0: c.start, trackIdx0: proj().tracks.indexOf(tr) } : null;
  }).filter(Boolean);
  if (!items.length) return;

  const rowTops = rowOffsets();
  const y0 = e.clientY;
  const startRow = rowAtClientY(y0, rowTops);
  let dtApplied = 0, trackShift = 0, didMove = false;
  const excl = new Set(items.map((i) => i.c.id));

  drag(e, {
    cursor: 'grabbing',
    onMove: (dx, dy, ev, moved) => {
      if (!moved) return;
      if (!didMove) { didMove = true; beginChange(); items.forEach((i) => i.el = els.tracks.querySelector(`.clip[data-clip="${i.c.id}"]`)); }
      let dt = x2t(dx);
      // הצמדה: בודקים את קצוות הקליפ הראשי
      const lead = items[0];
      const wanted = lead.start0 + dt;
      const snappedStart = snapTime(wanted, [...excl]);
      const snappedEnd = snapTime(wanted + lead.c.duration, [...excl]) - lead.c.duration;
      if (Math.abs(snappedStart - wanted) <= Math.abs(snappedEnd - wanted)) dt = snappedStart - lead.start0;
      else dt = snappedEnd - lead.start0;

      const minStart = Math.min(...items.map((i) => i.start0));
      if (minStart + dt < 0) dt = -minStart;
      dtApplied = dt;

      // מעבר בין ערוצים: לפי הערוץ שנמצא ממש מתחת לסמן
      trackShift = rowAtClientY(ev.clientY, rowTops) - startRow;
      for (const it of items) {
        const targetIdx = clamp(it.trackIdx0 + trackShift, 0, proj().tracks.length - 1);
        const target = proj().tracks[targetIdx];
        const ok = target && target.kind === kindTrackFor(it.c) && !target.locked;
        it.targetTrack = ok ? target : it.tr;
        if (it.el) {
          it.el.classList.add('dragging');
          it.el.style.left = `${t2x(it.start0 + dt)}px`;
          const row = proj().tracks.indexOf(it.targetTrack);
          it.el.style.transform = `translateY(${rowTops[row] - rowTops[it.trackIdx0]}px)`;
        }
      }
    },
    onEnd: (moved) => {
      items.forEach((i) => { if (i.el) { i.el.classList.remove('dragging'); i.el.style.transform = ''; } });
      if (!moved || !didMove) { commit('noop'); return; }
      // מוציאים את כולם ומניחים מחדש (דריסה)
      for (const it of items) it.tr.clips = it.tr.clips.filter((x) => x.id !== it.c.id);
      for (const it of items) {
        it.c.start = Math.max(0, snapFrame(it.start0 + dtApplied, fps()));
        it.c.tin = null;
        placeClip(it.targetTrack || it.tr, it.c);
      }
      proj().tracks.forEach(validateTransitions);
      commit('move');
    },
  });
}

const kindTrackFor = (c) => (c.kind === 'audio' ? 'audio' : 'video');

function rowOffsets() {
  const tops = [];
  let y = 0;
  for (const t of proj().tracks) { tops.push(y); y += t.height; }
  return tops;
}

/** אינדקס הערוץ שנמצא מתחת לנקודה על המסך */
function rowAtClientY(clientY, tops) {
  const rect = els.tracks.getBoundingClientRect();
  const y = clientY - rect.top;
  const tracks = proj().tracks;
  if (y < 0) return 0;
  for (let i = 0; i < tracks.length; i++) {
    if (y >= tops[i] && y < tops[i] + tracks[i].height) return i;
  }
  return tracks.length - 1;
}

function startTrim(e, clip, track, side) {
  if (track.locked) return;
  select(clip.id);
  const start0 = clip.start, dur0 = clip.duration, in0 = clip.inPoint;
  const room = sourceRoom(clip);                     // כמה מקור נשאר אחרי inPoint
  const headRoom = clip.kind === 'title' || (mediaById(clip.mediaId)?.type === 'image')
    ? Infinity : in0 / (clip.speed || 1);            // כמה אפשר להאריך אחורה
  const el = els.tracks.querySelector(`.clip[data-clip="${clip.id}"]`);
  let began = false;
  let newStart = start0, newDur = dur0, newIn = in0;

  drag(e, {
    cursor: 'ew-resize',
    onMove: (dx, dy, ev, moved) => {
      if (!moved) return;
      if (!began) { began = true; beginChange(); }
      const dt = x2t(dx);
      if (side === 'l') {
        let s = snapTime(start0 + dt, [clip.id]);
        s = clamp(s, Math.max(0, start0 - headRoom), start0 + dur0 - MIN_CLIP);
        newStart = s;
        newDur = dur0 - (s - start0);
        newIn = in0 + (s - start0) * (clip.speed || 1);
      } else {
        let e2 = snapTime(start0 + dur0 + dt, [clip.id]);
        e2 = clamp(e2, start0 + MIN_CLIP, start0 + (isFinite(room) ? room : 1e6));
        newDur = e2 - start0;
      }
      if (el) {
        el.style.left = `${t2x(newStart)}px`;
        el.style.width = `${Math.max(2, t2x(newDur))}px`;
      }
    },
    onEnd: (moved) => {
      if (!moved || !began) { commit('noop'); return; }
      clip.start = snapFrame(newStart, fps());
      clip.duration = Math.max(MIN_CLIP, snapFrame(newDur, fps()));
      clip.inPoint = Math.max(0, newIn);
      clip.aFadeIn = Math.min(clip.aFadeIn, clip.duration / 2);
      clip.aFadeOut = Math.min(clip.aFadeOut, clip.duration / 2);
      clip.vFadeIn = Math.min(clip.vFadeIn, clip.duration / 2);
      clip.vFadeOut = Math.min(clip.vFadeOut, clip.duration / 2);
      // דריסת מה שנקלע לדרך
      track.clips = track.clips.filter((x) => x.id !== clip.id);
      placeClip(track, clip);
      commit('trim');
    },
  });
}

function startTransResize(e, clip, track) {
  selectTransition(clip.id);
  const d0 = clip.tin.dur;
  const max = maxTransition(track, clip);
  let began = false, nd = d0;
  drag(e, {
    cursor: 'ew-resize',
    onMove: (dx, dy, ev, moved) => {
      if (!moved) return;
      if (!began) { began = true; beginChange(); }
      nd = clamp(d0 + x2t(dx) * 2, 0.06, max);
      const el = els.tracks.querySelector(`.trans[data-trans="${clip.id}"]`);
      if (el) el.style.width = `${Math.max(6, t2x(nd))}px`;
    },
    onEnd: (moved) => {
      if (!moved || !began) { commit('noop'); return; }
      clip.tin.dur = nd;
      commit('trans-dur');
    },
  });
}

/* ─────────── לחיצות על אזור ריק ─────────── */

function onTracksPointerDown(e) {
  if (e.button === 2) return;
  const trEl = e.target.closest('.track');
  if (state.tool === 'hand') { startPan(e); return; }
  if (trEl) state.activeTrack = trEl.dataset.track;

  if (state.tool === 'razor') {
    const rect = els.inner.getBoundingClientRect();
    const t = x2t(e.clientX - rect.left);
    beginChange();
    const n = splitAll(t);
    commit('split-all');
    if (!n) toast('אין קליפ לחתוך כאן');
    return;
  }
  if (!e.shiftKey) clearSelection();
  startMarquee(e);
  renderHeads();
}

function onTracksDblClick(e) {
  const clipEl = e.target.closest('.clip');
  if (!clipEl) return;
  const c = clipById(clipEl.dataset.clip);
  if (c) { select(c.id); setPlayhead(c.start); engine?.seek(state.playhead); }
}

function startMarquee(e) {
  const rect = els.inner.getBoundingClientRect();
  const x0 = e.clientX - rect.left, y0 = e.clientY - rect.top;
  const box = document.createElement('div');
  box.className = 'marquee';
  let added = false;

  drag(e, {
    onMove: (dx, dy, ev, moved) => {
      if (!moved) return;
      if (!added) { els.inner.appendChild(box); added = true; }
      const x1 = ev.clientX - rect.left, y1 = ev.clientY - rect.top;
      const l = Math.min(x0, x1), t = Math.min(y0, y1);
      box.style.left = `${l}px`; box.style.top = `${t}px`;
      box.style.width = `${Math.abs(x1 - x0)}px`; box.style.height = `${Math.abs(y1 - y0)}px`;

      const tops = rowOffsets();
      const ruler = RULER_H;
      const hit = [];
      proj().tracks.forEach((tr, i) => {
        const ty = tops[i] + ruler, tb = ty + tr.height;
        if (tb < t || ty > t + Math.abs(y1 - y0)) return;
        for (const c of tr.clips) {
          const cl = t2x(c.start), cr = t2x(clipEnd(c));
          if (cr > l && cl < l + Math.abs(x1 - x0)) hit.push(c.id);
        }
      });
      select(hit, ev.shiftKey);
    },
    onEnd: () => { box.remove(); },
  });
}

function startPan(e) {
  const sl0 = els.scroll.scrollLeft, st0 = els.scroll.scrollTop;
  drag(e, {
    cursor: 'grabbing',
    onMove: (dx, dy) => {
      els.scroll.scrollLeft = sl0 - dx;
      els.scroll.scrollTop = st0 - dy;
    },
  });
}

/* ─────────── תפריט הקשר ─────────── */

function onContextMenu(e) {
  const clipEl = e.target.closest('.clip');
  const transEl = e.target.closest('.trans');
  e.preventDefault();
  if (transEl) {
    const c = clipById(transEl.dataset.trans);
    transitionMenu(e.clientX, e.clientY, c, trackOfClip(c.id));
    return;
  }
  if (!clipEl) {
    openMenu(e.clientX, e.clientY, [
      { label: 'הדבקת סמן כאן', fn: () => {
        const rect = els.inner.getBoundingClientRect();
        setPlayhead(x2t(e.clientX - rect.left)); engine?.seek(state.playhead);
      } },
      { label: 'בחירת הכל', fn: () => select(allClips().map((c) => c.id)) },
    ]);
    return;
  }
  const c = clipById(clipEl.dataset.clip);
  const track = trackOfClip(c.id);
  if (!state.selection.has(c.id)) select(c.id);

  const items = [
    { label: '✂ חיתוך בנקודת הסמן', fn: () => {
      beginChange(); const r = splitClip(track, c, state.playhead); commit('split');
      if (r) select(r.id); else toast('הסמן לא נמצא בתוך הקליפ');
    } },
    { label: '⧉ שכפול', fn: () => duplicateSelection() },
    { sep: true },
    { label: '🗑 מחיקה', fn: () => { beginChange(); removeClips([...state.selection]); commit('delete'); } },
    { label: '⇤ מחיקה וסגירת החור', fn: () => { beginChange(); rippleDelete([...state.selection]); commit('ripple'); } },
    { sep: true },
    { label: 'מעבר בהתחלה ▸ מעבר הדרגתי', fn: () => applyTransition(c, track, 'dissolve') },
    { label: 'עוד מעברים…', fn: () => transitionMenu(e.clientX, e.clientY, c, track) },
  ];
  if (c.kind !== 'title') {
    items.push({ sep: true },
      { label: c.mute ? '🔊 ביטול השתקה' : '🔇 השתקת הקליפ', fn: () => { beginChange(); c.mute = !c.mute; commit('mute'); } });
  }
  openMenu(e.clientX, e.clientY, items);
}

function transitionMenu(x, y, clip, track) {
  const max = maxTransition(track, clip);
  if (max <= 0.02) { toast('צריך קליפ צמוד לפני כדי לשים מעבר', 'err'); return; }
  const items = TRANSITIONS.map((t) => ({
    label: (clip.tin?.type === t.id ? '✓ ' : '') + t.name,
    fn: () => applyTransition(clip, track, t.id),
  }));
  if (clip.tin) {
    items.push({ sep: true }, { label: '✕ הסרת המעבר', fn: () => { beginChange(); clip.tin = null; commit('trans-remove'); } });
    items.push({ label: 'משך: 0.3 שנ׳', fn: () => setTransDur(clip, track, 0.3) });
    items.push({ label: 'משך: 1 שנ׳', fn: () => setTransDur(clip, track, 1) });
    items.push({ label: 'משך: 2 שנ׳', fn: () => setTransDur(clip, track, 2) });
  }
  openMenu(x, y, items);
}

function setTransDur(clip, track, d) {
  beginChange();
  clip.tin.dur = Math.min(d, maxTransition(track, clip));
  commit('trans-dur');
}

export function applyTransition(clip, track, type, dur) {
  beginChange();
  const ok = setTransition(track, clip, type, dur);
  commit('transition');
  if (!ok) toast('אי אפשר לשים כאן מעבר: צריך שני קליפים צמודים', 'err');
  else selectTransition(clip.id);
  return ok;
}

export function openMenu(x, y, items) {
  const menu = $('#ctxMenu');
  menu.innerHTML = '';
  for (const it of items) {
    if (it.sep) { const s = document.createElement('div'); s.className = 'ctx-sep'; menu.appendChild(s); continue; }
    if (it.header) { const h = document.createElement('div'); h.className = 'ctx-label'; h.textContent = it.header; menu.appendChild(h); continue; }
    const b = document.createElement('button');
    b.textContent = it.label;
    b.addEventListener('click', () => { closeMenu(); it.fn?.(); });
    menu.appendChild(b);
  }
  menu.classList.remove('hidden');
  const r = menu.getBoundingClientRect();
  menu.style.left = `${Math.min(x, innerWidth - r.width - 8)}px`;
  menu.style.top = `${Math.min(y, innerHeight - r.height - 8)}px`;
  setTimeout(() => {
    window.addEventListener('pointerdown', closeMenu, { once: true });
    window.addEventListener('blur', closeMenu, { once: true });
  }, 0);
}

export function closeMenu() { $('#ctxMenu').classList.add('hidden'); }

/* ─────────── גרירה מהפאנל ─────────── */

function dropInfo(e) {
  const rect = els.inner.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top - RULER_H;    // מינוס גובה הסרגל
  let t = Math.max(0, x2t(x));
  const tops = rowOffsets();
  let idx = -1;
  for (let i = 0; i < proj().tracks.length; i++) {
    if (y >= tops[i] && y < tops[i] + proj().tracks[i].height) { idx = i; break; }
  }
  return { t, trackIdx: idx, track: idx >= 0 ? proj().tracks[idx] : null, y, tops };
}

function onDragOver(e) {
  if (!pendingDrop) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
  const info = dropInfo(e);

  if (pendingDrop.kind === 'transition') {
    // מסמנים את הגבול הקרוב ביותר
    const target = nearestBoundary(info);
    els.ghost.classList.toggle('hidden', !target);
    if (target) {
      els.ghost.style.left = `${t2x(target.clip.start) - 12}px`;
      els.ghost.style.top = `${RULER_H + target.top}px`;
      els.ghost.style.width = '24px';
      els.ghost.style.height = `${target.track.height}px`;
    }
    return;
  }

  const m = pendingDrop.media;
  const dur = m.type === 'image' ? 5 : m.duration;
  let t = state.snap ? snapTime(info.t, []) : info.t;
  els.ghost.classList.remove('hidden');
  els.ghost.style.left = `${t2x(t)}px`;
  els.ghost.style.width = `${Math.max(4, t2x(dur))}px`;
  const idx = info.trackIdx >= 0 ? info.trackIdx : 0;
  els.ghost.style.top = `${RULER_H + info.tops[idx]}px`;
  els.ghost.style.height = `${proj().tracks[idx].height}px`;
}

function nearestBoundary(info) {
  if (!info.track || info.track.kind !== 'video') {
    // גם באודיו אפשר, אבל נעדיף וידאו
  }
  const track = info.track;
  if (!track) return null;
  let best = null, bestD = 40 / state.zoom;
  for (const c of track.clips) {
    if (!c.tin && !prevAdjacent(track, c)) continue;
    const d = Math.abs(c.start - info.t);
    if (d < bestD) { bestD = d; best = c; }
  }
  if (!best) return null;
  return { clip: best, track, top: info.tops[proj().tracks.indexOf(track)] };
}

function onDrop(e) {
  if (!pendingDrop) return;
  e.preventDefault();
  els.ghost.classList.add('hidden');
  const info = dropInfo(e);

  if (pendingDrop.kind === 'transition') {
    const target = nearestBoundary(info);
    if (!target) { toast('שחררו את המעבר על הגבול שבין שני קליפים', 'err'); pendingDrop = null; return; }
    applyTransition(target.clip, target.track, pendingDrop.type);
    pendingDrop = null;
    return;
  }

  const m = pendingDrop.media;
  const t = state.snap ? snapTime(info.t, []) : snapFrame(info.t, fps());
  const wantKind = m.type === 'audio' ? 'audio' : 'video';
  let track = info.track && info.track.kind === wantKind && !info.track.locked ? info.track : null;
  const dur = m.type === 'image' ? 5 : m.duration;
  if (!track) track = findFreeTrack(wantKind, t, t + dur);

  beginChange();
  const clip = makeClip(m, { trackId: track.id, start: t });
  placeClip(track, clip);
  // וידאו עם פס קול: מוסיפים גם קליפ אודיו מקושר בערוץ אודיו
  commit('drop');
  select(clip.id);
  pendingDrop = null;
}

/* ─────────── פעולות שנגישות מבחוץ ─────────── */

export function duplicateSelection() {
  const ids = [...state.selection];
  if (!ids.length) return;
  beginChange();
  const made = [];
  for (const id of ids) {
    const c = clipById(id), tr = trackOfClip(id);
    if (!c || !tr) continue;
    const copy = JSON.parse(JSON.stringify(c));
    copy.id = `clip_${Math.random().toString(36).slice(2, 9)}`;
    copy.start = clipEnd(c);
    copy.tin = null;
    placeClip(tr, copy);
    made.push(copy.id);
  }
  commit('duplicate');
  select(made);
}

/** זום סביב נקודת עוגן: הזמן anchorT נשאר באותו מקום על המסך */
export function zoomBy(factor, anchorT = null) {
  const sc = els.scroll;
  const t = anchorT ?? x2t(sc.scrollLeft + sc.clientWidth / 2);
  const screenX = anchorT != null ? t2x(anchorT) - sc.scrollLeft : sc.clientWidth / 2;
  state.zoom = clamp(state.zoom * factor, MIN_ZOOM, MAX_ZOOM);
  renderAll();
  sc.scrollLeft = Math.max(0, t2x(t) - screenX);
  syncZoomSlider();
}

export function setZoom(z) {
  state.zoom = clamp(z, MIN_ZOOM, MAX_ZOOM);
  renderAll();
  syncZoomSlider();
}

export function zoomToFit() {
  const d = duration();
  if (d <= 0) { setZoom(60); return; }
  setZoom(clamp(((els.scroll.clientWidth - 40) / d), MIN_ZOOM, MAX_ZOOM));
  els.scroll.scrollLeft = 0;
}

function syncZoomSlider() {
  const r = $('#zoomRange');
  if (!r) return;
  const p = Math.log(state.zoom / MIN_ZOOM) / Math.log(MAX_ZOOM / MIN_ZOOM);
  r.value = String(Math.round(p * 100));
}

export function zoomFromSlider(v) {
  const z = MIN_ZOOM * Math.pow(MAX_ZOOM / MIN_ZOOM, v / 100);
  setZoom(z);
}

function refreshSelection() {
  $$('.clip', els.tracks).forEach((el) => {
    el.classList.toggle('sel', state.selection.has(el.dataset.clip));
  });
  $$('.trans', els.tracks).forEach((el) => {
    el.classList.toggle('sel', state.selTransition === el.dataset.trans);
  });
}
