/* app.js: חיבור כל החלקים. ממשק, קיצורי מקלדת, שמירה אוטומטית וניהול הפרויקט */

import {
  $, $$, clamp, tc, shortDur, bytes, drag, toast, debounce, inField, download, baseName,
} from './util.js';
import {
  state, proj, fps, duration, allClips, clipById, trackOfClip, mediaById, clipEnd,
  emitter, beginChange, commit, undo, redo, canUndo, canRedo, select, clearSelection,
  setPlayhead, placeClip, makeClip, makeTitleClip, splitAll, removeClips, rippleDelete,
  rippleRange, addTrack, findFreeTrack, serialize, load as loadProject, reset as resetProject,
  TRANSITIONS, transitionName, sortTrack,
} from './state.js';
import {
  importFiles, removeMedia, rt, hasFile, mediaLabel, relink,
  idbSaveFile, idbSaveProject, idbLoadProject, idbClear, rehydrate,
} from './media.js';
import { Engine } from './engine.js';
import {
  initTimeline, renderAll as renderTimeline, updatePlayhead, updateBand, ensureVisible,
  setPendingDrop, zoomBy, zoomToFit, zoomFromSlider, openMenu, duplicateSelection, applyTransition,
} from './timeline.js';
import { initInspector, render as renderInspector } from './inspector.js';
import { initExportUI, isExporting } from './export.js';
import { paintPreview } from './transitions.js';
import { log } from './logger.js';

const L = log.tag('app');
const engine = new Engine($('#stage'));
window.__vedit = { state, engine, proj, rt, log };   // נוח לניפוי שגיאות מהקונסולה

/* ═══════════════ אתחול ═══════════════ */

function boot() {
  initTimeline(engine);
  initInspector(engine);
  initExportUI(engine);
  buildTransitionPanel();
  wireTabs();
  wireTopbar();
  wireTransport();
  wireMediaPool();
  wireTitles();
  wireTimelineToolbar();
  wireSplitters();
  wireKeyboard();
  wireGlobalDrop();
  wireHelp();
  wireDiagnostics();

  engine.onTick = (t) => {
    updatePlayhead();
    ensureVisible(t);
  };
  engine.onPlayState = syncPlayButton;
  emitter.on('change', () => {
    refreshMediaPool();
    updateHistoryButtons();
    updateTotals();
    engine.render(state.playhead);
    autosave();
  });
  emitter.on('history', updateHistoryButtons);
  emitter.on('media', refreshMediaPool);
  emitter.on('assets', debounce(() => { refreshMediaPool(); renderInspector(); }, 350));
  emitter.on('inspector-live', debounce(() => renderTimeline(), 150));

  updateTotals();
  engine.render(0);
  restoreSession();
}

/* ═══════════════ טאבים ═══════════════ */

function wireTabs() {
  $$('.tab').forEach((t) => {
    t.addEventListener('click', () => {
      $$('.tab').forEach((x) => x.classList.remove('active'));
      $$('.tabpage').forEach((x) => x.classList.remove('active'));
      t.classList.add('active');
      $(`.tabpage[data-page="${t.dataset.tab}"]`).classList.add('active');
    });
  });
}

/* ═══════════════ מאגר המדיה ═══════════════ */

function wireMediaPool() {
  $('#btnImport').addEventListener('click', () => $('#fileInput').click());
  $('#fileInput').addEventListener('change', async (e) => {
    const files = [...e.target.files];
    e.target.value = '';
    if (files.length) await doImport(files);
  });
}

async function doImport(files) {
  toast(`מייבא ${files.length} קבצים…`);
  const added = await importFiles(files);
  if (!added.length) return;
  added.forEach((m) => idbSaveFile(m));
  toast(`נוספו ${added.length} פריטים למאגר`, 'ok');
  // אם הטיימליין ריק, מכניסים אליו אוטומטית את הפריט הראשון שנוסף
  if (!allClips().length) appendMedia(added[0]);
}

function refreshMediaPool() {
  const host = $('#mediaPool');
  const items = proj().media;
  if (!items.length) {
    host.innerHTML = `<div class="pool-empty"><div class="big">🎞️</div>
      <p>גררו לכאן וידאו, אודיו או תמונות</p><p class="dim">או לחצו על "ייבוא קבצים"</p></div>`;
    return;
  }
  host.innerHTML = '';
  for (const m of items) {
    const d = document.createElement('div');
    d.className = 'mitem';
    d.draggable = true;
    d.dataset.media = m.id;

    const th = document.createElement('div');
    th.className = 'thumb';
    const r = rt(m.id);
    if (m.type === 'image' && r?.url) th.style.backgroundImage = `url("${r.url}")`;
    else if (r?.thumbs?.length) {
      const cv = document.createElement('canvas');
      cv.width = 64; cv.height = 38;
      cv.getContext('2d').drawImage(r.thumbs[Math.floor(r.thumbs.length / 2)], 0, 0, 64, 38);
      th.style.backgroundImage = `url("${cv.toDataURL('image/jpeg', 0.7)}")`;
    } else {
      th.textContent = m.type === 'audio' ? '🎵' : m.type === 'image' ? '🖼️' : '🎬';
    }

    const meta = document.createElement('div');
    meta.className = 'meta';
    const missing = !hasFile(m.id);
    meta.innerHTML = `<div class="mname" title="${escapeAttr(m.name)}">${escapeHtml(m.name)}</div>
      <div class="msub">${missing ? '⚠ הקובץ חסר. לחצו כאן לקישור מחדש'
        : `${mediaLabel(m)} · ${shortDur(m.duration)} · ${bytes(m.size || 0)}`}</div>`;

    const del = document.createElement('button');
    del.className = 'mdel'; del.textContent = '✕'; del.title = 'הסרה מהמאגר';
    del.addEventListener('click', (e) => {
      e.stopPropagation();
      const used = allClips().some((c) => c.mediaId === m.id);
      if (used && !confirm(`"${m.name}" נמצא בשימוש בטיימליין. להסיר אותו ואת הקליפים שלו?`)) return;
      removeMedia(m.id);
    });

    d.append(th, meta, del);

    if (missing) {
      d.addEventListener('click', () => relinkPrompt(m));
    } else {
      d.addEventListener('dragstart', (ev) => {
        setPendingDrop({ kind: 'media', media: m });
        ev.dataTransfer.effectAllowed = 'copy';
        ev.dataTransfer.setData('text/plain', m.id);
      });
      d.addEventListener('dragend', () => setPendingDrop(null));
      d.addEventListener('dblclick', () => appendMedia(m));
      d.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        openMenu(e.clientX, e.clientY, [
          { label: '＋ הוספה לסוף הטיימליין', fn: () => appendMedia(m) },
          { label: '＋ הוספה בנקודת הסמן', fn: () => appendMedia(m, state.playhead) },
          { sep: true },
          { label: '✕ הסרה מהמאגר', fn: () => removeMedia(m.id) },
        ]);
      });
    }
    host.appendChild(d);
  }
}

function relinkPrompt(m) {
  const inp = document.createElement('input');
  inp.type = 'file';
  inp.accept = m.type === 'audio' ? 'audio/*' : m.type === 'image' ? 'image/*' : 'video/*';
  inp.addEventListener('change', async () => {
    if (inp.files[0]) {
      await relink(m, inp.files[0]);
      toast(`"${m.name}" חובר מחדש`, 'ok');
      engine.releaseAll();
      engine.seek(state.playhead);
    }
  });
  inp.click();
}

/** הוספת פריט מדיה לטיימליין */
function appendMedia(m, at = null) {
  const kind = m.type === 'audio' ? 'audio' : 'video';
  const dur = m.type === 'image' ? 5 : m.duration;
  let track = proj().tracks.find((t) => t.id === state.activeTrack && t.kind === kind && !t.locked);
  if (!track) track = proj().tracks.filter((t) => t.kind === kind && !t.locked).pop();
  if (!track) track = addTrack(kind);

  const start = at != null ? at
    : track.clips.length ? Math.max(...track.clips.map(clipEnd)) : 0;

  beginChange();
  const clip = makeClip(m, { trackId: track.id, start });
  placeClip(track, clip);
  commit('append');
  select(clip.id);
  state.activeTrack = track.id;
  toast(`"${m.name}" נוסף לטיימליין`, 'ok', 1600);
}

/* ═══════════════ פאנל המעברים ═══════════════ */

function buildTransitionPanel() {
  const host = $('#transitionList');
  host.innerHTML = '';
  for (const t of TRANSITIONS) {
    const card = document.createElement('div');
    card.className = 'tcard';
    card.draggable = true;
    card.title = `${t.name}. גררו אל הגבול בין שני קליפים, או לחצו פעמיים כדי להחיל על הקליפ הנבחר`;

    const cv = document.createElement('canvas');
    cv.width = 116; cv.height = 42;
    card.appendChild(cv);
    const cap = document.createElement('div');
    cap.textContent = t.name;
    card.appendChild(cap);
    host.appendChild(card);

    paintPreview(cv, t.id, 0.55);

    // אנימציה קטנה בריחוף
    let anim = null;
    card.addEventListener('pointerenter', () => {
      let p = 0;
      cancelAnimationFrame(anim);
      const step = () => {
        p += 0.02;
        paintPreview(cv, t.id, p % 1);
        anim = requestAnimationFrame(step);
      };
      step();
    });
    card.addEventListener('pointerleave', () => {
      cancelAnimationFrame(anim);
      paintPreview(cv, t.id, 0.55);
    });

    card.addEventListener('dragstart', (ev) => {
      setPendingDrop({ kind: 'transition', type: t.id });
      ev.dataTransfer.effectAllowed = 'copy';
      ev.dataTransfer.setData('text/plain', t.id);
    });
    card.addEventListener('dragend', () => setPendingDrop(null));
    card.addEventListener('dblclick', () => {
      const id = [...state.selection][0];
      const c = id && clipById(id);
      if (!c) { toast('בחרו קודם קליפ בטיימליין', 'err'); return; }
      applyTransition(c, trackOfClip(c.id), t.id);
    });
  }
}

/* ═══════════════ כותרות ═══════════════ */

function wireTitles() {
  $('#btnAddTitle').addEventListener('click', () => {
    const text = $('#titleText').value || 'כותרת';
    const size = parseInt($('#titleSize').value, 10) || 72;
    const color = $('#titleColor').value;
    const style = $('#titleStyle').value;

    // הכותרת יושבת על ערוץ וידאו מעל הקיים
    const vTracks = proj().tracks.filter((t) => t.kind === 'video');
    const start = state.playhead;
    let track = vTracks.find((t) => !t.locked
      && !t.clips.some((c) => c.start < start + 4 - 1e-4 && clipEnd(c) > start + 1e-4));
    beginChange();
    if (!track) track = addTrack('video');
    const clip = makeTitleClip({ trackId: track.id, start, text, size, color, style });
    placeClip(track, clip);
    commit('add-title');
    select(clip.id);
    toast('כותרת נוספה', 'ok', 1600);
  });
}

/* ═══════════════ סרגל עליון ═══════════════ */

function wireTopbar() {
  $('#projectName').addEventListener('change', (e) => {
    beginChange(); proj().name = e.target.value || 'פרויקט'; commit('rename');
  });
  $('#btnUndo').addEventListener('click', () => { undo(); afterHistory(); });
  $('#btnRedo').addEventListener('click', () => { redo(); afterHistory(); });

  $('#projRes').addEventListener('change', (e) => {
    const [w, h] = e.target.value.split('x').map(Number);
    beginChange(); proj().width = w; proj().height = h; commit('res');
    engine.render(state.playhead);
  });
  $('#projFps').addEventListener('change', (e) => {
    beginChange(); proj().fps = parseInt(e.target.value, 10); commit('fps');
  });

  $('#btnNewProj').addEventListener('click', async () => {
    if (allClips().length && !confirm('לפתוח פרויקט חדש? כל מה שלא יוצא ייעלם.')) return;
    engine.pause(); engine.releaseAll();
    resetProject();
    await idbClear();
    syncProjectControls();
    zoomToFit();
    toast('פרויקט חדש', 'ok');
  });

  $('#btnSaveProj').addEventListener('click', () => {
    const data = serialize();
    const blob = new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' });
    download(blob, `${baseName(proj().name || 'vedit')}.vedit.json`);
    toast('הפרויקט נשמר לקובץ. הווידאו עצמו לא נשמר בתוכו, ובפתיחה צריך לקשר אותו מחדש.', 'ok', 5000);
  });

  $('#btnOpenProj').addEventListener('click', () => {
    const inp = document.createElement('input');
    inp.type = 'file'; inp.accept = '.json,application/json';
    inp.addEventListener('change', async () => {
      const f = inp.files[0];
      if (!f) return;
      try {
        const data = JSON.parse(await f.text());
        engine.pause(); engine.releaseAll();
        loadProject(data);
        syncProjectControls();
        const missing = await rehydrate(proj().media);
        zoomToFit();
        engine.seek(0);
        if (missing.length) toast(`הפרויקט נטען. ${missing.length} קבצים חסרים, לחצו עליהם במאגר כדי לקשר אותם מחדש.`, 'err', 6000);
        else toast('הפרויקט נטען', 'ok');
      } catch (err) {
        console.error(err);
        toast('לא הצלחתי לקרוא את קובץ הפרויקט', 'err');
      }
    });
    inp.click();
  });
}

function syncProjectControls() {
  $('#projectName').value = proj().name;
  $('#projRes').value = `${proj().width}x${proj().height}`;
  $('#projFps').value = String(proj().fps);
}

function afterHistory() {
  engine.releaseAll();
  engine.seek(state.playhead, true);
  renderTimeline();
  renderInspector();
  updateHistoryButtons();
}

function updateHistoryButtons() {
  $('#btnUndo').disabled = !canUndo();
  $('#btnRedo').disabled = !canRedo();
}

function updateTotals() {
  const d = duration();
  $('#tcTotal').textContent = tc(d, fps());
  const info = $('#monInfo');
  const n = allClips().length;
  info.textContent = n ? `${proj().width}×${proj().height} · ${n} קליפים · ${shortDur(d)}` : '';

  // רמז למסך ריק
  const ov = $('#stageOverlay');
  if (!n) {
    ov.innerHTML = `<div class="stage-hint">
      <div class="big">🎬</div>
      <h2>ברוכים הבאים ל-vedit</h2>
      <p>גררו לכאן קובצי וידאו, או לחצו על <b>ייבוא קבצים</b> בפאנל המדיה.</p>
      <p class="dim">הכול קורה בדפדפן שלכם. שום קובץ לא נשלח לשום שרת.</p>
    </div>`;
  } else if (ov.firstChild) {
    ov.innerHTML = '';
  }
}

/* ═══════════════ בקרות ניגון ═══════════════ */

function wireTransport() {
  $('#btnPlay').addEventListener('click', togglePlay);
  $('#btnStart').addEventListener('click', () => gotoTime(0));
  $('#btnEnd').addEventListener('click', () => gotoTime(duration()));
  $('#btnPrevFrame').addEventListener('click', () => stepFrame(-1));
  $('#btnNextFrame').addEventListener('click', () => stepFrame(1));
  $('#btnSplitT').addEventListener('click', doSplit);
  $('#btnMarkIn').addEventListener('click', () => markIn());
  $('#btnMarkOut').addEventListener('click', () => markOut());
  $('#btnClearMarks').addEventListener('click', () => { state.inPoint = null; state.outPoint = null; updateBand(); });
  $('#btnSnapshot').addEventListener('click', snapshot);

  $('#zoomFit').addEventListener('change', applyMonitorZoom);
  window.addEventListener('resize', debounce(applyMonitorZoom, 120));
  applyMonitorZoom();

  emitter.on('playhead', () => { $('#tcCur').textContent = tc(state.playhead, fps()); });
}

function applyMonitorZoom() {
  const v = $('#zoomFit').value;
  const cv = $('#stage');
  if (v === 'fit') {
    cv.style.width = ''; cv.style.height = '';
    cv.style.maxWidth = '100%'; cv.style.maxHeight = '100%';
  } else {
    const s = parseFloat(v);
    cv.style.maxWidth = 'none'; cv.style.maxHeight = 'none';
    cv.style.width = `${proj().width * s}px`;
    cv.style.height = `${proj().height * s}px`;
  }
}

/** מסונכרן מהמנוע, כך שגם סיום טבעי של הניגון מעדכן את הכפתור */
function syncPlayButton(playing) {
  const b = $('#btnPlay');
  b.textContent = playing ? '⏸' : '▶';
  b.classList.toggle('on', playing);
  if (!playing) shuttle = 0;
}

function togglePlay() {
  if (engine.playing) engine.pause();
  else { engine.setRate(1); engine.play(); }
}

function stopPlay() {
  if (engine.playing) engine.pause();
}

function gotoTime(t) {
  stopPlay();
  setPlayhead(t);
  engine.seek(state.playhead, true);
  ensureVisible(state.playhead);
}

function stepFrame(dir, big = false) {
  stopPlay();
  const d = big ? 1 : 1 / fps();
  setPlayhead(state.playhead + dir * d);
  engine.seek(state.playhead, true);
  ensureVisible(state.playhead);
}

function doSplit() {
  beginChange();
  const n = splitAll(state.playhead);
  commit('split');
  if (n) toast(`נחתכו ${n} קליפים`, 'ok', 1400);
  else toast('אין קליפ בנקודת הסמן', 'err', 1600);
}

function markIn() {
  state.inPoint = state.playhead;
  if (state.outPoint != null && state.outPoint <= state.inPoint) state.outPoint = null;
  updateBand();
  toast(`נקודת כניסה: ${tc(state.inPoint, fps())}`, '', 1400);
}

function markOut() {
  state.outPoint = state.playhead;
  if (state.inPoint != null && state.inPoint >= state.outPoint) state.inPoint = null;
  updateBand();
  toast(`נקודת יציאה: ${tc(state.outPoint, fps())}`, '', 1400);
}

function cutMarkedRange() {
  const a = state.inPoint, b = state.outPoint;
  if (a == null || b == null || b <= a) { toast('סמנו קודם כניסה (I) ויציאה (O)', 'err'); return; }
  beginChange();
  rippleRange(a, b);
  commit('ripple-range');
  state.outPoint = null;
  setPlayhead(a);
  updateBand();
  engine.seek(state.playhead, true);
  toast(`הוסר קטע של ${shortDur(b - a)}`, 'ok');
}

async function snapshot() {
  await engine.waitReady(state.playhead);
  engine.render(state.playhead);
  $('#stage').toBlob((blob) => {
    download(blob, `${baseName(proj().name || 'vedit')}_${tc(state.playhead, fps()).replace(/:/g, '-')}.png`);
    toast('הפריים נשמר כתמונה', 'ok');
  }, 'image/png');
}

/* ═══════════════ סרגל הטיימליין ═══════════════ */

function wireTimelineToolbar() {
  $$('#toolModes .tool').forEach((b) => {
    b.addEventListener('click', () => setTool(b.dataset.tool));
  });
  $('#btnSplit').addEventListener('click', doSplit);
  $('#btnDelete').addEventListener('click', deleteSelection);
  $('#btnRipple').addEventListener('click', rippleDeleteSelection);
  $('#btnDup').addEventListener('click', duplicateSelection);
  $('#btnCutSel').addEventListener('click', cutMarkedRange);
  $('#btnAddVideoTrack').addEventListener('click', () => { beginChange(); addTrack('video'); commit('add-track'); });
  $('#btnAddAudioTrack').addEventListener('click', () => { beginChange(); addTrack('audio'); commit('add-track'); });
  $('#chkSnap').addEventListener('change', (e) => { state.snap = e.target.checked; });
  $('#btnZoomIn').addEventListener('click', () => zoomBy(1.35));
  $('#btnZoomOut').addEventListener('click', () => zoomBy(1 / 1.35));
  $('#btnZoomFitTl').addEventListener('click', zoomToFit);
  $('#zoomRange').addEventListener('input', (e) => zoomFromSlider(parseFloat(e.target.value)));
}

function setTool(t) {
  state.tool = t;
  $$('#toolModes .tool').forEach((b) => b.classList.toggle('active', b.dataset.tool === t));
  const cursor = t === 'razor' ? 'crosshair' : t === 'hand' ? 'grab' : '';
  $('#tracks').style.cursor = cursor;
}

function deleteSelection() {
  if (!state.selection.size) { toast('לא נבחר כלום'); return; }
  const n = state.selection.size;
  beginChange();
  removeClips([...state.selection]);
  commit('delete');
  toast(`נמחקו ${n} קליפים`, '', 1400);
}

function rippleDeleteSelection() {
  if (!state.selection.size) { toast('לא נבחר כלום'); return; }
  beginChange();
  rippleDelete([...state.selection]);
  commit('ripple');
}

/* ═══════════════ העתקה והדבקה ═══════════════ */

let clipboard = [];

function copySelection(cut = false) {
  const ids = [...state.selection];
  if (!ids.length) { toast('לא נבחר כלום'); return; }
  clipboard = ids.map((id) => JSON.parse(JSON.stringify(clipById(id)))).filter(Boolean);
  toast(`${clipboard.length} קליפים הועתקו`, '', 1400);
  if (cut) { beginChange(); removeClips(ids); commit('cut'); }
}

function pasteClipboard() {
  if (!clipboard.length) { toast('אין מה להדביק'); return; }
  const base = Math.min(...clipboard.map((c) => c.start));
  beginChange();
  const made = [];
  for (const src of clipboard) {
    const c = JSON.parse(JSON.stringify(src));
    c.id = `clip_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
    c.start = Math.max(0, state.playhead + (src.start - base));
    c.tin = null;
    const kind = c.kind === 'audio' ? 'audio' : 'video';
    let tr = proj().tracks.find((t) => t.id === c.trackId && !t.locked);
    if (!tr) tr = findFreeTrack(kind, c.start, c.start + c.duration);
    c.trackId = tr.id;
    placeClip(tr, c);
    made.push(c.id);
  }
  commit('paste');
  select(made);
  toast(`הודבקו ${made.length} קליפים`, 'ok', 1400);
}

/* ═══════════════ מחיצות נגררות ═══════════════ */

function wireSplitters() {
  $$('.vsplit').forEach((sp) => {
    sp.addEventListener('pointerdown', (e) => {
      const which = sp.dataset.split;
      const panel = which === 'left' ? $('#leftPanel') : $('#inspector');
      const w0 = panel.getBoundingClientRect().width;
      drag(e, {
        cursor: 'col-resize',
        onMove: (dx) => {
          const w = clamp(which === 'left' ? w0 + dx : w0 - dx, 170, 460);
          panel.style.width = `${w}px`;
        },
        onEnd: () => { renderTimeline(); applyMonitorZoom(); },
      });
    });
  });

  $('.hsplit').addEventListener('pointerdown', (e) => {
    const panel = $('#timelinePanel');
    const h0 = panel.getBoundingClientRect().height;
    drag(e, {
      cursor: 'row-resize',
      onMove: (dy) => { panel.style.height = `${clamp(h0 - dy, 120, innerHeight - 220)}px`; },
      onEnd: () => { renderTimeline(); applyMonitorZoom(); },
    });
  });
}

/* ═══════════════ גרירת קבצים לחלון ═══════════════ */

function wireGlobalDrop() {
  let depth = 0;
  const zone = $('#dropZone');
  const isFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');

  window.addEventListener('dragenter', (e) => {
    if (!isFiles(e)) return;
    depth++; zone.classList.remove('hidden');
  });
  window.addEventListener('dragover', (e) => { if (isFiles(e)) e.preventDefault(); });
  window.addEventListener('dragleave', () => { if (--depth <= 0) { depth = 0; zone.classList.add('hidden'); } });
  window.addEventListener('drop', async (e) => {
    if (!isFiles(e)) return;
    e.preventDefault();
    depth = 0; zone.classList.add('hidden');
    const files = [...(e.dataTransfer.files || [])];
    if (files.length) await doImport(files);
  });
}

/* ═══════════════ קיצורי מקלדת ═══════════════ */

let shuttle = 0;

function wireKeyboard() {
  window.addEventListener('keydown', (e) => {
    if (inField() || isExporting()) return;
    const ctrl = e.ctrlKey || e.metaKey;
    const k = e.key;

    if (ctrl && (k === 'z' || k === 'Z')) {
      e.preventDefault();
      if (e.shiftKey) redo(); else undo();
      afterHistory();
      return;
    }
    if (ctrl && (k === 'y' || k === 'Y')) { e.preventDefault(); redo(); afterHistory(); return; }
    if (ctrl && (k === 'a' || k === 'A')) { e.preventDefault(); select(allClips().map((c) => c.id)); return; }
    if (ctrl && e.shiftKey && (k === 'd' || k === 'D')) { e.preventDefault(); openDiag(); return; }
    if (ctrl && (k === 'd' || k === 'D')) { e.preventDefault(); duplicateSelection(); return; }
    if (ctrl && (k === 'k' || k === 'K')) { e.preventDefault(); doSplit(); return; }
    if (ctrl && (k === 'c' || k === 'C')) { e.preventDefault(); copySelection(false); return; }
    if (ctrl && (k === 'x' || k === 'X')) { e.preventDefault(); copySelection(true); return; }
    if (ctrl && (k === 'v' || k === 'V')) { e.preventDefault(); pasteClipboard(); return; }
    if (ctrl && (k === 's' || k === 'S')) { e.preventDefault(); $('#btnSaveProj').click(); return; }
    if (ctrl) return;

    switch (k) {
      case ' ':
        e.preventDefault(); shuttle = 0; togglePlay(); break;
      case 'k': case 'K':
        e.preventDefault(); shuttle = 0; stopPlay(); break;
      case 'l': case 'L': {
        e.preventDefault();
        const next = shuttle >= 1 ? Math.min(shuttle * 2, 8) : 1;
        if (!engine.playing) engine.play();
        shuttle = next;
        engine.setRate(shuttle);
        toast(`${shuttle}× קדימה`, '', 900);
        break;
      }
      case 'j': case 'J': {
        e.preventDefault();
        // ניגון לאחור אינו נתמך על ידי אלמנטי וידאו, ולכן קופצים אחורה בקפיצות
        stopPlay();
        setPlayhead(state.playhead - 0.5);
        engine.seek(state.playhead, true);
        ensureVisible(state.playhead);
        break;
      }
      case 'ArrowLeft': e.preventDefault(); stepFrame(-1, e.shiftKey); break;
      case 'ArrowRight': e.preventDefault(); stepFrame(1, e.shiftKey); break;
      case 'Home': e.preventDefault(); gotoTime(0); break;
      case 'End': e.preventDefault(); gotoTime(duration()); break;
      case 's': case 'S': e.preventDefault(); doSplit(); break;
      case 'v': case 'V': setTool('select'); break;
      case 'c': case 'C': setTool('razor'); break;
      case 'h': case 'H': setTool('hand'); break;
      case 'i': case 'I': markIn(); break;
      case 'o': case 'O': markOut(); break;
      case 'x': case 'X': if (e.shiftKey) { e.preventDefault(); cutMarkedRange(); } break;
      case 'Delete': case 'Backspace':
        e.preventDefault();
        if (e.shiftKey) rippleDeleteSelection(); else deleteSelection();
        break;
      case 'Escape': clearSelection(); break;
      case '+': case '=': e.preventDefault(); zoomBy(1.35); break;
      case '-': case '_': e.preventDefault(); zoomBy(1 / 1.35); break;
      case 'z': case 'Z': if (e.shiftKey) zoomToFit(); break;
      case 'm': case 'M': {
        const tr = proj().tracks.find((t) => t.id === state.activeTrack);
        if (tr) { beginChange(); tr.muted = !tr.muted; commit('mute-track'); }
        break;
      }
      case '?': $('#modalHelp').classList.remove('hidden'); break;
      default: break;
    }
  });

  window.addEventListener('keyup', (e) => {
    if (e.key === 'l' || e.key === 'L') { /* השארת המהירות עד ללחיצה על K/רווח */ }
  });
}

/* ═══════════════ אבחון ═══════════════ */

/** הסבר בעברית לכל סיבה שבגללה פריים לא צויר */
const SKIP_TEXT = {
  'not-ready': ['הדפדפן עוד לא פענח את הפריים הזה',
    'אם זה נתקע ככה, כנראה שהקודק של הקובץ לא נתמך בדפדפן הזה (למשל HEVC/H.265 או ProRes). נסו קובץ MP4/H.264.'],
  'no-size': ['אין לפריים מידות',
    'הדפדפן לא הצליח לקרוא את גודל התמונה מהקובץ.'],
  'no-player': ['הקובץ של הקליפ לא זמין',
    'אולי הוא הוסר מהמאגר. לחצו על הפריט במאגר כדי לקשר אותו מחדש.'],
  'media-missing': ['פריט המדיה נמחק', 'הקליפ מצביע על קובץ שכבר לא קיים בפרויקט.'],
  'drawImage-threw': ['הציור על הקנבס נכשל', 'זו כנראה תקלה בדפדפן או בזיכרון הגרפי.'],
  'alpha-zero': ['הקליפ שקוף לגמרי', 'בדקו את השקיפות או את הדעיכות באינספקטור.'],
  'no-element': ['לא נוצר נגן לקליפ', ''],
};

/* ההודעה מופיעה רק אם הפריים נשאר ריק יותר משנייה, כדי לא להבהב בכל דילוג קצר */
let blankTimer = null;

function showStageStatus(reason) {
  clearTimeout(blankTimer);
  if (!reason) {
    const ov0 = $('#stageOverlay');
    if (ov0.dataset.mode === 'status') { ov0.innerHTML = ''; delete ov0.dataset.mode; updateTotals(); }
    return;
  }
  blankTimer = setTimeout(() => paintStageStatus(reason), 1100);
}

function paintStageStatus(reason) {
  const ov = $('#stageOverlay');
  if (ov.dataset.mode === 'status' && ov.dataset.reason === reason) return;
  const [title, body] = SKIP_TEXT[reason] || ['אי אפשר להציג את הפריים', ''];
  ov.dataset.mode = 'status';
  ov.dataset.reason = reason;
  ov.innerHTML = `<div class="stage-status">
    <div class="st-title">⚠ ${title}</div>
    <div class="st-body">${body}<br>לחצו על 🩺 בסרגל העליון כדי לראות דוח אבחון מלא.</div>
    <div class="st-code">reason: ${reason}</div>
  </div>`;
}

/** מסקנה אוטומטית בראש הדוח, כדי שאפשר יהיה לקרוא אותו במבט אחד */
function verdict() {
  const d = engine.diagnostics();
  const notes = [];
  if (!proj().media.length) notes.push('לא יובאו קבצים.');
  for (const p of d.players) {
    if (p.errorCode) notes.push(`שגיאת פענוח בקובץ "${p.media}" (code ${p.errorCode}). הקודק כנראה לא נתמך בדפדפן הזה.`);
    else if (p.readyState < 2 && p.ageMs > 3000) notes.push(`"${p.media}" לא הגיע לפריים ראשון תוך ${(p.ageMs / 1000).toFixed(1)} שניות (readyState=${p.readyState}). חשד לקודק לא נתמך.`);
    else if (p.videoSize === '0x0' && p.kind === 'video') notes.push(`"${p.media}" נטען בלי מסלול וידאו קריא.`);
  }
  if (d.blankFrame) notes.push(`הפריים האחרון יצא ריק (סיבה: ${d.lastSkip}).`);
  if (d.audio !== 'not-created' && d.audio.state !== 'running') notes.push(`הקשר האודיו במצב ${d.audio.state}.`);
  const errs = log.entries().filter((e) => e.lvl === 'error');
  if (errs.length) notes.push(`${errs.length} שגיאות ביומן, הראשונה: ${errs[0].msg}`);
  return notes;
}

function openDiag() {
  const notes = verdict();
  const v = $('#diagVerdict');
  v.className = `note ${notes.length ? 'verdict-bad' : 'verdict-ok'}`;
  v.textContent = notes.length ? `נמצאו סימנים: ${notes.join(' ')}` : 'לא נמצאו תקלות בולטות.';
  $('#diagText').value = log.report();
  $('#modalDiag').classList.remove('hidden');
}

function wireDiagnostics() {
  // ספקי מידע לדוח
  log.provider('project', () => ({
    name: proj().name,
    resolution: `${proj().width}x${proj().height}@${proj().fps}`,
    duration: +duration().toFixed(2),
    zoom: state.zoom, tool: state.tool, snap: state.snap,
    selection: state.selection.size,
    tracks: proj().tracks.map((t) => ({
      name: t.name, kind: t.kind, clips: t.clips.length,
      muted: t.muted, hidden: t.hidden, locked: t.locked,
    })),
  }));
  log.provider('media', () => proj().media.map((m) => {
    const r = rt(m.id);
    return {
      name: m.name, type: m.type, sizeBytes: m.size,
      duration: +Number(m.duration || 0).toFixed(2),
      pixels: `${m.width}x${m.height}`, hasAudio: m.hasAudio,
      fileLinked: !!r?.url, thumbs: r?.thumbs?.length ?? 0, peaks: r?.peaks?.length ?? 0,
    };
  }));
  log.provider('clips', () => allClips().map((c) => ({
    name: c.name, kind: c.kind, start: +c.start.toFixed(2), dur: +c.duration.toFixed(2),
    in: +c.inPoint.toFixed(2), speed: c.speed, opacity: c.opacity,
    fit: c.fit || 'contain', transition: c.tin ? `${c.tin.type}/${c.tin.dur.toFixed(2)}` : null,
  })));

  engine.onBlank = showStageStatus;
  engine.onMediaError = (media, err) => {
    toast(`הדפדפן לא הצליח לפענח את "${media.name}" (קוד ${err?.code ?? '?'}). כנראה קודק לא נתמך.`, 'err', 7000);
  };

  $('#btnDiag').addEventListener('click', openDiag);
  $('#diagClose').addEventListener('click', () => $('#modalDiag').classList.add('hidden'));
  $('#diagRefresh').addEventListener('click', openDiag);
  $('#diagDownload').addEventListener('click', () => {
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    download(new Blob([$('#diagText').value], { type: 'text/plain;charset=utf-8' }), `vedit-diagnostics-${stamp}.txt`);
  });
  $('#diagCopy').addEventListener('click', async () => {
    const text = $('#diagText').value;
    try {
      await navigator.clipboard.writeText(text);
      toast('הדוח הועתק ללוח', 'ok');
    } catch {
      $('#diagText').select();
      document.execCommand?.('copy');
      toast('הדוח נבחר, הקישו Ctrl+C', '', 4000);
    }
  });
  $('#diagClear').addEventListener('click', () => { log.clear(); openDiag(); toast('היומן נוקה'); });
  $('#modalDiag').addEventListener('click', (e) => {
    if (e.target.id === 'modalDiag') $('#modalDiag').classList.add('hidden');
  });

  // מחווה ראשונה של המשתמש: משחררים את גרף האודיו. חייב לקרות אחרי אינטראקציה,
  // ואלמנט וידאו שמחובר להקשר מושהה עלול לא לפענח פריימים בכלל.
  const unlock = () => {
    engine.unlockAudio();
    L.info('first user gesture');
    window.removeEventListener('pointerdown', unlock);
    window.removeEventListener('keydown', unlock);
  };
  window.addEventListener('pointerdown', unlock);
  window.addEventListener('keydown', unlock);

  L.info('app booted', { href: location.href });
}

function wireHelp() {
  $('#btnHelp').addEventListener('click', () => $('#modalHelp').classList.remove('hidden'));
  $('#helpClose').addEventListener('click', () => $('#modalHelp').classList.add('hidden'));
  $('#modalHelp').addEventListener('click', (e) => {
    if (e.target.id === 'modalHelp') $('#modalHelp').classList.add('hidden');
  });
}

/* ═══════════════ שמירה אוטומטית ושחזור ═══════════════ */

const autosave = debounce(() => {
  try { idbSaveProject(serialize()); } catch {}
}, 1200);

async function restoreSession() {
  let saved = null;
  try { saved = await idbLoadProject(); } catch {}
  if (!saved || !saved.tracks?.some((t) => t.clips?.length)) { syncProjectControls(); return; }
  try {
    loadProject(saved);
    syncProjectControls();
    const missing = await rehydrate(proj().media);
    renderTimeline();
    zoomToFit();
    engine.seek(0, true);
    if (missing.length) {
      toast(`שוחזר הפרויקט האחרון. ${missing.length} קבצים חסרים, לחצו עליהם במאגר לקישור מחדש.`, 'err', 6000);
    } else {
      toast('שוחזר הפרויקט האחרון שעבדתם עליו', 'ok', 3500);
    }
  } catch (e) {
    console.warn('restore failed', e);
  }
}

window.addEventListener('beforeunload', (e) => {
  if (isExporting()) { e.preventDefault(); e.returnValue = ''; }
});

/* ═══════════════ עזרים ═══════════════ */

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
}
const escapeAttr = escapeHtml;

/* ═══════════════ יציאה לדרך ═══════════════ */

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
