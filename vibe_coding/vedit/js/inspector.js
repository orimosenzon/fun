/* inspector.js: פאנל המאפיינים של הקליפ או של המעבר הנבחר */

import { $, clamp, tc, shortDur, debounce } from './util.js';
import {
  state, proj, fps, clipById, trackOfClip, mediaById, clipEnd, emitter,
  beginChange, commit, TRANSITIONS, transitionName, maxTransition, sourceRoom, placeClip,
} from './state.js';

let engine = null;
let openGroups = new Set(['transform', 'audio', 'trans', 'title']);

/* בזמן שהמשתמש גורר מחוון בפאנל אסור לבנות אותו מחדש, כי זה מנתק את הגרירה.
   לכן דוחים את הרענון עד לשחרור העכבר. */
let interacting = false;
let renderPending = false;

export function initInspector(eng) {
  engine = eng;
  const body = $('#inspBody');
  body.addEventListener('pointerdown', () => { interacting = true; });
  window.addEventListener('pointerup', () => {
    if (!interacting) return;
    interacting = false;
    if (renderPending) { renderPending = false; render(); }
  });
  emitter.on('selection', render);
  emitter.on('change', render);
  render();
}

/* עורך ערך: נרשם להיסטוריה פעם אחת לכל "סשן" של גרירה */
let editing = false;
const endEdit = debounce(() => { if (editing) { editing = false; commit('inspector'); } }, 350);

function edit(apply) {
  if (!editing) { editing = true; beginChange(); }
  apply();
  engine?.render(state.playhead);
  emitter.emit('inspector-live');
  endEdit();
}

/* ─────────── בנייה של שדות ─────────── */

function group(title, key, inner) {
  const d = document.createElement('details');
  d.className = 'igroup';
  d.open = openGroups.has(key);
  d.addEventListener('toggle', () => { d.open ? openGroups.add(key) : openGroups.delete(key); });
  const s = document.createElement('summary');
  s.textContent = title;
  d.appendChild(s);
  const b = document.createElement('div');
  b.className = 'gbody';
  inner(b);
  d.appendChild(b);
  return d;
}

function slider(host, { label, min, max, step, value, fmt, onInput, onReset }) {
  const f = document.createElement('div');
  f.className = 'field';
  const l = document.createElement('label');
  l.textContent = label;
  if (onReset) {
    l.style.cursor = 'pointer';
    l.title = 'לחיצה כפולה לאיפוס';
    l.addEventListener('dblclick', () => { edit(onReset); render(); });
  }
  const r = document.createElement('input');
  r.type = 'range'; r.min = min; r.max = max; r.step = step; r.value = value;
  const v = document.createElement('span');
  v.className = 'val';
  v.textContent = fmt ? fmt(value) : value;
  r.addEventListener('input', () => {
    const num = parseFloat(r.value);
    v.textContent = fmt ? fmt(num) : num;
    edit(() => onInput(num));
  });
  f.append(l, r, v);
  host.appendChild(f);
  return r;
}

function number(host, { label, min, max, step, value, fmt, onInput }) {
  const f = document.createElement('div');
  f.className = 'field';
  const l = document.createElement('label'); l.textContent = label;
  const n = document.createElement('input');
  n.type = 'number'; n.min = min; n.max = max; n.step = step; n.value = value;
  n.addEventListener('change', () => {
    let num = parseFloat(n.value);
    if (!isFinite(num)) num = value;
    num = clamp(num, min, max);
    n.value = num;
    edit(() => onInput(num));
  });
  f.append(l, n);
  host.appendChild(f);
  return n;
}

function selectField(host, { label, value, options, onChange }) {
  const f = document.createElement('div');
  f.className = 'field';
  const l = document.createElement('label'); l.textContent = label;
  const s = document.createElement('select');
  for (const o of options) {
    const op = document.createElement('option');
    op.value = o.value; op.textContent = o.label;
    if (o.value === value) op.selected = true;
    s.appendChild(op);
  }
  s.addEventListener('change', () => { edit(() => onChange(s.value)); render(); });
  f.append(l, s);
  host.appendChild(f);
  return s;
}

function checkbox(host, { label, value, onChange }) {
  const f = document.createElement('div');
  f.className = 'field';
  const l = document.createElement('label'); l.textContent = label;
  const c = document.createElement('input');
  c.type = 'checkbox'; c.checked = !!value;
  c.addEventListener('change', () => { edit(() => onChange(c.checked)); });
  f.append(l, c);
  host.appendChild(f);
  return c;
}

function row(host, buttons) {
  const d = document.createElement('div');
  d.className = 'btn-row';
  for (const b of buttons) {
    const el = document.createElement('button');
    el.textContent = b.label;
    if (b.danger) el.className = 'danger';
    el.addEventListener('click', b.fn);
    d.appendChild(el);
  }
  host.appendChild(d);
}

/* ─────────── ציור הפאנל ─────────── */

export function render() {
  const host = $('#inspBody');
  if (!host) return;
  if (interacting) { renderPending = true; return; }
  host.innerHTML = '';

  // מעבר נבחר
  if (state.selTransition) {
    const c = clipById(state.selTransition);
    const track = c && trackOfClip(c.id);
    if (c?.tin && track) { renderTransition(host, c, track); return; }
  }

  const ids = [...state.selection];
  if (!ids.length) {
    host.innerHTML = '<div class="insp-empty">בחרו קליפ בטיימליין<br><span class="dim">כדי לערוך את מאפייניו</span></div>';
    return;
  }
  if (ids.length > 1) { renderMulti(host, ids); return; }

  const c = clipById(ids[0]);
  if (!c) return;
  const track = trackOfClip(c.id);
  const media = mediaById(c.mediaId);

  const head = document.createElement('div');
  head.innerHTML = `<div class="clip-title">${escapeHtml(c.name)}</div>
    <div class="clip-sub">${labelKind(c)} · ${shortDur(c.duration)} · מתחיל ב-${tc(c.start, fps())}</div>`;
  host.appendChild(head);

  /* ── כותרת ── */
  if (c.kind === 'title') {
    host.appendChild(group('טקסט', 'title', (b) => {
      const f = document.createElement('div');
      f.className = 'field stack';
      const l = document.createElement('label'); l.textContent = 'תוכן';
      const ta = document.createElement('textarea');
      ta.rows = 3; ta.value = c.text || '';
      ta.addEventListener('input', () => edit(() => {
        c.text = ta.value;
        c.name = ta.value.split('\n')[0].slice(0, 24) || 'כותרת';
      }));
      f.append(l, ta);
      b.appendChild(f);

      number(b, { label: 'גודל גופן', min: 8, max: 400, step: 1, value: c.fontSize || 72,
        onInput: (v) => { c.fontSize = v; } });

      const cf = document.createElement('div');
      cf.className = 'field';
      const cl = document.createElement('label'); cl.textContent = 'צבע';
      const ci = document.createElement('input');
      ci.type = 'color'; ci.value = c.color || '#ffffff';
      ci.addEventListener('input', () => edit(() => { c.color = ci.value; }));
      cf.append(cl, ci);
      b.appendChild(cf);

      selectField(b, { label: 'סגנון', value: c.titleStyle || 'shadow', onChange: (v) => { c.titleStyle = v; },
        options: [
          { value: 'plain', label: 'רגיל' }, { value: 'shadow', label: 'עם צל' },
          { value: 'box', label: 'על רקע כהה' }, { value: 'lower-third', label: 'כותרת תחתונה' },
        ] });
    }));
  }

  /* ── תמונה/וידאו: טרנספורם ── */
  if (c.kind !== 'audio' && track?.kind === 'video') {
    host.appendChild(group('תמונה ומיקום', 'transform', (b) => {
      slider(b, { label: 'שקיפות', min: 0, max: 1, step: 0.01, value: c.opacity ?? 1,
        fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { c.opacity = v; },
        onReset: () => { c.opacity = 1; } });
      slider(b, { label: 'גודל', min: 0.05, max: 4, step: 0.01, value: c.scale ?? 1,
        fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { c.scale = v; },
        onReset: () => { c.scale = 1; } });
      slider(b, { label: 'מיקום ↔', min: -1, max: 1, step: 0.005, value: c.posX ?? 0,
        fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { c.posX = v; },
        onReset: () => { c.posX = 0; } });
      slider(b, { label: 'מיקום ↕', min: -1, max: 1, step: 0.005, value: c.posY ?? 0,
        fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { c.posY = v; },
        onReset: () => { c.posY = 0; } });
      slider(b, { label: 'סיבוב', min: -180, max: 180, step: 1, value: c.rotation ?? 0,
        fmt: (v) => `${v}°`, onInput: (v) => { c.rotation = v; }, onReset: () => { c.rotation = 0; } });
      if (c.kind !== 'title') {
        selectField(b, { label: 'התאמה למסגרת', value: c.fit || 'contain', onChange: (v) => { c.fit = v; },
          options: [
            { value: 'contain', label: 'הכל בפנים' },
            { value: 'cover', label: 'מילוי המסגרת' },
            { value: 'stretch', label: 'מתיחה' },
          ] });
        checkbox(b, { label: 'היפוך אופקי', value: c.flipH, onChange: (v) => { c.flipH = v; } });
      }
      row(b, [{ label: 'איפוס הכל', fn: () => { edit(() => {
        c.opacity = 1; c.scale = 1; c.posX = 0; c.posY = 0; c.rotation = 0; c.flipH = false;
      }); render(); } }]);
    }));

    host.appendChild(group('דעיכת וידאו', 'vfade', (b) => {
      slider(b, { label: 'כניסה', min: 0, max: Math.max(0.1, c.duration / 2), step: 0.05, value: c.vFadeIn || 0,
        fmt: (v) => `${v.toFixed(2)}s`, onInput: (v) => { c.vFadeIn = v; }, onReset: () => { c.vFadeIn = 0; } });
      slider(b, { label: 'יציאה', min: 0, max: Math.max(0.1, c.duration / 2), step: 0.05, value: c.vFadeOut || 0,
        fmt: (v) => `${v.toFixed(2)}s`, onInput: (v) => { c.vFadeOut = v; }, onReset: () => { c.vFadeOut = 0; } });
      row(b, [
        { label: 'מ-שחור 1 שנ׳', fn: () => { edit(() => { c.vFadeIn = Math.min(1, c.duration / 2); }); render(); } },
        { label: 'אל שחור 1 שנ׳', fn: () => { edit(() => { c.vFadeOut = Math.min(1, c.duration / 2); }); render(); } },
      ]);
    }));

    if (c.kind !== 'title') {
      host.appendChild(group('צבע', 'color', (b) => {
        const f = c.filters || (c.filters = { brightness: 100, contrast: 100, saturate: 100, blur: 0 });
        slider(b, { label: 'בהירות', min: 0, max: 250, step: 1, value: f.brightness,
          fmt: (v) => `${v}%`, onInput: (v) => { f.brightness = v; }, onReset: () => { f.brightness = 100; } });
        slider(b, { label: 'ניגודיות', min: 0, max: 250, step: 1, value: f.contrast,
          fmt: (v) => `${v}%`, onInput: (v) => { f.contrast = v; }, onReset: () => { f.contrast = 100; } });
        slider(b, { label: 'רוויה', min: 0, max: 250, step: 1, value: f.saturate,
          fmt: (v) => `${v}%`, onInput: (v) => { f.saturate = v; }, onReset: () => { f.saturate = 100; } });
        slider(b, { label: 'טשטוש', min: 0, max: 30, step: 0.5, value: f.blur,
          fmt: (v) => `${v}px`, onInput: (v) => { f.blur = v; }, onReset: () => { f.blur = 0; } });
        row(b, [
          { label: 'שחור-לבן', fn: () => { edit(() => { f.saturate = 0; }); render(); } },
          { label: 'איפוס', fn: () => { edit(() => { f.brightness = 100; f.contrast = 100; f.saturate = 100; f.blur = 0; }); render(); } },
        ]);
      }));
    }
  }

  /* ── אודיו ── */
  if (c.kind !== 'title' && media?.hasAudio) {
    host.appendChild(group('אודיו', 'audio', (b) => {
      slider(b, { label: 'עוצמה', min: 0, max: 2, step: 0.01, value: c.volume ?? 1,
        fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { c.volume = v; }, onReset: () => { c.volume = 1; } });
      checkbox(b, { label: 'השתקה', value: c.mute, onChange: (v) => { c.mute = v; } });
      slider(b, { label: 'דעיכת כניסה', min: 0, max: Math.max(0.1, c.duration / 2), step: 0.05, value: c.aFadeIn || 0,
        fmt: (v) => `${v.toFixed(2)}s`, onInput: (v) => { c.aFadeIn = v; }, onReset: () => { c.aFadeIn = 0; } });
      slider(b, { label: 'דעיכת יציאה', min: 0, max: Math.max(0.1, c.duration / 2), step: 0.05, value: c.aFadeOut || 0,
        fmt: (v) => `${v.toFixed(2)}s`, onInput: (v) => { c.aFadeOut = v; }, onReset: () => { c.aFadeOut = 0; } });
    }));
  }

  /* ── מהירות ותזמון ── */
  host.appendChild(group('תזמון', 'time', (b) => {
    if (c.kind !== 'title' && media && media.type !== 'image') {
      selectField(b, { label: 'מהירות', value: String(c.speed || 1),
        options: [
          { value: '0.25', label: '0.25× (איטי מאוד)' }, { value: '0.5', label: '0.5× (איטי)' },
          { value: '1', label: '1× (רגיל)' }, { value: '1.5', label: '1.5×' },
          { value: '2', label: '2× (מהיר)' }, { value: '4', label: '4×' },
        ],
        onChange: (v) => {
          const ns = parseFloat(v), old = c.speed || 1;
          const srcLen = c.duration * old;
          c.speed = ns;
          c.duration = srcLen / ns;
        } });
    }
    // אחרי שינוי מספרי מניחים את הקליפ מחדש, כדי שלא ייווצרו חפיפות עם השכנים
    const replace = () => {
      const tr = trackOfClip(c.id);
      if (!tr) return;
      tr.clips = tr.clips.filter((x) => x.id !== c.id);
      placeClip(tr, c);
    };
    number(b, { label: 'התחלה (שנ׳)', min: 0, max: 100000, step: 0.1, value: +c.start.toFixed(3),
      onInput: (v) => { c.start = v; replace(); } });
    number(b, { label: 'משך (שנ׳)', min: 0.05, max: 100000, step: 0.1, value: +c.duration.toFixed(3),
      onInput: (v) => { c.duration = Math.min(v, sourceRoom(c)); replace(); } });
    const info = document.createElement('div');
    info.className = 'clip-sub';
    info.textContent = media
      ? `מקור: ${escapeHtml(media.name)} · נקודת כניסה ${tc(c.inPoint, fps())}`
      : 'קליפ מחולל';
    b.appendChild(info);
  }));

  /* ── מעבר בראש הקליפ ── */
  const track2 = trackOfClip(c.id);
  if (track2) {
    const max = maxTransition(track2, c);
    host.appendChild(group('מעבר בהתחלה', 'trans', (b) => {
      if (max <= 0.02) {
        const p = document.createElement('div');
        p.className = 'clip-sub';
        p.textContent = 'כדי לשים מעבר צריך קליפ צמוד לפני הקליפ הזה.';
        b.appendChild(p);
        return;
      }
      selectField(b, { label: 'סוג', value: c.tin?.type || '',
        options: [{ value: '', label: '(ללא)' }, ...TRANSITIONS.map((t) => ({ value: t.id, label: t.name }))],
        onChange: (v) => {
          if (!v) c.tin = null;
          else c.tin = { type: v, dur: clamp(c.tin?.dur ?? 1, 0.06, max) };
        } });
      if (c.tin) {
        slider(b, { label: 'משך', min: 0.06, max: +max.toFixed(2), step: 0.02, value: Math.min(c.tin.dur, max),
          fmt: (v) => `${v.toFixed(2)}s`, onInput: (v) => { c.tin.dur = v; } });
      }
    }));
  }
}

function renderTransition(host, c, track) {
  const max = maxTransition(track, c);
  const head = document.createElement('div');
  head.innerHTML = `<div class="clip-title">מעבר: ${transitionName(c.tin.type)}</div>
    <div class="clip-sub">לפני "${escapeHtml(c.name)}" · ${c.tin.dur.toFixed(2)} שניות</div>`;
  host.appendChild(head);

  host.appendChild(group('מאפייני המעבר', 'trans', (b) => {
    selectField(b, { label: 'סוג', value: c.tin.type,
      options: TRANSITIONS.map((t) => ({ value: t.id, label: t.name })),
      onChange: (v) => { c.tin.type = v; } });
    slider(b, { label: 'משך', min: 0.06, max: +Math.max(0.1, max).toFixed(2), step: 0.02,
      value: Math.min(c.tin.dur, max), fmt: (v) => `${v.toFixed(2)}s`,
      onInput: (v) => { c.tin.dur = v; } });
    row(b, [
      { label: 'הסרת המעבר', danger: true, fn: () => {
        beginChange(); c.tin = null; state.selTransition = null; commit('trans-remove');
      } },
    ]);
  }));
}

function renderMulti(host, ids) {
  const clips = ids.map(clipById).filter(Boolean);
  const head = document.createElement('div');
  head.innerHTML = `<div class="clip-title">${clips.length} קליפים נבחרו</div>
    <div class="clip-sub">שינוי יחול על כולם</div>`;
  host.appendChild(head);

  host.appendChild(group('פעולות מרובות', 'multi', (b) => {
    slider(b, { label: 'עוצמה', min: 0, max: 2, step: 0.01, value: 1, fmt: (v) => `${Math.round(v * 100)}%`,
      onInput: (v) => clips.forEach((c) => { c.volume = v; }) });
    slider(b, { label: 'שקיפות', min: 0, max: 1, step: 0.01, value: 1, fmt: (v) => `${Math.round(v * 100)}%`,
      onInput: (v) => clips.forEach((c) => { c.opacity = v; }) });
    row(b, [
      { label: 'השתקת כולם', fn: () => { beginChange(); clips.forEach((c) => { c.mute = true; }); commit('mute-multi'); } },
      { label: 'ביטול השתקה', fn: () => { beginChange(); clips.forEach((c) => { c.mute = false; }); commit('mute-multi'); } },
    ]);
    row(b, [
      { label: 'מעבר הדרגתי לכולם', fn: () => {
        beginChange();
        for (const c of clips) {
          const tr = trackOfClip(c.id);
          const m = tr ? maxTransition(tr, c) : 0;
          if (m > 0.05) c.tin = { type: 'dissolve', dur: Math.min(0.7, m) };
        }
        commit('trans-multi');
      } },
    ]);
  }));
}

const labelKind = (c) => ({ video: 'וידאו', audio: 'אודיו', image: 'תמונה', title: 'כותרת' }[c.kind] || c.kind);

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
}
