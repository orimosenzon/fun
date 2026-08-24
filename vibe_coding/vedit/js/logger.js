/* logger.js: יומן אבחון
 *
 * הרעיון: כשמשהו לא עובד אצל המשתמש אין לי גישה לקונסולה שלו. לכן כל אירוע
 * משמעותי נרשם לחוצץ טבעתי בזיכרון, השורות האחרונות נשמרות ב-localStorage
 * (כדי לשרוד גם רענון או קריסה), ואפשר לייצא בלחיצה אחת דוח מלא שכולל את
 * היומן יחד עם צילום מצב חי של המנוע, המדיה והדפדפן.
 *
 * שימוש:  const L = log.tag('engine');  L.info('player created', {id});
 * דוח:    Ctrl+Shift+D  או כפתור האבחון בסרגל העליון
 */

const MAX_ENTRIES = 3000;
const PERSIST_ENTRIES = 500;
const LS_KEY = 'vedit.diag';

const t0 = performance.now();
const entries = [];
let seq = 0;
let persistTimer = null;

/** ספקי מידע לצילום מצב: name → () => object */
const providers = new Map();

const now = () => +(performance.now() - t0).toFixed(1);

function push(lvl, tag, msg, data) {
  const e = { i: ++seq, t: now(), lvl, tag, msg, data };
  entries.push(e);
  if (entries.length > MAX_ENTRIES) entries.shift();

  if (lvl === 'error') {
    console.error(`[vedit/${tag}] ${msg}`, data ?? '');
    persistSoon(0);
  } else if (lvl === 'warn') {
    console.warn(`[vedit/${tag}] ${msg}`, data ?? '');
    persistSoon();
  } else if (window.__veditVerbose) {
    console.debug(`[vedit/${tag}] ${msg}`, data ?? '');
  }
  return e;
}

function persistSoon(delay = 3000) {
  clearTimeout(persistTimer);
  persistTimer = setTimeout(persistNow, delay);
}

function persistNow() {
  try {
    const tail = entries.slice(-PERSIST_ENTRIES);
    localStorage.setItem(LS_KEY, JSON.stringify({ at: new Date().toISOString(), tail }));
  } catch { /* מכסת אחסון מלאה, לא נורא */ }
}

/* ─────────── ויסות: אירועים שקורים 60 פעם בשנייה ─────────── */

const seen = new Map();

/** נרשם רק בפעם הראשונה עבור המפתח הזה */
function first(key) {
  if (seen.has(key)) return false;
  seen.set(key, 1);
  return true;
}

/** נרשם לכל היותר פעם ב-ms, ומדווח כמה פעמים דילגנו */
function throttled(key, ms) {
  const rec = seen.get(key);
  const n = performance.now();
  if (!rec || typeof rec !== 'object') { seen.set(key, { last: n, skipped: 0 }); return { ok: true, skipped: 0 }; }
  if (n - rec.last < ms) { rec.skipped++; return { ok: false, skipped: rec.skipped }; }
  const skipped = rec.skipped;
  rec.last = n; rec.skipped = 0;
  return { ok: true, skipped };
}

export function resetThrottles(prefix = '') {
  for (const k of [...seen.keys()]) if (!prefix || String(k).startsWith(prefix)) seen.delete(k);
}

/* ─────────── ה-API ─────────── */

function make(tag) {
  return {
    debug: (msg, data) => push('debug', tag, msg, data),
    info:  (msg, data) => push('info', tag, msg, data),
    warn:  (msg, data) => push('warn', tag, msg, data),
    error: (msg, data) => push('error', tag, msg, data),
    /** נרשם פעם אחת בלבד לכל מפתח */
    once: (key, lvl, msg, data) => { if (first(`${tag}:${key}`)) push(lvl, tag, msg, data); },
    /** נרשם לכל היותר פעם ב-ms */
    every: (key, ms, lvl, msg, data) => {
      const r = throttled(`${tag}:${key}`, ms);
      if (r.ok) push(lvl, tag, msg, r.skipped ? { ...data, _repeated: r.skipped } : data);
    },
    /** מודד כמה זמן לקחה פעולה אסינכרונית */
    timer: (label) => {
      const s = performance.now();
      return (extra) => push('info', tag, `${label} took ${(performance.now() - s).toFixed(0)}ms`, extra);
    },
  };
}

export const log = {
  tag: make,
  /** רישום ספק מידע לצילום המצב */
  provider(name, fn) { providers.set(name, fn); },
  entries: () => entries,
  clear() { entries.length = 0; seen.clear(); try { localStorage.removeItem(LS_KEY); } catch {} },
  snapshot,
  report,
  first,
};

const L = make('log');

/* ─────────── צילום מצב ─────────── */

function snapshot() {
  const out = {};
  for (const [name, fn] of providers) {
    try { out[name] = fn(); } catch (e) { out[name] = { _error: String(e) }; }
  }
  return out;
}

function env() {
  const c = document.createElement('canvas');
  let gl = null;
  try { gl = c.getContext('webgl2') || c.getContext('webgl'); } catch {}
  const mr = [];
  if (typeof MediaRecorder !== 'undefined') {
    for (const m of ['video/mp4;codecs=avc1.42E01E,mp4a.40.2', 'video/mp4;codecs=avc1.42E01E,opus',
      'video/mp4', 'video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus']) {
      try { if (MediaRecorder.isTypeSupported(m)) mr.push(m); } catch {}
    }
  }
  const v = document.createElement('video');
  const canPlay = {};
  for (const m of ['video/mp4; codecs="avc1.42E01E"', 'video/mp4; codecs="hvc1"',
    'video/webm; codecs="vp9"', 'video/quicktime']) {
    canPlay[m] = v.canPlayType(m) || 'no';
  }
  return {
    ua: navigator.userAgent,
    platform: navigator.platform,
    languages: navigator.languages?.join(','),
    screen: `${screen.width}x${screen.height} dpr=${devicePixelRatio}`,
    window: `${innerWidth}x${innerHeight}`,
    memoryGB: navigator.deviceMemory ?? '?',
    cores: navigator.hardwareConcurrency ?? '?',
    webgl: gl ? (gl.getParameter(gl.VERSION) || 'yes') : 'NONE',
    recorderFormats: mr,
    canPlayType: canPlay,
    audioContext: typeof AudioContext !== 'undefined' || typeof webkitAudioContext !== 'undefined',
    uptimeSec: +(now() / 1000).toFixed(1),
  };
}

const LVL_MARK = { debug: '  ', info: '  ', warn: '! ', error: 'X ' };

/** בונה דוח טקסט מלא להעתקה או להורדה */
function report() {
  const lines = [];
  const p = (s = '') => lines.push(s);

  // הדוח כולו באנגלית בכוונה: הוא מוצג בתיבה חד-כיוונית, וערבוב עברית עם
  // מספרים וקוד היה מתהפך על המסך ובהדבקה.
  p('====================================================');
  p(' vedit diagnostics report');
  p(` generated: ${new Date().toISOString()}`);
  p('====================================================');
  p();

  p('-- ENVIRONMENT --');
  for (const [k, v] of Object.entries(env())) {
    p(`  ${k}: ${typeof v === 'object' ? JSON.stringify(v) : v}`);
  }
  p();

  p('-- CURRENT STATE --');
  p(JSON.stringify(snapshot(), null, 1));
  p();

  const prev = loadPersisted();
  if (prev?.tail?.length) {
    p(`-- LOG FROM PREVIOUS RUN (${prev.at}) --`);
    for (const e of prev.tail.slice(-120)) p(fmt(e));
    p();
  }

  p(`-- LOG, THIS RUN (${entries.length} entries) --`);
  for (const e of entries) p(fmt(e));
  p();
  p('-- END OF REPORT --');
  return lines.join('\n');
}

function fmt(e) {
  const d = e.data === undefined ? ''
    : ' ' + safeJson(e.data);
  return `${String(e.t).padStart(9)}ms ${LVL_MARK[e.lvl] || '  '}[${e.tag}] ${e.msg}${d}`;
}

function safeJson(o) {
  try {
    return JSON.stringify(o, (k, v) => {
      if (typeof v === 'number') return Number.isInteger(v) ? v : +v.toFixed(3);
      if (v instanceof Error) return `${v.name}: ${v.message}`;
      return v;
    });
  } catch { return String(o); }
}

function loadPersisted() {
  try { return JSON.parse(localStorage.getItem(LS_KEY) || 'null'); } catch { return null; }
}

/* ─────────── לכידת תקלות גלובליות ─────────── */

window.addEventListener('error', (e) => {
  push('error', 'window', e.message, {
    src: `${e.filename}:${e.lineno}:${e.colno}`,
    stack: e.error?.stack?.split('\n').slice(0, 4).join(' | '),
  });
});

window.addEventListener('unhandledrejection', (e) => {
  push('error', 'promise', 'unhandled rejection', {
    reason: String(e.reason?.message || e.reason),
    stack: e.reason?.stack?.split('\n').slice(0, 4).join(' | '),
  });
});

window.addEventListener('beforeunload', persistNow);
document.addEventListener('visibilitychange', () => {
  push('info', 'page', `visibility: ${document.visibilityState}`);
  if (document.visibilityState === 'hidden') persistNow();
});

L.info('logger ready', { persisted: !!loadPersisted() });

/* נגיש מהקונסולה: __vedit.log.report() / copy(__vedit.log.report()) */
export default log;
