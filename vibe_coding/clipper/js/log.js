/* log.js: יומן אבחון. נטען ראשון, לפני כל השאר.
 *
 * כל שלב חשוב נרשם (טעינת קובץ, נתוני הווידאו, תמונות ממוזערות, צורת גל, ייצוא, שגיאות),
 * וגם כל שגיאה שלא נתפסה וכל פעם שהדף קפא. היומן נשמר ב-localStorage כל כמה שניות, כך
 * שגם אם הלשונית נתקעה וצריך לטעון מחדש, היומן של הפעם הקודמת עדיין שם.
 * כפתור "Log" מוריד קובץ טקסט עם שני היומנים, הנוכחי והקודם.
 */
window.C = window.C || {};

C.log = (() => {
  const VERSION = '2026-10-08a';
  const KEY = 'clipper.log', PREV = 'clipper.log.prev', MAX = 1500;
  const t0 = performance.now();
  const started = new Date().toISOString();
  const lines = [];
  let dirty = false;

  // היומן של הפעם הקודמת עובר הצידה, ונשמר עד הפעם הבאה
  let previous = '';
  try {
    previous = localStorage.getItem(KEY) || '';
    if (previous) localStorage.setItem(PREV, previous); else previous = localStorage.getItem(PREV) || '';
  } catch {}

  const stamp = () => ((performance.now() - t0) / 1000).toFixed(2).padStart(8);
  const fmt = (x) => {
    if (x == null) return '';
    if (x instanceof Error) return `${x.name}: ${x.message}${x.stack ? '\n' + x.stack.split('\n').slice(1, 4).join('\n') : ''}`;
    if (typeof x === 'object') { try { return JSON.stringify(x); } catch { return String(x); } }
    return String(x);
  };

  function add(level, ev, data) {
    const line = `${stamp()} ${level.padEnd(5)} ${ev}${data === undefined ? '' : ' ' + fmt(data)}`;
    lines.push(line);
    if (lines.length > MAX) lines.splice(0, lines.length - MAX);
    dirty = true;
    if (level === 'ERROR') save();
  }

  function text() {
    return `Clipper ${VERSION} · session started ${started}\n` + lines.join('\n');
  }
  function save() {
    if (!dirty) return;
    dirty = false;
    try { localStorage.setItem(KEY, text().slice(-200000)); } catch {}
  }
  setInterval(save, 2000);
  addEventListener('pagehide', save);

  // שגיאות שלא נתפסו
  addEventListener('error', (e) => {
    // הודעה תמימה של הדפדפן כשגודל משתנה בזמן ציור; לא תקלה, ורק מבלבלת ביומן
    if (/ResizeObserver loop/.test(e.message)) return;
    add('ERROR', 'uncaught', e.error || `${e.message} @ ${e.filename}:${e.lineno}`);
  });
  addEventListener('unhandledrejection', (e) => add('ERROR', 'unhandled promise', e.reason));
  for (const lvl of ['error', 'warn']) {
    const orig = console[lvl].bind(console);
    console[lvl] = (...a) => { add(lvl === 'error' ? 'ERROR' : 'WARN', 'console', a.map(fmt).join(' ')); orig(...a); };
  }

  // שומר קפיאות: אם הטיימר מאחר ביותר משנייה וחצי, הדף היה תקוע (כשהלשונית גלויה)
  let last = performance.now();
  setInterval(() => {
    const now = performance.now();
    const gap = now - last;
    last = now;
    if (gap > 1500 && !document.hidden) add('WARN', 'freeze', { ms: Math.round(gap) });
  }, 500);
  document.addEventListener('visibilitychange', () => add('INFO', 'visibility', document.visibilityState));

  // פרטי הסביבה, פעם אחת בהתחלה
  const env = () => {
    const mr = typeof MediaRecorder !== 'undefined';
    const mem = navigator.deviceMemory ? `${navigator.deviceMemory}GB` : '?';
    return {
      ua: navigator.userAgent, lang: navigator.language, screen: `${screen.width}x${screen.height}@${devicePixelRatio}`,
      window: `${innerWidth}x${innerHeight}`, cores: navigator.hardwareConcurrency, memory: mem,
      mediaRecorder: mr, mp4aac: mr && MediaRecorder.isTypeSupported('video/mp4;codecs=avc1.640028,mp4a.40.2'),
      hevc: document.createElement('video').canPlayType('video/mp4; codecs="hvc1"'),
      url: location.href,
    };
  };
  add('INFO', 'start', env());

  function download() {
    save();
    const body = `${text()}\n\n===== previous session =====\n${previous || '(none)'}\n`;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([body], { type: 'text/plain' }));
    a.download = `clipper-log-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  return {
    VERSION,
    info: (ev, data) => add('INFO', ev, data),
    warn: (ev, data) => add('WARN', ev, data),
    error: (ev, data) => add('ERROR', ev, data),
    text, download,
    /** מודד משך של פעולה אסינכרונית ורושם התחלה, סוף או כישלון */
    async time(ev, fn, data) {
      const s = performance.now();
      add('INFO', ev + ' …', data);
      try {
        const r = await fn();
        add('INFO', ev + ' ok', { ms: Math.round(performance.now() - s) });
        return r;
      } catch (e) {
        add('ERROR', ev + ' failed', { ms: Math.round(performance.now() - s), err: fmt(e) });
        throw e;
      }
    },
  };
})();
