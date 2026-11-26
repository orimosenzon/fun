const KEEP_ROWS = 5000;

// Logging must never turn a request into a failure, and it has to work when the database is
// the thing that broke. So every entry also goes to the console, which `npm run dev` saves
// to logs/dev.log.
export async function writeLog(db, entry) {
  console.log(JSON.stringify({ log: true, ...entry }));
  try {
    await db.batch([
      db.prepare(`INSERT INTO app_log (at, source, level, request_id, user_id, method, path, status, message, stack)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
        .bind(Date.now(), entry.source, entry.level, entry.requestId ?? null, entry.userId ?? null,
          entry.method ?? null, entry.path ?? null, entry.status ?? null,
          entry.message?.slice(0, 500) ?? null, entry.stack?.slice(0, 4000) ?? null),
      db.prepare('DELETE FROM app_log WHERE id <= (SELECT MAX(id) FROM app_log) - ?').bind(KEEP_ROWS),
    ]);
  } catch (error) {
    console.error(JSON.stringify({ log: true, level: 'error', message: `could not write log: ${error.message}` }));
  }
}

const JOURNAL_MS = 30 * 24 * 60 * 60 * 1000;

// The kind of device and browser, in a few words, from how the browser names itself.
export function deviceName(agent) {
  if (typeof agent !== 'string' || !agent) return null;
  const number = (pattern) => agent.match(pattern)?.[1]?.replace(/_/g, '.') ?? '';
  const system = /Android/.test(agent) ? `Android ${number(/Android ([\d.]+)/)}`
    : /iPhone|iPad|iPod/.test(agent) ? `iOS ${number(/OS ([\d_]+)/)}`
      : /Windows/.test(agent) ? 'Windows' : /CrOS/.test(agent) ? 'ChromeOS' : /Macintosh/.test(agent) ? 'Mac' : /Linux/.test(agent) ? 'Linux' : 'other';
  const browsers = [['Samsung', /SamsungBrowser\/(\d+)/], ['Edge', /Edg[A-Za-z]*\/(\d+)/], ['Opera', /OPR\/(\d+)/], ['Firefox', /(?:Firefox|FxiOS)\/(\d+)/],
    ['Chrome', /(?:Chrome|CriOS)\/(\d+)/], ['Safari', /Version\/(\d+)[\d.]* .*Safari/]];
  const found = browsers.find(([, pattern]) => pattern.test(agent));
  const browser = found ? `${found[0]} ${number(found[1])}` : 'unknown browser';
  return `${system.trim()} · ${browser}${/; wv\)/.test(agent) ? ' (webview)' : ''}`.slice(0, 80);
}

// One line in the journal. Like the error log, it must never turn a request into a failure.
export async function writeJournal(db, entry) {
  const now = Date.now();
  try {
    await db.batch([
      db.prepare('INSERT INTO journal (at, kind, user_id, session, device, detail, request_id) VALUES (?, ?, ?, ?, ?, ?, ?)')
        .bind(now, entry.kind, entry.userId ?? null, entry.session ?? null, entry.device ?? null, entry.detail?.slice(0, 300) ?? null, entry.requestId ?? null),
      db.prepare('DELETE FROM journal WHERE at < ?').bind(now - JOURNAL_MS),
    ]);
  } catch (error) {
    console.error(JSON.stringify({ log: true, level: 'error', message: `could not write journal: ${error.message}` }));
  }
}
