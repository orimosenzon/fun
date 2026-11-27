const KEEP_ROWS = 5000;
const HOUR_MS = 60 * 60 * 1000;
// Anyone on the internet can make the server write a line: a browser reports an error, a
// request is refused, the network is opened by someone who is not logged in. So what one member,
// or everyone who is not logged in together, can write in an hour is limited. A crash of the
// server itself is always written.
const LINES_PER_HOUR = 300;

// Logging must never turn a request into a failure, and it has to work when the database is
// the thing that broke. So every entry also goes to the console, which `npm run dev` saves
// to logs/dev.log. Each kind of line (crashes, refusals, browser errors) keeps its own newest
// lines, so a flood of one kind never pushes out another.
export async function writeLog(db, entry) {
  console.log(JSON.stringify({ log: true, ...entry }));
  const now = Date.now();
  const always = entry.source === 'server' && entry.level === 'error';
  try {
    await db.batch([
      db.prepare(`INSERT INTO app_log (at, source, level, request_id, user_id, method, path, status, message, stack)
        SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10
        WHERE ?11 OR (SELECT COUNT(*) FROM app_log WHERE source = ?2 AND level = ?3 AND user_id IS ?5 AND at > ?12) < ?13`)
        .bind(now, entry.source, entry.level, entry.requestId ?? null, entry.userId ?? null,
          entry.method ?? null, entry.path ?? null, entry.status ?? null,
          entry.message?.slice(0, 500) ?? null, entry.stack?.slice(0, 4000) ?? null,
          always ? 1 : 0, now - HOUR_MS, LINES_PER_HOUR),
      db.prepare(`DELETE FROM app_log WHERE source = ?1 AND level = ?2
        AND id <= (SELECT id FROM app_log WHERE source = ?1 AND level = ?2 ORDER BY id DESC LIMIT 1 OFFSET ?3)`)
        .bind(entry.source, entry.level, KEEP_ROWS),
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

// One line in the journal. Like the error log, it must never turn a request into a failure, and
// it holds the same limit: so many lines an hour for one member, and for all who are not logged
// in together.
export async function writeJournal(db, entry) {
  const now = Date.now();
  try {
    await db.batch([
      db.prepare(`INSERT INTO journal (at, kind, user_id, session, device, detail, request_id)
        SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7
        WHERE (SELECT COUNT(*) FROM journal WHERE user_id IS ?3 AND at > ?8) < ?9`)
        .bind(now, entry.kind, entry.userId ?? null, entry.session ?? null, entry.device ?? null, entry.detail?.slice(0, 300) ?? null, entry.requestId ?? null,
          now - HOUR_MS, LINES_PER_HOUR),
      db.prepare('DELETE FROM journal WHERE at < ?').bind(now - JOURNAL_MS),
    ]);
  } catch (error) {
    console.error(JSON.stringify({ log: true, level: 'error', message: `could not write journal: ${error.message}` }));
  }
}
