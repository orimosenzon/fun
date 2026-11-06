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
