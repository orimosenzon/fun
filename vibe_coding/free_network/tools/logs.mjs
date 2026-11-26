// Reads what the network wrote down about itself.
//
//   node tools/logs.mjs                    the error log (warnings and errors)
//   node tools/logs.mjs --journal          the journal: openings, logins, logouts, Google's answers
//   node tools/logs.mjs --sessions         the sessions that are open: whose, since when, last used, on what
//
// --remote reads the live network (default: the local one). --limit 50, --user ID, --kind login,
// --hours 24, --id CODE (a request id, in either log), --persist-to DIR.
import { execFileSync } from 'node:child_process';

const args = process.argv.slice(2);
const option = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : null);
const number = (name) => {
  const value = option(name);
  if (value !== null && !/^\d+$/.test(value)) throw new Error(`${name} takes a number`);
  return value === null ? null : Number(value);
};
const limit = number('--limit') ?? 50;
const user = number('--user');
const hours = number('--hours');
const id = option('--id');
const kind = option('--kind');
if (id && !/^[\w-]+$/.test(id)) throw new Error('bad id');
if (kind && !/^[\w-]+$/.test(kind)) throw new Error('bad kind');
const remote = args.includes('--remote');

function query(sql) {
  const output = execFileSync('npx', ['wrangler', 'd1', 'execute',
    ...(remote ? ['free-network-pilot', '--env', 'live', '--remote'] : ['free-network', '--local']),
    ...(option('--persist-to') ? ['--persist-to', option('--persist-to')] : []),
    '--json', '--command', sql,
  ], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], env: { ...process.env, CI: '1' } });
  return JSON.parse(output)[0].results;
}

const time = (ms) => (ms ? new Date(ms).toISOString().replace('T', ' ').slice(0, 19) : '-');
const where = (...parts) => {
  const kept = parts.filter(Boolean);
  return kept.length ? `WHERE ${kept.join(' AND ')}` : '';
};
const since = hours ? `at > ${Date.now() - hours * 60 * 60 * 1000}` : null;
// Names make the lines readable. They stay on this computer.
const NAMED = `(SELECT CASE WHEN name = '' THEN '(no name)' ELSE name END FROM users WHERE id = user_id) AS name`;

if (args.includes('--sessions')) {
  const rows = query(`SELECT substr(token_hash, 1, 8) AS session, user_id, ${NAMED}, created_at, seen_at, expires_at, device,
      (SELECT COUNT(*) FROM push_devices WHERE session_hash = token_hash) AS pushes
    FROM sessions ${where(user ? `user_id = ${user}` : null, `expires_at > ${Date.now()}`)} ORDER BY COALESCE(seen_at, created_at) DESC LIMIT ${limit}`);
  console.log('times are UTC');
  for (const row of rows) {
    console.log(`${row.session}  member ${row.user_id} ${row.name}  began ${time(row.created_at)}  last used ${time(row.seen_at)}  ${row.device ?? 'device unknown'}${row.pushes ? '  push on' : ''}`);
  }
  console.log(`${rows.length} sessions`);
} else if (args.includes('--journal')) {
  const rows = query(`SELECT *, ${NAMED} FROM journal
    ${where(since, user ? `user_id = ${user}` : null, kind ? `kind = '${kind}'` : null, id ? `request_id = '${id}'` : null)} ORDER BY id DESC LIMIT ${limit}`).reverse();
  console.log('times are UTC');
  for (const row of rows) {
    const who = row.user_id ? `member ${row.user_id} ${row.name}` : 'nobody logged in';
    console.log(`${time(row.at)}  ${row.kind.padEnd(6)}  ${who}  [${row.session ?? 'no session'}]  ${row.device ?? ''}`);
    if (row.detail) console.log(`    ${row.detail}`);
  }
  console.log(`${rows.length} lines`);
} else {
  const rows = query(`SELECT * FROM app_log ${where(since, user ? `user_id = ${user}` : null, id ? `request_id = '${id}'` : null)} ORDER BY id DESC LIMIT ${limit}`).reverse();
  for (const row of rows) {
    console.log(`${time(row.at)} ${row.level} ${row.source} ${row.request_id ?? '-'} user=${row.user_id ?? '-'} ${row.method ?? ''} ${row.path ?? ''} ${row.status ?? ''}`);
    console.log(`  ${row.message}`);
    if (row.stack) console.log(row.stack.replace(/^/gm, '    '));
  }
  console.log(`${rows.length} entries`);
}
