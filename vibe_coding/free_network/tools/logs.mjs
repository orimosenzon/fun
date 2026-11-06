// Reads the error log. Usage: node tools/logs.mjs [--remote] [--limit 50] [--id CODE] [--persist-to DIR]
import { execFileSync } from 'node:child_process';

const args = process.argv.slice(2);
const option = (name) => (args.includes(name) ? args[args.indexOf(name) + 1] : null);
const limit = Number(option('--limit') ?? 50);
const id = option('--id');
if (id && !/^[\w-]+$/.test(id)) throw new Error('bad id');

const where = id ? `WHERE request_id = '${id}'` : '';
const output = execFileSync('npx', ['wrangler', 'd1', 'execute', 'free-network',
  args.includes('--remote') ? '--remote' : '--local',
  ...(option('--persist-to') ? ['--persist-to', option('--persist-to')] : []),
  '--json', '--command', `SELECT * FROM app_log ${where} ORDER BY id DESC LIMIT ${limit}`,
], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], env: { ...process.env, CI: '1' } });

const rows = JSON.parse(output)[0].results.reverse();
for (const row of rows) {
  const time = new Date(row.at).toISOString().replace('T', ' ').slice(0, 19);
  console.log(`${time} ${row.level} ${row.source} ${row.request_id ?? '-'} user=${row.user_id ?? '-'} ${row.method ?? ''} ${row.path ?? ''} ${row.status ?? ''}`);
  console.log(`  ${row.message}`);
  if (row.stack) console.log(row.stack.replace(/^/gm, '    '));
}
console.log(`${rows.length} entries`);
