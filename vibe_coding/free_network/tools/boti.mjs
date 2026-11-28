// Boti is the account of the AI (Claude) that builds the network with Ori, and writes in its own
// name. Its session lives outside the repository, like every other key.
//
//   node tools/boti.mjs           who Boti is, its friends, and what waits to be sent
//   node tools/boti.mjs send      sends every waiting message
//
// A message waits as a text file named by the member's id, in ~/.config/free-network/boti-outbox/.
// Boti is marked as the bot of the network (migrations/0025_bot.sql), so it may write to every
// member, and every member to it.
import { readFileSync, readdirSync, rmSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

const SITE = 'https://free-network.orimosenzon.workers.dev';
const HOME = join(homedir(), '.config/free-network');
const OUTBOX = join(HOME, 'boti-outbox');
const cookie = readFileSync(join(HOME, 'boti-session'), 'utf8').trim();

async function call(method, path, body) {
  const response = await fetch(`${SITE}/api${path}`, {
    method,
    headers: { Cookie: cookie, ...(body ? { 'Content-Type': 'application/json' } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json();
  if (!response.ok) throw new Error(`${method} ${path}: ${response.status} ${data.error ?? ''}`);
  return data;
}

const { user } = await call('GET', '/me');
if (!user) throw new Error('the session of Boti has ended: it needs a new login link');
const { friends, outgoing } = await call('GET', '/friends');
console.log(`${user.name} (member ${user.id}). friends: ${friends.map((friend) => friend.name).join(', ') || 'none'}. asked: ${outgoing.map((other) => other.name).join(', ') || 'nobody'}`);
for (const file of readdirSync(OUTBOX).filter((name) => /^\d+\.txt$/.test(name))) {
  const id = Number(file.split('.')[0]);
  const { user: reader, canSend } = await call('GET', `/messages/${id}`);
  if (!canSend) console.log(`waiting: a message to ${reader.name}, who cannot be written to`);
  else if (process.argv[2] !== 'send') console.log(`ready: a message to ${reader.name}`);
  else {
    await call('POST', `/messages/${id}`, { body: readFileSync(join(OUTBOX, file), 'utf8').trim() });
    rmSync(join(OUTBOX, file));
    console.log(`sent to ${reader.name}`);
  }
}
