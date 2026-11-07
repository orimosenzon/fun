// Prints a one-time login link for the admin of the pilot. Everyone else is invited from /admin.
import { execFileSync } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';

const ADMIN_EMAIL = 'admin@invite.invalid';
const SITE = 'https://free-network.orimosenzon.workers.dev';
const WEEK_MS = 7 * 24 * 60 * 60 * 1000;

const token = randomBytes(32).toString('base64url');
const hash = createHash('sha256').update(token).digest('hex');
const now = Date.now();
execFileSync('npx', ['wrangler', 'd1', 'execute', 'free-network-pilot', '--env', 'pilot', '--remote', '--command',
  `INSERT INTO login_tokens (token_hash, email, created_at, expires_at) VALUES ('${hash}', '${ADMIN_EMAIL}', ${now}, ${now + WEEK_MS})`],
{ stdio: 'ignore' });
console.log(`${SITE}/login#token=${token}`);
