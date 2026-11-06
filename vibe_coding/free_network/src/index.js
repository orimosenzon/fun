import { Hono } from 'hono';
import { deleteCookie, getCookie, setCookie } from 'hono/cookie';

const SESSION_COOKIE = 'session';
const SESSION_MS = 90 * 24 * 60 * 60 * 1000;
const LOGIN_TOKEN_MS = 15 * 60 * 1000;
const LOGIN_REQUESTS_PER_HOUR = 5;
const PAGE_SIZE = 30;
const MAX_POST = 5000;
const MAX_COMMENT = 2000;
const MAX_NAME = 60;

const FRIEND_IDS = `
  SELECT addressee_id FROM friendships WHERE requester_id = ?1 AND status = 'accepted'
  UNION
  SELECT requester_id FROM friendships WHERE addressee_id = ?1 AND status = 'accepted'`;

function randomToken() {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

async function sha256(text) {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export function findYoutubeId(text) {
  for (const [raw] of text.matchAll(/https?:\/\/[^\s<>"]+/g)) {
    try {
      const url = new URL(raw);
      const host = url.hostname.replace(/^(www|m|music)\./, '');
      let id = null;
      if (host === 'youtu.be') id = url.pathname.slice(1);
      else if (host === 'youtube.com' || host === 'youtube-nocookie.com') {
        id = url.searchParams.get('v') ?? url.pathname.match(/^\/(?:embed|shorts|live)\/([^/]+)/)?.[1];
      }
      if (/^[\w-]{11}$/.test(id ?? '')) return id;
    } catch {
      // Not a valid URL, keep looking.
    }
  }
  return null;
}

function cleanText(value, max) {
  if (typeof value !== 'string') return null;
  const text = value.trim();
  return text && text.length <= max ? text : null;
}

async function readJson(c) {
  try {
    return await c.req.json();
  } catch {
    return {};
  }
}

function parseId(value) {
  return /^\d{1,15}$/.test(value) ? Number(value) : null;
}

function isLocalhost(c) {
  return ['localhost', '127.0.0.1'].includes(new URL(c.req.url).hostname);
}

function serializePost(row, viewer) {
  return {
    id: row.id,
    author: { id: row.author_id, name: row.author_name },
    body: row.body,
    youtubeId: row.youtube_id,
    audience: row.audience,
    createdAt: row.created_at,
    commentCount: row.comment_count,
    mine: viewer?.id === row.author_id,
  };
}

function serializeComment(row, viewer, post) {
  return {
    id: row.id,
    author: { id: row.author_id, name: row.author_name },
    body: row.body,
    createdAt: row.created_at,
    removedBy: row.removed_by,
    canRemove: !row.removed_by && !!viewer && (viewer.id === row.author_id || viewer.id === post.author_id),
  };
}

const POST_SELECT = `
  SELECT p.*, u.name AS author_name,
    (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.id) AS comment_count
  FROM posts p JOIN users u ON u.id = p.author_id`;

async function areFriends(db, a, b) {
  const row = await db
    .prepare(`SELECT 1 FROM friendships WHERE status = 'accepted'
      AND ((requester_id = ?1 AND addressee_id = ?2) OR (requester_id = ?2 AND addressee_id = ?1))`)
    .bind(a, b)
    .first();
  return !!row;
}

async function loadVisiblePost(c, id) {
  const db = c.env.DB;
  const viewer = c.get('user');
  const post = await db.prepare(`${POST_SELECT} WHERE p.id = ? AND p.deleted_at IS NULL`).bind(id).first();
  if (!post) return null;
  if (post.audience === 'public' || viewer?.id === post.author_id) return post;
  return viewer && (await areFriends(db, viewer.id, post.author_id)) ? post : null;
}

const app = new Hono();

app.use('/api/*', async (c, next) => {
  const token = getCookie(c, SESSION_COOKIE);
  if (token) {
    const user = await c.env.DB
      .prepare(`SELECT u.id, u.name, u.email FROM sessions s JOIN users u ON u.id = s.user_id
        WHERE s.token_hash = ? AND s.expires_at > ?`)
      .bind(await sha256(token), Date.now())
      .first();
    if (user) c.set('user', user);
  }
  await next();
});

async function requireUser(c, next) {
  if (!c.get('user')) return c.json({ error: 'צריך להתחבר' }, 401);
  await next();
}

app.post('/api/auth/request', async (c) => {
  const { email: rawEmail } = await readJson(c);
  const email = typeof rawEmail === 'string' ? rawEmail.trim().toLowerCase() : '';
  if (email.length > 254 || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return c.json({ error: 'כתובת המייל לא תקינה' }, 400);
  }

  const now = Date.now();
  const recent = await c.env.DB
    .prepare('SELECT COUNT(*) AS n FROM login_tokens WHERE email = ? AND created_at > ?')
    .bind(email, now - 60 * 60 * 1000)
    .first();
  if (recent.n >= LOGIN_REQUESTS_PER_HOUR) return c.json({ error: 'יותר מדי ניסיונות, נסו שוב בעוד שעה' }, 429);

  // The dev shortcut works only on localhost, even if the variable leaks into production.
  const devLogin = c.env.DEV_LOGIN === '1' && isLocalhost(c);
  if (!devLogin) return c.json({ error: 'שליחת מיילים עוד לא הוגדרה' }, 503);

  const token = randomToken();
  await c.env.DB
    .prepare('INSERT INTO login_tokens (token_hash, email, created_at, expires_at) VALUES (?, ?, ?, ?)')
    .bind(await sha256(token), email, now, now + LOGIN_TOKEN_MS)
    .run();
  return c.json({ ok: true, devLink: `${new URL(c.req.url).origin}/login#token=${token}` });
});

// The link in the email opens a page that POSTs here, so mail scanners that prefetch links
// cannot use up the token.
app.post('/api/auth/verify', async (c) => {
  const { token } = await readJson(c);
  if (typeof token !== 'string') return c.json({ error: 'הקישור לא תקין' }, 400);

  const now = Date.now();
  const used = await c.env.DB
    .prepare(`UPDATE login_tokens SET used_at = ?1
      WHERE token_hash = ?2 AND used_at IS NULL AND expires_at > ?1 RETURNING email`)
    .bind(now, await sha256(token))
    .first();
  if (!used) return c.json({ error: 'הקישור פג או שכבר השתמשו בו' }, 400);

  await c.env.DB
    .prepare('INSERT INTO users (email, created_at) VALUES (?, ?) ON CONFLICT (email) DO NOTHING')
    .bind(used.email, now)
    .run();
  const user = await c.env.DB.prepare('SELECT id FROM users WHERE email = ?').bind(used.email).first();

  const session = randomToken();
  await c.env.DB
    .prepare('INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)')
    .bind(await sha256(session), user.id, now + SESSION_MS)
    .run();
  setCookie(c, SESSION_COOKIE, session, {
    httpOnly: true,
    sameSite: 'Lax',
    secure: new URL(c.req.url).protocol === 'https:',
    path: '/',
    maxAge: SESSION_MS / 1000,
  });
  return c.json({ ok: true });
});

app.post('/api/auth/logout', async (c) => {
  const token = getCookie(c, SESSION_COOKIE);
  if (token) await c.env.DB.prepare('DELETE FROM sessions WHERE token_hash = ?').bind(await sha256(token)).run();
  deleteCookie(c, SESSION_COOKIE, { path: '/' });
  return c.json({ ok: true });
});

app.get('/api/me', (c) => {
  const user = c.get('user');
  return c.json({ user: user ? { id: user.id, name: user.name } : null });
});

app.put('/api/me', requireUser, async (c) => {
  const name = cleanText((await readJson(c)).name, MAX_NAME);
  if (!name) return c.json({ error: 'צריך שם, עד 60 תווים' }, 400);
  await c.env.DB.prepare('UPDATE users SET name = ? WHERE id = ?').bind(name, c.get('user').id).run();
  return c.json({ ok: true });
});

// How many people a friends post will reach. The composer shows this before publishing.
app.get('/api/audience', requireUser, async (c) => {
  const row = await c.env.DB
    .prepare(`SELECT COUNT(*) AS n FROM (${FRIEND_IDS})`)
    .bind(c.get('user').id)
    .first();
  return c.json({ friends: row.n });
});

// The whole feed algorithm: every post by you and your friends, newest first.
app.get('/api/feed', requireUser, async (c) => {
  const user = c.get('user');
  const before = parseId(c.req.query('before') ?? '') ?? Number.MAX_SAFE_INTEGER;
  const { results } = await c.env.DB
    .prepare(`${POST_SELECT}
      WHERE p.deleted_at IS NULL AND p.id < ?2
        AND (p.author_id = ?1 OR p.author_id IN (${FRIEND_IDS}))
      ORDER BY p.id DESC LIMIT ?3`)
    .bind(user.id, before, PAGE_SIZE)
    .all();
  return c.json({ posts: results.map((row) => serializePost(row, user)), hasMore: results.length === PAGE_SIZE });
});

app.post('/api/posts', requireUser, async (c) => {
  const input = await readJson(c);
  const body = cleanText(input.body, MAX_POST);
  if (!body) return c.json({ error: 'הפוסט ריק או ארוך מדי' }, 400);
  if (!['friends', 'public'].includes(input.audience)) return c.json({ error: 'קהל לא תקין' }, 400);

  const row = await c.env.DB
    .prepare(`INSERT INTO posts (author_id, body, youtube_id, audience, created_at)
      VALUES (?, ?, ?, ?, ?) RETURNING id`)
    .bind(c.get('user').id, body, findYoutubeId(body), input.audience, Date.now())
    .first();
  return c.json({ id: row.id }, 201);
});

// Every comment comes back, in the order it was written, with no paging and no ranking.
app.get('/api/posts/:id', async (c) => {
  const id = parseId(c.req.param('id'));
  const post = id && (await loadVisiblePost(c, id));
  if (!post) return c.json({ error: 'הפוסט לא נמצא' }, 404);

  const viewer = c.get('user');
  const { results } = await c.env.DB
    .prepare(`SELECT m.*, u.name AS author_name FROM comments m JOIN users u ON u.id = m.author_id
      WHERE m.post_id = ? ORDER BY m.id`)
    .bind(id)
    .all();
  return c.json({
    post: serializePost(post, viewer),
    comments: results.map((row) => serializeComment(row, viewer, post)),
  });
});

app.delete('/api/posts/:id', requireUser, async (c) => {
  const id = parseId(c.req.param('id'));
  const result = await c.env.DB
    .prepare('UPDATE posts SET deleted_at = ? WHERE id = ? AND author_id = ? AND deleted_at IS NULL')
    .bind(Date.now(), id, c.get('user').id)
    .run();
  if (!result.meta.changes) return c.json({ error: 'הפוסט לא נמצא' }, 404);
  return c.json({ ok: true });
});

app.post('/api/posts/:id/comments', requireUser, async (c) => {
  const id = parseId(c.req.param('id'));
  const post = id && (await loadVisiblePost(c, id));
  if (!post) return c.json({ error: 'הפוסט לא נמצא' }, 404);

  const body = cleanText((await readJson(c)).body, MAX_COMMENT);
  if (!body) return c.json({ error: 'התגובה ריקה או ארוכה מדי' }, 400);
  const row = await c.env.DB
    .prepare('INSERT INTO comments (post_id, author_id, body, created_at) VALUES (?, ?, ?, ?) RETURNING id')
    .bind(id, c.get('user').id, body, Date.now())
    .first();
  return c.json({ id: row.id }, 201);
});

app.delete('/api/comments/:id', requireUser, async (c) => {
  const id = parseId(c.req.param('id'));
  const user = c.get('user');
  const comment = id && (await c.env.DB
    .prepare(`SELECT m.author_id, p.author_id AS post_owner_id FROM comments m
      JOIN posts p ON p.id = m.post_id WHERE m.id = ? AND m.removed_by IS NULL`)
    .bind(id)
    .first());
  if (!comment) return c.json({ error: 'התגובה לא נמצאה' }, 404);

  const removedBy = user.id === comment.author_id ? 'author' : user.id === comment.post_owner_id ? 'post_owner' : null;
  if (!removedBy) return c.json({ error: 'אין הרשאה' }, 403);
  await c.env.DB.prepare('UPDATE comments SET body = NULL, removed_by = ? WHERE id = ?').bind(removedBy, id).run();
  return c.json({ ok: true });
});

app.get('/api/users', requireUser, async (c) => {
  const q = (c.req.query('q') ?? '').trim();
  if (q.length < 2) return c.json({ users: [] });
  const pattern = `%${q.replace(/[\\%_]/g, '\\$&')}%`;
  const { results } = await c.env.DB
    .prepare(`SELECT id, name FROM users WHERE name LIKE ? ESCAPE '\\' AND name <> '' AND id <> ? ORDER BY name LIMIT 20`)
    .bind(pattern, c.get('user').id)
    .all();
  return c.json({ users: results });
});

app.get('/api/friends', requireUser, async (c) => {
  const { results } = await c.env.DB
    .prepare(`SELECT u.id, u.name, f.status, f.requester_id = ?1 AS outgoing
      FROM friendships f
      JOIN users u ON u.id = CASE WHEN f.requester_id = ?1 THEN f.addressee_id ELSE f.requester_id END
      WHERE f.requester_id = ?1 OR f.addressee_id = ?1
      ORDER BY u.name`)
    .bind(c.get('user').id)
    .all();
  const person = ({ id, name }) => ({ id, name });
  return c.json({
    friends: results.filter((r) => r.status === 'accepted').map(person),
    incoming: results.filter((r) => r.status === 'pending' && !r.outgoing).map(person),
    outgoing: results.filter((r) => r.status === 'pending' && r.outgoing).map(person),
  });
});

// Asking someone who already asked you accepts their request.
app.post('/api/friends/:id', requireUser, async (c) => {
  const other = parseId(c.req.param('id'));
  const me = c.get('user').id;
  const target = other && other !== me
    && (await c.env.DB.prepare(`SELECT id FROM users WHERE id = ? AND name <> ''`).bind(other).first());
  if (!target) return c.json({ error: 'המשתמש לא נמצא' }, 404);

  const accepted = await c.env.DB
    .prepare(`UPDATE friendships SET status = 'accepted' WHERE requester_id = ? AND addressee_id = ?`)
    .bind(other, me)
    .run();
  if (accepted.meta.changes) return c.json({ status: 'accepted' });

  await c.env.DB
    .prepare(`INSERT INTO friendships (requester_id, addressee_id, status, created_at)
      VALUES (?, ?, 'pending', ?) ON CONFLICT DO NOTHING`)
    .bind(me, other, Date.now())
    .run();
  return c.json({ status: 'pending' });
});

// Cancels a request, declines one, or ends a friendship.
app.delete('/api/friends/:id', requireUser, async (c) => {
  await c.env.DB
    .prepare(`DELETE FROM friendships
      WHERE (requester_id = ?1 AND addressee_id = ?2) OR (requester_id = ?2 AND addressee_id = ?1)`)
    .bind(c.get('user').id, parseId(c.req.param('id')))
    .run();
  return c.json({ ok: true });
});

app.notFound((c) => c.json({ error: 'לא נמצא' }, 404));

export default app;
