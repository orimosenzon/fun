// Runs the real worker on a throwaway local database and checks the promises the network makes.
import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const PORT = 8799;
const PROD_PORT = 8798;
const root = mkdtempSync(join(tmpdir(), 'free-network-test-'));
const servers = [];

// Each server needs its own database directory: two cannot share one.
function startServer(port, extraArgs) {
  const state = join(root, String(port));
  execFileSync('npx', ['wrangler', 'd1', 'migrations', 'apply', 'free-network', '--local', '--persist-to', state], {
    stdio: 'ignore',
    env: { ...process.env, CI: '1' },
  });
  const server = spawn('npx', ['wrangler', 'dev', '--port', String(port), '--inspector-port', String(port + 1000),
    '--persist-to', state, ...extraArgs], { stdio: 'ignore', detached: true });
  servers.push(server);
}

async function waitFor(port) {
  for (let i = 0; i < 120; i++) {
    try {
      await fetch(`http://localhost:${port}/api/me`);
      return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw new Error(`server on ${port} did not start`);
}

function client(port = PORT) {
  let cookie = '';
  return async function call(method, path, body) {
    const response = await fetch(`http://localhost:${port}/api${path}`, {
      method,
      headers: { ...(body ? { 'Content-Type': 'application/json' } : {}), ...(cookie ? { Cookie: cookie } : {}) },
      body: Buffer.isBuffer(body) ? body : body ? JSON.stringify(body) : undefined,
    });
    const setCookie = response.headers.get('set-cookie');
    if (setCookie) cookie = setCookie.split(';')[0];
    const bytes = await response.arrayBuffer();
    const isJson = response.headers.get('content-type')?.includes('json');
    return { status: response.status, bytes, data: isJson ? JSON.parse(Buffer.from(bytes)) : null };
  };
}

async function signUp(name) {
  const call = client();
  const { data } = await call('POST', '/auth/request', { email: `${name}@example.com` });
  const token = data.devLink.split('#token=')[1];
  assert.equal((await call('POST', '/auth/verify', { token })).status, 200);
  await call('PUT', '/me', { name });
  const { data: { user } } = await call('GET', '/me');
  return { call, id: user.id, token };
}

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

let alice, bob, carol, anon;

test('login token works once only', async () => {
  alice = await signUp('alice');
  bob = await signUp('bob');
  carol = await signUp('carol');
  anon = client();
  assert.equal((await anon('POST', '/auth/verify', { token: alice.token })).status, 400);
});

test('login requests are rate limited per email', async () => {
  const call = client();
  for (let i = 0; i < 5; i++) assert.equal((await call('POST', '/auth/request', { email: 'x@example.com' })).status, 200);
  assert.equal((await call('POST', '/auth/request', { email: 'X@example.com' })).status, 429);
});

test('friend request becomes friendship and the audience count follows', async () => {
  assert.equal((await alice.call('GET', '/audience')).data.friends, 0);
  assert.equal((await alice.call('POST', `/friends/${bob.id}`)).data.status, 'pending');
  assert.equal((await alice.call('GET', '/audience')).data.friends, 0);
  assert.equal((await bob.call('POST', `/friends/${alice.id}`)).data.status, 'accepted');
  assert.equal((await alice.call('GET', '/audience')).data.friends, 1);
  assert.equal((await bob.call('GET', '/audience')).data.friends, 1);
});

let friendsPost, publicPost;

test('a friend gets every post, newest first, with no filtering', async () => {
  const ids = [];
  for (let i = 0; i < 5; i++) {
    ids.push((await alice.call('POST', '/posts', { body: `פוסט ${i}`, audience: 'friends' })).data.id);
  }
  friendsPost = ids[0];
  publicPost = (await alice.call('POST', '/posts', { body: 'לכולם', audience: 'public' })).data.id;
  const feed = (await bob.call('GET', '/feed')).data.posts.map((post) => post.id);
  assert.deepEqual(feed, [publicPost, ...ids.reverse()]);
});

test('friends posts stay private from strangers and logged out visitors', async () => {
  assert.equal((await carol.call('GET', '/feed')).data.posts.length, 0);
  assert.equal((await carol.call('GET', `/posts/${friendsPost}`)).status, 404);
  assert.equal((await anon('GET', `/posts/${friendsPost}`)).status, 404);
  assert.equal((await carol.call('POST', `/posts/${friendsPost}/comments`, { body: 'היי' })).status, 404);
  assert.equal((await anon('GET', '/feed')).status, 401);
});

test('a public post opens without an account', async () => {
  assert.equal((await anon('GET', `/posts/${publicPost}`)).data.post.body, 'לכולם');
});

test('ending a friendship ends access', async () => {
  const dave = await signUp('dave');
  await dave.call('POST', `/friends/${alice.id}`);
  await alice.call('POST', `/friends/${dave.id}`);
  assert.equal((await dave.call('GET', `/posts/${friendsPost}`)).status, 200);
  await dave.call('DELETE', `/friends/${alice.id}`);
  assert.equal((await dave.call('GET', `/posts/${friendsPost}`)).status, 404);
});

test('all comments return in the order written, links intact', async () => {
  const bodies = ['ראשונה', 'https://example.com/a', 'שלישית'];
  for (const [i, body] of bodies.entries()) {
    await [bob, carol, bob][i].call('POST', `/posts/${publicPost}/comments`, { body });
  }
  const { comments } = (await anon('GET', `/posts/${publicPost}`)).data;
  assert.deepEqual(comments.map((item) => item.body), bodies);
});

test('a removed comment keeps its place and says who removed it', async () => {
  const before = (await alice.call('GET', `/posts/${publicPost}`)).data.comments;
  const [first, second] = before;
  assert.equal((await carol.call('DELETE', `/comments/${first.id}`)).status, 403);
  assert.equal((await alice.call('DELETE', `/comments/${first.id}`)).status, 200);
  assert.equal((await carol.call('DELETE', `/comments/${second.id}`)).status, 200);
  const after = (await anon('GET', `/posts/${publicPost}`)).data.comments;
  assert.deepEqual(after.map((item) => item.id), before.map((item) => item.id));
  assert.deepEqual(after.map((item) => item.removedBy), ['post_owner', 'author', null]);
  assert.deepEqual(after.map((item) => item.body), [null, null, 'שלישית']);
});

test('a YouTube link in the text becomes a video', async () => {
  const urls = ['https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=3', 'https://youtu.be/dQw4w9WgXcQ', 'https://youtube.com/shorts/dQw4w9WgXcQ'];
  for (const url of urls) {
    const { id } = (await alice.call('POST', '/posts', { body: `ראו ${url}`, audience: 'public' })).data;
    assert.equal((await anon('GET', `/posts/${id}`)).data.post.youtubeId, 'dQw4w9WgXcQ');
  }
  const { id } = (await alice.call('POST', '/posts', { body: 'https://evil.example/youtube.com/watch?v=dQw4w9WgXcQ', audience: 'public' })).data;
  assert.equal((await anon('GET', `/posts/${id}`)).data.post.youtubeId, null);
});

test('a YouTube link in a comment becomes a video, until the comment is removed', async () => {
  const { id } = (await bob.call('POST', `/posts/${publicPost}/comments`, { body: 'https://youtu.be/dQw4w9WgXcQ' })).data;
  const find = async () => (await anon('GET', `/posts/${publicPost}`)).data.comments.find((item) => item.id === id);
  assert.equal((await find()).youtubeId, 'dQw4w9WgXcQ');
  await bob.call('DELETE', `/comments/${id}`);
  assert.equal((await find()).youtubeId, null);
});

test('only the author can delete a post, and it disappears', async () => {
  const { id } = (await alice.call('POST', '/posts', { body: 'זמני', audience: 'public' })).data;
  assert.equal((await bob.call('DELETE', `/posts/${id}`)).status, 404);
  assert.equal((await alice.call('DELETE', `/posts/${id}`)).status, 200);
  assert.equal((await anon('GET', `/posts/${id}`)).status, 404);
});

// A tiny JPEG shaped like a phone photo: an Exif block holding a location, then the picture data.
function phoneJpeg() {
  const exif = Buffer.from('Exif\0\0GPS-32.47,34.97');
  return Buffer.concat([
    Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x04, 0x4a, 0x46]),
    Buffer.from([0xff, 0xe1, (exif.length + 2) >> 8, (exif.length + 2) & 255]), exif,
    Buffer.from([0xff, 0xdb, 0x00, 0x04, 0x01, 0x02]),
    Buffer.from([0xff, 0xda, 0x00, 0x02, 0x11, 0x22, 0x33, 0xff, 0xd9]),
  ]);
}

const upload = (user, data) => user.call('POST', '/images', data);
const fetchImage = (user, id) => user('GET', `/images/${id}`);

test('location data never survives an upload', async () => {
  const { status, data } = await upload(alice, phoneJpeg());
  assert.equal(status, 201);
  const { id } = (await alice.call('POST', '/posts', { body: '', audience: 'public', images: [data.id] })).data;
  assert.deepEqual((await anon('GET', `/posts/${id}`)).data.post.images, [data.id]);
  const stored = Buffer.from((await anon('GET', `/images/${data.id}`)).bytes);
  assert.ok(!stored.includes('GPS'));
  assert.ok(stored.includes(Buffer.from([0xff, 0xda, 0x00, 0x02, 0x11, 0x22, 0x33, 0xff, 0xd9])));
});

test('only JPEG files within the size limit are accepted', async () => {
  assert.equal((await upload(alice, Buffer.from('<svg onload="alert(1)"/>'))).status, 415);
  assert.equal((await upload(alice, Buffer.alloc(1024 * 1024 + 1))).status, 413);
});

test('an image follows the audience of its post', async () => {
  const image = (await upload(alice, phoneJpeg())).data.id;
  assert.equal((await fetchImage(alice.call, image)).status, 200);
  assert.equal((await fetchImage(bob.call, image)).status, 404);
  await alice.call('POST', '/posts', { body: 'פרטי', audience: 'friends', images: [image] });
  assert.equal((await fetchImage(bob.call, image)).status, 200);
  assert.equal((await fetchImage(carol.call, image)).status, 404);
  assert.equal((await fetchImage(anon, image)).status, 404);
});

test('an image cannot be attached twice or by someone else', async () => {
  const image = (await upload(alice, phoneJpeg())).data.id;
  assert.equal((await bob.call('POST', '/posts', { body: 'x', audience: 'public', images: [image] })).status, 400);
  assert.equal((await alice.call('POST', '/posts', { body: 'x', audience: 'friends', images: [image] })).status, 201);
  assert.equal((await alice.call('POST', '/posts', { body: 'y', audience: 'public', images: [image] })).status, 400);
});

test('deleting a post deletes its images', async () => {
  const image = (await upload(alice, phoneJpeg())).data.id;
  const { id } = (await alice.call('POST', '/posts', { body: 'x', audience: 'public', images: [image] })).data;
  assert.equal((await fetchImage(anon, image)).status, 200);
  await alice.call('DELETE', `/posts/${id}`);
  assert.equal((await fetchImage(alice.call, image)).status, 404);
});

test('a post needs text or an image, and at most four images', async () => {
  assert.equal((await alice.call('POST', '/posts', { body: '  ', audience: 'public' })).status, 400);
  const images = [];
  for (let i = 0; i < 5; i++) images.push((await upload(alice, phoneJpeg())).data.id);
  assert.equal((await alice.call('POST', '/posts', { body: 'x', audience: 'public', images })).status, 400);
});

test('name search treats wildcards literally', async () => {
  assert.equal((await bob.call('GET', '/users?q=al')).data.users.length, 1);
  assert.equal((await bob.call('GET', '/users?q=%25%25')).data.users.length, 0);
});

test('a crash reaches the log with a code the user can quote', async () => {
  const state = join(root, String(PROD_PORT));
  const wrangler = ['wrangler', 'd1', 'execute', 'free-network', '--local', '--persist-to', state, '--command'];
  execFileSync('npx', [...wrangler, 'DROP TABLE images'], { stdio: 'ignore', env: { ...process.env, CI: '1' } });
  const { status, data } = await client(PROD_PORT)('GET', '/posts/1');
  assert.equal(status, 500);
  assert.match(data.requestId, /^[0-9a-f]{8}$/);
  assert.ok(!JSON.stringify(data).includes('images'));
  const log = execFileSync('node', ['tools/logs.mjs', '--persist-to', state, '--id', data.requestId], { encoding: 'utf8' });
  assert.match(log, /no such table: images/);
  assert.match(log, /GET \/api\/posts\/1 500/);
});

test('browser errors are logged, and post text is not', async () => {
  const state = join(root, String(PORT));
  await client()('POST', '/client-log', { message: 'TypeError: boom', stack: 'at app.js:1', path: '/p/1' });
  await alice.call('POST', '/posts', { body: 'טקסט סודי', audience: 'nobody' });
  const log = execFileSync('node', ['tools/logs.mjs', '--persist-to', state], { encoding: 'utf8' });
  assert.match(log, /TypeError: boom/);
  assert.match(log, /POST \/api\/posts 400/);
  assert.ok(!log.includes('טקסט סודי'));
});

test('without DEV_LOGIN a login link is never returned', async () => {
  const { status, data } = await client(PROD_PORT)('POST', '/auth/request', { email: 'y@example.com' });
  assert.equal(status, 503);
  assert.equal(data.devLink, undefined);
});

let failed = 0;
try {
  startServer(PORT, ['--var', 'DEV_LOGIN:1']);
  startServer(PROD_PORT, []);
  await Promise.all([waitFor(PORT), waitFor(PROD_PORT)]);
  for (const [name, fn] of tests) {
    try {
      await fn();
      console.log(`ok   ${name}`);
    } catch (error) {
      failed++;
      console.log(`FAIL ${name}\n     ${error.message}`);
    }
  }
} finally {
  for (const server of servers) {
    try {
      process.kill(-server.pid);
    } catch {
      // Already gone.
    }
  }
  rmSync(root, { recursive: true, force: true });
}
console.log(failed ? `${failed} failed` : `all ${tests.length} passed`);
process.exit(failed ? 1 : 0);
