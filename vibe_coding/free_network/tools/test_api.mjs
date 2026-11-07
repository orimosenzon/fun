// Runs the real worker on a throwaway local database and checks the promises the network makes.
import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const PORT = 8799;
const PROD_PORT = 8798;
const DEMO_PORT = 8796;
const STORAGE_PORT = 8795;
const UPLOADS_PORT = 8794;
const PILOT_PORT = 8793;
const YOUTUBE_PORT = 8792;
const VIDEO_ID = 'abcdefghijk';
const youtubeRequests = [];

// Stands in for YouTube, and records what the worker sends it.
const fakeYoutube = createServer((request, response) => {
  if (request.url === '/token') {
    let body = '';
    request.on('data', (part) => { body += part; });
    request.on('end', () => {
      // A code like "google-x" is a person with no address; "google-x:a@b.com" has one, "!" unverified.
      const [code, email] = new URLSearchParams(body).get('code').split(':');
      const claims = { iss: 'https://accounts.google.com', aud: code === 'wrong-audience' ? 'other' : 'test-client', sub: code };
      if (email) Object.assign(claims, { email: email.replace('!', ''), email_verified: !email.includes('!') });
      const encoded = Buffer.from(JSON.stringify(claims)).toString('base64url');
      response.setHeader('Content-Type', 'application/json');
      response.end(JSON.stringify({ id_token: `x.${encoded}.y` }));
    });
    return;
  }
  youtubeRequests.push(request);
  if (request.url.startsWith('/oembed') && request.url.includes(VIDEO_ID)) {
    response.setHeader('Content-Type', 'application/json');
    response.end(JSON.stringify({ title: 'סרטון לדוגמה' }));
  } else if (request.url === `/vi/${VIDEO_ID}/hqdefault.jpg`) {
    response.end(Buffer.from([0xff, 0xd8, 0xff, 0xd9]));
  } else {
    response.statusCode = 404;
    response.end();
  }
});
const root = mkdtempSync(join(tmpdir(), 'free-network-test-'));
const servers = [];

// Each server needs its own database directory: two cannot share one.
function startServer(port, extraArgs, { database = 'free-network', seed } = {}) {
  const state = join(root, String(port));
  const envArgs = extraArgs.filter((arg) => arg.startsWith('--env') || arg === 'demo');
  const quiet = { stdio: 'ignore', env: { ...process.env, CI: '1' } };
  execFileSync('npx', ['wrangler', 'd1', 'migrations', 'apply', database, '--local', '--persist-to', state, ...envArgs], quiet);
  if (seed) execFileSync('npx', ['wrangler', 'd1', 'execute', database, '--local', '--persist-to', state, ...envArgs, '--file', seed], quiet);
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
  const call = async function call(method, path, body) {
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
  call.cookie = () => cookie;
  return call;
}

async function signUp(name, port = PORT) {
  const call = client(port);
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

test('a comment can be an image, and it follows the audience of the post', async () => {
  const noa = await signUp('noa');
  const yael = await signUp('yael');
  const outsider = await signUp('outsider');
  await noa.call('POST', `/friends/${yael.id}`);
  await yael.call('POST', `/friends/${noa.id}`);
  const { data: { id: postId } } = await noa.call('POST', '/posts', { body: 'לחברים', audience: 'friends' });
  const image = (await upload(yael, phoneJpeg())).data.id;
  assert.equal((await yael.call('POST', `/posts/${postId}/comments`, { image })).status, 201);
  const { data } = await noa.call('GET', `/posts/${postId}`);
  assert.equal(data.comments[0].image, image);
  assert.equal(data.comments[0].body, '');
  assert.equal((await fetchImage(noa.call, image)).status, 200);
  assert.equal((await fetchImage(outsider.call, image)).status, 404);
  assert.equal((await fetchImage(anon, image)).status, 404);
});

test('a comment needs text or an image, and an image is used once', async () => {
  const roni = await signUp('roni');
  const { data: { id: postId } } = await roni.call('POST', '/posts', { body: 'x', audience: 'public' });
  assert.equal((await roni.call('POST', `/posts/${postId}/comments`, { body: '  ' })).status, 400);
  const image = (await upload(roni, phoneJpeg())).data.id;
  assert.equal((await alice.call('POST', `/posts/${postId}/comments`, { image })).status, 400);
  assert.equal((await roni.call('POST', `/posts/${postId}/comments`, { image })).status, 201);
  assert.equal((await roni.call('POST', `/posts/${postId}/comments`, { image })).status, 400);
  assert.equal((await roni.call('POST', '/posts', { body: '', audience: 'public', images: [image] })).status, 400);
});

test('removing a comment or its post really deletes the image', async () => {
  const tal = await signUp('tal');
  const { data: { id: postId } } = await tal.call('POST', '/posts', { body: 'x', audience: 'public' });
  const first = (await upload(tal, phoneJpeg())).data.id;
  const second = (await upload(tal, phoneJpeg())).data.id;
  const { data: { id: commentId } } = await tal.call('POST', `/posts/${postId}/comments`, { image: first });
  await tal.call('POST', `/posts/${postId}/comments`, { body: 'עם תמונה', image: second });
  await tal.call('DELETE', `/comments/${commentId}`);
  assert.equal((await fetchImage(tal.call, first)).status, 404);
  assert.equal((await tal.call('GET', `/posts/${postId}`)).data.comments[0].image, null);
  assert.equal((await fetchImage(anon, second)).status, 200);
  await tal.call('DELETE', `/posts/${postId}`);
  assert.equal((await fetchImage(tal.call, second)).status, 404);
});

test('a reply to a reply stays at the second level, and replies cannot cross posts', async () => {
  const omer = await signUp('omer');
  const { data: { id: postId } } = await omer.call('POST', '/posts', { body: 'x', audience: 'public' });
  const { data: { id: otherPost } } = await omer.call('POST', '/posts', { body: 'y', audience: 'public' });
  const send = (id, body, parentId) => omer.call('POST', `/posts/${id}/comments`, { body, parentId });
  const top = (await send(postId, 'ראשונה')).data.id;
  const reply = (await send(postId, 'תשובה', top)).data.id;
  await send(postId, 'תשובה לתשובה', reply);
  const { comments } = (await anon('GET', `/posts/${postId}`)).data;
  assert.deepEqual(comments.map((item) => item.parentId), [null, top, top]);
  assert.equal((await send(otherPost, 'זר', top)).status, 404);
  await omer.call('DELETE', `/comments/${reply}`);
  assert.equal((await send(postId, 'להוסרה', reply)).status, 404);
});

test('a video title and picture come through the server, without the reader', async () => {
  const dan = await signUp('dan');
  assert.equal((await dan.call('GET', `/youtube/${VIDEO_ID}`)).data.title, 'סרטון לדוגמה');
  const picture = await dan.call('GET', `/youtube/${VIDEO_ID}/thumb`);
  assert.equal(picture.status, 200);
  assert.equal(Buffer.from(picture.bytes).length, 4);
  assert.equal(youtubeRequests.length, 2);
  for (const { headers } of youtubeRequests) assert.equal(headers.cookie, undefined);
});

test('a missing video falls back quietly, and odd addresses are refused', async () => {
  assert.equal((await anon('GET', '/youtube/zzzzzzzzzzz')).data.title, null);
  assert.equal((await anon('GET', '/youtube/zzzzzzzzzzz/thumb')).status, 404);
  assert.equal((await anon('GET', '/youtube/short')).status, 404);
  assert.equal((await anon('GET', '/youtube/..%2f..%2fetc00/thumb')).status, 404);
});

const setPicture = (user, kind, imageId) => user.call('PUT', '/me/images', { kind, imageId });

test('a profile picture appears beside posts and comments, even for visitors', async () => {
  const dana = await signUp('dana');
  const { data: { id } } = await upload(dana, phoneJpeg());
  assert.equal((await setPicture(dana, 'avatar', id)).status, 200);
  const { data: post } = await dana.call('POST', '/posts', { body: 'שלום', audience: 'public' });
  await dana.call('POST', `/posts/${post.id}/comments`, { body: 'תגובה' });
  const { data } = await anon('GET', `/posts/${post.id}`);
  assert.equal(data.post.author.avatar, id);
  assert.equal(data.comments[0].author.avatar, id);
  assert.equal((await fetchImage(anon, id)).status, 200);
});

test('replacing or removing a picture deletes the old one', async () => {
  const eli = await signUp('eli');
  const first = (await upload(eli, phoneJpeg())).data.id;
  const second = (await upload(eli, phoneJpeg())).data.id;
  await setPicture(eli, 'cover', first);
  await setPicture(eli, 'cover', second);
  assert.equal((await fetchImage(anon, first)).status, 404);
  assert.equal((await eli.call('GET', '/me')).data.user.cover, second);
  await setPicture(eli, 'cover', null);
  assert.equal((await fetchImage(anon, second)).status, 404);
  assert.equal((await eli.call('GET', '/me')).data.user.cover, null);
});

test('a picture cannot be stolen, reused, or moved into a post', async () => {
  const gal = await signUp('gal');
  const own = (await upload(gal, phoneJpeg())).data.id;
  assert.equal((await setPicture(alice, 'avatar', own)).status, 400);
  assert.equal((await setPicture(gal, 'nonsense', own)).status, 400);
  await setPicture(gal, 'avatar', own);
  assert.equal((await setPicture(gal, 'cover', own)).status, 400);
  assert.equal((await gal.call('POST', '/posts', { body: '', audience: 'public', images: [own] })).status, 400);
});

test('a profile shows a stranger only public posts, and a friend everything', async () => {
  const hila = await signUp('hila');
  const ido = await signUp('ido');
  await hila.call('POST', '/posts', { body: 'לכולם', audience: 'public' });
  await hila.call('POST', '/posts', { body: 'לחברים', audience: 'friends' });
  const stranger = (await ido.call('GET', `/users/${hila.id}`)).data;
  assert.deepEqual(stranger.posts.map((post) => post.body), ['לכולם']);
  assert.equal(stranger.friendship, null);
  await ido.call('POST', `/friends/${hila.id}`);
  assert.equal((await ido.call('GET', `/users/${hila.id}`)).data.friendship, 'outgoing');
  assert.equal((await hila.call('GET', `/users/${ido.id}`)).data.friendship, 'incoming');
  await hila.call('POST', `/friends/${ido.id}`);
  const friend = (await ido.call('GET', `/users/${hila.id}`)).data;
  assert.deepEqual(friend.posts.map((post) => post.body), ['לחברים', 'לכולם']);
  assert.equal(friend.friendship, 'friends');
  assert.equal((await hila.call('GET', `/users/${hila.id}`)).data.friendship, 'self');
  assert.equal((await client()('GET', `/users/${hila.id}`)).status, 401);
  assert.equal((await ido.call('GET', '/users/99999')).status, 404);
});

test('uploads stop at the storage limit, for everyone', async () => {
  const first = await signUp('first', STORAGE_PORT);
  const second = await signUp('second', STORAGE_PORT);
  assert.equal((await upload(first, phoneJpeg())).status, 201);
  const refused = await upload(second, phoneJpeg());
  assert.equal(refused.status, 429);
  assert.match(refused.data.error, /מגבלת האחסון/);
});

test('uploads stop at the daily limit of the whole network', async () => {
  const first = await signUp('first', UPLOADS_PORT);
  const second = await signUp('second', UPLOADS_PORT);
  assert.equal((await upload(first, phoneJpeg())).status, 201);
  assert.equal((await upload(second, phoneJpeg())).status, 201);
  assert.equal((await upload(first, phoneJpeg())).status, 429);
});

// One admin for all pilot tests: logins are rate limited per address, as in production.
let bossAccount;
async function theBoss() {
  bossAccount ??= await signUp('boss', PILOT_PORT);
  return bossAccount;
}

async function openInvite(link) {
  const call = client(PILOT_PORT);
  const verified = await call('POST', '/auth/verify', { token: link.split('#token=')[1] });
  return { call, verified };
}

test('the admin invites a member, and the link works once', async () => {
  const boss = await theBoss();
  assert.equal((await boss.call('GET', '/me')).data.user.admin, true);
  const { status, data } = await boss.call('POST', '/admin/invites', {});
  assert.equal(status, 201);
  const member = await openInvite(data.link);
  assert.equal(member.verified.status, 200);
  await member.call('PUT', '/me', { name: 'חברה' });
  assert.equal((await member.call('GET', '/me')).data.user.admin, false);
  assert.equal((await openInvite(data.link)).verified.status, 400);
});

test('someone who joins by invitation starts as a friend of the admin', async () => {
  const boss = await theBoss();
  const { data } = await boss.call('POST', '/admin/invites', {});
  const { call } = await openInvite(data.link);
  assert.equal((await boss.call('GET', '/friends')).data.friends.some((user) => user.name === ''), false);
  await call('PUT', '/me', { name: 'מוזמנת' });
  await boss.call('POST', '/posts', { body: 'ברוכה הבאה', audience: 'friends' });
  assert.equal((await call('GET', '/feed')).data.posts[0].body, 'ברוכה הבאה');
  assert.ok((await boss.call('GET', '/friends')).data.friends.some((user) => user.name === 'מוזמנת'));
});

test('the waiting friend requests are counted, for the addressee only', async () => {
  const shir = await signUp('shir');
  const ziv = await signUp('ziv');
  await shir.call('POST', `/friends/${ziv.id}`);
  assert.equal((await ziv.call('GET', '/me')).data.user.friendRequests, 1);
  assert.equal((await shir.call('GET', '/me')).data.user.friendRequests, 0);
  await ziv.call('POST', `/friends/${shir.id}`);
  assert.equal((await ziv.call('GET', '/me')).data.user.friendRequests, 0);
});

// Plays both redirects of a Google sign-in, with `code` standing for who chose to sign in.
async function googleFlow(code, { user, link = false, forgedState = false } = {}) {
  const base = `http://localhost:${PILOT_PORT}/api/auth/google`;
  const session = user?.call.cookie() ?? '';
  const start = await fetch(`${base}${link ? '?link=1' : ''}`, { redirect: 'manual', headers: { Cookie: session } });
  const state = new URL(start.headers.get('location'), base).searchParams.get('state');
  const stateCookie = start.headers.get('set-cookie')?.split(';')[0] ?? '';
  const back = await fetch(`${base}/callback?code=${code}&state=${forgedState ? 'forged' : state}`, {
    redirect: 'manual',
    headers: { Cookie: [session, stateCookie].filter(Boolean).join('; ') },
  });
  const cookie = back.headers.getSetCookie().find((line) => line.startsWith('session='))?.split(';')[0];
  return { result: new URL(back.headers.get('location'), base).searchParams.get('google'), cookie };
}

test('a linked Google account opens the same account on another device', async () => {
  const lior = await signUp('lior', PILOT_PORT);
  assert.equal((await googleFlow('google-lior', { user: lior, link: true })).result, 'linked');
  assert.equal((await lior.call('GET', '/me')).data.user.googleLinked, true);
  const { result, cookie } = await googleFlow('google-lior');
  assert.equal(result, 'login');
  const me = await fetch(`http://localhost:${PILOT_PORT}/api/me`, { headers: { Cookie: cookie } });
  assert.equal((await me.json()).user.id, lior.id);
});

test('Google never creates an account, and a forged redirect gets nothing', async () => {
  const boss = await theBoss();
  const before = (await boss.call('GET', '/admin/overview')).data.users;
  const unknown = await googleFlow('google-stranger');
  assert.equal(unknown.result, 'unknown');
  assert.equal(unknown.cookie, undefined);
  assert.equal((await boss.call('GET', '/admin/overview')).data.users, before);
  const forged = await googleFlow('google-lior', { forgedState: true });
  assert.equal(forged.result, 'failed');
  assert.equal(forged.cookie, undefined);
  assert.equal((await googleFlow('wrong-audience')).result, 'failed');
});

test('one Google account belongs to one person, and it can be unlinked', async () => {
  const maya = await signUp('maya', PILOT_PORT);
  assert.equal((await googleFlow('google-lior', { user: maya, link: true })).result, 'taken');
  assert.equal((await googleFlow('google-maya', { user: maya, link: true })).result, 'linked');
  assert.equal((await maya.call('DELETE', '/me/google')).status, 200);
  assert.equal((await googleFlow('google-maya')).result, 'unknown');
});

test('an invited Google address joins linked, as a friend, and the invitation is used up', async () => {
  const boss = await theBoss();
  assert.equal((await boss.call('POST', '/admin/google-invites', { email: 'New.Friend@Gmail.com' })).data.email, 'newfriend@gmail.com');
  assert.equal((await boss.call('GET', '/admin/overview')).data.googleInvites.length, 1);
  const joined = await googleFlow('google-new:newfriend@gmail.com');
  assert.equal(joined.result, 'joined');
  const headers = { Cookie: joined.cookie };
  const me = await (await fetch(`http://localhost:${PILOT_PORT}/api/me`, { headers })).json();
  assert.equal(me.user.googleLinked, true);
  assert.equal(me.user.name, '');
  const friends = await (await fetch(`http://localhost:${PILOT_PORT}/api/friends`, { headers })).json();
  assert.deepEqual(friends.friends.map((user) => user.name), ['boss']);
  assert.equal((await boss.call('GET', '/admin/overview')).data.googleInvites.length, 0);
  assert.equal((await googleFlow('google-new:newfriend@gmail.com')).result, 'login');
  assert.equal((await googleFlow('google-other:newfriend@gmail.com')).result, 'unknown');
});

test('a Google invitation needs a verified address, and only the admin can send or cancel one', async () => {
  const boss = await theBoss();
  await boss.call('POST', '/admin/google-invites', { email: 'careful@gmail.com' });
  assert.equal((await googleFlow('google-careful:careful@gmail.com!')).result, 'unknown');
  const stranger = await signUp('stranger', PILOT_PORT);
  assert.equal((await stranger.call('POST', '/admin/google-invites', { email: 'x@gmail.com' })).status, 403);
  assert.equal((await boss.call('POST', '/admin/google-invites', { email: 'not an address' })).status, 400);
  await boss.call('DELETE', '/admin/google-invites/careful%40gmail.com');
  assert.equal((await googleFlow('google-careful:careful@gmail.com')).result, 'unknown');
});

test('without Google settings the routes do not exist', async () => {
  const response = await fetch(`http://localhost:${PROD_PORT}/api/auth/google`, { redirect: 'manual' });
  assert.equal(response.status, 404);
  assert.equal((await client(PROD_PORT)('GET', '/me')).data.googleLogin, false);
});

test('a new login link returns a member to the same account', async () => {
  const boss = await theBoss();
  const { data: { members } } = await boss.call('GET', '/admin/overview');
  const known = members.find((user) => user.name === 'חברה');
  const { data } = await boss.call('POST', '/admin/invites', { userId: known.id });
  const { call } = await openInvite(data.link);
  assert.equal((await call('GET', '/me')).data.user.id, known.id);
  assert.equal((await boss.call('POST', '/admin/invites', { userId: 9999 })).status, 404);
});

test('only the admin reaches the admin screen', async () => {
  const stranger = await signUp('stranger', PILOT_PORT);
  assert.equal((await stranger.call('GET', '/admin/overview')).status, 403);
  assert.equal((await stranger.call('POST', '/admin/invites', {})).status, 403);
  const nobody = client(PILOT_PORT);
  assert.equal((await nobody('GET', '/admin/overview')).status, 401);
  assert.equal((await nobody('POST', '/admin/invites', {})).status, 401);
  assert.equal((await nobody('GET', '/me')).data.inviteOnly, true);
  assert.equal((await client(PROD_PORT)('GET', '/admin/overview')).status, 401);
});

test('the admin screen reports usage against the limits', async () => {
  const boss = await theBoss();
  await upload(boss, phoneJpeg());
  const { usage, users } = (await boss.call('GET', '/admin/overview')).data;
  assert.equal(usage.today, 1);
  assert.equal(usage.bytes, 23);
  assert.equal(usage.storageLimit, 8 * 1024 ** 3);
  assert.equal(usage.uploadLimit, 2000);
  assert.ok(users >= 3);
});

test('the demo opens with one click and shows the invented feed', async () => {
  const call = client(DEMO_PORT);
  assert.equal((await call('GET', '/me')).data.demo, true);
  assert.equal((await call('POST', '/auth/demo')).status, 200);
  assert.equal((await call('GET', '/me')).data.user.name, 'אורח בדמו');
  assert.equal((await call('GET', '/feed')).data.posts.length, 4);
});

test('the demo refuses every write and stores nothing', async () => {
  const call = client(DEMO_PORT);
  await call('POST', '/auth/demo');
  const writes = [
    ['POST', '/posts', { body: 'x', audience: 'public' }],
    ['POST', '/posts/3/comments', { body: 'x' }],
    ['PUT', '/me', { name: 'x' }],
    ['POST', '/friends/5'],
    ['DELETE', '/friends/2'],
    ['POST', '/auth/request', { email: 'x@example.com' }],
    ['POST', '/client-log', { message: 'x' }],
  ];
  for (const [method, path, body] of writes) assert.equal((await call(method, path, body)).status, 403, path);
  assert.equal((await call('GET', '/feed')).data.posts.length, 4);
  assert.equal((await call('GET', '/friends')).data.friends.length, 3);
});

test('logging out of the demo leaves it open for the next visitor', async () => {
  const first = client(DEMO_PORT);
  await first('POST', '/auth/demo');
  await first('POST', '/auth/logout');
  const second = client(DEMO_PORT);
  await second('POST', '/auth/demo');
  assert.equal((await second('GET', '/me')).data.user.id, 1);
});

test('the demo does not serve /why, and demo login exists only there', async () => {
  for (const path of ['/why', '/why.html']) {
    const response = await fetch(`http://localhost:${DEMO_PORT}${path}`, { redirect: 'manual' });
    assert.equal(response.status, 302, path);
    assert.equal(response.headers.get('location'), '/', path);
  }
  assert.equal((await client(PROD_PORT)('POST', '/auth/demo')).status, 404);
});

test('without DEV_LOGIN a login link is never returned', async () => {
  const { status, data } = await client(PROD_PORT)('POST', '/auth/request', { email: 'y@example.com' });
  assert.equal(status, 503);
  assert.equal(data.devLink, undefined);
});

// A run cut short must not leave servers holding the ports for the next one.
function stopServers() {
  for (const server of servers) {
    try {
      process.kill(-server.pid);
    } catch {
      // Already gone.
    }
  }
}
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => {
    stopServers();
    process.exit(1);
  });
}

let failed = 0;
try {
  await new Promise((resolve) => fakeYoutube.listen(YOUTUBE_PORT, resolve));
  startServer(PORT, ['--var', 'DEV_LOGIN:1', '--var', `YOUTUBE_ORIGIN:http://localhost:${YOUTUBE_PORT}`]);
  startServer(PROD_PORT, []);
  startServer(DEMO_PORT, ['--env', 'demo'], { database: 'free-network-demo', seed: 'tools/demo_seed.sql' });
  startServer(STORAGE_PORT, ['--var', 'DEV_LOGIN:1', '--var', 'STORAGE_LIMIT_BYTES:30']);
  startServer(UPLOADS_PORT, ['--var', 'DEV_LOGIN:1', '--var', 'UPLOADS_PER_DAY:2']);
  startServer(PILOT_PORT, ['--var', 'DEV_LOGIN:1', '--var', 'INVITE_ONLY:1', '--var', 'ADMIN_EMAIL:boss@example.com',
    '--var', 'GOOGLE_CLIENT_ID:test-client', '--var', 'GOOGLE_CLIENT_SECRET:test-secret',
    '--var', `GOOGLE_TOKEN_URL:http://localhost:${YOUTUBE_PORT}/token`]);
  await Promise.all([PORT, PROD_PORT, DEMO_PORT, STORAGE_PORT, UPLOADS_PORT, PILOT_PORT].map(waitFor));
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
  stopServers();
  fakeYoutube.close();
  rmSync(root, { recursive: true, force: true });
}
console.log(failed ? `${failed} failed` : `all ${tests.length} passed`);
process.exit(failed ? 1 : 0);
