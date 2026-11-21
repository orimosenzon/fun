// Builds the little network the film is shot in: invented people, posts and comments on a
// throwaway local database, then leaves the server running for the camera (film/shoot.py).
//   node film/stage.mjs          build and serve on localhost:8811
// Dana is the one the film follows. She joins the way a real guest does: by an emailed link.
import { execFileSync, spawn } from 'node:child_process';
import { readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, '..');
const STATE = join(HERE, '.state');
const PORT = 8811;
const SITE = `http://localhost:${PORT}`;
const HOUR = 3600 * 1000;
const wrangler = (...args) => execFileSync('npx', ['wrangler', ...args], { cwd: ROOT, stdio: 'ignore', env: { ...process.env, CI: '1' } });
const sql = (command) => wrangler('d1', 'execute', 'free-network', '--local', '--persist-to', STATE, '--command', command);

function serve() {
  const server = spawn('npx', ['wrangler', 'dev', '--port', String(PORT), '--inspector-port', String(PORT + 1000), '--persist-to', STATE,
    '--var', 'DEV_LOGIN:1', '--var', 'INVITE_ONLY:1', '--var', 'ADMIN_EMAIL:ori@film.example', '--var', 'GOOGLE_CLIENT_ID:film'],
  { cwd: ROOT, stdio: 'ignore', detached: true });
  server.unref();
  return server;
}

async function ready() {
  for (let i = 0; i < 120; i++) {
    try {
      await fetch(`${SITE}/api/me`);
      return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  throw new Error('the server did not start');
}

function client() {
  let cookie = '';
  const call = async (method, path, body) => {
    const response = await fetch(`${SITE}/api${path}`, {
      method,
      headers: { ...(body ? { 'Content-Type': Buffer.isBuffer(body) ? 'image/jpeg' : 'application/json' } : {}), ...(cookie ? { Cookie: cookie } : {}) },
      body: Buffer.isBuffer(body) ? body : body ? JSON.stringify(body) : undefined,
    });
    const setCookie = response.headers.get('set-cookie');
    if (setCookie) cookie = setCookie.split(';')[0];
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(`${method} ${path}: ${response.status} ${data.error ?? ''}`);
    return data;
  };
  call.cookie = () => cookie;
  return call;
}

async function join_(key, name) {
  const call = client();
  const { devLink } = await call('POST', '/auth/request', { email: `${key}@film.example` });
  await call('POST', '/auth/verify', { token: devLink.split('#token=')[1] });
  await call('PUT', '/me', { name });
  const { user } = await call('GET', '/me');
  return { call, id: user.id, name };
}

const photo = (name) => readFileSync(join(HERE, 'photos', `${name}.jpg`));
const upload = async (who, name) => (await who.call('POST', '/images', photo(name))).id;

rmSync(STATE, { recursive: true, force: true });
wrangler('d1', 'migrations', 'apply', 'free-network', '--local', '--persist-to', STATE);
let server = serve();
await ready();

const ori = await join_('ori', 'אורי מוסנזון');
const people = {};
for (const [key, name] of [['noa', 'נועה לוי'], ['yoav', 'יואב שגיא'], ['tamar', 'תמר אבני'], ['ron', 'רון גלעד'],
  ['maya', 'מאיה פרץ'], ['gil', 'גיל דור'], ['yael', 'יעל נחום'], ['amit', 'עמית שחר'], ['shira', 'שירה בן דוד']]) {
  people[key] = await join_(key, name);
}
const { noa, yoav, tamar, ron, maya, gil, yael, shira } = people;

// The invitation by email, exactly as the admin screen makes it.
const invite = await ori.call('POST', '/admin/email-invites', { email: 'dana.cohen@gmail.com', name: 'דנה כהן' });
const dana = { call: client(), name: 'דנה כהן' };
await dana.call('POST', '/auth/verify', { token: invite.link.split('#token=')[1] });
dana.id = (await dana.call('GET', '/me')).user.id;

// One circle of friends: everyone knows everyone, as in the first days of a small network.
const circle = [ori, dana, noa, yoav, tamar, ron, maya, gil, yael];
for (const [i, one] of circle.entries()) {
  for (const other of circle.slice(i + 1)) {
    await one.call('POST', `/friends/${other.id}`);
    await other.call('POST', `/friends/${one.id}`);
  }
}
await shira.call('POST', `/friends/${dana.id}`);

await noa.call('PUT', '/me/images', { kind: 'avatar', imageId: await upload(noa, 'sunset') });
await yoav.call('PUT', '/me/images', { kind: 'avatar', imageId: await upload(yoav, 'cat') });

const places = [[dana, 32.47, 34.97], [ori, 32.52, 35.0], [noa, 32.08, 34.78], [tamar, 31.77, 35.21], [ron, 52.52, 13.4], [maya, 40.71, -74.01], [yael, 48.85, 2.35]];
for (const [who, lat, lng] of places) await who.call('PUT', '/me/place', { lat, lng });
await noa.call('PUT', '/me/status', { text: 'חזרנו מהצפון, הכול ירוק. מי בא לקפה בשבת?' });
await maya.call('PUT', '/me/status', { text: 'שבוע ראשון בעבודה החדשה, מתרגלת לרכבת התחתית' });

// Posts in the order they were written. `ago` is how many hours back each one is moved afterwards.
const posts = [];
async function post(who, ago, body, audience, pictures = []) {
  const images = [];
  for (const name of pictures) images.push(await upload(who, name));
  const { id } = await who.call('POST', '/posts', { body, audience, images });
  posts.push({ id, ago });
  return id;
}
const comments = [];
async function comment(who, postId, ago, body, parentId) {
  const { id } = await who.call('POST', `/posts/${postId}/comments`, { body, parentId });
  comments.push({ id, ago });
  return id;
}

const electrician = await post(yoav, 74, 'מישהו מכיר חשמלאי טוב באזור? נשרף לנו הלוח בבית.', 'friends');
await comment(noa, electrician, 73, 'עבד אצלנו מישהו מצוין, שולחת לך את הטלפון.');
await comment(tamar, electrician, 72, 'גם אני אשמח לשם שלו.');
const bread = await post(tamar, 50, 'אפיתי לחם מחמצת ראשון. יצא יותר טוב ממה שציפיתי.', 'public', ['bread']);
await comment(gil, bread, 49, 'נראה מעולה. יש מתכון?');
const garden = await post(noa, 27, 'מחר בבוקר נפגשים בגינה הקהילתית לשתילה. מביאים כפפות, יהיה קפה.', 'public', ['garden']);
const when = await comment(yoav, garden, 26.5, 'באיזו שעה?');
await comment(noa, garden, 26.2, 'בתשע. מי שמגיע מוקדם עוזר לפרוק את השתילים.', when);
await comment(tamar, garden, 25, 'אני מביאה עוגה.');
await comment(ron, garden, 23, 'למי שרוצה לקרוא על גינות קהילתיות: https://he.wikipedia.org/wiki/גינה_קהילתית');
await comment(yoav, garden, 22.5, 'תודה, נתראה שם.', when);
await post(ron, 21, 'שקיעה מהחוף אתמול.', 'friends', ['sunset']);
await post(gil, 6, 'סרט קצר וחמוד לשבת בבוקר עם הילדים:\nhttps://www.youtube.com/watch?v=aqz-KE-bpKQ', 'public');
await post(yoav, 3, 'נמצא חתלתול אפור ליד תחנת האוטובוס. רגוע וחברותי. מי מכיר?', 'friends', ['cat']);
await post(ori, 1, 'ברוכים הבאים לרשת החופשית. הכול כאן עוד חדש, אז ספרו לי מה חסר ומה מפריע.', 'friends', ['grove']);

// The clock is turned back with the server down, because two programs cannot share the database.
process.kill(-server.pid);
await new Promise((resolve) => setTimeout(resolve, 1500));
const shift = (table, rows) => rows.map(({ id, ago }) => `UPDATE ${table} SET created_at = created_at - ${Math.round(ago * HOUR)} WHERE id = ${id};`).join(' ');
sql(`${shift('posts', posts)} ${shift('comments', comments)}
  UPDATE users SET status_at = status_at - ${5 * HOUR} WHERE id = ${noa.id};
  UPDATE users SET status_at = status_at - ${30 * HOUR} WHERE id = ${maya.id};
  UPDATE users SET created_at = created_at - ${40 * 24 * HOUR} WHERE id NOT IN (${dana.id}, ${gil.id});
  UPDATE users SET place_at = place_at - ${60 * 24 * HOUR} WHERE id <> ${ron.id};`);
server = serve();
await ready();

// A second invitation waits unanswered, for the picture of the admin screen.
await ori.call('POST', '/admin/email-invites', { email: 'omer@example.org', name: 'עומר רז' });
writeFileSync(join(HERE, '.stage.json'), JSON.stringify({
  site: SITE, pid: server.pid, garden, invite: invite.link,
  sessions: { dana: dana.call.cookie(), ori: ori.call.cookie() }, ids: { dana: dana.id, ori: ori.id, noa: noa.id },
}, null, 1));
console.log(`the stage is up on ${SITE} (server pid ${server.pid}); film/.stage.json has the sessions`);
