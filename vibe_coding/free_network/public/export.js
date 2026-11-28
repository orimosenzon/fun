// Everything a member has here, taken out as one file: a ZIP with the data in an open format
// (data.json), a page that reads without the network (index.html), and the pictures. It is
// built in the member's own browser, from what the server hands only to them.

const FORMAT = 'free-network-export';
const FETCH_TOGETHER = 4;

// ---- A ZIP file, the simplest kind: files stored as they are. Pictures are already packed.

const CRC = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

export function zip(files, when = new Date()) {
  const encoder = new TextEncoder();
  const time = (when.getHours() << 11) | (when.getMinutes() << 5) | (when.getSeconds() >> 1);
  const date = ((when.getFullYear() - 1980) << 9) | ((when.getMonth() + 1) << 5) | when.getDate();
  const parts = [];
  const central = [];
  let offset = 0;
  for (const file of files) {
    const name = encoder.encode(file.name);
    const data = typeof file.data === 'string' ? encoder.encode(file.data) : file.data;
    const sum = crc32(data);
    // What both headers say about the file: version, UTF-8 names, stored, time, sum and sizes.
    const about = new DataView(new ArrayBuffer(26));
    [[0, 20], [2, 0x0800], [4, 0], [6, time], [8, date]].forEach(([at, value]) => about.setUint16(at, value, true));
    about.setUint32(10, sum, true);
    about.setUint32(14, data.length, true);
    about.setUint32(18, data.length, true);
    about.setUint16(22, name.length, true);
    const local = new DataView(new ArrayBuffer(4));
    local.setUint32(0, 0x04034b50, true);
    parts.push(local.buffer, about.buffer, name, data);
    const entry = new DataView(new ArrayBuffer(6));
    entry.setUint32(0, 0x02014b50, true);
    entry.setUint16(4, 20, true);
    const where = new DataView(new ArrayBuffer(14));
    where.setUint32(10, offset, true);
    central.push(entry.buffer, about.buffer, where.buffer, name);
    offset += 30 + name.length + data.length;
  }
  const centralSize = central.reduce((n, part) => n + part.byteLength, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true);
  end.setUint16(8, files.length, true);
  end.setUint16(10, files.length, true);
  end.setUint32(12, centralSize, true);
  end.setUint32(16, offset, true);
  return new Blob([...parts, ...central, end.buffer], { type: 'application/zip' });
}

// ---- The page that reads without the network. Plain HTML, no scripts, pictures from the folder.

const esc = (text) => String(text ?? '').replace(/[&<>"']/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[ch]);
const when = (ms) => (ms ? new Intl.DateTimeFormat('he-IL', { dateStyle: 'medium', timeStyle: 'short' }).format(ms) : '');
const words = (text) => (text ? `<p class="words" dir="auto">${esc(text)}</p>` : '');
const picture = (id, have) => (!id ? '' : have.has(id) ? `<img src="images/${esc(id)}.jpg" alt="">` : '<p class="gone">(תמונה שלא נשמרה)</p>');
const AUDIENCE = { public: 'ציבורי', friends: 'חברים' };

function commentHtml(item, have) {
  if (item.removed) return `<div class="comment${item.parentId ? ' reply' : ''}"><p class="gone">תגובה שהוסרה</p></div>`;
  return `<div class="comment${item.parentId ? ' reply' : ''}"><p class="who"><b>${esc(item.author.name)}</b> · ${when(item.createdAt)}${item.editedAt ? ' · נערך' : ''}</p>${words(item.body)}${picture(item.image, have)}</div>`;
}

function postHtml(post, have) {
  const felt = post.reactions.map((group) => `${esc(group.kind)} ${group.people.map((person) => esc(person.name)).join(', ')}`).join(' · ');
  const quoted = !post.shared ? '' : post.shared.post
    ? `<blockquote><p class="who"><b>${esc(post.shared.post.author.name)}</b> · ${when(post.shared.post.createdAt)}</p>${words(post.shared.post.body)}${post.shared.post.images.map((id) => picture(id, have)).join('')}</blockquote>`
    : '<blockquote><p class="gone">הפוסט המקורי כבר לא זמין</p></blockquote>';
  return `<article id="post-${post.id}"><p class="who">${when(post.createdAt)} · ${AUDIENCE[post.audience] ?? esc(post.audience)}${post.editedAt ? ' · נערך' : ''}</p>
${words(post.body)}${post.images.map((id) => picture(id, have)).join('')}
${post.page ? `<p class="link">דף חי: <a href="${esc(post.page.url)}" dir="ltr">${esc(post.page.title || post.page.url)}</a></p>` : ''}${quoted}
${felt ? `<p class="felt">${felt}</p>` : ''}
${post.comments.length ? `<div class="comments">${post.comments.map((item) => commentHtml(item, have)).join('\n')}</div>` : ''}</article>`;
}

export function readerPage(data, have) {
  const friends = data.friends.filter((friend) => friend.status === 'friends');
  const sections = [
    ['posts', `הפוסטים שלי (${data.posts.length})`, data.posts.slice().reverse().map((post) => postHtml(post, have)).join('\n')],
    ['comments', `תגובות שכתבתי אצל אחרים (${data.comments.length})`, data.comments.slice().reverse().map((item) => `<article><p class="who">${when(item.createdAt)} · על פוסט של <b>${esc(item.post.author.name)}</b>${item.editedAt ? ' · נערך' : ''}</p>
${item.post.words ? `<blockquote>${words(item.post.words)}</blockquote>` : ''}${words(item.body)}${picture(item.image, have)}</article>`).join('\n')],
    ['messages', `הודעות פרטיות (${data.conversations.length} שיחות)`, data.conversations.map((talk) => `<details><summary>${esc(talk.with.name)} · ${talk.messages.length} הודעות</summary>
${talk.messages.map((message) => `<div class="message ${message.mine ? 'mine' : 'theirs'}"><p class="who"><b>${esc(message.mine ? data.profile.name : talk.with.name)}</b> · ${when(message.at)}</p>${message.removed ? '<p class="gone">הודעה שהוסרה</p>' : `${words(message.body)}${picture(message.image, have)}`}</div>`).join('\n')}</details>`).join('\n')],
    ['friends', `חברים (${friends.length})`, friends.length ? `<ul>${friends.map((friend) => `<li>${esc(friend.name)}${friend.since ? ` · מאז ${when(friend.since)}` : ''}</li>`).join('')}</ul>` : ''],
  ];
  return `<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(data.profile.name)} · הרשת החופשית</title>
<style>
body { font-family: system-ui, Arial, sans-serif; line-height: 1.6; color: #0f2a38; background: #f6ecdc; margin: 0; padding: 24px 16px 80px; }
main { max-width: 720px; margin: 0 auto; }
h1 { margin: 0 0 4px; } h2 { margin: 40px 0 12px; border-bottom: 2px solid #e2cfae; padding-bottom: 6px; }
nav a { margin-inline-end: 14px; color: #1e4a63; }
article, details { background: #fffaf1; border-radius: 14px; padding: 14px 18px; margin: 0 0 14px; }
.who, .felt, .link, .lead { color: #5d6b70; font-size: 14px; margin: 0 0 6px; }
.words { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0 0 8px; }
img { max-width: 100%; border-radius: 10px; display: block; margin: 8px 0; }
blockquote { margin: 8px 0; padding: 8px 14px; border-inline-start: 3px solid #e2cfae; }
.comments { border-top: 1px solid #eadcc3; margin-top: 10px; padding-top: 10px; }
.comment { margin: 0 0 10px; } .reply { margin-inline-start: 28px; }
.message { margin: 10px 0; padding: 8px 12px; border-radius: 12px; background: #f3e7d2; } .mine { background: #e3eef3; }
.gone { color: #8a8f8c; font-style: italic; margin: 0; }
summary { cursor: pointer; font-weight: 600; }
</style></head><body><main>
<h1>${esc(data.profile.name)}</h1>
<p class="lead">כל מה שהיה לך ברשת החופשית, כפי שנשמר ב-${when(data.exportedAt)}. הדף הזה נקרא בלי חיבור לרשת. אותו מידע, בצורה שתוכנות יכולות לקרוא, נמצא בקובץ data.json.</p>
<nav>${sections.map(([id, title]) => `<a href="#${id}">${title}</a>`).join('')}</nav>
${sections.map(([id, title, body]) => `<h2 id="${id}">${title}</h2>\n${body || '<p class="gone">אין</p>'}`).join('\n')}
</main></body></html>`;
}

const README = `הרשת החופשית: כל המידע שלך

index.html   לפתוח בדפדפן. הכול קריא בלי חיבור לרשת.
data.json    אותו מידע בפורמט פתוח (JSON), למי שרוצה להעביר אותו למקום אחר.
images/      התמונות שלך, ושל מה שנכתב אצלך.

מה יש כאן: הפרופיל וההעדפות שלך, הפוסטים שלך עם התגובות והרגשות שקיבלו, תגובות שכתבת
אצל אחרים, הרגשות שהבעת, ההודעות הפרטיות שלך (שני הצדדים), ורשימת החברים.
יש כאן גם מילים של אחרים שנכתבו אליך או אצלך. הן נשמרות לשימושך האישי.

The Free Network: everything you had there.
index.html reads offline. data.json holds the same in an open format ("${FORMAT}", version 1):
profile, friends, posts (each with its comments and reactions), comments (yours, on posts of
others), reactions (yours), conversations (private messages, both sides). Times are milliseconds
since 1970 (UTC). A picture id names the file images/<id>.jpg.
`;

// Every picture that the data names.
function pictureIds(data) {
  const ids = new Set([data.profile.avatar, data.profile.cover]);
  for (const post of data.posts) {
    post.images.forEach((id) => ids.add(id));
    post.shared?.post?.images.forEach((id) => ids.add(id));
    post.comments.forEach((item) => ids.add(item.image));
  }
  data.comments.forEach((item) => ids.add(item.image));
  data.conversations.forEach((talk) => talk.messages.forEach((message) => ids.add(message.image)));
  ids.delete(null);
  ids.delete(undefined);
  return [...ids];
}

// Builds the file. `progress` hears a line about what is being done now.
export async function exportAll({ api, progress = () => {} }) {
  progress('אוספים את המידע שלך…');
  const data = await api('/me/export');
  const ids = pictureIds(data);
  const files = [];
  const have = new Set();
  let next = 0;
  let done = 0;
  await Promise.all(Array.from({ length: FETCH_TOGETHER }, async () => {
    while (next < ids.length) {
      const id = ids[next++];
      const response = await fetch(`/api/images/${id}`).catch(() => null);
      if (response?.ok) {
        files.push({ name: `images/${id}.jpg`, data: new Uint8Array(await response.arrayBuffer()) });
        have.add(id);
      }
      progress(`מורידים תמונות: ${++done} מתוך ${ids.length}`);
    }
  }));
  // A picture that could not be read (its post is no longer open to the member) is named, not hidden.
  data.missingImages = ids.filter((id) => !have.has(id));
  progress('אורזים…');
  const day = new Date(data.exportedAt).toISOString().slice(0, 10);
  return {
    name: `free-network-${day}.zip`,
    blob: zip([{ name: 'README.txt', data: README }, { name: 'index.html', data: readerPage(data, have) },
      { name: 'data.json', data: JSON.stringify(data, null, 2) }, ...files]),
    counts: { posts: data.posts.length, pictures: have.size, missing: data.missingImages.length },
  };
}
