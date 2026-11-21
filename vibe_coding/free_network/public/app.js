import { friendsGlobe } from '/globe.js';
import { updatePanel } from '/update.js';

const app = document.getElementById('app');
// Beside the feed: the update on the left once there is room for it, and the globe on the
// right once there is room for both.
const SIDE_UPDATE = matchMedia('(min-width: 1000px)');
const SIDE_GLOBE = matchMedia('(min-width: 1240px)');
const nav = document.getElementById('nav');
const MAX_IMAGE_SIDE = 1600;
const MAX_IMAGES = 4;
const MAX_REPORTS = 20;
let reports = 0;
const ICONS = {
  user: 'M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M12 3a4 4 0 1 0 0 8a4 4 0 1 0 0-8',
  info: 'M12 22a10 10 0 1 0 0-20a10 10 0 0 0 0 20M12 16v-4M12 8h.01',
  logout: 'M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9',
  comment: 'M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z',
  link: 'M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71',
  trash: 'M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6M10 11v6M14 11v6',
  image: 'M5 3h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zM8.5 10a1.5 1.5 0 1 0 0-3a1.5 1.5 0 0 0 0 3M21 15l-5-5L5 21',
  home: 'M3 11l9-8 9 8M5 10v10h14V10',
  settings: 'M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6',
  camera: 'M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2zM12 17a4 4 0 1 0 0-8a4 4 0 0 0 0 8',
  leaf: 'M11 20A7 7 0 0 1 4 13C4 7 10 4 20 4c0 10-3 16-9 16zM4 21c2-6 6-9 11-11',
  users: 'M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2M9 3a4 4 0 1 0 0 8a4 4 0 1 0 0-8M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75',
  globe: 'M12 2a10 10 0 1 0 0 20a10 10 0 0 0 0-20M2 12h20M12 2a15.3 15.3 0 0 1 4 10a15.3 15.3 0 0 1-4 10a15.3 15.3 0 0 1-4-10a15.3 15.3 0 0 1 4-10',
  search: 'M11 3a8 8 0 1 0 0 16a8 8 0 0 0 0-16M21 21l-4.35-4.35',
  close: 'M18 6L6 18M6 6l12 12',
  share: 'M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8M16 6l-4-4-4 4M12 2v13',
  right: 'M9 18l6-6-6-6',
  left: 'M15 18l-6-6 6-6',
  reply: 'M9 14L4 9l5-5M20 20v-7a4 4 0 0 0-4-4H4',
  eye: 'M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8zM12 9a3 3 0 1 0 0 6a3 3 0 0 0 0-6',
  check: 'M20 6L9 17l-5-5',
  plus: 'M12 5v14M5 12h14',
  minus: 'M5 12h14',
  flag: 'M4 22V4M4 4h13l-2 4 2 4H4',
  sun: 'M12 17a5 5 0 1 0 0-10a5 5 0 0 0 0 10M12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4',
};
let me = null;
let demo = false;
let inviteOnly = false;
let googleLogin = false;
const GOOGLE_MESSAGES = {
  linked: 'חשבון הגוגל חובר. מעכשיו אפשר להיכנס איתו מכל מכשיר.',
  login: null,
  joined: null,
  unknown: 'חשבון הגוגל הזה לא מוכר כאן. נכנסים עם חשבון Google של הכתובת שאליה נשלחה ההזמנה, או עם הקישור האישי שבמייל.',
  taken: 'חשבון הגוגל הזה כבר מחובר למשתמש אחר.',
  failed: 'הכניסה עם גוגל לא הצליחה, נסו שוב.',
};
const scrollByPath = new Map();
history.scrollRestoration = 'manual';
let movedInside = false;
let shownPath = null;
let seenUpTo;
// Someone who has just joined sees the welcome page once, until they press on into the network.
const WELCOME_KEY = 'welcome';

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (key in node) node[key] = value;
    else node.setAttribute(key, value);
  }
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

// Sends browser-side failures to the server log, so they can be read without asking the user.
// The address hash is left out because a login link keeps its token there.
function report(message, stack) {
  if (reports >= MAX_REPORTS) return;
  reports++;
  fetch('/api/client-log', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: String(message), stack, path: location.pathname }),
    keepalive: true,
  }).catch(() => {});
}

window.addEventListener('error', (e) => report(e.message, e.error?.stack));
window.addEventListener('unhandledrejection', (e) => report(e.reason?.message ?? e.reason, e.reason?.stack));

async function api(path, { method = 'GET', body } = {}) {
  const isFile = body instanceof Blob;
  let response;
  try {
    response = await fetch(`/api${path}`, {
      method,
      headers: body ? { 'Content-Type': isFile ? body.type : 'application/json' } : {},
      body: isFile ? body : body ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    report(`${method} ${path}: ${err.message}`);
    throw new Error('אין חיבור לשרת, נסו שוב');
  }
  const data = await response.json().catch(() => ({}));
  if (response.ok) return data;
  if (response.status < 500) throw new Error(data.error ?? 'הבקשה נדחתה');

  const code = data.requestId ?? response.headers.get('X-Request-Id');
  if (!data.requestId) report(`${method} ${path}: ${response.status} without a server log`);
  throw new Error(`${data.error ?? 'משהו השתבש, נסו שוב'}${code ? ` (קוד ${code})` : ''}`);
}

async function go(path) {
  movedInside = true;
  scrollByPath.set(location.pathname, scrollY);
  history.pushState(null, '', path);
  await route();
  scrollTo(0, 0);
}

function link(path, ...content) {
  return el('a', { href: path, onclick: (e) => { e.preventDefault(); go(path); } }, ...content);
}

// On a phone this opens the share sheet, so a link goes straight to WhatsApp.
async function shareLink(url, button) {
  if (navigator.share) {
    try {
      await navigator.share({ url });
    } catch {
      // The reader closed the share sheet.
    }
    return;
  }
  await navigator.clipboard.writeText(url);
  const label = button.querySelector('.label') ?? button;
  const before = label.textContent;
  label.textContent = 'הקישור הועתק';
  setTimeout(() => { label.textContent = before; }, 2000);
}

// A question or a message in the page's own style, in place of the browser's grey box.
function dialogBox(message, { confirmLabel = 'אישור', cancelLabel = 'ביטול', danger = false } = {}) {
  return new Promise((resolve) => {
    const box = el('dialog', { className: 'dialog' });
    const close = (answer) => {
      box.close();
      box.remove();
      resolve(answer);
    };
    box.append(el('div', { className: 'dialog-body' },
      el('p', { dir: 'auto' }, message),
      el('div', { className: 'dialog-actions' },
        el('button', { className: danger ? 'danger-fill' : '', type: 'button', onclick: () => close(true) }, confirmLabel),
        cancelLabel ? el('button', { className: 'ghost', type: 'button', onclick: () => close(false) }, cancelLabel) : null)));
    box.addEventListener('cancel', (e) => {
      e.preventDefault();
      close(false);
    });
    box.addEventListener('click', (e) => {
      if (e.target === box) close(false);
    });
    document.body.append(box);
    box.showModal();
  });
}

const ask = (message, confirmLabel, danger = true) => dialogBox(message, { confirmLabel, danger });
const tell = (message) => dialogBox(message, { confirmLabel: 'הבנתי', cancelLabel: null });

// Pictures open over the page, so reading a post never throws the reader into another tab.
function lightbox(ids, start) {
  let at = start;
  const picture = el('img', { alt: 'תמונה מהפוסט' });
  const counter = el('span', { className: 'lightbox-count' });
  const box = el('dialog', { className: 'lightbox' });
  const show = (step) => {
    at = (at + step + ids.length) % ids.length;
    picture.src = `/api/images/${ids[at]}`;
    counter.textContent = ids.length > 1 ? `${at + 1} מתוך ${ids.length}` : '';
  };
  const arrow = (step, side, label) => el('button', {
    className: `lightbox-arrow ${side}`,
    type: 'button',
    ariaLabel: label,
    onclick: (e) => {
      e.stopPropagation();
      show(step);
    },
  }, icon(side));
  box.append(picture, el('button', { className: 'lightbox-close', type: 'button', ariaLabel: 'סגירה' }, icon('close')), counter);
  // The page reads right to left, so the next picture waits on the left.
  if (ids.length > 1) box.append(arrow(-1, 'right', 'התמונה הקודמת'), arrow(1, 'left', 'התמונה הבאה'));
  let touchX = null;
  box.addEventListener('touchstart', (e) => { touchX = e.touches[0].clientX; }, { passive: true });
  box.addEventListener('touchend', (e) => {
    const moved = e.changedTouches[0].clientX - touchX;
    if (ids.length > 1 && Math.abs(moved) > 40) {
      e.preventDefault();
      show(moved > 0 ? 1 : -1);
    }
  });
  box.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') show(1);
    if (e.key === 'ArrowRight') show(-1);
  });
  box.addEventListener('click', () => box.close());
  box.addEventListener('close', () => box.remove());
  show(0);
  document.body.append(box);
  box.showModal();
}

function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('class', 'icon');
  svg.setAttribute('aria-hidden', 'true');
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', ICONS[name]);
  svg.append(path);
  return svg;
}

function labelled(name, text) {
  return el('span', { className: 'with-icon' }, icon(name), el('span', { className: 'label' }, text));
}

// A link is shown the way a person would read it: no protocol, Hebrew letters instead of %D7 codes.
function linkText(url) {
  let text = url.replace(/^https?:\/\/(www\.)?/, '').replace(/\/$/, '');
  try {
    text = decodeURI(text);
  } catch {
    // Not valid encoding, so it stays as written.
  }
  return text.length > 64 ? `${text.slice(0, 61)}…` : text;
}

// Text is added as text nodes only, so user content can never become HTML.
// Links are clickable for everyone, in posts and in comments alike.
function richText(text) {
  const node = el('p', { className: 'text', dir: 'auto' });
  let last = 0;
  for (const match of text.matchAll(/https?:\/\/[^\s<>"]+/g)) {
    const url = match[0].replace(/[.,;:!?'"”׳״)\]]+$/, '');
    node.append(text.slice(last, match.index));
    node.append(el('a', { className: 'url', href: url, title: url, dir: 'ltr', target: '_blank', rel: 'noopener nofollow ugc' }, linkText(url)));
    last = match.index + url.length;
  }
  node.append(text.slice(last));
  return node;
}

// The picture and title come through our server. Nothing reaches Google until the reader clicks.
function video(id) {
  const title = el('span', { className: 'video-title', dir: 'auto' }, 'סרטון מיוטיוב, לחצו לטעינה');
  const picture = el('img', {
    className: 'video-picture',
    src: `/api/youtube/${id}/thumb`,
    alt: '',
    loading: 'lazy',
    onerror: () => picture.remove(),
  });
  const button = el('button', { className: 'video', type: 'button' }, picture, el('span', { className: 'play' }), title);
  api(`/youtube/${id}`).then((details) => {
    if (details.title) title.textContent = details.title;
  }).catch(() => {});
  button.addEventListener('click', () => {
    button.replaceWith(el('div', { className: 'video' }, el('iframe', {
      src: `https://www.youtube-nocookie.com/embed/${id}?autoplay=1`,
      title: 'סרטון מיוטיוב',
      allow: 'autoplay; encrypted-media; picture-in-picture; fullscreen',
      allowFullscreen: true,
      referrerPolicy: 'strict-origin-when-cross-origin',
    })));
  });
  return button;
}

// Shrinks the photo and re-encodes it, which also drops the GPS location the phone stored in it.
async function prepareImage(file) {
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = el('canvas', { width: Math.round(bitmap.width * scale), height: Math.round(bitmap.height * scale) });
  const context = canvas.getContext('2d');
  context.fillStyle = '#fff';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.8));
}

function gallery(ids) {
  return el('div', { className: `gallery n${ids.length}` }, ...ids.map((id, index) => el('a', {
    href: `/api/images/${id}`,
    target: '_blank',
    onclick: (e) => {
      if (e.ctrlKey || e.metaKey || e.shiftKey) return;
      e.preventDefault();
      lightbox(ids, index);
    },
  }, el('img', { src: `/api/images/${id}`, alt: 'תמונה מהפוסט', loading: 'lazy' }))));
}

function svg(tag, attributes = {}, ...children) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  node.append(...children);
  return node;
}

// Until a member picks a picture, theirs is drawn from their first name: the whole name when it
// is short, and broken into two rows when it is long. Tiny pictures keep only the first letter.
function nameArt(name) {
  const letters = [...(name.trim().split(/\s+/)[0] ?? '')].slice(0, 10);
  if (!letters.length) letters.push('?');
  const rows = letters.length <= 4 ? [letters] : [letters.slice(0, Math.ceil(letters.length / 2)), letters.slice(Math.ceil(letters.length / 2))];
  const widest = Math.max(...rows.map((row) => row.length));
  const size = Math.min(rows.length === 1 ? 52 : 34, 80 / (0.62 * widest));
  const line = (row, index) => svg('text', {
    x: 50,
    y: 50 + (index - (rows.length - 1) / 2) * size * 1.08,
    'font-size': size.toFixed(1),
    'text-anchor': 'middle',
    'dominant-baseline': 'central',
  }, row.join(''));
  return svg('svg', { viewBox: '0 0 100 100', class: 'name-art' },
    svg('g', { class: 'name-full' }, ...rows.map(line)),
    svg('text', { class: 'name-initial', x: 50, y: 50, 'font-size': 52, 'text-anchor': 'middle', 'dominant-baseline': 'central' }, letters[0]));
}

function avatar(user, { small = false, large = false } = {}) {
  const size = small ? ' small' : large ? ' large' : '';
  const content = user.avatar ? el('img', { src: `/api/images/${user.avatar}`, alt: '' }) : nameArt(user.name ?? '');
  return el('span', { className: `avatar c${(user.id - 1) % 6}${size}`, ariaHidden: 'true' }, content);
}

function googleButton(href, label) {
  return el('a', { className: 'google-button', href }, el('img', { src: '/google.svg', alt: '' }), label);
}

function profileLink(user, ...content) {
  return link(`/u/${user.id}`, ...content);
}

const relative = new Intl.RelativeTimeFormat('he', { numeric: 'auto' });
const format = (options) => new Intl.DateTimeFormat('he-IL', options);

function daysAgo(ms) {
  const midnight = (date) => new Date(date).setHours(0, 0, 0, 0);
  return Math.round((midnight(Date.now()) - midnight(ms)) / 86400000);
}

// Recent times read the way people say them. The exact date and hour are in the tooltip.
function when(ms) {
  const minutes = Math.floor((Date.now() - ms) / 60000);
  const days = daysAgo(ms);
  const hour = format({ timeStyle: 'short' }).format(ms);
  let text;
  if (minutes < 1) text = 'עכשיו';
  else if (minutes < 60) text = relative.format(-minutes, 'minute');
  else if (days === 0) text = relative.format(-Math.floor(minutes / 60), 'hour');
  else if (days === 1) text = `אתמול ב-${hour}`;
  else if (days < 7) text = `${format({ weekday: 'long' }).format(ms)} ב-${hour}`;
  else {
    const sameYear = new Date(ms).getFullYear() === new Date().getFullYear();
    text = `${format({ day: 'numeric', month: 'long', ...(sameYear ? {} : { year: 'numeric' }) }).format(ms)}, ${hour}`;
  }
  return el('time', { dateTime: new Date(ms).toISOString(), title: format({ dateStyle: 'full', timeStyle: 'short' }).format(ms) }, text);
}

function dayLabel(ms) {
  const days = daysAgo(ms);
  if (days === 0) return 'היום';
  if (days === 1) return 'אתמול';
  return format({ weekday: 'long', day: 'numeric', month: 'long' }).format(ms);
}

function audienceChip(audience) {
  const open = audience === 'public';
  return el('span', {
    className: `chip ${open ? 'public' : 'friends'}`,
    title: open ? 'כל מי שיש לו את הקישור יכול לקרוא' : 'רק החברים של הכותב רואים את הפוסט',
  }, icon(open ? 'globe' : 'users'), open ? 'ציבורי' : 'חברים');
}

function postCard(post, { onDeleted } = {}) {
  const url = `/p/${post.id}`;
  const count = !post.commentCount ? 'להגיב' : post.commentCount === 1 ? 'תגובה אחת' : `${post.commentCount} תגובות`;
  const actions = el('div', { className: 'actions' }, link(url, labelled('comment', count)));
  if (post.audience === 'public') {
    const copy = el('button', { className: 'plain', type: 'button' }, labelled(navigator.share ? 'share' : 'link', navigator.share ? 'שיתוף' : 'העתקת קישור'));
    copy.addEventListener('click', () => shareLink(new URL(url, location.origin).href, copy));
    actions.append(copy);
  }
  const remove = !post.mine ? null : el('button', {
    className: 'icon-button danger',
    type: 'button',
    ariaLabel: 'מחיקת הפוסט',
    title: 'מחיקת הפוסט',
    onclick: async () => {
      if (!await ask('למחוק את הפוסט?', 'מחיקה')) return;
      try {
        await api(`/posts/${post.id}`, { method: 'DELETE' });
        onDeleted?.();
      } catch (err) {
        tell(err.message);
      }
    },
  }, icon('trash'));
  return el('article', { className: 'card post' },
    el('div', { className: 'head' }, profileLink(post.author, avatar(post.author)), el('div', { className: 'who' },
      profileLink(post.author, el('span', { className: 'author', dir: 'auto' }, post.author.name)),
      el('span', { className: 'meta' }, link(url, when(post.createdAt)), audienceChip(post.audience))), remove),
    post.body ? richText(post.body) : null,
    post.images.length ? gallery(post.images) : null,
    post.youtubeId ? video(post.youtubeId) : null,
    actions);
}

async function composer(onPosted) {
  const { friends } = await api('/audience');
  const text = el('textarea', {
    placeholder: 'מה חדש?',
    maxLength: 5000,
    dir: 'auto',
    onfocus: () => text.closest('form').classList.add('open'),
    onkeydown: (e) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) text.closest('form').requestSubmit();
    },
  });
  const note = el('span', {});
  const error = el('p', { className: 'error', hidden: true });
  const radio = (value, label, picture, checked) => el('label', {},
    el('input', { type: 'radio', name: 'audience', value, checked, onchange: explain }), icon(picture), label);
  const choices = el('div', { className: 'audience', role: 'radiogroup', ariaLabel: 'מי יראה את הפוסט' },
    radio('friends', 'חברים', 'users', true), radio('public', 'ציבורי', 'globe', false));

  function audience() {
    return choices.querySelector('input:checked').value;
  }

  function explain() {
    const who = friends === 1 ? 'החבר היחיד שלך' : `${friends} החברים שלך`;
    const reach = friends ? `יופיע בפיד של ${who}, בלי סינון ובלי דירוג.` : 'עוד אין לך חברים, אז הפוסט יופיע רק אצלך.';
    note.textContent = audience() === 'public' ? `${reach} כל מי שיש לו את הקישור יוכל לקרוא אותו.` : reach;
  }
  explain();

  let photos = [];
  const previews = el('div', { className: 'previews' });
  const showPhotos = () => previews.replaceChildren(...photos.map((photo) => el('button', {
    className: 'preview',
    type: 'button',
    title: 'הסרת התמונה',
    onclick: () => {
      photos = photos.filter((other) => other !== photo);
      showPhotos();
    },
  }, el('img', { src: URL.createObjectURL(photo), alt: 'תמונה שנבחרה' }))));
  const picker = el('input', { type: 'file', accept: 'image/*', multiple: true, hidden: true });
  picker.addEventListener('change', async () => {
    error.hidden = true;
    try {
      for (const file of [...picker.files].slice(0, MAX_IMAGES - photos.length)) photos.push(await prepareImage(file));
      showPhotos();
    } catch {
      error.textContent = 'אי אפשר לקרוא את התמונה הזו';
      error.hidden = false;
    }
    picker.value = '';
  });
  const addPhoto = el('button', { className: 'plain', type: 'button', ariaLabel: 'הוספת תמונה', title: 'הוספת תמונה', onclick: () => picker.click() }, labelled('image', 'תמונה'));

  const submit = el('button', { type: 'submit', title: 'אפשר גם Ctrl+Enter' }, 'פרסום');
  return el('form', {
    className: 'card composer',
    onsubmit: async (e) => {
      e.preventDefault();
      submit.disabled = true;
      submit.textContent = 'מפרסם…';
      error.hidden = true;
      try {
        const uploads = photos.map((photo) => api('/images', { method: 'POST', body: photo }));
        const images = (await Promise.all(uploads)).map((upload) => upload.id);
        await api('/posts', { method: 'POST', body: { body: text.value, audience: audience(), images } });
        onPosted();
      } catch (err) {
        error.textContent = err.message;
        error.hidden = false;
        submit.disabled = false;
        submit.textContent = 'פרסום';
      }
    },
  }, el('div', { className: 'composer-top' }, avatar(me), text),
  el('div', { className: 'composer-more' }, previews, picker,
    el('div', { className: 'composer-bar' }, choices, addPhoto, submit),
    el('p', { className: 'reach' }, icon('eye'), note), error));
}

// Remembers, on this device only, the newest post already shown here.
function seenStore(value) {
  try {
    const key = `seen:${me.id}`;
    if (value) localStorage.setItem(key, value);
    return Number(localStorage.getItem(key)) || 0;
  } catch {
    return 0;
  }
}

async function globe() {
  const { friends } = await api('/globe');
  return friendsGlobe({ me, friends, ui: { el, icon, avatar, go, api, tell, demo } });
}

async function globePage() {
  app.replaceChildren(el('div', { className: 'card globe-page' },
    el('h1', {}, 'החברים שלי בעולם'),
    await globe()));
}

// Turns the globe to a place. Without a globe on the screen, its page opens first.
async function flyGlobe(spot) {
  const waits = !document.querySelector('.globe');
  if (waits) await go('/globe');
  setTimeout(() => document.querySelector('.globe')?.dispatchEvent(new CustomEvent('fly', { detail: spot })), waits ? 400 : 0);
}

function update({ all = false, top = false } = {}) {
  seenUpTo ??= seenStore();
  return updatePanel({ me, seen: seenUpTo, all, top, ui: { el, icon, avatar, link, api, when, tell, demo, flyGlobe } });
}

function updatePage() {
  app.replaceChildren(el('div', { className: 'card update-page' }, update({ all: true })));
}

async function feedPage() {
  const list = el('div', { className: 'feed' });
  const more = el('button', { className: 'ghost more', type: 'button', hidden: true }, 'פוסטים ישנים יותר');
  const end = el('div', { className: 'feed-end', hidden: true },
    el('img', { src: '/favicon.svg', alt: '' }),
    el('p', {}, 'הגעת לסוף. זה כל מה שנכתב, אין כאן גלילה אינסופית.'));
  seenUpTo ??= seenStore();
  let oldest = null;
  let lastDay = null;
  let fresh = 0;
  let marked = !seenUpTo;

  async function load() {
    const { posts, hasMore } = await api(`/feed${oldest ? `?before=${oldest}` : ''}`);
    if (!oldest && posts.length) seenStore(posts[0].id);
    for (const post of posts) {
      if (post.id > seenUpTo && !post.mine) fresh++;
      if (!marked && post.id <= seenUpTo) {
        marked = true;
        if (fresh) list.append(el('p', { className: 'seen-mark' }, el('span', {}, 'מכאן ומטה כבר ראית')));
      }
      const day = dayLabel(post.createdAt);
      if (day !== lastDay) list.append(el('h2', { className: 'day' }, el('span', {}, day)));
      lastDay = day;
      list.append(postCard(post, { onDeleted: route }));
    }
    if (posts.length) oldest = posts.at(-1).id;
    more.hidden = !hasMore;
    end.hidden = hasMore || !list.children.length;
    if (!list.children.length) {
      list.append(el('div', { className: 'card empty' },
        el('img', { src: '/favicon.svg', alt: '' }),
        el('h2', {}, 'שקט כאן בינתיים'),
        el('p', { className: 'muted' }, 'כתבו את הפוסט הראשון, או ', link('/friends', 'מצאו חברים'), ' מהשכונה.')));
    }
  }
  more.addEventListener('click', async () => {
    more.disabled = true;
    try {
      await load();
    } catch (err) {
      tell(err.message);
    }
    more.disabled = false;
  });
  await load();

  // Beside the feed: the update on the left, folded into a line per circle, and the globe on
  // the right. On a narrow screen the folded update sits above the posts instead.
  const side = SIDE_GLOBE.matches ? el('aside', { className: 'globe-side' }, await globe()) : null;
  const news = SIDE_UPDATE.matches
    ? el('aside', { className: 'card update-side' }, update())
    : el('aside', { className: 'card update-top' }, update({ top: true }));
  app.replaceChildren(...[
    side,
    news,
    await composer(route),
    el('p', { className: 'feed-rule' }, 'כל הפוסטים שלך ושל החברים שלך, מהחדש לישן. זה כל האלגוריתם.'),
    list,
    more,
    end].filter(Boolean));
}

function comment(item, reload, onReply) {
  const bubble = el('div', { className: 'bubble' },
    el('div', { className: 'comment-head' },
      profileLink(item.author, el('span', { className: 'author', dir: 'auto' }, item.author.name)),
      el('span', { className: 'meta' }, when(item.createdAt))));
  const node = el('div', { className: `comment${item.parentId ? ' reply' : ''}` }, profileLink(item.author, avatar(item.author, { small: true })), bubble);
  if (item.removedBy) {
    const by = item.removedBy === 'author' ? 'הכותב' : 'בעל הפוסט';
    bubble.append(el('p', { className: 'text removed' }, `התגובה הוסרה על ידי ${by}`));
    return node;
  }
  if (item.body) bubble.append(richText(item.body));
  if (item.image) bubble.append(gallery([item.image]));
  if (item.youtubeId) bubble.append(video(item.youtubeId));
  const actions = el('div', { className: 'comment-actions' });
  if (onReply) actions.append(el('button', { className: 'plain', type: 'button', onclick: onReply }, labelled('reply', 'הגבה')));
  if (item.canRemove) {
    actions.append(el('button', {
      className: 'plain danger',
      type: 'button',
      onclick: async () => {
        if (!await ask('להסיר את התגובה? במקומה יישאר סימון שהוסרה.', 'הסרה')) return;
        try {
          await api(`/comments/${item.id}`, { method: 'DELETE' });
          reload();
        } catch (err) {
          tell(err.message);
        }
      },
    }, 'הסרה'));
  }
  if (actions.children.length) bubble.append(actions);
  return node;
}

function commentForm(postId, parentId, placeholder) {
  const text = el('textarea', {
    placeholder,
    maxLength: 2000,
    dir: 'auto',
    onkeydown: (e) => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) text.form.requestSubmit();
    },
  });
  const error = el('p', { className: 'error', hidden: true });
  const submit = el('button', { type: 'submit', title: 'אפשר גם Ctrl+Enter' }, 'שליחה');
  const previews = el('div', { className: 'previews' });
  const picker = el('input', { type: 'file', accept: 'image/*', hidden: true });
  let photo = null;
  const showPhoto = () => previews.replaceChildren(...(photo ? [el('button', {
    className: 'preview',
    type: 'button',
    title: 'הסרת התמונה',
    onclick: () => {
      photo = null;
      showPhoto();
    },
  }, el('img', { src: URL.createObjectURL(photo), alt: 'תמונה שנבחרה' }))] : []));
  picker.addEventListener('change', async () => {
    error.hidden = true;
    try {
      photo = await prepareImage(picker.files[0]);
      showPhoto();
    } catch {
      error.textContent = 'אי אפשר לקרוא את התמונה הזו';
      error.hidden = false;
    }
    picker.value = '';
  });
  return el('form', {
    className: 'comment-form',
    onsubmit: async (e) => {
      e.preventDefault();
      submit.disabled = true;
      error.hidden = true;
      try {
        const image = photo ? (await api('/images', { method: 'POST', body: photo })).id : null;
        await api(`/posts/${postId}/comments`, { method: 'POST', body: { body: text.value, image, parentId } });
        route();
      } catch (err) {
        error.textContent = err.message;
        error.hidden = false;
        submit.disabled = false;
      }
    },
  }, text, previews, picker, el('div', { className: 'row spread' },
    el('button', { className: 'plain', type: 'button', onclick: () => picker.click() }, labelled('image', 'תמונה')),
    submit), error);
}

function commentThread(postId, top, replies) {
  const group = el('div', { className: 'comment-group' });
  const openReply = () => {
    let form = group.querySelector('.reply-form');
    if (!form) {
      form = commentForm(postId, top.id, 'כתיבת תגובה לתגובה');
      form.classList.add('reply-form');
      under.append(form);
    }
    form.querySelector('textarea').focus();
  };
  const onReply = me?.name ? openReply : null;
  const under = el('div', { className: 'replies' }, ...replies.map((item) => comment(item, route, top.removedBy ? null : onReply)));
  group.append(comment(top, route, top.removedBy ? null : onReply), under);
  return group;
}

async function postPage(id) {
  const { post, comments } = await api(`/posts/${id}`);
  const back = !me ? null : el('a', {
    className: 'back',
    href: '/',
    onclick: (e) => {
      e.preventDefault();
      if (movedInside) history.back();
      else go('/');
    },
  }, icon('right'), 'חזרה');
  const section = el('section', { className: 'card thread' },
    el('h2', {}, comments.length ? `תגובות (${comments.length})` : 'תגובות', icon('leaf')),
    el('p', { className: 'muted' }, comments.length ? 'כל התגובות, לפי סדר הכתיבה.' : 'עוד אין תגובות.'),
    ...comments.filter((item) => !item.parentId).map((top) => commentThread(id, top,
      comments.filter((item) => item.parentId === top.id))));

  if (me?.name) section.append(commentForm(id, null, 'כתיבת תגובה'));
  else section.append(el('p', {}, link('/login', 'התחברו'), ' כדי להגיב.'));
  app.replaceChildren(...[back, postCard(post, { onDeleted: () => go('/') }), section].filter(Boolean));
}

function person(user, ...actions) {
  return el('div', { className: 'person' },
    el('div', { className: 'head' }, profileLink(user, avatar(user, { small: true }), el('span', { dir: 'auto' }, user.name))),
    el('div', { className: 'row', style: 'margin-top: 0' }, ...actions));
}

function friendAction(label, method, id, { plain = false, confirmText = null } = {}) {
  return el('button', {
    className: plain ? 'plain' : 'small',
    type: 'button',
    onclick: async () => {
      if (confirmText && !await ask(confirmText, label)) return;
      try {
        await api(`/friends/${id}`, { method });
        await refreshMe();
        route();
      } catch (err) {
        tell(err.message);
      }
    },
  }, label);
}

async function friendsPage() {
  const { friends, incoming, outgoing } = await api('/friends');
  const known = new Set([...friends, ...incoming, ...outgoing].map((user) => user.id));
  const results = el('div');
  const search = el('input', { type: 'search', placeholder: 'חיפוש לפי שם', dir: 'auto', ariaLabel: 'חיפוש אנשים לפי שם' });
  let timer = null;
  let asked = 0;
  search.addEventListener('input', () => {
    clearTimeout(timer);
    const q = search.value.trim();
    if (q.length < 2) {
      asked++;
      results.replaceChildren();
      return;
    }
    // Waits for a pause in the typing, and drops an answer that a newer search has overtaken.
    timer = setTimeout(async () => {
      const mine = ++asked;
      const { users } = await api(`/users?q=${encodeURIComponent(q)}`);
      if (mine !== asked) return;
      const found = users.filter((user) => !known.has(user.id));
      const already = users.length - found.length;
      results.replaceChildren(...found.map((user) => person(user, friendAction('בקשת חברות', 'POST', user.id))));
      if (!found.length) {
        results.append(el('p', { className: 'note' }, already ? 'מי שנמצא כבר מופיע ברשימות שלמטה.' : 'לא נמצא אף אחד בשם הזה.'));
      }
    }, 250);
  });

  const card = el('section', { className: 'card people' },
    el('h1', {}, 'חברים'),
    el('label', { className: 'search' }, icon('search'), search),
    results);
  if (incoming.length) {
    card.append(el('h2', {}, 'בקשות אליך'), ...incoming.map((user) => person(user,
      friendAction('אישור', 'POST', user.id), friendAction('דחייה', 'DELETE', user.id, { plain: true }))));
  }
  if (outgoing.length) {
    card.append(el('h2', {}, 'ממתינות לאישור'), ...outgoing.map((user) => person(user,
      friendAction('ביטול', 'DELETE', user.id, { plain: true }))));
  }
  card.append(el('h2', {}, `החברים שלך (${friends.length})`), ...friends.map((user) => person(user,
    friendAction('הסרה', 'DELETE', user.id, { plain: true, confirmText: `להסיר את ${user.name} מהחברים שלך?` }))));
  if (!friends.length) card.append(el('p', { className: 'note' }, 'עוד אין לך חברים כאן. חפשו מישהו לפי שם ושלחו בקשת חברות.'));
  app.replaceChildren(card);
}

async function savePicture(kind, imageId) {
  await api('/me/images', { method: 'PUT', body: { kind, imageId } });
  ({ user: me } = await api('/me'));
  await route();
}

// On your own page, a picture is its own button: pressing it replaces it.
function editablePicture(kind, label, content, error) {
  const picker = el('input', { type: 'file', accept: 'image/*', hidden: true });
  const button = el('button', {
    className: `picture-edit ${kind}-edit`,
    type: 'button',
    ariaLabel: label,
    title: label,
    onclick: () => picker.click(),
  }, content, el('span', { className: 'badge' }, icon('camera')));
  picker.addEventListener('change', async () => {
    error.hidden = true;
    button.classList.add('busy');
    try {
      const { id } = await api('/images', { method: 'POST', body: await prepareImage(picker.files[0]) });
      await savePicture(kind, id);
    } catch (err) {
      error.textContent = err.message;
      error.hidden = false;
      button.classList.remove('busy');
    }
  });
  return [button, picker];
}

async function profilePage(id) {
  const { user, friendship, posts, hasMore } = await api(`/users/${id}`);
  const list = el('div');
  const more = el('button', { className: 'plain', type: 'button', hidden: !hasMore }, 'פוסטים ישנים יותר');
  let oldest = posts.at(-1)?.id;
  const show = (items) => list.append(...items.map((post) => postCard(post, { onDeleted: route })));
  show(posts);
  more.addEventListener('click', async () => {
    const next = await api(`/users/${id}?before=${oldest}`);
    show(next.posts);
    oldest = next.posts.at(-1)?.id ?? oldest;
    more.hidden = !next.hasMore;
  });
  if (!posts.length) list.append(el('p', { className: 'feed-rule' }, 'אין כאן פוסטים שאפשר להציג.'));

  const mine = friendship === 'self';
  const error = el('p', { className: 'error', hidden: true });
  const face = avatar(user, { large: true });
  const cover = el('div', { className: `cover${user.cover ? '' : ' plain-cover'}` }, user.cover ? el('img', { src: `/api/images/${user.cover}`, alt: '' }) : null);
  const body = el('div', { className: 'profile-body' },
    ...(mine ? editablePicture('avatar', 'החלפת תמונת הפרופיל', face, error) : [face]),
    el('h1', { dir: 'auto' }, user.name));
  if (mine) {
    const rename = el('button', {
      className: 'plain',
      type: 'button',
      onclick: () => {
        const field = el('input', { type: 'text', required: true, maxLength: 60, value: me.name, dir: 'auto', ariaLabel: 'השם שיוצג לאחרים' });
        rename.replaceWith(el('form', {
          className: 'rename',
          onsubmit: async (e) => {
            e.preventDefault();
            try {
              await api('/me', { method: 'PUT', body: { name: field.value } });
              await refreshMe();
              route();
            } catch (err) {
              error.textContent = err.message;
              error.hidden = false;
            }
          },
        }, field, el('button', { className: 'small', type: 'submit' }, 'שמירה'), el('button', { className: 'plain', type: 'button', onclick: route }, 'ביטול')));
        field.focus();
      },
    }, 'שינוי שם');
    const remove = (kind, label) => el('button', {
      className: 'plain danger',
      type: 'button',
      onclick: async () => await ask('להסיר את התמונה?', 'הסרה') && savePicture(kind, null),
    }, label);
    const google = !googleLogin ? null : me.googleLinked
      ? el('div', { className: 'row' }, el('span', { className: 'muted' }, 'חשבון גוגל מחובר לכניסה.'), el('button', {
        className: 'plain',
        type: 'button',
        onclick: async () => {
          if (!await ask('לנתק את חשבון הגוגל? אחרי זה אפשר יהיה להיכנס רק עם קישור.', 'ניתוק')) return;
          await api('/me/google', { method: 'DELETE' });
          await refreshMe();
          route();
        },
      }, 'ניתוק'))
      : el('div', { className: 'row' }, googleButton('/api/auth/google?link=1', 'חיבור חשבון Google'));
    const lettered = user.avatar ? null : el('p', { className: 'note' }, 'התמונה שלך מצוירת בינתיים מהשם הפרטי. אפשר להעלות תמונה, או לחבר חשבון Google ולקבל את התמונה שלו.');
    body.append(...[
      error,
      google,
      lettered,
      el('p', { className: 'note' }, 'לחיצה על תמונה מחליפה אותה. התמונות האלה גלויות כמו השם שלך, גם למי שקורא פוסט ציבורי בלי להתחבר.'),
      el('div', { className: 'row' },
        rename,
        user.avatar ? remove('avatar', 'הסרת תמונת הפרופיל') : null,
        user.cover ? remove('cover', 'הסרת תמונת הכותרת') : null,
        el('button', { className: 'plain phone-only', type: 'button', onclick: logout }, 'יציאה מהחשבון'))].filter(Boolean));
  } else if (friendship === 'friends') {
    body.append(el('p', { className: 'chip friends' }, icon('check'), 'חברים'));
  } else if (friendship === 'incoming') {
    body.append(el('div', { className: 'row' }, friendAction('אישור בקשת החברות', 'POST', user.id)));
  } else if (friendship === 'outgoing') {
    body.append(el('p', { className: 'muted' }, 'בקשת החברות ממתינה לאישור.'));
  } else if (!friendship) {
    body.append(el('div', { className: 'row' }, friendAction('בקשת חברות', 'POST', user.id)));
  }

  app.replaceChildren(el('section', { className: 'card profile' },
    ...(mine ? editablePicture('cover', 'החלפת תמונת הכותרת', cover, error) : [cover]), body), list, more);
}

function meter(label, value, max, text) {
  return el('div', { className: 'meter' },
    el('div', { className: 'row spread' }, el('span', {}, label), el('span', { className: 'muted' }, `${text} · ${(value / max * 100).toFixed(1)}%`)),
    el('progress', { value, max }));
}

async function adminPage() {
  const { usage, users, posts, comments, members, emailInvites } = await api('/admin/overview');
  const gb = (bytes) => (bytes / 1024 ** 3).toFixed(2);
  const result = el('div');

  async function invite(userId) {
    const { link: url } = await api('/admin/invites', { method: 'POST', body: { userId } });
    const copy = el('button', { type: 'button' }, navigator.share ? 'שליחה' : 'העתקה');
    copy.addEventListener('click', () => shareLink(url, copy));
    result.replaceChildren(
      el('input', { type: 'text', readOnly: true, value: url, dir: 'ltr' }),
      el('p', { className: 'note' }, 'הקישור עובד פעם אחת, במשך שבוע. שלחו אותו רק לאדם שהוא מיועד לו.'),
      el('div', { className: 'row' }, copy));
  }

  // The invitation goes out from the admin's own mailbox, so it arrives as a letter from a
  // friend. Each line of the box is one person: a name, a comma, an address.
  const people = el('textarea', { className: 'invite-lines', placeholder: 'דנה כהן, dana@gmail.com', dir: 'auto', rows: 3, required: true, ariaLabel: 'מוזמנים: שם, פסיק, כתובת מייל. שורה לכל אחד' });
  const sent = el('div', { className: 'invite-results' });
  const pendingBox = el('div');
  const firstName = (name) => (name ?? '').trim().split(/\s+/)[0] ?? '';

  // The letter goes to the address as it was typed. Gmail ignores the dots, but people do not.
  function letter({ name, email, typed = email, link: url, expiresAt }) {
    const until = format({ day: 'numeric', month: 'long' }).format(expiresAt);
    const subject = 'הזמנה אישית לרשת החופשית';
    const body = [
      `היי${firstName(name) ? ` ${firstName(name)}` : ''},`,
      '',
      'אני בונה רשת חברתית חדשה. אין בה פרסומות, ואין בה אלגוריתם שמחליט בשבילנו מה נראה. היא עדיין קטנה ונמצאת בניסוי בין חברים, ואשמח לראות אותך בין הראשונים.',
      '',
      'סרטון קצר שמסביר מה זה ולמה:',
      `${location.origin}/intro`,
      '',
      `הקישור האישי שלך לכניסה (עובד פעם אחת, עד ${until}):`,
      url,
      '',
      'אשמח לשמוע מה חסר ומה מפריע. בשביל זה הניסוי.',
      firstName(me.name),
    ].join('\n');
    const query = (pairs) => new URLSearchParams(pairs).toString().replace(/\+/g, '%20');
    const copy = el('button', { className: 'plain', type: 'button' }, 'העתקת המכתב');
    copy.addEventListener('click', async () => {
      await navigator.clipboard.writeText(body);
      copy.textContent = 'הועתק';
    });
    return el('div', { className: 'invite-letter' },
      el('div', { className: 'row spread' },
        el('span', {}, el('strong', { dir: 'auto' }, name || typed), name ? el('span', { className: 'muted', dir: 'ltr' }, ` ${typed}`) : null),
        el('span', { className: 'row' },
          el('a', { className: 'send', target: '_blank', rel: 'noopener', href: `https://mail.google.com/mail/?${query({ view: 'cm', fs: '1', to: typed, su: subject, body })}` }, 'פתיחה ב-Gmail'),
          el('a', { className: 'other', href: `mailto:${encodeURIComponent(typed)}?${query({ subject, body })}` }, 'תוכנת מייל אחרת'),
          copy)),
      el('pre', { className: 'letter-text', dir: 'auto' }, body));
  }

  async function inviteByEmail(line) {
    const parts = line.split(/[,\t;<>]+/).map((part) => part.trim()).filter(Boolean);
    const email = parts.find((part) => part.includes('@'));
    if (!email) throw new Error(`בשורה "${line}" אין כתובת מייל`);
    const invite = await api('/admin/email-invites', { method: 'POST', body: { email, name: parts.filter((part) => part !== email).join(' ') } });
    return { ...invite, typed: email };
  }

  const emailForm = el('form', {
    onsubmit: async (e) => {
      e.preventDefault();
      sent.replaceChildren();
      const left = [];
      for (const line of people.value.split('\n').map((text) => text.trim()).filter(Boolean)) {
        try {
          sent.append(letter(await inviteByEmail(line)));
        } catch (err) {
          left.push(line);
          sent.append(el('p', { className: 'error' }, `${line}: ${err.message}`));
        }
      }
      people.value = left.join('\n');
      if (sent.querySelector('.invite-letter')) {
        sent.prepend(el('p', { className: 'note' }, 'ההזמנות מוכנות. לחיצה על "פתיחה ב-Gmail" פותחת מכתב מוכן, ונשאר רק לשלוח. את הקישור אי אפשר להציג שוב אחר כך, רק ליצור חדש.'));
      }
      showPending((await api('/admin/overview')).emailInvites);
    },
  }, people, el('div', { className: 'row' }, el('button', { type: 'submit' }, 'יצירת הזמנות')));

  function showPending(list) {
    pendingBox.replaceChildren(...(list.length ? [el('h2', {}, `הוזמנו ועוד לא נכנסו (${list.length})`)] : []), ...list.map((item) => el('div', { className: 'person' },
      el('span', {}, item.name ? el('span', { dir: 'auto' }, `${item.name} · `) : null, el('span', { dir: 'ltr' }, item.email)),
      el('div', { className: 'row' },
        el('button', {
          className: 'plain',
          type: 'button',
          title: 'יוצר קישור חדש ומבטל את הקודם',
          onclick: async () => {
            try {
              sent.replaceChildren(letter(await api('/admin/email-invites', { method: 'POST', body: { email: item.email, name: item.name ?? '' } })));
              sent.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
            } catch (err) {
              tell(err.message);
            }
          },
        }, 'מכתב חדש'),
        el('button', {
          className: 'plain danger',
          type: 'button',
          onclick: async () => {
            if (!await ask(`לבטל את ההזמנה של ${item.name || item.email}? הקישור שנשלח יפסיק לעבוד.`, 'ביטול ההזמנה')) return;
            await api(`/admin/email-invites/${encodeURIComponent(item.email)}`, { method: 'DELETE' });
            showPending(list.filter((other) => other !== item));
          },
        }, 'ביטול')))));
  }
  showPending(emailInvites);

  app.replaceChildren(
    el('section', { className: 'card' },
      el('h1', {}, 'ניהול'),
      el('p', { className: 'muted' }, `${users} משתמשים · ${posts} פוסטים · ${comments} תגובות`),
      meter('אחסון תמונות', usage.bytes, usage.storageLimit, `${gb(usage.bytes)} מתוך ${gb(usage.storageLimit)}GB`),
      meter('העלאות ב-24 השעות האחרונות', usage.today, usage.uploadLimit, `${usage.today} מתוך ${usage.uploadLimit}`)),
    el('section', { className: 'card' },
      el('h2', {}, 'הזמנה במייל'),
      el('p', { className: 'muted' }, 'שורה לכל מוזמן: שם, פסיק, כתובת מייל. לכל אחד נוצר קישור אישי ומכתב מוכן עם הקישור ועם סרטון ההסבר, ושולחים אותו מהמייל שלך. מי שהוזמן בכתובת של Google יכול גם להיכנס ישר עם Google, והתמונה שלו מגיעה משם.'),
      emailForm, sent, pendingBox),
    el('section', { className: 'card' },
      el('h2', {}, 'הזמנה בקישור, בלי מייל'),
      el('p', { className: 'muted' }, 'למי שמעדיפים וואטסאפ. בלי כתובת אי אפשר להיכנס עם Google עד שמחברים אותו מהדף האישי.'),
      el('div', { className: 'row' }, el('button', { type: 'button', onclick: () => invite() }, 'יצירת קישור הזמנה')),
      result,
      el('h2', {}, 'חברי הרשת'),
      ...members.map((member) => person({ ...member, name: member.name || 'עוד לא בחר שם' }, el('button', {
        className: 'plain',
        type: 'button',
        onclick: () => invite(member.id),
      }, 'קישור כניסה חדש')))));
}

function promise(picture, title, text) {
  return el('li', {}, el('span', { className: 'promise-icon' }, icon(picture)), el('div', {}, el('strong', {}, title), text));
}

function loginPage() {
  const email = el('input', { type: 'email', required: true, placeholder: 'כתובת מייל', dir: 'ltr' });
  const message = el('p', { className: 'muted' });
  const inviteNote = el('div', { className: 'card' },
    el('p', { className: 'flush' }, 'הרשת נמצאת בניסוי סגור. נכנסים עם הקישור האישי שבמייל ההזמנה, או עם חשבון Google של הכתובת שאליה היא נשלחה.'));
  const demoEntry = el('div', { className: 'card' },
    el('p', { style: 'margin-top: 0' }, 'כאן אפשר להציץ בממשק עם אנשים ופוסטים בדויים.'),
    el('div', { className: 'row' }, el('button', {
      type: 'button',
      onclick: async () => {
        await api('/auth/demo', { method: 'POST' });
        await start();
        go('/');
      },
    }, 'כניסה לדמו')));
  app.replaceChildren(el('section', { className: 'hero' },
    el('img', { src: '/favicon.svg', alt: '' }),
    el('h1', {}, 'השכונה שלך, בלי הרעש'),
    el('p', {}, 'הרשת החופשית של פרדס חנה-כרכור. מקום לחברים ולשכנים, שלא מנסה למכור לך כלום.')),
  googleLogin ? el('div', { className: 'card' }, el('p', { className: 'flush' }, 'הוזמנת בכתובת של Google, או כבר חיברת חשבון?'),
    el('div', { className: 'row' }, googleButton('/api/auth/google', 'כניסה עם Google'))) : '',
  demo ? demoEntry : inviteOnly ? inviteNote : el('form', {
    className: 'card',
    onsubmit: async (e) => {
      e.preventDefault();
      try {
        const { devLink } = await api('/auth/request', { method: 'POST', body: { email: email.value } });
        message.replaceChildren(el('a', { href: devLink }, 'קישור כניסה (מצב פיתוח)'));
      } catch (err) {
        message.textContent = err.message;
      }
    },
  },
  el('p', { style: 'margin-top: 0' }, 'נכנסים עם קישור למייל, בלי סיסמה.'),
  email,
  el('div', { className: 'row' }, el('button', { type: 'submit' }, 'שליחת קישור כניסה')),
  message),
  el('ul', { className: 'promises' },
    promise('eye', 'החברים שלך באמת יראו', 'כל פוסט מגיע לפיד של כל החברים, בלי סינון.'),
    promise('comment', 'כל התגובות, לפי הסדר', 'בלי "הרלוונטיות ביותר" ובלי תגובות שנעלמות.'),
    promise('leaf', 'בלי פרסומות', 'פרויקט ללא מטרות רווח, שיעבור לעמותה. אין מי שמרוויח מהזמן שלך.')),
  el('p', { className: 'feed-rule' }, el('a', { href: '/intro' }, 'סרטון קצר: מה זה ולמה'), ' · ', el('a', { href: '/why' }, 'למה בכלל צריך רשת חדשה?')));
}

function welcomeFlag(value) {
  try {
    if (value === true) sessionStorage.setItem(WELCOME_KEY, '1');
    if (value === false) sessionStorage.removeItem(WELCOME_KEY);
    return sessionStorage.getItem(WELCOME_KEY) === '1';
  } catch {
    return false;
  }
}

// The first screen of a new member: the name they will go by, their picture, and where to
// learn what this place is. Whoever was invited by name only has to confirm it.
function welcomePage() {
  const name = el('input', { type: 'text', required: true, maxLength: 60, placeholder: 'השם שיוצג לאחרים', value: me.name ?? '', dir: 'auto' });
  const error = el('p', { className: 'error', hidden: true });
  const first = (me.name ?? '').trim().split(/\s+/)[0];
  const face = avatar(me, { large: true });
  const google = googleLogin && !me.googleLinked
    ? el('div', { className: 'welcome-google' },
      googleButton('/api/auth/google?link=1', me.avatar ? 'חיבור חשבון Google' : 'להביא את התמונה מ-Google'),
      el('p', { className: 'note' }, 'חיבור החשבון מאפשר גם להיכנס מכל מכשיר בלחיצה. אנחנו מקבלים מגוגל רק שם, כתובת ותמונה.'))
    : null;
  app.replaceChildren(el('form', {
    className: 'card welcome',
    onsubmit: async (e) => {
      e.preventDefault();
      try {
        if (name.value.trim() !== me.name) await api('/me', { method: 'PUT', body: { name: name.value } });
        welcomeFlag(false);
        await start();
      } catch (err) {
        error.textContent = err.message;
        error.hidden = false;
      }
    },
  },
  el('h1', { dir: 'auto' }, first ? `ברוכים הבאים, ${first}` : 'ברוכים הבאים'),
  el('p', { className: 'muted' }, 'הרשת החופשית היא רשת חברתית בלי פרסומות ובלי בעלים שמרוויחים ממנה. ',
    el('a', { href: '/intro', target: '_blank' }, 'סרטון קצר מסביר מה זה ולמה.')),
  me.name ? el('div', { className: 'welcome-face' }, ...editablePicture('avatar', 'בחירת תמונת פרופיל', face, error),
    el('p', { className: 'note' }, me.avatar ? 'זו התמונה שלך. לחיצה עליה מחליפה אותה.' : 'בינתיים התמונה שלך מצוירת מהשם הפרטי. לחיצה עליה מעלה תמונה אחרת.')) : null,
  me.name ? google : null,
  el('label', { className: 'field' }, el('span', {}, 'השם שיוצג לאחרים'), name),
  el('div', { className: 'row' }, el('button', { type: 'submit' }, 'כניסה לרשת')), error));
}

async function logout() {
  await api('/auth/logout', { method: 'POST' });
  me = null;
  seenUpTo = undefined;
  go('/login');
  renderNav();
}

async function refreshMe() {
  ({ user: me } = await api('/me'));
  renderNav();
}

function renderNav() {
  const why = el('a', { href: '/why' }, labelled('info', 'למה אנחנו'));
  if (!me) {
    nav.replaceChildren(...[why, link('/login', labelled('user', 'כניסה'))].filter(Boolean));
    return;
  }
  const admin = me.admin ? link('/admin', labelled('settings', 'ניהול')) : null;
  const mine = me.name ? link(`/u/${me.id}`, labelled('user', 'הדף שלי')) : null;
  const home = link('/', labelled('home', 'בית'));
  const friends = link('/friends', labelled('users', 'חברים'));
  if (me.friendRequests) {
    friends.classList.add('has-count');
    friends.append(el('span', { className: 'count', ariaLabel: `${me.friendRequests} בקשות חברות` }, String(me.friendRequests)));
  }
  const world = link('/globe', labelled('globe', 'גלובוס'));
  const news = link('/update', labelled('sun', 'העדכון'));
  nav.replaceChildren(...[home, news, mine, friends, world, admin, why, el('button', {
    className: 'plain desktop-only',
    type: 'button',
    onclick: logout,
  }, labelled('logout', 'יציאה'))].filter(Boolean));
  markNav();
}

function markNav() {
  for (const item of nav.querySelectorAll('a')) item.classList.toggle('active', item.pathname === location.pathname);
}

function skeleton() {
  return el('div', { className: 'skeleton', ariaHidden: 'true' }, ...[0, 1, 2].map(() => el('div', { className: 'card' },
    el('div', { className: 'bone head-bone' }), el('div', { className: 'bone' }), el('div', { className: 'bone short' }))));
}

async function route() {
  markNav();
  const path = location.pathname;
  const arriving = path !== shownPath;
  // Only a move to another page gets the placeholder and the entrance. A refresh in place stays still.
  app.classList.toggle('arriving', arriving);
  const waiting = arriving ? setTimeout(() => app.replaceChildren(skeleton()), 160) : null;
  const post = location.pathname.match(/^\/p\/(\d+)$/);
  const profile = location.pathname.match(/^\/u\/(\d+)$/);
  try {
    await show(post, profile);
  } catch (err) {
    app.replaceChildren(el('div', { className: 'card empty' }, el('p', { className: 'error' }, err.message),
      el('div', { className: 'row center' }, el('button', { className: 'ghost', type: 'button', onclick: route }, 'לנסות שוב'))));
  } finally {
    clearTimeout(waiting);
    shownPath = path;
  }
}

async function show(post, profile) {
  if (post) await postPage(post[1]);
  else if (!me) loginPage();
  else if (!me.name || welcomeFlag()) welcomePage();
  else if (profile) await profilePage(profile[1]);
  else if (location.pathname === '/friends') await friendsPage();
  else if (location.pathname === '/globe') await globePage();
  else if (location.pathname === '/update') updatePage();
  else if (location.pathname === '/admin') await adminPage();
  else await feedPage();
}

async function start() {
  const token = new URLSearchParams(location.hash.slice(1)).get('token');
  let linkFailed = false;
  if (token) {
    history.replaceState(null, '', '/');
    try {
      if ((await api('/auth/verify', { method: 'POST', body: { token } })).joined) welcomeFlag(true);
    } catch {
      linkFailed = true;
    }
  }
  ({ user: me, demo, inviteOnly, googleLogin } = await api('/me'));
  // Opening an old link again is normal: someone already logged in just continues to the site.
  const google = new URLSearchParams(location.search).get('google');
  if (google) {
    history.replaceState(null, '', location.pathname);
    if (google === 'joined') welcomeFlag(true);
    if (GOOGLE_MESSAGES[google]) tell(GOOGLE_MESSAGES[google]);
  }
  if (linkFailed && !me) tell('הקישור הזה כבר נוצל או שפג תוקפו. בקשו קישור כניסה חדש ממי שהזמין אתכם.');
  document.getElementById('demo-banner').hidden = !demo;
  renderNav();
  if (location.pathname === '/login' && me) history.replaceState(null, '', '/');
  await route();
}

// A phone keeps the page open for days, so the count is refreshed on coming back to it.
document.addEventListener('visibilitychange', () => {
  if (!document.hidden && me) refreshMe().catch(() => {});
});
window.addEventListener('popstate', async () => {
  await route();
  scrollTo(0, scrollByPath.get(location.pathname) ?? 0);
});
window.addEventListener('hashchange', start);
start();
