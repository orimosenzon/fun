// The update: what happened lately, in circles that widen from friends to the moshava,
// Israel and the world. Each circle folds into one line with a peek at what is inside, and
// opens to a short list with pictures. The reader's device remembers which circles they keep
// open. Nothing here is ranked: each list is newest first.

import { face, feelingLabel } from '/reactions.js';

const BOARD_URL = 'https://pardespedia.info/wiki/עמוד_ראשי#אירועי_תרבות_ובילוי_קרובים';
const PLANS_URL = 'https://orimosenzon.github.io/fun/vibe_coding/transparency/ma-bonim/';
const HE_NEWS_URL = 'https://he.wikipedia.org/wiki/תבנית:חדשות_ואקטואליה';
const EN_NEWS_URL = 'https://en.wikipedia.org/wiki/Portal:Current_events';
const DAY_MS = 86400000;
const SHOWN = 4;
const OPEN_KEY = 'update-open';

const format = (options) => new Intl.DateTimeFormat('he-IL', options);

function localDay(ms) {
  const date = new Date(ms);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`;
}

function eventDay(day) {
  if (day === localDay(Date.now())) return 'היום';
  if (day === localDay(Date.now() + DAY_MS)) return 'מחר';
  return format({ weekday: 'long' }).format(new Date(`${day}T12:00`));
}

function quietFor(last) {
  const months = Math.floor((Date.now() - last) / (30 * DAY_MS));
  if (months < 2) return 'יותר מחודש';
  return months === 2 ? 'חודשיים' : `${months} חודשים`;
}

function daysLeft(day) {
  const days = Math.round((Date.parse(`${day}T12:00`) - Date.parse(`${localDay(Date.now())}T12:00`)) / DAY_MS);
  if (days <= 0) return 'היום אחרון';
  if (days === 1) return 'נשאר יום';
  if (days === 2) return 'נשארו יומיים';
  return `נשארו ${days} ימים`;
}

// Beside the feed the friends and the moshava start open; above it, on a narrow screen,
// everything starts folded so the posts are not pushed down. Each place remembers on its own.
function openRings(top) {
  try {
    const saved = JSON.parse(localStorage.getItem(top ? `${OPEN_KEY}-top` : OPEN_KEY));
    if (Array.isArray(saved)) return new Set(saved);
  } catch {
    // No storage: the defaults below.
  }
  return new Set(top ? [] : ['friends', 'moshava']);
}

function keepOpen(kind, open, top) {
  try {
    const rings = openRings(top);
    if (open) rings.add(kind);
    else rings.delete(kind);
    localStorage.setItem(top ? `${OPEN_KEY}-top` : OPEN_KEY, JSON.stringify([...rings]));
  } catch {
    // Nothing to remember it in.
  }
}

// `all` opens every circle, for the update's own page; `top` is the update above the posts.
export function updatePanel({ me, seen, ui, all = false, top = false }) {
  const { el, icon, avatar, link, api, when, tell, flyGlobe } = ui;
  const body = el('div', { className: 'update-body' },
    el('div', { className: 'bone' }), el('div', { className: 'bone short' }), el('div', { className: 'bone' }));
  const root = el('section', { className: 'update' },
    el('header', { className: 'update-head' },
      el('h2', {}, all ? 'העדכון' : link('/update', 'העדכון')),
      el('span', { className: 'update-date' }, format({ weekday: 'long', day: 'numeric', month: 'long' }).format(Date.now()))),
    body);
  const open = openRings(top);

  const outside = (url, className, ...content) => (url
    ? el('a', { className, href: url, target: '_blank', rel: 'noopener' }, ...content)
    : el('div', { className }, ...content));
  const picture = (src) => el('img', { src, alt: '', loading: 'lazy', decoding: 'async', onerror: (e) => e.target.remove() });
  const thumb = (src, fallback) => el('span', { className: 'row-thumb' }, src ? picture(src) : null, src ? null : fallback);

  // A folded circle: one line with a peek, and the list inside.
  function ring(kind, pictureName, title, peek, ...children) {
    const details = el('details', { className: `update-ring ring-${kind}`, open: all || open.has(kind) },
      el('summary', {},
        el('span', { className: 'ring-icon' }, icon(pictureName)),
        el('span', { className: 'ring-title' }, title),
        el('span', { className: 'ring-peek', dir: 'auto' }, peek),
        el('span', { className: 'ring-chevron' }, icon('left'))),
      el('div', { className: 'ring-body' }, ...children.flat().filter(Boolean)));
    if (!all) details.addEventListener('toggle', () => keepOpen(kind, details.open, top));
    return details;
  }

  // The first few rows, and a button for the rest.
  function rows(list) {
    const items = list.flat().filter(Boolean);
    if (all || items.length <= SHOWN + 1) return items;
    const rest = items.slice(SHOWN);
    rest.forEach((item) => { item.hidden = true; });
    const more = el('button', {
      className: 'plain ring-more',
      type: 'button',
      onclick: () => {
        rest.forEach((item) => { item.hidden = false; });
        more.remove();
      },
    }, `ועוד ${rest.length}`);
    return [...items, more];
  }

  // A friend as a round picture, with what is new about them underneath.
  function bubble(user, { badge, mark, caption, quote, path, onclick }) {
    const content = [
      el('span', { className: `face-ring${mark ? ` ${mark}` : ''}` }, avatar(user, { large: true }), badge ? el('span', { className: `face-badge${badge.classList?.contains('face-feeling') ? ' feeling' : ''}` }, badge) : null),
      el('span', { className: 'face-name', dir: 'auto' }, user.name),
      quote ? el('span', { className: 'face-quote', dir: 'auto' }, quote) : null,
      caption ? el('span', { className: 'face-caption' }, caption) : null,
    ];
    if (onclick) return el('button', { className: 'face', type: 'button', onclick }, ...content);
    const node = link(path ?? `/u/${user.id}`, ...content);
    node.className = 'face';
    return node;
  }

  // The reader's own bubble: their line for friends, written or changed in place.
  function myBubble() {
    const wrap = el('div', { className: 'face-mine' });
    const form = el('form', { className: 'status-form', hidden: true });
    function draw() {
      wrap.replaceChildren(bubble(me, {
        badge: me.status ? null : icon('plus'),
        mark: 'mine',
        quote: me.status?.text ?? null,
        caption: me.status ? 'השורה שלך' : 'מה נשמע?',
        onclick: () => openForm(),
      }));
    }
    function openForm() {
      const input = el('input', { type: 'text', maxLength: 140, dir: 'auto', placeholder: 'מה נשמע? משפט אחד לחברים', ariaLabel: 'מה נשמע? משפט אחד לחברים', value: me.status?.text ?? '' });
      form.replaceChildren(
        input,
        el('button', { className: 'small', type: 'submit' }, 'שיתוף'),
        me.status ? el('button', {
          className: 'plain danger',
          type: 'button',
          onclick: async () => {
            try {
              await api('/me/status', { method: 'DELETE' });
              me.status = null;
              form.hidden = true;
              draw();
            } catch (err) {
              tell(err.message);
            }
          },
        }, 'מחיקה') : null,
        el('button', { className: 'plain', type: 'button', onclick: () => { form.hidden = true; } }, 'ביטול'),
        el('p', { className: 'update-source' }, 'רק החברים שלך רואים, והשורה נמחקת מעצמה אחרי שבוע.'));
      form.hidden = false;
      input.focus();
    }
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = form.querySelector('input').value.trim();
      if (!text) return;
      try {
        ({ status: me.status } = await api('/me/status', { method: 'PUT', body: { text } }));
        form.hidden = true;
        draw();
      } catch (err) {
        tell(err.message);
      }
    });
    draw();
    return { bubble: wrap, form };
  }

  function friendsRing({ statuses, fresh, replies, felt, moved, joined, quiet }) {
    const people = [
      ...replies.map((item) => [item.user, bubble(item.user, {
        badge: icon('comment'), mark: 'hot', path: `/p/${item.postId}`,
        caption: item.onMine ? 'הגיב לפוסט שלך' : 'ענה לתגובה שלך',
      })]),
      ...felt.map((item) => [item.user, bubble(item.user, {
        badge: el('span', { className: 'face-feeling', title: feelingLabel(item.kind) }, face(item.kind)), mark: 'hot', path: `/p/${item.postId}`,
        caption: item.onPost ? 'הגיב ברגש לפוסט שלך' : 'הגיב ברגש לתגובה שלך',
      })]),
      ...fresh.map((item) => [item.user, bubble(item.user, {
        badge: String(item.count), mark: 'hot',
        caption: item.count === 1 ? 'פוסט חדש' : 'פוסטים חדשים',
      })]),
      ...statuses.map((item) => [item.user, bubble(item.user, { quote: item.text, mark: 'says' })]),
      ...moved.map((item) => [item.user, bubble(item.user, {
        badge: icon('globe'), mark: 'moved', caption: 'במקום חדש', onclick: () => flyGlobe(item.place),
      })]),
      ...joined.map((item) => [item.user, bubble(item.user, { badge: 'חדש', mark: 'new', caption: el('span', {}, 'הצטרף ', when(item.at)) })]),
      ...quiet.map((item) => [item.user, bubble(item.user, {
        mark: 'quiet', caption: item.last ? `שקט כבר ${quietFor(item.last)}. אולי לכתוב?` : 'עוד לא כתב כאן. אולי לכתוב?',
      })]),
    ];
    const faces = [...new Map(people.map(([user]) => [user.id, user])).values()];
    const peek = faces.length
      ? el('span', { className: 'peek-faces' }, ...faces.slice(0, 4).map((user) => avatar(user, { small: true })),
        faces.length > 4 ? el('span', { className: 'peek-count' }, `+${faces.length - 4}`) : null)
      : 'שקט';
    const mine = myBubble();
    return ring('friends', 'users', 'החברים', peek,
      el('div', { className: 'faces' }, mine?.bubble, ...people.map(([, node]) => node)),
      mine?.form,
      people.length ? null : el('p', { className: 'update-none' }, 'שקט אצל החברים בימים האחרונים.'));
  }

  function communityRing({ events, moreEvents, plans }) {
    if (!events.length && !plans.length) return null;
    const short = (day) => format({ day: 'numeric', month: 'numeric' }).format(new Date(`${day}T12:00`));
    const today = events.filter((event) => eventDay(event.day) === 'היום').length;
    const peek = today ? `${today} אירועים היום` : events.length ? events[0].name : `${plans.length} תוכניות פתוחות`;
    return ring('moshava', 'home', 'במושבה', peek,
      rows(events.map((event) => outside(event.url, 'update-row',
        thumb(event.image, '🎭'),
        el('span', { className: 'row-text' },
          el('span', { className: 'row-chip' }, [eventDay(event.day), event.time].filter(Boolean).join(' · ')),
          el('span', { className: 'row-title', dir: 'auto' }, event.name),
          event.venue ? el('span', { className: 'row-small' }, event.venue) : null)))),
      events.length ? el('p', { className: 'update-source' },
        el('a', { href: BOARD_URL, target: '_blank', rel: 'noopener' }, moreEvents ? `עוד ${moreEvents} אירועים בפרדספדיה` : 'לוח האירועים בפרדספדיה')) : null,
      plans.length ? el('h4', {}, 'תוכניות בנייה שאפשר להתנגד להן') : null,
      plans.map((plan) => outside(plan.url, 'update-row plan',
        el('span', { className: 'row-thumb plan-art' }, '🏗️'),
        el('span', { className: 'row-text' },
          el('span', { className: 'row-chip urgent' }, `${daysLeft(plan.shuts)}, עד ${short(plan.shuts)}`),
          el('span', { className: 'row-title', dir: 'auto' }, plan.title),
          plan.adds ? el('span', { className: 'row-small' }, plan.adds) : null))),
      plans.length ? el('p', { className: 'update-source' },
        el('a', { href: PLANS_URL, target: '_blank', rel: 'noopener' }, 'כל התוכניות ב"מה בונים לנו"')) : null);
  }

  function newsRows(items, dir) {
    return rows(items.map((item) => outside(item.url, `update-row news${item.key ? ` ch-${item.key}` : ''}`,
      thumb(item.image, icon('globe')),
      el('span', { className: 'row-text' },
        item.channel ? el('span', { className: 'row-chip channel' }, item.channel) : null,
        el('span', { className: 'row-title', dir }, item.text)))));
  }

  // Israel: headlines from four channels in turn, each marked with its channel.
  function israelRing(items) {
    if (!items?.length) return null;
    return ring('israel', 'flag', 'בישראל', items[0].text, newsRows(items, 'rtl'),
      el('p', { className: 'update-source' }, 'כותרות מ-N12, חדשות 13, עכשיו 14 וכאן חדשות, לסירוגין.'));
  }

  function newsRing(kind, title, items, sourceUrl, sourceLabel, dir) {
    if (!items.length) return null;
    return ring(kind, 'globe', title, items[0].text, newsRows(items, dir),
      el('p', { className: 'update-source' }, el('a', { href: sourceUrl, target: '_blank', rel: 'noopener' }, sourceLabel), ', ברישיון CC BY-SA'));
  }

  api(`/update${seen ? `?seen=${seen}` : ''}`).then((data) => {
    body.replaceChildren(...[
      friendsRing(data.friends),
      communityRing(data.community),
      israelRing(data.israel),
      newsRing('country', 'בעולם', data.country, HE_NEWS_URL, 'מתוך ויקיפדיה העברית', 'rtl'),
      newsRing('world', 'בעולם, באנגלית', data.world, EN_NEWS_URL, 'מתוך ויקיפדיה האנגלית', 'ltr'),
      all ? el('p', { className: 'update-end' }, 'זה כל העדכון. אין לו המשך לגלול אליו.') : null].filter(Boolean));
  }).catch((err) => {
    body.replaceChildren(el('p', { className: 'error' }, err.message));
  });
  return root;
}
