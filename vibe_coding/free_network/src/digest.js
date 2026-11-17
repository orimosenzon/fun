// News from outside the network. The server fetches each source a few times a day and keeps
// the result, so a reader's browser never contacts the source and nobody learns who is reading.
// Every source is best effort: when one fails, the last good copy is served.

const FRESH_MS = 3 * 60 * 60 * 1000;
const RETRY_MS = 15 * 60 * 1000;
const FETCH_TIMEOUT_MS = 6000;
// Raised whenever what is kept for a source changes shape, so an older copy is fetched anew.
const CACHE_VERSION = 2;
const HE_MONTHS = ['ינואר', 'פברואר', 'מרץ', 'אפריל', 'מאי', 'יוני', 'יולי', 'אוגוסט', 'ספטמבר', 'אוקטובר', 'נובמבר', 'דצמבר'];
const PLANS_SITE = 'https://orimosenzon.github.io/fun/vibe_coding/transparency/ma-bonim/';

const SOURCES = {
  events: {
    url: (env) => `${env.PARDESPEDIA_ORIGIN ?? 'https://pardespedia.info'}/api.php?action=parse&page=${encodeURIComponent('עמוד ראשי')}&prop=wikitext&format=json&formatversion=2`,
    read: (data, now) => boardEvents(data.parse.wikitext, now),
  },
  plans: {
    url: (env) => env.PLANS_URL ?? 'https://raw.githubusercontent.com/orimosenzon/derech-kitzur-data/main/data/objections.json',
    read: (data) => data.plans.filter((plan) => !plan.hidden && plan.shuts).map((plan) => ({
      title: String(plan.address || plan.name),
      adds: plan.adds ? String(plan.adds) : null,
      shuts: String(plan.shuts),
      url: `${PLANS_SITE}#p-${encodeURIComponent(plan.num)}`,
    })),
  },
  country: {
    url: (env) => `${env.WIKI_HE_ORIGIN ?? 'https://he.wikipedia.org'}/w/api.php?action=parse&page=${encodeURIComponent('תבנית:חדשות ואקטואליה')}&prop=text&format=json&formatversion=2`,
    read: (data) => wikiNews(data.parse.text, 'https://he.wikipedia.org'),
    pictures: (env) => env.WIKI_HE_ORIGIN ?? 'https://he.wikipedia.org',
  },
  world: {
    url: (env) => `${env.WIKI_EN_ORIGIN ?? 'https://en.wikipedia.org'}/w/api.php?action=parse&page=Template:In_the_news&prop=text&format=json&formatversion=2`,
    read: (data) => wikiNews(data.parse.text, 'https://en.wikipedia.org'),
    pictures: (env) => env.WIKI_EN_ORIGIN ?? 'https://en.wikipedia.org',
  },
  // Israel, from four channels. N12 and 14 publish feeds; Kan and 13 turn away automatic
  // readers, so their headlines come through the Google News feed of each site.
  n12: {
    url: (env) => env.N12_FEED ?? 'https://rcs.mako.co.il/rss/31750a2610f26110VgnVCM1000005201000aRCRD.xml',
    text: true,
    read: (xml) => feedItems(xml),
  },
  c14: {
    url: (env) => env.C14_FEED ?? 'https://www.c14.co.il/feed/',
    text: true,
    read: (xml) => feedItems(xml),
  },
  kan: {
    url: (env) => env.KAN_FEED ?? googleNews('kan.org.il'),
    text: true,
    read: (xml) => feedItems(xml),
  },
  c13: {
    url: (env) => env.C13_FEED ?? googleNews('13tv.co.il'),
    text: true,
    read: (xml) => feedItems(xml),
  },
};

export const CHANNELS = [
  { key: 'n12', name: 'N12' },
  { key: 'c13', name: 'חדשות 13' },
  { key: 'c14', name: 'עכשיו 14' },
  { key: 'kan', name: 'כאן חדשות' },
];

function googleNews(site) {
  return `https://news.google.com/rss/search?q=${encodeURIComponent(`site:${site} when:1d`)}&hl=he&gl=IL&ceid=IL:he`;
}

const unwrap = (value) => value.replace(/^\s*<!\[CDATA\[([\s\S]*?)\]\]>\s*$/, '$1');

// The headlines of an RSS feed: title, link, a picture when the feed carries one, and the time.
// Only the headline is kept, never the body of the article.
export function feedItems(xml) {
  return [...xml.matchAll(/<item\b[^>]*>([\s\S]*?)<\/item>/g)].map(([, item]) => {
    const tag = (name) => {
      const found = item.match(new RegExp(`<${name}\\b[^>]*>([\\s\\S]*?)<\/${name}>`));
      return found ? unwrap(found[1]) : '';
    };
    const decoded = (html) => html.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&amp;/g, '&');
    const picture = tag('image').trim()
      || item.match(/<(?:media:content|media:thumbnail|enclosure)\b[^>]*url="([^"]+)"/)?.[1]
      || decoded(tag('content:encoded') + tag('description')).match(/<img[^>]+src=["']([^"']+)["']/)?.[1]
      || null;
    const at = Date.parse(tag('pubDate'));
    return {
      // A Google News headline ends with " - <site name>".
      text: plainText(decoded(tag('title'))).replace(/\s+-\s+[^-]{2,40}$/, item.includes('<source') ? '' : '$&'),
      url: tag('link').trim() || null,
      image: picture && /^https:\/\//.test(picture) ? picture.replace(/&amp;/g, '&') : null,
      at: Number.isFinite(at) ? at : null,
    };
  }).filter((item) => item.text && item.url && /^https:\/\//.test(item.url));
}

// One headline from each channel in turn, newest first within each, so none of them leads the row.
export function israelNews(news, perChannel = 3) {
  const lists = CHANNELS.map((channel) => (news[channel.key] ?? [])
    .slice().sort((a, b) => (b.at ?? 0) - (a.at ?? 0)).slice(0, perChannel)
    .map((item) => ({ ...item, channel: channel.name, key: channel.key })));
  const out = [];
  for (let i = 0; i < perChannel; i++) for (const list of lists) if (list[i]) out.push(list[i]);
  return out;
}

// Where pictures may come from. The reader's browser gets them through this server, like the rest.
export const PICTURE_HOSTS = ['upload.wikimedia.org', 'thumb.wikimedia.org', 'pardespedia.info', 'img.mako.co.il', 'www.c14.co.il'];
const USER_AGENT = 'free-network-digest/1.0 (https://free-network.orimosenzon.workers.dev)';

// The lead picture of the article each news item is about, when it has one. Best effort:
// a news line without a picture is still news.
async function addPictures(items, origin) {
  const titles = items.map((item) => item.url && decodeURIComponent(item.url.split('/wiki/')[1] ?? '').replace(/_/g, ' ')).filter(Boolean);
  if (!titles.length) return items;
  try {
    const query = new URLSearchParams({ action: 'query', prop: 'pageimages', piprop: 'thumbnail', pithumbsize: '480', redirects: '1', format: 'json', formatversion: '2', titles: titles.join('|') });
    const response = await fetch(`${origin}/w/api.php?${query}`, { headers: { 'User-Agent': USER_AGENT }, signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) });
    const data = await response.json();
    const renamed = new Map([...(data.query?.normalized ?? []), ...(data.query?.redirects ?? [])].map((step) => [step.from, step.to]));
    const pictures = new Map((data.query?.pages ?? []).filter((page) => page.thumbnail).map((page) => [page.title, page.thumbnail.source]));
    return items.map((item) => {
      let title = item.url && decodeURIComponent(item.url.split('/wiki/')[1] ?? '').replace(/_/g, ' ');
      for (let i = 0; i < 3 && renamed.has(title); i++) title = renamed.get(title);
      const picture = pictures.get(title);
      return picture && PICTURE_HOSTS.includes(new URL(picture).host) ? { ...item, image: picture } : item;
    });
  } catch {
    return items;
  }
}

function plainText(html) {
  return html
    .replace(/<[^>]+>/g, '')
    .replace(/&#(\d+);/g, (_, code) => String.fromCodePoint(Number(code)))
    .replace(/&#x([0-9a-f]+);/gi, (_, code) => String.fromCodePoint(parseInt(code, 16)))
    .replace(/&quot;/g, '"').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&')
    .replace(/\s+/g, ' ')
    .trim();
}

// The news box of a Wikipedia main page: one line per item, linked to the article it is about.
export function wikiNews(html, origin) {
  const list = html.match(/<ul>([\s\S]*?)<\/ul>/)?.[1] ?? '';
  return [...list.matchAll(/<li>([\s\S]*?)<\/li>/g)].map(([, item]) => {
    const body = item
      .replace(/\s*\(<a [^>]*class="external[^>]*>[^<]*<\/a>\)\s*$/, '')
      .replace(/\s*<small>[\s\S]*?<\/small>/g, '')
      .replace(/\s*<i>\([^<]*\)<\/i>/g, '');
    const article = body.match(/<b>(?:<i>)?<a href="(\/wiki\/[^"]+)"/)
      ?? body.match(/<a href="(\/wiki\/[^"]+)"[^>]*>(?:<i>)?<b>/)
      ?? body.match(/<a href="(\/wiki\/[^"]+)"/);
    return { text: plainText(body), url: article ? origin + article[1] : null };
  }).filter((item) => item.text);
}

export function israelDay(now) {
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Jerusalem' }).format(now);
}

// The events board on the Pardespedia main page is written by a script, a heading per day and
// a card per event. The headings carry no year, so each takes the year nearest to now.
export function boardEvents(wikitext, now) {
  const board = wikitext.split('AUTO:START')[1]?.split('AUTO:END')[0] ?? '';
  const origin = 'https://pardespedia.info';
  const today = israelDay(now);
  const events = [];
  for (const part of board.split(/border-bottom:2px solid #e3bcac[^>]*>/).slice(1)) {
    const heading = part.match(/(\d{1,2}) ב([א-ת]+)/);
    const month = heading ? HE_MONTHS.indexOf(heading[2]) : -1;
    if (month < 0) continue;
    const year = Number(today.slice(0, 4));
    const day = [year - 1, year, year + 1]
      .map((y) => `${y}-${String(month + 1).padStart(2, '0')}-${heading[1].padStart(2, '0')}`)
      .sort((a, b) => Math.abs(Date.parse(a) - Date.parse(today)) - Math.abs(Date.parse(b) - Date.parse(today)))[0];
    const cards = part.split('font-size:125%; font-weight:bold; line-height:1.3;">');
    for (let i = 1; i < cards.length; i++) {
      const card = cards[i];
      const title = card.split('</div>')[0];
      // The card's picture comes just before its title.
      const file = [...cards[i - 1].matchAll(/\[\[(?:קובץ|File|Image|תמונה):([^|\]]+)/g)].at(-1)?.[1].trim();
      if (title.includes('בוטל')) continue;
      const linked = title.match(/^\[(https?:\/\/\S+) ([^\]]*)\]/);
      const venue = card.match(/📍<\/span>(?:\[\S+ ([^\]]*)\]|([^<]*))/);
      events.push({
        day,
        name: plainText(linked ? linked[2] : title),
        url: linked ? linked[1] : null,
        time: card.match(/🕗<\/span>([^<]*)</)?.[1].trim() || null,
        venue: plainText(venue?.[1] ?? venue?.[2] ?? '') || null,
        image: file ? `${origin}/index.php?title=Special:Redirect/file/${encodeURIComponent(file)}&width=480` : null,
      });
    }
  }
  return events.filter((event) => event.name);
}

async function refresh(env, key, now) {
  const source = SOURCES[key];
  const response = await fetch(source.url(env), {
    headers: { 'User-Agent': USER_AGENT },
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  });
  if (!response.ok) throw new Error(`${key}: ${response.status}`);
  const items = source.read(source.text ? await response.text() : await response.json(), now);
  if (!Array.isArray(items) || !items.length) throw new Error(`${key}: nothing to read`);
  return source.pictures ? addPictures(items, source.pictures(env)) : items;
}

// Returns a list for each source, empty when it was never fetched.
export async function outsideNews(env, now = Date.now()) {
  const { results } = await env.DB.prepare('SELECT key, body, fetched_at, tried_at FROM digest_cache').all();
  const cached = Object.fromEntries(results.map((row) => [row.key, row]));
  const stored = (key) => `${key}@${CACHE_VERSION}`;
  const news = {};
  await Promise.all(Object.keys(SOURCES).map(async (key) => {
    const row = cached[stored(key)];
    news[key] = row ? JSON.parse(row.body) : [];
    if (row && now - row.tried_at < FRESH_MS) return;
    try {
      news[key] = await refresh(env, key, now);
      await env.DB
        .prepare(`INSERT INTO digest_cache (key, body, fetched_at, tried_at) VALUES (?1, ?2, ?3, ?3)
          ON CONFLICT (key) DO UPDATE SET body = ?2, fetched_at = ?3, tried_at = ?3`)
        .bind(stored(key), JSON.stringify(news[key]), now)
        .run();
    } catch (error) {
      console.log(JSON.stringify({ digest: key, error: error.message }));
      // A failed source is tried again in a quarter of an hour, not on every request.
      await env.DB
        .prepare(`INSERT INTO digest_cache (key, body, fetched_at, tried_at) VALUES (?1, '[]', 0, ?2)
          ON CONFLICT (key) DO UPDATE SET tried_at = ?2`)
        .bind(stored(key), now - FRESH_MS + RETRY_MS)
        .run();
    }
  }));

  const today = israelDay(now);
  news.events = news.events.filter((event) => event.day >= today);
  news.plans = news.plans.filter((plan) => plan.shuts >= today).sort((a, b) => a.shuts.localeCompare(b.shuts));
  return news;
}
