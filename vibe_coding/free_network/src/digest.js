// News from outside the network. The server fetches each source a few times a day and keeps
// the result, so a reader's browser never contacts the source and nobody learns who is reading.
// Every source is best effort: when one fails, the last good copy is served.

const FRESH_MS = 3 * 60 * 60 * 1000;
const RETRY_MS = 15 * 60 * 1000;
const FETCH_TIMEOUT_MS = 6000;
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
  },
  world: {
    url: (env) => `${env.WIKI_EN_ORIGIN ?? 'https://en.wikipedia.org'}/w/api.php?action=parse&page=Template:In_the_news&prop=text&format=json&formatversion=2`,
    read: (data) => wikiNews(data.parse.text, 'https://en.wikipedia.org'),
  },
};

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
    for (const card of part.split('font-size:125%; font-weight:bold; line-height:1.3;">').slice(1)) {
      const title = card.split('</div>')[0];
      if (title.includes('בוטל')) continue;
      const linked = title.match(/^\[(https?:\/\/\S+) ([^\]]*)\]/);
      const venue = card.match(/📍<\/span>(?:\[\S+ ([^\]]*)\]|([^<]*))/);
      events.push({
        day,
        name: plainText(linked ? linked[2] : title),
        url: linked ? linked[1] : null,
        time: card.match(/🕗<\/span>([^<]*)</)?.[1].trim() || null,
        venue: plainText(venue?.[1] ?? venue?.[2] ?? '') || null,
      });
    }
  }
  return events.filter((event) => event.name);
}

async function refresh(env, key, now) {
  const source = SOURCES[key];
  const response = await fetch(source.url(env), {
    headers: { 'User-Agent': 'free-network-digest/1.0' },
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  });
  if (!response.ok) throw new Error(`${key}: ${response.status}`);
  const items = source.read(await response.json(), now);
  if (!Array.isArray(items) || !items.length) throw new Error(`${key}: nothing to read`);
  return items;
}

// Returns { events, plans, country, world }, each a list, or empty when never fetched.
export async function outsideNews(env, now = Date.now()) {
  const { results } = await env.DB.prepare('SELECT key, body, fetched_at, tried_at FROM digest_cache').all();
  const cached = Object.fromEntries(results.map((row) => [row.key, row]));
  const news = {};
  await Promise.all(Object.keys(SOURCES).map(async (key) => {
    const row = cached[key];
    news[key] = row ? JSON.parse(row.body) : [];
    if (row && now - row.tried_at < FRESH_MS) return;
    try {
      news[key] = await refresh(env, key, now);
      await env.DB
        .prepare(`INSERT INTO digest_cache (key, body, fetched_at, tried_at) VALUES (?1, ?2, ?3, ?3)
          ON CONFLICT (key) DO UPDATE SET body = ?2, fetched_at = ?3, tried_at = ?3`)
        .bind(key, JSON.stringify(news[key]), now)
        .run();
    } catch (error) {
      console.log(JSON.stringify({ digest: key, error: error.message }));
      // A failed source is tried again in a quarter of an hour, not on every request.
      await env.DB
        .prepare(`INSERT INTO digest_cache (key, body, fetched_at, tried_at) VALUES (?1, '[]', 0, ?2)
          ON CONFLICT (key) DO UPDATE SET tried_at = ?2`)
        .bind(key, now - FRESH_MS + RETRY_MS)
        .run();
    }
  }));

  const today = israelDay(now);
  news.events = news.events.filter((event) => event.day >= today);
  news.plans = news.plans.filter((plan) => plan.shuts >= today).sort((a, b) => a.shuts.localeCompare(b.shuts));
  return news;
}
