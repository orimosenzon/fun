// Live pages: a web page that runs inside a post. The page runs in the reader's browser, in a
// frame that is closed off from the network (sandboxed, from another address), so it can do all
// it does without reaching anyone's account here. This server only reads the page once, for the
// title, the few words and the picture that readers see before they choose to load it.

const MAX_URL = 1000;
const MAX_READ = 512 * 1024;
const MAX_TITLE = 200;
const MAX_TEXT = 300;
const READ_MS = 5000;
const AGENT = 'free-network-page/1.0 (https://free-network.orimosenzon.workers.dev)';

// The address of a live page, cleaned, or null. Only https (a page from http would be blocked
// inside a secure one), no name and password in the address, and never the network itself: a
// page of its own address in the frame would no longer be closed off from it.
export function pageUrl(raw, { own, local = false } = {}) {
  if (typeof raw !== 'string') return null;
  const text = raw.trim();
  if (!text || text.length > MAX_URL) return null;
  let url;
  try {
    url = new URL(text);
  } catch {
    return null;
  }
  const allowed = url.protocol === 'https:' || (local && url.protocol === 'http:' && url.hostname === 'localhost');
  if (!allowed || url.username || url.password || (!url.hostname.includes('.') && !local)) return null;
  if (own && url.host === own) return null;
  return url.href;
}

const ENTITIES = { amp: '&', lt: '<', gt: '>', quot: '"', apos: "'", nbsp: ' ' };

function decode(text) {
  return text
    .replace(/&#x([0-9a-f]+);/gi, (_, hex) => String.fromCodePoint(Number.parseInt(hex, 16)))
    .replace(/&#(\d+);/g, (_, dec) => String.fromCodePoint(Number(dec)))
    .replace(/&([a-z]+);/gi, (whole, name) => ENTITIES[name.toLowerCase()] ?? whole)
    .replace(/\s+/g, ' ')
    .trim();
}

const clip = (text, size) => (text && text.length > size ? `${text.slice(0, size - 1)}…` : text || null);

// The content of a <meta> tag, whichever order its attributes come in.
function meta(html, names) {
  for (const tag of html.match(/<meta\b[^>]*>/gi) ?? []) {
    const name = tag.match(/\b(?:property|name)\s*=\s*["']([^"']+)["']/i)?.[1]?.toLowerCase();
    if (!names.includes(name)) continue;
    const content = tag.match(/\bcontent\s*=\s*(?:"([^"]*)"|'([^']*)')/i);
    if (content) return decode(content[1] ?? content[2]);
  }
  return null;
}

// Whether the site lets other pages show it in a frame. A site that says nothing allows it.
export function framable(headers, own) {
  const policy = (headers.get('Content-Security-Policy') ?? '').split(/[;,]/)
    .map((part) => part.trim().split(/\s+/)).find(([name]) => name?.toLowerCase() === 'frame-ancestors');
  if (policy) {
    return policy.slice(1).some((source) => {
      const value = source.toLowerCase().replace(/^https:\/\//, '').replace(/\/$/, '');
      if (value === '*' || value === 'https:') return true;
      if (value.startsWith('*.')) return !!own && own.endsWith(value.slice(1));
      return !!own && value === own;
    });
  }
  const options = (headers.get('X-Frame-Options') ?? '').toLowerCase();
  return !/deny|sameorigin|allow-from/.test(options);
}

async function readText(response) {
  const reader = response.body?.getReader();
  if (!reader) return '';
  const parts = [];
  let size = 0;
  while (size < MAX_READ) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    size += value.length;
  }
  reader.cancel().catch(() => {});
  const all = new Uint8Array(size);
  let at = 0;
  for (const part of parts) {
    all.set(part, at);
    at += part.length;
  }
  return new TextDecoder().decode(all.subarray(0, MAX_READ));
}

// Reads the page once: what it is called, a line about it, its picture, and whether it may be
// framed. A page that cannot be read is still a page: it is then shown by its address.
export async function readPage(url, { own, local = false } = {}) {
  const found = { title: null, text: null, picture: null, framable: true };
  const response = await fetch(url, {
    headers: { 'User-Agent': AGENT, Accept: 'text/html,*/*;q=0.5' },
    redirect: 'follow',
    signal: AbortSignal.timeout(READ_MS),
  }).catch(() => null);
  if (!response) return found;
  found.framable = framable(response.headers, own);
  if (!response.ok || !/html/i.test(response.headers.get('Content-Type') ?? '')) return found;
  const html = await readText(response).catch(() => '');
  const title = meta(html, ['og:title', 'twitter:title']) ?? html.match(/<title[^>]*>([^<]*)<\/title>/i)?.[1];
  found.title = clip(title && decode(title), MAX_TITLE);
  found.text = clip(meta(html, ['og:description', 'twitter:description', 'description']), MAX_TEXT);
  const picture = meta(html, ['og:image', 'og:image:url', 'twitter:image']);
  if (picture) {
    try {
      const at = new URL(picture, response.url || url);
      if (at.protocol === 'https:' || (local && at.hostname === 'localhost')) found.picture = clip(at.href, MAX_URL);
    } catch {
      // No picture, then.
    }
  }
  return found;
}

// The picture of a page, passed through this server so the reader's browser contacts nobody else.
export async function pagePicture(src) {
  const response = await fetch(src, {
    headers: { 'User-Agent': AGENT },
    signal: AbortSignal.timeout(READ_MS),
    cf: { cacheEverything: true, cacheTtl: 86400 },
  }).catch(() => null);
  const type = response?.headers.get('Content-Type') ?? '';
  if (!response?.ok || !/^image\/(jpeg|png|gif|webp)/.test(type)) return null;
  return new Response(response.body, {
    headers: { 'Content-Type': type, 'Cache-Control': 'private, max-age=86400', 'X-Content-Type-Options': 'nosniff' },
  });
}
