/* מה בונים לנו: phase 1, "what is open now".
 *
 * Two sources, both read from the browser:
 *
 *  1. objections.json in the derech-kitzur-data repo. Written every morning at
 *     07:15 by dereh_kitzur/watch_objections.py, which reads Xplan *and* each
 *     deposited plan's page on מבא"ת. That second read matters: on 27/9/2026
 *     Xplan had a closing date for 5 of the 9 plans that were really open.
 *     The browser cannot do that read itself (מבא"ת has no CORS and wants a
 *     reCAPTCHA v3 token), so the cron does it and this page only displays it.
 *
 *  2. Xplan itself, live, for the lists that need no מבא"ת: plans on their way
 *     to a window, and plans approved in the last year. Xplan answers CORS for
 *     github.io and localhost.
 */
'use strict';

const OBJECTIONS = 'https://raw.githubusercontent.com/orimosenzon/derech-kitzur-data/main/data/objections.json';
const XPLAN = 'https://ags.iplan.gov.il/arcgisiplan/rest/services/PlanningPublic/Xplan/MapServer/1/query';
const AREA = "plan_area_name LIKE '%פרדס חנה%'";
const SITE = 'https://orimosenzon.github.io/fun/vibe_coding/transparency/ma-bonim/';
const LOCAL = 3;          // pl_by_auth_of: 3 = local committee, 2 = district

const DAY = 864e5;
const today = (() => { const d = new Date(); d.setHours(0, 0, 0, 0); return d; })();

const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = (d, opts = { day: 'numeric', month: 'long' }) => d.toLocaleDateString('he-IL', opts);
const fromIso = s => { const [y, m, d] = s.split('-').map(Number); return new Date(y, m - 1, d); };
// Xplan stores a day as midnight UTC; the UTC date is the day itself.
const fromStamp = ms => { const u = new Date(ms); return new Date(u.getUTCFullYear(), u.getUTCMonth(), u.getUTCDate()); };
const daysLeft = d => Math.round((d - today) / DAY);

/* Plan names are written for the committee: "שינוי בהוראות בניה גוש 12783
 * חלקה 92 רח' עציון פח"כ". Drop the town (everything here is in it) and the
 * trailing abbreviations; leave the rest, since gush and parcel are what a
 * resident will search for. */
function tidy(name) {
  return String(name || '')
    .replace(/\s*,?\s*(ב?פרדס[\s-]*חנה[\s-]*כרכור|ב?פרדס[\s-]*חנה|ב?פח"כ|ב?פ"ח)\s*\.?\s*$/u, '')
    .replace(/\s+/g, ' ').trim();
}

/* What a plan adds, in words. Same columns and wording as build_plans.adds()
 * in dereh_kitzur, so the two sites say the same thing about the same plan. */
const QUANT = [
  ['quantity_delta_120', n => n === 1 ? 'יחידת דיור אחת' : `${fmtN(n)} יחידות דיור`],
  ['quantity_delta_125', n => `${fmtN(n)} מ"ר שטחי מגורים`],
  ['quantity_delta_75', n => `${fmtN(n)} מ"ר מסחר`],
  ['quantity_delta_60', n => `${fmtN(n)} מ"ר תעסוקה`],
  ['quantity_delta_80', n => `${fmtN(n)} מ"ר מבני ציבור`],
];
function fmtN(n) { return Math.round(n).toLocaleString('en-US'); }
function adds(at) {
  const parts = [];
  for (const [f, say] of QUANT) {
    const n = Number(at[f]);
    if (!n) continue;
    parts.push(n > 0 ? say(n) : 'גריעה של ' + say(-n));
  }
  if (!parts.length) return '';
  return parts.length === 1 ? parts[0] : parts.slice(0, -1).join(', ') + ' ו' + parts.at(-1);
}
// Hebrew writes "ו-300", with a hyphen, when the conjunction meets a digit.
// objections.json comes from Python without it, so both paths pass through here.
const hy = s => String(s || '').replace(/(^|\s)ו(\d)/g, '$1ו-$2');

/* ------------------------------------------------------------ open now */

let openPlans = [];

function whereToObject(p) {
  return p.local
    ? 'מגישים לוועדה המקומית לתכנון ובנייה פרדס חנה-כרכור, עם העתק לוועדה המחוזית חיפה.'
    : 'אפשר להגיש מקוון בדף התוכנית במבא"ת ("הגשת התנגדות לתוכנית"), או לוועדה המחוזית חיפה עם העתק לוועדה המקומית.';
}

function card(p) {
  const shuts = fromIso(p.shuts);
  const left = daysLeft(shuts);
  const leftWord = left === 0 ? 'נסגר היום' : left === 1 ? 'יום אחרון מחר' : 'ימים';
  const tags = [];
  if (left <= 7) tags.push('<span class="tag t-open">נסגר השבוע</span>');
  if (!p.exact) tags.push('<span class="tag t-soon">מועד משוער</span>');
  if (p.hidden) tags.push('<span class="tag t-blue">לא מופיע באתר הארצי</span>');
  const title = p.address || tidy(p.name);
  const sub = p.address ? tidy(p.name) : '';
  return `<article class="card${left <= 7 ? ' soon' : ''}" id="p-${esc(p.num)}">
    <div class="clock"><b>${left}</b><span>${left > 1 ? leftWord : ''}</span></div>
    <h3>${esc(title)} ${tags.join(' ')}</h3>
    <div>${p.adds ? `<span class="adds">${esc(hy(p.adds))}</span>` : '<span class="muted">שינוי בהוראות, בלי תוספת שטחים</span>'}
      ${sub ? `<div class="small muted">${esc(sub)}</div>` : ''}</div>
    <div class="meta">
      <span>אפשר להתנגד עד <b>${fmt(shuts, { weekday: 'long', day: 'numeric', month: 'long' })}</b></span>
      <span>${p.local ? 'תוכנית של הוועדה המקומית' : 'תוכנית של הוועדה המחוזית'}</span>
      <span class="mono">${esc(p.num)}</span>
    </div>
    ${!p.exact && p.basis ? `<div class="how small">${esc(p.basis)}</div>` : ''}
    <div class="how">${whereToObject(p)}</div>
    <div class="links"><a href="${esc(p.url)}" target="_blank" rel="noopener">מסמכי התוכנית במבא"ת</a>
      ${p.centre ? `<a href="#map" data-fly="${esc(p.num)}">במפה</a>` : ''}</div>
  </article>`;
}

let map, layers = {};
function drawMap(plans) {
  if (!window.L) { $('map').hidden = true; return; }
  map = L.map('map', { scrollWheelZoom: false, attributionControl: true }).setView([32.473, 34.97], 13);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19, attribution: '© OpenStreetMap',
  }).addTo(map);
  const red = getComputedStyle(document.documentElement).getPropertyValue('--open').trim() || '#b3261e';
  const bounds = [];
  for (const p of plans) {
    const rings = (p.rings || []).map(r => r.map(([x, y]) => [y, x]));
    let layer;
    if (rings.length) {
      layer = L.polygon(rings, { color: red, weight: 2.5, fillOpacity: 0.25 });
      rings.forEach(r => r.forEach(pt => bounds.push(pt)));
    } else if (p.centre) {
      layer = L.circleMarker([p.centre[1], p.centre[0]], { radius: 9, color: red, fillOpacity: 0.5 });
      bounds.push([p.centre[1], p.centre[0]]);
    }
    if (!layer) continue;
    const left = daysLeft(fromIso(p.shuts));
    layer.bindPopup(`<b>${esc(p.address || tidy(p.name))}</b><br>${esc(hy(p.adds))}<br>עוד ${left} ימים · <a href="#p-${esc(p.num)}">פרטים</a>`);
    layer.addTo(map);
    layers[p.num] = layer;
  }
  if (bounds.length) map.fitBounds(bounds, { padding: [30, 30], maxZoom: 16 });
}

document.addEventListener('click', e => {
  const a = e.target.closest('[data-fly]');
  if (!a || !map) return;
  const layer = layers[a.dataset.fly];
  if (!layer) return;
  e.preventDefault();
  $('map').scrollIntoView({ behavior: 'smooth', block: 'center' });
  if (layer.getBounds) map.fitBounds(layer.getBounds(), { maxZoom: 17 }); else map.setView(layer.getLatLng(), 17);
  layer.openPopup();
});

function shareText(plans) {
  const head = `פתוח עכשיו להתנגדות בפרדס חנה-כרכור (${fmt(today, { day: 'numeric', month: 'numeric' })}):`;
  if (!plans.length) return `${head}\nאין כרגע תוכניות פתוחות להתנגדות.\n${SITE}`;
  const lines = plans.map(p => {
    const shuts = fromIso(p.shuts), left = daysLeft(shuts);
    const what = p.adds ? `: ${hy(p.adds)}` : '';
    const est = p.exact ? '' : ' (משוער)';
    return `• ${p.address || tidy(p.name)}${what}. עד ${fmt(shuts, { day: 'numeric', month: 'numeric' })}${est}, עוד ${left} ימים`;
  });
  return [head, ...lines, '', `פרטים, מסמכים ואיך מתנגדים: ${SITE}`].join('\n');
}

$('copy').addEventListener('click', async () => {
  const text = shareText(openPlans);
  const box = $('share-text');
  box.value = text;
  try {
    await navigator.clipboard.writeText(text);
    $('copy-note').textContent = 'הועתק. אפשר להדביק בקבוצה.';
    box.hidden = true;
  } catch {
    box.hidden = false; box.focus(); box.select();
    $('copy-note').textContent = 'לא הצלחתי להעתיק אוטומטית. הטקסט מסומן למטה, העתיקו אותו ידנית.';
  }
});

async function loadOpen() {
  let doc;
  try {
    const r = await fetch(OBJECTIONS, { cache: 'no-cache' });
    if (!r.ok) throw new Error(r.status);
    doc = await r.json();
  } catch (err) {
    $('open-list').innerHTML = `<div class="empty err">לא הצלחתי לטעון את רשימת התוכניות הפתוחות (${esc(err.message)}). נסו לרענן.</div>`;
    $('fresh').textContent = '';
    return;
  }
  // The file is written in the morning; a plan whose day has passed since then is shut.
  openPlans = (doc.plans || []).filter(p => p.shuts && daysLeft(fromIso(p.shuts)) >= 0)
    .sort((a, b) => a.shuts.localeCompare(b.shuts));
  const checked = fromIso(doc.day);
  const stale = daysLeft(checked) < -1;
  $('fresh').innerHTML = `נבדק ${daysLeft(checked) === 0 ? 'הבוקר' : 'ב-' + fmt(checked)}: ${doc.checked} תוכניות בשלב ההפקדה`
    + (stale ? ' <span class="tag t-soon">הבדיקה היומית לא רצה, המידע ישן</span>' : '');
  $('open-count').textContent = openPlans.length ? `${openPlans.length} תוכניות` : '';
  $('open-list').innerHTML = openPlans.length
    ? openPlans.map(card).join('')
    : '<div class="empty">אין כרגע תוכנית פתוחה להתנגדות. הרשימה למטה מראה מה בדרך.</div>';
  drawMap(openPlans);
}

/* ------------------------------------------------------------ Xplan lists */

async function xplan(where, fields, order) {
  const body = new URLSearchParams({
    where: `${AREA} AND ${where}`, outFields: fields.join(','), returnGeometry: 'false',
    orderByFields: order || '', f: 'json',
  });
  const r = await fetch(XPLAN, { method: 'POST', body });
  if (!r.ok) throw new Error(`Xplan ${r.status}`);
  const doc = await r.json();
  if (doc.error) throw new Error(doc.error.message || 'Xplan');
  return (doc.features || []).map(f => f.attributes);
}

const BASE = ['pl_number', 'pl_name', 'pl_url', 'internet_short_status', 'pl_by_auth_of',
  ...QUANT.map(q => q[0])];

function row(at, side, info) {
  const what = hy(adds(at));
  return `<div class="row">
    <div class="name"><a href="${esc(at.pl_url || '#')}" target="_blank" rel="noopener">${esc(tidy(at.pl_name) || at.pl_number)}</a></div>
    <div class="side">${side}</div>
    <div class="info">${what ? `<span><b>${esc(what)}</b></span>` : ''}${info}<span class="mono">${esc(at.pl_number)}</span></div>
  </div>`;
}

function listHtml(items, max) {
  if (!items.length) return '<div class="empty">אין כרגע.</div>';
  const head = items.slice(0, max).join(''), rest = items.slice(max);
  return `<div class="list">${head}</div>` + (rest.length
    ? `<details class="more"><summary>עוד ${rest.length}</summary><div class="list" style="margin-top:8px">${rest.join('')}</div></details>` : '');
}

/* A deposited plan whose window has already shut, by what Xplan knows: a last day
 * in the past, or a newspaper notice more than two months (and two days) ago.
 * Local plans with neither stay listed; their window is invisible to Xplan, which
 * is exactly the gap watch_objections.py reads מבא"ת to close. */
function shut(at) {
  const last = at.pl_rejection_date || at.pl_last_deposit_date;
  if (last) return fromStamp(last) < today;
  if (at.pl_date_advertise) {
    const d = fromStamp(at.pl_date_advertise); d.setMonth(d.getMonth() + 2); d.setDate(d.getDate() + 2);
    return d < today;
  }
  return false;
}

const units = at => Number(at.quantity_delta_120) || 0;
const size = at => QUANT.reduce((s, [f]) => s + Math.abs(Number(at[f]) || 0), 0);

async function loadComing() {
  try {
    const plans = await xplan(
      "internet_short_status IN ('החלטה בדיון','בתהליך הפקדה','פרסום הפקדה')",
      [...BASE, 'depositing_date', 'pl_rejection_date', 'pl_last_deposit_date', 'pl_date_advertise']);
    const openNums = new Set(openPlans.map(p => p.num));
    const yearAgo = today - 365 * DAY;
    // In deposit but not open, with a deposit decision in the last year: on its way to a window.
    // Older ones are stuck at "deposit" with nobody moving them on; they are not "coming".
    const coming = plans.filter(at => !openNums.has((at.pl_number || '').trim()))
      .filter(at => at.internet_short_status === 'החלטה בדיון'
        || (at.depositing_date && at.depositing_date >= yearAgo && !shut(at)))
      .sort((a, b) => units(b) - units(a) || size(b) - size(a));
    $('coming-count').textContent = coming.length ? `${coming.length} תוכניות` : '';
    $('coming-list').innerHTML = listHtml(coming.map(at => {
      const stage = at.internet_short_status === 'החלטה בדיון'
        ? '<span class="tag t-blue">בדיון בוועדה</span>' : '<span class="tag t-soon">הוחלט להפקיד</span>';
      const when = at.depositing_date ? `<span>החלטה על הפקדה: ${fmt(fromStamp(at.depositing_date), { day: 'numeric', month: 'numeric', year: 'numeric' })}</span>` : '';
      const who = `<span>${at.pl_by_auth_of === LOCAL ? 'ועדה מקומית' : 'ועדה מחוזית'}</span>`;
      return row(at, stage, when + who);
    }), 8);
  } catch (err) {
    $('coming-list').innerHTML = `<div class="empty err">מינהל התכנון לא ענה (${esc(err.message)}). נסו לרענן בעוד כמה דקות.</div>`;
  }
}

async function loadApproved() {
  try {
    const since = new Date(today - 365 * DAY);
    const stamp = `${since.getFullYear()}-${String(since.getMonth() + 1).padStart(2, '0')}-${String(since.getDate()).padStart(2, '0')}`;
    const plans = await xplan(
      `internet_short_status IN ('פרסום אישור','התכנית אושרה') AND pl_date_8 >= DATE '${stamp}'`,
      [...BASE, 'pl_date_8'], 'pl_date_8 DESC');
    $('approved-count').textContent = plans.length ? `${plans.length} תוכניות` : '';
    $('approved-list').innerHTML = listHtml(plans.map(at => {
      const d = at.pl_date_8 ? fmt(fromStamp(at.pl_date_8), { day: 'numeric', month: 'numeric', year: 'numeric' }) : '';
      return row(at, `<span class="tag t-ok">ברשומות ${d}</span>`, `<span>${at.pl_by_auth_of === LOCAL ? 'ועדה מקומית' : 'ועדה מחוזית'}</span>`);
    }), 8);
  } catch (err) {
    $('approved-list').innerHTML = `<div class="empty err">מינהל התכנון לא ענה (${esc(err.message)}). נסו לרענן בעוד כמה דקות.</div>`;
  }
}

loadOpen().then(loadComing);
loadApproved();
