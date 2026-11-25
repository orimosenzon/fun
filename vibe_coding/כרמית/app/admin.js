// Ori and Carmit manage quizzes (חידונים) here: titled, ordered selections from one shared pool of riddles.
// Images are shrunk in the browser and kept inside the Firestore documents as data URLs, so no Storage
// bucket (and no billing) is needed. Routing is by hash: '' = all quizzes, '#pool', '#quiz/<id>', '#bundle/<id>'.
import { app, db, fs, ADMINS, LEVELS, GENERAL, loadRiddles, loadQuizzes, loadBundles, loadLetters, loadCategories, loadRatings, loadComments, loadVisits, loadGever, sendGever, markOwnDevice, dayKey, primaryAnswer, DEFAULT_COVER, levelOf, levelLabel,
  hasHeading, fillHeading, safeUrl } from './fb.js?v=202611251657';

const au = await import('https://www.gstatic.com/firebasejs/10.14.1/firebase-auth.js');
const auth = au.getAuth(app);
auth.languageCode = 'he';

const $ = (id) => document.getElementById(id);
const MAX_SIDE = 1280;
const MAX_B64 = 380_000; // two images must fit together in one 1 MiB document

let riddles = [];        // the pool, newest first
let byId = new Map();
let quizzes = [];
let letters = [];        // letters to send along with a quiz, before it or after it (any quiz)
let categories = [];     // the categories of riddles, in the order they were made; "כללי" is built in, not one of them
let poolCat = null;      // the pool's filter: null = every riddle, '' = "כללי", or a category's id
let bundles = [];        // bundles (מקבצים): collections of quizzes, each with a link of its own
let gever = [];          // requests to Claude in the "גבר גבר" tab, each with its answer once there is one
let current = null;      // the quiz open in the quiz view
let currentBundle = null;  // the bundle open in the bundle view
let picked = new Set();  // riddle ids checked in the quiz list, to copy or move to another quiz
let lastPick = null;     // anchor for shift-click ranges

let editing = null;      // id of the riddle being edited in the dialog, or null for a new one
let addToQuiz = null;    // a new riddle saved from the quiz view joins this quiz
let image = '';          // data URLs of the dialog form
let answerImage = '';

const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
};
const plural = (n, one, many) => (n === 1 ? one : `${n} ${many}`);
const riddleLabel = (r) => (r.answerType === 'image' ? 'תשובה בתמונה' : primaryAnswer(r));
const quizRiddles = (q) => q.riddleIds.map((id) => byId.get(id)).filter(Boolean);
const quizzesWith = (rid) => quizzes.filter((q) => q.riddleIds.includes(rid));
// A riddle's category id, or '' for "כללי": no category, or one that was deleted since.
const catId = (r) => (categories.some((c) => c.id === r.category) ? r.category : '');
const catName = (id) => categories.find((c) => c.id === id)?.name || GENERAL;
const inCat = (id) => riddles.filter((r) => catId(r) === id);
// A locked quiz (quiz.locked, "קבוע") is one that was sent and must stay as it is: nothing in it can change, and
// neither can the riddles in it, wherever they are edited from. These are its fields that the player sees.
const LOCKED_FIELDS = ['title', 'cover', 'riddleIds', 'shuffle', 'level', 'heading', 'opening', 'closing'];
// A letter marked in a locked quiz is locked with it: it can't be edited or deleted, only cloned or archived.
const letterLockedIn = (l) => quizzes.filter((q) => q.locked && (q.opening === l.id || q.closing === l.id));
const lockedWith = (rid) => quizzesWith(rid).filter((q) => q.locked);
const quizNames = (qs) => qs.map((q) => `"${q.title || 'ללא שם'}"`).join(', ');
// The link to send: the quiz on its own. The player shows the letters marked in the quiz at the moment it is opened.
function playUrl(q) {
  return new URL(`index.html?q=${q.id}`, location.href).href;
}

function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => t.classList.remove('show'), 2200);
}

// ---------- sign-in ----------

$('login-btn').addEventListener('click', async () => {
  try {
    await au.signInWithPopup(auth, new au.GoogleAuthProvider());
  } catch (e) {
    if (e.code !== 'auth/popup-closed-by-user') $('login-msg').textContent = `הכניסה נכשלה: ${e.message}`;
  }
});
$('logout-btn').addEventListener('click', () => au.signOut(auth));

au.onAuthStateChanged(auth, async (user) => {
  const ok = user && ADMINS.includes((user.email || '').toLowerCase());
  $('login-screen').classList.toggle('hidden', !!ok);
  $('who-box').classList.toggle('hidden', !ok);
  if (!ok) document.querySelectorAll('.view').forEach((v) => v.classList.add('hidden'));
  if (user && !ok) {
    $('login-msg').textContent = `החשבון ${user.email} לא מורשה לנהל חידות. רק אורי וכרמית יכולים.`;
    $('login-btn').textContent = 'כניסה עם חשבון אחר';
    au.signOut(auth);
  }
  if (ok) {
    $('who').textContent = user.email;
    markOwnDevice();   // from now on visits from this browser count as ours, not as a player's
    await reload();
    route();
  }
});

async function reload() {
  [riddles, quizzes, letters, bundles, categories, gever] = await Promise.all([loadRiddles(), loadQuizzes(),
    loadLetters().catch((e) => { console.error(e); return []; }),
    loadBundles().catch((e) => { console.error(e); return []; }),
    loadCategories().catch((e) => { console.error(e); return []; }),
    loadGever().catch((e) => { console.error(e); return []; })]);
  byId = new Map(riddles.map((r) => [r.id, r]));
}

// ---------- routing ----------

window.addEventListener('hashchange', route);
// Every view opens at its top, also after a reload (the browser would restore the old position).
if ('scrollRestoration' in history) history.scrollRestoration = 'manual';

function route() {
  if (!auth.currentUser) return;
  hidePeek();
  const [, quizId] = location.hash.match(/^#quiz\/(.+)$/) || [];
  const quiz = quizId && quizzes.find((q) => q.id === quizId);
  const [, bundleId] = location.hash.match(/^#bundle\/(.+)$/) || [];
  const bundle = bundleId && bundles.find((b) => b.id === bundleId);
  $('home-view').classList.toggle('hidden', !!(quiz || bundle));
  $('quiz-view').classList.toggle('hidden', !quiz);
  $('bundle-view').classList.toggle('hidden', !bundle);
  scrollTo({ top: 0 });
  currentBundle = bundle || null;
  if (quiz) return openQuiz(quiz);
  current = null;
  if (bundle) return openBundle(bundle);
  const tab = { '#pool': 'pool', '#bundles': 'bundles', '#path': 'path', '#letters': 'letters', '#stats': 'stats', '#gever': 'gever' }[location.hash] || 'quizzes';
  setTab(tab);
  renderHome();
  if (tab === 'stats') loadStats();
  if (tab === 'gever') renderGever();
  geverBadge();
}

function setTab(tab) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === tab));
  $('tab-quizzes').classList.toggle('hidden', tab !== 'quizzes');
  $('tab-pool').classList.toggle('hidden', tab !== 'pool');
  $('tab-bundles').classList.toggle('hidden', tab !== 'bundles');
  $('tab-path').classList.toggle('hidden', tab !== 'path');
  $('tab-letters').classList.toggle('hidden', tab !== 'letters');
  $('tab-stats').classList.toggle('hidden', tab !== 'stats');
  $('tab-gever').classList.toggle('hidden', tab !== 'gever');
}
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => {
  history.replaceState(null, '', t.dataset.tab === 'quizzes' ? location.pathname : `#${t.dataset.tab}`);
  route();
}));

// ---------- home: the quiz gallery ----------

function renderHome() {
  $('quiz-count').textContent = quizzes.length;
  $('pool-count').textContent = riddles.length;
  renderQuizGrid();
  renderBundles();
  renderPath();
  renderPool();
  renderAllLetters();
}

// ---------- statistics: how hard players found each riddle ----------

let ratings = null;      // every rating ever given; loaded when the tab is first opened
let visits = null;       // one per browser per day per page (the start screen or a quiz)
let comments = [];       // what players wrote about riddles, beside the rating

async function loadStats(force) {
  const body = $('stats-body');
  if (!ratings || force) {
    body.replaceChildren(el('p', 'help', 'טוענים…'));
    $('visits-body').replaceChildren();
    try {
      [ratings, visits, comments] = await Promise.all([loadRatings(), loadVisits(),
        loadComments().catch((e) => { console.error(e); return []; })]);
    } catch (e) {
      console.error(e);
      body.replaceChildren(el('p', 'help warn', 'לא הצלחנו לטעון את הנתונים. נסו לרענן.'));
      return;
    }
  }
  renderVisits();
  renderComments();
  renderStats();
}
$('stats-refresh').addEventListener('click', () => loadStats(true));

// { easy, medium, hard, n, correct } over a list of ratings
function tally(list) {
  const t = { easy: 0, medium: 0, hard: 0, n: 0, correct: 0 };
  for (const r of list) {
    if (!(r.level in t)) continue;
    t[r.level]++;
    t.n++;
    if (r.correct) t.correct++;
  }
  return t;
}

// What the players think on the whole: the average of easy = 1, medium = 2, hard = 3, cut into thirds.
// With fewer than three ratings one person decides it, so no verdict yet.
const MIN_VERDICT = 3;
function verdict(t) {
  if (t.n < MIN_VERDICT) return null;
  const avg = (t.easy + 2 * t.medium + 3 * t.hard) / t.n;
  return LEVELS[avg < 5 / 3 ? 0 : avg <= 7 / 3 ? 1 : 2];
}

const pct = (part, n) => `${Math.round((100 * part) / n)}%`;

// One horizontal bar split into easy / medium / hard, each part as wide as its share.
function levelBar(t) {
  const bar = el('div', 'lv-bar');
  bar.setAttribute('role', 'img');
  bar.setAttribute('aria-label', LEVELS.map((L) => `${L.name}: ${t[L.id]}`).join(', '));
  for (const L of LEVELS) {
    if (!t[L.id]) continue;
    const seg = el('span', `lv-seg lv-${L.id}`, t[L.id] / t.n >= 0.1 ? String(t[L.id]) : '');
    seg.style.flexGrow = t[L.id];
    seg.title = `${L.name}: ${t[L.id]} מתוך ${t.n} (${pct(t[L.id], t.n)})`;
    bar.appendChild(seg);
  }
  return bar;
}

const countsText = (t) => LEVELS.map((L) => `${L.name} ${t[L.id]}`).join(' · ');

function statTile(value, label) {
  const tile = el('div', 'stat-tile');
  tile.append(el('b', '', value), el('span', '', label));
  return tile;
}

function riddleStatRow(r, t) {
  const li = el('li', 'stat-riddle');
  const img = el('img', 'thumb');
  img.src = r.image;
  img.alt = '';
  img.loading = 'lazy';
  const main = el('div', 'stat-main');
  const head = el('div', 'stat-head');
  head.append(el('b', '', riddleLabel(r)), el('span', 'help', `${plural(t.n, 'דירוג אחד', 'דירוגים')} · ${pct(t.correct, t.n)} פתרו`));
  main.append(head, levelBar(t), el('span', 'help stat-counts', countsText(t)));
  li.append(img, main);
  return li;
}

function quizStatCard(q, list) {
  const t = tally(list);
  const card = el('section', 'pane stat-quiz');
  const head = el('div', 'pane-head');
  const title = el('h2', '', q.title || 'חידון ללא שם');
  const meta = el('span', 'help', plural(t.n, 'דירוג אחד', 'דירוגים'));
  head.append(title, meta);
  card.append(head, levelBar(t), el('span', 'help stat-counts', countsText(t)));

  const v = verdict(t);
  const line = el('p', 'stat-verdict');
  const declared = levelLabel(q);
  if (!v) {
    line.textContent = `עוד מעט מדי דירוגים כדי לומר מה השחקנים חושבים (צריך לפחות ${MIN_VERDICT}).`;
    line.classList.add('help');
  } else {
    line.append('לפי השחקנים: ', el('b', '', `רמה ${v.name}`), declared ? ` · הוגדר: ${declared}` : ' · הוגדר: בלי רמה');
    if (declared && v.id !== levelOf(q)) line.append(el('span', 'stat-gap', ' ← שונה ממה שהוגדר'));
  }
  card.appendChild(line);

  // The riddles in the quiz's own order; a riddle since taken out of the quiz still shows its ratings, at the end.
  const byRiddle = Map.groupBy(list, (r) => r.riddle);
  const ids = [...q.riddleIds.filter((id) => byRiddle.has(id)), ...[...byRiddle.keys()].filter((id) => !q.riddleIds.includes(id))];
  const rows = ids.filter((id) => byId.has(id)).map((id) => riddleStatRow(byId.get(id), tally(byRiddle.get(id))));
  if (rows.length) {
    const details = el('details', 'stat-details');
    details.appendChild(el('summary', '', `לפי חידה (${rows.length})`));
    const ul = el('ul', 'stat-riddles');
    ul.append(...rows);
    details.appendChild(ul);
    card.appendChild(details);
  }
  return card;
}

const DAYS_SHOWN = 30;
const shortDay = (key) => `${Number(key.slice(8))}/${Number(key.slice(5, 7))}`;
const uniq = (list) => new Set(list.map((v) => v.visitor)).size;

// A bar for each of the last 30 days: how many different real visitors came that day.
function daysChart(real) {
  const perDay = Map.groupBy(real, (v) => v.day);
  const days = [];
  for (let i = DAYS_SHOWN - 1; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    const key = dayKey(d);
    days.push({ key, n: uniq(perDay.get(key) || []) });
  }
  const max = Math.max(1, ...days.map((d) => d.n));
  const fig = el('figure', 'days-chart');
  fig.appendChild(el('figcaption', 'help', `מבקרים בכל יום, ${DAYS_SHOWN} הימים האחרונים (היום מימין). היום העמוס ביותר: ${max}.`));
  const plot = el('div', 'days-plot');
  plot.dir = 'ltr';
  days.forEach((d, i) => {
    const col = el('div', 'day-col');
    col.title = `${shortDay(d.key)}: ${plural(d.n, 'מבקר אחד', 'מבקרים')}`;
    const bar = el('i', 'day-bar');
    bar.style.height = `${(100 * d.n) / max}%`;
    if (!d.n) bar.classList.add('zero');
    // A date under every seventh bar, counted back from today.
    const label = el('span', 'day-label', (DAYS_SHOWN - 1 - i) % 7 === 0 ? shortDay(d.key) : '');
    const track = el('div', 'day-track');
    if (d.n === max && d.n > 0) track.appendChild(el('b', 'day-top', String(d.n)));
    track.appendChild(bar);
    col.append(track, label);
    plot.appendChild(col);
  });
  fig.appendChild(plot);
  return fig;
}

function renderVisits() {
  const body = $('visits-body');
  // A browser that was ever marked as ours is ours on all its days, also those before it first signed in.
  const ours = new Set(visits.filter((v) => v.own).map((v) => v.visitor));
  const real = visits.filter((v) => !ours.has(v.visitor));
  const own = visits.filter((v) => ours.has(v.visitor));
  if (!visits.length) {
    body.replaceChildren(el('p', 'help', 'עוד לא נרשמו כניסות.'));
    return;
  }
  const today = dayKey();
  const weekAgo = new Date();
  weekAgo.setDate(weekAgo.getDate() - 6);
  const weekKey = dayKey(weekAgo);
  const visitDays = (list) => new Set(list.map((v) => `${v.visitor} ${v.day}`)).size;
  const tiles = el('div', 'stat-tiles');
  tiles.append(
    statTile(String(uniq(real)), 'מבקרים אמיתיים בסך הכול'),
    statTile(String(uniq(real.filter((v) => v.day >= weekKey))), 'ב-7 הימים האחרונים'),
    statTile(String(uniq(real.filter((v) => v.day === today))), 'היום'),
    statTile(String(visitDays(real)), 'ביקורים (מבקר ביום)'),
  );
  const ownLine = el('p', 'help', own.length
    ? `בנפרד, הכניסות שלכם (כרמית ואורי): ${plural(visitDays(own), 'ביקור אחד', 'ביקורים')} מ-${plural(ours.size, 'דפדפן אחד', 'דפדפנים')}. הן לא נכללות באף מספר אחר כאן.`
    : 'עוד לא נרשמו כניסות מהמכשירים שלכם.');
  const out = [tiles, ownLine];
  if (real.length) {
    const chart = el('section', 'pane');
    chart.appendChild(daysChart(real));
    out.push(chart);
    // Per quiz: how many real visitors opened it, pressed start, and reached the results.
    const perQuiz = Map.groupBy(real.filter((v) => v.quiz), (v) => v.quiz);
    const rows = quizzes.filter((q) => perQuiz.has(q.id))
      .map((q) => {
        const list = perQuiz.get(q.id);
        return { q, opened: uniq(list), started: uniq(list.filter((v) => v.started)), finished: uniq(list.filter((v) => v.finished)),
          solo: uniq(list.filter((v) => v.solo)) };
      })
      .sort((a, b) => b.opened - a.opened);
    // A bundle's own screen (its list of quizzes) is a page too: how many opened it. The quizzes in it are counted above.
    const bundleRows = bundles.filter((b) => perQuiz.has(`s:${b.id}`))
      .map((b) => ({ b, opened: uniq(perQuiz.get(`s:${b.id}`)) })).sort((x, y) => y.opened - x.opened);
    if (rows.length || bundleRows.length) {
      const pane = el('section', 'pane');
      pane.appendChild(el('h2', 'stat-h', 'לפי חידון'));
      const wrap = el('div', 'table-wrap');
      const table = el('table', 'visits-table');
      const head = el('tr');
      ['חידון', 'פתחו', 'התחילו', 'סיימו', 'הגיעו מקישור שנשלח'].forEach((h) => head.appendChild(el('th', '', h)));
      table.appendChild(head);
      for (const r of rows) {
        const tr = el('tr');
        tr.append(el('td', '', r.q.title || 'חידון ללא שם'), el('td', 'num', String(r.opened)), el('td', 'num', String(r.started)),
          el('td', 'num', String(r.finished)), el('td', 'num', String(r.solo)));
        table.appendChild(tr);
      }
      for (const r of bundleRows) {
        const tr = el('tr');
        tr.append(el('td', '', `מקבץ: ${r.b.title || 'ללא שם'}`), el('td', 'num', String(r.opened)), el('td', 'num', '–'), el('td', 'num', '–'), el('td', 'num', '–'));
        table.appendChild(tr);
      }
      wrap.appendChild(table);
      pane.append(wrap, el('p', 'help', `דף הפתיחה (רשימת החידונים): ${plural(uniq(real.filter((v) => !v.quiz)), 'מבקר אחד', 'מבקרים')}.`));
      out.push(pane);
    }
  } else {
    out.push(el('p', 'help', 'עוד לא נרשמו מבקרים אמיתיים.'));
  }
  body.replaceChildren(...out);
}

// What players wrote, newest first: the words, and around them which riddle, in which quiz, when, and how it went.
const stamp = (c) => (c && c.toMillis ? c.toMillis() : +c || 0);
function renderComments() {
  const body = $('comments-body');
  $('comments-count').textContent = comments.length || '';
  if (!comments.length) {
    body.replaceChildren(el('p', 'help', 'עוד לא כתבו כלום. מה ששחקנים יכתבו יופיע כאן.'));
    return;
  }
  const list = el('ul', 'comment-list');
  for (const c of [...comments].sort((a, b) => stamp(b.created) - stamp(a.created))) {
    const r = byId.get(c.riddle);
    const q = quizzes.find((x) => x.id === c.quiz);
    const li = el('li', 'pane comment');
    if (r) {
      const img = el('img', 'thumb');
      img.src = r.image;
      img.alt = '';
      img.loading = 'lazy';
      li.appendChild(img);
    }
    const main = el('div', 'stat-main');
    const head = el('div', 'stat-head');
    const when = stamp(c.created) ? new Date(stamp(c.created)).toLocaleString('he-IL', { day: 'numeric', month: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' }) : '';
    head.append(el('b', '', r ? riddleLabel(r) : 'חידה שנמחקה'), el('span', 'help', [q ? q.title || 'חידון ללא שם' : 'חידון שנמחק', when].filter(Boolean).join(' · ')));
    const L = LEVELS.find((x) => x.id === c.level);
    const facts = [c.correct ? 'פתר' : 'לא פתר', c.hint && 'נעזר ברמז', L ? `סימן "${L.name}"` : 'לא סימן רמת קושי'].filter(Boolean).join(' · ');
    const foot = el('div', 'stat-head');
    const del = el('button', 'link-btn del', 'מחיקה');
    del.addEventListener('click', async () => {
      if (!confirm('למחוק את המשוב הזה? אי אפשר לבטל את זה.')) return;
      try {
        await fs.deleteDoc(fs.doc(db, 'comments', c.id));
      } catch (err) { return toast(`המחיקה נכשלה: ${err.message}`); }
      comments = comments.filter((x) => x !== c);
      renderComments();
      toast('המשוב נמחק');
    });
    foot.append(el('span', 'help', facts), del);
    main.append(head, el('p', 'comment-text', c.text), foot);
    li.appendChild(main);
    list.appendChild(li);
  }
  body.replaceChildren(list);
}

function renderStats() {
  const body = $('stats-body');
  // Ratings of a quiz that was deleted since have nowhere to show.
  const byQuiz = Map.groupBy(ratings.filter((r) => quizzes.some((q) => q.id === r.quiz)), (r) => r.quiz);
  const all = tally([...byQuiz.values()].flat());
  $('stats-count').textContent = all.n || '';
  if (!all.n) {
    body.replaceChildren(el('p', 'help', 'עוד אין דירוגים. הם יופיעו כאן אחרי ששחקנים יסמנו כמה קשה הייתה להם חידה.'));
    return;
  }
  const tiles = el('div', 'stat-tiles');
  tiles.append(
    statTile(String(all.n), 'דירוגים בסך הכול'),
    ...LEVELS.map((L) => statTile(pct(all[L.id], all.n), `סימנו "${L.name}"`)),
  );
  const total = el('section', 'pane');
  total.append(el('h2', 'stat-h', 'כל החידונים יחד'), levelBar(all), el('span', 'help stat-counts', countsText(all)));
  // Quizzes with the most ratings first: that's where there is the most to learn.
  const cards = quizzes.filter((q) => byQuiz.has(q.id))
    .sort((a, b) => byQuiz.get(b.id).length - byQuiz.get(a.id).length)
    .map((q) => quizStatCard(q, byQuiz.get(q.id)));
  const silent = quizzes.filter((q) => !byQuiz.has(q.id) && q.riddleIds.length);
  body.replaceChildren(tiles, total, ...cards);
  if (silent.length) body.appendChild(el('p', 'help', `עוד בלי דירוגים: ${silent.map((q) => q.title || 'חידון ללא שם').join(', ')}.`));
}

// ---------- the path: which quizzes players see, and in what order ----------

const onPath = () => quizzes.filter((q) => Number.isInteger(q.path)).sort((a, b) => a.path - b.path);

// Writes the new positions: only quizzes whose position changed are saved.
async function setPath(ids) {
  const writes = [];
  for (const q of quizzes) {
    const i = ids.indexOf(q.id);
    const pos = i < 0 ? null : i;
    if ((Number.isInteger(q.path) ? q.path : null) === pos) continue;
    q.path = pos;
    writes.push(fs.updateDoc(fs.doc(db, 'quizzes', q.id), { path: pos, updated: fs.serverTimestamp() }));
  }
  renderPath();
  try {
    await Promise.all(writes);
    toast('הסדר נשמר');
  } catch (err) {
    toast(`השמירה נכשלה: ${err.message}`);
  }
}

// A row for a quiz in an ordered list of quizzes (the path, a bundle), or beside it waiting to be added.
// i = its place in the list, or null for one that is not in it. to = { add(q), remove(q), addTitle, removeTitle }.
function quizRow(q, i, to) {
  const li = el('li', 'q-item path-item');
  li.dataset.id = q.id;
  const img = coverImg(q);
  const meta = el('div', 'meta');
  meta.appendChild(el('b', '', q.title || 'ללא שם'));
  const n = quizRiddles(q).length;
  meta.appendChild(el('div', 'sub', [levelLabel(q), n ? plural(n, 'חידה אחת', 'חידות') : 'אין חידות, לא יוצג'].filter(Boolean).join(' · ')));
  const open = el('a', 'link-btn', 'עריכה');
  open.href = `#quiz/${q.id}`;
  if (i === null) {
    li.title = to.addTitle;
    li.append(img, meta, open, el('span', 'add', '+'));
    li.addEventListener('click', (e) => { if (e.target !== open) to.add(q); });
  } else {
    const out = el('button', 'icon-btn', '✕');
    out.title = to.removeTitle;
    out.addEventListener('click', () => to.remove(q));
    li.append(el('span', 'grip', '⠿'), el('span', 'num', String(i + 1).padStart(2, '0')), img, meta, open, out);
  }
  li.draggable = true;
  li.addEventListener('dragstart', (e) => {
    e.dataTransfer.setData('text/plain', `quiz:${q.id}`);
    e.dataTransfer.effectAllowed = 'move';
    requestAnimationFrame(() => li.classList.add('dragging'));
  });
  li.addEventListener('dragend', () => li.classList.remove('dragging'));
  return li;
}

// Dropping a quiz on an ordered list inserts it at the pointer: reorders one already in it, or adds one from the
// side. ids() = the list's quiz ids now, set(ids) = save the new order.
function quizDrop(list, ids, set) {
  const marker = el('li', 'drop-marker');
  const at = (y) => {
    const items = [...list.querySelectorAll('.q-item:not(.dragging)')];
    const i = items.findIndex((it) => y < it.getBoundingClientRect().top + it.offsetHeight / 2);
    return { items, i: i < 0 ? items.length : i };
  };
  list.addEventListener('dragover', (e) => {
    e.preventDefault();
    const { items, i } = at(e.clientY);
    if (items[i]) list.insertBefore(marker, items[i]);
    else list.appendChild(marker);
  });
  list.addEventListener('dragleave', (e) => { if (!list.contains(e.relatedTarget)) marker.remove(); });
  list.addEventListener('drop', (e) => {
    e.preventDefault();
    marker.remove();
    const [, id] = e.dataTransfer.getData('text/plain').match(/^quiz:(.+)$/) || [];
    if (!id || !quizzes.some((q) => q.id === id)) return;
    const { items, i } = at(e.clientY);
    const before = items[i]?.dataset.id;
    const rest = ids().filter((x) => x !== id);
    rest.splice(before ? rest.indexOf(before) : rest.length, 0, id);
    set(rest);
  });
}

const PATH_ROW = {
  add: (q) => setPath([...onPath().map((x) => x.id), q.id]),
  remove: (q) => setPath(onPath().map((x) => x.id).filter((id) => id !== q.id)),
  addTitle: 'הוספה לסוף המסלול',
  removeTitle: 'הוצאה מהמסלול (החידון לא נמחק)',
};

function renderPath() {
  const path = onPath();
  const rest = quizzes.filter((q) => !path.includes(q));
  $('path-count').textContent = path.length;
  $('path-in-count').textContent = path.length;
  $('path-out-count').textContent = rest.length;
  $('path-fallback').classList.toggle('hidden', path.length > 0);
  const list = $('path-list');
  list.innerHTML = '';
  if (!path.length) list.appendChild(el('li', 'empty-drop', 'גוררים לכאן חידונים, או לוחצים עליהם'));
  path.forEach((q, i) => list.appendChild(quizRow(q, i, PATH_ROW)));
  const out = $('path-out');
  out.innerHTML = '';
  rest.forEach((q) => out.appendChild(quizRow(q, null, PATH_ROW)));
}
quizDrop($('path-list'), () => onPath().map((q) => q.id), setPath);

// ---------- bundles (מקבצים) ----------
// A bundle is a titled, ordered collection of quizzes that is sent as one link (index.html?s=<id>). Serial: the
// player gets them one after the other, each finished quiz opening the next, the rest dimmed and locked; not
// serial: all open. The quizzes stay where they are: a quiz can be in several bundles, and on the path too.

const bundleQuizzes = (b) => (b.quizIds || []).map((id) => quizzes.find((q) => q.id === id)).filter(Boolean);
const bundleUrl = (b) => new URL(`index.html?s=${b.id}`, location.href).href;
const bundleKind = (b) => (b.serial ? 'סדרתי' : 'כולם פתוחים');

function renderBundles() {
  $('bundle-count').textContent = bundles.length || '';
  const grid = $('bundle-grid');
  const add = el('button', 'bundle-card new-bundle');
  add.id = 'bundle-new';
  add.append(el('span', 'plus', '+'), el('span', '', 'מקבץ חדש'));
  add.addEventListener('click', createBundle);
  const cards = bundles.map((b) => {
    const qs = bundleQuizzes(b);
    const card = el('a', 'bundle-card');
    card.href = `#bundle/${b.id}`;
    // Its first quizzes' covers in a row, the way the player will meet them.
    const strip = el('div', 'bundle-strip');
    qs.slice(0, 4).forEach((q) => strip.appendChild(coverImg(q)));
    if (!qs.length) strip.appendChild(el('span', 'help', 'עוד אין חידונים'));
    if (qs.length > 4) strip.appendChild(el('span', 'more', `+${qs.length - 4}`));
    const cap = el('div', 'caption');
    cap.append(el('b', '', b.title || 'ללא שם'),
      el('span', '', [qs.length ? plural(qs.length, 'חידון אחד', 'חידונים') : null, `${b.serial ? '🔒 ' : ''}${bundleKind(b)}`].filter(Boolean).join(' · ')));
    card.append(strip, cap);
    return card;
  });
  grid.replaceChildren(add, ...cards);
}

async function createBundle() {
  const data = { title: 'מקבץ חדש', quizIds: [], serial: false, by: auth.currentUser.email };
  try {
    const ref = await fs.addDoc(fs.collection(db, 'bundles'), { ...data, created: fs.serverTimestamp() });
    bundles.unshift({ id: ref.id, ...data });
  } catch (err) { return toast(`יצירת המקבץ נכשלה: ${err.message}`); }
  location.hash = `#bundle/${bundles[0].id}`;
  setTimeout(() => { $('bundle-title').focus(); $('bundle-title').select(); }, 50);
}

function setBundleSave(st) {
  const box = $('bundle-save');
  box.textContent = { saving: 'שומרים…', saved: 'נשמר ✓', error: 'השמירה נכשלה' }[st] || '';
  box.className = `save-state ${st}`;
}

async function saveBundle(patch) {
  const b = currentBundle;
  Object.assign(b, patch);
  setBundleSave('saving');
  try {
    await fs.updateDoc(fs.doc(db, 'bundles', b.id), { ...patch, updated: fs.serverTimestamp() });
    if (b === currentBundle) setBundleSave('saved');
  } catch (err) {
    setBundleSave('error');
    toast(`השמירה נכשלה: ${err.message}`);
  }
}

function openBundle(b) {
  clearTimeout(bundleTimer);
  $('bundle-title').value = b.title;
  setBundleSave('');
  $('bundle-play').href = bundleUrl(b);
  renderBundle();
  scrollTo({ top: 0 });
}

const BUNDLE_ROW = {
  add: (q) => setBundleIds([...currentBundle.quizIds, q.id]),
  remove: (q) => setBundleIds(currentBundle.quizIds.filter((id) => id !== q.id)),
  addTitle: 'הוספה לסוף המקבץ',
  removeTitle: 'הוצאה מהמקבץ (החידון לא נמחק)',
};

function setBundleIds(ids) {
  saveBundle({ quizIds: ids });
  renderBundle();
}

function renderBundle() {
  const b = currentBundle;
  const inside = bundleQuizzes(b);
  const rest = quizzes.filter((q) => !inside.includes(q));
  document.querySelector(`input[name="bundle-serial"][value="${b.serial ? 'yes' : 'no'}"]`).checked = true;
  $('bundle-serial-help').textContent = b.serial
    ? 'השחקן מתחיל מהחידון הראשון. כל שאר החידונים מופיעים מעומעמים ונעולים, וכל חידון שהוא מסיים פותח את הבא. בכל ביקור חדש הוא מתחיל שוב מהראשון.'
    : 'השחקן רואה את כל החידונים פתוחים, ובוחר בעצמו מאיפה להתחיל ובאיזה סדר.';
  $('bundle-stats').textContent = inside.length ? `${plural(inside.length, 'חידון אחד', 'חידונים')} · ${bundleKind(b)}` : 'עוד אין חידונים';
  $('bundle-in-count').textContent = inside.length;
  $('bundle-out-count').textContent = rest.length;
  const list = $('bundle-in');
  list.innerHTML = '';
  if (!inside.length) list.appendChild(el('li', 'empty-drop', 'גוררים לכאן חידונים, או לוחצים עליהם'));
  inside.forEach((q, i) => list.appendChild(quizRow(q, i, BUNDLE_ROW)));
  $('bundle-out').replaceChildren(...rest.map((q) => quizRow(q, null, BUNDLE_ROW)));
}
quizDrop($('bundle-in'), () => bundleQuizzes(currentBundle).map((q) => q.id), setBundleIds);

let bundleTimer = null;
$('bundle-title').addEventListener('input', () => {
  clearTimeout(bundleTimer);
  setBundleSave('saving');
  bundleTimer = setTimeout(() => saveBundle({ title: $('bundle-title').value.trim() }), 600);
});
$('bundle-title').addEventListener('keydown', (e) => { if (e.key === 'Enter') e.target.blur(); });
document.querySelectorAll('input[name="bundle-serial"]').forEach((r) => r.addEventListener('change', () => {
  saveBundle({ serial: r.value === 'yes' });
  renderBundle();
}));

$('bundle-delete').addEventListener('click', async () => {
  const b = currentBundle;
  if (!confirm(`למחוק את המקבץ "${b.title}"?\nהחידונים שבו לא נמחקים. מי שקיבל קישור למקבץ כבר לא יוכל לפתוח אותו.`)) return;
  try {
    await fs.deleteDoc(fs.doc(db, 'bundles', b.id));
  } catch (err) { return toast(`המחיקה נכשלה: ${err.message}`); }
  bundles = bundles.filter((x) => x !== b);
  toast('המקבץ נמחק');
  location.hash = '#bundles';
});

$('bundle-share').addEventListener('click', () => {
  const b = currentBundle;
  // Made inside the click itself, as for a quiz: Safari on a phone only lets a click copy.
  copyLink(shortLink({ s: b.id }).catch((err) => {
    console.error(err);
    toast('לא הצלחנו ליצור קישור קצר, הועתק הקישור הארוך.');
    return bundleUrl(b);
  }), 'הקישור למקבץ:');
});

function coverImg(q) {
  const img = el('img', q.cover ? 'cover' : 'cover default');
  img.src = q.cover || DEFAULT_COVER;
  img.alt = '';
  return img;
}

function renderQuizGrid() {
  const grid = $('quiz-grid');
  grid.innerHTML = '';

  const add = el('button', 'quiz-card new-quiz');
  add.innerHTML = '<span class="plus">+</span><span>חידון חדש</span>';
  add.addEventListener('click', createQuiz);
  grid.appendChild(add);

  for (const q of quizzes) {
    const rs = quizRiddles(q);
    const card = el('a', 'quiz-card');
    card.href = `#quiz/${q.id}`;
    // A few of its riddles peek out from behind the cover like tilted pages.
    const stack = el('div', 'stack');
    rs.slice(0, 3).forEach((r, i) => {
      const leaf = el('img', `leaf leaf-${i}`);
      leaf.src = r.image;
      leaf.alt = '';
      stack.appendChild(leaf);
    });
    const front = el('div', 'front');
    front.appendChild(coverImg(q));
    stack.appendChild(front);
    card.appendChild(stack);
    const cap = el('div', 'caption');
    const count = rs.length ? plural(rs.length, 'חידה אחת', 'חידות') : 'עוד אין חידות';
    cap.append(el('b', '', `${q.locked ? '🔒 ' : ''}${q.title || 'ללא שם'}`), el('span', '', [levelLabel(q), count, q.shuffle && 'בסדר אקראי'].filter(Boolean).join(' · ')));
    card.appendChild(cap);
    card.addEventListener('mouseenter', () => schedulePeek(card, q));
    card.addEventListener('mouseleave', hidePeek);
    grid.appendChild(card);
  }
}

async function createQuiz() {
  const data = { title: 'חידון חדש', cover: '', riddleIds: [], level: 'easy', by: auth.currentUser.email };
  const ref = await fs.addDoc(fs.collection(db, 'quizzes'), { ...data, created: fs.serverTimestamp() });
  quizzes.unshift({ id: ref.id, ...data });
  location.hash = `#quiz/${ref.id}`;
  setTimeout(() => { $('quiz-title').focus(); $('quiz-title').select(); }, 50);
}

// ---------- hover peek ----------

const canHover = matchMedia('(hover: hover)').matches;

function schedulePeek(card, q) {
  if (!canHover) return;
  clearTimeout(schedulePeek.timer);
  schedulePeek.timer = setTimeout(() => showPeek(card, q), 280);
}

function showPeek(card, q) {
  const rs = quizRiddles(q);
  const peek = $('peek');
  peek.innerHTML = '';
  peek.appendChild(el('div', 'peek-title', q.title || 'ללא שם'));
  if (!rs.length) {
    peek.appendChild(el('p', 'peek-empty', 'החידון ריק. לחיצה פותחת אותו כדי להוסיף חידות.'));
  } else {
    const grid = el('div', 'peek-grid');
    rs.slice(0, 12).forEach((r, i) => {
      const cell = el('div', 'peek-cell');
      const img = el('img');
      img.src = r.image;
      img.alt = '';
      cell.append(img, el('span', 'peek-ans', `${i + 1}. ${riddleLabel(r)}`));
      grid.appendChild(cell);
    });
    peek.appendChild(grid);
    if (rs.length > 12) peek.appendChild(el('p', 'peek-more', `ועוד ${rs.length - 12}`));
  }
  // Beside the card, on whichever side has room.
  const box = card.getBoundingClientRect();
  peek.classList.add('show');
  const w = peek.offsetWidth;
  const h = peek.offsetHeight;
  let x = box.left - w - 14;
  if (x < 8) x = box.right + 14;
  if (x + w > innerWidth - 8) x = Math.max(8, innerWidth - w - 8);
  const y = Math.min(Math.max(8, box.top), innerHeight - h - 8);
  peek.style.left = `${x}px`;
  peek.style.top = `${Math.max(8, y)}px`;
}

function hidePeek() {
  clearTimeout(schedulePeek.timer);
  $('peek').classList.remove('show');
}
addEventListener('scroll', hidePeek, { passive: true });

// ---------- pool tab ----------

const matches = (r, term) => !term || [r.question, r.hint, ...(r.answers || [])]
  .some((s) => (s || '').toLowerCase().includes(term));

// The row of categories above the pool: each one filters the pool to its riddles, and says how many it has.
function renderCatBar() {
  if (poolCat && !categories.some((c) => c.id === poolCat)) poolCat = null;
  const chip = (value, name, n, cls = '') => {
    const b = el('button', `cat-chip ${cls}`);
    b.type = 'button';
    b.append(name, el('span', 'count', String(n)));
    b.setAttribute('aria-pressed', String(poolCat === value));
    b.addEventListener('click', () => { poolCat = value; renderPool(); });
    return b;
  };
  $('cat-chips').replaceChildren(
    chip(null, 'הכול', riddles.length, 'all'),
    ...categories.map((c) => chip(c.id, c.name, inCat(c.id).length)),
    chip('', GENERAL, inCat('').length, 'general'),
  );
}

// A category picker: Carmit's categories, "כללי", and a way to make a new one on the spot.
const NEW_CAT = '+new';
function fillCatSelect(sel, value) {
  sel.replaceChildren(...categories.map((c) => new Option(c.name, c.id)), new Option(GENERAL, ''), new Option('+ קטגוריה חדשה…', NEW_CAT));
  sel.value = categories.some((c) => c.id === value) ? value : '';
  sel.dataset.value = sel.value;
  sel.classList.toggle('general', !sel.value);
}
// apply(id) gets the chosen category; for "new" the categories window opens, and the one made there is chosen.
function onCatSelect(sel, apply) {
  sel.addEventListener('change', () => {
    if (sel.value !== NEW_CAT) return apply(sel.value);
    sel.value = sel.dataset.value;
    openCats((c) => apply(c.id));
  });
}

// The category is only for Carmit, the player never sees it: so it can change also on a riddle of a locked quiz.
async function setCategory(r, id) {
  if (catId(r) === id) return;
  try {
    await fs.updateDoc(fs.doc(db, 'riddles', r.id), { category: id, updated: fs.serverTimestamp() });
  } catch (err) {
    renderPool();
    return toast(`השמירה נכשלה: ${err.message}`);
  }
  r.category = id;
  toast(`החידה עברה לקטגוריה "${catName(id)}"`);
  renderPool();
}

function renderPool() {
  renderCatBar();
  const term = $('pool-search').value.trim().toLowerCase();
  const grid = $('pool-grid');
  grid.innerHTML = '';
  for (const r of riddles.filter((x) => (poolCat === null || catId(x) === poolCat) && matches(x, term))) {
    const tile = el('div', 'pool-tile');
    const img = el('img');
    img.src = r.image;
    img.alt = '';
    const body = el('div', 'tile-body');
    body.appendChild(el('b', r.answerType === 'image' ? 'tag' : 'ans', riddleLabel(r)));
    if (r.hint) body.appendChild(el('div', 'sub', `רמז: ${r.hint}`));
    if (r.aiImage) body.appendChild(el('div', 'sub ai-mark', 'התמונה נוצרה בבינה מלאכותית'));
    const cat = el('select', 'cat-select');
    cat.title = 'הקטגוריה של החידה';
    cat.setAttribute('aria-label', 'קטגוריה');
    fillCatSelect(cat, catId(r));
    onCatSelect(cat, (id) => setCategory(r, id));
    body.appendChild(cat);
    const qs = quizzesWith(r.id);
    const chips = el('div', 'chips');
    if (qs.length) {
      qs.forEach((q) => {
        const chip = el('a', 'chip', q.title || 'ללא שם');
        chip.href = `#quiz/${q.id}`;
        chips.appendChild(chip);
      });
    } else {
      chips.appendChild(el('span', 'chip lonely', 'לא בשום חידון'));
    }
    body.appendChild(chips);
    const actions = el('div', 'tile-actions');
    const edit = el('button', 'link-btn', 'עריכה');
    edit.addEventListener('click', () => openRiddle(r));
    const clone = el('button', 'link-btn', 'שיבוט');
    clone.title = 'חידה חדשה במאגר, זהה לזו, שאפשר לשנות בלי לגעת במקור';
    clone.addEventListener('click', () => cloneRiddle(r));
    const del = el('button', 'link-btn del', 'מחיקה');
    del.addEventListener('click', () => deleteRiddle(r));
    actions.append(edit, clone, del);
    tile.append(img, body, actions);
    grid.appendChild(tile);
  }
  if (!grid.children.length) {
    grid.appendChild(el('p', 'help', term ? 'לא נמצאו חידות.' : poolCat === null ? 'המאגר ריק.'
      : `עוד אין חידות בקטגוריה "${catName(poolCat)}". אפשר לשייך אליה חידה מהכרטיס שלה, או ליצור חידה חדשה.`));
  }
}
$('pool-search').addEventListener('input', renderPool);
$('pool-new').addEventListener('click', () => openRiddle(null));

// ---------- categories ----------
// The window where Carmit adds, renames and deletes categories. Deleting one deletes no riddle: its riddles go
// back to "כללי".

const catDialog = $('cat-dialog');
let catAdded = null;     // opened from a picker's "new category": what to do with the category made here

function catError(msg) {
  $('cat-error').textContent = msg;
  $('cat-error').classList.toggle('hidden', !msg);
}

const sameName = (a, b) => a.replace(/\s+/g, ' ').trim() === b.replace(/\s+/g, ' ').trim();
// '' when the name is fine for a category (other than the one being renamed), or what is wrong with it.
function nameProblem(name, except = null) {
  if (!name) return 'צריך לכתוב שם לקטגוריה.';
  if (sameName(name, GENERAL)) return `"${GENERAL}" כבר קיימת: שם נמצאת כל חידה שלא שויכה לקטגוריה.`;
  if (categories.some((c) => c !== except && sameName(c.name, name))) return `כבר יש קטגוריה בשם "${name}".`;
  return '';
}

function openCats(onAdded = null) {
  catAdded = onAdded;
  $('cat-form').reset();
  catError('');
  renderCats();
  catDialog.showModal();
  if (onAdded) $('cat-name').focus();
}

function renderCats() {
  const count = (n) => (n ? plural(n, 'חידה אחת', 'חידות') : 'אין חידות');
  const rows = categories.map((c) => {
    const li = el('li', 'cat-row');
    const name = el('input', 'cat-rename');
    name.type = 'text';
    name.maxLength = 40;
    name.value = c.name;
    name.autocomplete = 'off';
    name.title = 'כותבים כאן כדי לשנות את השם';
    name.setAttribute('aria-label', `שם הקטגוריה "${c.name}"`);
    name.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); name.blur(); } });
    name.addEventListener('change', () => renameCategory(c, name));
    const del = el('button', 'link-btn del', 'מחיקה');
    del.type = 'button';
    del.addEventListener('click', () => deleteCategory(c));
    li.append(name, el('span', 'sub', count(inCat(c.id).length)), del);
    return li;
  });
  const general = el('li', 'cat-row general');
  general.append(el('b', '', GENERAL), el('span', 'sub', count(inCat('').length)), el('span', 'sub fixed', 'ברירת המחדל, תמיד כאן'));
  $('cat-list').replaceChildren(...rows, general);
}

// After any change to the categories: the window itself, the page behind it, and the picker in an open riddle form.
function refreshCats() {
  renderCats();
  current ? renderQuiz() : renderHome();
  fillCatSelect($('riddle-category'), $('riddle-category').dataset.value);
}

$('cat-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const name = $('cat-name').value.replace(/\s+/g, ' ').trim();
  catError(nameProblem(name));
  if (nameProblem(name)) return $('cat-name').focus();
  $('cat-add').disabled = true;
  try {
    const data = { name, by: auth.currentUser.email };
    const ref = await fs.addDoc(fs.collection(db, 'categories'), { ...data, created: fs.serverTimestamp() });
    const c = { id: ref.id, ...data, created: Date.now() };
    categories.push(c);
    $('cat-form').reset();
    refreshCats();
    if (catAdded) {
      // Made on the way to choosing a category for a riddle: choose it, and back to the riddle.
      catAdded(c);
      catDialog.close();
    } else {
      toast(`הקטגוריה "${name}" נוספה`);
      $('cat-name').focus();
    }
  } catch (err) {
    catError(`ההוספה נכשלה: ${err.message}`);
  } finally {
    $('cat-add').disabled = false;
  }
});

async function renameCategory(c, input) {
  const name = input.value.replace(/\s+/g, ' ').trim();
  if (name === c.name) { input.value = c.name; return; }
  const problem = nameProblem(name, c);
  catError(problem);
  if (problem) { input.value = c.name; return; }
  try {
    await fs.updateDoc(fs.doc(db, 'categories', c.id), { name, updated: fs.serverTimestamp() });
  } catch (err) {
    input.value = c.name;
    return catError(`שינוי השם נכשל: ${err.message}`);
  }
  c.name = name;
  refreshCats();
  toast(`השם שונה ל"${name}"`);
}

async function deleteCategory(c) {
  const rs = inCat(c.id);
  const what = rs.length
    ? `\n${plural(rs.length, 'החידה שבה תעבור', 'החידות שבה יעברו')} ל"${GENERAL}". שום חידה לא נמחקת.`
    : '\nאין בה חידות.';
  if (!confirm(`למחוק את הקטגוריה "${c.name}"?${what}`)) return;
  catError('');
  try {
    await fs.deleteDoc(fs.doc(db, 'categories', c.id));
  } catch (err) { return catError(`המחיקה נכשלה: ${err.message}`); }
  categories = categories.filter((x) => x !== c);
  // From here the riddles already count as "כללי" (their category no longer exists); this only tidies the documents.
  try {
    await Promise.all(rs.map((r) => {
      r.category = '';
      return fs.updateDoc(fs.doc(db, 'riddles', r.id), { category: '', updated: fs.serverTimestamp() });
    }));
  } catch (err) { console.error(err); }
  refreshCats();
  toast(rs.length ? `הקטגוריה נמחקה, ו${plural(rs.length, 'חידה אחת עברה', 'חידות עברו')} ל"${GENERAL}"` : 'הקטגוריה נמחקה');
}

$('cat-manage').addEventListener('click', () => openCats());
$('cat-close').addEventListener('click', () => catDialog.close());
$('cat-done').addEventListener('click', () => catDialog.close());
catDialog.addEventListener('click', (e) => { if (e.target === catDialog) catDialog.close(); });

// A copy of a riddle as a new riddle in the pool, opened for editing. In a quiz the copy goes right after the
// original, or takes its place (replace), e.g. when the original can't be edited because another quiz is locked.
async function cloneRiddle(r, quiz = null, replace = false) {
  const { id, created, updated, ...data } = r;
  data.by = auth.currentUser.email;
  let copy;
  try {
    const ref = await fs.addDoc(fs.collection(db, 'riddles'), { ...data, created: fs.serverTimestamp() });
    copy = { id: ref.id, ...data };
    riddles.unshift(copy);
    byId.set(copy.id, copy);
    if (quiz && quiz === current && !quiz.locked) {
      const ids = [...quiz.riddleIds];
      const at = ids.indexOf(r.id);
      if (at < 0) ids.push(copy.id);
      else ids.splice(replace ? at : at + 1, replace ? 1 : 0, copy.id);
      await saveQuiz({ riddleIds: ids });
    }
  } catch (err) {
    return toast(`השיבוט נכשל: ${err.message}`);
  }
  current ? renderQuiz() : renderHome();
  toast(replace ? 'נוצר עותק והוא מחליף את המקור בחידון הזה' : 'נוצר עותק של החידה');
  openRiddle(copy);
  $('form-title').textContent = 'עריכת העותק';
}

async function deleteRiddle(r) {
  const locked = lockedWith(r.id);
  if (locked.length) return alert(`אי אפשר למחוק את החידה: היא נמצאת בחידון קבוע (${quizNames(locked)}).`);
  const qs = quizzesWith(r.id);
  const where = qs.length ? `\nהיא תוסר גם מ: ${qs.map((q) => q.title).join(', ')}.` : '';
  if (!confirm(`למחוק את החידה מהמאגר? אי אפשר לבטל את זה.${where}`)) return;
  await fs.deleteDoc(fs.doc(db, 'riddles', r.id));
  await Promise.all(qs.map((q) => {
    q.riddleIds = q.riddleIds.filter((id) => id !== r.id);
    return fs.updateDoc(fs.doc(db, 'quizzes', q.id), { riddleIds: fs.arrayRemove(r.id), updated: fs.serverTimestamp() });
  }));
  riddles = riddles.filter((x) => x.id !== r.id);
  byId.delete(r.id);
  toast('החידה נמחקה');
  current ? renderQuiz() : renderHome();
}

// ---------- one quiz ----------

function openQuiz(q) {
  current = q;
  picked = new Set();
  lastPick = null;
  $('quiz-title').value = q.title;
  $('side-search').value = '';
  $('side-source').value = '';
  clearTimeout(headingTimer);
  $('heading-text').value = (q.heading || {}).text || '';
  setSaveState('');
  renderHeading();
  renderLetters();
  renderOrder();
  renderLevel();
  renderAsk();
  renderQuiz();
  renderLock();
  scrollTo({ top: 0 });
}

function renderQuiz() {
  const q = current;
  const cover = $('cover-img');
  cover.src = q.cover || DEFAULT_COVER;
  cover.classList.toggle('default', !q.cover);
  $('cover-remove').classList.toggle('hidden', !q.cover);
  const rs = quizRiddles(q);
  $('in-count').textContent = rs.length;
  $('quiz-stats').textContent = rs.length ? plural(rs.length, 'חידה אחת', 'חידות') : 'עוד אין חידות';
  renderQuizList(rs);
  renderSideSource();
  renderSidePool();
}

// A locked quiz shows everything and lets nothing be changed; sending, playing, cloning and the letters stay open.
function renderLock() {
  const locked = !!current.locked;
  $('quiz-view').classList.toggle('locked', locked);
  $('lock-banner').classList.toggle('hidden', !locked);
  $('quiz-lock').textContent = locked ? '🔓 ביטול הקיבוע' : '🔒 קיבוע החידון';
  $('quiz-lock').title = locked ? 'החידון יחזור להיות ניתן לשינוי' : 'נועל את החידון ואת החידות שבו מפני שינויים, למשל אחרי שנשלח';
  for (const id of ['quiz-title', 'heading-text', 'heading-img-btn', 'heading-img-remove', 'heading-reset', 'cover-btn',
    'cover-remove', 'quiz-delete', 'quiz-new-riddle', 'pick-move']) $(id).disabled = locked;
  document.querySelectorAll('input[name="quiz-order"], input[name="quiz-level"]').forEach((r) => { r.disabled = locked; });
  if (locked) $('heading-from').disabled = true;
}

$('quiz-lock').addEventListener('click', async () => {
  const lock = !current.locked;
  if (!lock && !confirm(`לבטל את הקיבוע של "${current.title}"?
אחרי זה אפשר יהיה לשנות אותו ואת החידות שבו, ומי שכבר קיבל אותו יראה את השינויים.`)) return;
  await saveQuiz({ locked: lock });
  renderHeading();
  renderQuiz();
  renderLock();
  toast(lock ? 'החידון קבוע: אי אפשר לשנות אותו עד שמבטלים את הקיבוע' : 'הקיבוע בוטל');
});

// A new quiz with the same riddles (the very same ones, not copies) and the same settings. The copy is not
// locked and not on the players' path.
$('quiz-clone').addEventListener('click', async () => {
  const q = current;
  const data = { title: `${q.title || 'חידון'} (עותק)`, cover: q.cover || '', riddleIds: [...q.riddleIds], level: levelOf(q),
    shuffle: !!q.shuffle, heading: q.heading || { text: '', image: '' }, opening: q.opening || '', closing: q.closing || '',
    askRating: q.askRating !== false, by: auth.currentUser.email };
  try {
    const ref = await fs.addDoc(fs.collection(db, 'quizzes'), { ...data, created: fs.serverTimestamp() });
    quizzes.unshift({ id: ref.id, ...data });
    location.hash = `#quiz/${ref.id}`;
    toast(`נוצר עותק של "${q.title}". עכשיו עורכים את העותק.`);
  } catch (err) { toast(`השיבוט נכשל: ${err.message}`); }
});

let saveTimer = null;
function setSaveState(s) {
  const st = $('save-state');
  st.textContent = { saving: 'שומרים…', saved: 'נשמר ✓', error: 'השמירה נכשלה' }[s] || '';
  st.className = `save-state ${s}`;
}

async function saveQuiz(patch) {
  const q = current;
  // The last line of defence: whatever the screen let through, a locked quiz does not change.
  if (q.locked && LOCKED_FIELDS.some((k) => k in patch)) return toast('החידון קבוע, אי אפשר לשנות אותו. אפשר לשבט אותו או לבטל את הקיבוע.');
  Object.assign(q, patch);
  setSaveState('saving');
  try {
    await fs.updateDoc(fs.doc(db, 'quizzes', q.id), { ...patch, updated: fs.serverTimestamp() });
    if (q === current) setSaveState('saved');
  } catch (err) {
    setSaveState('error');
    toast(`השמירה נכשלה: ${err.message}`);
  }
}

$('quiz-title').addEventListener('input', () => {
  clearTimeout(saveTimer);
  setSaveState('saving');
  saveTimer = setTimeout(() => saveQuiz({ title: $('quiz-title').value.trim() }), 600);
});
$('quiz-title').addEventListener('keydown', (e) => { if (e.key === 'Enter') e.target.blur(); });

// The player sees the riddles in the order set here, or reshuffled on every start.
function renderOrder() {
  const shuffle = !!current.shuffle;
  document.querySelector(`input[name="quiz-order"][value="${shuffle ? 'shuffle' : 'fixed'}"]`).checked = true;
  $('order-help').textContent = shuffle
    ? 'השחקן יקבל את החידות בסדר אקראי, חדש בכל משחק. הסדר כאן משמש רק לתצוגה המקדימה.'
    : 'גוררים כדי לשנות את הסדר. זה הסדר שבו השחקן יראה את החידות.';
}
document.querySelectorAll('input[name="quiz-order"]').forEach((r) => r.addEventListener('change', () => {
  saveQuiz({ shuffle: r.value === 'shuffle' });
  renderOrder();
}));

// The player climbs easy → medium → hard; a quiz from before levels existed counts as easy.
// 'none' is for a quiz that mixes levels: no label at all.
function renderLevel() {
  document.querySelector(`input[name="quiz-level"][value="${levelOf(current)}"]`).checked = true;
}
document.querySelectorAll('input[name="quiz-level"]').forEach((r) => r.addEventListener('change', () => {
  saveQuiz({ level: r.value });
}));

// quiz.askRating: whether the player is asked, after each riddle, how hard it was. On unless switched off.
function renderAsk() {
  document.querySelector(`input[name="quiz-ask"][value="${current.askRating === false ? 'no' : 'yes'}"]`).checked = true;
}
document.querySelectorAll('input[name="quiz-ask"]').forEach((r) => r.addEventListener('change', () => {
  saveQuiz({ askRating: r.value === 'yes' });
}));

$('cover-btn').addEventListener('click', () => $('cover-file').click());
$('cover-file').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    await saveQuiz({ cover: await readImage(file) });
    renderQuiz();
  } catch (err) { toast(err.message); }
});
$('cover-remove').addEventListener('click', async () => { await saveQuiz({ cover: '' }); renderQuiz(); });

// ---------- short links ----------
// What gets sent is a short link, orimosenzon.github.io/fun/carmit/?<code>. That page carries the preview
// (Wordplay & Wonder logo, "חידות כרמית כונסרי") and forwards to the quiz. links/<code> says where to: the quiz
// and its letters. The same quiz with the same letters always gets the same code back.
const SHORT_BASE = new URL('../../../carmit/', location.href).href;
const CODE_CHARS = 'abcdefghijkmnpqrstuvwxyz23456789';  // no l, o, 0, 1: nothing to misread

function randomCode() {
  return [...crypto.getRandomValues(new Uint8Array(6))].map((n) => CODE_CHARS[n % CODE_CHARS.length]).join('');
}

// One code per quiz: the link only names the quiz, and the letters come from the quiz itself when it is opened.
// (links/<code> still has b and a, empty in new links; in links made before 21/11/2026 they are no longer read.)
// A bundle gets a code of its own the same way: target = { q: quizId } or { s: bundleId } (then q is '').
async function shortLink(target) {
  const [field, id] = target.s ? ['s', target.s] : ['q', target.q];
  const found = await fs.getDocs(fs.query(fs.collection(db, 'links'), fs.where(field, '==', id)));
  if (found.docs.length) {
    // The newest one, so the same quiz keeps giving the link Carmit sent last.
    const at = (d) => { const c = d.data().created; return c ? (c.toMillis ? c.toMillis() : +c) : 0; };
    return SHORT_BASE + '?' + found.docs.map((d) => [at(d), d.id]).sort((x, y) => y[0] - x[0])[0][1];
  }
  for (let i = 0; i < 5; i++) {
    const code = randomCode();
    if ((await fs.getDoc(fs.doc(db, 'links', code))).exists()) continue;
    await fs.setDoc(fs.doc(db, 'links', code), { q: target.q || '', b: '', a: '', ...(target.s ? { s: target.s } : {}), key: `${field}:${id}`,
      by: auth.currentUser.email, created: fs.serverTimestamp() });
    return SHORT_BASE + '?' + code;
  }
  throw new Error('no free code');
}

$('share-btn').addEventListener('click', () => {
  const quiz = current;
  // Made inside the click itself: Safari on a phone only lets a click copy, not something that finishes later.
  const url = shortLink({ q: quiz.id }).catch((err) => {
    console.error(err);
    toast('לא הצלחנו ליצור קישור קצר, הועתק הקישור הארוך.');
    return playUrl(quiz);
  });
  copyLink(url);
});

// url may be a promise of the link (the short link is still being made).
async function copyLink(url, what = 'הקישור לחידון:') {
  try {
    if (typeof ClipboardItem === 'undefined') throw new Error('no ClipboardItem');
    const blob = Promise.resolve(url).then((u) => new Blob([u], { type: 'text/plain' }));
    await navigator.clipboard.write([new ClipboardItem({ 'text/plain': blob })]);
  } catch {
    const u = await url;
    try { await navigator.clipboard.writeText(u); } catch { return prompt(what, u); }
  }
  toast(`הקישור הועתק: ${await url}`);
}

// ---------- the heading above the quiz ----------

const headingKey = (h) => `${(h.text || '').trim()}|${h.image || ''}`;

function renderHeading() {
  const h = current.heading || {};
  $('heading-img').src = h.image || '';
  $('heading-img').classList.toggle('hidden', !h.image);
  $('heading-img-hint').classList.toggle('hidden', !!h.image);
  $('heading-img-remove').classList.toggle('hidden', !h.image);
  $('heading-reset').classList.toggle('hidden', !hasHeading(h));
  // The preview shows exactly what the player will see, the usual line when there is no heading.
  const preview = $('heading-preview');
  if (hasHeading(h)) fillHeading(preview, h);
  else preview.innerHTML = $('default-motif').innerHTML;
  // Headings of the other quizzes, each once, to take over as they are.
  const sel = $('heading-from');
  sel.innerHTML = '';
  sel.appendChild(new Option('לקחת כותרת מחידון אחר…', ''));
  const seen = new Map();
  for (const q of quizzes) {
    if (q === current || !hasHeading(q.heading) || headingKey(q.heading) === headingKey(h)) continue;
    const k = headingKey(q.heading);
    if (!seen.has(k)) seen.set(k, { heading: q.heading, titles: [] });
    seen.get(k).titles.push(q.title || 'ללא שם');
  }
  for (const [k, { heading, titles }] of seen) {
    const first = (heading.text || '').trim().split('\n')[0] || '(תמונה בלבד)';
    const label = `${heading.image ? '🖼 ' : ''}${first.length > 40 ? `${first.slice(0, 40)}…` : first}  ·  מתוך: ${titles.join(', ')}`;
    sel.appendChild(new Option(label, k));
  }
  if (seen.size) sel.appendChild(new Option('ההסבר הרגיל (עברית × English)', 'default'));
  sel.disabled = !seen.size || !!current.locked;
  if (!seen.size) sel.options[0].text = 'אין עוד כותרות בחידונים אחרים';
  sel.value = '';
}

function saveHeading(h) {
  saveQuiz({ heading: { text: (h.text || '').trim(), image: h.image || '' } });
  $('heading-text').value = current.heading.text;
  renderHeading();
}

let headingTimer = null;
$('heading-text').addEventListener('input', () => {
  clearTimeout(headingTimer);
  setSaveState('saving');
  // The preview follows every keystroke; the save waits for a pause.
  const h = { ...(current.heading || {}), text: $('heading-text').value };
  if (hasHeading(h)) fillHeading($('heading-preview'), h);
  else $('heading-preview').innerHTML = $('default-motif').innerHTML;
  headingTimer = setTimeout(async () => {
    await saveQuiz({ heading: { text: $('heading-text').value.trim(), image: (current.heading || {}).image || '' } });
    renderHeading();
  }, 600);
});
$('heading-from').addEventListener('change', (e) => {
  const k = e.target.value;
  if (!k) return;
  if (k === 'default') return saveHeading({ text: '', image: '' });
  const src = quizzes.find((q) => q !== current && hasHeading(q.heading) && headingKey(q.heading) === k);
  if (src) { saveHeading(src.heading); toast(`הכותרת של "${src.title}" הועתקה לחידון הזה`); }
});
$('heading-img-btn').addEventListener('click', () => $('heading-file').click());
$('heading-file').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;
  try {
    saveHeading({ text: $('heading-text').value, image: await readImage(file, 600, 120_000) });
  } catch (err) { toast(err.message); }
});
$('heading-img-remove').addEventListener('click', () => saveHeading({ text: $('heading-text').value, image: '' }));
$('heading-reset').addEventListener('click', () => {
  if (confirm('למחוק את הכותרת של החידון ולחזור להסבר הרגיל?')) saveHeading({ text: '', image: '' });
});

// ---------- letters ----------
// A letter stands on its own: text, a picture and a button, none of it tied to a quiz. In a quiz Carmit marks
// which letter goes before it (quiz.opening, רישא) and which after it (quiz.closing, סיפא), and the one share
// button copies the quiz's link. The player reads the choice from the quiz when the link is opened, so what Carmit
// sees here is what everyone with the link sees, also with a link sent earlier. (Until 21/11/2026 the link itself
// carried the letters of the moment it was copied; Carmit changed a letter and the people she had sent to didn't see it.)

function letterById(id) { return letters.find((l) => l.id === id); }
function letterGist(l) {
  const t = (l.text || '').replace(/\s+/g, ' ').trim();
  return t.length > 110 ? `${t.slice(0, 110)}…` : t || (l.image ? '(תמונה בלבד)' : '');
}
function letterName(l) { return l.name || 'מכתב ללא שם'; }
// The quiz's chosen letters, skipping one that was deleted since.
function chosen(q) { return { opening: letterById(q.opening) || null, closing: letterById(q.closing) || null }; }

function letterCard(letter) {
  const li = el('li', 'letter-row');
  li.dataset.id = letter.id;
  if (letter.image) {
    const img = el('img');
    img.src = letter.image;
    img.alt = '';
    li.appendChild(img);
  }
  const meta = el('div', 'meta');
  meta.appendChild(el('b', '', letterName(letter)));
  meta.appendChild(el('div', 'sub', letterGist(letter)));
  if (safeUrl(letter.linkUrl)) meta.appendChild(el('div', 'sub mono', `כפתור: ${letter.linkText || 'לפרטים נוספים'} → ${letter.linkUrl}`));
  li.appendChild(meta);
  return li;
}

// In the quiz: every letter, each marked as not sent, before the quiz or after it.
function renderLetters() {
  const list = $('letters-list');
  list.innerHTML = '';
  const { opening, closing } = chosen(current);
  // Archived letters stay out of the way, except one that is marked in this very quiz.
  const shown = letters.filter((l) => !l.archived || l === opening || l === closing);
  for (const letter of shown) {
    const li = letterCard(letter);
    const role = letter === opening ? 'opening' : letter === closing ? 'closing' : 'none';
    li.classList.toggle('picked', role !== 'none');
    const seg = el('div', 'order-row letter-role');
    seg.setAttribute('role', 'radiogroup');
    seg.setAttribute('aria-label', `תפקיד המכתב "${letterName(letter)}"`);
    for (const [value, label] of [['none', 'לא נשלח'], ['opening', 'רישא: לפני'], ['closing', 'סיפא: אחרי']]) {
      const lab = el('label', 'seg');
      const input = el('input');
      input.type = 'radio';
      input.name = `role-${letter.id}`;
      input.value = value;
      input.checked = value === role;
      input.disabled = !!current.locked;
      input.addEventListener('change', () => setRole(letter, value));
      lab.append(input, el('span', '', label));
      seg.appendChild(lab);
    }
    const actions = el('div', 'tile-actions');
    actions.append(seg, ...letterButtons(letter));
    li.appendChild(actions);
    list.appendChild(li);
  }
  if (!shown.length) list.appendChild(el('p', 'help', letters.length ? 'כל המכתבים בארכיון. אפשר להחזיר מכתב מהארכיון בלשונית "מכתבים".' : 'עוד אין מכתבים. "+ מכתב חדש" כותב את הראשון.'));
  renderShare();
}

// One letter per place: marking a letter "before" takes the place from the one that held it.
function setRole(letter, role) {
  const { opening, closing } = chosen(current);
  const patch = { opening: opening ? opening.id : '', closing: closing ? closing.id : '' };
  if (patch.opening === letter.id) patch.opening = '';
  if (patch.closing === letter.id) patch.closing = '';
  if (role !== 'none') patch[role] = letter.id;
  saveQuiz(patch);
  renderLetters();
}

// The one place a link comes from: the share button, with the letters chosen right now.
function renderShare() {
  const { opening, closing } = chosen(current);
  $('play-link').href = playUrl(current);
  const parts = [opening && `לפני החידון: "${letterName(opening)}"`, closing && `אחרי החידון: "${letterName(closing)}"`].filter(Boolean);
  $('share-with').textContent = parts.length ? `מי שפותח את הקישור רואה מכתב ${parts.join(', ומכתב ')}.` : 'מי שפותח את הקישור רואה את החידון בלי מכתבים.';
}

// The letters tab: all letters, to write, edit and delete; which quizzes they are marked in.
function renderAllLetters() {
  const active = letters.filter((l) => !l.archived);
  const archived = letters.filter((l) => l.archived);
  $('all-letters-count').textContent = active.length || '';
  const row = (letter) => {
    const li = letterCard(letter);
    const uses = quizzes.filter((q) => q.opening === letter.id || q.closing === letter.id)
      .map((q) => `${q.locked ? '🔒 ' : ''}${q.title || 'ללא שם'} (${q.opening === letter.id ? 'לפני' : 'אחרי'})`);
    if (uses.length) li.querySelector('.meta').appendChild(el('div', 'sub', `מסומן ב: ${uses.join(', ')}`));
    const actions = el('div', 'tile-actions');
    actions.append(...letterButtons(letter));
    li.appendChild(actions);
    return li;
  };
  const list = $('all-letters');
  list.innerHTML = '';
  list.append(...active.map(row));
  if (!active.length) list.appendChild(el('p', 'help', letters.length ? 'כל המכתבים בארכיון.' : 'עוד אין מכתבים.'));
  // The archive: letters Carmit is done with. They are out of every list, and still open for whoever got them.
  const wasOpen = $('letters-archive').open;
  $('letters-archive').classList.toggle('hidden', !archived.length);
  $('letters-archive').open = wasOpen && !!archived.length;
  $('archive-count').textContent = archived.length;
  $('archived-letters').replaceChildren(...archived.map(row));
}

// Edit, clone, archive, delete: the same buttons wherever a letter is listed.
function letterButtons(letter) {
  const btn = (cls, text, title, fn) => {
    const b = el('button', cls, text);
    if (title) b.title = title;
    b.addEventListener('click', fn);
    return b;
  };
  return [
    btn('link-btn', 'עריכה', '', () => openLetter(letter)),
    btn('link-btn', 'שיבוט', 'מכתב חדש, זהה לזה, שאפשר לשנות בלי לגעת במקור', () => cloneLetter(letter)),
    letter.archived
      ? btn('link-btn', 'החזרה מהארכיון', '', () => setArchived(letter, false))
      : btn('link-btn muted', 'לארכיון', 'המכתב יוצא מהרשימות, ומי שכבר קיבל אותו ממשיך לראות אותו', () => setArchived(letter, true)),
    btn('link-btn del', 'מחיקה', '', () => deleteLetter(letter)),
  ];
}

async function cloneLetter(letter) {
  const { id, created, updated, archived, ...data } = letter;
  data.name = `${letterName(letter)} (עותק)`;
  data.by = auth.currentUser.email;
  let copy;
  try {
    const ref = await fs.addDoc(fs.collection(db, 'letters'), { ...data, created: fs.serverTimestamp() });
    copy = { id: ref.id, ...data };
    letters.unshift(copy);
  } catch (err) { return toast(`השיבוט נכשל: ${err.message}`); }
  refreshLetters();
  toast('נוצר עותק של המכתב');
  openLetter(copy);
  $('letter-form-title').textContent = 'עריכת העותק';
}

async function setArchived(letter, on) {
  try {
    await fs.updateDoc(fs.doc(db, 'letters', letter.id), { archived: on, updated: fs.serverTimestamp() });
  } catch (err) { return toast(`הפעולה נכשלה: ${err.message}`); }
  letter.archived = on;
  refreshLetters();
  toast(on ? 'המכתב עבר לארכיון. מי שכבר קיבל אותו ממשיך לראות אותו.' : 'המכתב חזר מהארכיון');
}

function refreshLetters() {
  renderAllLetters();
  if (current) renderLetters();
}

async function deleteLetter(letter) {
  const locked = letterLockedIn(letter);
  if (locked.length) {
    alert(`אי אפשר למחוק את המכתב: הוא מצורף לחידון קבוע (${quizNames(locked)}), ומי שקיבל את החידון רואה אותו.\nאם כבר אין בו צורך, "לארכיון" מוציא אותו מהרשימות בלי לפגוע במה שנשלח.`);
    return false;
  }
  const uses = quizzes.filter((q) => q.opening === letter.id || q.closing === letter.id).map((q) => `"${q.title || 'ללא שם'}"`);
  const where = uses.length ? `\nהוא מסומן ב: ${uses.join(', ')}, והסימון יוסר.` : '';
  if (!confirm(`למחוק את המכתב "${letterName(letter)}"?${where}\nמי שיפתח קישור שנשלח איתו יגיע לחידון בלי המכתב.`)) return false;
  try {
    await fs.deleteDoc(fs.doc(db, 'letters', letter.id));
    // Quizzes that had it marked go back to no letter in that place.
    for (const q of quizzes) {
      const patch = {};
      if (q.opening === letter.id) patch.opening = '';
      if (q.closing === letter.id) patch.closing = '';
      if (Object.keys(patch).length) {
        Object.assign(q, patch);
        await fs.updateDoc(fs.doc(db, 'quizzes', q.id), patch);
      }
    }
  } catch (err) { return toast(`המחיקה נכשלה: ${err.message}`); }
  letters = letters.filter((l) => l !== letter);
  toast('המכתב נמחק');
  refreshLetters();
  return true;
}

const letterDialog = $('letter-dialog');
let letterEditing = null;
let letterImage = '';

function letterError(msg) {
  $('letter-error').textContent = msg;
  $('letter-error').classList.toggle('hidden', !msg);
}
function setLetterImage(data) {
  letterImage = data;
  showPreview($('letter-preview'), data);
  $('letter-img-remove').classList.toggle('hidden', !data);
}

function openLetter(letter) {
  const locked = letter ? letterLockedIn(letter) : [];
  if (locked.length) {
    if (confirm(`המכתב מצורף לחידון קבוע (${quizNames(locked)}), ולכן אי אפשר לשנות אותו.\nליצור עותק שלו ולערוך אותו? המכתב המקורי והחידון הקבוע לא ישתנו.`)) cloneLetter(letter);
    return;
  }
  letterEditing = letter;
  $('letter-form').reset();
  letterError('');
  $('letter-form-title').textContent = letter ? 'עריכת מכתב' : 'מכתב חדש';
  $('letter-name').value = letter?.name || '';
  $('letter-text').value = letter?.text || '';
  $('letter-link-text').value = letter?.linkText || '';
  $('letter-link-url').value = letter?.linkUrl || '';
  setLetterImage(letter?.image || '');
  $('letter-delete').classList.toggle('hidden', !letter);
  letterDialog.showModal();
}
$('letter-new').addEventListener('click', () => openLetter(null));
$('letter-new-home').addEventListener('click', () => openLetter(null));
$('letter-close').addEventListener('click', () => letterDialog.close());
$('letter-cancel').addEventListener('click', () => letterDialog.close());
$('letter-img-remove').addEventListener('click', () => setLetterImage(''));
$('letter-delete').addEventListener('click', async () => {
  if (await deleteLetter(letterEditing)) letterDialog.close();
});
letterDialog.addEventListener('click', (e) => { if (e.target === letterDialog) letterDialog.close(); });

$('letter-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  letterError('');
  const data = {
    name: $('letter-name').value.trim(),
    text: $('letter-text').value.trim(),
    image: letterImage,
    linkText: $('letter-link-text').value.trim(),
    linkUrl: $('letter-link-url').value.trim(),
    by: auth.currentUser.email,
  };
  if (!data.text && !data.image) return letterError('צריך לכתוב משהו במכתב, או לבחור תמונה.');
  if (data.linkUrl && !safeUrl(data.linkUrl)) return letterError('הכתובת של הכפתור לא נראית תקינה.');
  if (data.linkText && !data.linkUrl) return letterError('לכפתור יש טקסט אבל אין לאן הוא מוביל.');
  if (data.linkUrl) data.linkUrl = safeUrl(data.linkUrl);
  $('letter-save').disabled = true;
  try {
    if (letterEditing) {
      await fs.updateDoc(fs.doc(db, 'letters', letterEditing.id), { ...data, updated: fs.serverTimestamp() });
      Object.assign(letterEditing, data);
      toast('המכתב עודכן. גם מי שכבר קיבל קישור איתו יראה את הגרסה החדשה.');
    } else {
      const ref = await fs.addDoc(fs.collection(db, 'letters'), { ...data, created: fs.serverTimestamp() });
      letters.unshift({ id: ref.id, ...data });
      toast('המכתב נשמר');
    }
    letterDialog.close();
    refreshLetters();
  } catch (err) {
    letterError(`השמירה נכשלה: ${err.message}`);
  } finally {
    $('letter-save').disabled = false;
  }
});

$('quiz-delete').addEventListener('click', async () => {
  if (!confirm(`למחוק את החידון "${current.title}"?\nהחידות עצמן נשארות במאגר.`)) return;
  await fs.deleteDoc(fs.doc(db, 'quizzes', current.id));
  // Bundles that had it go on without it (the player skips a missing quiz anyway; this tidies the documents).
  for (const b of bundles.filter((x) => (x.quizIds || []).includes(current.id))) {
    b.quizIds = b.quizIds.filter((id) => id !== current.id);
    fs.updateDoc(fs.doc(db, 'bundles', b.id), { quizIds: b.quizIds, updated: fs.serverTimestamp() }).catch((e) => console.error(e));
  }
  quizzes = quizzes.filter((q) => q !== current);
  toast('החידון נמחק');
  location.hash = '';
});

$('quiz-new-riddle').addEventListener('click', () => openRiddle(null, current));

function setIds(ids) {
  saveQuiz({ riddleIds: ids });
  renderQuiz();
}

function renderQuizList(rs) {
  const list = $('quiz-list');
  list.innerHTML = '';
  if (!rs.length) {
    list.appendChild(el('li', 'empty-drop', 'גוררים לכאן חידות מהמאגר, או לוחצים עליהן'));
  }
  rs.forEach((r, i) => {
    const li = el('li', picked.has(r.id) ? 'q-item picked' : 'q-item');
    li.draggable = !current.locked;
    li.dataset.id = r.id;
    const box = el('input', 'pick');
    box.type = 'checkbox';
    box.checked = picked.has(r.id);
    box.title = 'סימון להעתקה או העברה';
    box.addEventListener('click', (e) => togglePick(r.id, box.checked, e.shiftKey));
    const img = el('img');
    img.src = r.image;
    img.alt = '';
    const meta = el('div', 'meta');
    meta.appendChild(el('b', r.answerType === 'image' ? 'tag' : 'ans', riddleLabel(r)));
    if (catId(r)) meta.appendChild(el('span', 'cat-tag', catName(catId(r))));
    if (r.hint || r.question) meta.appendChild(el('div', 'sub', [r.question, r.hint && `רמז: ${r.hint}`].filter(Boolean).join(' · ')));
    const edit = el('button', 'link-btn', 'עריכה');
    edit.addEventListener('click', () => openRiddle(r));
    const clone = el('button', 'link-btn', 'שיבוט');
    clone.title = current.locked ? 'עותק של החידה במאגר, שאפשר לשנות' : 'עותק של החידה, מיד אחריה בחידון';
    clone.addEventListener('click', () => cloneRiddle(r, current));
    const out = el('button', 'icon-btn', '✕');
    out.title = 'הוצאה מהחידון (החידה נשארת במאגר)';
    out.addEventListener('click', () => setIds(current.riddleIds.filter((id) => id !== r.id)));
    if (current.locked) { edit.classList.add('hidden'); out.classList.add('hidden'); }
    li.append(box, el('span', 'grip', '⠿'), el('span', 'num', String(i + 1).padStart(2, '0')), img, meta, edit, clone, out);
    li.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/plain', r.id);
      e.dataTransfer.effectAllowed = 'move';
      requestAnimationFrame(() => li.classList.add('dragging'));
    });
    li.addEventListener('dragend', () => li.classList.remove('dragging'));
    list.appendChild(li);
  });
  renderPickBar(rs);
}

// ---------- copy / move checked riddles to another quiz ----------

function togglePick(id, on, range) {
  const ids = quizRiddles(current).map((r) => r.id);
  const a = ids.indexOf(lastPick);
  const b = ids.indexOf(id);
  const span = range && a >= 0 ? ids.slice(Math.min(a, b), Math.max(a, b) + 1) : [id];
  span.forEach((x) => (on ? picked.add(x) : picked.delete(x)));
  lastPick = id;
  renderQuizList(quizRiddles(current));
}

const pickedIds = () => quizRiddles(current).map((r) => r.id).filter((id) => picked.has(id));

function renderPickBar(rs) {
  const n = pickedIds().length;
  $('pick-bar').classList.toggle('hidden', !rs.length);
  $('pick-all').checked = n > 0 && n === rs.length;
  $('pick-all').indeterminate = n > 0 && n < rs.length;
  $('pick-count').textContent = n ? `נבחרו ${n}` : 'בחירת הכל';
  $('pick-actions').classList.toggle('hidden', !n);
  const sel = $('pick-target');
  const keep = sel.value;
  sel.innerHTML = '';
  sel.appendChild(new Option('לחידון…', ''));
  // A locked quiz takes no new riddles.
  quizzes.filter((q) => q !== current && !q.locked).forEach((q) => sel.appendChild(new Option(q.title || 'ללא שם', q.id)));
  sel.appendChild(new Option('+ חידון חדש…', 'new'));
  sel.value = [...sel.options].some((o) => o.value === keep) ? keep : '';
}

$('pick-all').addEventListener('change', (e) => {
  picked = new Set(e.target.checked ? current.riddleIds : []);
  lastPick = null;
  renderQuizList(quizRiddles(current));
});
$('pick-clear').addEventListener('click', () => { picked = new Set(); renderQuizList(quizRiddles(current)); });
$('pick-copy').addEventListener('click', () => transfer(false));
$('pick-move').addEventListener('click', () => transfer(true));

// Copy appends the checked riddles to the target quiz, skipping ones already there; move also takes them out of this one.
async function transfer(move) {
  const ids = pickedIds();
  const choice = $('pick-target').value;
  if (!ids.length) return;
  if (!choice) { $('pick-target').focus(); return toast('בוחרים קודם לאיזה חידון'); }
  if (move && current.locked) return toast('החידון קבוע, אפשר רק להעתיק ממנו.');
  let target;
  let dup = 0;
  try {
    if (choice === 'new') {
      const title = prompt('שם לחידון החדש:', 'חידון חדש');
      if (title == null) return;
      const data = { title: title.trim() || 'חידון חדש', cover: '', riddleIds: ids, level: 'easy', by: auth.currentUser.email };
      const ref = await fs.addDoc(fs.collection(db, 'quizzes'), { ...data, created: fs.serverTimestamp() });
      target = { id: ref.id, ...data };
      quizzes.unshift(target);
    } else {
      target = quizzes.find((q) => q.id === choice);
      const fresh = ids.filter((id) => !target.riddleIds.includes(id));
      dup = ids.length - fresh.length;
      const riddleIds = [...target.riddleIds, ...fresh];
      await fs.updateDoc(fs.doc(db, 'quizzes', target.id), { riddleIds, updated: fs.serverTimestamp() });
      target.riddleIds = riddleIds;
    }
    if (move) await saveQuiz({ riddleIds: current.riddleIds.filter((id) => !picked.has(id)) });
  } catch (err) {
    return toast(`הפעולה נכשלה: ${err.message}`);
  }
  const n = ids.length;
  const verb = move ? (n === 1 ? 'הועברה' : 'הועברו') : (n === 1 ? 'הועתקה' : 'הועתקו');
  const extra = !dup ? '' : dup === n ? ' (כבר היו שם)' : dup === 1 ? ' (אחת מהן כבר הייתה שם)' : ` (${dup} מהן כבר היו שם)`;
  toast(`${n === 1 ? 'חידה אחת' : `${n} חידות`} ${verb} ל"${target.title}"${extra}`);
  picked = new Set();
  renderQuiz();
  $('pick-target').value = target.id; // a quiz just created stays the target for the next batch
}

// The side pool can be narrowed to the riddles of one other quiz, to pull from it.
function renderSideSource() {
  const sel = $('side-source');
  const keep = sel.value;
  sel.innerHTML = '';
  sel.appendChild(new Option('מכל המאגר', ''));
  sel.appendChild(new Option('רק כאלה שלא בשום חידון', 'none'));
  categories.forEach((c) => sel.appendChild(new Option(`קטגוריה: ${c.name}`, `cat:${c.id}`)));
  if (categories.length) sel.appendChild(new Option(`קטגוריה: ${GENERAL}`, 'cat:'));
  quizzes.filter((q) => q !== current).forEach((q) => sel.appendChild(new Option(`מתוך: ${q.title || 'ללא שם'}`, q.id)));
  sel.value = [...sel.options].some((o) => o.value === keep) ? keep : '';
}

let sideShown = [];

function renderSidePool() {
  const term = $('side-search').value.trim().toLowerCase();
  const source = $('side-source').value;
  const out = riddles.filter((r) => !current.riddleIds.includes(r.id));
  $('out-count').textContent = out.length;
  let from = out;
  if (source === 'none') from = out.filter((r) => !quizzesWith(r.id).length);
  else if (source.startsWith('cat:')) from = out.filter((r) => catId(r) === source.slice(4));
  else if (source) {
    const src = quizzes.find((q) => q.id === source);
    from = src ? quizRiddles(src).filter((r) => !current.riddleIds.includes(r.id)) : out;
  }
  sideShown = from.filter((x) => matches(x, term));
  const all = $('side-add-all');
  all.classList.toggle('hidden', !(source || term) || sideShown.length < 2);
  all.textContent = `+ הוספת כל ה-${sideShown.length} לסוף החידון`;
  const box = $('side-pool');
  box.innerHTML = '';
  for (const r of sideShown) {
    const item = el('button', 'side-item');
    item.draggable = true;
    item.title = 'הוספה לחידון';
    const img = el('img');
    img.src = r.image;
    img.alt = '';
    const n = quizzesWith(r.id).length;
    item.append(img, el('span', r.answerType === 'image' ? 'tag' : 'ans', riddleLabel(r)),
      el('span', 'uses', n ? `ב-${plural(n, 'חידון אחד', 'חידונים')}` : 'חדשה'), el('span', 'add', '+'));
    item.addEventListener('click', () => { setIds([...current.riddleIds, r.id]); toast('נוספה לסוף החידון'); });
    item.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/plain', r.id);
      e.dataTransfer.effectAllowed = 'copyMove';
    });
    box.appendChild(item);
  }
  if (!box.children.length) {
    box.appendChild(el('p', 'help', term || source ? 'לא נמצאו חידות.' : 'כל החידות שבמאגר כבר בחידון הזה.'));
  }
}
$('side-search').addEventListener('input', renderSidePool);
$('side-source').addEventListener('change', renderSidePool);
$('side-add-all').addEventListener('click', () => {
  const n = sideShown.length;
  setIds([...current.riddleIds, ...sideShown.map((r) => r.id)]);
  toast(`${n} חידות נוספו לסוף החידון`);
});

// Dropping on the list inserts at the pointer: reorders a riddle already in it, or adds one from the pool.
const list = $('quiz-list');
const marker = el('li', 'drop-marker');

function dropIndex(y) {
  const items = [...list.querySelectorAll('.q-item:not(.dragging)')];
  const i = items.findIndex((it) => y < it.getBoundingClientRect().top + it.offsetHeight / 2);
  return { items, i: i < 0 ? items.length : i };
}

list.addEventListener('dragover', (e) => {
  e.preventDefault();
  const { items, i } = dropIndex(e.clientY);
  if (items[i]) list.insertBefore(marker, items[i]);
  else list.appendChild(marker);
});
list.addEventListener('dragleave', (e) => { if (!list.contains(e.relatedTarget)) marker.remove(); });
list.addEventListener('drop', (e) => {
  e.preventDefault();
  const id = e.dataTransfer.getData('text/plain');
  const { items, i } = dropIndex(e.clientY);
  marker.remove();
  if (!byId.has(id)) return;
  const before = items[i]?.dataset.id;
  const ids = current.riddleIds.filter((x) => x !== id);
  const at = before ? ids.indexOf(before) : ids.length;
  ids.splice(at, 0, id);
  setIds(ids);
});

// ---------- images ----------

function readImage(file, maxSide = MAX_SIDE, maxB64 = MAX_B64) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      let scale = Math.min(1, maxSide / Math.max(img.width, img.height));
      for (let attempt = 0; attempt < 8; attempt++) {
        const c = document.createElement('canvas');
        c.width = Math.round(img.width * scale);
        c.height = Math.round(img.height * scale);
        const ctx = c.getContext('2d');
        ctx.fillStyle = '#fff';
        ctx.fillRect(0, 0, c.width, c.height);
        ctx.drawImage(img, 0, 0, c.width, c.height);
        for (const q of [0.85, 0.75, 0.65, 0.55]) {
          const data = c.toDataURL('image/jpeg', q);
          if (data.length < maxB64) return resolve(data);
        }
        scale *= 0.8;
      }
      reject(new Error('התמונה גדולה מדי'));
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('לא הצלחתי לקרוא את התמונה')); };
    img.src = url;
  });
}

function showPreview(img, data) {
  img.src = data || '';
  img.classList.toggle('hidden', !data);
  img.closest('.drop').classList.toggle('filled', !!data);
}

// A drop zone takes a click (file picker) or a dragged-in file.
function dropZone(zone, input, preview, set, onError = showError, size = []) {
  const take = async (file) => {
    if (!file) return;
    try { const data = await readImage(file, ...size); set(data); showPreview(preview, data); onError(''); }
    catch (err) { onError(err.message); }
  };
  input.addEventListener('change', (e) => { take(e.target.files[0]); e.target.value = ''; });
  zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('over');
    take(e.dataTransfer.files[0]);
  });
}
// A file dropped just outside a zone would otherwise make the browser leave the page to show it.
for (const ev of ['dragover', 'drop']) {
  addEventListener(ev, (e) => { if (e.dataTransfer?.types.includes('Files')) e.preventDefault(); });
}
dropZone($('image-drop'), $('image-file'), $('image-preview'), (d) => { image = d; });
dropZone($('answer-drop'), $('answer-file'), $('answer-preview'), (d) => { answerImage = d; });
dropZone($('letter-drop'), $('letter-file'), $('letter-preview'), setLetterImage, letterError, [1000, 300_000]);

// ---------- answers ----------

const answerType = () => document.querySelector('input[name="answer-type"]:checked').value;

document.querySelectorAll('input[name="answer-type"]').forEach((r) => r.addEventListener('change', () => setType(answerType())));

function addAnswerRow(value = '', primary = false) {
  const row = document.createElement('div');
  row.className = 'answer-row';
  row.innerHTML = `
    <label class="star" title="התשובה העיקרית"><input type="radio" name="primary"><span>⭐</span></label>
    <input type="text" class="answer-text" autocomplete="off" placeholder="תשובה">
    <button type="button" class="link-btn remove" title="הסרה">✕</button>`;
  row.querySelector('.answer-text').value = value;
  row.querySelector('input[name="primary"]').checked = primary;
  row.querySelector('.remove').addEventListener('click', () => {
    const wasPrimary = row.querySelector('input[name="primary"]').checked;
    row.remove();
    const first = document.querySelector('#answers-list input[name="primary"]');
    if (wasPrimary && first) first.checked = true;
    if (!first) addAnswerRow('', true);
  });
  $('answers-list').appendChild(row);
  return row;
}
$('add-answer').addEventListener('click', () => {
  const noPrimary = !document.querySelector('#answers-list input[name="primary"]:checked');
  addAnswerRow('', noPrimary).querySelector('.answer-text').focus();
});

// ---------- the riddle dialog ----------

const dialog = $('riddle-dialog');

function showError(msg) {
  $('form-error').textContent = msg;
  $('form-error').classList.toggle('hidden', !msg);
}

function setType(type) {
  document.querySelector(`input[name="answer-type"][value="${type}"]`).checked = true;
  $('text-answers').classList.toggle('hidden', type !== 'text');
  $('image-answer').classList.toggle('hidden', type !== 'image');
}

function resetForm() {
  editing = null;
  image = '';
  answerImage = '';
  $('riddle-form').reset();
  showPreview($('image-preview'), '');
  showPreview($('answer-preview'), '');
  $('answers-list').innerHTML = '';
  addAnswerRow('', true);
  setType('text');
  showError('');
}

// r = null opens an empty form; quiz = the quiz a new riddle should join.
function openRiddle(r, quiz = null) {
  const locked = r ? lockedWith(r.id) : [];
  if (locked.length) {
    // From a quiz that is not locked itself, the editable copy takes the original's place there.
    const here = current && !current.locked && current.riddleIds.includes(r.id);
    const what = here ? `
בחידון "${current.title}" העותק יחליף את המקור. בחידון הקבוע דבר לא ישתנה.` : '';
    if (confirm(`החידה נמצאת בחידון קבוע (${quizNames(locked)}), ולכן אי אפשר לשנות אותה.
ליצור עותק שלה ולערוך אותו?${what}`)) cloneRiddle(r, here ? current : null, true);
    return;
  }
  resetForm();
  addToQuiz = quiz;
  // A new riddle made while the pool shows one category starts in that category.
  fillCatSelect($('riddle-category'), r ? catId(r) : (!quiz && poolCat) || '');
  $('form-title').textContent = r ? 'עריכת חידה' : quiz ? `חידה חדשה ב"${quiz.title}"` : 'חידה חדשה למאגר';
  if (r) {
    editing = r.id;
    image = r.image;
    answerImage = r.answerImage || '';
    showPreview($('image-preview'), image);
    showPreview($('answer-preview'), answerImage);
    $('question').value = r.question || '';
    $('hint').value = r.hint || '';
    $('ai-image').checked = !!r.aiImage;
    setType(r.answerType);
    if (r.answerType === 'text' && r.answers.length) {
      $('answers-list').innerHTML = '';
      r.answers.forEach((a, i) => addAnswerRow(a, i === r.primary));
    }
    const n = quizzesWith(r.id).length;
    if (n > 1) $('form-title').textContent += ` (מופיעה ב-${n} חידונים)`;
  }
  dialog.showModal();
}

onCatSelect($('riddle-category'), (id) => fillCatSelect($('riddle-category'), id));
$('cancel-btn').addEventListener('click', () => dialog.close());
$('dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });

$('riddle-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  showError('');
  const type = answerType();
  const rows = [...document.querySelectorAll('#answers-list .answer-row')];
  let answers = [];
  let primary = 0;
  rows.forEach((row) => {
    const text = row.querySelector('.answer-text').value.trim();
    if (!text) return;
    if (row.querySelector('input[name="primary"]').checked) primary = answers.length;
    answers.push(text);
  });

  if (!image) return showError('צריך לבחור תמונה לחידה.');
  if (type === 'text' && !answers.length) return showError('צריך לכתוב לפחות תשובה אחת.');
  if (type === 'image' && !answerImage) return showError('צריך לבחור תמונה לתשובה.');
  if (type === 'image') { answers = []; primary = 0; }

  const data = {
    image,
    question: $('question').value.trim(),
    hint: $('hint').value.trim(),
    category: $('riddle-category').dataset.value || '',
    aiImage: $('ai-image').checked,
    answerType: type,
    answers,
    primary,
    answerImage: type === 'image' ? answerImage : '',
    by: auth.currentUser.email,
  };
  $('save-btn').disabled = true;
  $('save-btn').textContent = 'שומרים…';
  try {
    if (editing) {
      await fs.updateDoc(fs.doc(db, 'riddles', editing), { ...data, updated: fs.serverTimestamp() });
      Object.assign(byId.get(editing), data);
      toast('החידה עודכנה');
    } else {
      const ref = await fs.addDoc(fs.collection(db, 'riddles'), { ...data, created: fs.serverTimestamp() });
      const r = { id: ref.id, ...data };
      riddles.unshift(r);
      byId.set(r.id, r);
      if (addToQuiz && addToQuiz === current) {
        await saveQuiz({ riddleIds: [...current.riddleIds, r.id] });
        toast('החידה נשמרה ונוספה לחידון');
      } else {
        toast('החידה נשמרה במאגר');
      }
    }
    dialog.close();
    current ? renderQuiz() : renderHome();
  } catch (err) {
    showError(`השמירה נכשלה: ${err.message}`);
  } finally {
    $('save-btn').disabled = false;
    $('save-btn').textContent = 'שמירה';
  }
});

// ---------- "גבר גבר": requests to Claude ----------
// Carmit writes what to change; once an hour tools/gever_cron.sh hands the new requests to Claude, who changes the
// app and writes the answer on the request. A request can continue an earlier one (followUp), e.g. to answer a
// question Claude asked; it then shows right under it.

const GEVER_SEEN = 'carmit-gever-seen';
const STATUS = {
  new: 'ממתינה לקלוד (הוא בודק פעם בשעה)',
  working: 'קלוד עובד על זה עכשיו',
  done: '✓ טופלה',
  question: 'לקלוד יש שאלה',
};
const personName = (email) => ({ 'carmitconsari@gmail.com': 'כרמית', 'orimosenzon@gmail.com': 'אורי' })[email] || email || '';
const fmtWhen = (t) => (stamp(t) ? new Date(stamp(t)).toLocaleString('he-IL', { day: 'numeric', month: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' }) : '');
let geverImage = '';
let geverFollow = '';

// The tab's count: answers that came since this browser last opened the tab.
function geverBadge() {
  let seen = 0;
  try { seen = +localStorage.getItem(GEVER_SEEN) || 0; } catch { /* private mode */ }
  const open = !$('tab-gever').classList.contains('hidden');
  const fresh = gever.filter((g) => g.reply && stamp(g.replied) > seen).length;
  $('gever-count').textContent = open || !fresh ? '' : `${fresh} חדש`;
}

function renderGever() {
  try { localStorage.setItem(GEVER_SEEN, String(Date.now())); } catch { /* private mode */ }
  const list = $('gever-list');
  if (!gever.length) {
    list.replaceChildren(el('li', 'help', 'עוד לא נכתבו בקשות. הבקשה הראשונה שתכתבי תופיע כאן, ואחריה התשובה של קלוד.'));
    return;
  }
  // Threads: a request that continues another shows right under it; threads by their newest message, newest first.
  const byIdG = new Map(gever.map((g) => [g.id, g]));
  const root = (g) => { const seen = new Set(); while (g.followUp && byIdG.has(g.followUp) && !seen.has(g.id)) { seen.add(g.id); g = byIdG.get(g.followUp); } return g; };
  const threads = Map.groupBy([...gever].sort((a, b) => stamp(a.created) - stamp(b.created)), (g) => root(g).id);
  const last = (t) => Math.max(...t.map((g) => Math.max(stamp(g.created), stamp(g.replied))));
  const items = [];
  for (const t of [...threads.values()].sort((a, b) => last(b) - last(a))) {
    t.forEach((g, i) => items.push(geverItem(g, i > 0)));
  }
  list.replaceChildren(...items);
}

function geverItem(g, followup) {
  const li = el('li', `pane gever-item${followup ? ' followup' : ''}`);
  const head = el('div', 'gever-head');
  const status = g.status in STATUS ? g.status : 'new';
  head.append(el('b', '', personName(g.by)), el('span', 'help', fmtWhen(g.created)), el('span', `gever-status ${status}`, STATUS[status]));
  li.append(head, el('p', 'gever-text', g.text));
  if (g.image) {
    const img = el('img', 'gever-shot');
    img.src = g.image;
    img.alt = 'צילום מסך';
    img.loading = 'lazy';
    img.addEventListener('click', () => window.open().document.write(`<img src="${g.image}" style="max-width:100%">`));
    li.appendChild(img);
  }
  if (g.reply) {
    const box = el('div', 'gever-reply');
    const rh = el('div', 'gever-head');
    rh.append(el('span', 'who', 'קלוד'), el('span', 'help', fmtWhen(g.replied)));
    box.append(rh, el('p', 'gever-text', g.reply));
    li.appendChild(box);
  }
  const foot = el('div', 'gever-foot');
  const more = el('button', 'link-btn', g.status === 'question' ? '↩ לענות לקלוד' : '↩ להמשיך את הבקשה הזאת');
  more.addEventListener('click', () => setFollow(g));
  foot.appendChild(more);
  if (status === 'new') {
    const del = el('button', 'link-btn del', 'ביטול הבקשה');
    del.addEventListener('click', async () => {
      if (!confirm('לבטל את הבקשה? קלוד עוד לא ראה אותה.')) return;
      try { await fs.deleteDoc(fs.doc(db, 'gever', g.id)); }
      catch (err) { return toast(`הביטול נכשל: ${err.message}`); }
      gever = gever.filter((x) => x !== g);
      renderGever();
      toast('הבקשה בוטלה');
    });
    foot.appendChild(del);
  }
  li.appendChild(foot);
  return li;
}

function setFollow(g) {
  geverFollow = g ? g.id : '';
  $('gever-follow').classList.toggle('hidden', !g);
  if (g) {
    const words = g.text.length > 60 ? `${g.text.slice(0, 60)}…` : g.text;
    $('gever-follow').querySelector('span').textContent = `המשך ל: "${words}"`;
    scrollTo({ top: 0 });
    $('gever-text').focus();
  }
}
$('gever-follow-cancel').addEventListener('click', () => setFollow(null));

function setGeverImage(data) {
  geverImage = data;
  $('gever-img').src = data || '';
  $('gever-img-box').classList.toggle('hidden', !data);
}
const geverError = (msg) => { $('gever-error').textContent = msg; $('gever-error').classList.toggle('hidden', !msg); };
async function takeGeverImage(file) {
  if (!file) return;
  try { setGeverImage(await readImage(file)); geverError(''); } catch (err) { geverError(err.message); }
}
$('gever-file').addEventListener('change', (e) => { takeGeverImage(e.target.files[0]); e.target.value = ''; });
$('gever-img-remove').addEventListener('click', () => setGeverImage(''));
// A screenshot can also be pasted straight into the text box.
$('gever-text').addEventListener('paste', (e) => {
  const file = [...(e.clipboardData?.files || [])].find((f) => f.type.startsWith('image/'));
  if (file) { e.preventDefault(); takeGeverImage(file); }
});
$('gever-text').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $('gever-form').requestSubmit(); }
});

$('gever-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = $('gever-text').value.trim();
  if (!text) return geverError('מה לכתוב לקלוד? התיבה ריקה.');
  $('gever-send').disabled = true;
  try {
    const by = (auth.currentUser?.email || '').toLowerCase();
    const { id } = await sendGever({ by, text, image: geverImage, followUp: geverFollow });
    gever.push({ id, by, text, image: geverImage, followUp: geverFollow, status: 'new', created: Date.now() });
  } catch (err) {
    return geverError(`השליחה נכשלה: ${err.message}`);
  } finally {
    $('gever-send').disabled = false;
  }
  $('gever-text').value = '';
  setGeverImage('');
  setFollow(null);
  geverError('');
  renderGever();
  toast('נשלח. קלוד יענה כאן בתוך שעה בערך.');
});
