// Ori and Carmit manage quizzes (שעשועונים) here: titled, ordered selections from one shared pool of riddles.
// Images are shrunk in the browser and kept inside the Firestore documents as data URLs, so no Storage
// bucket (and no billing) is needed. Routing is by hash: '' = all quizzes, '#pool', '#quiz/<id>'.
import { app, db, fs, ADMINS, loadRiddles, loadQuizzes, primaryAnswer, DEFAULT_COVER } from './fb.js';

const au = await import('https://www.gstatic.com/firebasejs/10.14.1/firebase-auth.js');
const auth = au.getAuth(app);
auth.languageCode = 'he';

const $ = (id) => document.getElementById(id);
const MAX_SIDE = 1280;
const MAX_B64 = 380_000; // two images must fit together in one 1 MiB document

let riddles = [];        // the pool, newest first
let byId = new Map();
let quizzes = [];
let current = null;      // the quiz open in the quiz view

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
const playUrl = (q) => new URL(`index.html?q=${q.id}`, location.href).href;

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
    await reload();
    route();
  }
});

async function reload() {
  [riddles, quizzes] = await Promise.all([loadRiddles(), loadQuizzes()]);
  byId = new Map(riddles.map((r) => [r.id, r]));
}

// ---------- routing ----------

window.addEventListener('hashchange', route);

function route() {
  if (!auth.currentUser) return;
  hidePeek();
  const [, quizId] = location.hash.match(/^#quiz\/(.+)$/) || [];
  const quiz = quizId && quizzes.find((q) => q.id === quizId);
  $('home-view').classList.toggle('hidden', !!quiz);
  $('quiz-view').classList.toggle('hidden', !quiz);
  if (quiz) return openQuiz(quiz);
  current = null;
  setTab(location.hash === '#pool' ? 'pool' : 'quizzes');
  renderHome();
}

function setTab(tab) {
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === tab));
  $('tab-quizzes').classList.toggle('hidden', tab !== 'quizzes');
  $('tab-pool').classList.toggle('hidden', tab !== 'pool');
}
document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => {
  history.replaceState(null, '', t.dataset.tab === 'pool' ? '#pool' : location.pathname);
  route();
}));

// ---------- home: the quiz gallery ----------

function renderHome() {
  $('quiz-count').textContent = quizzes.length;
  $('pool-count').textContent = riddles.length;
  renderQuizGrid();
  renderPool();
}

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
  add.innerHTML = '<span class="plus">+</span><span>שעשועון חדש</span>';
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
    cap.append(el('b', '', q.title || 'ללא שם'), el('span', '', rs.length ? plural(rs.length, 'חידה אחת', 'חידות') : 'עוד אין חידות'));
    card.appendChild(cap);
    card.addEventListener('mouseenter', () => schedulePeek(card, q));
    card.addEventListener('mouseleave', hidePeek);
    grid.appendChild(card);
  }
}

async function createQuiz() {
  const data = { title: 'שעשועון חדש', cover: '', riddleIds: [], by: auth.currentUser.email };
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
    peek.appendChild(el('p', 'peek-empty', 'השעשועון ריק. לחיצה פותחת אותו כדי להוסיף חידות.'));
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

function renderPool() {
  const term = $('pool-search').value.trim().toLowerCase();
  const grid = $('pool-grid');
  grid.innerHTML = '';
  for (const r of riddles.filter((x) => matches(x, term))) {
    const tile = el('div', 'pool-tile');
    const img = el('img');
    img.src = r.image;
    img.alt = '';
    const body = el('div', 'tile-body');
    body.appendChild(el('b', r.answerType === 'image' ? 'tag' : 'ans', riddleLabel(r)));
    if (r.hint) body.appendChild(el('div', 'sub', `רמז: ${r.hint}`));
    const qs = quizzesWith(r.id);
    const chips = el('div', 'chips');
    if (qs.length) {
      qs.forEach((q) => {
        const chip = el('a', 'chip', q.title || 'ללא שם');
        chip.href = `#quiz/${q.id}`;
        chips.appendChild(chip);
      });
    } else {
      chips.appendChild(el('span', 'chip lonely', 'לא בשום שעשועון'));
    }
    body.appendChild(chips);
    const actions = el('div', 'tile-actions');
    const edit = el('button', 'link-btn', 'עריכה');
    edit.addEventListener('click', () => openRiddle(r));
    const del = el('button', 'link-btn del', 'מחיקה');
    del.addEventListener('click', () => deleteRiddle(r));
    actions.append(edit, del);
    tile.append(img, body, actions);
    grid.appendChild(tile);
  }
  if (!grid.children.length) grid.appendChild(el('p', 'help', term ? 'לא נמצאו חידות.' : 'המאגר ריק.'));
}
$('pool-search').addEventListener('input', renderPool);
$('pool-new').addEventListener('click', () => openRiddle(null));

async function deleteRiddle(r) {
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
  $('quiz-title').value = q.title;
  $('side-search').value = '';
  $('play-link').href = playUrl(q);
  setSaveState('');
  renderQuiz();
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
  renderSidePool();
}

let saveTimer = null;
function setSaveState(s) {
  const st = $('save-state');
  st.textContent = { saving: 'שומרים…', saved: 'נשמר ✓', error: 'השמירה נכשלה' }[s] || '';
  st.className = `save-state ${s}`;
}

async function saveQuiz(patch) {
  const q = current;
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

$('share-btn').addEventListener('click', async () => {
  const url = playUrl(current);
  try {
    await navigator.clipboard.writeText(url);
    toast('הקישור הועתק. אפשר להדביק ולשלוח.');
  } catch {
    prompt('הקישור לשעשועון:', url);
  }
});

$('quiz-delete').addEventListener('click', async () => {
  if (!confirm(`למחוק את השעשועון "${current.title}"?\nהחידות עצמן נשארות במאגר.`)) return;
  await fs.deleteDoc(fs.doc(db, 'quizzes', current.id));
  quizzes = quizzes.filter((q) => q !== current);
  toast('השעשועון נמחק');
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
    const li = el('li', 'q-item');
    li.draggable = true;
    li.dataset.id = r.id;
    const img = el('img');
    img.src = r.image;
    img.alt = '';
    const meta = el('div', 'meta');
    meta.appendChild(el('b', r.answerType === 'image' ? 'tag' : 'ans', riddleLabel(r)));
    if (r.hint || r.question) meta.appendChild(el('div', 'sub', [r.question, r.hint && `רמז: ${r.hint}`].filter(Boolean).join(' · ')));
    const edit = el('button', 'link-btn', 'עריכה');
    edit.addEventListener('click', () => openRiddle(r));
    const out = el('button', 'icon-btn', '✕');
    out.title = 'הוצאה מהשעשועון (החידה נשארת במאגר)';
    out.addEventListener('click', () => setIds(current.riddleIds.filter((id) => id !== r.id)));
    li.append(el('span', 'grip', '⠿'), el('span', 'num', String(i + 1).padStart(2, '0')), img, meta, edit, out);
    li.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/plain', r.id);
      e.dataTransfer.effectAllowed = 'move';
      requestAnimationFrame(() => li.classList.add('dragging'));
    });
    li.addEventListener('dragend', () => li.classList.remove('dragging'));
    list.appendChild(li);
  });
}

function renderSidePool() {
  const term = $('side-search').value.trim().toLowerCase();
  const out = riddles.filter((r) => !current.riddleIds.includes(r.id));
  $('out-count').textContent = out.length;
  const box = $('side-pool');
  box.innerHTML = '';
  for (const r of out.filter((x) => matches(x, term))) {
    const item = el('button', 'side-item');
    item.draggable = true;
    item.title = 'הוספה לשעשועון';
    const img = el('img');
    img.src = r.image;
    img.alt = '';
    const n = quizzesWith(r.id).length;
    item.append(img, el('span', r.answerType === 'image' ? 'tag' : 'ans', riddleLabel(r)),
      el('span', 'uses', n ? `ב-${plural(n, 'שעשועון אחד', 'שעשועונים')}` : 'חדשה'), el('span', 'add', '+'));
    item.addEventListener('click', () => { setIds([...current.riddleIds, r.id]); toast('נוספה לסוף השעשועון'); });
    item.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/plain', r.id);
      e.dataTransfer.effectAllowed = 'copyMove';
    });
    box.appendChild(item);
  }
  if (!box.children.length) {
    box.appendChild(el('p', 'help', term ? 'לא נמצאו חידות.' : 'כל החידות שבמאגר כבר בשעשועון הזה.'));
  }
}
$('side-search').addEventListener('input', renderSidePool);

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

function readImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      let scale = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
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
          if (data.length < MAX_B64) return resolve(data);
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
function dropZone(zone, input, preview, set) {
  const take = async (file) => {
    if (!file) return;
    try { const data = await readImage(file); set(data); showPreview(preview, data); showError(''); }
    catch (err) { showError(err.message); }
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
  resetForm();
  addToQuiz = quiz;
  $('form-title').textContent = r ? 'עריכת חידה' : quiz ? `חידה חדשה ב"${quiz.title}"` : 'חידה חדשה למאגר';
  if (r) {
    editing = r.id;
    image = r.image;
    answerImage = r.answerImage || '';
    showPreview($('image-preview'), image);
    showPreview($('answer-preview'), answerImage);
    $('question').value = r.question || '';
    $('hint').value = r.hint || '';
    setType(r.answerType);
    if (r.answerType === 'text' && r.answers.length) {
      $('answers-list').innerHTML = '';
      r.answers.forEach((a, i) => addAnswerRow(a, i === r.primary));
    }
    const n = quizzesWith(r.id).length;
    if (n > 1) $('form-title').textContent += ` (מופיעה ב-${n} שעשועונים)`;
  }
  dialog.showModal();
}

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
        toast('החידה נשמרה ונוספה לשעשועון');
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
