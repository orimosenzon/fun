// Ori and Carmit add, edit and delete riddles here. Images are shrunk in the browser and kept
// inside the riddle's Firestore document as a data URL, so no Storage bucket (and no billing) is needed.
import { app, db, fs, ADMINS, loadRiddles, primaryAnswer } from './fb.js';

const au = await import('https://www.gstatic.com/firebasejs/10.14.1/firebase-auth.js');
const auth = au.getAuth(app);
auth.languageCode = 'he';

const $ = (id) => document.getElementById(id);
const MAX_SIDE = 1280;
const MAX_B64 = 380_000; // two images must fit together in one 1 MiB document

let riddles = [];
let editing = null;      // id of the riddle being edited, or null for a new one
let image = '';          // data URLs of the current form
let answerImage = '';

// ---------- sign-in ----------

$('login-btn').addEventListener('click', async () => {
  try {
    await au.signInWithPopup(auth, new au.GoogleAuthProvider());
  } catch (e) {
    if (e.code !== 'auth/popup-closed-by-user') $('login-msg').textContent = `הכניסה נכשלה: ${e.message}`;
  }
});
$('logout-btn').addEventListener('click', () => au.signOut(auth));

au.onAuthStateChanged(auth, (user) => {
  const ok = user && ADMINS.includes((user.email || '').toLowerCase());
  $('login-screen').classList.toggle('hidden', !!ok);
  $('admin-screen').classList.toggle('hidden', !ok);
  if (user && !ok) {
    $('login-msg').textContent = `החשבון ${user.email} לא מורשה לנהל חידות. רק אורי וכרמית יכולים.`;
    $('login-btn').textContent = 'כניסה עם חשבון אחר';
    au.signOut(auth);
  }
  if (ok) {
    $('who').textContent = user.email;
    refresh();
    resetForm();
  }
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

function showPreview(el, data) {
  el.src = data || '';
  el.classList.toggle('hidden', !data);
}

$('image-file').addEventListener('change', async (e) => {
  if (!e.target.files[0]) return;
  try { image = await readImage(e.target.files[0]); showPreview($('image-preview'), image); }
  catch (err) { showError(err.message); }
});
$('answer-file').addEventListener('change', async (e) => {
  if (!e.target.files[0]) return;
  try { answerImage = await readImage(e.target.files[0]); showPreview($('answer-preview'), answerImage); }
  catch (err) { showError(err.message); }
});

// ---------- answers ----------

const answerType = () => document.querySelector('input[name="answer-type"]:checked').value;

document.querySelectorAll('input[name="answer-type"]').forEach((r) => r.addEventListener('change', () => {
  $('text-answers').classList.toggle('hidden', answerType() !== 'text');
  $('image-answer').classList.toggle('hidden', answerType() !== 'image');
}));

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

// ---------- form ----------

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
  $('form-title').textContent = 'חידה חדשה';
  $('cancel-btn').classList.add('hidden');
  showPreview($('image-preview'), '');
  showPreview($('answer-preview'), '');
  $('answers-list').innerHTML = '';
  addAnswerRow('', true);
  setType('text');
  showError('');
}

function editRiddle(r) {
  resetForm();
  editing = r.id;
  image = r.image;
  answerImage = r.answerImage || '';
  $('form-title').textContent = 'עריכת חידה';
  $('cancel-btn').classList.remove('hidden');
  showPreview($('image-preview'), image);
  showPreview($('answer-preview'), answerImage);
  $('question').value = r.question || '';
  $('hint').value = r.hint || '';
  setType(r.answerType);
  if (r.answerType === 'text' && r.answers.length) {
    $('answers-list').innerHTML = '';
    r.answers.forEach((a, i) => addAnswerRow(a, i === r.primary));
  }
  $('riddle-form').scrollIntoView({ behavior: 'smooth' });
}

$('cancel-btn').addEventListener('click', resetForm);

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
    } else {
      await fs.addDoc(fs.collection(db, 'riddles'), { ...data, created: fs.serverTimestamp() });
    }
    resetForm();
    await refresh();
  } catch (err) {
    showError(`השמירה נכשלה: ${err.message}`);
  } finally {
    $('save-btn').disabled = false;
    $('save-btn').textContent = 'שמירה';
  }
});

// ---------- list ----------

async function refresh() {
  riddles = await loadRiddles();
  $('count').textContent = riddles.length;
  const list = $('riddle-list');
  list.innerHTML = '';
  for (const r of riddles) {
    const item = document.createElement('div');
    item.className = 'riddle-item';
    const answer = r.answerType === 'image'
      ? '<span class="tag">תשובה בתמונה</span>'
      : `<b></b>${r.answers.length > 1 ? ` <span class="tag">+${r.answers.length - 1}</span>` : ''}`;
    item.innerHTML = `
      <img src="" alt="">
      <div class="meta"><div class="ans">${answer}</div><div class="sub"></div></div>
      <div class="actions">
        <button type="button" class="link-btn edit">עריכה</button>
        <button type="button" class="link-btn del">מחיקה</button>
      </div>`;
    item.querySelector('img').src = r.image;
    if (r.answerType === 'text') item.querySelector('.ans b').textContent = primaryAnswer(r);
    item.querySelector('.sub').textContent = [r.question, r.hint && `רמז: ${r.hint}`].filter(Boolean).join(' · ');
    item.querySelector('.edit').addEventListener('click', () => editRiddle(r));
    item.querySelector('.del').addEventListener('click', async () => {
      if (!confirm('למחוק את החידה הזאת? אי אפשר לבטל את זה.')) return;
      await fs.deleteDoc(fs.doc(db, 'riddles', r.id));
      if (editing === r.id) resetForm();
      refresh();
    });
    list.appendChild(item);
  }
}
