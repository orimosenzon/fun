// editor.js - עורך העולם: מספרים סיפור רקע, ומודל התמונות של Azure מצייר

import { LOCATIONS } from './world.js';

const REGION_NAMES = {
  shire: 'השאייר והמערב',
  eriador: 'אריאדור',
  misty_mountains: 'הרי הערפל',
  rohan: 'רוהאן',
  gondor: 'גונדור',
  mordor: 'מורדור',
};

// סדר המסע מערבה-מזרחה, כדי שהרשימה תתנהג כמו הדרך ולא כמו א"ב
const REGION_ORDER = ['shire', 'eriador', 'misty_mountains', 'rohan', 'gondor', 'mordor'];

const $ = (id) => document.getElementById(id);

const el = {
  modelBadge: $('model-badge'),
  progressBadge: $('progress-badge'),
  list: $('loc-list'),
  filter: $('loc-filter'),
  empty: $('ed-empty'),
  detail: $('ed-detail'),
  name: $('d-name'),
  meta: $('d-meta'),
  description: $('d-description'),
  lore: $('d-lore'),
  visual: $('d-visual'),
  saveState: $('save-state'),
  saveBtn: $('save-btn'),
  genBtn: $('gen-btn'),
  promptBtn: $('prompt-btn'),
  promptView: $('prompt-view'),
  imgView: $('img-view'),
  imgPlaceholder: $('img-placeholder'),
  imgSpinner: $('img-spinner'),
  imgNote: $('img-note'),
  worldBtn: $('world-lore-btn'),
  worldModal: $('world-modal'),
  closeWorld: $('close-world'),
  wTitle: $('w-title'),
  wPremise: $('w-premise'),
  wStyle: $('w-style'),
  saveWorld: $('save-world'),
  worldSaveState: $('world-save-state'),
};

// state מהשרת: location_id -> רשומה
let canon = {};
let lore = {};
let selectedId = null;
let dirty = false;

// ── תקשורת ──────────────────────────────────────────────────────────────

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status}`);
  return data;
}

async function syncStaticLocations() {
  // world.js הוא מקור האמת לשם/אזור/תיאור. מעתיקים לשרת פעם אחת בעלייה
  // כדי שהוא יוכל לבנות פרומפטים בלי הדפדפן.
  const payload = {};
  for (const [id, loc] of Object.entries(LOCATIONS)) {
    payload[id] = { name: loc.name, region: loc.region, description: loc.description };
  }
  await api('/api/editor/sync', { method: 'POST', body: JSON.stringify({ locations: payload }) });
}

async function loadWorld() {
  const data = await api('/api/editor/world');
  lore = data.lore;
  canon = Object.fromEntries(data.locations.map((l) => [l.location_id, l]));
  renderModelBadge(data.image_model);
  renderList();
  renderWorldForm();
}

// ── תצוגה ───────────────────────────────────────────────────────────────

function renderModelBadge(model) {
  if (model) {
    el.modelBadge.textContent = `מודל: ${model}`;
    el.modelBadge.className = 'ed-badge ed-badge-ok';
  } else {
    el.modelBadge.textContent = 'אין מודל תמונות פרוס';
    el.modelBadge.className = 'ed-badge ed-badge-error';
  }
  el.genBtn.disabled = !model;
}

function renderProgress() {
  const ids = Object.keys(LOCATIONS);
  const withImage = ids.filter((id) => canon[id]?.image_path).length;
  el.progressBadge.textContent = `${withImage} / ${ids.length} מצוירים`;
}

function renderList() {
  const term = el.filter.value.trim();
  const byRegion = new Map();
  for (const [id, loc] of Object.entries(LOCATIONS)) {
    if (term && !loc.name.includes(term) && !id.includes(term)) continue;
    if (!byRegion.has(loc.region)) byRegion.set(loc.region, []);
    byRegion.get(loc.region).push({ id, loc });
  }

  el.list.innerHTML = '';
  for (const region of REGION_ORDER) {
    const entries = byRegion.get(region);
    if (!entries) continue;
    const head = document.createElement('div');
    head.className = 'ed-region';
    head.textContent = REGION_NAMES[region] || region;
    el.list.appendChild(head);

    for (const { id, loc } of entries) {
      const rec = canon[id] || {};
      const btn = document.createElement('button');
      btn.className = 'ed-item' + (id === selectedId ? ' active' : '');
      btn.dataset.id = id;
      btn.innerHTML = `<span>${loc.name}</span>
        <span class="ed-dots">
          <span class="ed-dot ${rec.lore ? 'on-lore' : ''}" title="סיפור רקע"></span>
          <span class="ed-dot ${rec.image_path ? 'on-image' : ''}" title="תמונה"></span>
        </span>`;
      btn.addEventListener('click', () => select(id));
      el.list.appendChild(btn);
    }
  }
  renderProgress();
}

function renderDetail() {
  const loc = LOCATIONS[selectedId];
  const rec = canon[selectedId] || {};
  el.empty.classList.add('hidden');
  el.detail.classList.remove('hidden');
  el.name.textContent = loc.name;
  el.meta.textContent = `${REGION_NAMES[loc.region] || loc.region} · ${selectedId}`;
  el.description.value = rec.description || loc.description || '';
  el.lore.value = rec.lore || '';
  el.visual.value = rec.visual_notes || '';
  el.promptView.classList.add('hidden');
  setSaveState('');
  renderImage();
}

function renderImage() {
  const rec = canon[selectedId] || {};
  el.imgSpinner.classList.add('hidden');
  el.imgNote.className = 'ed-img-note';
  if (rec.image_path) {
    // מעקף מטמון: אותו נתיב נכתב מחדש בכל יצירה
    el.imgView.src = `${rec.image_path}?v=${encodeURIComponent(rec.updated_at || '')}`;
    el.imgView.classList.remove('hidden');
    el.imgPlaceholder.classList.add('hidden');
    el.imgNote.textContent = rec.updated_at
      ? `נוצר ${new Date(rec.updated_at).toLocaleString('he-IL')}`
      : '';
  } else {
    el.imgView.classList.add('hidden');
    el.imgPlaceholder.classList.remove('hidden');
    el.imgNote.textContent = '';
  }
}

function renderWorldForm() {
  el.wTitle.value = lore.title || '';
  el.wPremise.value = lore.premise || '';
  el.wStyle.value = lore.style_bible || '';
}

function setSaveState(text, isError = false) {
  el.saveState.textContent = text;
  el.saveState.style.color = isError ? '#a33' : '';
}

// ── פעולות ──────────────────────────────────────────────────────────────

async function select(id) {
  if (dirty && !confirm('יש שינויים שלא נשמרו. לעבור בכל זאת?')) return;
  dirty = false;
  selectedId = id;
  renderList();
  renderDetail();
}

async function saveLocation() {
  if (!selectedId) return;
  el.saveBtn.disabled = true;
  setSaveState('שומר…');
  try {
    const data = await api(`/api/editor/location/${selectedId}`, {
      method: 'POST',
      body: JSON.stringify({
        lore: el.lore.value,
        visual_notes: el.visual.value,
        description: el.description.value,
      }),
    });
    canon[selectedId] = data.location;
    dirty = false;
    setSaveState('נשמר');
    renderList();
  } catch (e) {
    setSaveState(`שגיאה: ${e.message}`, true);
  } finally {
    el.saveBtn.disabled = false;
  }
}

async function generateImage() {
  if (!selectedId) return;
  if (dirty) await saveLocation();
  el.genBtn.disabled = true;
  el.imgSpinner.classList.remove('hidden');
  el.imgNote.className = 'ed-img-note';
  el.imgNote.textContent = '';
  try {
    const data = await api(`/api/editor/location/${selectedId}/image`, {
      method: 'POST',
      body: JSON.stringify({}),
    });
    canon[selectedId] = {
      ...(canon[selectedId] || {}),
      image_path: data.image_path,
      image_prompt: data.prompt,
      updated_at: new Date().toISOString(),
    };
    renderImage();
    renderList();
  } catch (e) {
    el.imgSpinner.classList.add('hidden');
    el.imgNote.className = 'ed-img-note error';
    el.imgNote.textContent = `נכשל: ${e.message}`;
  } finally {
    el.genBtn.disabled = false;
  }
}

async function togglePrompt() {
  if (!el.promptView.classList.contains('hidden')) {
    el.promptView.classList.add('hidden');
    return;
  }
  if (dirty) await saveLocation();
  try {
    const data = await api(`/api/editor/location/${selectedId}/prompt`);
    el.promptView.textContent = data.prompt;
    el.promptView.classList.remove('hidden');
  } catch (e) {
    el.promptView.textContent = `שגיאה: ${e.message}`;
    el.promptView.classList.remove('hidden');
  }
}

async function saveWorldLore() {
  el.saveWorld.disabled = true;
  el.worldSaveState.textContent = 'שומר…';
  try {
    const data = await api('/api/editor/world', {
      method: 'POST',
      body: JSON.stringify({
        title: el.wTitle.value,
        premise: el.wPremise.value,
        style_bible: el.wStyle.value,
      }),
    });
    lore = data.lore;
    el.worldSaveState.textContent = 'נשמר';
  } catch (e) {
    el.worldSaveState.textContent = `שגיאה: ${e.message}`;
  } finally {
    el.saveWorld.disabled = false;
  }
}

// ── חיווט ───────────────────────────────────────────────────────────────

el.filter.addEventListener('input', renderList);
el.saveBtn.addEventListener('click', saveLocation);
el.genBtn.addEventListener('click', generateImage);
el.promptBtn.addEventListener('click', togglePrompt);
[el.lore, el.visual, el.description].forEach((node) => {
  node.addEventListener('input', () => { dirty = true; setSaveState('לא נשמר'); });
});

el.worldBtn.addEventListener('click', () => el.worldModal.classList.remove('hidden'));
el.closeWorld.addEventListener('click', () => el.worldModal.classList.add('hidden'));
el.worldModal.addEventListener('click', (e) => {
  if (e.target === el.worldModal) el.worldModal.classList.add('hidden');
});
el.saveWorld.addEventListener('click', saveWorldLore);

// ctrl+s לשמירה, ctrl+enter ליצירת תמונה — העורך הוא כלי עבודה
document.addEventListener('keydown', (e) => {
  if (!(e.ctrlKey || e.metaKey)) return;
  if (e.key === 's') { e.preventDefault(); saveLocation(); }
  if (e.key === 'Enter') { e.preventDefault(); generateImage(); }
});

window.addEventListener('beforeunload', (e) => {
  if (dirty) { e.preventDefault(); e.returnValue = ''; }
});

(async function boot() {
  try {
    await syncStaticLocations();
    await loadWorld();
  } catch (e) {
    el.modelBadge.textContent = `השרת לא זמין: ${e.message}`;
    el.modelBadge.className = 'ed-badge ed-badge-error';
  }
})();
