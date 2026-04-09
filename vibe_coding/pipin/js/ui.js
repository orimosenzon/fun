// ui.js - לוגיקת ממשק

import { ITEMS, NPCS, LOCATIONS } from './world.js';

// fallback גלובלי לתמונות NPC חסרות
window.npcImgFallback = function(el) {
  el.onerror = null;
  const name = el.alt || 'Tolkien character';
  const prompt = 'Fantasy portrait of ' + name + ', Tolkien character, painterly style, no text';
  el.src = 'https://image.pollinations.ai/prompt/' + encodeURIComponent(prompt) + '?width=128&height=128&nologo=true';
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// אלמנטים
const els = {
  topbarLocation: $('#topbar-location'),
  locationImage: $('#location-image'),
  imagePlaceholder: $('#location-image-placeholder'),
  npcsBar: $('#npcs-bar'),
  locationName: $('#location-name'),
  locationDescription: $('#location-description'),
  ambientText: $('#ambient-text'),
  dialogueArea: $('#dialogue-area'),
  suggestions: $('#suggestions'),
  inputForm: $('#input-form'),
  userInput: $('#user-input'),
  sendBtn: $('.send-btn'),
  inventoryBtn: $('#inventory-btn'),
  inventoryCount: $('#inventory-count'),
  inventoryModal: $('#inventory-modal'),
  closeInventory: $('#close-inventory'),
  inventoryList: $('#inventory-list'),
  worldMap: $('#world-map'),
  loading: $('#loading'),
  startScreen: $('#start-screen'),
  startBtn: $('#start-btn'),
  loadBtn: $('#load-btn'),
  apiKeyInput: $('#api-key-input'),
};

// ═══════════════════════════════
//   עדכון מקום
// ═══════════════════════════════

function updateLocation(location) {
  // Header
  els.topbarLocation.textContent = location.name;

  // Image — show placeholder until generated image arrives
  els.locationImage.classList.add('hidden');
  els.imagePlaceholder.classList.remove('hidden');

  // Text
  els.locationName.textContent = location.name;
  els.locationName.classList.add('fade-in');
  els.locationDescription.textContent = location.description;
  els.locationDescription.classList.add('fade-in');
  els.ambientText.textContent = location.ambient;

  // Clear dialogue
  els.dialogueArea.innerHTML = '';

  // Reset fade-in
  setTimeout(() => {
    els.locationName.classList.remove('fade-in');
    els.locationDescription.classList.remove('fade-in');
  }, 400);
}

function setLocationImage(src) {
  els.locationImage.src = src;
  els.locationImage.classList.remove('hidden');
  els.imagePlaceholder.classList.add('hidden');
}

// ═══════════════════════════════
//   שושנת רוחות
// ═══════════════════════════════

function updateCompass(exits) {
  const dirs = ['north', 'south', 'east', 'west'];
  const btnIds = ['compass-n', 'compass-s', 'compass-e', 'compass-w'];

  dirs.forEach((dir, i) => {
    const btn = $(`#${btnIds[i]}`);
    if (exits[dir]) {
      btn.disabled = false;
    } else {
      btn.disabled = true;
    }
  });
}

function onCompassClick(callback) {
  $$('.compass-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (!btn.disabled) {
        callback(btn.dataset.dir);
      }
    });
  });
}

// ═══════════════════════════════
//   NPCs
// ═══════════════════════════════

function updateNPCs(npcIds, onNPCClick) {
  els.npcsBar.innerHTML = '';
  npcIds.forEach(id => {
    const npc = NPCS[id];
    if (!npc) return;

    const avatar = document.createElement('div');
    avatar.className = 'npc-avatar';
    avatar.innerHTML = `
      <img class="npc-avatar-img" src="${npc.portrait}" alt="${npc.name}"
           onerror="npcImgFallback(this)">
      <span class="npc-avatar-name">${npc.name}</span>
    `;
    avatar.addEventListener('click', () => onNPCClick(npc));
    els.npcsBar.appendChild(avatar);
  });
}

// ═══════════════════════════════
//   דיאלוגים
// ═══════════════════════════════

function addDialogue(speakerNpcId, text) {
  const npc = NPCS[speakerNpcId];
  const bubble = document.createElement('div');
  bubble.className = 'dialogue-bubble';
  bubble.innerHTML = `
    <img class="dialogue-portrait" src="${npc ? npc.portrait : ''}" alt="${npc ? npc.name : ''}"
         onerror="npcImgFallback(this)">
    <div class="dialogue-content">
      <div class="dialogue-speaker">${npc ? npc.name : ''}</div>
      <div class="dialogue-text">${text}</div>
    </div>
  `;
  els.dialogueArea.appendChild(bubble);
  scrollTextToBottom();
}

function addNarration(text) {
  const p = document.createElement('p');
  p.className = 'location-description fade-in';
  p.textContent = text;
  els.dialogueArea.appendChild(p);
  scrollTextToBottom();
}

function addPlayerAction(text) {
  const div = document.createElement('div');
  div.className = 'player-action';
  div.textContent = `> ${text}`;
  els.dialogueArea.appendChild(div);
  scrollTextToBottom();
}

function scrollTextToBottom() {
  const panel = els.dialogueArea.closest('.text-panel');
  setTimeout(() => {
    panel.scrollTop = panel.scrollHeight;
  }, 50);
}

// ═══════════════════════════════
//   הצעות פעולה
// ═══════════════════════════════

function updateSuggestions(options, onSuggestionClick) {
  els.suggestions.innerHTML = '';
  if (!options || options.length === 0) return;

  options.forEach(text => {
    const chip = document.createElement('button');
    chip.className = 'suggestion-chip';
    chip.textContent = text;
    chip.addEventListener('click', () => onSuggestionClick(text));
    els.suggestions.appendChild(chip);
  });
}

// ═══════════════════════════════
//   תרמיל
// ═══════════════════════════════

function updateInventory(inventoryIds) {
  els.inventoryCount.textContent = inventoryIds.length;

  els.inventoryList.innerHTML = '';
  if (inventoryIds.length === 0) {
    els.inventoryList.innerHTML = '<div class="inventory-empty">התרמיל ריק</div>';
    return;
  }

  inventoryIds.forEach(id => {
    const item = ITEMS[id];
    if (!item) return;

    const div = document.createElement('div');
    div.className = 'inventory-item';
    div.innerHTML = `
      <span class="inventory-item-icon">${item.icon}</span>
      <div class="inventory-item-info">
        <div class="inventory-item-name">${item.name}</div>
        <div class="inventory-item-desc">${item.description}</div>
      </div>
    `;
    els.inventoryList.appendChild(div);
  });
}

function initInventoryModal() {
  els.inventoryBtn.addEventListener('click', () => {
    els.inventoryModal.classList.remove('hidden');
  });
  els.closeInventory.addEventListener('click', () => {
    els.inventoryModal.classList.add('hidden');
  });
  els.inventoryModal.addEventListener('click', (e) => {
    if (e.target === els.inventoryModal) {
      els.inventoryModal.classList.add('hidden');
    }
  });
}

// ═══════════════════════════════
//   קלט
// ═══════════════════════════════

function onSubmit(callback) {
  els.inputForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = els.userInput.value.trim();
    if (!text) return;
    els.userInput.value = '';
    callback(text);
  });
}

function setInputText(text) {
  els.userInput.value = text;
  els.userInput.focus();
}

function setInputEnabled(enabled) {
  els.userInput.disabled = !enabled;
  els.sendBtn.disabled = !enabled;
}

// ═══════════════════════════════
//   Loading
// ═══════════════════════════════

// ═══════════════════════════════
//   מפה
// ═══════════════════════════════

function updateMap(visitedLocations, currentLocationId) {
  const svg = els.worldMap;
  svg.innerHTML = '';

  if (!visitedLocations || visitedLocations.length === 0) return;

  // BFS to compute grid coordinates from 'hobbiton'
  const coords = {};
  const dirOffset = { north: [0, -1], south: [0, 1], east: [1, 0], west: [-1, 0] };
  const queue = [{ id: 'hobbiton', x: 0, y: 0 }];
  const seen = new Set(['hobbiton']);
  coords['hobbiton'] = { x: 0, y: 0 };

  while (queue.length) {
    const { id, x, y } = queue.shift();
    const loc = LOCATIONS[id];
    if (!loc) continue;
    for (const [dir, targetId] of Object.entries(loc.exits)) {
      if (targetId && !seen.has(targetId)) {
        seen.add(targetId);
        const [dx, dy] = dirOffset[dir];
        coords[targetId] = { x: x + dx, y: y + dy };
        queue.push({ id: targetId, x: x + dx, y: y + dy });
      }
    }
  }

  const visitedSet = new Set(visitedLocations);
  const visCoords = visitedLocations.filter(id => coords[id]).map(id => coords[id]);
  if (visCoords.length === 0) return;

  const minX = Math.min(...visCoords.map(c => c.x));
  const maxX = Math.max(...visCoords.map(c => c.x));
  const minY = Math.min(...visCoords.map(c => c.y));
  const maxY = Math.max(...visCoords.map(c => c.y));

  const W = 180, H = 320, pad = 28, cell = 40;
  const totalW = Math.max((maxX - minX) * cell + pad * 2, W);
  const totalH = Math.max((maxY - minY) * cell + pad * 2, H);

  svg.setAttribute('viewBox', `0 0 ${totalW} ${totalH}`);

  const toSVG = (gx, gy) => ({
    x: pad + (gx - minX) * cell,
    y: pad + (gy - minY) * cell
  });

  const ns = 'http://www.w3.org/2000/svg';

  // Lines first (behind nodes)
  visitedLocations.forEach(id => {
    const loc = LOCATIONS[id];
    if (!loc || !coords[id]) return;
    const from = toSVG(coords[id].x, coords[id].y);
    Object.values(loc.exits).forEach(targetId => {
      if (targetId && visitedSet.has(targetId) && coords[targetId]) {
        const to = toSVG(coords[targetId].x, coords[targetId].y);
        const line = document.createElementNS(ns, 'line');
        line.setAttribute('x1', from.x); line.setAttribute('y1', from.y);
        line.setAttribute('x2', to.x);   line.setAttribute('y2', to.y);
        line.setAttribute('stroke', '#d4cdbf');
        line.setAttribute('stroke-width', '1.5');
        svg.appendChild(line);
      }
    });
  });

  // Nodes
  visitedLocations.forEach(id => {
    if (!coords[id]) return;
    const { x, y } = toSVG(coords[id].x, coords[id].y);
    const isCurrent = id === currentLocationId;
    const loc = LOCATIONS[id];

    const g = document.createElementNS(ns, 'g');

    if (isCurrent) {
      const pulse = document.createElementNS(ns, 'circle');
      pulse.setAttribute('cx', x); pulse.setAttribute('cy', y);
      pulse.setAttribute('r', 12);
      pulse.setAttribute('fill', '#e8f0eb');
      g.appendChild(pulse);
    }

    const circle = document.createElementNS(ns, 'circle');
    circle.setAttribute('cx', x); circle.setAttribute('cy', y);
    circle.setAttribute('r', isCurrent ? 7 : 5);
    circle.setAttribute('fill', isCurrent ? '#3d6b4f' : '#c9a84c');
    circle.setAttribute('stroke', '#fff');
    circle.setAttribute('stroke-width', '1.5');
    g.appendChild(circle);

    const label = document.createElementNS(ns, 'text');
    label.setAttribute('x', x);
    label.setAttribute('y', y + (isCurrent ? 22 : 18));
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('font-size', '8');
    label.setAttribute('fill', isCurrent ? '#3d6b4f' : '#7a6e5d');
    label.setAttribute('font-family', 'Heebo, sans-serif');
    label.setAttribute('font-weight', isCurrent ? '600' : '400');
    label.textContent = loc ? loc.name : id;
    g.appendChild(label);

    svg.appendChild(g);
  });
}

function showLoading() {
  els.loading.classList.remove('hidden');
}

function hideLoading() {
  els.loading.classList.add('hidden');
}

// ═══════════════════════════════
//   מסך פתיחה
// ═══════════════════════════════

function onStartGame(callback) {
  els.startBtn.addEventListener('click', () => {
    const apiKey = els.apiKeyInput.value.trim();
    if (!apiKey) {
      els.apiKeyInput.style.borderColor = '#e74c3c';
      els.apiKeyInput.focus();
      return;
    }
    callback(apiKey, false);
  });

  els.loadBtn.addEventListener('click', () => {
    const apiKey = els.apiKeyInput.value.trim();
    if (!apiKey) {
      els.apiKeyInput.style.borderColor = '#e74c3c';
      els.apiKeyInput.focus();
      return;
    }
    callback(apiKey, true);
  });

  // Check if saved game exists
  if (localStorage.getItem('pipin_save')) {
    els.loadBtn.classList.remove('hidden');
  }

  // Restore saved API key
  const savedKey = localStorage.getItem('pipin_api_key');
  if (savedKey) {
    els.apiKeyInput.value = savedKey;
  }
}

function hideStartScreen() {
  els.startScreen.classList.add('hidden');
}

export {
  updateLocation,
  setLocationImage,
  updateMap,
  updateCompass,
  onCompassClick,
  updateNPCs,
  addDialogue,
  addNarration,
  addPlayerAction,
  updateSuggestions,
  updateInventory,
  initInventoryModal,
  onSubmit,
  setInputText,
  setInputEnabled,
  showLoading,
  hideLoading,
  onStartGame,
  hideStartScreen,
};
