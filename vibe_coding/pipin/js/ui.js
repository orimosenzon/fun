// ui.js - לוגיקת ממשק

import { ITEMS, NPCS } from './world.js';

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

  // Image
  const img = els.locationImage;
  img.classList.add('hidden');
  els.imagePlaceholder.classList.remove('hidden');

  const testImg = new Image();
  testImg.onload = () => {
    img.src = location.image;
    img.classList.remove('hidden');
    els.imagePlaceholder.classList.add('hidden');
  };
  testImg.onerror = () => {
    img.classList.add('hidden');
    els.imagePlaceholder.classList.remove('hidden');
  };
  testImg.src = location.image;

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
           onerror="this.style.display='none'">
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
         onerror="this.style.display='none'">
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
