// game.js - מנוע המשחק הראשי

import { LOCATIONS, ITEMS, NPCS } from './world.js';
import { initAPI, sendAction, generateLocationImage, getGuestLocationImage, getConversationHistory, setConversationHistory, fetchCanonLocation, saveCanonLocation, fetchPlayerStats } from './api.js';
import * as UI from './ui.js';

let isGuest = false;

// ═══════════════════════════════
//   מצב המשחק
// ═══════════════════════════════

let gameState = {
  currentLocation: 'hobbiton',
  inventory: ['pipe_weed', 'walking_stick'],
  visitedLocations: ['hobbiton'],
  metNPCs: [],
  events: [],
  turnCount: 0,
  health: 'טוב',
};

// ═══════════════════════════════
//   אתחול
// ═══════════════════════════════

function init() {
  UI.initInventoryModal();

  UI.onStartGame((apiKey, loadSave) => {
    localStorage.setItem('pipin_api_key', apiKey);
    initAPI(apiKey);

    if (loadSave) {
      loadGame();
    }

    UI.hideStartScreen();
    renderCurrentLocation();
    sendInitialPrompt();
    refreshPlayerStats();
  });

  UI.onGuestStart(() => {
    isGuest = true;
    UI.setGuestMode();
    UI.hideStartScreen();
    renderGuestLocation();
  });

  UI.onCompassClick(handleDirection);

  UI.onSubmit(handlePlayerAction);
}

// ═══════════════════════════════
//   רינדור מקום נוכחי
// ═══════════════════════════════

function renderCurrentLocation() {
  const loc = LOCATIONS[gameState.currentLocation];
  if (!loc) return;

  UI.updateLocation(loc);
  UI.updateCompass(loc.exits);
  UI.updateInventory(gameState.inventory);

  // Find NPCs at current location
  const npcsHere = getNPCsAtLocation(gameState.currentLocation);
  UI.updateNPCs(npcsHere, handleNPCClick);

  // Load image from cache or generate (non-blocking)
  loadLocationImage(loc);

  // Update map
  UI.updateMap(gameState.visitedLocations, gameState.currentLocation);
}

async function loadLocationImage(loc) {
  try {
    const src = await generateLocationImage(loc.id, loc.name, loc.description);
    UI.setLocationImage(src);
  } catch (err) {
    console.warn('Image generation failed:', err);
  }
}

async function renderGuestLocation() {
  const loc = LOCATIONS[gameState.currentLocation];
  if (!loc) return;

  UI.updateLocation(loc);
  UI.updateCompass(loc.exits);

  // Image via Pollinations — no server save
  UI.setLocationImage(getGuestLocationImage(loc.id, loc.name, loc.description));

  UI.updateMap(gameState.visitedLocations, gameState.currentLocation);

  // Show canon narrative if exists, else unexplored message
  const canon = await fetchCanonLocation(loc.id);
  if (canon?.narrative) {
    UI.addNarration(`📜 ${canon.narrative}`);
  } else {
    UI.addNarration('מקום זה טרם נחקר. שחקן עם מפתח Gemini API יוכל לחקור אותו לראשונה.');
  }
}

function getNPCsAtLocation(locationId) {
  const loc = LOCATIONS[locationId];
  if (!loc) return [];
  return loc.npcs.filter(id => NPCS[id]);
}

// ═══════════════════════════════
//   פעולות שחקן
// ═══════════════════════════════

function handleDirection(dir) {
  const loc = LOCATIONS[gameState.currentLocation];
  const target = loc.exits[dir];
  if (!target || !LOCATIONS[target]) return;

  if (isGuest) {
    gameState.currentLocation = target;
    if (!gameState.visitedLocations.includes(target)) {
      gameState.visitedLocations.push(target);
    }
    LOCATIONS[target].visited = true;
    renderGuestLocation();
    return;
  }

  const dirNames = { north: 'צפונה', south: 'דרומה', east: 'מזרחה', west: 'מערבה' };
  handlePlayerAction(`הולך ${dirNames[dir]} אל ${LOCATIONS[target].name}`);
}

function handleNPCClick(npc) {
  UI.setInputText(`דבר עם ${npc.name}`);
}

async function handlePlayerAction(action) {
  UI.setInputEnabled(false);
  UI.showLoading();
  UI.addPlayerAction(action);

  const loc = LOCATIONS[gameState.currentLocation];
  const npcsHere = getNPCsAtLocation(gameState.currentLocation).map(id => NPCS[id]);

  try {
    const response = await sendAction(gameState, action, loc, npcsHere);
    await applyResponse(response);
  } catch (err) {
    console.error('Error:', err);
    if (err.message.includes('429')) {
      UI.addNarration('גנדלף מניח יד על כתפך: "סבלנות, חביבי — הכוחות הקסומים זקוקים לרגע מנוחה. נסה שוב עוד דקה."');
    } else {
      UI.addNarration('משהו השתבש... נסה שוב.');
    }
  }

  UI.setInputEnabled(true);
  UI.hideLoading();
  gameState.turnCount++;

  // Auto-save every 5 turns
  if (gameState.turnCount % 5 === 0) {
    saveGame();
  }
}

// ═══════════════════════════════
//   Canon location (shared world)
// ═══════════════════════════════

async function refreshPlayerStats() {
  const stats = await fetchPlayerStats();
  const count = stats.locations_canonized;
  const el = document.getElementById('player-stats');
  const countEl = document.getElementById('player-location-count');
  if (el && countEl) {
    countEl.textContent = count;
    el.classList.toggle('hidden', count === 0);
    el.title = `${count} מקומות שחקרת בפעם הראשונה`;
  }
}

async function handleFirstVisit(locationId, freshNarrative) {
  const canon = await fetchCanonLocation(locationId);
  if (canon?.narrative) {
    UI.addNarration(`📜 ${canon.narrative}`);
  } else if (freshNarrative) {
    saveCanonLocation(locationId, freshNarrative, null);
    refreshPlayerStats();
  }
}

// ═══════════════════════════════
//   טיפול בתגובת AI
// ═══════════════════════════════

async function applyResponse(response) {
  // Description
  if (response.description) {
    UI.addNarration(response.description);
  }

  // Dialogue
  if (response.dialogue) {
    const speaker = response.dialogue.speaker;
    const text = response.dialogue.text;
    if (speaker && text) {
      UI.addDialogue(speaker, text);
    }
  }

  // State changes
  const changes = response.state_changes;
  if (changes) {
    // Location change
    if (changes.location && LOCATIONS[changes.location]) {
      const isNewLocation = !gameState.visitedLocations.includes(changes.location);
      gameState.currentLocation = changes.location;
      if (isNewLocation) {
        gameState.visitedLocations.push(changes.location);
        await handleFirstVisit(changes.location, response.description);
      }
      LOCATIONS[changes.location].visited = true;
      renderCurrentLocation();
    }

    // Inventory add
    if (changes.inventory_add) {
      changes.inventory_add.forEach(itemId => {
        if (ITEMS[itemId] && !gameState.inventory.includes(itemId)) {
          gameState.inventory.push(itemId);
        }
      });
      UI.updateInventory(gameState.inventory);
    }

    // Inventory remove
    if (changes.inventory_remove) {
      changes.inventory_remove.forEach(itemId => {
        const idx = gameState.inventory.indexOf(itemId);
        if (idx !== -1) gameState.inventory.splice(idx, 1);
      });
      UI.updateInventory(gameState.inventory);
    }

    // Event
    if (changes.event) {
      gameState.events.push(changes.event);
    }

    // NPC met
    if (changes.npc_met && !gameState.metNPCs.includes(changes.npc_met)) {
      gameState.metNPCs.push(changes.npc_met);
    }
  }

  // Suggestions
  if (response.options) {
    UI.updateSuggestions(response.options, handlePlayerAction);
  }
}

// ═══════════════════════════════
//   פרומפט ראשוני
// ═══════════════════════════════

async function sendInitialPrompt() {
  UI.setInputEnabled(false);
  UI.showLoading();

  const loc = LOCATIONS[gameState.currentLocation];
  const npcsHere = getNPCsAtLocation(gameState.currentLocation).map(id => NPCS[id]);

  try {
    const response = await sendAction(
      gameState,
      'פיפין מתעורר בבוקר חדש בהוביטון. תאר את הסצנה הפותחת.',
      loc,
      npcsHere
    );
    applyResponse(response);
  } catch (err) {
    console.error('Error:', err);
    // Fallback - show static description
    UI.addNarration(loc.description);
    UI.updateSuggestions(['הסתכל מסביב', 'דבר עם גנדלף', 'לך מזרחה'], handlePlayerAction);
  }

  UI.setInputEnabled(true);
  UI.hideLoading();
}

// ═══════════════════════════════
//   שמירה/טעינה
// ═══════════════════════════════

function saveGame() {
  const save = {
    gameState,
    conversationHistory: getConversationHistory(),
    timestamp: Date.now()
  };
  localStorage.setItem('pipin_save', JSON.stringify(save));
}

function loadGame() {
  const raw = localStorage.getItem('pipin_save');
  if (!raw) return;

  try {
    const save = JSON.parse(raw);
    gameState = save.gameState;
    if (save.conversationHistory) {
      setConversationHistory(save.conversationHistory);
    }
  } catch (e) {
    console.error('Failed to load save:', e);
  }
}

// ═══════════════════════════════
//   התחלה
// ═══════════════════════════════

init();
