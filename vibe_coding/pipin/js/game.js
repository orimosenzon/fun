// game.js - מנוע המשחק הראשי

import { LOCATIONS, ITEMS, NPCS } from './world.js';
import { initAPI, sendAction, generateLocationImage, getConversationHistory, setConversationHistory } from './api.js';
import * as UI from './ui.js';

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
    // placeholder stays visible — no crash
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
    applyResponse(response);
  } catch (err) {
    console.error('Error:', err);
    UI.addNarration('משהו השתבש... נסה שוב.');
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
//   טיפול בתגובת AI
// ═══════════════════════════════

function applyResponse(response) {
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
      gameState.currentLocation = changes.location;
      if (!gameState.visitedLocations.includes(changes.location)) {
        gameState.visitedLocations.push(changes.location);
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
