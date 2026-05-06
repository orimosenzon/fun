// api.js - תקשורת עם Gemini API

import { SYSTEM_PROMPT, buildTurnPrompt } from './prompts.js';
import { logError, logWarn, logInfo } from './logger.js';

// Empty string = relative URLs, works both locally (flask serves the page) and on Render
const SERVER_URL = '';

let apiKey = '';
let conversationHistory = [];

function getPlayerId() {
  let id = localStorage.getItem('pipin_player_id');
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem('pipin_player_id', id);
  }
  return id;
}

// ── Canon location (shared world) ───────────────────────────────────────

async function fetchCanonLocation(locationId) {
  try {
    const res = await fetch(`${SERVER_URL}/api/world/location/${locationId}`);
    if (res.ok) {
      const data = await res.json();
      if (data.found) return data;
    }
  } catch (_) { /* server unavailable — graceful fallback */ }
  return null;
}

async function fetchPlayerStats() {
  try {
    const res = await fetch(`${SERVER_URL}/api/world/player/${getPlayerId()}/stats`);
    if (res.ok) return await res.json();
  } catch (_) { /* server unavailable */ }
  return { locations_canonized: 0, location_ids: [] };
}

async function saveCanonLocation(locationId, narrative, imageData) {
  try {
    await fetch(`${SERVER_URL}/api/world/location/${locationId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ player_id: getPlayerId(), narrative, image_data: imageData }),
    });
  } catch (_) { /* server unavailable — silent */ }
}

function initAPI(key) {
  apiKey = key;
  conversationHistory = [];
}

async function sendAction(gameState, action, locationData, npcData) {
  const turnPrompt = buildTurnPrompt(gameState, action, locationData, npcData);

  // Keep conversation history manageable (last 20 turns)
  if (conversationHistory.length > 40) {
    conversationHistory = conversationHistory.slice(-20);
  }

  conversationHistory.push({
    role: 'user',
    parts: [{ text: turnPrompt }]
  });

  const requestBody = {
    system_instruction: {
      parts: [{ text: SYSTEM_PROMPT }]
    },
    contents: conversationHistory,
    generationConfig: {
      temperature: 0.9,
      topP: 0.95,
      maxOutputTokens: 1024,
      responseMimeType: "application/json",
    }
  };

  const models = ['gemini-2.0-flash-lite', 'gemini-2.0-flash'];
  let response, lastErr;

  for (const model of models) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });
    if (!response.ok) {
      lastErr = await response.text();
      logWarn(`Gemini ${model} failed`, { status: response.status, model, body: lastErr.slice(0, 300) });
    }
    if (response.ok) break;
    if (response.status !== 429) throw new Error(`Gemini API error: ${response.status} - ${lastErr}`);
  }

  if (!response.ok) {
    logError(`All Gemini models failed`, { status: response.status, body: lastErr?.slice(0, 300) });
    throw new Error(`Gemini API error: ${response.status} - ${lastErr}`);
  }

  const data = await response.json();

  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) {
    logError('No text in Gemini response', { data: JSON.stringify(data).slice(0, 300) });
    throw new Error('No response from Gemini');
  }

  // Add AI response to history
  conversationHistory.push({
    role: 'model',
    parts: [{ text }]
  });

  // Parse JSON response
  const parsed = JSON.parse(text);
  return parsed;
}

// cache in-memory
const imageCache = new Map();

function buildPollinationsUrl(prompt, width = 512, height = 512) {
  const encoded = encodeURIComponent(prompt);
  return `https://image.pollinations.ai/prompt/${encoded}?width=${width}&height=${height}&nologo=true&model=flux`;
}

async function generateLocationImage(locationId, locationName, description) {
  if (imageCache.has(locationId)) return imageCache.get(locationId);

  // Check server for canon image
  const canon = await fetchCanonLocation(locationId);
  if (canon?.image_data) {
    imageCache.set(locationId, canon.image_data);
    return canon.image_data;
  }

  // Generate new image via Pollinations (no API key needed)
  const prompt = `Fantasy illustration of a Middle-earth location: ${locationName}. ${description}. Tolkien style, Alan Lee watercolor, painterly, warm lighting, no text, no people in foreground.`;
  const imageUrl = buildPollinationsUrl(prompt, 512, 512);
  imageCache.set(locationId, imageUrl);

  // Save image URL to server (fire-and-forget)
  saveCanonLocation(locationId, null, imageUrl);

  return imageUrl;
}

async function generateNPCPortrait(npcId, npcName) {
  const cacheKey = `npc_${npcId}`;
  if (imageCache.has(cacheKey)) return imageCache.get(cacheKey);

  const prompt = `Fantasy portrait of ${npcName}, Tolkien character, Alan Lee style, painterly, dramatic lighting, no text, close-up face.`;
  const imageUrl = buildPollinationsUrl(prompt, 256, 256);
  imageCache.set(cacheKey, imageUrl);
  return imageUrl;
}

// Guest mode — returns Pollinations URL with no server interaction
function getGuestLocationImage(locationId, locationName, description) {
  if (imageCache.has(locationId)) return imageCache.get(locationId);
  const prompt = `Fantasy illustration of a Middle-earth location: ${locationName}. ${description}. Tolkien style, Alan Lee watercolor, painterly, warm lighting, no text, no people in foreground.`;
  const url = buildPollinationsUrl(prompt, 512, 512);
  imageCache.set(locationId, url);
  return url;
}

function getConversationHistory() {
  return conversationHistory;
}

function setConversationHistory(history) {
  conversationHistory = history;
}

export { initAPI, sendAction, generateLocationImage, generateNPCPortrait, getGuestLocationImage, getConversationHistory, setConversationHistory, getPlayerId, fetchCanonLocation, saveCanonLocation, fetchPlayerStats };
