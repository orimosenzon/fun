// api.js - תקשורת עם Gemini API

import { SYSTEM_PROMPT, buildTurnPrompt } from './prompts.js';
import { logError, logWarn, logInfo } from './logger.js';

const SERVER_URL = 'http://localhost:8765';

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

// cache in-memory (base64 is too large for localStorage)
const imageCache = new Map();

async function callImagen(prompt, aspectRatio = '1:1') {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/imagen-3.0-generate-002:predict?key=${apiKey}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      instances: [{ prompt }],
      parameters: { sampleCount: 1, aspectRatio }
    })
  });
  if (!response.ok) {
    const err = await response.text();
    logError('Imagen failed', { status: response.status, body: err.slice(0, 300) });
    throw new Error(`Imagen error: ${response.status}`);
  }
  const data = await response.json();
  const b64 = data.predictions?.[0]?.bytesBase64Encoded;
  if (!b64) throw new Error('No image from Imagen');
  return `data:image/png;base64,${b64}`;
}

async function generateLocationImage(locationId, locationName, description) {
  if (imageCache.has(locationId)) return imageCache.get(locationId);

  // Check server for canon image
  const canon = await fetchCanonLocation(locationId);
  if (canon?.image_data) {
    imageCache.set(locationId, canon.image_data);
    return canon.image_data;
  }

  // Generate new image (first visitor)
  const prompt = `Fantasy illustration of a Middle-earth location: ${locationName}. ${description}. Tolkien style, painterly, warm lighting, no text, no people in foreground.`;
  const dataUrl = await callImagen(prompt, '1:1');
  imageCache.set(locationId, dataUrl);

  // Save image to server (fire-and-forget — narrative saved separately by game.js)
  saveCanonLocation(locationId, null, dataUrl);

  return dataUrl;
}

async function generateNPCPortrait(npcId, npcName) {
  const cacheKey = `npc_${npcId}`;
  if (imageCache.has(cacheKey)) return imageCache.get(cacheKey);

  const prompt = `Fantasy portrait of ${npcName}, Tolkien character, painterly style, dramatic lighting, no text, close-up face.`;
  const dataUrl = await callImagen(prompt, '1:1');
  imageCache.set(cacheKey, dataUrl);
  return dataUrl;
}

function getConversationHistory() {
  return conversationHistory;
}

function setConversationHistory(history) {
  conversationHistory = history;
}

export { initAPI, sendAction, generateLocationImage, generateNPCPortrait, getConversationHistory, setConversationHistory, getPlayerId, fetchCanonLocation, saveCanonLocation };
