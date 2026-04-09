// api.js - תקשורת עם Gemini API

import { SYSTEM_PROMPT, buildTurnPrompt } from './prompts.js';

let apiKey = '';
let conversationHistory = [];

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
    if (!response.ok) lastErr = await response.text();
    if (response.ok) break;
    if (response.status !== 429) throw new Error(`Gemini API error: ${response.status} - ${lastErr}`);
  }

  if (!response.ok) {
    throw new Error(`Gemini API error: ${response.status} - ${lastErr}`);
  }

  const data = await response.json();

  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) {
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

async function generateLocationImage(locationId, locationName, description) {
  const cacheKey = `pipin_img_${locationId}`;
  const cached = localStorage.getItem(cacheKey);
  if (cached) return cached;

  const prompt = `Fantasy illustration of a Middle-earth location: ${locationName}. ${description}. Tolkien style, painterly, warm lighting, no text, no people in foreground.`;
  const encodedPrompt = encodeURIComponent(prompt);
  const url = `https://image.pollinations.ai/prompt/${encodedPrompt}?width=512&height=512&nologo=true&seed=${locationId.split('').reduce((a, c) => a + c.charCodeAt(0), 0)}`;

  // Return the URL directly — Pollinations serves images as URLs, no need to fetch/b64
  localStorage.setItem(cacheKey, url);
  return url;
}

function getConversationHistory() {
  return conversationHistory;
}

function setConversationHistory(history) {
  conversationHistory = history;
}

export { initAPI, sendAction, generateLocationImage, getConversationHistory, setConversationHistory };
