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

  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody)
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`Gemini API error: ${response.status} - ${err}`);
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

function getConversationHistory() {
  return conversationHistory;
}

function setConversationHistory(history) {
  conversationHistory = history;
}

export { initAPI, sendAction, getConversationHistory, setConversationHistory };
