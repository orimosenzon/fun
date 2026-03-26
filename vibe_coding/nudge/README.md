# Nudge — Personal Finance Companion

> A gentle nudge in the right direction. No judgment, just support.

## Vision

Nudge is an Android app for personal income and expense management, built around a conversational interface. Instead of manually filling forms and spreadsheets, the user simply *talks* to the app — in whatever language feels natural — and Nudge takes care of the rest.

The core philosophy: **supportive, not judgmental**. Most finance apps make you feel guilty. Nudge learns your life — your income, commitments, constraints, and goals — and helps you navigate with empathy and practical advice.

This is a multi-user product, not a personal tool.

---

## Core Features (Planned)

### 1. Natural Language Input
- "קניתי קפה ב-20 שקל"
- "Got paid today, 12,000"
- "Bought groceries, around 300"

The user speaks or types freely. Nudge parses intent, amount, category, and date automatically.

### 2. Voice Interface
- Primary input: voice (speech-to-text)
- Fallback: text input
- Supports Hebrew, Arabic, Russian, English (and more via Claude)

**Important:** Claude API is text-only. Voice requires a separate STT/TTS pipeline (see Architecture).

### 3. Contextual Memory
Nudge learns and remembers:
- Recurring income (salary dates, freelance patterns)
- Fixed commitments (rent, mortgage, subscriptions)
- Personal constraints ("I'm on a tight budget this month")
- Financial goals ("saving for a trip")

### 4. Supportive AI Advisor
- Proactive nudges: "You're on track this week"
- Gentle alerts: "Looks like you're close to your dining budget — here's a way to balance it out"
- Tone: warm, practical, never shaming

### 5. Financial Overview
- Monthly summary (income vs. expenses)
- Category breakdown
- Trend insights ("Your grocery spend is 20% lower than last month")

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Mobile | Flutter (Dart) | Android-first, iOS-ready later |
| AI / Conversation | Claude API (Anthropic) | Text-only: NLU, structured extraction, advice |
| Speech-to-Text (STT) | Android SpeechRecognizer | Built-in, free, good Hebrew support |
| Text-to-Speech (TTS) | Android TextToSpeech | Built-in, free — upgrade to Google Cloud TTS if quality insufficient |
| Local Storage | SQLite via `sqflite` | Privacy-first, works offline |
| Cloud Sync (optional) | Supabase | Auth + DB + edge functions if multi-device needed |
| Backend (optional) | Supabase Edge Functions | Claude API calls from server side if needed |

### Why Flutter?
- Single codebase for Android (and iOS later)
- Rich UI components, smooth animations
- Large ecosystem, good Claude/Supabase SDKs

### Why Claude API?
- Best-in-class multilingual understanding (Hebrew, Arabic, Russian, English)
- Can be precisely instructed with a system prompt for tone and behavior
- Long context window — can hold financial history in a conversation
- Excellent at structured extraction (amount, category, date) from casual speech

### STT / TTS alternatives

| Service | Hebrew | Cost |
|---|---|---|
| Android SpeechRecognizer | Good | Free |
| Google Cloud STT | Excellent | Free up to 60 min/month, then ~$0.006/15 sec |
| OpenAI Whisper API | Very good | ~$0.006/min |
| Android TextToSpeech | Fair (robotic) | Free |
| Google Cloud TTS | Good | Free up to 4M chars/month |

**Recommendation:** start with Android built-ins; upgrade to Google Cloud if quality is insufficient.

---

## Architecture Overview

Claude is **text-in, text-out**. Voice requires a three-stage pipeline:

```
User voice
    │
    ▼ [STT — Android SpeechRecognizer]
  text
    │
    ▼ [Claude API]
  structured response + advice text
    │
    ├── saves transaction → SQLite
    │
    ▼ [TTS — Android TextToSpeech]
User hears response
```

The **system prompt** given to Claude includes:
- The user's known income sources and amounts
- Recurring commitments (rent, subscriptions, etc.)
- Current month budget targets
- Tone instructions (supportive, practical, no judgment)
- Recent transaction history (summarized — see Context Management below)

---

## Cost Analysis

### Per-user monthly estimate

Assumptions for a typical active user:
- 5 transactions logged per day
- 1–2 advice requests per day
- ~7 Claude API calls/day → ~210 calls/month
- ~2,500 tokens per call (system prompt + history + message + response)

| Scenario | Cost / user / month |
|---|---|
| Claude Haiku + Android STT | **~$0.45** |
| Claude Sonnet + Android STT | **~$1.70** |
| Claude Sonnet + Whisper API | **~$1.80** |

### Business model implication

A subscription of **$3–5/month per user** comfortably covers costs and leaves margin.

### Claude API pricing reference (verify current rates at console.anthropic.com)

| Model | Input | Output |
|---|---|---|
| Claude Haiku | ~$0.80 / 1M tokens | ~$4 / 1M tokens |
| Claude Sonnet | ~$3 / 1M tokens | ~$15 / 1M tokens |

---

## Context Management (Key Technical Challenge)

As a user accumulates months of history, the system prompt grows — and so does the cost per call. Without management, costs could double or triple over time.

**Strategy:**
- Summarize old transactions periodically rather than keeping raw history
- Store only recent transactions (last 30–60 days) in the prompt
- Keep a compact "financial profile" (commitments, goals, patterns) as a separate summary
- Use RAG (retrieval-augmented generation) for deeper history if needed

This is one of the most important architectural decisions in the project.

---

## Project Status

- [x] Concept defined
- [x] Tech stack chosen
- [x] Project named: **Nudge**
- [x] Cost analysis completed
- [ ] Flutter project scaffold
- [ ] Basic chat UI
- [ ] Claude API integration
- [ ] Transaction parsing (NLP → structured data)
- [ ] Local SQLite storage
- [ ] Voice input (STT + TTS)
- [ ] Financial overview screen
- [ ] User profile / commitments setup
- [ ] Context management / summarization
- [ ] Proactive nudges / alerts

---

## Design Principles

1. **Zero friction** — logging an expense should take under 5 seconds
2. **Multilingual by default** — the user speaks their language, not the app's
3. **Privacy-first** — financial data stays on device unless the user opts into sync
4. **Supportive tone** — every message should feel like a friend, not an accountant
5. **Progressive disclosure** — simple on the surface, powerful underneath

---

## Name & Branding Notes

**Nudge** — a gentle push in the right direction. Reflects:
- Behavioral economics concept (Thaler & Sunstein's *Nudge* theory)
- Non-coercive guidance
- Small actions with big impact over time

Color palette ideas: warm greens and soft neutrals (trust, growth, calm — not the cold blues of banking apps).
