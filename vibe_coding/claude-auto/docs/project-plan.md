# Claude Auto — Project Plan

## Vision
A voice-first Android assistant for driving.
Hands-free, eyes-on-the-road. Activated by wake word.
Powered by Claude Sonnet 4.6.

## Phase 1 — MVP (3 actions)
- Call someone
- Send WhatsApp
- Navigate to a place

## Future Phases
- More phone actions (calendar, reminders, music, etc.)
- Hebrew language support
- Custom wake word
- Optional cloud STT for better accuracy

---

## Tech Stack

| Component       | Technology                          |
|----------------|--------------------------------------|
| Platform        | Android (Kotlin)                    |
| Wake Word       | Porcupine (Picovoice) — free tier   |
| STT             | Vosk — local, offline, free         |
| Brain           | Claude Sonnet 4.6 (API)             |
| WhatsApp        | Accessibility Service automation    |
| Calls           | Android Intent ACTION_CALL          |
| Navigation      | Android Intent → Google Maps        |
| TTS (feedback)  | Android built-in TextToSpeech       |

---

## Flow

```
[Wake Word "Claude"]
    → [Record user speech]
    → [Vosk: speech → text]
    → [Claude API: extract intent + entities]
    → [Voice confirmation if needed]
    → [Execute Android action]
```

### Example
User: "Claude, send a WhatsApp to Mom: I'm on my way"

1. Porcupine detects wake word
2. Vosk transcribes: "send a WhatsApp to Mom: I'm on my way"
3. Claude returns: `{ action: "whatsapp", contact: "Mom", message: "I'm on my way" }`
4. TTS asks: "Send to Mom: I'm on my way. Say yes or no."
5. User: "yes"
6. App opens WhatsApp, pre-fills message, Accessibility Service taps Send

---

## Build Roadmap

### Step 1 — Skeleton
- Basic Android app
- Background Service that stays alive
- Microphone permission

### Step 2 — Wake Word
- Integrate Porcupine SDK
- Detect "Claude" (or custom word)
- Visual/audio feedback when activated

### Step 3 — STT
- Integrate Vosk
- Download English model
- Transcribe speech after wake word

### Step 4 — Claude Integration
- Send transcribed text to Claude Sonnet 4.6
- Structured JSON response: `{ action, contact, message, destination }`
- Handle ambiguous input gracefully

### Step 5 — Actions
- **Call:** resolve contact name → `Intent ACTION_CALL`
- **Navigate:** `Intent` to Google Maps with destination
- **WhatsApp:** open WhatsApp via URI scheme → Accessibility Service taps Send

### Step 6 — Voice Confirmation
- TTS reads back the intended action
- Listen for "yes" / "no" / "cancel"
- Execute or abort

---

## Key Decisions
- **English only** for now (Vosk has Hebrew models for later)
- **WhatsApp automatic send** via Accessibility Service (requires user permission, works like AutoResponder for WA)
- **Contacts access** — app reads device contacts, Claude matches by name
- **Always-on background service** — needs battery optimization exemption

---

## Known Risks / Challenges
- WhatsApp UI changes may break Accessibility Service automation
- Porcupine free tier limits custom wake words (may need to use a built-in keyword)
- Background service may be killed by Android battery optimization — needs `foreground service` with notification
- STT accuracy in noisy car environment
