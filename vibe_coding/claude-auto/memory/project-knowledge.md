# Claude Auto — Project Knowledge

## What is this?
Voice assistant for driving on Android.
Activated by wake word. Powered by Claude Sonnet 4.6.
Goal: perform most useful phone actions hands-free while driving.

## Current Status
Planning phase. No code written yet.
User does not have Android Studio installed yet — next session starts from there.

## MVP Actions
1. Call someone
2. Send WhatsApp (fully automatic with voice confirmation)
3. Navigate to a place

## Tech Choices
- Platform: Android, Kotlin
- Wake word: Porcupine (Picovoice)
- STT: Vosk (local, offline, free)
- Brain: Claude Sonnet 4.6
- WhatsApp automation: Accessibility Service
- Language: English only (for now)

## Important Decisions
- WhatsApp must send fully automatically (no manual tap) — using Accessibility Service
- Contacts: app reads from device contacts
- Voice confirmation before sending/calling
- Background service with foreground notification to avoid being killed

## Next Steps (start of next session)
1. Install Android Studio
2. Create new Android project (Kotlin, Empty Activity)
3. Verify it builds
4. Add background foreground service skeleton
