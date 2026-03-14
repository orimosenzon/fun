# Karaoke / Lyrics Sync — Project Knowledge

## Idea
A web app that turns any YouTube song into a karaoke experience:
1. User searches for a song (YouTube search)
2. App fetches synced lyrics (YouTube captions or LRClib)
3. Browser plays the YouTube video and highlights the current line/word in real time

## Tech Stack
- Python + Flask (backend)
- yt-dlp (YouTube search + captions download)
- LRClib API (synced lyrics fallback)
- MusicBrainz API (song credits: lyricist, composer, arranger, performer)
- YouTube IFrame API + JavaScript (synchronized frontend player)
- Deployed on Railway

## Key Design Decisions
- No audio download or Whisper — uses YouTube captions or LRClib for sync
- Lyrics + credits cached as JSON in `static/` (keyed by URL hash)
- sync offset (לכוונון עיתוי) saved per-song in `localStorage`
- Language auto-detected from lyrics text (Unicode script ranges)
- Credits labels localized per language (he/en/ar/ru/fr/es/it/de/pt/ja/ko/zh)

## File Structure
```
karaoke/
  app.py              # Flask server
  transcriber.py      # yt-dlp + LRClib + MusicBrainz logic
  templates/
    index.html        # Player UI (search → results → player screens)
  static/             # Transcript/lyrics cache (JSON per song)
  requirements.txt
  memory/
    project-knowledge.md
```

## API Routes
- POST /api/process        — accepts { url, title }, returns full song data
- POST /api/job_id         — returns job ID for polling
- GET  /api/status/<id>    — returns processing stage + elapsed time
- GET  /api/search?q=...   — YouTube search filtered to songs with LRClib lyrics
- GET  /api/debug/credits  — debug endpoint for MusicBrainz credit fetch

## Song Data Format (cached JSON)
```json
{
  "id": "...", "title": "...", "url": "...", "source": "lrclib|youtube_captions|cached",
  "segments": [{ "text": "...", "start": 0.0, "end": 3.5, "words": [...] }],
  "lyricist": "...", "composer": "...", "arranger": "...", "performer": "...",
  "lang": "he|en|es|..."
}
```

## Planned Features (Future)

### Line-by-Line Translation
Allow the user to select any target language and see a translation displayed alongside each lyric line in real time.
- Translation done at processing time (cached) or on-demand
- Candidate APIs: LibreTranslate (free/self-hosted), DeepL, or Claude
- UI: toggle button to show/hide translation; translated line appears beneath the original in a muted color
- Language selector (dropdown) in the player header
- Translated text stored in the cached JSON per target language to avoid re-translating

## Status
- [x] Project scaffolded and deployed on Railway
- [x] YouTube search + LRClib lyrics (with YouTube captions fallback)
- [x] Word-level sync highlighting
- [x] YouTube IFrame video player with sync offset controls
- [x] Sync offset saved/restored per song (localStorage)
- [x] Song credits: lyricist, composer, arranger, performer (MusicBrainz)
- [x] Localized credit labels (12 languages)
- [ ] Line-by-line translation feature
