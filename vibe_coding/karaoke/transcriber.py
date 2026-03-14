# Copyright (C) 2026 Ori Mosenzon and Claude (Anthropic AI)
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# See the LICENSE file for details.

import os
import re
import json
import hashlib
import glob as glob_mod
import threading
import urllib.request
import urllib.parse
import yt_dlp

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


def url_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def search_songs(query: str) -> list:
    """Search YouTube and return songs that have LRClib lyrics."""
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "no_warnings": True,
        "extractor_args": {"youtube": {"js_runtimes": ["nodejs"]}},
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch10:{query}", download=False)
    except Exception as e:
        print(f"YouTube search failed: {e}")
        return []

    candidates = []
    for entry in (info.get("entries") or []):
        if not entry or not entry.get("id"):
            continue
        vid_id = entry["id"]
        candidates.append({
            "title": entry.get("title", "Unknown"),
            "url": f"https://www.youtube.com/watch?v={vid_id}",
            "thumbnail": f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg",
        })

    confirmed = []
    lock = threading.Lock()

    def check_one(c):
        if _check_lrclib(c["title"]):
            with lock:
                confirmed.append(c)

    threads = [threading.Thread(target=check_one, args=(c,)) for c in candidates]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=6)

    return confirmed[:5]


def _check_lrclib(title: str) -> bool:
    """Quick check if LRClib has synced lyrics for this title."""
    query = urllib.parse.urlencode({"q": title})
    url = f"https://lrclib.net/api/search?{query}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "KaraokeApp/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            results = json.loads(resp.read())
        return any(r.get("syncedLyrics") for r in results)
    except Exception:
        return False


def process_url(url: str, title: str = "", on_stage=None):
    def stage(s):
        if on_stage:
            on_stage(s)

    vid_id = url_id(url)
    transcript_path = os.path.join(STATIC_DIR, f"{vid_id}.json")

    if os.path.exists(transcript_path):
        stage("cached")
        with open(transcript_path) as f:
            cached = json.load(f)
        # Backfill credits if missing
        if not cached.get("lyricist") and not cached.get("composer") and not cached.get("performer"):
            credits = _fetch_credits(cached.get("title", title))
            cached["lyricist"]  = credits.get("lyricist")
            cached["composer"]  = credits.get("composer")
            cached["arranger"]  = credits.get("arranger")
            cached["performer"] = credits.get("performer")
            cached["lang"] = _detect_language(_lyrics_text(cached.get("segments", [])))
            with open(transcript_path, "w") as f:
                json.dump(cached, f, ensure_ascii=False)
        return cached

    # Try YouTube captions first, then LRClib
    stage("captions")
    captions = _try_youtube_captions(url, vid_id)
    if captions:
        segments, source = captions, "youtube_captions"
    else:
        stage("lrclib")
        lrc = _try_lrclib(title)
        if lrc:
            segments, source = lrc, "lrclib"
        else:
            return {"error": "No lyrics found for this song. Try a more popular track."}

    credits = _fetch_credits(title)
    lang = _detect_language(_lyrics_text(segments))
    data = {
        "id": vid_id, "title": title, "url": url,
        "segments": segments, "source": source,
        "lyricist":  credits.get("lyricist"),
        "composer":  credits.get("composer"),
        "arranger":  credits.get("arranger"),
        "performer": credits.get("performer"),
        "lang": lang,
    }
    with open(transcript_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)

    return data


def _try_youtube_captions(url: str, vid_id: str):
    """Download YouTube auto-captions (VTT) and parse them. Returns segments or None."""
    vtt_base = os.path.join(STATIC_DIR, vid_id)
    ydl_opts = {
        "writeautomaticsub": True,
        "writesubtitles": True,
        "subtitleslangs": ["en", "en-orig"],
        "subtitlesformat": "vtt",
        "skip_download": True,
        "outtmpl": vtt_base,
        "quiet": True,
        "extractor_args": {"youtube": {"js_runtimes": ["nodejs"]}},
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception:
        return None

    # Find the downloaded VTT file
    vtt_files = glob_mod.glob(f"{vtt_base}*.vtt")
    if not vtt_files:
        return None

    try:
        segments = _parse_vtt(vtt_files[0])
        if segments:
            print(f"Using YouTube captions ({len(segments)} segments)")
            return segments
    except Exception as e:
        print(f"VTT parse failed: {e}")
    return None


def _parse_vtt(path: str):
    """Parse YouTube VTT with embedded word timestamps."""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    segments = []
    # Each cue block: timestamp line + content lines
    cue_pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})[^\n]*\n(.*?)(?=\n\n|\Z)',
        re.DOTALL
    )

    seen_texts = set()

    for m in cue_pattern.finditer(content):
        seg_start = _ts(m.group(1))
        seg_end = _ts(m.group(2))
        raw = m.group(3).strip()

        # Remove positioning tags like <c>, keep text and timestamps
        # YouTube format: <00:00:01.000><c> word</c>
        text_only = re.sub(r'<[^>]+>', '', raw).strip()
        if not text_only or text_only in seen_texts:
            continue
        seen_texts.add(text_only)

        # Parse word-level timestamps
        words = _parse_vtt_words(raw, seg_start, seg_end)

        segments.append({
            "text": text_only,
            "start": seg_start,
            "end": seg_end,
            "words": words,
        })

    return segments


def _parse_vtt_words(raw: str, seg_start: float, seg_end: float):
    """Extract word-level timing from a YouTube VTT cue line."""
    # Pattern: optional <timestamp> followed by <c> word </c>
    # Example: <00:00:02.219><c> Hello</c><00:00:02.459><c> world</c>
    token_pattern = re.compile(r'(?:(\d{2}:\d{2}:\d{2}[.,]\d{3}))?\s*<c>(.*?)</c>', re.DOTALL)

    words = []
    tokens = token_pattern.findall(raw)

    for i, (ts_str, word_text) in enumerate(tokens):
        word = word_text.strip()
        if not word:
            continue
        start = _ts(ts_str) if ts_str else seg_start
        # End is the next token's start, or seg_end for the last
        if i + 1 < len(tokens) and tokens[i + 1][0]:
            end = _ts(tokens[i + 1][0])
        else:
            end = seg_end
        words.append({"word": word, "start": start, "end": end})

    return words


def _ts(s: str) -> float:
    """Parse HH:MM:SS.mmm or HH:MM:SS,mmm to seconds."""
    s = s.replace(',', '.')
    parts = s.split(':')
    h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + sec


def _try_lrclib(title: str):
    """Search LRClib.net for synced lyrics by song title."""
    query = urllib.parse.urlencode({"q": title})
    url = f"https://lrclib.net/api/search?{query}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "KaraokeApp/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            results = json.loads(resp.read())
    except Exception as e:
        print(f"LRClib search failed: {e}")
        return None

    for result in results:
        lrc = result.get("syncedLyrics")
        if lrc:
            segments = _parse_lrc(lrc)
            if segments:
                print(f"Using LRClib lyrics ({len(segments)} lines)")
                return segments
    return None


def _parse_lrc(lrc: str):
    """Parse LRC format [mm:ss.xx] text into segments."""
    pattern = re.compile(r'\[(\d{2}):(\d{2}\.\d+)\](.*)')
    lines = []
    for m in pattern.finditer(lrc):
        start = int(m.group(1)) * 60 + float(m.group(2))
        text = m.group(3).strip()
        if text:
            lines.append((start, text))

    segments = []
    for i, (start, text) in enumerate(lines):
        end = lines[i + 1][0] if i + 1 < len(lines) else start + 5.0
        segments.append({
            "text": text,
            "start": round(start, 3),
            "end": round(end, 3),
            "words": [],
        })
    return segments


def _lyrics_text(segments: list) -> str:
    """Concatenate all segment texts for language detection."""
    return " ".join(s.get("text", "") for s in segments[:20])


def _detect_language(text: str) -> str:
    """Detect language from lyrics text using Unicode script ranges."""
    if not text:
        return "en"
    counts = {
        "he": sum(1 for c in text if "\u0590" <= c <= "\u05FF"),
        "ar": sum(1 for c in text if "\u0600" <= c <= "\u06FF"),
        "ru": sum(1 for c in text if "\u0400" <= c <= "\u04FF"),
        "ja": sum(1 for c in text if "\u3040" <= c <= "\u30FF" or "\u4E00" <= c <= "\u9FFF"),
        "ko": sum(1 for c in text if "\uAC00" <= c <= "\uD7AF"),
        "zh": sum(1 for c in text if "\u4E00" <= c <= "\u9FFF"),
    }
    threshold = len(text) * 0.08
    best = max(counts, key=counts.get)
    if counts[best] > threshold:
        return best
    return "en"


def _clean_title(title: str) -> str:
    """Strip YouTube-style suffixes and 'Artist - ' prefix for cleaner MusicBrainz searches."""
    cleaned = re.sub(r'\s*[\(\[][^\)\]]*[\)\]]', '', title).strip()
    if ' - ' in cleaned:
        cleaned = cleaned.split(' - ', 1)[1].strip()
    return cleaned or title


def _fetch_credits(title: str) -> dict:
    """Fetch lyricist and composer from MusicBrainz. Returns dict with 'lyricist' and/or 'composer'."""
    headers = {"User-Agent": "KaraokeApp/1.0 (open-source karaoke project)"}
    search_title = _clean_title(title)
    try:
        # Step 1: search recording
        query = urllib.parse.urlencode({"query": f'recording:"{search_title}"', "fmt": "json", "limit": "5"})
        req = urllib.request.Request(
            f"https://musicbrainz.org/ws/2/recording?{query}", headers=headers
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            recordings = json.loads(resp.read()).get("recordings", [])
        if not recordings:
            return {}
        rec = recordings[0]
        mbid = rec["id"]

        # Extract performer from artist-credit in search result
        artist_credits = rec.get("artist-credit", [])
        performer_names = [a["artist"]["name"] for a in artist_credits if isinstance(a, dict) and "artist" in a]
        performer = ", ".join(performer_names) or None

        # Step 2: recording → work-rels + artist-rels (for arranger at recording level)
        req = urllib.request.Request(
            f"https://musicbrainz.org/ws/2/recording/{mbid}?inc=work-rels+artist-rels&fmt=json", headers=headers
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            rec_data = json.loads(resp.read())

        arrangers = []
        for rel in rec_data.get("relations", []):
            if rel.get("target-type") == "artist":
                role = rel.get("type", "").lower()
                name = rel.get("artist", {}).get("name", "")
                if role == "arranger" and name:
                    arrangers.append(name)

        work_mbid = next(
            (r["work"]["id"] for r in rec_data.get("relations", []) if r.get("target-type") == "work"),
            None,
        )
        if not work_mbid:
            return {"lyricist": None, "composer": None, "arranger": ", ".join(arrangers) or None, "performer": performer}

        # Step 3: work → artist relations (composer / lyricist / writer / arranger)
        req = urllib.request.Request(
            f"https://musicbrainz.org/ws/2/work/{work_mbid}?inc=artist-rels&fmt=json", headers=headers
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            work_data = json.loads(resp.read())

        lyricists, composers = [], []
        for rel in work_data.get("relations", []):
            role = rel.get("type", "").lower()
            name = rel.get("artist", {}).get("name", "")
            if not name:
                continue
            if role == "lyricist":
                lyricists.append(name)
            elif role == "composer":
                composers.append(name)
            elif role == "writer":
                lyricists.append(name)
                composers.append(name)
            elif role == "arranger":
                arrangers.append(name)

        return {
            "lyricist":  ", ".join(lyricists) or None,
            "composer":  ", ".join(composers) or None,
            "arranger":  ", ".join(arrangers) or None,
            "performer": performer,
        }
    except Exception as e:
        print(f"Credits fetch failed: {e}")
        return {}


