#!/usr/bin/env python3
"""
Voice Agent — Alexa-style, powered by Gemini (same model as OpenClaw).

Flow:
  mic → VAD → faster-whisper (local STT) → Gemini chat → gTTS → ffmpeg → aplay

Requirements:
    pip install google-genai pyaudio webrtcvad gtts faster-whisper
    export GEMINI_API_KEY="your-key"   # https://aistudio.google.com/apikey

No Google Cloud service account needed. STT runs fully offline.
"""

import argparse
import io
import os
import subprocess
import sys
import time
import wave
from typing import Optional

import pyaudio
import webrtcvad
from faster_whisper import WhisperModel
from google import genai
from google.genai import types
from gtts import gTTS

# ── Audio recording config ─────────────────────────────────────────────────────
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SIZE = SAMPLE_RATE * FRAME_MS // 1000
VAD_MODE = 2               # 0 = lenient … 3 = aggressive
SILENCE_FRAMES = 25        # ~750 ms silence → stop recording
LEADIN_FRAMES = 8          # ~240 ms pre-speech kept in buffer

# ── Gemini config ──────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_FALLBACKS = ["gemini-1.5-flash", "gemini-2.0-flash-lite"]

SYSTEM_PROMPT = """\
You are a helpful voice assistant, like Alexa — but smarter.
Always reply in the same language the user spoke (Hebrew or English).
Keep answers SHORT and spoken-friendly: no bullet points, no markdown, no lists.
One to three natural sentences max unless the user asks for more.\
"""

# ── Whisper STT config ─────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "small"   # tiny / base / small / medium / large


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_hebrew(text: str) -> bool:
    return any("\u0590" <= c <= "\u05FF" for c in text)


def _build_wav(frames: list[bytes]) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def _suppress_alsa(func):
    """Run func() while suppressing ALSA stderr spam."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_err = os.dup(2)
    os.dup2(devnull, 2)
    try:
        return func()
    finally:
        os.dup2(old_err, 2)
        os.close(old_err)
        os.close(devnull)


# ── Recording ──────────────────────────────────────────────────────────────────

def record_utterance() -> Optional[bytes]:
    """Wait for speech, record until ~750 ms silence. Returns WAV bytes."""
    vad = webrtcvad.Vad(VAD_MODE)
    pa = _suppress_alsa(pyaudio.PyAudio)
    stream = _suppress_alsa(lambda: pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=FRAME_SIZE,
    ))

    print("  \033[90m[listening...]\033[0m", end="", flush=True)

    ring: list[bytes] = []
    frames: list[bytes] = []
    triggered = False
    silence_count = 0

    try:
        while True:
            chunk = stream.read(FRAME_SIZE, exception_on_overflow=False)
            is_speech = vad.is_speech(chunk, SAMPLE_RATE)

            if not triggered:
                ring.append(chunk)
                if len(ring) > LEADIN_FRAMES:
                    ring.pop(0)
                if is_speech:
                    triggered = True
                    frames.extend(ring)
                    ring.clear()
                    print("\r  \033[31m[recording...]\033[0m", end="", flush=True)
            else:
                frames.append(chunk)
                if is_speech:
                    silence_count = 0
                else:
                    silence_count += 1
                    if silence_count >= SILENCE_FRAMES:
                        break
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    if not frames:
        return None

    print("\r\033[K", end="", flush=True)
    return _build_wav(frames)


# ── STT (local, faster-whisper) ────────────────────────────────────────────────

def load_whisper() -> WhisperModel:
    print(f"  \033[90mloading Whisper ({WHISPER_MODEL_SIZE})...\033[0m", end="", flush=True)
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    print("\r\033[K", end="", flush=True)
    return model


def transcribe(whisper: WhisperModel, wav_bytes: bytes) -> str:
    """Transcribe locally using faster-whisper."""
    buf = io.BytesIO(wav_bytes)
    segments, info = whisper.transcribe(
        buf,
        language=None,           # auto-detect
        task="transcribe",
        beam_size=5,
        vad_filter=True,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


# ── TTS + Playback ─────────────────────────────────────────────────────────────

def speak(text: str, lang: str = "auto") -> None:
    """Synthesize with gTTS and play via ffmpeg → aplay."""
    tts_lang = "iw" if (lang == "he" or (lang == "auto" and _is_hebrew(text))) else "en"

    buf = io.BytesIO()
    gTTS(text=text, lang=tts_lang, slow=False).write_to_fp(buf)
    mp3_bytes = buf.getvalue()

    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-i", "pipe:0", "-ar", "22050", "-ac", "1", "-f", "s16le", "pipe:1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    pcm, _ = ffmpeg.communicate(input=mp3_bytes)

    aplay = subprocess.Popen(
        ["aplay", "-q", "-r", "22050", "-f", "S16_LE", "-c", "1", "-"],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    aplay.communicate(input=pcm)


# ── Gemini chat ────────────────────────────────────────────────────────────────

def _gemini_send_with_retry(chat, text: str, max_retries: int = 3) -> str:
    """Send a message with automatic retry on 429 rate-limit errors."""
    import re
    for attempt in range(max_retries):
        try:
            return chat.send_message(text).text.strip()
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                delay = 30
                m = re.search(r"retryDelay.*?(\d+)s", msg)
                if m:
                    delay = int(m.group(1)) + 1
                if attempt < max_retries - 1:
                    print(f"\r\033[K  \033[33m⏳ rate limit — ממתין {delay}s...\033[0m", end="", flush=True)
                    time.sleep(delay)
                else:
                    raise
            else:
                raise
    return ""


# ── Main agent ─────────────────────────────────────────────────────────────────

class VoiceAgent:
    def __init__(self, lang: str = "auto"):
        self.lang = lang

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key or api_key in ("your-key-here", ""):
            print(
                "\033[31mError: GEMINI_API_KEY לא מוגדר.\033[0m\n"
                "הוסף ל-~/.bashrc:\n"
                "  export GEMINI_API_KEY='your-key'\n"
                "קבל מפתח: https://aistudio.google.com/apikey",
                file=sys.stderr,
            )
            sys.exit(1)

        self.client = genai.Client(api_key=api_key)
        self._model_index = 0
        self._all_models = [GEMINI_MODEL] + GEMINI_FALLBACKS
        self._cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=300,
        )
        self.chat = self._new_chat()
        self.whisper = load_whisper()

    def _new_chat(self):
        model = self._all_models[self._model_index]
        print(f"  \033[90mמודל: {model}\033[0m")
        return self.client.chats.create(model=model, config=self._cfg)

    def _next_model(self) -> bool:
        """Switch to next fallback model. Returns False if none left."""
        if self._model_index + 1 >= len(self._all_models):
            return False
        self._model_index += 1
        self.chat = self._new_chat()
        return True

    def run(self) -> None:
        print(f"\n\033[1;36m🦞 OpenClaw Voice Agent\033[0m")
        print("Speak in Hebrew or English. Ctrl+C to quit.\n")

        while True:
            try:
                wav = record_utterance()
                if wav is None:
                    continue

                print("\r\033[K  STT...", end="", flush=True)
                text = transcribe(self.whisper, wav)
                if not text:
                    print("\r\033[K  (quiet — skip)")
                    continue
                print(f"\r\033[K  \033[1mYou:\033[0m {text}")

                print("  thinking...", end="", flush=True)
                try:
                    reply = _gemini_send_with_retry(self.chat, text)
                except Exception as exc:
                    msg = str(exc)
                    if ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "404" in msg) and self._next_model():
                        print(f"\r\033[K  switching model...", end="", flush=True)
                        reply = _gemini_send_with_retry(self.chat, text)
                    else:
                        raise
                print(f"\r\033[K  \033[1;36mGemini:\033[0m {reply}\n")

                speak(reply, lang=self.lang)

            except KeyboardInterrupt:
                print("\n\n\033[90mBye!\033[0m")
                break
            except Exception as exc:
                print(f"\n  \033[31mError: {exc}\033[0m\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    global VAD_MODE, WHISPER_MODEL_SIZE

    parser = argparse.ArgumentParser(description="Voice agent powered by Gemini")
    parser.add_argument(
        "--lang", choices=["he", "en", "auto"], default="auto",
        help="TTS language override (default: auto-detect)"
    )
    parser.add_argument(
        "--vad", type=int, choices=[0, 1, 2, 3], default=VAD_MODE,
        help="VAD aggressiveness 0–3 (default: 2)"
    )
    parser.add_argument(
        "--whisper", choices=["tiny", "base", "small", "medium"], default=WHISPER_MODEL_SIZE,
        help="Whisper model size (default: small)"
    )
    args = parser.parse_args()
    VAD_MODE = args.vad
    WHISPER_MODEL_SIZE = args.whisper

    VoiceAgent(lang=args.lang).run()


if __name__ == "__main__":
    main()
