#!/usr/bin/env python3
"""
WhatsApp Voice Message Transcriber
Uses Google Cloud Speech-to-Text via OpenClaw's CLI provider interface.

Usage:
    python3 transcribe.py <audio_file_path>

Output:
    Transcription text on stdout (OpenClaw reads this).
    Errors on stderr.
"""

import sys
import os
import subprocess
import tempfile
from google.cloud import speech


def convert_to_wav(input_path: str) -> tuple[str, bool]:
    """
    Convert audio file to 16kHz mono WAV for Google STT.
    WhatsApp voice messages are OGG/Opus format.
    Returns (wav_path, should_cleanup).
    """
    if input_path.endswith(".wav"):
        return input_path, False

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = tmp.name
    tmp.close()

    subprocess.run(
        ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", wav_path, "-y"],
        check=True,
        capture_output=True,
    )
    return wav_path, True


def transcribe(audio_path: str) -> str:
    wav_path, cleanup = convert_to_wav(audio_path)

    try:
        with open(wav_path, "rb") as f:
            content = f.read()
    finally:
        if cleanup:
            os.unlink(wav_path)

    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="he-IL",
        alternative_language_codes=["en-US", "ar-IL"],
        enable_automatic_punctuation=True,
    )

    response = client.recognize(config=config, audio=audio)

    if not response.results:
        return "[לא זוהתה דיבור]"

    return " ".join(
        result.alternatives[0].transcript for result in response.results
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 transcribe.py <audio_file>", file=sys.stderr)
        sys.exit(1)

    audio_path = sys.argv[1]

    if not os.path.exists(audio_path):
        print(f"Error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = transcribe(audio_path)
        print(result)
    except Exception as e:
        print(f"Transcription error: {e}", file=sys.stderr)
        sys.exit(1)
