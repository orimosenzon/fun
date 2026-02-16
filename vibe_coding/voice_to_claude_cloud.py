#!/usr/bin/env python3
"""
Voice-to-Claude: Cloud-based speech recognition (Google).
Uses arecord + Google Speech API for fast, accurate Hebrew.
"""

import subprocess
import sys
import time
import tempfile
import json
import base64
import urllib.request
from pathlib import Path


# --- Config ---
MIC_DEVICE = "plughw:2,0"
LANGUAGE = "he-IL"  # Hebrew. "en-US" for English
SAMPLE_RATE = 16000
SILENCE_TIMEOUT_S = 1.0
SPEECH_START_FRAMES = 10
FRAME_DURATION_MS = 30


def check_dependencies():
    for cmd in ["arecord", "xdotool", "xclip"]:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            print(f"Missing: {cmd}")
            sys.exit(1)
    try:
        import webrtcvad  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "webrtcvad"], check=True)


def find_vscode_window():
    result = subprocess.run(
        ["xdotool", "search", "--name", "Visual Studio Code"],
        capture_output=True, text=True
    )
    windows = result.stdout.strip().split("\n")
    return windows[-1] if windows and windows[0] else None


def paste_text(text):
    window_id = find_vscode_window()
    subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
    time.sleep(0.05)
    if window_id:
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id], check=True)
        time.sleep(0.2)
    subprocess.run(["xdotool", "key", "ctrl+v"], check=True)
    time.sleep(0.1)
    subprocess.run(["xdotool", "key", "Return"], check=True)


def open_mic_stream():
    proc = subprocess.Popen(
        ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
         "-c", "1", "-t", "raw", "--buffer-size=4096"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return proc


def read_frame(stream, frame_size):
    data = stream.stdout.read(frame_size)
    if len(data) < frame_size:
        return None
    return data


def save_flac(frames, filepath):
    """Save as FLAC for smaller upload size."""
    import wave
    import struct
    wav_path = filepath + ".wav"
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    # Convert to FLAC using arecord's built-in or just send WAV
    return wav_path


def google_speech_recognize(audio_path, language="he-IL"):
    """Use Google's free speech recognition API."""
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio, language=language)


def main():
    check_dependencies()
    import webrtcvad

    # Check if speech_recognition is available
    try:
        import speech_recognition  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "SpeechRecognition"], check=True)

    vad = webrtcvad.Vad(2)
    frame_size = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2
    silence_frames_needed = int(SILENCE_TIMEOUT_S * 1000 / FRAME_DURATION_MS)
    min_frames = int(0.5 * 1000 / FRAME_DURATION_MS)

    print("\n" + "=" * 50)
    print("  Voice-to-Claude: Cloud Mode (Google)")
    print("  Just talk! Fast & accurate Hebrew")
    print("  Press Ctrl+C to quit")
    print("=" * 50 + "\n")

    import collections

    while True:
        mic = open_mic_stream()
        try:
            # Wait for speech
            speech_count = 0
            ring_buffer = collections.deque(maxlen=SPEECH_START_FRAMES)

            print("👂 Listening...")

            while True:
                frame = read_frame(mic, frame_size)
                if frame is None:
                    break
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
                ring_buffer.append((frame, is_speech))
                speech_count = speech_count + 1 if is_speech else 0
                if speech_count >= SPEECH_START_FRAMES:
                    print("🎙️  Recording...")
                    break

            # Record until silence
            recorded_frames = [f for f, _ in ring_buffer]
            silence_count = 0
            total_frames = len(recorded_frames)

            while True:
                frame = read_frame(mic, frame_size)
                if frame is None:
                    break
                recorded_frames.append(frame)
                total_frames += 1
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
                if is_speech:
                    silence_count = 0
                else:
                    silence_count += 1
                if silence_count >= silence_frames_needed and total_frames >= min_frames:
                    break

        finally:
            mic.terminate()
            mic.wait()

        if total_frames < min_frames:
            continue

        duration = total_frames * FRAME_DURATION_MS / 1000

        tmpfile = tempfile.mktemp(suffix=".wav")
        audio_path = save_flac(recorded_frames, tmpfile)

        t0 = time.time()
        try:
            text = google_speech_recognize(audio_path, LANGUAGE)
            t1 = time.time()
            if text:
                print(f'📝 "{text}" ({t1-t0:.1f}s)')
                paste_text(text)
                print("✅ Pasted!\n")
        except Exception as e:
            t1 = time.time()
            print(f"❌ {e} ({t1-t0:.1f}s)\n")

        Path(audio_path).unlink(missing_ok=True)
        Path(tmpfile).unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
