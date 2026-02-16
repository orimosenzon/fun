#!/usr/bin/env python3
"""
Voice-to-Claude: Hands-free voice input for Claude Code.
Listens continuously, detects speech automatically, transcribes and pastes.
Just talk naturally - no buttons needed!
"""

import subprocess
import sys
import time
import wave
import struct
import collections
import tempfile
from pathlib import Path


# --- Config ---
MIC_DEVICE = "plughw:2,0"
WHISPER_MODEL = "small"
SAMPLE_RATE = 16000
LANGUAGE = "he"  # "he" for Hebrew, "en" for English, None for auto-detect
CHANNELS = 1
FRAME_DURATION_MS = 30  # 30ms frames for VAD
SILENCE_TIMEOUT_S = 1.0  # seconds of silence to stop recording
SPEECH_START_FRAMES = 10  # consecutive speech frames to start recording
MIN_RECORDING_S = 0.5  # minimum recording length to transcribe


def install_if_missing(package, pip_name=None):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {pip_name or package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", pip_name or package],
                       check=True, capture_output=True)


def check_dependencies():
    for cmd in ["arecord", "xdotool", "xclip"]:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            print(f"Missing: {cmd}")
            print("Install: sudo apt install alsa-utils xdotool xclip")
            sys.exit(1)
    install_if_missing("webrtcvad", "webrtcvad")
    install_if_missing("faster_whisper", "faster-whisper")


def load_whisper():
    from faster_whisper import WhisperModel
    print(f"Loading Whisper '{WHISPER_MODEL}'... (first time downloads ~500MB)")
    try:
        model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
        print("Running on GPU!")
    except Exception as e:
        print(f"GPU failed ({e}), using CPU...")
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("Model loaded!")
    return model


def transcribe(model, filepath):
    kwargs = {"beam_size": 1, "vad_filter": True}
    if LANGUAGE:
        kwargs["language"] = LANGUAGE
    segments, _ = model.transcribe(filepath, **kwargs)
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()


def find_vscode_window():
    """Find the VSCode window ID."""
    result = subprocess.run(
        ["xdotool", "search", "--name", "Visual Studio Code"],
        capture_output=True, text=True
    )
    windows = result.stdout.strip().split("\n")
    return windows[-1] if windows and windows[0] else None


def paste_text(text):
    window_id = find_vscode_window()
    if not window_id:
        print("⚠️  VSCode window not found, pasting to active window")
        window_id = None

    subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
    time.sleep(0.05)

    if window_id:
        subprocess.run(["xdotool", "windowactivate", "--sync", window_id], check=True)
        time.sleep(0.2)

    subprocess.run(["xdotool", "key", "ctrl+v"], check=True)
    time.sleep(0.1)
    subprocess.run(["xdotool", "key", "Return"], check=True)


def save_wav(frames, filepath):
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))


def open_mic_stream():
    """Open a raw PCM stream from the microphone using arecord."""
    proc = subprocess.Popen(
        ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", str(SAMPLE_RATE),
         "-c", str(CHANNELS), "-t", "raw", "--buffer-size=4096"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return proc


def read_frame(stream, frame_size):
    """Read one frame of audio from the stream."""
    data = stream.stdout.read(frame_size)
    if len(data) < frame_size:
        return None
    return data


def main():
    check_dependencies()
    import webrtcvad

    model = load_whisper()
    vad = webrtcvad.Vad(2)  # aggressiveness 0-3 (2 = balanced)

    frame_size = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) * 2  # bytes per frame (16-bit)
    silence_frames_needed = int(SILENCE_TIMEOUT_S * 1000 / FRAME_DURATION_MS)
    min_frames = int(MIN_RECORDING_S * 1000 / FRAME_DURATION_MS)

    print("\n" + "=" * 50)
    print("  Voice-to-Claude: Hands-Free Mode")
    print("  Just talk! I'm listening...")
    print("  Press Ctrl+C to quit")
    print("=" * 50 + "\n")

    while True:
        mic = open_mic_stream()
        try:
            # --- Wait for speech ---
            speech_count = 0
            ring_buffer = collections.deque(maxlen=SPEECH_START_FRAMES)

            while True:
                frame = read_frame(mic, frame_size)
                if frame is None:
                    break

                is_speech = vad.is_speech(frame, SAMPLE_RATE)
                ring_buffer.append((frame, is_speech))
                speech_count = speech_count + 1 if is_speech else 0

                if speech_count >= SPEECH_START_FRAMES:
                    print("🎙️  Speech detected - recording...")
                    break

            # --- Record until silence ---
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
        print(f"⏳ Transcribing {duration:.1f}s of audio...")

        tmpfile = tempfile.mktemp(suffix=".wav")
        save_wav(recorded_frames, tmpfile)

        t0 = time.time()
        text = transcribe(model, tmpfile)
        t1 = time.time()
        Path(tmpfile).unlink(missing_ok=True)

        if text:
            print(f"📝 \"{text}\" (transcribed in {t1-t0:.1f}s)")
            paste_text(text)
            print("✅ Pasted! Listening again...\n")
        else:
            print("❌ No speech recognized. Listening...\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
