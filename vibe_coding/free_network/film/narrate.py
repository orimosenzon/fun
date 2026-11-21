"""Narration for the film: one recording per scene, and when each line of it is spoken.

  python3 film/narrate.py [--engine gemini|azure] [--only id,id] [--force]

Reads film/script.json, writes film/audio/<scene>.wav and film/audio/timing.json. Gemini sounds the
most natural but its free tier allows ten requests a day, so a scene is one request, and the
lines inside it are found by the pauses between them. Azure (Hila) has no such limit.
Keys are the shared ones in ~/.config/ai (never in the repository).
"""
import argparse, io, json, re, subprocess, sys, time, urllib.request, wave
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).parent
AUDIO = HERE / "audio"
SPOKEN = {}  # spelling for the voice only, where it reads a word wrong


def env(name, var):
    for line in (Path.home() / ".config/ai" / name).read_text().splitlines():
        if line.startswith(var + "="):
            return line.split("=", 1)[1].strip().strip("\"'")


def gemini(text, voice, path, model="gemini-3.8-flash-tts"):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=env("gemini.env", "GEMINI_API_KEY"))
    config = types.GenerateContentConfig(response_modalities=["AUDIO"], speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))))
    for attempt in range(6):
        try:
            part = client.models.generate_content(model=model, contents=text, config=config).candidates[0].content.parts[0].inline_data
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 5 or "per day" in str(e) or not any(c in str(e) for c in ("429", "500", "503")):
                raise
            time.sleep(15 * (attempt + 1))
    data = part.data
    if part.mime_type != "audio/wav":  # raw PCM: 16 bit, 24 kHz, mono
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000), w.writeframes(data)
        data = buf.getvalue()
    path.write_bytes(data)


def azure(text, voice, path):
    ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="he-IL">'
            f'<voice name="{voice}">{escape(text)}</voice></speak>')
    req = urllib.request.Request(
        f"https://{env('azure.env', 'AZURE_AI_REGION')}.tts.speech.microsoft.com/cognitiveservices/v1", data=ssml.encode(),
        headers={"Ocp-Apim-Subscription-Key": env("azure.env", "AZURE_OPENAI_API_KEY"), "Content-Type": "application/ssml+xml",
                 "User-Agent": "free-network-film", "X-Microsoft-OutputFormat": "riff-48khz-16bit-mono-pcm"})
    path.write_bytes(urllib.request.urlopen(req, timeout=90).read())


def duration(path):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]))


def pauses(path):
    """Silences inside the recording, as (start, end), longest first."""
    log = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af", "silencedetect=noise=-38dB:d=0.15", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    return list(zip(starts, ends))


def line_times(path, lines):
    """When each line starts and ends. A line break is the pause nearest to where the text says
    it should fall, judged by the share of letters before it."""
    total = duration(path)
    gaps = [g for g in pauses(path) if g[0] > 0.3 and g[1] < total - 0.2]
    letters = [len(re.sub(r"\s", "", line)) for line in lines]
    cuts, used = [], set()
    for i in range(1, len(lines)):
        expected = total * sum(letters[:i]) / sum(letters)
        best = min((g for g in gaps if g not in used), key=lambda g: abs((g[0] + g[1]) / 2 - expected), default=None)
        if best and abs((best[0] + best[1]) / 2 - expected) < 2.5:
            used.add(best)
            cuts.append(best)
        else:
            cuts.append((expected, expected))
    cuts.sort()
    edges = [0.0] + [c for cut in cuts for c in cut] + [total]
    return [[round(edges[2 * i], 2), round(edges[2 * i + 1], 2)] for i in range(len(lines))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="gemini")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    script = json.loads((HERE / "script.json").read_text())
    AUDIO.mkdir(exist_ok=True)
    only = set(filter(None, args.only.split(",")))
    timing = {}
    for scene in script["scenes"]:
        path = AUDIO / f"{scene['id']}.wav"
        if (args.force or not path.exists()) and (not only or scene["id"] in only):
            text = " ".join(scene["lines"])
            for written, spoken in SPOKEN.items():
                text = text.replace(written, spoken)
            if args.engine == "gemini":
                gemini(text, script.get("voice", "Algenib"), path)
            else:
                azure(text, "he-IL-Hila:DragonLatestNeural", path)
            print("made", path.name, file=sys.stderr)
        if path.exists():
            timing[scene["id"]] = {"length": round(duration(path), 2), "lines": line_times(path, scene["lines"])}
    (AUDIO / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=1))
    for key, value in timing.items():
        print(key, value["length"], value["lines"])
    print("total", round(sum(v["length"] for v in timing.values()), 1))


main()
