"""Checks the narration by ear, as far as a machine can: speech recognition (Azure, Hebrew) writes
down what it hears in each recording, to compare with the script. A word the voice says wrong
usually comes back as a different word.   python3 film/listen.py [scene ...]
The key is the shared one in ~/.config/ai/azure.env (never in the repository)."""
import json, subprocess, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).parent


def env(var):
    for line in (Path.home() / ".config/ai/azure.env").read_text().splitlines():
        if line.startswith(var + "="):
            return line.split("=", 1)[1].strip().strip("\"'")


script = json.loads((HERE / "script.json").read_text())
for scene in script["scenes"]:
    path = HERE / "audio" / f"{scene['id']}.wav"
    if not path.exists() or (sys.argv[1:] and scene["id"] not in sys.argv[1:]):
        continue
    wav = subprocess.check_output(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-ar", "16000", "-ac", "1", "-f", "wav", "-"])
    request = urllib.request.Request(
        f"https://{env('AZURE_AI_REGION')}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=he-IL",
        data=wav, headers={"Ocp-Apim-Subscription-Key": env("AZURE_OPENAI_API_KEY"), "Content-Type": "audio/wav; codecs=audio/pcm; samplerate=16000", "Accept": "application/json"})
    heard = json.load(urllib.request.urlopen(request, timeout=90)).get("DisplayText", "")
    print(f"== {scene['id']}\nכתוב: {' '.join(scene['lines'])}\nנשמע: {heard}\n", flush=True)
