"""תשתית משותפת לסרטונים: טקסט עברי, קריינות (edge-tts, Azure או Gemini) וכיתוביות.

זטה והראשוניים (zeta_primes.py) עדיין מחזיק עותק משלו של הקוד הזה.
"""
import asyncio
import hashlib
import io
import re
import time
import wave
from pathlib import Path

import edge_tts
from pydub import AudioSegment, silence

from manimlib import *

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "audio"
VOICE = "he-IL-AvriNeural"
SPEECH_GAP = 0.35  # שקט בין משפט למשפט
FONT = "Rubik"
BG = "#14161c"

LTR_START = r"[A-Za-z0-9α-ωπ½²³ˢ√∛∞]"
LTR_END = r"[A-Za-z0-9α-ωπ½²³ˢ√∛∞°)]"
# רצף לטיני/מתמטי בתוך משפט עברי: מתחיל ונגמר בתו "חזק", כך שפיסוק וסוגריים עבריים נשארים בחוץ
LTR_RUN = re.compile(rf"{LTR_START}(?:[^֐-׿]*{LTR_END})?")


def heb(text, size=36, color=WHITE, **kwargs):
    """טקסט עברי. ביטויים כמו π(x) או 3 = 2·2 נעטפים ב-LRI/PDI כדי שלא יתהפכו."""
    text = LTR_RUN.sub(lambda m: "⁦" + m.group(0) + "⁩", text)
    return Text(text, font=FONT, font_size=size, fill_color=color, **kwargs)


NARRATOR = ("edge", VOICE)  # (מנוע, קול). סרטון יכול לבחור אחר עם use_azure / use_gemini


def use_azure(voice="he-IL-Hila:DragonLatestNeural"):
    """קריינות ב-Azure Speech, על חשבון קרדיט Azure (~/.config/ai/azure.env).
    Hila Dragon ניצחה בהשוואה עיוורת את Avri של edge-tts ואת הקולות הרב-לשוניים."""
    global NARRATOR
    NARRATOR = ("azure", voice)


def use_gemini(voice="Algenib", model="gemini-3.8-flash-tts"):
    """קריינות ב-Gemini TTS, הטבעית ביותר בהשוואה. מפתח ב-~/.config/ai/gemini.env.
    בלי הנחיית סגנון: המודל מקריא אותה בקול. בשכבה החינמית רק 10 בקשות ביום."""
    global NARRATOR
    NARRATOR = ("gemini", voice, model)


def _env(name, var):
    for line in (Path.home() / ".config/ai" / name).read_text().splitlines():
        if line.startswith(var + "="):
            return line.split("=", 1)[1].strip()


def _azure_tts(text, path):
    import urllib.request
    from xml.sax.saxutils import escape

    voice = NARRATOR[1]
    ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="he-IL">'
            f'<voice name="{voice}">{escape(text)}</voice></speak>')
    region = _env("azure.env", "AZURE_AI_REGION")
    req = urllib.request.Request(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1", data=ssml.encode(),
        headers={"Ocp-Apim-Subscription-Key": _env("azure.env", "AZURE_OPENAI_API_KEY"),
                 "Content-Type": "application/ssml+xml", "User-Agent": "math_sims",
                 "X-Microsoft-OutputFormat": "audio-48khz-192kbitrate-mono-mp3"})
    for attempt in range(6):
        try:
            data = urllib.request.urlopen(req, timeout=60).read()
            break
        except Exception:  # noqa: BLE001
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
    _save_trimmed(AudioSegment.from_file(io.BytesIO(data), format="mp3"), path)


def _gemini_tts(text, path):
    from google import genai
    from google.genai import types

    _, voice, model = NARRATOR
    client = genai.Client(api_key=_env("gemini.env", "GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice))))
    for attempt in range(12):
        try:
            part = client.models.generate_content(model=model, contents=text, config=config) \
                .candidates[0].content.parts[0].inline_data
            break
        except Exception as e:  # noqa: BLE001
            # עומס רגעי שווה המתנה; מכסה יומית לא
            if attempt == 11 or "per day" in str(e) or not any(c in str(e) for c in ("429", "500", "503")):
                raise
            time.sleep(min(10 * (attempt + 1), 60))
    data = part.data
    if part.mime_type != "audio/wav":  # PCM גולמי: 16 ביט, 24kHz, מונו
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000), w.writeframes(data)
        data = buf.getvalue()
    _save_trimmed(AudioSegment.from_file(io.BytesIO(data), format="wav"), path)


def _save_trimmed(audio, path):
    """חותך את השקט שבקצוות, כדי שהתזמון ייקבע לפי הדיבור עצמו."""
    lead = silence.detect_leading_silence(audio, silence_threshold=-45)
    tail = silence.detect_leading_silence(audio.reverse(), silence_threshold=-45)
    audio[max(lead - 50, 0):len(audio) - max(tail - 100, 0)].export(path, format="wav")


SPOKEN = {}  # כתיב להקראה בלבד, למילים שהמנוע הוגה לא נכון. למשל {"פורייה": "פוריֶה"}


def tts(text):
    """קובץ הקראה של הטקסט (במטמון לפי הקול והטקסט) ואורכו בשניות."""
    for written, spoken in SPOKEN.items():
        text = text.replace(written, spoken)
    engine = NARRATOR[0]
    if engine == "edge":
        key = hashlib.sha1(f"{VOICE}|{text}".encode()).hexdigest()[:16]
        path = AUDIO_DIR / f"{key}.mp3"
    else:
        key = hashlib.sha1("|".join((*NARRATOR, text)).encode()).hexdigest()[:16]
        path = AUDIO_DIR / f"{engine[0]}_{key}.wav"
    if not path.exists():
        AUDIO_DIR.mkdir(exist_ok=True)
        if engine == "edge":
            asyncio.run(edge_tts.Communicate(text, VOICE).save(str(path)))
        else:
            {"azure": _azure_tts, "gemini": _gemini_tts}[engine](text, path)
    return path, AudioSegment.from_file(path).duration_seconds


class HebrewScene(Scene):
    default_camera_config = dict(background_color=BG)

    def setup(self):
        super().setup()
        self.cap = None
        self.speech_end = 0

    def say(self, text):
        """מתחיל להקריא מהרגע הזה. האנימציות שאחריו רצות במקביל להקראה."""
        self.wait_speech()
        path, duration = tts(text)
        self.add_sound(str(path))
        self.speech_end = self.time + duration + SPEECH_GAP

    def wait_speech(self):
        if self.speech_end > self.time:
            self.wait(self.speech_end - self.time)

    def caption(self, text, say=None, size=34, wait=0.5):
        """כיתובית קצרה על המסך, והקראה של say (או של הכיתובית עצמה)."""
        self.say(say or text)
        new = heb(text, size).to_edge(DOWN, buff=0.4)
        if self.cap is None:
            self.play(FadeIn(new, shift=0.2 * UP))
        else:
            self.play(FadeOut(self.cap, shift=0.2 * UP), FadeIn(new, shift=0.2 * UP))
        self.cap = new
        self.wait(wait)

    def clear_caption(self):
        if self.cap is not None:
            self.play(FadeOut(self.cap))
            self.cap = None

    def fade_all(self, run_time=1):
        self.wait_speech()
        self.play(*map(FadeOut, self.mobjects), run_time=run_time)
        self.cap = None
