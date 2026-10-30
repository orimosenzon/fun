"""תשתית משותפת לסרטונים: טקסט עברי, קריינות ב-edge-tts וכיתוביות.

זטה והראשוניים (zeta_primes.py) עדיין מחזיק עותק משלו של הקוד הזה.
"""
import asyncio
import hashlib
import re
from pathlib import Path

import edge_tts
from pydub import AudioSegment

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


def tts(text):
    """קובץ הקראה של הטקסט (במטמון לפי הקול והטקסט) ואורכו בשניות."""
    key = hashlib.sha1(f"{VOICE}|{text}".encode()).hexdigest()[:16]
    path = AUDIO_DIR / f"{key}.mp3"
    if not path.exists():
        AUDIO_DIR.mkdir(exist_ok=True)
        asyncio.run(edge_tts.Communicate(text, VOICE).save(str(path)))
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
