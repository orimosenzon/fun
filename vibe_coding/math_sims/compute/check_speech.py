"""בדיקת הגייה: מקריא כל שורה ב-edge-tts ומתמלל בחזרה עם ivrit-ai, כדי לראות אם המנוע הבין ראשי תיבות ושמות.

הרצה: python3 compute/check_speech.py "שורה ראשונה" "שורה שנייה" ...
(צריך python3 של המערכת, שבו מותקן faster_whisper)
"""
import asyncio
import hashlib
import sys
import tempfile
from pathlib import Path

import edge_tts
from faster_whisper import WhisperModel

VOICE = "he-IL-AvriNeural"
model = WhisperModel("ivrit-ai/whisper-large-v3-turbo-ct2", device="cpu", compute_type="int8")
tmp = Path(tempfile.mkdtemp())
for line in sys.argv[1:]:
    path = tmp / (hashlib.sha1(line.encode()).hexdigest()[:10] + ".mp3")
    asyncio.run(edge_tts.Communicate(line, VOICE).save(str(path)))
    segs, _ = model.transcribe(str(path), language="he")
    print(f"{line}\n  → {' '.join(s.text.strip() for s in segs)}")
