#!/usr/bin/env bash
# הוספת ישיבת מליאה חדשה ל"מה אמרו במועצה": כל השרשרת בפקודה אחת.
#
# קודם מוסיפים את הישיבה ל-data/meetings.json (מזהה יוטיוב, תאריך, כותרת), ואז:
#   tools/update_council.sh <מזהה יוטיוב> ["הודעת קומיט ל-Space"]
#
# 1. תמלול (ivrit-ai על ה-GPU, בתקרת זיכרון של 7G)
# 2. זיהוי דוברים (pyannote, בסביבה של הפרויקט)
# 3. שמות לדוברים (Azure, כ-2 סנט)
# 4. ייצוא ל-NotebookLM
# 5. בניית האינדקס מחדש (bge-m3, כ-10 דקות)
# 6. פריסה ל-HF Space
set -euo pipefail

VID="${1:?usage: update_council.sh <youtube-id> [msg]}"
MSG="${2:-ישיבה חדשה $VID}"
cd "$(dirname "$0")/.."

systemd-run --user --scope -p MemoryMax=7G python3 tools/transcribe_meetings.py "$VID"
ls private/transcripts/*_"$VID".json >/dev/null  # נכשל אם התמלול לא נוצר
.venv/bin/python tools/diarize_meetings.py "$VID"
python3 tools/name_speakers.py "$VID"
python3 tools/export_notebooklm.py
.venv/bin/python rag/build_index.py
rag/deploy.sh "$MSG"
