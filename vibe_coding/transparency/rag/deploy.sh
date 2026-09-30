#!/usr/bin/env bash
# פריסת "מה אמרו במועצה" ל-Hugging Face Space, בדפוס של remez/deploy.sh.
#
# 1. מושך את השכפול המקומי של ה-Space (כדי לא לדרוס שינויים שנעשו באתר).
# 2. מעתיק את הקוד מ-rag/ ואת האינדקס מ-private/index (הנתונים עם השמות).
# 3. קומיט ודחיפה ב-SSH. הקבצים הגדולים עוברים ב-git-lfs, כי HF דוחה בינאריים.
#
# הגדרה ראשונה: יוצרים Space ריק מסוג Docker באתר, ואז:
#   git clone git@hf.co:spaces/orimosenzon/$SPACE_NAME ~/fun/council-space
# שימוש:
#   ./deploy.sh "הודעת קומיט"
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
SPACE_NAME="${SPACE_NAME:-phk-council}"
SPACE="${COUNCIL_SPACE_DIR:-$HOME/fun/council-space}"
MSG="${1:-update $(date +%F)}"

if [[ ! -d "$SPACE/.git" ]]; then
    echo "❌ אין שכפול של ה-Space ב-$SPACE" >&2
    echo "   git clone git@hf.co:spaces/orimosenzon/$SPACE_NAME \"$SPACE\"" >&2
    exit 1
fi

git -C "$SPACE" pull --rebase -q || true
# Space מסוג Gradio (Docker עבר לתשלום): app.py של ה-Space הוא space/app_gradio.py,
# והשרת שלנו נכנס בשם server.py
cp "$SRC/app.py" "$SPACE/server.py"
cp "$SRC/space/app_gradio.py" "$SPACE/app.py"
cp "$SRC/engine.py" "$SRC/index.html" "$SRC/space/requirements.txt" "$SRC/space/README.md" "$SPACE/"
mkdir -p "$SPACE/index"
cp "$SRC/../private/index/chunks.json" "$SRC/../private/index/emb.npy" "$SPACE/index/"

cd "$SPACE"
git lfs install --local >/dev/null
git lfs track "index/*" >/dev/null
git add -A
if git diff --cached --quiet; then
    echo "אין שינויים לפריסה."
    exit 0
fi
git commit -q -m "$MSG"
git push -q
echo "✅ נדחף. הבנייה לוקחת כמה דקות: https://huggingface.co/spaces/orimosenzon/$SPACE_NAME"
