<div dir="rtl">

# פריסה ל-Hugging Face Spaces (חינמי)

הגרסה הענן היא **זרימת הדפדפן** (העלאת PDF, הורדת JSON) — לא העטיפה הנייטיב.
אבישי רק פותח URL, מזדהה, ועובד. אין התקנה.

## מה כבר מוכן בריפו

- `Dockerfile` — אימג' ווב (בלי תלויות Qt), gunicorn על פורט 7860, worker יחיד.
- `.dockerignore` — לא שולח venv/desktop/memory ל-Space.
- **Basic Auth** — נאכף אוטומטית כש-`HASKALA_USER`/`HASKALA_PASS` מוגדרים.
- **אבטחה** — endpoints מבוססי-path (`/load?path=`, decode-by-path) מוחזרים 403 בענן; רק העלאה דרך הדפדפן פעילה.

## שלבים (חד-פעמי)

1. **תקרת הוצאה חודשית.** משתמשים במפתח ה-Anthropic הקיים — ודאו שעל החשבון מוגדרת **תקרת הוצאה חודשית** ב-[Anthropic Console](https://console.anthropic.com/) (Billing/Limits). זו רשת הביטחון: גם אם מישהו עוקף את ה-Basic Auth, ההפסד חסום בתקרה. (מפתח ייעודי נפרד היה מוסיף בידוד ביטול/מעקב, אך לא נדרש כאן.)

2. **צור Space.** ב-[huggingface.co](https://huggingface.co/) (חשבון חינמי) → **New Space** → SDK: **Docker** → Template: **Blank** → Visibility: **Public** (ה-Basic Auth מגן; Private Space דורש לכל מבקר חשבון HF).

3. **דחוף את קוד האפליקציה ל-Space.** ה-Space הוא ריפו git נפרד. דוחפים רק את קבצי האפליקציה (לא את כל המונוריפו):

</div>

```bash
git clone https://huggingface.co/spaces/<user>/<space-name> /tmp/haskala-space
cd ~/fun/vibe_coding/haskala
cp Dockerfile .dockerignore app.py requirements.txt /tmp/haskala-space/
cp -r templates /tmp/haskala-space/

cd /tmp/haskala-space
# צור README.md עם ה-frontmatter שבסעיף 4
git add -A && git commit -m "deploy haskala OCR" && git push
```

<div dir="rtl">

4. **README.md ל-Space** (חובה — ה-frontmatter קובע פורט/SDK). צור `/tmp/haskala-space/README.md`:

</div>

```
---
title: Haskala OCR
emoji: 📝
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

בודק איכות OCR לתרגילים בכתב יד.
```

<div dir="rtl">

5. **Secrets ב-Space.** Space → Settings → **Variables and secrets** → הוסף שלושה **Secrets**:
   - `ANTHROPIC_API_KEY` — המפתח הייעודי מסעיף 1
   - `HASKALA_USER` — שם המשתמש של אבישי
   - `HASKALA_PASS` — סיסמה חזקה

   > בלי `HASKALA_USER`/`HASKALA_PASS` האפליקציה תעלה **בלי אימות** (יש אזהרה בלוג). חובה להגדיר לפני שמשתפים את ה-URL.

6. ה-Space בונה אוטומטית. כשמסיים — שלח לאבישי את ה-URL + שם המשתמש והסיסמה. הדפדפן יבקש אותם פעם אחת.

## עדכון גרסה (אחרי ההתקנה הראשונית)

הסקריפט `deploy.sh` מסנכרן את הקבצים מ-`haskala/` ל-clone המקומי
של ה-Space (`~/fun/haskala-space/`), עושה commit ו-push דרך SSH.

```bash
./deploy.sh                       # קומיט אוטומטי לפי הסבג'קט האחרון בריפו
./deploy.sh "fix RTL on tooltip"  # קומיט עם הודעה מותאמת
```

מה הסקריפט עושה: `git pull --rebase` מה-Space → rsync של
`app.py`, `Dockerfile`, `.dockerignore`, `requirements.txt`, `templates/`,
`rubrics/`, ו-`deploy/space-readme.md` (כ-`README.md`) → commit → push.
הוא **לא** דוחף `desktop.py`, `run.sh`, `venv/`, `memory/`, `*.log`, או `.env`.

**דרישות חד-פעמיות:**
- Clone של ה-Space ב-`~/fun/haskala-space/` עם remote SSH:
  `git clone git@hf.co:spaces/orimosenzon/haskala-ocr ~/fun/haskala-space`
- מפתח SSH הציבורי (`~/.ssh/id_ed25519.pub`) הועלה ל-https://huggingface.co/settings/keys.

## הערות תפעול

- **Worker יחיד בכוונה** — תור ה-OCR (`JOBS`) חי בזיכרון התהליך. לטסטר יחיד זה מספיק; אל תעלה workers.
- **שינה** — Space חינמי נרדם רק אחרי ~48ש' חוסר-פעילות; ההתעוררות הבאה לוקחת ~30ש' פעם אחת.
- **README של ה-Space** מתוחזק ב-`deploy/space-readme.md` (יש בו frontmatter של HF). אל תערוך אותו ידנית ב-`~/fun/haskala-space/README.md` — הקומיש הבא ידרוס אותו.
- **אל תדחוף `.env`** ל-Space — הסודות באים מ-Secrets בלבד.

</div>
