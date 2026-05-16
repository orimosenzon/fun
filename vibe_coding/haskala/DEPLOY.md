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

1. **מפתח Anthropic ייעודי עם תקרה.** ב-[Anthropic Console](https://console.anthropic.com/) → API Keys → צור מפתח חדש *רק לפרויקט הזה*. ב-Billing/Limits → קבע **תקרת הוצאה חודשית נמוכה** (למשל $10). זה גג הביטחון — גם אם משהו משתבש, ההפסד חסום.

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

## הערות תפעול

- **Worker יחיד בכוונה** — תור ה-OCR (`JOBS`) חי בזיכרון התהליך. לטסטר יחיד זה מספיק; אל תעלה workers.
- **שינה** — Space חינמי נרדם רק אחרי ~48ש' חוסר-פעילות; ההתעוררות הבאה לוקחת ~30ש' פעם אחת.
- **עדכון גרסה** — דחוף שוב את הקבצים ל-ריפו ה-Space; הוא בונה מחדש לבד.
- **אל תדחוף `.env`** ל-Space — הסודות באים מ-Secrets בלבד.

</div>
