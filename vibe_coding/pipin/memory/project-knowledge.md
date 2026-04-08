# פיפין — ידע כללי על הפרויקט

## מה זה
משחק הרפתקאות טקסטואלי-ויזואלי בדפדפן. השחקן מגלם את פיפין (פרגרין טוק) בארץ התיכונה.
Pure frontend — HTML/JS, ללא שרת. רץ מ-localhost עם `python3 -m http.server 8765`.

## ארכיטקטורה
- `index.html` + `style.css` — ממשק משתמש RTL
- `js/game.js` — מנוע משחק ראשי, ניהול state
- `js/api.js` — תקשורת עם Gemini API (נרטיב + Imagen תמונות)
- `js/ui.js` — עדכוני ממשק, מפה, תמונות, דיאלוגים
- `js/world.js` — מבנה נתונים סטטי: ~30 מקומות, דמויות, חפצים
- `js/prompts.js` — system prompt ל-Gemini, בניית turn prompt

## עיקרון המפתח: Lazy World Generation
העולם נבנה תוך כדי משחק:
- ביקור ראשון במקום → Gemini יוצר נרטיב, Imagen יוצר תמונה → נשמרים ב-localStorage
- ביקור חוזר → טוען מ-cache, ללא קריאה ל-API
- cache keys: `pipin_img_{locationId}`, `pipin_save`, `pipin_api_key`

## API
- מודל נרטיב: `gemini-2.0-flash-lite` (free tier)
- מודל תמונות: `imagen-3.0-generate-002` (ייתכן שדורש billing)
- API key: מוכנס ידנית במסך פתיחה, נשמר ב-localStorage

## מפה
מפה SVG דינמית בעמודה ימנית (200px). נבנית ב-BFS מ-hobbiton לפי הקשרים ב-world.js.
צמתים: זהב = ביקרת, ירוק = מיקום נוכחי (עם halo).
