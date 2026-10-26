# חידות לכרמית: ידע על הפרויקט

- **האפליקציה:** `app/` סטטית (HTML/CSS/JS עם ES modules, בלי build), באוויר ב-GitHub Pages: orimosenzon.github.io/fun/vibe_coding/%D7%9B%D7%A8%D7%9E%D7%99%D7%AA/app/index.html
- **החידות ב-Firestore** מ-26/10/2026, בפרויקט `carmit-riddles` של orimosenzon@gmail.com, בקולקציה `riddles`. אין יותר riddles.js ו-build_riddles.py.
- **מבנה חידה:** `image` (data URL), `question` ו-`hint` (אופציונליים), `answerType` ('text' או 'image'), `answers` (רשימה), `primary` (האינדקס של התשובה העיקרית), `answerImage`, `by`, `created`.
- התמונות נשמרות בתוך המסמך (JPEG, צלע ארוכה עד 1280, פחות מ-380K תווים לתמונה). לכן לא צריך Storage ולא חיוב.
- **ניהול:** `app/admin.html` עם כניסת Google. רק orimosenzon@gmail.com ו-carmitconsari@gmail.com מורשים. האכיפה ב-`firebase/firestore.rules`, והפריסה ב-`firebase/deploy_rules.sh` (דרך gcloud, בלי firebase login).
- **תשובה בתמונה:** השחקן חושב, לוחץ "הצגת התשובה" ומסמן לבד אם צדק.
- **בדיקת תשובה טקסטואלית:** לא תלויה ברישיות, ברווחים, בפיסוק או באותיות סופיות. נכון = שווה לאחת התשובות האפשריות.
- `tools/seed_firestore.py` טען את 16 החידות הראשונות (seed-01..16). **לא להריץ שוב**, הוא ידרוס עריכות מהאדמין.
