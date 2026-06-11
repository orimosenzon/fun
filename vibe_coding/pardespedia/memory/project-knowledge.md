# pardespedia — ידע פרויקט

## מהות
כלי CLI לניהול דפים בוויקי הקהילתי pardespedia.info (ויקי של פרדס חנה-כרכר) דרך MediaWiki API.

## מבנה
- `wiki_client.py` — קליינט ל-API: לוגין, get/edit/create, search, list, categories
- `pardespedia.py` — CLI: get, edit, create, search, list, cats
- `credentials.json` — פרטי הבוט (ב-.gitignore, לא בריפו). יש `credentials.json.example` כתבנית.

## חשבון
הבוט מתחבר כמשתמש **"אורי מוסנזון בוט"** (חשבון נפרד שאורי פתח בוויקי, לא Special:BotPasswords). לוגין נבדק ועובד (2026-06-11).

## כלל עריכה מחייב — לכבד תוכן קיים
ה-API מחליף את כל הוויקיטקסט של הדף במה ששולחים. לכן **לפני כל עריכת דף קיים: חובה למשוך את התוכן הנוכחי (`get`), לבצע את השינוי הנקודתי בתוכו, ולשלוח את הדף המלא המעודכן.** לעולם לא לשלוח תוכן שנכתב מאפס על דף קיים — זה מוחק את מה שהיה (ההיסטוריה נשמרת, אבל הגרסה החיה נדרסת).
- `create` משתמש בדגל `createonly` ונכשל אם הדף קיים — בכוונה. אם הדף קיים, עוברים ל-flow של edit (get → שינוי → שליחה).

## הרצה
```bash
cd ~/fun/vibe_coding/pardespedia
python3 pardespedia.py list [--prefix X]
python3 pardespedia.py get "שם דף"
python3 pardespedia.py edit "שם דף" content.wiki --summary "..."
python3 pardespedia.py create "שם דף חדש" content.wiki --summary "..."
```

## צעדים אפשריים להמשך
- pagination ב-list (כרגע מוגבל ל-50)
- תמיכה ב-stdin במקום קובץ ב-edit/create
