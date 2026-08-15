---
name: projects_page — דף הפרויקטים
description: ניהול index.html — הדף המרכזי שמציג את כל פרויקטי vibe_coding
type: project
---

# projects_page — ניהול דף הפרויקטים

## מיקום
- הדף: `/home/ori/fun/vibe_coding/index.html` — **נוצר אוטומטית, לא לערוך ביד**
- URL ציבורי: https://orimosenzon.github.io/fun/vibe_coding/index.html

## איך זה עובד (מ-16.8.2026)
הדף עבר מכתיבה ידנית לבנייה מנתונים. שלושה קבצים ב-`projects_page/`:

- `projects.json` — מקור האמת. כל פרויקט: שם, אייקון, ערכת צבע, תגיות, תיאור,
  תגי live/wip, קישורים, ואיור SVG.
- `template.html` — שלד הדף (CSS, header, footer) עם `{{HEADER}}` ו-`{{SECTIONS}}`.
- `build.py` — מרכיב את השניים ל-`index.html`.

```
python3 projects_page/build.py            # בנייה
python3 projects_page/build.py --check    # ביקורת בלי בנייה
```

תיעוד מלא כולל סכמת שדות: `projects_page/README.md`.
`extract.py` היא המיגרציה החד-פעמית שהפיקה את ה-JSON מהדף הישן, שמורה לתיעוד.

## שני דברים שהמכניקה נותנת בחינם
1. **סטטיסטיקות מחושבות** — מספר פרויקטים, live demos, AI-powered וקטגוריות
   נספרים מהנתונים. קודם הם היו מספרים קשיחים ב-HTML והתיישנו (הראו 32 כשהיו 38).
2. **`--check` מוצא פערים** — פרויקט בלי כרטיס, קישור לקובץ שנמחק, וקישור לקובץ
   שקיים מקומית אבל לא ב-git ולכן יחזיר 404 בשידור חי.

## עדכון אוטומטי
hook של pre-commit ב-`projects_page/pre-commit`. התקנה:

```
ln -sf ../../vibe_coding/projects_page/pre-commit /home/ori/fun/.git/hooks/pre-commit
```

בכל קומיט הוא בונה מחדש את הדף, מוסיף אותו לקומיט, ומדפיס אזהרה על פערים
בלי לחסום.

## למה חשוב שהתיקייה תהיה ב-git
GitHub Pages משרת רק מה שנדחף. פרויקט שקיים מקומית ולא בריפו — הקישור אליו
בדף יחזיר 404. זה מה שקרה ל-golem, sidekick, wikivoice, bloom ו-ירדן.
`haskala` מסומן `"private": true` כי הוא בריפו נפרד בכוונה.

## הערות תוכן
- `job-search`, `factotum`, `billing-check` מחוץ לדף בכוונה — ב-`SKIP` ב-`build.py`.
- `map_search / smart_map / ms / ms_es / ms_mob / wip / tmp.html` הן איטרציות
  קודמות של Smart Business Search, גם הן ב-`SKIP`.
