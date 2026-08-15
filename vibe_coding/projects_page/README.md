# projects_page — דף הפרויקטים

הדף `vibe_coding/index.html` **נוצר אוטומטית**. אל תערוך אותו ישירות, כל עריכה
בו תימחק בבנייה הבאה.

## מקור האמת

| קובץ | תפקיד |
|---|---|
| `projects.json` | כל הפרויקטים: שם, אייקון, תגיות, תיאור, קישורים, איורי SVG |
| `template.html` | שלד הדף: CSS, כותרת, פוטר. הפרויקטים נכנסים ל-`{{SECTIONS}}` |
| `build.py` | בונה את `index.html` מהשניים |
| `extract.py` | מיגרציה חד-פעמית שהפיקה את `projects.json` מהדף שנכתב ביד |
| `pre-commit` | hook שמריץ את הבנייה בכל קומיט |

## שימוש

```
python3 projects_page/build.py            # בנה מחדש את index.html
python3 projects_page/build.py --check    # בדוק פערים בלי לבנות
```

הסטטיסטיקות בראש הדף (מספר פרויקטים, live demos, AI-powered, קטגוריות)
מחושבות מהנתונים ולא נכתבות ביד.

## התקנת ה-hook

```
ln -sf ../../vibe_coding/projects_page/pre-commit /home/ori/fun/.git/hooks/pre-commit
```

מרגע זה כל קומיט בונה מחדש את הדף ומוסיף אותו לקומיט, ומדפיס אזהרה אם יש
פרויקט בלי כרטיס או קישור שבור. האזהרה לא חוסמת את הקומיט.

## מה `--check` מוצא

1. **תיקיות וקבצי HTML בלי כרטיס בדף** — פרויקט חדש שנוסף ולא הופיע.
2. **קישורים לקבצים שלא קיימים** — הקובץ נמחק או שונה שמו.
3. **קישורים לקבצים שלא ב-git** — קיימים מקומית אבל יחזירו 404 בשידור חי,
   כי GitHub Pages משרת רק מה שנדחף.

חריגים מכוונים נכנסים ל-`SKIP` בראש `build.py`.

## הוספת פרויקט

הוסף אובייקט למערך `projects` של הסקציה המתאימה ב-`projects.json`:

```json
{
  "title": "שם הפרויקט",
  "icon": "🎯",
  "theme": "t-purple",
  "dir": "my_project",
  "tags": ["Flask", "Claude API"],
  "desc": "מה הפרויקט עושה, במשפט או שניים.",
  "badges": [{ "type": "live", "label": "LIVE" }],
  "links": [
    { "href": "my_project/index.html", "style": "primary", "label": "▶ פתח" },
    {
      "href": "https://github.com/orimosenzon/fun/tree/master/vibe_coding/my_project",
      "style": "secondary",
      "label": "📁 קוד מקור",
      "external": true
    }
  ]
}
```

שדות רשות:

- `dir` — התיקייה שהכרטיס מייצג. נחוץ רק כשאף קישור לא מזכיר אותה
  (למשל פרויקט שמתארח בריפו אחר), אחרת `--check` יתלונן שהפרויקט חסר.
- `badges` — `live` (עם נקודה מהבהבת) או `wip`. ה-`label` חופשי:
  `⚙ בפיתוח`, `⚙ בתכנון`, `🖥 דסקטופ`.
- `illus` — SVG של 72×72 שמוצג בריבוע שליד הכרטיס.
- `small` — כרטיס צר בלי עמודת איור.
- `ai` — עוקף את הזיהוי האוטומטי לספירת ה-AI-Powered.
- `private` — הפרויקט לא אמור להיות בריפו הציבורי, אל תזהיר עליו.
- `pending_publish` — הקוד עוד לא פורסם ולכן אין קישור. הכרטיס נבנה בלי שורת
  הכפתורים, ו-`--check` מזכיר. כשהריפו יעלה, הוסף קישור והסר את השדה.

ערכות צבע: `t-purple`, `t-cyan`, `t-indigo`, `t-blue`, `t-teal`, `t-amber`,
`t-pink`, `t-green`, `t-red`, `t-orange`.
