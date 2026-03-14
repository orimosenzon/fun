---
name: bezier-animate project knowledge
description: General knowledge about the Bezier curve animation editor project
type: project
---

# פרויקט עורך אנימציית עקומות בזייה

## מיקום
`/home/ori/fun/vibe_coding/animate/`

## מה זה
עורך אנימציה מבוסס web (HTML5 Canvas + Vanilla JS) לציור עקומות בזייה קוביות ויצירת אנימציות מרובות פריימים.

## סטאק
- **Frontend only**: HTML + CSS + Vanilla JavaScript
- **Canvas API** לציור
- **אין שרת** — קובץ HTML סטטי שעובד ישירות בדפדפן
- **פורמט שמירה**: `.clp` (JSON מותאם אישית)

## פיצ'רים עיקריים
- ציור עקומות בזייה קוביות על canvas
- מצב ציור (רק בפריים 1) + מצב עריכה (כל פריים)
- גרירת anchor points ו-control points
- undo/redo (Ctrl+Z / Ctrl+Y)
- זום עם גלגלת עכבר
- ניגון אנימציה עם תמיכה בהצגת/הסתרת נקודות בקרה
- ציר זמן (timeline) עם thumbnails של כל הפריימים
- שמירה/טעינה של פרויקט בפורמט `.clp`

## קבצים
- `anim9.html` — **הגרסה הנוכחית** (עם ציר זמן)
- `anim8.html` — גרסה קודמת (משופרת)
- `anim7.html`, `anim6.html` — גרסאות ישנות יותר
- `*.clp` — קבצי אנימציה שמורים (animation.clp, nose.clp, וכו')

## פורמט `.clp`
JSON עם `animationFrames`: מערך של פריימים, כל פריים מכיל מערך של רצפים (sequences).
כל רצף: `id`, `anchors` (נקודות עגינה), `segments` (פלחי בזייה עם p0, cp1, cp2, p1).

## Why:
פרויקט creative/educational — עורך להנפשת עקומות. פותח בגרסאות איטרטיביות.
