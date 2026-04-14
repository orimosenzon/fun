---
name: esla_vibe project knowledge
description: ידע כללי על פרויקט esla_vibe — ויזואליזציה אינטראקטיבית לאלגברה לינארית
type: project
---

# esla_vibe — Linear Algebra Visualization Suite

## מקור
תכנות מחדש של https://github.com/orimosenzon/fun/tree/master/esla
הפרויקט המקורי (2009) — כלים חינוכיים ב-Python 2 + Tkinter + VPython.

## מטרה
אפליקציית web אחת עם 5 כלים ויזואליים ללימוד אלגברה לינארית.

## סטאק
- **Dash 2.x** — framework ראשי (callbacks, layout)
- **Plotly 5.x** — גרפים 2D ו-3D
- **NumPy 2.x** — חישובים נומריים
- **SymPy 1.13** — מתמטיקה סימבולית (Gaussian elimination עם שברים)
- **dash-bootstrap-components** — עיצוב (CYBORG dark theme)

## הרצה
```bash
cd /home/ori/fun/vibe_coding/esla_vibe
pip3 install -r requirements.txt
python3 app.py
# → http://localhost:5050
```

## מבנה קבצים
```
app.py                    ← entry point (Dash + tabs)
views/
  transform.py            ← Tab 1: טרנספורמציות לינאריות 2D
  rowcol.py               ← Tab 2: שורות ועמודות
  gaussian.py             ← Tab 3: אלימינציה גאוסית שלב-שלב
  vecspan.py              ← Tab 4: span וקטורים ב-3D
  surface3d.py            ← Tab 5: פונקציות 3D
utils/
  linalg.py               ← math utilities (shared)
assets/custom.css         ← dark theme overrides
requirements.txt
```

## מה כל כלי מדגים

### Tab 1 — Transformations (trans.py rewrite)
- גריד לפני ואחרי הטרנספורמציה (סגנון 3Blue1Brown)
- הגדרות מוכנות: סיבוב, שיקוף, גזירה, הטלה, הגדלה...
- Slider t∈[0,1] מאניחם את הטרנספורמציה בצורה חלקה
- לחץ על הגרף להוספת נקודות וצפה לאן הן הולכות
- אפשרויות: eigenvectors, מקבילית det(T), וקטורי בסיס

### Tab 2 — Row/Col (rawcol.py rewrite)
- 4 inputs לאלמנטי המטריצה [[a,b],[c,d]]
- מציג בנפרד: וקטורי שורה (כחול) ו-וקטורי עמודה (כתום)
- כשהדטרמיננטה → 0, שניהם מתיישרים בו-זמנית

### Tab 3 — Gaussian Elimination (gEliminate.py rewrite)
- מקלדת מטריצה מוגדלת (תומך שברים)
- ניווט שלב-שלב: שורה פיבוט מודגשת
- 4 דוגמאות מובנות כולל מטריצה סינגולרית

### Tab 4 — Vector Span 3D (vec_comb.py rewrite)
- עד 4 וקטורים ב-R³ עם sliders לכל רכיב
- מציג span: קו (rank 1), מישור (rank 2), R³ (rank 3)
- מזהה תלות לינארית

### Tab 5 — 3D Surface (3dFunction.py rewrite)
- נוסחה חופשית f(x,y): sin, cos, exp, sqrt...
- סקאלת צבעים, טווח וב-רזולוציה ניתנים לשינוי

## Next steps אפשריים
- הוסף tab לכפל מטריצות עם ויזואליזציה
- הוסף tab לאוטוערכים/וקטורים עצמיים (power iteration)
- הוסף tab לפירוק SVD
- הוסף מספרים מרוכבים (complex.py) — מוכפלים כטרנספורמציה לינארית
- הוסף אנימציה אוטומטית (dcc.Interval) לסליידר הטרנספורמציה
