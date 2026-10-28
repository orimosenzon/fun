# סימולציות מתמטיות (math_sims)

סרטוני מתמטיקה עם **ManimGL**, הספרייה של גראנט סנדרסון מערוץ 3Blue1Brown
(הריפו `3b1b/manim`, ולא הגרסה הקהילתית ManimCE).

## הסרטון הראשון: זטה של רימן והמספרים הראשוניים

`videos/zeta_primes.mp4`, ‏3:07 דקות, 1080p, כיתוביות בעברית. תשע סצנות:

1. **Title**: פתיחה
2. **PrimeGrid**: הראשוניים עד 100 על לוח
3. **PrimeCounting**: π(x) כמדרגות, הקירוב של גאוס x/ln x והקירוב החלק של רימן R(x)
4. **ZetaDefinition**: הגדרת זטה, והסכום החלקי של ζ(2) מתכנס ל-π²/6
5. **EulerProduct**: מכפלת אוילר: סכום על כל המספרים = מכפלה על הראשוניים
6. **ComplexZeros**: המישור המרוכב, הקוטב ב-1, האפסים הטריוויאליים, הרצועה והקו הקריטי
7. **CriticalLineSpiral**: מסלול ζ(½+it) עד t=50, כל מעבר בראשית הוא אפס
8. **ExplicitFormula**: הנוסחה המפורשת: R(x) פחות גלים מ-1, 3, 10, 30, 100 אפסים משחזר את המדרגות
9. **Hypothesis**: השערת רימן ומשמעותה לחסם על השגיאה

## הרצה

```bash
.venv/bin/python compute/zeta_data.py   # פעם אחת, ~5.5 דקות: אפסים, ערכי זטה, הקירובים
./render.sh                             # כל הסצנות ב-1080p ואיחוד ל-videos/zeta_primes.mp4
./render.sh -l                          # טיוטה מהירה באיכות נמוכה
ONLY="ExplicitFormula" ./render.sh -l   # רק סצנות מסוימות (האיחוד עדיין לוקח את כולן)
```

תצוגה אינטראקטיבית של סצנה בחלון (בלי `-w`):

```bash
PATH=~/.TinyTeX/bin/x86_64-linux:$PATH .venv/bin/manimgl scenes/zeta_primes.py CriticalLineSpiral
```

## מה הותקן

- `.venv/` עם `manimgl` 1.7.2, `ManimPango` (נבנה מקוד מקור מול `libpango1.0-dev`, `libcairo2-dev` מ-apt)
- LaTeX: ‏TinyTeX ב-`~/.TinyTeX` (בלי sudo), עם החבילות של התבנית של ManimGL ו-`dvisvgm`
- פונט עברי Rubik ב-`~/.local/share/fonts`

## עברית ב-ManimGL

`Text` מרנדר עברית נכון דרך Pango, אבל ביטוי כמו `π(x)` או `12 = 2·2·3` בתוך משפט עברי מתהפך.
הפונקציה `heb()` עוטפת רצפים לטיניים/מתמטיים בתווי בידוד כיווניות (LRI/PDI) ופותרת את זה.
נוסחאות של ממש נכתבות ב-`Tex` בנפרד מהטקסט העברי.
