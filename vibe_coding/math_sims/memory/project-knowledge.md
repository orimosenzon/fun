# math_sims: ידע על הפרויקט

**מטרה:** סרטוני סימולציות מתמטיות עם ManimGL (הספרייה של 3b1b). נפתח 28/10/2026.

**סטאק:** Python 3.10 ב-`.venv` של הפרויקט, manimgl 1.7.2, numpy, mpmath. LaTeX מ-TinyTeX (`~/.TinyTeX/bin/x86_64-linux` חייב להיות ב-PATH, `render.sh` דואג לזה). פונט Rubik ב-`~/.local/share/fonts`.

## החלטות
- ManimGL ולא ManimCE, כי אורי ביקש את הספרייה של סנדרסון.
- חישובים כבדים (mpmath) נעשים מראש ב-`compute/` ונשמרים ל-`data/*.npz`, הסצנות רק קוראות.
- כל הכיתוביות בעברית דרך `heb()` שעוטפת ביטויים לטיניים ב-LRI/PDI. בלי מספרים שליליים בתוך כיתובית עברית (מתבלגן), ובלי מקף ארוך.
- סרטון = קובץ סצנות עם רשימת `SCENES` בסוף, `render.sh` מרנדר ומאחד עם ffmpeg.

## מלכודות ב-ManimGL 1.7.2
- `ComplexPlane.n2p` לא וקטורי. למערך משתמשים ב-`plane.c2p(z.real, z.imag)`.
- `set_opacity` / `fade` על polyline צובעים גם את המילוי (או קורסים). לעמעום: `set_stroke(opacity=...)`.
- `animate.set_value` על Integer משעה את ה-updaters שלו. לשים updater על ValueTracker נפרד.

## סרטונים
- `videos/zeta_primes.mp4` (28/10/2026): זטה והראשוניים, 9 סצנות, 3:07.

## צעדים הבאים אפשריים
- קריינות (TTS עברי) מסונכרנת לסצנות
- סרטונים נוספים בסדרה
- כרטיס בדף הפרויקטים
