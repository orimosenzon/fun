# jetbike_film — סרט טיסת ניסוי של MJ-5

סרט אנימציה (~2 דק׳, 1080p30, סטריאו) של רוכב על אופנוע סילון: 4 מנועי עילוי עם צינורות פלטה כלפי מטה + מנוע שיוט אחורי. כל התנועה מסימולציה: גוף קשיח 6DOF ב-2kHz, מחשב טיסה ב-200Hz עם הקצאה דו-מהירותית (דחף דיפרנציאלי איטי + כנפונים מהירים), רוח, אפקט קרקע, צריכת דלק.

- נגן אינטראקטיבי: index.html (שרת סטטי).
- רינדור: `tools/render.py frames|audio-data|encode`, סאונד: `tools/audio.py`. הפלט ב-out/ (מחוץ ל-git, הסרט 310MB).
- GPU בכרום headless דורש `--use-angle=vulkan --enable-features=Vulkan --ignore-gpu-blocklist`.
- באגים שנפתרו: autoClear של three מנקה יעד אדיטיבי (צבירת תת-פריימים ובלום); Float32 לא ניתן לבלנדינג בלי EXT_float_blend → HalfFloat.
- המשך המנוע: jetbike_game.
