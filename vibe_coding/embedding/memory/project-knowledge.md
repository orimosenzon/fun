# פרויקט שיכון (Embedding) — ידע כללי

## מטרה
אפליקציית לימוד אינטראקטיבית שמדגימה איך עובדים word embeddings — בצורה ויזואלית וחווייתית.

## סטאק
- Flask (Python)
- GloVe מודל: `glove-wiki-gigaword-100` דרך `gensim`
- sklearn — PCA + t-SNE להקרנת וקטורים ל-2D
- Frontend: HTML/JS (תבנית אחת: `templates/index.html`)

## מצבי שימוש (מה בנינו)

### 1. Quiz
שאלות אנלוגיה אוטומטיות: מלך : מלכה :: X : ?
המשתמש בוחר מתוך 4 אפשרויות.

### 2. Free Mode
המשתמש בוחר 3 מילים מתוך pool ומחשב אנלוגיה חופשית (A : B :: C : ?)

### 3. Visualize
הקרנת וקטורי מילים ל-2D — בחירה בין PCA ו-t-SNE, סינון לפי קטגוריה.

### 4. Nearest Words
הקלדת מילה → מציג 7 שכנות קרובות ביותר בחלל הוקטורים.

## קטגוריות מובנות
- Animal Sounds
- Countries & Capitals
- Gender Pairs
- Object & Material
- Profession & Tool
- Big & Small

## קבצים מרכזיים
- `app.py` — Flask routes
- `analogy_engine.py` — לוגיקת השיכון (טעינת מודל, אנלוגיות, הקרנה)
- `templates/index.html` — ה-Frontend כולו
