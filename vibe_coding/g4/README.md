# g4 — Gemma 4 Local Chatbot

צ'אטבוט מקומי מבוסס Google Gemma 4 (E2B) דרך Ollama.

## חומרה
- GPU: GTX 1060 6GB — מספיק ל-E2B quantized
- RAM: 16GB

## התקנה

### שלב 1 — התקן Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### שלב 2 — הורד את המודל
```bash
ollama pull gemma4:e2b
```
(ייקח כמה דקות — המודל ~3GB)

### שלב 3 — הרץ את הצ'אטבוט
```bash
python3 chat.py
```

## שימוש
- כתוב כל שאלה/בקשה בשפה חופשית (עברית או אנגלית)
- המודל עונה בסטרימינג (מילה מילה בזמן אמת)
- ההיסטוריה נשמרת במהלך הסשן
- `exit` / `quit` / `יציאה` לסיום
