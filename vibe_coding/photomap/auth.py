"""סיסמה למפת התמונות.

הסיסמה עצמה לא נשמרת בשום מקום: ב-data/auth.json (מחוץ ל-git) יש רק מלח וגיבוב PBKDF2,
וסוד לחתימת עוגיות. אין קובץ כזה = אין סיסמה (מתאים רק לשימוש מקומי).

קביעת סיסמה:
    python3 serve.py --set-password        (קורא את הסיסמה מהמקלדת או מ-stdin)

העוגייה היא "תוקף.חתימה" בחתימת HMAC, כך שאין צורך לשמור רשימת כניסות.
ניסיונות שגויים מוגבלים לכל השרת יחד (מאחורי tunnel כל הבקשות מגיעות מאותה כתובת),
וזה מה שהופך סיסמה קצרה לבטוחה מספיק מול ניחוש דרך הרשת.
"""
import hashlib
import hmac
import json
import secrets
import threading
import time
from pathlib import Path

PATH = Path(__file__).resolve().parent / "data" / "auth.json"
COOKIE = "photomap"
DAYS = 30
ITERATIONS = 300_000
MAX_FAILS, WINDOW = 10, 600      # עד 10 ניסיונות שגויים ב-10 דקות, לכל השרת

_fails, _guard = [], threading.Lock()


def _load():
    return json.loads(PATH.read_text()) if PATH.exists() else None


def enabled():
    return PATH.exists()


def set_password(pw):
    salt = secrets.token_bytes(16)
    PATH.parent.mkdir(exist_ok=True)
    PATH.write_text(json.dumps({
        "salt": salt.hex(),
        "hash": hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, ITERATIONS).hex(),
        "iterations": ITERATIONS,
        "secret": secrets.token_hex(32),   # סוד חדש = כל העוגיות הקודמות מתבטלות
    }))
    PATH.chmod(0o600)


def locked():
    now = time.time()
    with _guard:
        _fails[:] = [t for t in _fails if now - t < WINDOW]
        return len(_fails) >= MAX_FAILS


def check(pw):
    a = _load()
    got = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(a["salt"]), a["iterations"]).hex()
    ok = hmac.compare_digest(got, a["hash"])
    if not ok:
        with _guard:
            _fails.append(time.time())
        time.sleep(1)
    return ok


def _sign(exp, secret):
    return hmac.new(bytes.fromhex(secret), str(exp).encode(), hashlib.sha256).hexdigest()


def cookie_header():
    exp = int(time.time()) + DAYS * 86400
    return f"{COOKIE}={exp}.{_sign(exp, _load()['secret'])}; Max-Age={DAYS * 86400}; Path=/; HttpOnly; SameSite=Lax"


def logout_header():
    return f"{COOKIE}=; Max-Age=0; Path=/; HttpOnly; SameSite=Lax"


def valid(cookie_str):
    for part in (cookie_str or "").split(";"):
        name, _, val = part.strip().partition("=")
        if name != COOKIE:
            continue
        exp, _, sig = val.partition(".")
        if exp.isdigit() and int(exp) > time.time() and hmac.compare_digest(sig, _sign(int(exp), _load()["secret"])):
            return True
    return False


def login_page(error=""):
    msg = {"bad": "סיסמה שגויה", "locked": "יותר מדי ניסיונות. נסו שוב בעוד כמה דקות."}.get(error, "")
    return f"""<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>מפת התמונות</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📍</text></svg>">
<style>
  :root {{ --bg: #f3f4f7; --card: #fff; --fg: #1d2330; --muted: #6b7385; --line: #d9dde6; --accent: #e0533d; }}
  @media (prefers-color-scheme: dark) {{ :root {{ --bg: #12151c; --card: #1b1f29; --fg: #eef0f5; --muted: #9aa1b2; --line: #2e3442; }} }}
  body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: var(--bg); color: var(--fg);
         font-family: Heebo, system-ui, sans-serif; padding: 16px; box-sizing: border-box; }}
  form {{ box-sizing: border-box; background: var(--card); padding: 32px 28px; border-radius: 16px; width: 100%; max-width: 340px;
         box-shadow: 0 8px 30px rgba(0,0,0,.12); text-align: center; }}
  h1 {{ margin: 0 0 4px; font-size: 24px; }}
  p {{ margin: 0 0 22px; color: var(--muted); }}
  input {{ width: 100%; box-sizing: border-box; font: inherit; font-size: 17px; padding: 11px 14px; border-radius: 10px;
          border: 1px solid var(--line); background: var(--bg); color: var(--fg); text-align: center; }}
  button {{ margin-top: 12px; width: 100%; font: inherit; font-size: 17px; padding: 11px; border: 0; border-radius: 10px;
           background: var(--accent); color: #fff; cursor: pointer; }}
  .err {{ color: var(--accent); min-height: 1.4em; margin: 10px 0 0; font-size: 15px; }}
</style></head>
<body><form method="post" action="/login">
  <div style="font-size:44px">📍</div>
  <h1>מפת התמונות</h1>
  <p>הכניסה בסיסמה</p>
  <input type="password" name="password" autocomplete="current-password" autofocus required aria-label="סיסמה">
  <button>כניסה</button>
  <div class="err">{msg}</div>
</form></body></html>"""
