import io
import json
import os
import sqlite3
from datetime import datetime

from anthropic import Anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from gtts import gTTS

load_dotenv()

app = Flask(__name__)
client = Anthropic()
DB_PATH = os.path.join(os.path.dirname(__file__), 'nudge.db')

CATEGORY_NAMES = {
    'food': 'אוכל',
    'transport': 'תחבורה',
    'housing': 'דיור',
    'health': 'בריאות',
    'entertainment': 'בידור',
    'clothing': 'ביגוד',
    'utilities': 'שירותים',
    'subscriptions': 'מנויים',
    'other': 'אחר',
}

SYSTEM_PROMPT = """You are Nudge, a warm and supportive personal finance assistant.
Respond in the same language the user uses (Hebrew, English, etc.).
Be friendly, brief, and non-judgmental. Never lecture or moralize.

Your tasks:
1. Parse expense/income reports from natural language
2. Answer questions about the user's spending
3. Give gentle, practical financial advice

ALWAYS respond with valid JSON in this exact format (no markdown, no extra text):
{
  "action": "add_expense" | "add_income" | "set_balance" | "query" | "advice",
  "data": {
    "amount": <number or null>,
    "category": "<category or null>",
    "description": "<brief description in the user's language, or null>",
    "is_recurring": <true or false>
  },
  "response": "<your warm, brief reply in the user's language>"
}

Categories (use English keys):
- food: restaurants, cafes, groceries, supermarket
- transport: bus, train, taxi, fuel, parking
- housing: rent, mortgage, maintenance, cleaning
- health: pharmacy, doctor, dentist, gym
- entertainment: movies, games, concerts, outings
- clothing: clothes, shoes, accessories
- utilities: electricity, water, gas, internet, phone
- subscriptions: Netflix, Spotify, apps, recurring services
- other: anything else

is_recurring = true for: rent, subscriptions, regular bills, monthly fixed expenses
is_recurring = false for: one-time purchases, occasional expenses

Action guide:
- "add_expense": user spent money ("הוצאתי", "קניתי", "שילמתי", "spent", "bought", "paid")
- "add_income": user received money ("נכנסה משכורת", "קיבלתי", "received", "got paid")
- "set_balance": user states their current balance ("יש לי", "היתרה שלי", "my balance is")
- "query": user asks about their spending ("כמה הוצאתי", "how much did I spend")
- "advice": user asks for advice or just chatting

For query/advice, data fields can be null."""


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        amount REAL NOT NULL,
        category TEXT NOT NULL DEFAULT 'other',
        description TEXT DEFAULT '',
        is_recurring INTEGER DEFAULT 0,
        date TEXT DEFAULT CURRENT_DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_settings (
        id INTEGER PRIMARY KEY,
        balance REAL DEFAULT 0
    )''')
    c.execute('INSERT OR IGNORE INTO user_settings (id, balance) VALUES (1, 0)')
    conn.commit()
    conn.close()


def get_balance():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT balance FROM user_settings WHERE id=1')
    row = c.fetchone()
    conn.close()
    return row[0] if row else 0


def add_expense(amount, category, description, is_recurring):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    date = datetime.now().strftime('%Y-%m-%d')
    c.execute(
        'INSERT INTO expenses (amount, category, description, is_recurring, date) VALUES (?, ?, ?, ?, ?)',
        (amount, category or 'other', description or '', 1 if is_recurring else 0, date)
    )
    c.execute('UPDATE user_settings SET balance = balance - ? WHERE id=1', (amount,))
    conn.commit()
    conn.close()
    return get_balance()


def add_income(amount, description):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE user_settings SET balance = balance + ? WHERE id=1', (amount,))
    conn.commit()
    conn.close()
    return get_balance()


def set_balance(amount):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE user_settings SET balance = ? WHERE id=1', (amount,))
    conn.commit()
    conn.close()
    return amount


def get_monthly_summary():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT category, SUM(amount), COUNT(*)
                 FROM expenses
                 WHERE strftime('%Y-%m', date) = strftime('%Y-%m', 'now')
                 GROUP BY category
                 ORDER BY SUM(amount) DESC''')
    categories = [
        {'category': row[0], 'name': CATEGORY_NAMES.get(row[0], row[0]), 'total': row[1], 'count': row[2]}
        for row in c.fetchall()
    ]
    c.execute('''SELECT is_recurring, SUM(amount)
                 FROM expenses
                 WHERE strftime('%Y-%m', date) = strftime('%Y-%m', 'now')
                 GROUP BY is_recurring''')
    fixed = 0
    occasional = 0
    for row in c.fetchall():
        if row[0]:
            fixed = row[1]
        else:
            occasional = row[1]
    conn.close()
    return {'categories': categories, 'fixed': fixed, 'occasional': occasional}


def get_recent_expenses(limit=5):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT amount, category, description, is_recurring, date
                 FROM expenses ORDER BY created_at DESC LIMIT ?''', (limit,))
    expenses = [
        {'amount': r[0], 'category': r[1], 'description': r[2], 'is_recurring': bool(r[3]), 'date': r[4]}
        for r in c.fetchall()
    ]
    conn.close()
    return expenses


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'empty message'}), 400

    balance = get_balance()
    recent = get_recent_expenses(3)
    summary = get_monthly_summary()

    context_parts = [f"Current account balance: ₪{balance:.2f}"]
    if recent:
        recent_str = ', '.join([f"₪{e['amount']} on {e['description']}" for e in recent])
        context_parts.append(f"Recent expenses: {recent_str}")
    month_total = summary['fixed'] + summary['occasional']
    if month_total > 0:
        context_parts.append(f"This month's total spending: ₪{month_total:.2f}")

    full_message = user_message + '\n\n[Context: ' + '; '.join(context_parts) + ']'

    response = client.messages.create(
        model='claude-haiku-4-5-20251001',
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': full_message}]
    )

    raw = response.content[0].text.strip()
    # Strip markdown code fences if present
    if raw.startswith('```'):
        raw = raw.split('```')[1]
        if raw.startswith('json'):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return jsonify({'response': raw, 'balance': balance, 'summary': summary, 'action': 'error'})

    action = parsed.get('action', 'advice')
    action_data = parsed.get('data') or {}
    friendly_response = parsed.get('response', '')
    new_balance = balance

    if action == 'add_expense' and action_data.get('amount'):
        new_balance = add_expense(
            amount=float(action_data['amount']),
            category=action_data.get('category', 'other'),
            description=action_data.get('description', ''),
            is_recurring=action_data.get('is_recurring', False),
        )
    elif action == 'add_income' and action_data.get('amount'):
        new_balance = add_income(float(action_data['amount']), action_data.get('description', ''))
    elif action == 'set_balance' and action_data.get('amount') is not None:
        new_balance = set_balance(float(action_data['amount']))

    return jsonify({
        'response': friendly_response,
        'balance': new_balance,
        'summary': get_monthly_summary(),
        'action': action,
    })


@app.route('/api/expenses')
def api_expenses():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, amount, category, description, is_recurring, date
                 FROM expenses ORDER BY created_at DESC LIMIT 100''')
    expenses = [
        {'id': r[0], 'amount': r[1], 'category': r[2],
         'category_name': CATEGORY_NAMES.get(r[2], r[2]),
         'description': r[3], 'is_recurring': bool(r[4]), 'date': r[5]}
        for r in c.fetchall()
    ]
    conn.close()
    return jsonify(expenses)


@app.route('/api/tts', methods=['POST'])
def api_tts():
    text = (request.json or {}).get('text', '')
    if not text:
        return jsonify({'error': 'empty'}), 400
    buf = io.BytesIO()
    gTTS(text=text, lang='iw').write_to_fp(buf)
    buf.seek(0)
    return send_file(buf, mimetype='audio/mpeg')


@app.route('/api/summary')
def api_summary():
    return jsonify({'balance': get_balance(), **get_monthly_summary()})


init_db()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f'✓ Nudge is running → http://localhost:{port}')
    app.run(debug=False, host='0.0.0.0', port=port)
