#!/usr/bin/env python3
# server.py - Flask dev server with error logging for pipin

import json
import os
import sqlite3
import traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)
LOG_FILE = 'pipin_errors.log'
DB_FILE = os.environ.get('DB_FILE', 'pipin_world.db')
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'images', 'locations')


# ── DB ──────────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


# עמודות שנוספו אחרי הגרסה הראשונה. ההגירה רצה בכל עלייה של השרת כדי
# שמסד נתונים ישן מהשדה ימשיך לעבוד בלי טיפול ידני.
_LOCATION_COLUMNS = {
    'name': 'TEXT',
    'region': 'TEXT',
    'description': 'TEXT',
    'lore': 'TEXT',
    'visual_notes': 'TEXT',
    'image_prompt': 'TEXT',
    'image_path': 'TEXT',
    'updated_at': 'TEXT',
}


def init_db():
    with get_db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS locations (
            location_id  TEXT PRIMARY KEY,
            player_id    TEXT,
            created_at   TEXT,
            narrative    TEXT,
            image_data   TEXT
        )''')
        existing = {r['name'] for r in conn.execute('PRAGMA table_info(locations)')}
        for col, decl in _LOCATION_COLUMNS.items():
            if col not in existing:
                conn.execute(f'ALTER TABLE locations ADD COLUMN {col} {decl}')

        # סיפור-העל של העולם. שורה אחת בלבד (id=1): מה קורה בארץ התיכונה
        # בתקופה הזאת, ומהי שפת הציור המשותפת לכל 30 המקומות.
        conn.execute('''CREATE TABLE IF NOT EXISTS world_lore (
            id          INTEGER PRIMARY KEY CHECK (id = 1),
            title       TEXT,
            premise     TEXT,
            style_bible TEXT,
            updated_at  TEXT
        )''')
        conn.execute('INSERT OR IGNORE INTO world_lore (id) VALUES (1)')


def write_log(entry: dict):
    entry['server_time'] = datetime.now().isoformat()
    line = json.dumps(entry, ensure_ascii=False)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    # Also print to terminal for live visibility
    level = entry.get('level', 'ERROR').upper()
    msg = entry.get('message', '')
    src = entry.get('source', '')
    print(f"[{entry['server_time']}] [{level}] {msg}" + (f"  ({src})" if src else ''))

# ── World API ───────────────────────────────────────────────────────────

@app.route('/api/world/player/<player_id>/stats', methods=['GET'])
def get_player_stats(player_id):
    with get_db() as conn:
        rows = conn.execute(
            'SELECT location_id FROM locations WHERE player_id=? ORDER BY created_at',
            (player_id,)
        ).fetchall()
    return jsonify({
        'player_id': player_id,
        'locations_canonized': len(rows),
        'location_ids': [r['location_id'] for r in rows],
    })


@app.route('/api/world/location/<location_id>', methods=['GET'])
def get_location(location_id):
    with get_db() as conn:
        row = conn.execute(
            'SELECT * FROM locations WHERE location_id=?', (location_id,)
        ).fetchone()
    if not row:
        return jsonify({'found': False}), 404
    return jsonify({
        'found': True,
        'narrative': row['narrative'],
        # תמונה שנוצרה בעורך גוברת על כתובת חיצונית ישנה
        'image_data': row['image_path'] or row['image_data'],
        'lore': row['lore'],
        'player_id': row['player_id'],
        'created_at': row['created_at'],
    })


@app.route('/api/world/location/<location_id>', methods=['POST'])
def save_location(location_id):
    """Idempotent canon save. Narrative and image arrive in separate calls
    (text is fast, image generation is slow), so we let either field fill an
    empty slot — but never overwrite a value that's already canonized."""
    data = request.get_json(force=True, silent=True) or {}
    new_narrative = data.get('narrative')
    new_image = data.get('image_data')
    with get_db() as conn:
        row = conn.execute(
            'SELECT player_id, narrative, image_data FROM locations WHERE location_id=?',
            (location_id,)
        ).fetchone()
        if row is None:
            conn.execute(
                'INSERT INTO locations (location_id, player_id, created_at, narrative, image_data) VALUES (?,?,?,?,?)',
                (location_id, data.get('player_id'), datetime.now().isoformat(),
                 new_narrative, new_image)
            )
            return jsonify({'saved': True, 'created': True})

        updates, params = [], []
        if new_narrative and not row['narrative']:
            updates.append('narrative=?'); params.append(new_narrative)
        if new_image and not row['image_data']:
            updates.append('image_data=?'); params.append(new_image)
        if not updates:
            return jsonify({'saved': False, 'reason': 'already_canonized'})
        params.append(location_id)
        conn.execute(f"UPDATE locations SET {', '.join(updates)} WHERE location_id=?", params)
    return jsonify({'saved': True, 'updated': updates})


# ── Editor API ──────────────────────────────────────────────────────────
# בונים את העולם ההתחלתי: אורי משוטט, מספר סיפור רקע לכל מקום, ומודל
# התמונות של Azure מצייר אותו. המפתח נשאר כאן ולא מגיע לדפדפן.

def _row_to_location(row):
    return {
        'location_id': row['location_id'],
        'name': row['name'],
        'region': row['region'],
        'description': row['description'],
        'lore': row['lore'],
        'visual_notes': row['visual_notes'],
        'narrative': row['narrative'],
        'image_path': row['image_path'],
        'image_prompt': row['image_prompt'],
        'updated_at': row['updated_at'],
    }


def _get_lore(conn):
    row = conn.execute('SELECT * FROM world_lore WHERE id=1').fetchone()
    return {
        'title': row['title'] or '',
        'premise': row['premise'] or '',
        'style_bible': row['style_bible'] or DEFAULT_STYLE_BIBLE,
        'updated_at': row['updated_at'],
    }


# ברירת מחדל לשפת הציור. זה מה שגורם ל-30 מקומות שונים להיראות כמו עולם
# אחד ולא כמו 30 תמונות שהוזמנו בנפרד, ולכן הוא נשלח בכל בקשת תמונה.
DEFAULT_STYLE_BIBLE = (
    "Painterly fantasy landscape illustration in the tradition of Alan Lee and "
    "John Howe: soft watercolour washes over fine graphite underdrawing, muted "
    "earth palette of moss green, weathered stone grey, ochre and dusk blue, "
    "diffuse natural light, deep atmospheric perspective with mist in the "
    "distance, human-scale detail that makes the landscape feel vast. "
    "No text, no lettering, no watermark, no signature, no modern objects."
)


def build_image_prompt(lore: dict, loc: dict) -> str:
    """שפת הציור המשותפת, ואז המקום הספציפי. הסדר חשוב: המודל מתייחס
    לתחילת הפרומפט כמסגרת ולשאר כתוכן."""
    parts = [lore['style_bible'], '']
    if lore.get('premise'):
        parts += [f"World context: {lore['premise']}", '']
    parts.append(f"Depict: {loc.get('name') or loc['location_id']}")
    if loc.get('description'):
        parts.append(loc['description'])
    if loc.get('lore'):
        parts.append(loc['lore'])
    # הכוונה ויזואלית באנגלית. באה אחרי הסיפור כדי שהיא תקבע את הקומפוזיציה
    # בפועל, ולא רק תרמוז עליה.
    if loc.get('visual_notes'):
        parts.append(loc['visual_notes'])
    parts.append('Wide establishing shot of the place itself. No people in the '
                 'foreground.')
    return '\n'.join(p for p in parts if p is not None)


@app.route('/api/editor/world', methods=['GET'])
def editor_get_world():
    with get_db() as conn:
        lore = _get_lore(conn)
        rows = conn.execute(
            'SELECT * FROM locations ORDER BY region, location_id').fetchall()
    return jsonify({
        'lore': lore,
        'locations': [_row_to_location(r) for r in rows],
        'image_model': _image_model_name(),
    })


@app.route('/api/editor/world', methods=['POST'])
def editor_save_world():
    data = request.get_json(force=True, silent=True) or {}
    with get_db() as conn:
        conn.execute(
            'UPDATE world_lore SET title=?, premise=?, style_bible=?, updated_at=? WHERE id=1',
            (data.get('title', ''), data.get('premise', ''),
             data.get('style_bible') or DEFAULT_STYLE_BIBLE,
             datetime.now().isoformat()))
        lore = _get_lore(conn)
    return jsonify({'saved': True, 'lore': lore})


@app.route('/api/editor/sync', methods=['POST'])
def editor_sync():
    """הלקוח מחזיק את 30 המקומות ב-world.js; זה מעתיק את השדות הסטטיים
    (שם, אזור, תיאור) לשרת, כדי שיוכל לבנות פרומפטים בעצמו."""
    data = request.get_json(force=True, silent=True) or {}
    locations = data.get('locations') or {}
    now = datetime.now().isoformat()
    with get_db() as conn:
        for loc_id, loc in locations.items():
            conn.execute('''INSERT INTO locations (location_id, created_at, name, region, description)
                            VALUES (?,?,?,?,?)
                            ON CONFLICT(location_id) DO UPDATE SET
                              name=excluded.name,
                              region=excluded.region,
                              description=COALESCE(locations.description, excluded.description)''',
                         (loc_id, now, loc.get('name'), loc.get('region'),
                          loc.get('description')))
    return jsonify({'synced': len(locations)})


@app.route('/api/editor/location/<location_id>', methods=['POST'])
def editor_save_location(location_id):
    """שומר את סיפור הרקע שאורי סיפר. בניגוד לקנון של השחקנים, כאן מותר
    לדרוס — זה כלי עריכה, והמחבר אמור לתקן את עצמו."""
    data = request.get_json(force=True, silent=True) or {}
    fields = {k: data[k] for k in ('lore', 'visual_notes', 'narrative', 'description',
                                   'name', 'region')
              if k in data}
    if not fields:
        return jsonify({'saved': False, 'reason': 'nothing_to_save'}), 400
    fields['updated_at'] = datetime.now().isoformat()
    with get_db() as conn:
        conn.execute('INSERT OR IGNORE INTO locations (location_id, created_at) VALUES (?,?)',
                     (location_id, datetime.now().isoformat()))
        sets = ', '.join(f'{k}=?' for k in fields)
        conn.execute(f'UPDATE locations SET {sets} WHERE location_id=?',
                     [*fields.values(), location_id])
        row = conn.execute('SELECT * FROM locations WHERE location_id=?',
                           (location_id,)).fetchone()
    return jsonify({'saved': True, 'location': _row_to_location(row)})


@app.route('/api/editor/location/<location_id>/prompt', methods=['GET'])
def editor_preview_prompt(location_id):
    """מה בדיוק יישלח למודל. מקור אמת אחד לעורך ולסקריפט האצווה."""
    with get_db() as conn:
        row = conn.execute('SELECT * FROM locations WHERE location_id=?',
                           (location_id,)).fetchone()
        lore = _get_lore(conn)
    if row is None:
        return jsonify({'error': 'unknown location'}), 404
    return jsonify({'prompt': build_image_prompt(lore, _row_to_location(row))})


def _image_model_name():
    try:
        import azure_images
        return azure_images.available_model()
    except Exception:
        return None


@app.route('/api/editor/location/<location_id>/image', methods=['POST'])
def editor_generate_image(location_id):
    data = request.get_json(force=True, silent=True) or {}
    with get_db() as conn:
        conn.execute('INSERT OR IGNORE INTO locations (location_id, created_at) VALUES (?,?)',
                     (location_id, datetime.now().isoformat()))
        row = conn.execute('SELECT * FROM locations WHERE location_id=?',
                           (location_id,)).fetchone()
        lore = _get_lore(conn)

    loc = _row_to_location(row)
    prompt = data.get('prompt') or build_image_prompt(lore, loc)

    try:
        import azure_images
        png = azure_images.generate(prompt, size=data.get('size', '1024x1024'))
    except Exception as e:
        write_log({'level': 'ERROR', 'message': f'image generation failed for {location_id}',
                   'source': 'editor', 'detail': str(e)[:500],
                   'trace': traceback.format_exc()[-800:]})
        return jsonify({'ok': False, 'error': str(e)[:400]}), 502

    os.makedirs(IMAGE_DIR, exist_ok=True)
    filename = f'{location_id}.png'
    with open(os.path.join(IMAGE_DIR, filename), 'wb') as f:
        f.write(png)
    rel = f'images/locations/{filename}'

    with get_db() as conn:
        conn.execute('UPDATE locations SET image_path=?, image_prompt=?, updated_at=? WHERE location_id=?',
                     (rel, prompt, datetime.now().isoformat(), location_id))
    return jsonify({'ok': True, 'image_path': rel, 'prompt': prompt,
                    'bytes': len(png)})


@app.route('/api/editor/status', methods=['GET'])
def editor_status():
    model = _image_model_name()
    return jsonify({'image_model': model, 'ready': bool(model)})


# ── Logging ─────────────────────────────────────────────────────────────

@app.route('/log', methods=['POST'])
def log_endpoint():
    try:
        data = request.get_json(force=True, silent=True) or {}
        write_log(data)
        return jsonify({'ok': True})
    except Exception as e:
        print(f'[LOG ENDPOINT ERROR] {e}')
        return jsonify({'ok': False}), 500

@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_static(path):
    if path.startswith('api/'):
        return jsonify({'error': 'not found'}), 404
    return send_from_directory('.', path)

init_db()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8765))
    print(f'Pipin server running on port {port}')
    app.run(host='0.0.0.0', port=port, debug=False)
